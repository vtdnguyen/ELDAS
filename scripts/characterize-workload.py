#!/usr/bin/env python3
"""W0.1 — Workload characterisation for ELDAS traces.

Measures the quantities a load scenario is *supposed* to be defined by, for any
trace CSV in the Alibaba openb schema (the schema WM-1 will also emit):

  * offered load          rho = sum(demand_i * duration_i) / (capacity * T)
  * arrival process       rate, C_a^2, IDC(w) at several window widths
  * job-size distribution mean/median/p90/p99/CV, heavy-tail work share
  * instantaneous demand  peak & time-weighted percentiles, saturation time
  * composition           QoS mix, pod-phase mix, GPU-task fraction

Every formula mirrors the Java side so the numbers describe the simulation that
actually runs:

  duration  = max(0, deletion_time - max(creation_time, scheduled_time))
                                          (AlibabaTraceReader.TaskRecord.duration)
  pes       = max(1, cpu_milli // 1000)   (AlibabaTraceReader.TaskRecord.pesNeeded)
  gpu cards = num_gpu                     (SimulationManager.canHost: whole cards only;
                                           gpu_milli only drives GPU *affinity*)
  horizon T = span of creation times      (the episode ends at the last arrival and
                                           energy is integrated up to it, so this is
                                           the window capacity is billed over)

Scenario filters reproduce ScenarioFilter.java / eval/trace_loader.py exactly, so
the tool can characterise both the LEGACY slices and a flat WM-1 trace (ALL).

Standard library only — runs on the host without the Docker image.

Usage
-----
  python scripts/characterize-workload.py --trace data/alibaba-trace/openb_pod_list_default.csv
  python scripts/characterize-workload.py --trace <csv> --topology config/topology-hetero.json
  python scripts/characterize-workload.py --trace <csv> --format markdown --out report.md
  python scripts/characterize-workload.py --trace <csv> --scenarios ALL --format json
"""

from __future__ import annotations

import argparse
import csv
import datetime as _dt
import hashlib
import json
import math
import os
import shlex
import subprocess
import sys
from dataclasses import dataclass

# ── Defaults mirroring SimulationConfig.DEFAULT_HOST / DEFAULT_DC ───────────
DEFAULT_HOSTS = 10
DEFAULT_PES_PER_HOST = 64
DEFAULT_GPUS_PER_HOST = 8

# ── ScenarioFilter.java constants ───────────────────────────────────────────
LOW_FRACTION = 0.25
BURST_WINDOW_SEC = 3600.0
BURST_PERCENTILE = 0.80

SCENARIOS = ("LEGACY_LOW", "LEGACY_HIGH", "LEGACY_BURST", "ALL")

SEC_PER_HOUR = 3600.0
SEC_PER_DAY = 86400.0

# ── Deadline model, mirroring SimulationConfig (W1.5) ───────────────────────
#   deadline = creation + duration * slackFactor(qos) + slackFloor(qos)
# The floor was added because the multiplicative term alone left 24.6 % of openb
# tasks with under 60 s of absolute budget, where the fixed 5 s wake latency
# dominates and energy-saving suspends scored as SLA violations regardless of load.
QOS_SLACK_FACTOR = {"LS": 1.1, "Guaranteed": 1.3, "Burstable": 1.7, "BE": 3.0}
QOS_SLACK_DEFAULT = 1.5
SCHEDULING_LAG_P90_SEC = 106.0
QOS_FLOOR_FACTOR = {"LS": 1.0, "Guaranteed": 2.0, "Burstable": 5.0, "BE": 20.0}
QOS_FLOOR_DEFAULT = 1.5

# A task whose absolute slack budget is below this is violated by any realistic
# scheduling delay (wake latency alone is 5 s in PowerSpec).
SLACK_TIGHT_SEC = 60.0


# ════════════════════════════════════════════════════════════════════════════
#  Trace model
# ════════════════════════════════════════════════════════════════════════════

@dataclass(frozen=True, slots=True)
class Task:
    name: str
    cpu_milli: int
    memory_mib: int
    num_gpu: int
    gpu_milli: int
    qos: str
    pod_phase: str
    creation_time: float
    deletion_time: float
    scheduled_time: float

    @property
    def duration(self) -> float:
        return max(0.0, self.deletion_time - max(self.creation_time, self.scheduled_time))

    @property
    def pes(self) -> int:
        return max(1, self.cpu_milli // 1000)

    @property
    def needs_gpu(self) -> bool:
        """Affinity signal — whole cards OR a fractional share (G2.2)."""
        return self.num_gpu > 0 or self.gpu_milli > 0

    @property
    def slack_budget(self) -> float:
        """Absolute lateness tolerated before C_SLA is charged, in seconds.

        A task that starts at once finishes at creation + duration, so the budget is
        the bounded-slowdown allowance duration * (f - 1) plus the absolute floor.
        Without the floor the allowance is proportional to the job's own length and a
        short job gets essentially none.
        """
        f = QOS_SLACK_FACTOR.get(self.qos, QOS_SLACK_DEFAULT)
        floor = SCHEDULING_LAG_P90_SEC * QOS_FLOOR_FACTOR.get(self.qos, QOS_FLOOR_DEFAULT)
        return self.duration * (f - 1.0) + floor

    @property
    def slack_budget_no_floor(self) -> float:
        """The pre-W1.5 budget, kept so the 'before' number stays auditable."""
        return self.duration * (QOS_SLACK_FACTOR.get(self.qos, QOS_SLACK_DEFAULT) - 1.0)


def _pint(s: str, fallback: int = 0) -> int:
    s = s.strip()
    if not s or s.lower() == "nan":
        return fallback
    return int(float(s))


def _pfloat(s: str, fallback: float = 0.0) -> float:
    s = s.strip()
    if not s or s.lower() == "nan":
        return fallback
    return float(s)


def read_trace(path: str) -> list[Task]:
    """Parse an openb-schema CSV. Bad rows are skipped, matching the Java reader."""
    tasks: list[Task] = []
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        next(reader, None)  # header
        for row in reader:
            if len(row) < 11:
                continue
            try:
                tasks.append(Task(
                    name=row[0].strip(),
                    cpu_milli=_pint(row[1]),
                    memory_mib=_pint(row[2]),
                    num_gpu=_pint(row[3]),
                    gpu_milli=_pint(row[4]),
                    qos=row[6].strip(),
                    pod_phase=row[7].strip(),
                    creation_time=_pfloat(row[8]),
                    deletion_time=_pfloat(row[9]),
                    scheduled_time=_pfloat(row[10]),
                ))
            except (ValueError, IndexError):
                continue
    tasks.sort(key=lambda t: t.creation_time)
    return tasks


# ════════════════════════════════════════════════════════════════════════════
#  Scenario filters — byte-for-byte mirror of ScenarioFilter.java
# ════════════════════════════════════════════════════════════════════════════

def filter_scenario(tasks: list[Task], scenario: str, *, drop_failed: bool = False) -> list[Task]:
    schedulable = [t for t in tasks if t.pod_phase != "Pending"]
    if drop_failed:
        schedulable = [t for t in schedulable if t.pod_phase != "Failed"]

    scenario = scenario.upper()
    if scenario in ("ALL", "LEGACY_HIGH", "HIGH"):
        # ALL and LEGACY_HIGH coincide on a legacy trace; on a WM-1 trace the
        # file already *is* one scenario, so ALL is the passthrough to use.
        return list(schedulable)
    if scenario in ("LEGACY_LOW", "LOW"):
        if not schedulable:
            return []
        count = max(1, int(len(schedulable) * LOW_FRACTION))
        return list(schedulable[:count])
    if scenario in ("LEGACY_BURST", "BURST"):
        return _filter_burst(schedulable)
    raise ValueError(f"unknown scenario: {scenario}")


def _filter_burst(tasks: list[Task]) -> list[Task]:
    if not tasks:
        return []
    min_t = tasks[0].creation_time
    windows: dict[int, list[Task]] = {}
    for t in tasks:
        windows.setdefault(int((t.creation_time - min_t) / BURST_WINDOW_SEC), []).append(t)
    sizes = sorted(len(w) for w in windows.values())
    idx = int(len(sizes) * BURST_PERCENTILE)
    threshold = sizes[min(idx, len(sizes) - 1)]
    result = [t for w in windows.values() if len(w) >= threshold for t in w]
    result.sort(key=lambda t: t.creation_time)
    return result


# ════════════════════════════════════════════════════════════════════════════
#  Cluster capacity
# ════════════════════════════════════════════════════════════════════════════

@dataclass(frozen=True, slots=True)
class Capacity:
    label: str
    hosts: int
    total_pes: int
    total_gpus: int


def capacity_from_args(hosts: int, pes: int, gpus: int) -> Capacity:
    return Capacity(f"homogeneous {hosts}x({pes} PE, {gpus} GPU)",
                    hosts, hosts * pes, hosts * gpus)


def capacity_from_topology(path: str, dflt_pes: int, dflt_gpus: int) -> Capacity:
    """Mirror of TopologyConfig.load(): absent SKU fields inherit the defaults."""
    with open(path, encoding="utf-8") as f:
        cfg = json.load(f)
    skus = cfg.get("skus") or []
    if not skus:
        raise ValueError(f"{path}: no 'skus' array")
    hosts = total_pe = total_gpu = 0
    parts = []
    for i, s in enumerate(skus):
        count = int(s.get("count") or 0)
        if count <= 0:
            raise ValueError(f"{path}: SKU {i} has count <= 0")
        vcpu = max(1, int(s.get("vcpu", dflt_pes)))
        gpu = max(0, int(s.get("gpu", dflt_gpus)))
        hosts += count
        total_pe += count * vcpu
        total_gpu += count * gpu
        parts.append(f"{count}x{s.get('name', f'sku-{i}')}({vcpu} PE,{gpu} GPU)")
    return Capacity(f"{cfg.get('name', os.path.basename(path))}: " + " + ".join(parts),
                    hosts, total_pe, total_gpu)


# ════════════════════════════════════════════════════════════════════════════
#  Statistics helpers
# ════════════════════════════════════════════════════════════════════════════

def _quantile(sorted_vals: list[float], p: float) -> float:
    if not sorted_vals:
        return float("nan")
    return sorted_vals[min(len(sorted_vals) - 1, max(0, int(p * (len(sorted_vals) - 1))))]


def _mean_var(vals: list[float]) -> tuple[float, float]:
    """Population mean and variance (matches the IDC/C_a^2 definitions)."""
    n = len(vals)
    if n == 0:
        return float("nan"), float("nan")
    m = sum(vals) / n
    return m, sum((v - m) ** 2 for v in vals) / n


def index_of_dispersion(times: list[float], window: float, t0: float, t1: float) -> float:
    """IDC = Var(N_w) / E(N_w) over fixed-width windows.

    Empty windows MUST be counted: the openb trace leaves 77 % of its 1 h windows
    empty, and dropping them collapses the variance and understates burstiness by
    an order of magnitude. Poisson => IDC = 1; bursty => IDC >> 1.
    """
    span = max(t1 - t0, 0.0)
    n_win = max(1, int(math.ceil(span / window)) if span > 0 else 1)
    counts = [0] * n_win
    for t in times:
        counts[min(n_win - 1, int((t - t0) / window))] += 1
    m, v = _mean_var([float(c) for c in counts])
    return v / m if m > 0 else float("nan")


def squared_cv_interarrival(times: list[float]) -> float:
    """C_a^2 = Var(inter-arrival) / mean(inter-arrival)^2. Poisson => 1."""
    if len(times) < 3:
        return float("nan")
    gaps = [times[i + 1] - times[i] for i in range(len(times) - 1)]
    m, v = _mean_var(gaps)
    return v / (m * m) if m > 0 else float("nan")


def demand_curve(tasks: list[Task], t0: float, t1: float, demand=None) -> list[tuple[float, float]]:
    """Time-weighted distribution of concurrent demand over [t0, t1].

    `demand` maps a task to the amount it holds (default: CPU PEs; pass
    `lambda t: t.num_gpu` for whole GPU cards). Returns [(level, seconds_at_level)].
    Occupancy interval of a task is [creation, creation + duration] clipped to the
    accounting window.
    """
    if t1 <= t0:
        return []
    demand = demand or (lambda t: t.pes)
    events: list[tuple[float, int]] = []
    for t in tasks:
        d = demand(t)
        if d <= 0:
            continue
        s = max(t0, t.creation_time)
        e = min(t1, t.creation_time + t.duration)
        if e > s:
            events.append((s, d))
            events.append((e, -d))
    if not events:
        return [(0.0, t1 - t0)]
    events.sort(key=lambda x: x[0])

    weights: dict[float, float] = {}
    level = 0.0
    prev = t0
    i = 0
    while i < len(events):
        t = events[i][0]
        if t > prev:
            weights[level] = weights.get(level, 0.0) + (t - prev)
            prev = t
        while i < len(events) and events[i][0] == t:      # all events at this instant
            level += events[i][1]
            i += 1
    if t1 > prev:
        weights[level] = weights.get(level, 0.0) + (t1 - prev)
    return sorted(weights.items())


def weighted_quantile(curve: list[tuple[float, float]], p: float) -> float:
    total = sum(w for _, w in curve)
    if total <= 0:
        return float("nan")
    target, acc = p * total, 0.0
    for level, w in curve:
        acc += w
        if acc >= target:
            return level
    return curve[-1][0]


# ════════════════════════════════════════════════════════════════════════════
#  Characterisation
# ════════════════════════════════════════════════════════════════════════════

def characterise(tasks: list[Task], cap: Capacity, windows: list[float],
                 horizon_override: float | None = None) -> dict:
    n = len(tasks)
    if n == 0:
        return {"n_tasks": 0}

    creations = [t.creation_time for t in tasks]
    t0 = creations[0]
    # For a *generated* trace the horizon is a known design parameter, not something to
    # infer: a bursty stream can fall silent well before T, and inferring the horizon
    # from the last arrival would shrink the denominator and overstate the load (BURST
    # measured 0.886 against a 0.85 target for exactly this reason). Pass --horizon-days
    # whenever the trace has a designed length.
    t1 = (t0 + horizon_override) if horizon_override else creations[-1]
    horizon = t1 - t0                       # accounting window (see module docstring)
    horizon_eff = max(horizon, 1.0)         # guard for degenerate single-arrival sets

    durations = sorted(t.duration for t in tasks)
    d_mean, d_var = _mean_var(durations)
    d_cv = math.sqrt(d_var) / d_mean if d_mean > 0 else float("nan")

    cpu_work = sum(t.pes * t.duration for t in tasks)            # PE-seconds
    gpu_work = sum(t.num_gpu * t.duration for t in tasks)        # card-seconds (whole cards)
    gpu_work_aff = sum(max(t.num_gpu, 1 if t.gpu_milli > 0 else 0) * t.duration
                       for t in tasks)                          # counting shares as 1 card

    rho_cpu = cpu_work / (cap.total_pes * horizon_eff)
    rho_gpu = gpu_work / (cap.total_gpus * horizon_eff) if cap.total_gpus else float("nan")
    rho_gpu_aff = (gpu_work_aff / (cap.total_gpus * horizon_eff)
                   if cap.total_gpus else float("nan"))

    # heavy tail — share of total CPU-work carried by the largest tasks
    work_desc = sorted((t.pes * t.duration for t in tasks), reverse=True)
    total_work = sum(work_desc) or 1.0
    share = lambda frac: sum(work_desc[:max(1, int(n * frac))]) / total_work

    slack = sorted(t.slack_budget for t in tasks)

    curve = demand_curve(tasks, t0, t1)
    peak_pe = max((lvl for lvl, _ in curve), default=0.0)
    sat_time = sum(w for lvl, w in curve if lvl > cap.total_pes)
    total_time = sum(w for _, w in curve) or 1.0

    # GPU is a separate capacity axis and can be the binding one: on the 3-SKU
    # hetero topology there are 18 cards vs 80 in the homogeneous default, so a
    # CPU-only saturation figure would be misleading there.
    gcurve = demand_curve(tasks, t0, t1, demand=lambda t: t.num_gpu)
    peak_gpu = max((lvl for lvl, _ in gcurve), default=0.0)
    gsat_time = (sum(w for lvl, w in gcurve if lvl > cap.total_gpus)
                 if cap.total_gpus else 0.0)
    gtotal_time = sum(w for _, w in gcurve) or 1.0

    counter = lambda key: {k: v for k, v in sorted(
        ((k, sum(1 for t in tasks if key(t) == k)) for k in {key(t) for t in tasks}),
        key=lambda kv: -kv[1])}

    return {
        "n_tasks": n,
        "horizon_sec": horizon,
        "horizon_days": horizon / SEC_PER_DAY,
        "makespan_days": (max(t.creation_time + t.duration for t in tasks) - t0) / SEC_PER_DAY,
        "arrival_rate_per_hour": n / (horizon_eff / SEC_PER_HOUR),

        "rho_cpu": rho_cpu,
        "rho_gpu": rho_gpu,
        "rho_gpu_affinity": rho_gpu_aff,
        "cpu_work_pe_sec": cpu_work,
        "gpu_work_card_sec": gpu_work,

        "dur_mean_h": d_mean / SEC_PER_HOUR,
        "dur_p50_h": _quantile(durations, 0.50) / SEC_PER_HOUR,
        "dur_p90_h": _quantile(durations, 0.90) / SEC_PER_HOUR,
        "dur_p99_h": _quantile(durations, 0.99) / SEC_PER_HOUR,
        "dur_max_h": durations[-1] / SEC_PER_HOUR,
        "dur_cv": d_cv,

        "ca2": squared_cv_interarrival(creations),
        "idc": {f"{int(w)}": index_of_dispersion(creations, w, t0, t1) for w in windows},

        "peak_pe_demand": peak_pe,
        "peak_pe_ratio": peak_pe / cap.total_pes,
        "p95_pe_ratio": weighted_quantile(curve, 0.95) / cap.total_pes,
        "p50_pe_ratio": weighted_quantile(curve, 0.50) / cap.total_pes,
        "saturated_time_frac": sat_time / total_time,

        "peak_gpu_demand": peak_gpu,
        "peak_gpu_ratio": peak_gpu / cap.total_gpus if cap.total_gpus else float("nan"),
        "p95_gpu_ratio": (weighted_quantile(gcurve, 0.95) / cap.total_gpus
                          if cap.total_gpus else float("nan")),
        "saturated_gpu_time_frac": gsat_time / gtotal_time,

        "work_share_top1pct": share(0.01),
        "work_share_top10pct": share(0.10),
        "n_top1pct": max(1, int(n * 0.01)),

        "slack_p50_sec": _quantile(slack, 0.50),
        "slack_p10_sec": _quantile(slack, 0.10),
        "slack_tight_frac": sum(1 for s in slack if s < SLACK_TIGHT_SEC) / n,

        "gpu_task_frac": sum(1 for t in tasks if t.needs_gpu) / n,
        "qos_mix": counter(lambda t: t.qos),
        "phase_mix": counter(lambda t: t.pod_phase),
    }


# ════════════════════════════════════════════════════════════════════════════
#  Automatic findings — the acceptance checks a load-scenario set must pass
# ════════════════════════════════════════════════════════════════════════════

#: two scenarios count as distinct load levels only if their offered loads differ
#: by at least this much, relative to the larger one.
MIN_PAIRWISE_RHO_SEP = 0.30


def findings(res: dict, names: list[str]) -> list[dict]:
    """Return the acceptance checks as dicts with EN (console) and VI (report) text.

    These are the same checks WM-1 must satisfy (PLAN-Workload-Model.md section 2.4),
    so re-running the tool on a generated trace shows the fix landing. The console
    strings stay pure ASCII (cp1252 on Windows); the Vietnamese ones go to markdown.
    """
    out: list[dict] = []

    def add(verdict, check_en, detail_en, check_vi, detail_vi):
        out.append({"verdict": verdict,
                    "check": {"en": check_en, "vi": check_vi},
                    "detail": {"en": detail_en, "vi": detail_vi}})

    live = [nm for nm in names if res[nm].get("n_tasks")]
    if not live:
        return out

    rhos = {nm: res[nm]["rho_cpu"] for nm in live}
    if len(live) > 1:
        # PAIRWISE, not min-vs-max: one well-separated outlier must not mask a
        # collapsed pair. On the legacy slices LOW and HIGH sit on top of each
        # other while BURST is far away, and a min/max spread would report PASS.
        pairs = [(a, b, abs(rhos[a] - rhos[b]) / max(rhos[a], rhos[b], 1e-12))
                 for i, a in enumerate(live) for b in live[i + 1:]]
        wa, wb, ws = min(pairs, key=lambda p: p[2])
        allp = "; ".join(f"{a} vs {b}: {s:.1%}" for a, b, s in pairs)
        add("PASS" if ws >= MIN_PAIRWISE_RHO_SEP else "FAIL",
            "every scenario pair separates on offered load",
            f"closest pair {wa} ({rhos[wa]:.3f}) vs {wb} ({rhos[wb]:.3f}) differ by only "
            f"{ws:.1%}, need >= {MIN_PAIRWISE_RHO_SEP:.0%} - all pairs: {allp}",
            "mọi cặp scenario phải tách biệt về offered load",
            f"cặp gần nhau nhất `{wa}` (ρ={rhos[wa]:.3f}) và `{wb}` (ρ={rhos[wb]:.3f}) chỉ "
            f"chênh **{ws:.1%}**, cần ≥ {MIN_PAIRWISE_RHO_SEP:.0%} mới gọi là hai mức tải "
            f"khác nhau · tất cả các cặp: {allp}")

        hz = {nm: res[nm]["horizon_days"] for nm in live}
        h_spread = ((max(hz.values()) - min(hz.values())) / max(hz.values())
                    if max(hz.values()) > 0 else 0.0)
        hz_txt = ", ".join(f"{nm}={hz[nm]:.1f}d" for nm in live)
        add("PASS" if h_spread <= 0.05 else "FAIL",
            "horizon comparable across scenarios",
            f"spread {h_spread:.1%} ({hz_txt}); energy in kWh is not comparable when the "
            f"billed window differs",
            "horizon so sánh được giữa các scenario",
            f"chênh **{h_spread:.1%}** ({hz_txt}) — energy (kWh) **không so sánh được** khi "
            f"cửa sổ tính năng lượng khác nhau")

    for nm in live:
        r = res[nm]
        # Consider BOTH capacity axes and name the binding one: on the hetero
        # topology GPU (18 cards) binds long before CPU (640 PEs).
        sat = max(r["saturated_time_frac"], r["saturated_gpu_time_frac"])
        gpu_binds = r["saturated_gpu_time_frac"] > r["saturated_time_frac"] or (
            not math.isnan(r["peak_gpu_ratio"]) and r["peak_gpu_ratio"] > r["peak_pe_ratio"])
        axis_en, axis_vi = ("GPU", "GPU") if gpu_binds else ("CPU", "CPU")
        peak = r["peak_gpu_ratio"] if gpu_binds else r["peak_pe_ratio"]
        both = (f"CPU {r['peak_pe_ratio']:.2f}x / GPU {_fmt(r['peak_gpu_ratio'], '{:.2f}')}x")
        if sat < 0.01 and peak < 1.5:
            add("INFO", f"{nm}: cluster almost never saturated (binding: {axis_en})",
                f"demand exceeds capacity {sat:.1%} of the time (peak {both}) - SLA "
                f"violations cannot be attributed to resource contention",
                f"`{nm}`: cụm gần như không bao giờ bão hoà (nghẽn ở {axis_vi})",
                f"nhu cầu vượt năng lực chỉ **{sat:.1%}** thời gian (đỉnh {both}) ⇒ vi phạm "
                f"SLA **không thể** quy cho tranh chấp tài nguyên")
        else:
            add("INFO", f"{nm}: {axis_en} is the binding resource",
                f"demand exceeds {axis_en} capacity {sat:.1%} of the time (peak {both})",
                f"`{nm}`: {axis_vi} là tài nguyên nghẽn cổ chai",
                f"nhu cầu vượt năng lực {axis_vi} **{sat:.1%}** thời gian (đỉnh {both}) ⇒ "
                f"đây mới là trục tạo ra tranh chấp thật")
        # Over-subscription is not just a load statement: SimulationManager drops
        # a task no host can take with reward {0,0} and NO C_SLA charge, so an
        # over-subscribed axis makes the constraint metric under-count.
        if r["saturated_gpu_time_frac"] > 0.01:
            add("FAIL", f"{nm}: offered GPU demand exceeds capacity",
                f"GPU demand is over capacity {r['saturated_gpu_time_frac']:.1%} of the "
                f"time (peak {r['peak_gpu_ratio']:.2f}x). Tasks no host can take are "
                f"dropped with reward {{0,0}} and no C_SLA charge, so the constraint "
                f"metric under-counts exactly where the cluster is most stressed",
                f"`{nm}`: nhu cầu GPU vượt quá năng lực cụm",
                f"nhu cầu GPU vượt năng lực **{r['saturated_gpu_time_frac']:.1%}** thời gian "
                f"(đỉnh **{r['peak_gpu_ratio']:.2f}×**). Task không host nào nhận được bị "
                f"`SimulationManager` **drop với reward {{0,0}} và KHÔNG bị tính C_SLA** ⇒ "
                f"chỉ số ràng buộc **đếm thiếu đúng ở vùng cụm căng nhất**")
        if r["work_share_top1pct"] > 0.50:
            add("FAIL", f"{nm}: heavy tail dominates",
                f"top 1% ({r['n_top1pct']} tasks) carry {r['work_share_top1pct']:.0%} of "
                f"CPU-work - effective sample size is far below n={r['n_tasks']}",
                f"`{nm}`: đuôi nặng chi phối",
                f"top 1% (**{r['n_top1pct']} task**) gánh **{r['work_share_top1pct']:.0%}** "
                f"tổng CPU-work ⇒ cỡ mẫu hữu hiệu thấp hơn rất nhiều so với n={r['n_tasks']:,}")
        add("INFO", f"{nm}: SLA budget is duration-proportional",
            f"deadline = creation + duration * f(qos) + floor(qos); absolute budget median "
            f"{r['slack_p50_sec']:.0f}s and {r['slack_tight_frac']:.0%} of tasks tolerate "
            f"< {SLACK_TIGHT_SEC:.0f}s - short jobs are violated by the 5s wake latency "
            f"alone, independently of load",
            f"`{nm}`: ngân sách SLA tỉ lệ với chính duration",
            f"`deadline = creation + duration · f(qos) + floor(qos)` ⇒ slack tuyệt đối median "
            f"**{r['slack_p50_sec']:.0f}s**, và **{r['slack_tight_frac']:.0%}** task chịu được "
            f"< {SLACK_TIGHT_SEC:.0f}s ⇒ job ngắn bị vi phạm chỉ vì độ trễ wake 5s, "
            f"**độc lập với mức tải**")
    return out


# ════════════════════════════════════════════════════════════════════════════
#  Rendering
# ════════════════════════════════════════════════════════════════════════════

def _fmt(v, spec="{:.3f}"):
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return "n/a"
    return spec.format(v)


# Console output stays pure ASCII: the Windows console is cp1252 and cannot
# print rho / Delta / arrows (Tracking_detail.md section 15.4).
ROWS_TEXT = [
    ("#tasks",                  "n_tasks",                "{:d}"),
    ("horizon (days)",          "horizon_days",           "{:.1f}"),
    ("makespan (days)",         "makespan_days",          "{:.1f}"),
    ("arrival (task/h)",        "arrival_rate_per_hour",  "{:.2f}"),
    ("rho_cpu",                 "rho_cpu",                "{:.3f}"),
    ("rho_gpu (cards)",         "rho_gpu",                "{:.3f}"),
    ("rho_gpu (affinity)",      "rho_gpu_affinity",       "{:.3f}"),
    ("peak PE / capacity",      "peak_pe_ratio",          "{:.2f}"),
    ("p95 PE / capacity",       "p95_pe_ratio",           "{:.2f}"),
    ("time CPU saturated",      "saturated_time_frac",    "{:.1%}"),
    ("peak GPU / capacity",     "peak_gpu_ratio",         "{:.2f}"),
    ("time GPU saturated",      "saturated_gpu_time_frac", "{:.1%}"),
    ("duration mean (h)",       "dur_mean_h",             "{:.2f}"),
    ("duration p50 (h)",        "dur_p50_h",              "{:.2f}"),
    ("duration p99 (h)",        "dur_p99_h",              "{:.1f}"),
    ("duration CV",             "dur_cv",                 "{:.1f}"),
    ("C_a^2 (interarrival)",    "ca2",                    "{:.1f}"),
    ("work share top 1%",       "work_share_top1pct",     "{:.1%}"),
    ("work share top 10%",      "work_share_top10pct",    "{:.1%}"),
    ("SLA slack p50 (s)",       "slack_p50_sec",          "{:.0f}"),
    ("SLA slack < 60s",         "slack_tight_frac",       "{:.1%}"),
    ("GPU-task fraction",       "gpu_task_frac",          "{:.1%}"),
]


def render_text(report: dict) -> str:
    names = report["scenarios_order"]
    res = report["scenarios"]
    out: list[str] = []
    out.append("=" * 78)
    out.append("ELDAS workload characterisation (W0.1)")
    out.append("=" * 78)
    out.append(f"trace     : {report['trace']}")
    out.append(f"sha256    : {report['trace_sha256'][:16]}...  ({report['trace_rows']} rows)")
    out.append(f"capacity  : {report['capacity']['label']}")
    out.append(f"            {report['capacity']['total_pes']} PE, "
               f"{report['capacity']['total_gpus']} GPU across "
               f"{report['capacity']['hosts']} hosts")
    out.append("")

    w = max(len(r[0]) for r in ROWS_TEXT) + 1
    header = "  ".join(f"{n:>14}" for n in names)
    out.append(f"{'metric':<{w}}{header}")
    out.append("-" * (w + len(header)))
    for label, key, spec in ROWS_TEXT:
        cells = []
        for nm in names:
            v = res[nm].get(key)
            cells.append(f"{_fmt(v, spec) if v is not None else 'n/a':>14}")
        out.append(f"{label:<{w}}" + "  ".join(cells))

    out.append("")
    out.append("IDC = Var(N_w)/E(N_w) over fixed windows (Poisson = 1.0; empty windows counted)")
    for wsec in report["windows"]:
        key = str(int(wsec))
        cells = "  ".join(f"{_fmt(res[nm]['idc'].get(key)):>14}" for nm in names)
        out.append(f"{'IDC @ ' + _win_label(wsec):<{w}}" + cells)

    out.append("")
    out.append("composition")
    for nm in names:
        r = res[nm]
        out.append(f"  {nm:<14} qos={r['qos_mix']}")
        out.append(f"  {'':<14} phase={r['phase_mix']}")

    if report.get("sensitivity"):
        out.append("")
        out.append("sensitivity: rho_cpu when 'Failed' pods are excluded")
        for nm in names:
            base = res[nm]["rho_cpu"]
            alt = report["sensitivity"][nm]["rho_cpu"]
            out.append(f"  {nm:<14} {base:.3f} -> {alt:.3f}  "
                       f"(n {res[nm]['n_tasks']} -> {report['sensitivity'][nm]['n_tasks']})")

    if report.get("findings"):
        out.append("")
        out.append("findings (acceptance checks for a load-scenario set)")
        for f in report["findings"]:
            out.append(f"  [{f['verdict']:<4}] {f['check']['en']}")
            out.append(f"           {f['detail']['en']}")
    out.append("")
    return "\n".join(out)


def _win_label(sec: float) -> str:
    if sec % SEC_PER_DAY == 0:
        return f"{int(sec // SEC_PER_DAY)}d"
    if sec % SEC_PER_HOUR == 0:
        return f"{int(sec // SEC_PER_HOUR)}h"
    return f"{int(sec // 60)}min"


ROWS_MD = [
    ("#task",                          "n_tasks",                "{:,d}"),
    ("Horizon (ngày)",                 "horizon_days",           "{:.1f}"),
    ("Makespan (ngày)",                "makespan_days",          "{:.1f}"),
    ("Arrival rate (task/h)",          "arrival_rate_per_hour",  "{:.2f}"),
    ("**ρ_cpu (offered load)**",       "rho_cpu",                "**{:.3f}**"),
    ("ρ_gpu (card nguyên)",            "rho_gpu",                "{:.3f}"),
    ("ρ_gpu (tính cả share)",          "rho_gpu_affinity",       "{:.3f}"),
    ("Peak PE / capacity",             "peak_pe_ratio",          "{:.2f}"),
    ("p95 PE / capacity",              "p95_pe_ratio",           "{:.2f}"),
    ("% thời gian CPU bão hoà",        "saturated_time_frac",    "{:.1%}"),
    ("Peak GPU / capacity",            "peak_gpu_ratio",         "{:.2f}"),
    ("% thời gian GPU bão hoà",        "saturated_gpu_time_frac", "{:.1%}"),
    ("Duration mean (h)",              "dur_mean_h",             "{:.2f}"),
    ("Duration median (h)",            "dur_p50_h",              "{:.2f}"),
    ("Duration p99 (h)",               "dur_p99_h",              "{:.1f}"),
    ("Duration CV",                    "dur_cv",                 "{:.1f}"),
    ("C_a² (inter-arrival)",           "ca2",                    "{:.1f}"),
    ("Work share top 1%",              "work_share_top1pct",     "{:.1%}"),
    ("Work share top 10%",             "work_share_top10pct",    "{:.1%}"),
    ("Slack SLA tuyệt đối p50 (s)",    "slack_p50_sec",          "{:.0f}"),
    ("% task có slack < 60 s",         "slack_tight_frac",       "{:.1%}"),
    ("Task cần GPU",                   "gpu_task_frac",          "{:.1%}"),
]

_VERDICT_ICON = {"PASS": "✅ PASS", "FAIL": "❌ FAIL", "INFO": "⚠️ INFO"}


def render_markdown(report: dict) -> str:
    names = report["scenarios_order"]
    res = report["scenarios"]
    cap = report["capacity"]
    o: list[str] = []

    o.append("# Đặc trưng workload đầu vào — ELDAS (W0.1/W0.2)")
    o.append("")
    o.append("> Sinh tự động bởi `scripts/characterize-workload.py`. **Không sửa tay** — "
             "chạy lại lệnh ở §5 để cập nhật.")
    o.append("")
    o.append(f"- Trace: `{report['trace']}` ({report['trace_rows']} dòng dữ liệu)")
    o.append(f"- SHA-256: `{report['trace_sha256']}`")
    o.append(f"- Cấu hình cluster: {cap['label']} "
             f"⇒ **{cap['total_pes']} PE**, **{cap['total_gpus']} GPU**, {cap['hosts']} host")
    o.append(f"- Sinh lúc: {report['generated_at']}"
             + (f" · git `{report['git_sha']}`" if report.get("git_sha") else ""))
    o.append("")
    for note in report.get("notes", []):
        o.append(f"> {note}")
        o.append("")

    o.append("## 1. Định nghĩa các đại lượng")
    o.append("")
    o.append("Mọi công thức khớp với phía Java để bảng mô tả **đúng** mô phỏng đang chạy:")
    o.append("")
    o.append("| Đại lượng | Công thức | Khớp với |")
    o.append("|---|---|---|")
    o.append("| `duration` | `max(0, deletion − max(creation, scheduled))` | "
             "`AlibabaTraceReader.TaskRecord.duration()` |")
    o.append("| `pes` | `max(1, cpu_milli / 1000)` | `TaskRecord.pesNeeded()` |")
    o.append("| GPU chiếm dụng | `num_gpu` (card nguyên) | "
             "`SimulationManager.canHost()` — `gpu_milli` chỉ quyết định *affinity* |")
    o.append("| Horizon `T` | `max(creation) − min(creation)` | "
             "episode kết thúc ở arrival cuối; energy tích phân tới đó |")
    o.append("| **Offered load** `ρ` | `Σ(demand_i · duration_i) / (capacity · T)` | "
             "utilization law; cách DeepRM (HotNets'16) định nghĩa mức tải |")
    o.append("| `C_a²` | `Var(inter-arrival) / mean(inter-arrival)²` | Poisson ⇒ 1 |")
    o.append("| `IDC(w)` | `Var(N_w) / E(N_w)`, **đếm cả cửa sổ rỗng** | "
             "Poisson ⇒ 1; bursty ⇒ ≫ 1 (Casale/Mi/Smirni, Middleware'08) |")
    o.append("| Slack SLA tuyệt đối | `duration · (f(qos) − 1)` | "
             "`SimulationConfig.qosToSlackFactor()` — LS 1.1 / Guaranteed 1.3 / "
             "Burstable 1.7 / BE 3.0 |")
    o.append("")
    o.append("⚠️ Cửa sổ rỗng **bắt buộc** phải được đếm khi tính IDC: trace hiện tại có "
             "77% cửa sổ 1 giờ là rỗng, bỏ chúng đi sẽ làm sập phương sai và báo cáo "
             "thiếu burstiness cả một bậc độ lớn.")
    o.append("")
    o.append("Bộ lọc `LEGACY_*` tái tạo nguyên văn `ScenarioFilter.java` "
             "(`LOW` = 25% task đầu theo `creation_time`; `HIGH` = toàn bộ task khả lập lịch; "
             "`BURST` = cửa sổ 1h có số task ≥ phân vị 80). "
             "`ALL` là passthrough — dùng cho trace WM-1 (mỗi file đã là một scenario).")
    o.append("")

    o.append("## 2. Bảng đặc trưng chính")
    o.append("")
    o.append("| Đại lượng | " + " | ".join(names) + " |")
    o.append("|---" * (len(names) + 1) + "|")
    for label, key, spec in ROWS_MD:
        cells = [_fmt(res[nm].get(key), spec) for nm in names]
        o.append(f"| {label} | " + " | ".join(cells) + " |")
    o.append("")

    if report.get("findings"):
        o.append("## 2.1 Kiểm tra nghiệm thu (tự động)")
        o.append("")
        o.append("Đây là bộ kiểm tra mà **một tập scenario tải hợp lệ phải vượt qua** "
                 "(PLAN-Workload-Model.md §2.4). Chạy lại công cụ trên trace WM-1 để "
                 "thấy các mục ❌ chuyển thành ✅.")
        o.append("")
        o.append("| Kết quả | Kiểm tra | Chi tiết |")
        o.append("|---|---|---|")
        for f in report["findings"]:
            o.append(f"| {_VERDICT_ICON.get(f['verdict'], f['verdict'])} "
                     f"| {f['check']['vi']} | {f['detail']['vi']} |")
        o.append("")

    o.append("## 3. Burstiness theo nhiều thang thời gian")
    o.append("")
    o.append("Burstiness **phụ thuộc thang đo**, nên phải báo cáo nhiều cửa sổ thay vì một.")
    o.append("")
    o.append("| IDC @ cửa sổ | " + " | ".join(names) + " |")
    o.append("|---" * (len(names) + 1) + "|")
    for wsec in report["windows"]:
        key = str(int(wsec))
        cells = [_fmt(res[nm]["idc"].get(key)) for nm in names]
        o.append(f"| {_win_label(wsec)} | " + " | ".join(cells) + " |")
    o.append("")

    o.append("## 4. Thành phần workload")
    o.append("")
    o.append("| Scenario | QoS mix | pod_phase mix |")
    o.append("|---|---|---|")
    for nm in names:
        r = res[nm]
        qos = ", ".join(f"{k} {v}" for k, v in r["qos_mix"].items())
        ph = ", ".join(f"{k} {v}" for k, v in r["phase_mix"].items())
        o.append(f"| {nm} | {qos} | {ph} |")
    o.append("")

    if report.get("sensitivity"):
        o.append("### 4.1 Độ nhạy — loại pod `Failed`")
        o.append("")
        o.append("Bộ lọc hiện tại chỉ loại `Pending`, nên vòng đời của pod **lỗi** "
                 "vẫn được tính như nhu cầu dịch vụ hợp lệ. Ảnh hưởng lên `ρ_cpu`:")
        o.append("")
        o.append("| Scenario | ρ_cpu (như hiện tại) | ρ_cpu (bỏ Failed) | #task |")
        o.append("|---|---:|---:|---|")
        for nm in names:
            s = report["sensitivity"][nm]
            o.append(f"| {nm} | {res[nm]['rho_cpu']:.3f} | {s['rho_cpu']:.3f} | "
                     f"{res[nm]['n_tasks']:,} → {s['n_tasks']:,} |")
        o.append("")
        o.append("Với `LEGACY_BURST`, bộ lọc được **chạy lại** sau khi loại `Failed`, nên "
                 "ngưỡng phân vị 80 và tập cửa sổ được chọn cũng đổi theo. Mức thay đổi "
                 "lớn của `ρ_cpu` ở dòng BURST vì vậy còn cho thấy bộ lọc burst **không "
                 "bền vững**: bỏ ~13% pod là đặc trưng tải của scenario đổi hẳn.")
        o.append("")

    o.append("## 5. Tái tạo")
    o.append("")
    o.append("```bash")
    o.append(report["command"])
    o.append("```")
    o.append("")
    return "\n".join(o)


# ════════════════════════════════════════════════════════════════════════════
#  CLI
# ════════════════════════════════════════════════════════════════════════════

def _sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _git_sha() -> str | None:
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                              capture_output=True, text=True, timeout=10,
                              check=True).stdout.strip() or None
    except Exception:
        return None


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="Characterise an ELDAS workload trace (offered load, burstiness, job sizes).")
    ap.add_argument("--trace", required=True, help="openb-schema CSV")
    ap.add_argument("--scenarios", default="LEGACY_LOW,LEGACY_HIGH,LEGACY_BURST",
                    help=f"comma list from {SCENARIOS} (default: the three legacy slices)")
    ap.add_argument("--hosts", type=int, default=DEFAULT_HOSTS)
    ap.add_argument("--pes-per-host", type=int, default=DEFAULT_PES_PER_HOST)
    ap.add_argument("--gpus-per-host", type=int, default=DEFAULT_GPUS_PER_HOST)
    ap.add_argument("--topology", default=None,
                    help="topology JSON (config/topology-hetero.json) — overrides --hosts/--*-per-host")
    ap.add_argument("--windows", default="600,3600,86400",
                    help="IDC window widths in seconds (default 10min,1h,1d)")
    ap.add_argument("--no-sensitivity", action="store_true",
                    help="skip the drop-Failed sensitivity block")
    ap.add_argument("--horizon-days", type=float, default=None,
                    help="designed horizon of a generated trace; without it the horizon "
                         "is inferred from the last arrival, which overstates the load "
                         "of a stream that falls silent early")
    ap.add_argument("--note", action="append", default=[],
                    help="free-text note rendered under the markdown header (repeatable) — "
                         "keeps the document fully generated instead of hand-edited")
    ap.add_argument("--format", choices=("text", "markdown", "json"), default="text")
    ap.add_argument("--out", default=None, help="write to file instead of stdout (UTF-8)")
    args = ap.parse_args(argv)

    if not os.path.isfile(args.trace):
        print(f"[ERROR] trace not found: {args.trace}", file=sys.stderr)
        return 2

    names = [s.strip().upper() for s in args.scenarios.split(",") if s.strip()]
    for nm in names:
        if nm not in SCENARIOS:
            print(f"[ERROR] unknown scenario {nm!r}; choose from {SCENARIOS}", file=sys.stderr)
            return 2

    try:
        windows = [float(w) for w in args.windows.split(",") if w.strip()]
    except ValueError:
        print(f"[ERROR] --windows must be numeric seconds, got {args.windows!r}", file=sys.stderr)
        return 2

    cap = (capacity_from_topology(args.topology, args.pes_per_host, args.gpus_per_host)
           if args.topology
           else capacity_from_args(args.hosts, args.pes_per_host, args.gpus_per_host))
    horizon_override = args.horizon_days * SEC_PER_DAY if args.horizon_days else None

    tasks = read_trace(args.trace)
    if not tasks:
        print(f"[ERROR] no parseable rows in {args.trace}", file=sys.stderr)
        return 1

    report = {
        "trace": args.trace.replace("\\", "/"),
        "trace_sha256": _sha256(args.trace),
        "trace_rows": len(tasks),
        "generated_at": _dt.datetime.now().strftime("%Y-%m-%d %H:%M"),
        "git_sha": _git_sha(),
        "capacity": {"label": cap.label, "hosts": cap.hosts,
                     "total_pes": cap.total_pes, "total_gpus": cap.total_gpus},
        "windows": windows,
        "notes": args.note,
        "scenarios_order": names,
        "horizon_override_sec": (args.horizon_days * SEC_PER_DAY
                                 if args.horizon_days else None),
        "scenarios": {nm: characterise(filter_scenario(tasks, nm), cap, windows,
                                       horizon_override)
                      for nm in names},
        # Rebuilt from argv so the reproduction line is exactly what was run,
        # including flags added to this tool later.
        "command": "python scripts/characterize-workload.py " + " ".join(
            shlex.quote(a.replace("\\", "/")) for a in (argv if argv is not None else sys.argv[1:])),
    }
    if not args.no_sensitivity:
        report["sensitivity"] = {
            nm: characterise(filter_scenario(tasks, nm, drop_failed=True), cap, windows,
                             horizon_override)
            for nm in names
        }
    report["findings"] = findings(report["scenarios"], names)

    if args.format == "json":
        text = json.dumps(report, indent=2, ensure_ascii=False)
    elif args.format == "markdown":
        text = render_markdown(report)
    else:
        text = render_text(report)

    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        with open(args.out, "w", encoding="utf-8", newline="\n") as f:
            f.write(text if text.endswith("\n") else text + "\n")
        print(f"[OK] wrote {args.out} ({args.format})")
    else:
        sys.stdout.reconfigure(errors="replace")
        print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
