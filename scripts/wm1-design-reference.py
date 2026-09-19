#!/usr/bin/env python3
"""WM-1 design reference — regenerates the locked numbers in PLAN-Workload-Model.md §3.

This is a deliberately small, dependency-free prototype of the WM-1 pipeline. It is NOT
the implementation (that is `rl-agent/src/workload/`). Its only job is to be the
*independent* source of the expected values that W1's acceptance tests compare against:
if `workload/` and this script disagree, one of them is wrong and work stops.

What it reproduces (PLAN §3):
  * §3.3 truncation moments  — E[work], E[gpu-work], fraction of jobs truncated at T
  * §3.5 burstiness anchors  — IDC of Philly/Helios, whole-trace and within a T window
  * §3.6 the scenario table  — fitted duty cycle p1, arrival count N, realised IDC

Design, all locked in the plan:
  job size   bootstrap whole rows from openb (Pending + Failed excluded), duration
             truncated at the arm horizon T
  arrivals   MMPP-2 ON/OFF modulator with mean ON = 3600 s; duty cycle p1 fitted by
             bisection so measured IDC(1h) hits the target; then EXACTLY N arrivals are
             placed uniformly on the ON set (fixed-N, PLAN §3.4)
  load       bottleneck / dominant-resource load: lambda = rho / max(E[work]/C_pe,
             E[gpu]/C_gpu), so the calibration is feasible on both topologies

Usage
-----
  python scripts/wm1-design-reference.py                 # scenario table (§3.6)
  python scripts/wm1-design-reference.py --anchors       # + burstiness anchors (§3.5)
  python scripts/wm1-design-reference.py --json          # machine-readable, for tests
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import statistics
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OPENB = os.path.join(ROOT, "data", "alibaba-trace", "openb_pod_list_default.csv")
REF_DIR = os.path.join(ROOT, "data", "reference-traces")

# ── Locked design constants (PLAN §3) ───────────────────────────────────────
E_ON_SEC = 3600.0            # mean ON sojourn of the MMPP-2 modulator
IDC_WINDOW = 3600.0          # scale at which the IDC target is defined
FIT_REPS = 12                # replicate streams averaged inside the bisection
FIT_ITERS = 32               # bisection iterations on p1
SEEDS = (42, 43, 44, 45, 46)

ARMS = {                     # arm -> (total PEs, total GPUs, horizon in days)
    "homo":   (640, 80,  4),
    "hetero": (640, 18, 12),
}
# W3.1 — largest single host per arm (PEs, RAM MiB, GPU cards). A task must fit ENTIRELY
# inside one host, so jobs above this can never be placed by any policy; they are excluded
# from the pool rather than left to produce guaranteed drops. Both arms use the same
# 64-vCPU / 256-GiB node; they differ only in the biggest GPU count (homo 8, hetero's
# gpu-heavy SKU 4).
HOST_LIMITS = {
    "homo":   (64, 256 * 1024, 8),
    "hetero": (64, 256 * 1024, 4),
}
SCENARIOS = (                # name, rho target, IDC(1h) target
    ("LOW",      0.30, 10.0),
    ("HIGH",     0.85, 10.0),
    ("BURST",    0.85, 40.0),
    ("OVERLOAD", 1.25, 10.0),
)


# ── Job-size pool ───────────────────────────────────────────────────────────

def load_pool(path: str) -> list[tuple[int, float, int, int, int]]:
    """(pes, duration, gpu_cards, ram_mib, gpu_milli) for schedulable, non-failed pods."""
    out = []
    with open(path, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if r["pod_phase"] in ("Pending", "Failed"):
                continue
            g = lambda k, d=0.0: (float(r[k]) if r[k].strip() not in ("", "nan") else d)
            c, dl, s = g("creation_time"), g("deletion_time"), g("scheduled_time")
            out.append((max(1, int(g("cpu_milli")) // 1000),
                        max(0.0, dl - max(c, s)),
                        int(g("num_gpu")),
                        int(g("memory_mib")),
                        int(g("gpu_milli"))))
    return out


def placeable(job, limits) -> bool:
    """Could an empty host of the largest size run this job? (W3.1)

    Mirrors ``SimulationManager.canHost`` on an idle host, GPU affinity included.
    """
    pes, _, gpus, ram, gpu_milli = job
    host_pes, host_ram, host_gpus = limits
    if pes > host_pes or ram > host_ram:
        return False
    if (gpus > 0 or gpu_milli > 0) and (host_gpus <= 0 or gpus > host_gpus):
        return False
    return True


def moments(pool, T: float, limits) -> tuple[float, float, float, float]:
    """E[pes*dur], E[gpu*dur], truncated fraction, unplaceable fraction."""
    fits = [j for j in pool if placeable(j, limits)]
    unplaceable = (len(pool) - len(fits)) / len(pool)
    trunc = sum(1 for _, d, _, _, _ in fits if d > T) / len(fits)
    cut = [(p, min(d, T), g) for p, d, g, _, _ in fits]
    return (sum(p * d for p, d, _ in cut) / len(cut),
            sum(g * d for _, d, g in cut) / len(cut),
            trunc, unplaceable)


# ── Arrival process ─────────────────────────────────────────────────────────

def on_intervals(p1: float, T: float, rng: random.Random) -> list[tuple[float, float]]:
    """ON/OFF modulator. p1 = r2/(r1+r2) is the stationary ON probability;
    mean ON sojourn is pinned to E_ON_SEC so p1 alone controls burstiness."""
    r1 = 1.0 / E_ON_SEC
    r2 = r1 * p1 / (1.0 - p1)
    t, on, out = 0.0, rng.random() < p1, []
    while t < T:
        end = min(t + rng.expovariate(r1 if on else r2), T)
        if on:
            out.append((t, end))
        t, on = end, not on
    return out


def generate(N: int, p1: float, T: float, rng: random.Random) -> list[float]:
    """Place EXACTLY N arrivals uniformly on the ON set (PLAN §3.4)."""
    iv = on_intervals(p1, T, rng)
    total = sum(b - a for a, b in iv)
    if total <= 0:
        return []
    out = []
    for _ in range(N):
        u = rng.random() * total
        for a, b in iv:
            if u <= b - a:
                out.append(a + u)
                break
            u -= b - a
    out.sort()
    return out


def idc(times, window: float, t0: float, t1: float) -> float:
    """Var(N_w)/E(N_w). Empty windows are counted — dropping them understates
    burstiness by an order of magnitude on sparse traces."""
    span = max(t1 - t0, 0.0)
    n = max(1, int(math.ceil(span / window)) if span > 0 else 1)
    counts = [0] * n
    for x in times:
        if t0 <= x < t1 or (x == t1 and n == 1):
            counts[min(n - 1, int((x - t0) / window))] += 1
    m = sum(counts) / n
    return (sum((c - m) ** 2 for c in counts) / n) / m if m > 0 else float("nan")


def fit_p1(N: int, target: float, T: float) -> float:
    """Bisect the duty cycle so the mean measured IDC(1h) hits `target`.
    Fitting numerically rather than from the asymptotic MMPP formula is deliberate:
    IDC(w) only approaches its asymptote for w far above the burst timescale, and at
    w = 1 h on a multi-day horizon we are squarely in the transition region."""
    lo, hi = 0.004, 0.98
    for _ in range(FIT_ITERS):
        mid = math.sqrt(lo * hi)
        got = sum(idc(generate(N, mid, T, random.Random(9000 + k)), IDC_WINDOW, 0.0, T)
                  for k in range(FIT_REPS)) / FIT_REPS
        if got > target:        # burstier than asked -> raise the duty cycle
            lo = mid
        else:
            hi = mid
    return math.sqrt(lo * hi)


# ── §3.6 scenario table ─────────────────────────────────────────────────────

def scenario_table(pool) -> list[dict]:
    rows = []
    for arm, (cap_pe, cap_gpu, days) in ARMS.items():
        T = days * 86400.0
        e_work, e_gpu, trunc, unplaceable = moments(pool, T, HOST_LIMITS[arm])
        for name, rho, idc_target in SCENARIOS:
            lam = rho / max(e_work / cap_pe, e_gpu / cap_gpu)
            N = round(lam * T)
            p1 = fit_p1(N, idc_target, T)
            per_seed = [idc(generate(N, p1, T, random.Random(s)), IDC_WINDOW, 0.0, T)
                        for s in SEEDS]
            mean = sum(per_seed) / len(per_seed)
            sd = (sum((x - mean) ** 2 for x in per_seed) / len(per_seed)) ** 0.5
            rows.append({
                "arm": arm, "scenario": name, "rho": rho, "idc_target": idc_target,
                "horizon_days": days, "e_work_pe_sec": e_work, "e_gpu_card_sec": e_gpu,
                "truncated_frac": trunc, "unplaceable_frac": unplaceable,
                "lambda_per_hour": lam * 3600, "n_task": N,
                "p1": p1, "idc_1h_mean": mean, "idc_1h_sd": sd,
                "idc_10min": sum(idc(generate(N, p1, T, random.Random(s)), 600.0, 0.0, T)
                                 for s in SEEDS) / len(SEEDS),
                "idc_12h": sum(idc(generate(N, p1, T, random.Random(s)), 43200.0, 0.0, T)
                               for s in SEEDS) / len(SEEDS),
            })
    return rows


# ── §3.5 burstiness anchors (Philly / Helios) ───────────────────────────────

def reference_arrivals(path: str) -> list[float]:
    """Philly/Helios expose inter-arrival times in `interval`; cumsum gives arrivals."""
    t, out = 0.0, []
    with open(path, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            t += float(r["interval"])
            out.append(t)
    return out


def anchors(window_days: int = 3) -> list[dict]:
    out = []
    for name in ("philly", "helios"):
        path = os.path.join(REF_DIR, f"{name}_data_training.csv")
        if not os.path.isfile(path):
            out.append({"trace": name, "error": f"missing {path} - run scripts/fetch-reference-traces.sh"})
            continue
        ts = reference_arrivals(path)
        t0, t1 = ts[0], ts[-1]
        W = window_days * 86400.0
        within, t = [], t0
        while t + W <= t1:
            if sum(1 for x in ts if t <= x < t + W) >= 50:
                within.append(idc(ts, IDC_WINDOW, t, t + W))
            t += W
        gaps = [ts[i + 1] - ts[i] for i in range(len(ts) - 1)]
        gm = sum(gaps) / len(gaps)
        out.append({
            "trace": name, "n": len(ts), "span_days": (t1 - t0) / 86400.0,
            "ca2": (sum((g - gm) ** 2 for g in gaps) / len(gaps)) / (gm * gm),
            "idc_1h_whole": idc(ts, IDC_WINDOW, t0, t1),
            f"idc_1h_within_{window_days}d_median": statistics.median(within) if within else None,
            f"idc_1h_within_{window_days}d_q1": (statistics.quantiles(within, n=4)[0]
                                                 if len(within) > 3 else None),
            f"idc_1h_within_{window_days}d_q3": (statistics.quantiles(within, n=4)[2]
                                                 if len(within) > 3 else None),
            "n_windows": len(within),
        })
    return out


# ── CLI ─────────────────────────────────────────────────────────────────────

def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Regenerate the locked WM-1 design numbers.")
    ap.add_argument("--anchors", action="store_true", help="also print the Philly/Helios anchors (§3.5)")
    ap.add_argument("--json", action="store_true", help="machine-readable output for tests")
    args = ap.parse_args(argv)

    if not os.path.isfile(OPENB):
        print(f"[ERROR] trace not found: {OPENB}", file=sys.stderr)
        return 2

    rows = scenario_table(load_pool(OPENB))
    anc = anchors() if args.anchors else None

    if args.json:
        print(json.dumps({"scenarios": rows, "anchors": anc}, indent=2))
        return 0

    seen = set()
    print("PLAN-Workload-Model.md section 3.6 - scenario table")
    print("%-7s %-9s %5s %5s | %6s %7s | %7s %6s %7s %7s"
          % ("arm", "scenario", "rho", "IDC*", "p1", "N", "IDC 1h", "sd", "IDC 10m", "IDC 12h"))
    for r in rows:
        if r["arm"] not in seen:
            seen.add(r["arm"])
            print("--- %s: T=%dd, truncation touches %.1f%% of jobs, %.2f%% excluded as "
                  "unplaceable, E[work]=%.0f PE-s ---"
                  % (r["arm"], r["horizon_days"], 100 * r["truncated_frac"],
                     100 * r["unplaceable_frac"], r["e_work_pe_sec"]))
        print("%-7s %-9s %5.2f %5.0f | %6.3f %7d | %7.1f %6.1f %7.1f %7.1f"
              % (r["arm"], r["scenario"], r["rho"], r["idc_target"], r["p1"], r["n_task"],
                 r["idc_1h_mean"], r["idc_1h_sd"], r["idc_10min"], r["idc_12h"]))

    if anc:
        print()
        print("section 3.5 - burstiness anchors")
        for a in anc:
            if "error" in a:
                print("  %-8s %s" % (a["trace"], a["error"]))
            else:
                print("  %-8s n=%d span=%.1fd  C_a^2=%.1f  IDC(1h) whole=%.1f  "
                      "within-3d median=%.1f IQR=[%.1f, %.1f] (%d win)"
                      % (a["trace"], a["n"], a["span_days"], a["ca2"], a["idc_1h_whole"],
                         a["idc_1h_within_3d_median"], a["idc_1h_within_3d_q1"],
                         a["idc_1h_within_3d_q3"], a["n_windows"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
