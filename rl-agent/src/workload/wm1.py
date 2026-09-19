"""WM-1 orchestrator — turn (arm, scenario, seed) into an openb-schema trace.

Composes the four independent blocks (PLAN §3.1):

    adapters  ->  jobsize (bootstrap + truncate at T)
                  calibrate (dominant-resource load -> lambda -> N)
                  arrivals (MMPP-2 duty cycle fitted to the IDC target -> N times)
              ->  schema.write_csv

Plus ``REPLAY``, which skips the model entirely and takes the busiest contiguous window
of the real trace (PLAN §3.7) so the thesis can report a controlled result and a
realistic one side by side.

Determinism is a hard requirement: the same (arm, scenario, seed) must reproduce the same
file byte for byte, because that is what makes a campaign auditable months later. Job sizes
and arrival times are drawn from separate, deterministically derived streams - see the
offsets below for why that matters beyond reproducibility.
"""

from __future__ import annotations

import hashlib
import json
import os
import random
import subprocess
from dataclasses import asdict, dataclass

from . import adapters, arrivals, schema
from .calibrate import ARMS, Capacity, measured_load, plan_load
from .jobsize import JobSizePool
from .schema import Task

#: (rho target, IDC(1h) target) per scenario — PLAN §3.6, all measured, none chosen.
SCENARIOS: dict[str, tuple[float, float]] = {
    "LOW": (0.30, 10.0),
    "HIGH": (0.85, 10.0),
    "BURST": (0.85, 40.0),
    "OVERLOAD": (1.25, 10.0),
}
REPLAY = "REPLAY"
ALL_SCENARIOS = (*SCENARIOS, REPLAY)

DEFAULT_SEEDS = (42, 43, 44, 45, 46)

# Job sizes and arrival times are drawn from SEPARATE, deterministically derived RNG
# streams. Two reasons, both structural:
#   * HIGH and BURST share a rho target and therefore an arrival count, so with split
#     streams they draw the *same job multiset* and differ only in when those jobs
#     arrive. That turns the burstiness comparison into a paired one, which is exactly
#     the controlled contrast the redesign exists to provide.
#   * Changing the arrival model can no longer perturb the job sample, so a load figure
#     cannot drift because a duty cycle changed.
# Plain integer offsets, not hashed tuples: Python's string hashing is randomised per
# process unless PYTHONHASHSEED is pinned, which would silently break reproducibility.
_JOB_STREAM_OFFSET = 1_000_003
_ARRIVAL_STREAM_OFFSET = 7_000_019


class GenerationError(RuntimeError):
    """Raised when a requested trace cannot be produced."""


@dataclass(frozen=True, slots=True)
class TraceSpec:
    """Everything needed to reproduce one generated trace."""

    arm: str
    scenario: str
    seed: int
    rho_target: float | None
    idc_target: float | None
    horizon_sec: float
    n_task: int
    duty_cycle: float | None
    lambda_per_hour: float | None
    rho_cpu: float
    rho_gpu: float
    bottleneck: str
    truncated_fraction: float
    #: W3.1 — share of source jobs excluded because no single host could run them.
    unplaceable_fraction: float
    e_work_pe_sec: float
    e_gpu_work_card_sec: float
    pool_size: int
    source_adapter: str
    source_sha256: str


# ── Generation ──────────────────────────────────────────────────────────────

def _busiest_window(tasks: list[Task], horizon_sec: float,
                    capacity: Capacity) -> list[Task]:
    """The busiest *realistic* contiguous window of length ``horizon_sec``.

    "Busiest" cannot simply mean most work. With this trace's tail the single heaviest
    4-day window contains **5 tasks** — one 145-day pod and four neighbours — at an
    offered load of 2.44. Replaying that would compare the scheduler against a degenerate
    instance, not against reality. Matching the HIGH load target alone fails the same way:
    the 4-day window closest to rho = 0.85 has 7 tasks.

    So the window must be busy on *both* axes: among windows carrying at least the median
    number of arrivals, take the one with the highest dominant-resource load. On the
    homogeneous arm that selects 776 arrivals at rho = 0.232; on the heterogeneous one,
    1 882 arrivals at rho_gpu = 0.932.

    Windows are anchored at arrivals, which is sufficient: the maximum of a sliding sum
    over a point process is always attained at a window starting on a point. Pending and
    Failed pods are dropped and durations truncated at the horizon, matching the job-size
    pool exactly, so REPLAY and the modelled scenarios are measured on one footing. Jobs no
    single host can run are excluded for the same reason as in the pool (W3.1): every policy
    drops them alike, so keeping them measures the cluster's inadequacy, not the workload.

    Finding worth reporting rather than glossing over: the busiest realistic 4-day window
    of openb sits at rho = 0.232, roughly a *quarter* of the modelled HIGH (0.85), and no
    real window reaches HIGH's arrival count. HIGH is therefore a stress level this trace
    does not naturally exhibit — which is a legitimate thing for a stress scenario to be,
    but it must be stated. On the heterogeneous arm the gap narrows sharply: with only
    18 cards the same window reaches rho_gpu = 0.932, close to saturation.

    (Both figures moved down in W3.1. The previous 0.454 / 1.572 counted pods that no host
    in the modelled cluster can run — a load that could never have been served, so it did
    not belong in the comparison.)
    """
    pool_tasks = [t for t in tasks
                  if t.pod_phase not in (schema.PHASE_PENDING, schema.PHASE_FAILED)
                  and capacity.fits_any_host(t)]
    if not pool_tasks:
        raise GenerationError("REPLAY needs a non-empty trace")

    n = len(pool_tasks)
    pe_sec = capacity.total_pes * horizon_sec
    gpu_sec = capacity.total_gpus * horizon_sec if capacity.total_gpus > 0 else 0.0

    windows: list[tuple[float, int, int, int]] = []   # (load, count, start, stop)
    j, w_cpu, w_gpu, count = 0, 0.0, 0.0, 0
    for i in range(n):
        if j < i:
            j, w_cpu, w_gpu, count = i, 0.0, 0.0, 0
        while (j < n
               and pool_tasks[j].creation_time < pool_tasks[i].creation_time + horizon_sec):
            t = pool_tasks[j]
            w_cpu += t.pes * min(t.duration, horizon_sec)
            w_gpu += t.num_gpu * min(t.duration, horizon_sec)
            count += 1
            j += 1
        load = max(w_cpu / pe_sec, (w_gpu / gpu_sec) if gpu_sec > 0 else 0.0)
        windows.append((load, count, i, j))
        head = pool_tasks[i]
        w_cpu -= head.pes * min(head.duration, horizon_sec)
        w_gpu -= head.num_gpu * min(head.duration, horizon_sec)
        count -= 1

    counts = sorted(w[1] for w in windows)
    median_count = counts[len(counts) // 2]
    candidates = [w for w in windows if w[1] >= median_count] or windows
    _, _, start, stop = max(candidates, key=lambda w: w[0])

    window = pool_tasks[start:stop]
    if not window:
        raise GenerationError("REPLAY window came out empty")

    # Rebase to zero so every scenario shares a [0, T) timeline, and truncate durations
    # at the horizon exactly as the job-size pool does, so REPLAY and the generated
    # scenarios are measured on the same footing.
    t0 = window[0].creation_time
    return [t.with_duration(min(t.duration, horizon_sec))
             .with_arrival(t.creation_time - t0, name=f"replay-{i:06d}")
            for i, t in enumerate(window)]


def generate_trace(source_tasks: list[Task], arm: str, scenario: str, seed: int, *,
                   source_adapter: str = "openb",
                   source_sha256: str = "") -> tuple[list[Task], TraceSpec]:
    """Build one trace and the spec that reproduces it."""
    if arm not in ARMS:
        raise GenerationError(f"unknown arm {arm!r}; known: {sorted(ARMS)}")
    if scenario not in ALL_SCENARIOS:
        raise GenerationError(f"unknown scenario {scenario!r}; known: {list(ALL_SCENARIOS)}")

    capacity, horizon = ARMS[arm]
    # capacity= filters out jobs no single host could ever run (W3.1) — without it every
    # scenario carries a handful of guaranteed drops and C_SLA gains a policy-independent
    # floor. Passing it here is what makes `droppedTasks == 0` an achievable acceptance
    # criterion on LOW/HIGH/BURST rather than an aspiration.
    pool = JobSizePool.from_trace(source_tasks, horizon, capacity=capacity)
    m = pool.moments

    if scenario == REPLAY:
        tasks = _busiest_window(source_tasks, horizon, capacity)
        rho_cpu, rho_gpu = measured_load(tasks, capacity, horizon)
        spec = TraceSpec(
            arm=arm, scenario=scenario, seed=seed,
            rho_target=None, idc_target=None, horizon_sec=horizon, n_task=len(tasks),
            duty_cycle=None, lambda_per_hour=None,
            rho_cpu=rho_cpu, rho_gpu=rho_gpu,
            bottleneck="gpu" if rho_gpu > rho_cpu else "cpu",
            truncated_fraction=m.truncated_fraction,
            unplaceable_fraction=m.unplaceable_fraction,
            e_work_pe_sec=m.e_work, e_gpu_work_card_sec=m.e_gpu_work, pool_size=m.size,
            source_adapter=source_adapter, source_sha256=source_sha256)
        return tasks, spec

    rho_target, idc_target = SCENARIOS[scenario]
    plan = plan_load(pool, capacity, rho_target, horizon)
    duty = arrivals.fit_duty_cycle(plan.n_task, idc_target, horizon)

    jobs = pool.sample_for_load(
        plan.n_task, random.Random(seed + _JOB_STREAM_OFFSET),
        pe_capacity=capacity.total_pes, gpu_capacity=capacity.total_gpus,
        horizon_sec=horizon, rho_target=rho_target)
    times = arrivals.generate(plan.n_task, duty, horizon,
                              random.Random(seed + _ARRIVAL_STREAM_OFFSET))
    prefix = f"wm1-{arm}-{scenario.lower()}"
    tasks = [job.with_arrival(t, name=f"{prefix}-{i:06d}")
             for i, (t, job) in enumerate(zip(times, jobs))]
    schema.validate_tasks(tasks)

    rho_cpu, rho_gpu = measured_load(tasks, capacity, horizon)
    spec = TraceSpec(
        arm=arm, scenario=scenario, seed=seed,
        rho_target=rho_target, idc_target=idc_target, horizon_sec=horizon,
        n_task=plan.n_task, duty_cycle=duty, lambda_per_hour=plan.lambda_per_hour,
        rho_cpu=rho_cpu, rho_gpu=rho_gpu, bottleneck=plan.bottleneck,
        truncated_fraction=m.truncated_fraction,
        unplaceable_fraction=m.unplaceable_fraction,
        e_work_pe_sec=m.e_work, e_gpu_work_card_sec=m.e_gpu_work, pool_size=m.size,
        source_adapter=source_adapter, source_sha256=source_sha256)
    return tasks, spec


# ── Batch generation to disk ────────────────────────────────────────────────

def _sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _git_sha() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                              capture_output=True, text=True, timeout=10,
                              check=True).stdout.strip()
    except Exception:
        return ""


def trace_path(out_root: str, arm: str, scenario: str, seed: int) -> str:
    """``<out_root>/<arm>/<SCENARIO>/seed<NN>.csv`` — arm is in the path so the two
    topologies can never overwrite each other (PLAN §4.1, risk R6)."""
    return os.path.join(out_root, arm, scenario, f"seed{seed}.csv")


def generate_arm(source_path: str, arm: str, out_root: str, *,
                 scenarios=ALL_SCENARIOS, seeds=DEFAULT_SEEDS,
                 source_adapter: str = "openb", verbose: bool = True) -> dict:
    """Generate every (scenario, seed) for one arm and write the manifest."""
    # require(), not load(): generation feeds a job-size pool, so an adapter without
    # CAP_JOBSIZE must fail HERE with a readable reason rather than three layers down.
    # Philly/Helios are exactly this case — they would emit tasks with memory_mib = 0 and
    # no QoS class, which the simulator accepts in silence (W4.2).
    tasks = adapters.require(source_adapter, adapters.CAP_JOBSIZE).load(source_path)
    src_sha = _sha256_file(source_path)
    capacity, horizon = ARMS[arm]

    entries = []
    for scenario in scenarios:
        for seed in seeds:
            out, spec = generate_trace(tasks, arm, scenario, seed,
                                       source_adapter=source_adapter,
                                       source_sha256=src_sha)
            path = trace_path(out_root, arm, scenario, seed)
            schema.write_csv(path, out)
            entry = asdict(spec)
            # Relative to the manifest's own directory, not to out_root: the manifest
            # lives at <out_root>/<arm>/, so an out_root-relative path would force every
            # reader to know it must climb one level first. ValidationRunner B22 got that
            # wrong on the first attempt and resolved <arm>/<arm>/... — a self-contained
            # path removes the ambiguity rather than documenting around it.
            entry["path"] = os.path.relpath(
                path, os.path.join(out_root, arm)).replace("\\", "/")
            entry["sha256"] = _sha256_file(path)
            entries.append(entry)
            if verbose:
                print(f"[wm1] {arm}/{scenario}/seed{seed}: {spec.n_task} tasks, "
                      f"rho_cpu={spec.rho_cpu:.3f} rho_gpu={spec.rho_gpu:.3f} "
                      f"({spec.bottleneck}) -> {entry['path']}")

    manifest = {
        "arm": arm,
        # Trace paths in "traces" are relative to THIS file's directory.
        "paths_relative_to": "this manifest's directory",
        "capacity": {"name": capacity.name, "hosts": capacity.hosts,
                     "total_pes": capacity.total_pes, "total_gpus": capacity.total_gpus},
        "horizon_sec": horizon,
        "horizon_days": horizon / 86400.0,
        "source": {"adapter": source_adapter,
                   "path": source_path.replace("\\", "/"), "sha256": src_sha},
        "git_sha": _git_sha(),
        "arrival_model": {
            "kind": "MMPP-2 ON/OFF, fixed-N placement",
            "mean_on_sec": arrivals.E_ON_SEC,
            "idc_window_sec": arrivals.IDC_WINDOW_SEC,
        },
        "scenarios": {k: {"rho": v[0], "idc_1h": v[1]} for k, v in SCENARIOS.items()},
        "seeds": list(seeds),
        "traces": entries,
    }
    mpath = os.path.join(out_root, arm, "wm1-manifest.json")
    os.makedirs(os.path.dirname(mpath), exist_ok=True)
    with open(mpath, "w", encoding="utf-8", newline="\n") as f:
        json.dump(manifest, f, indent=2)
        f.write("\n")
    if verbose:
        print(f"[wm1] manifest -> {mpath}  ({len(entries)} traces)")
    return manifest
