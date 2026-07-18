"""G2.5 — SLA-budget sweep: trace the Pareto front by scanning the budget ``d``.

Phase 2 does **not** sweep scalarisation weights (that was Phase 1). Instead,
each budget ``d`` defines a Constrained MDP whose PID-Lagrangian tunes ``λ``
itself; one converged policy per ``d`` contributes one point on the energy↔SLA
front (CLAUDE.md Lưu ý #7 / the zero-duality-gap argument, Paternain 2019).

The sweep then **verifies the front is monotone**: loosening ``d`` must not
lower SLA cost nor raise energy. A non-monotone point means that budget's policy
has not converged — it must be retrained, not plotted (Lưu ý #7). This script
reports the violation instead of silently emitting a bad front point.

Run (needs the gateway + sb3)::

    docker compose run --rm rl-agent python src/eval/sweep_budget.py \
        --scenario LOW --budgets 0.02,0.04,0.06,0.08,0.10 \
        --seeds 42,43,44,45,46 --total-timesteps 50000 --k-p 0.05 --k-i 0.05
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

# Support both `python src/eval/sweep_budget.py` and `python -m eval.sweep_budget`.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

try:
    from .aggregate import aggregate_by_method, check_budget_front_monotone, mean_ci95
    from .points import PointRecord, save_points
except ImportError:  # pragma: no cover - direct-script fallback
    from eval.aggregate import aggregate_by_method, check_budget_front_monotone, mean_ci95
    from eval.points import PointRecord, save_points


def _parse_floats(s: str) -> list[float]:
    return [float(x) for x in s.split(",") if x.strip()]


def _parse_ints(s: str) -> list[int]:
    return [int(x) for x in s.split(",") if x.strip()]


# ── Front assembly (pure — unit-testable without training) ──────────────────

def build_budget_front(points: list[PointRecord]) -> list[tuple[float, float, float]]:
    """Collapse per-seed points into one ``(d, mean_energy, mean_sla)`` per budget.

    Averaging across seeds BEFORE the monotonicity check is deliberate: a single
    seed's noise should not be mistaken for a convergence failure (Lưu ý #10).
    """
    by_d: dict[float, list[PointRecord]] = {}
    for p in points:
        d = float((p.extra or {}).get("budget_d", "nan"))
        by_d.setdefault(d, []).append(p)

    front: list[tuple[float, float, float]] = []
    for d, recs in sorted(by_d.items()):
        e = mean_ci95([r.energy_kwh for r in recs]).mean
        s = mean_ci95([r.sla_cost for r in recs]).mean
        front.append((d, e, s))
    return front


def summarise_sweep(points: list[PointRecord]) -> dict:
    """Build the budget front + monotonicity verdict + per-budget aggregates."""
    front = build_budget_front(points)
    report = check_budget_front_monotone(front)
    agg = aggregate_by_method(points)
    return {
        "front": [{"budget_d": d, "energy_kwh": e, "sla_cost": s}
                  for d, e, s in front],
        "monotone": report.monotone,
        "monotonicity_detail": report.detail,
        "per_budget": {
            m: {
                "energy_mean": a["energy"].mean, "energy_ci95": a["energy"].ci95,
                "sla_mean": a["sla"].mean, "sla_ci95": a["sla"].ci95,
                "seeds": a["seeds"], "n": a["n"],
            }
            for m, a in agg.items()
        },
    }


def print_sweep_summary(summary: dict) -> None:
    print("\n" + "=" * 78)
    print("  G2.5 — SLA-budget sweep front (energy ↔ SLA)")
    print("=" * 78)
    print(f"{'budget d':>10} {'energy kWh':>14} {'C_SLA':>16}")
    for row in summary["front"]:
        print(f"{row['budget_d']:>10.4g} {row['energy_kwh']:>14.2f} "
              f"{row['sla_cost']:>16.4e}")
    print("-" * 78)
    if summary["monotone"]:
        print("VERDICT: front is MONOTONE in d "
              "(looser d ⇒ higher SLA cost, lower energy) ✓")
    else:
        print("VERDICT: front is NON-MONOTONE ⇒ those budgets have NOT converged.")
        for d in summary["monotonicity_detail"]:
            print(f"  - {d}")
        print("  → retrain the offending budgets (more timesteps / tuned gains) "
              "before plotting (Lưu ý #7).")
    print("=" * 78 + "\n")


# ── Sweep driver (needs sb3 + gateway) ──────────────────────────────────────

def _job_list(budgets: list[float], seeds: list[int]) -> list[tuple[float, int]]:
    return [(d, s) for d in budgets for s in seeds]


def preflight_gateways(base_port: int, n: int, host: str | None = None) -> None:
    """Fail fast if the N gateways this sweep needs are not actually up.

    A budget sweep is a multi-hour job. Discovering at minute 3 that port
    ``base+3`` was never started — because the Java container was launched
    without ``NUM_GATEWAYS`` — wastes the whole run. This probes every port
    up-front and reports exactly what to do.
    """
    import socket

    host = host or os.environ.get("GATEWAY_HOST", "cloudsim-java")
    dead: list[int] = []
    for i in range(n):
        port = base_port + i
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.settimeout(3.0)
            if s.connect_ex((host, port)) != 0:
                dead.append(port)
    if dead:
        raise SystemExit(
            f"[sweep] PREFLIGHT FAILED: {len(dead)} of {n} gateways unreachable "
            f"at {host}: ports {dead}.\n"
            f"  Start the Java side with enough gateways before a long run:\n"
            f"    NUM_GATEWAYS={n} PY4J_PORT_MAX={base_port + n - 1} \\\n"
            f"        docker compose up -d --force-recreate cloudsim-java"
        )
    print(f"[sweep] preflight OK: {n} gateway(s) reachable on "
          f"{host}:{base_port}–{base_port + n - 1}")


def _run_one_job(args: tuple) -> PointRecord:
    """Train + evaluate a single (budget, seed). Runs in a worker process.

    Reads its gateway port from ``GATEWAY_PORT``, which ``_init_worker`` pins
    per process — that is how each concurrent run gets its own simulation.
    """
    (d, seed, scenario, total_timesteps, k_p, k_i, k_d, dual_every,
     output_dir, model_dir) = args

    from train_cmdp import evaluate_cmdp, train_cmdp  # lazy: needs sb3

    tag = f"cmdp-d{d:g}"
    port = os.environ.get("GATEWAY_PORT", "?")
    print(f"\n[sweep] ===== budget d={d:g}, seed={seed} (gateway :{port}) =====")
    model_path = Path(model_dir) / f"{scenario.lower()}-d{d:g}-s{seed}.zip"
    _, pid, conv = train_cmdp(
        scenario=scenario, seed=seed, total_timesteps=total_timesteps,
        budget_d=d, k_p=k_p, k_i=k_i, k_d=k_d, dual_every=dual_every,
        out_path=model_path,
    )
    row = evaluate_cmdp(model_path, scenario, seed, d, pid,
                        Path(output_dir) / f"baseline-{scenario}" / tag)
    return PointRecord(
        method=tag, scenario=scenario, seed=seed,
        energy_kwh=float(row["total_energy_kwh"]),
        sla_cost=float(row["total_sla_cost"]),
        extra={
            "budget_d": d,
            "lambda_final": float(pid.lambda_),
            "constraint": conv.get("constraint"),
            "j_tail_mean": conv.get("j_tail_mean"),
        },
    )


def _init_worker(port_queue) -> None:
    """Pin one gateway port to this worker process for its whole lifetime.

    Taking the port from a queue (rather than deriving it from a job index)
    guarantees no two *live* workers ever share a simulation, however the pool
    schedules jobs.
    """
    os.environ["GATEWAY_PORT"] = str(port_queue.get())


def run_sweep(
    scenario: str,
    budgets: list[float],
    seeds: list[int],
    total_timesteps: int,
    *,
    k_p: float,
    k_i: float,
    k_d: float,
    dual_every: int,
    output_dir: Path,
    model_dir: Path,
    parallel: int = 1,
    base_port: int | None = None,
) -> list[PointRecord]:
    """Train + evaluate one CMDP policy per (budget, seed); return the points.

    ``parallel`` runs that many *independent training runs* concurrently, each
    against its own gateway. This — not ``--n-envs`` — is the effective lever
    for the sweep: after SYS.2 the env is ~0.5 ms/step and the PPO learner is
    ~67% of the step budget, so vectorising envs inside one run buys little
    (measured 1.35×), while whole runs are embarrassingly parallel.

    Each run is numerically unaffected by the parallelism: separate process,
    separate JVM gateway, separate CloudSim instance, and ``torch_threads=1``
    keeps the processes from oversubscribing the CPU.
    """
    jobs = _job_list(budgets, seeds)
    port0 = int(os.environ.get("GATEWAY_PORT", "25333")) if base_port is None else base_port
    parallel = max(1, min(int(parallel), len(jobs)))

    payloads = [
        (d, seed, scenario, total_timesteps, k_p, k_i, k_d, dual_every,
         str(output_dir), str(model_dir))
        for d, seed in jobs
    ]

    if parallel == 1:
        # Sequential path — byte-for-byte the behaviour before C9 was addressed.
        return [_run_one_job(p) for p in payloads]

    preflight_gateways(port0, parallel)
    print(f"[sweep] running {len(jobs)} job(s) at concurrency {parallel} "
          f"(gateways :{port0}–{port0 + parallel - 1})")

    import multiprocessing as mp

    # "spawn" keeps each run's torch/JVM client state fully independent; the
    # workers are long-lived (one per gateway) so the startup cost is paid once.
    ctx = mp.get_context("spawn")
    port_queue = ctx.Queue()
    for i in range(parallel):
        port_queue.put(port0 + i)

    with ctx.Pool(parallel, initializer=_init_worker, initargs=(port_queue,)) as pool:
        points = pool.map(_run_one_job, payloads, chunksize=1)
    return list(points)


def main() -> None:
    ap = argparse.ArgumentParser(description="G2.5 — SLA-budget sweep → Pareto front")
    ap.add_argument("--scenario", default="LOW", choices=["LOW", "HIGH", "BURST"])
    ap.add_argument("--budgets", default="0.02,0.04,0.06,0.08,0.10",
                    help="comma-separated SLA budgets d (tight → loose)")
    ap.add_argument("--seeds", default="42",
                    help="comma-separated seeds (≥5 required for reportable CI)")
    ap.add_argument("--total-timesteps", type=int, default=50_000)
    ap.add_argument("--k-p", type=float, default=0.05)
    ap.add_argument("--k-i", type=float, default=0.05)
    ap.add_argument("--k-d", type=float, default=0.0)
    ap.add_argument("--dual-every", type=int, default=1)
    ap.add_argument("--output", default="/data/results")
    ap.add_argument("--models", default="/data/models")
    ap.add_argument("--parallel", type=int, default=1,
                    help="Concurrent training runs, one gateway each (C9 — the "
                         "effective lever for this sweep). Needs the Java side "
                         "started with NUM_GATEWAYS >= this. Default 1 = sequential.")
    ap.add_argument("--base-port", type=int, default=None,
                    help="First gateway port (default: $GATEWAY_PORT).")
    args = ap.parse_args()

    budgets = _parse_floats(args.budgets)
    seeds = _parse_ints(args.seeds)
    out_dir = Path(args.output)

    if len(seeds) < 5:
        print(f"[sweep] WARNING: {len(seeds)} seed(s) — CLAUDE.md Lưu ý #10 requires "
              f"≥5 seeds before any number is reportable. This run is a smoke/"
              f"wiring check, not a thesis result.")

    points = run_sweep(
        args.scenario, budgets, seeds, args.total_timesteps,
        k_p=args.k_p, k_i=args.k_i, k_d=args.k_d, dual_every=args.dual_every,
        output_dir=out_dir, model_dir=Path(args.models),
        parallel=args.parallel, base_port=args.base_port,
    )

    sweep_dir = out_dir / f"sweep-{args.scenario}"
    save_points(points, sweep_dir / "points.jsonl")
    summary = summarise_sweep(points)
    print_sweep_summary(summary)
    with open(sweep_dir / "sweep_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[sweep] points  → {sweep_dir / 'points.jsonl'}")
    print(f"[sweep] summary → {sweep_dir / 'sweep_summary.json'}")


if __name__ == "__main__":
    main()
