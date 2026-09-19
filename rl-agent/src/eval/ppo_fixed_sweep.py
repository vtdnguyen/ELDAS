"""G2.6+ — Fixed-weight PPO *front* (front-to-front baseline for CMDP-PID).

Why this exists
---------------
The Phase-1 fixed-weight PPO ("ppo-min") was a **single** scalarisation weight
(0.5/0.5), evaluated on **one** seed. That is not a fair opponent for the
CMDP-PID method, which produces a whole front by sweeping the SLA budget ``d``.
Comparing one hand-picked weight against a full front both (a) can't be judged
statistically (n=1) and (b) tilts hypervolume in the front's favour.

This collector fixes both. It trains one fixed-weight PPO per (weight, seed) and
writes per-seed points — the SAME ``points.jsonl`` schema the CMDP sweep uses —
to ``ppo-fixed-{scenario}/points.jsonl``. In the campaign:

  * each weight ``w`` becomes a ``ppo-w{w}`` method with mean ± 95% CI over seeds
    (so ``ppo-w0.5`` IS the classic ppo-min, now with ≥5 seeds — Lưu ý #10);
  * ``run_campaign.metric_family`` collapses ``ppo-w*`` → one ``ppo-fixed`` family
    so hypervolume/IGD+ score the **fixed-weight front vs. the CMDP-PID front** —
    a fair front-to-front comparison (manual weight tuning vs. principled budget
    tuning), which is the scientifically meaningful question.

Structure mirrors ``sweep_budget.py`` deliberately: same ``--parallel`` (one
gateway per concurrent run, port pinned per worker) and the same ``preflight``
that fails fast if the gateways aren't up — an overnight job must not die at
minute 3 (C9). Numerically each run is independent: separate process, JVM
gateway, CloudSim instance, and ``torch_threads=1`` (no CPU oversubscription).

Run (needs the gateway + sb3)::

    # just multi-seed ppo-min (cheap: 1 weight × 5 seeds)
    docker compose run --rm rl-agent python src/eval/ppo_fixed_sweep.py \
        --scenario HIGH --weights 0.5 --seeds 42,43,44,45,46 --total-timesteps 200000

    # full fixed-weight front (5 weights × 5 seeds), 4 runs at a time
    NUM_GATEWAYS=4 PY4J_PORT_MAX=25336 docker compose run --rm rl-agent \
        python src/eval/ppo_fixed_sweep.py --scenario HIGH \
        --weights 0.1,0.3,0.5,0.7,0.9 --seeds 42,43,44,45,46 \
        --total-timesteps 200000 --parallel 4
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

# Support both `python src/eval/ppo_fixed_sweep.py` and `python -m eval.ppo_fixed_sweep`.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

try:
    from .aggregate import aggregate_by_method, mean_ci95
    from .points import PointRecord, save_points
    from .sweep_budget import preflight_gateways, _init_worker
    from . import paths
except ImportError:  # pragma: no cover - direct-script fallback
    from eval.aggregate import aggregate_by_method, mean_ci95
    from eval.points import PointRecord, save_points
    from eval.sweep_budget import preflight_gateways, _init_worker
    from eval import paths


def _parse_floats(s: str) -> list[float]:
    return [float(x) for x in s.split(",") if x.strip()]


def _parse_ints(s: str) -> list[int]:
    return [int(x) for x in s.split(",") if x.strip()]


# ── One (weight, seed): train a fixed-weight PPO, evaluate one greedy episode ──

def _run_one_ppo(args: tuple) -> PointRecord:
    """Train + evaluate a single (weight_energy, seed). Runs in a worker process.

    Reads its gateway port from ``GATEWAY_PORT`` (pinned by ``_init_worker``), so
    each concurrent run drives its own CloudSim instance. The eval output dir is
    **unique per (w, seed)** on purpose: ``train_min.evaluate`` writes a
    ``baseline_results.json`` next to it, and a shared parent would let parallel
    workers race on that file. We build the point from the returned summary, not
    from that side-effect file.
    """
    (w, seed, scenario, total_timesteps, output_dir, model_dir, skip_existing) = args

    import numpy as np
    from train_min import train as _train, evaluate as _evaluate

    try:  # speed + determinism + no oversubscription under --parallel (SYS.5)
        from perf.tuning import set_torch_threads
        set_torch_threads(1, verbose=False)
    except Exception:
        pass

    weights = np.array([w, 1.0 - w], dtype=np.float32)
    tag = f"ppo-w{w:g}"
    port = os.environ.get("GATEWAY_PORT", "?")
    print(f"\n[ppo-fixed] ===== w_energy={w:g} (w_sla={1 - w:g}), seed={seed} "
          f"(gateway :{port}) =====")

    model_path = Path(model_dir) / "ppo-fixed" / f"{scenario.lower()}-w{w:g}-s{seed}.zip"
    log_dir = Path(model_dir) / "ppo-fixed" / f"{scenario.lower()}-w{w:g}-s{seed}-tb"
    eval_dir = Path(output_dir) / f"ppo-fixed-{scenario}" / f"w{w:g}-s{seed}" / "eval"

    if skip_existing and model_path.exists():
        print(f"[ppo-fixed] reuse existing model {model_path}")
    else:
        _train(scenario, seed, total_timesteps, weights, model_path, log_dir)

    summary = _evaluate(model_path, scenario, seed, weights, eval_dir)
    return PointRecord(
        method=tag, scenario=scenario, seed=seed,
        energy_kwh=float(summary["total_energy_kwh"]),
        sla_cost=float(summary["total_sla_cost"]),
        extra={
            "weight_energy": float(w),
            "weight_sla": float(round(1.0 - w, 6)),
            "steps": int(summary.get("steps", -1)),
        },
    )


def _job_list(weights: list[float], seeds: list[int]) -> list[tuple[float, int]]:
    return [(w, s) for w in weights for s in seeds]


def run_ppo_fixed_sweep(
    scenario: str,
    weights: list[float],
    seeds: list[int],
    total_timesteps: int,
    *,
    output_dir: Path,
    model_dir: Path,
    parallel: int = 1,
    base_port: int | None = None,
    skip_existing: bool = False,
) -> list[PointRecord]:
    """Train + evaluate one fixed-weight PPO per (weight, seed); return points.

    ``parallel`` runs that many independent training runs concurrently, each on
    its own gateway — the same lever the CMDP sweep uses (C9). Sequential
    (``parallel=1``) is byte-for-byte the single-run behaviour.
    """
    jobs = _job_list(weights, seeds)
    port0 = int(os.environ.get("GATEWAY_PORT", "25333")) if base_port is None else base_port
    parallel = max(1, min(int(parallel), len(jobs)))

    payloads = [
        (w, seed, scenario, total_timesteps, str(output_dir), str(model_dir),
         skip_existing)
        for w, seed in jobs
    ]

    if parallel == 1:
        return [_run_one_ppo(p) for p in payloads]

    preflight_gateways(port0, parallel)
    print(f"[ppo-fixed] running {len(jobs)} job(s) at concurrency {parallel} "
          f"(gateways :{port0}–{port0 + parallel - 1})")

    import multiprocessing as mp

    ctx = mp.get_context("spawn")
    port_queue = ctx.Queue()
    for i in range(parallel):
        port_queue.put(port0 + i)

    with ctx.Pool(parallel, initializer=_init_worker, initargs=(port_queue,)) as pool:
        points = pool.map(_run_one_ppo, payloads, chunksize=1)
    return list(points)


# ── Summary (the fixed-weight front, per weight seed-mean) ───────────────────

def summarise(points: list[PointRecord]) -> dict:
    """Per-weight aggregates (mean ± 95% CI over seeds) + the seed-mean front."""
    agg = aggregate_by_method(points)
    front = []
    for method in sorted(agg):
        a = agg[method]
        w = None
        for p in points:
            if p.method == method:
                w = float((p.extra or {}).get("weight_energy", "nan"))
                break
        front.append({
            "method": method, "weight_energy": w,
            "energy_kwh": a["energy"].mean, "sla_cost": a["sla"].mean,
        })
    return {
        "front": front,
        "per_weight": {
            m: {
                "energy_mean": a["energy"].mean, "energy_ci95": a["energy"].ci95,
                "sla_mean": a["sla"].mean, "sla_ci95": a["sla"].ci95,
                "seeds": a["seeds"], "n": a["n"],
            }
            for m, a in agg.items()
        },
    }


def print_summary(summary: dict) -> None:
    print("\n" + "=" * 78)
    print("  Fixed-weight PPO front (energy ↔ SLA), seed-mean per weight")
    print("=" * 78)
    print(f"{'w_energy':>10} {'energy kWh':>14} {'C_SLA':>16}")
    for row in summary["front"]:
        w = row["weight_energy"]
        w_str = f"{w:.2g}" if w == w else "?"     # NaN-safe
        print(f"{w_str:>10} {row['energy_kwh']:>14.2f} {row['sla_cost']:>16.4e}")
    print("=" * 78 + "\n")


def main() -> None:
    ap = argparse.ArgumentParser(
        description="G2.6+ — fixed-weight PPO front (front-to-front vs CMDP-PID)")
    ap.add_argument("--scenario", default="HIGH", help="LOW|HIGH|BURST slice the configured trace; with TRACE_PATTERN set, any generated scenario (incl. OVERLOAD, REPLAY) selects its own file")
    ap.add_argument("--weights", default="0.1,0.3,0.5,0.7,0.9",
                    help="comma-separated ENERGY weights w (w_sla = 1-w). "
                         "'0.5' alone = multi-seed ppo-min.")
    ap.add_argument("--seeds", default="42,43,44,45,46",
                    help="comma-separated seeds (≥5 required for reportable CI)")
    ap.add_argument("--total-timesteps", type=int, default=200_000)
    ap.add_argument("--output", default="/data/results")
    ap.add_argument("--models", default="/data/models")
    ap.add_argument("--parallel", type=int, default=1,
                    help="Concurrent training runs, one gateway each (C9). Needs "
                         "the Java side started with NUM_GATEWAYS >= this.")
    ap.add_argument("--base-port", type=int, default=None,
                    help="First gateway port (default: $GATEWAY_PORT).")
    ap.add_argument("--skip-existing", action="store_true",
                    help="Reuse a saved model if present (only re-evaluate) — "
                         "cheap re-runs after a partial crash.")
    args = ap.parse_args()

    # R5/R6: refuse to write WM-1 output onto the LEGACY results.
    paths.guard_results_root(args.output, what="the fixed-weight sweep")

    weights = _parse_floats(args.weights)
    seeds = _parse_ints(args.seeds)
    out_dir = Path(args.output)

    for w in weights:
        if not (0.0 <= w <= 1.0):
            raise SystemExit(f"[ppo-fixed] weight {w} out of [0,1] "
                             f"(these are ENERGY weights; w_sla = 1-w)")
    if len(seeds) < 5:
        print(f"[ppo-fixed] WARNING: {len(seeds)} seed(s) — Lưu ý #10 requires ≥5 "
              f"seeds before any number is reportable. Smoke/wiring check only.")

    points = run_ppo_fixed_sweep(
        args.scenario, weights, seeds, args.total_timesteps,
        output_dir=out_dir, model_dir=Path(args.models),
        parallel=args.parallel, base_port=args.base_port,
        skip_existing=args.skip_existing,
    )

    ppo_dir = out_dir / f"ppo-fixed-{args.scenario}"
    # merge: chạy thêm hạt giống KHÔNG được xoá hạt giống đã có (xem points.save_points)
    save_points(points, ppo_dir / "points.jsonl", merge=True)
    summary = summarise(points)
    print_summary(summary)
    with open(ppo_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[ppo-fixed] points  → {ppo_dir / 'points.jsonl'}")
    print(f"[ppo-fixed] summary → {ppo_dir / 'summary.json'}")
    print(f"[ppo-fixed] → run_campaign will fold ppo-w* into one 'ppo-fixed' "
          f"family for HV/IGD+ (front-to-front vs CMDP-PID).")


if __name__ == "__main__":
    main()
