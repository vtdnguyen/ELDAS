"""G2.3 — pymoo NSGA-II reference Pareto front for the STATIC task-placement
problem (energy ↔ SLA), with a GPU-affinity repair operator.

    Decision variable : host index for each task           (integer, 0..H-1)
    Objectives (min)  : [energy_kwh, sla_cost]              (see static_model)
    Hard constraint   : GPU task ⇒ GPU host  (AffinityRepair, G2.2)

**Scientific caveat (CLAUDE.md Lưu ý #9).** This is an *offline, static* solver
with full trace foreknowledge and no temporal dynamics — an idealised reference
/ upper-bound front, **not** a head-to-head competitor to the online RL
scheduler. Report it as such.

Run (inside Docker, no gateway needed)::

    docker compose run --rm --no-deps rl-agent \
        python src/eval/nsga2_baseline.py --scenario LOW --max-tasks 400

    # heterogeneous topology (CPU-GPU affinity active):
    docker compose run --rm --no-deps rl-agent \
        python src/eval/nsga2_baseline.py --scenario LOW \
        --topology /config/topology-hetero.json
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np

# eval/ is a package under src/; support both `python src/eval/nsga2_baseline.py`
# and `python -m eval.nsga2_baseline`.
try:
    from . import topology as topo_mod
    from . import trace_loader
    from . import paths
    from .static_model import StaticPlacementModel
except ImportError:  # pragma: no cover - direct-script fallback
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from eval import topology as topo_mod
    from eval import trace_loader
    from eval import paths
    from eval.static_model import StaticPlacementModel


# ── pymoo is optional at import time (tests import the model, not this) ──────

def _require_pymoo():
    try:
        from pymoo.algorithms.moo.nsga2 import NSGA2
        from pymoo.core.problem import Problem
        from pymoo.core.repair import Repair
        from pymoo.operators.crossover.sbx import SBX
        from pymoo.operators.mutation.pm import PM
        from pymoo.operators.repair.rounding import RoundingRepair
        from pymoo.operators.sampling.rnd import IntegerRandomSampling
        from pymoo.optimize import minimize
        from pymoo.termination import get_termination
        return dict(
            NSGA2=NSGA2, Problem=Problem, Repair=Repair, SBX=SBX, PM=PM,
            RoundingRepair=RoundingRepair, IntegerRandomSampling=IntegerRandomSampling,
            minimize=minimize, get_termination=get_termination,
        )
    except ImportError as e:  # pragma: no cover
        raise ImportError(
            "pymoo is required for the NSGA-II baseline. "
            "Install it: pip install 'pymoo>=0.6'"
        ) from e


# ── Build the pymoo problem + affinity repair ───────────────────────────────

def build_problem_and_repair(model: StaticPlacementModel, seed: int):
    """Construct the pymoo ``Problem`` and the ``AffinityRepair`` for ``model``."""
    P = _require_pymoo()

    class StaticPlacementProblem(P["Problem"]):
        def __init__(self):
            super().__init__(
                n_var=model.n_tasks,
                n_obj=2,
                n_constr=0,
                xl=0,
                xu=model.n_hosts - 1,
                vtype=int,
            )

        def _evaluate(self, X, out, *args, **kwargs):
            Xi = np.rint(X).astype(np.int64)
            np.clip(Xi, 0, model.n_hosts - 1, out=Xi)
            F = np.empty((Xi.shape[0], 2), dtype=np.float64)
            for k in range(Xi.shape[0]):
                F[k, 0], F[k, 1] = model.evaluate(Xi[k])
            out["F"] = F

    class AffinityRepair(P["Repair"]):
        """G2.2 — force every GPU task onto a GPU host (rounds+clips too)."""

        def __init__(self):
            super().__init__()
            self._rng = np.random.default_rng(seed)

        def _do(self, problem, X, **kwargs):
            Xi = np.rint(X).astype(np.int64)
            np.clip(Xi, 0, model.n_hosts - 1, out=Xi)
            for k in range(Xi.shape[0]):
                Xi[k] = model.repair_affinity(Xi[k], self._rng)
            return Xi.astype(float)

    return StaticPlacementProblem(), AffinityRepair()


# ── Non-dominated filtering (shared with pareto_metrics later) ───────────────

def non_dominated(F: np.ndarray) -> np.ndarray:
    """Return the boolean mask of non-dominated rows (minimisation, both obj)."""
    n = F.shape[0]
    keep = np.ones(n, dtype=bool)
    for i in range(n):
        if not keep[i]:
            continue
        # j dominates i if j ≤ i in all objs and < in at least one.
        dominated = np.all(F <= F[i], axis=1) & np.any(F < F[i], axis=1)
        if dominated.any():
            keep[i] = False
    return keep


# ── Driver ──────────────────────────────────────────────────────────────────

def build_reference_front(
    model: StaticPlacementModel,
    pop_size: int = 100,
    n_gen: int = 80,
    seed: int = 42,
    verbose: bool = False,
) -> dict:
    """Run NSGA-II and return the non-dominated reference front.

    Returns a dict with ``F`` (front objectives, n×2), ``X`` (assignments),
    plus the ``spread`` / ``packed`` anchor objectives for sanity.
    """
    P = _require_pymoo()
    problem, repair = build_problem_and_repair(model, seed)

    algorithm = P["NSGA2"](
        pop_size=pop_size,
        sampling=P["IntegerRandomSampling"](),
        crossover=P["SBX"](prob=0.9, eta=15, vtype=float, repair=P["RoundingRepair"]()),
        mutation=P["PM"](prob=1.0 / model.n_tasks, eta=20, vtype=float,
                         repair=P["RoundingRepair"]()),
        repair=repair,
        eliminate_duplicates=True,
    )

    res = P["minimize"](
        problem,
        algorithm,
        P["get_termination"]("n_gen", n_gen),
        seed=seed,
        verbose=verbose,
        save_history=False,
    )

    X = np.rint(np.atleast_2d(res.X)).astype(np.int64)
    F = np.atleast_2d(res.F).astype(np.float64)

    # Add the two analytic anchors so the reported front always spans the
    # full energy↔SLA range even if NSGA-II under-explores an extreme.
    spread = model.spread_assignment()
    packed = model.packed_assignment()
    anchors_X = np.vstack([spread, packed])
    anchors_F = np.array([model.evaluate(spread), model.evaluate(packed)])

    all_X = np.vstack([X, anchors_X])
    all_F = np.vstack([F, anchors_F])
    mask = non_dominated(all_F)
    front_F = all_F[mask]
    front_X = all_X[mask]

    # Sort the front by energy for readable output.
    order = np.argsort(front_F[:, 0])
    return {
        "F": front_F[order],
        "X": front_X[order],
        "spread": tuple(anchors_F[0]),
        "packed": tuple(anchors_F[1]),
        "n_eval": int(res.algorithm.evaluator.n_eval),
    }


# ── CLI ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="NSGA-II static reference Pareto front (energy vs SLA)"
    )
    parser.add_argument(
        "--scenario", default="LOW",
        help="LOW | HIGH | BURST slice the configured trace; with TRACE_PATTERN set, "
             "any generated scenario (incl. OVERLOAD, REPLAY) selects its own file",
    )
    parser.add_argument(
        "--trace", default=None,
        help="explicit trace CSV; overrides TRACE_PATTERN / TRACE_FILE",
    )
    parser.add_argument(
        "--topology", default=os.environ.get("TOPOLOGY_CONFIG") or None,
        help="topology JSON (default: homogeneous / TOPOLOGY_CONFIG env)",
    )
    parser.add_argument("--num-hosts", type=int, default=10,
                        help="host count for the homogeneous default")
    parser.add_argument(
        "--max-tasks", type=int, default=400,
        help="subsample the trace to this many tasks (0 = all). NSGA-II over "
             "the full trace is huge; the reference front is computed on a "
             "representative subset (documented idealisation).",
    )
    parser.add_argument("--pop-size", type=int, default=100)
    parser.add_argument("--n-gen", type=int, default=80)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--output", default="/data/results")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    # R5/R6: refuse to write WM-1 output onto the LEGACY results.
    paths.guard_results_root(args.output, what="the NSGA-II front")

    seed = args.seed if args.seed is not None else int(os.environ.get("RANDOM_SEED", "42"))

    hosts = topo_mod.load_hosts(args.topology, args.num_hosts)

    # W2.4 — resolve exactly like the simulator does. The reference front is only
    # meaningful if it is computed on the *same* workload the measured points come
    # from; reading the legacy trace here while the DES runs WM-1 would make every
    # hypervolume and IGD+ comparison against this front a comparison of two different
    # experiments.
    trace_path = trace_loader.resolve_trace_path(args.scenario, seed, trace=args.trace)
    patterned = trace_loader.uses_trace_pattern() and not args.trace
    filter_label = trace_loader.PASSTHROUGH if patterned else args.scenario
    tasks = trace_loader.filter_scenario(
        trace_loader.read_trace(trace_path), filter_label
    )
    if args.max_tasks and len(tasks) > args.max_tasks:
        # Even stride keeps the arrival-time spread (better than head slice).
        idx = np.linspace(0, len(tasks) - 1, args.max_tasks).astype(int)
        tasks = [tasks[i] for i in np.unique(idx)]

    print(f"[nsga2] trace={trace_path}"
          f"{' (WM-1, used whole)' if patterned else f' (legacy slice {args.scenario})'}")
    print(f"[nsga2] scenario={args.scenario} hosts={len(hosts)} "
          f"({'hetero' if args.topology else 'homogeneous'}) tasks={len(tasks)} "
          f"pop={args.pop_size} gen={args.n_gen} seed={seed}")

    model = StaticPlacementModel(hosts, tasks)
    result = build_reference_front(
        model, pop_size=args.pop_size, n_gen=args.n_gen, seed=seed,
        verbose=args.verbose,
    )

    F = result["F"]
    print(f"[nsga2] front points={len(F)} evals={result['n_eval']}")
    print(f"[nsga2]   anchor spread (max energy, min SLA): "
          f"E={result['spread'][0]:.2f} kWh, SLA={result['spread'][1]:.3e}")
    print(f"[nsga2]   anchor packed (min energy, max SLA): "
          f"E={result['packed'][0]:.2f} kWh, SLA={result['packed'][1]:.3e}")
    print("[nsga2] reference front (energy_kwh, sla_cost):")
    for e, s in F:
        print(f"           E={e:10.2f}  SLA={s:.4e}")

    # Persist for later hypervolume/IGD+ (G2.4) — fixed schema.
    out_dir = Path(args.output) / f"nsga2-{args.scenario}"
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "scenario": args.scenario,
        "trace": trace_path,
        "trace_mode": "wm1" if patterned else "legacy-slice",
        "topology": args.topology or "homogeneous",
        "num_hosts": len(hosts),
        "num_tasks": len(tasks),
        "seed": seed,
        "pop_size": args.pop_size,
        "n_gen": args.n_gen,
        "objectives": ["energy_kwh", "sla_cost"],
        "sense": ["min", "min"],
        "front": [[float(e), float(s)] for e, s in F],
        "anchor_spread": [float(x) for x in result["spread"]],
        "anchor_packed": [float(x) for x in result["packed"]],
    }
    out_file = out_dir / "reference_front.json"
    with open(out_file, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"[nsga2] reference front saved → {out_file}")


if __name__ == "__main__":
    main()
