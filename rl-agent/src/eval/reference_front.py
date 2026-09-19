"""W5.3 — Multi-algorithm static reference Pareto front.

``nsga2_baseline.py`` builds the reference front with one solver. That is a weak reference:
IGD+ measures how far a method sits from the reference front, so if NSGA-II happens to miss
a region of the true front, *every* method is scored generously in that region and nothing
in the numbers says so. The fix is the standard one — run several unrelated
multi-objective solvers on the same instance and take the non-dominated union, so a region
has to defeat all of them before it goes unrepresented.

Solvers used (all on the identical :class:`StaticPlacementModel` instance):

    NSGA-II     pymoo, via nsga2_baseline  — dominance rank + crowding distance
    MOEA/D      pymultiobjective           — decomposition into scalar subproblems
    SPEA2       pymultiobjective           — strength + density, external archive
    SMS-EMOA    pymultiobjective           — hypervolume-contribution selection

They are deliberately different families: a decomposition method, an archive/strength
method, and an indicator-based method. Four runs of NSGA-II with different seeds would add
points but not diversity of failure mode.

Integer variables through a continuous API
------------------------------------------
pymultiobjective optimises continuous vectors, while a placement is an integer host index
per task with a hard GPU-affinity constraint. Both are handled inside the objective
wrapper: a candidate is rounded, clipped, and affinity-repaired **before** evaluation, so
what the solver explores is always a feasible placement. Two consequences worth stating:

* the repair is **deterministic** (:func:`deterministic_repair`). The two objectives are
  separate callables that the library invokes independently for the same candidate; with a
  random repair they could land on *different* placements and the reported point would be a
  chimera — an energy figure from one assignment paired with an SLA figure from another.
  Determinism plus the memo below make that structurally impossible.
* the solver therefore optimises the repaired landscape, not the raw one. That is the same
  contract NSGA-II already runs under (``AffinityRepair``), so the fronts are comparable.

Scientific caveat (unchanged from ``nsga2_baseline``, CLAUDE.md Lưu ý #9)
------------------------------------------------------------------------
This is a *static* solver with full trace foreknowledge and no temporal dynamics. It is an
idealised reference, **not** a competitor to the online scheduler. Adding solvers makes the
reference tighter; it does not make it a fair opponent.

Run (offline, no gateway)::

    docker compose run --rm --no-deps rl-agent \\
        python src/eval/reference_front.py --scenario LOW --max-tasks 120
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np

try:
    from . import topology as topo_mod
    from . import trace_loader
    from .nsga2_baseline import build_reference_front, non_dominated
    from .static_model import StaticPlacementModel
    from . import paths
except ImportError:  # pragma: no cover - direct-script fallback
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from eval import topology as topo_mod
    from eval import trace_loader
    from eval.nsga2_baseline import build_reference_front, non_dominated
    from eval.static_model import StaticPlacementModel
    from eval import paths


#: Solvers contributed by pymultiobjective, with the attribute name it exports.
PMO_ALGORITHMS = {
    "moead": "multiobjective_evolutionary_algorithm_based_on_decomposition",
    "spea2": "strength_pareto_evolutionary_algorithm_2",
    "sms-emoa": "s_metric_selection_evolutionary_multiobjective_optimization_algorithm",
}

ALL_ALGORITHMS = ("nsga2", *PMO_ALGORITHMS)


class SolverUnavailable(RuntimeError):
    """pymultiobjective is not installed."""


def _require_pymultiobjective():
    try:
        import pyMultiobjective.algorithm as alg
    except ImportError as e:  # pragma: no cover - exercised by the guard test
        raise SolverUnavailable(
            "pymultiobjective is required for the multi-algorithm reference front "
            "(W5.3). Install it: pip install pymultiobjective"
        ) from e
    return alg


# ── Integer + affinity adapter ──────────────────────────────────────────────

def deterministic_repair(model: StaticPlacementModel, x) -> np.ndarray:
    """Round, clip, and move every GPU task onto a GPU host — with no RNG.

    ``StaticPlacementModel.repair_affinity`` picks a random GPU host, which is right for
    NSGA-II (it is applied once per individual, inside the algorithm) but wrong here: the
    library calls each objective separately for the same candidate, so a random repair
    would let the two objectives describe two different placements.

    The deterministic rule spreads offending tasks round-robin over the GPU hosts by task
    index, which preserves the spread NSGA-II's random choice provides on average.
    """
    a = np.rint(np.asarray(x, dtype=np.float64)).astype(np.int64)
    np.clip(a, 0, model.n_hosts - 1, out=a)
    bad = model.needs_gpu & ~model.host_has_gpu[a]
    if bad.any():
        gpu_hosts = model.gpu_host_indices()
        if gpu_hosts.size == 0:
            raise ValueError("GPU tasks present but topology has no GPU host")
        idx = np.nonzero(bad)[0]
        a[idx] = gpu_hosts[idx % gpu_hosts.size]
    return a


def make_objective_functions(model: StaticPlacementModel, cache_size: int = 4096,
                             record_values: set | None = None):
    """``([f_energy, f_sla], stats)`` — feasible-by-construction, memoised.

    The memo is not only a speed-up. The library asks for one objective at a time, so
    without it every candidate is repaired and evaluated twice; with it, both objectives are
    guaranteed to come from *one* evaluation of *one* placement.

    ``record_values`` (opt-in, ``None`` in production) collects every objective pair this
    wrapper ever returned. It exists so a test can assert the strongest property of the
    continuous→integer adapter: every point on a solver's front is a value some **real
    feasible placement** produced, not an artefact of the library exploring between integers.
    """
    cache: dict[bytes, tuple[float, float]] = {}
    stats = {"evaluations": 0, "cache_hits": 0}

    def evaluate(x) -> tuple[float, float]:
        a = deterministic_repair(model, x)
        key = a.tobytes()
        hit = cache.get(key)
        if hit is not None:
            stats["cache_hits"] += 1
            return hit
        val = model.evaluate(a)
        stats["evaluations"] += 1
        if len(cache) < cache_size:
            cache[key] = val
        if record_values is not None:
            record_values.add(val)
        return val

    return [lambda x: evaluate(x)[0], lambda x: evaluate(x)[1]], stats


# ── Running one pymultiobjective solver ─────────────────────────────────────

def run_pmo_algorithm(name: str, model: StaticPlacementModel, *,
                      pop_size: int = 40, generations: int = 40,
                      seed: int = 42, verbose: bool = False,
                      record_values: set | None = None) -> dict:
    """Run one pymultiobjective solver and return its non-dominated objective points.

    Only the objective columns are kept. The variable columns are the *unrepaired*
    continuous vector, so they do not describe the placement that produced the objectives —
    reporting them would invite exactly the confusion the repair is there to avoid.
    """
    if name not in PMO_ALGORITHMS:
        raise ValueError(f"unknown solver {name!r}; expected {sorted(PMO_ALGORITHMS)}")
    alg = _require_pymultiobjective()
    fn = getattr(alg, PMO_ALGORITHMS[name])

    functions, stats = make_objective_functions(model, record_values=record_values)
    lo = [0] * model.n_tasks
    hi = [model.n_hosts - 1] * model.n_tasks

    # pymultiobjective seeds from numpy's global RNG, so pin it for reproducibility.
    np.random.seed(seed)

    kwargs = dict(min_values=lo, max_values=hi, list_of_functions=functions,
                  generations=generations, verbose=verbose)
    # The population argument is spelled differently per algorithm family: the
    # decomposition/indicator solvers derive population size from `references`.
    if name == "spea2":
        kwargs.update(population_size=pop_size, archive_size=pop_size)
    else:
        kwargs.update(references=pop_size)

    t0 = time.perf_counter()
    out = np.asarray(fn(**kwargs), dtype=np.float64)
    elapsed = time.perf_counter() - t0

    F = out[:, -2:]
    F = F[np.isfinite(F).all(axis=1)]
    if len(F) == 0:                                     # pragma: no cover - defensive
        return {"F": np.empty((0, 2)), "seconds": elapsed, **stats}
    F = F[non_dominated(F)]
    return {"F": F[np.argsort(F[:, 0])], "seconds": elapsed, **stats}


# ── Union front ─────────────────────────────────────────────────────────────

def union_front(fronts: dict[str, np.ndarray]) -> dict:
    """Non-dominated union of every solver's points, with provenance.

    ``contributors`` records which solvers reached each surviving point — the evidence for
    whether the extra solvers earned their place or merely re-found what NSGA-II already
    had. Ties are shared: two solvers landing on the identical point both get credit.
    """
    labels: list[str] = []
    rows: list[np.ndarray] = []
    for name, F in fronts.items():
        F = np.atleast_2d(np.asarray(F, dtype=np.float64))
        if F.size == 0:
            continue
        for r in F:
            labels.append(name)
            rows.append(r)
    if not rows:
        raise ValueError("no points from any solver")

    allF = np.vstack(rows)
    keep = non_dominated(allF)
    front = allF[keep]

    # Deduplicate identical points, merging their provenance.
    seen: dict[tuple[float, float], set[str]] = {}
    for r, lab in zip(allF[keep], [labels[i] for i in np.nonzero(keep)[0]]):
        seen.setdefault((float(r[0]), float(r[1])), set()).add(lab)
    # A point one solver found may also be *on* another solver's front even if that copy
    # was dropped as a duplicate by non_dominated; credit it too.
    for r, lab in zip(allF, labels):
        key = (float(r[0]), float(r[1]))
        if key in seen:
            seen[key].add(lab)

    pts = np.array(sorted(seen), dtype=np.float64)
    contributors = [sorted(seen[(float(p[0]), float(p[1]))]) for p in pts]
    return {"F": pts, "contributors": contributors}


def _unique_rows(F) -> np.ndarray:
    """Distinct rows, order-stable. See :func:`solver_igd_report` for why it matters."""
    F = np.atleast_2d(np.asarray(F, dtype=np.float64))
    if F.size == 0:
        return F.reshape(0, 2)
    _, idx = np.unique(F, axis=0, return_index=True)
    return F[np.sort(idx)]


def weakly_dominates_all(front: np.ndarray, other: np.ndarray) -> bool:
    """Is every point of ``other`` matched or beaten by some point of ``front``?

    The acceptance property for W5.3: the union front may never be *worse* anywhere than
    the single-solver front it replaces.
    """
    front = np.atleast_2d(np.asarray(front, dtype=np.float64))
    other = np.atleast_2d(np.asarray(other, dtype=np.float64))
    for p in other:
        if not np.any(np.all(front <= p, axis=1)):
            return False
    return True


def front_gain(union: np.ndarray, baseline: np.ndarray) -> dict:
    """How much the extra solvers actually added over the baseline front.

    Both sides are deduplicated first. ``non_dominated`` keeps exact duplicates (neither
    row strictly dominates the other), and NSGA-II does return them — its raw front had 15
    rows for 14 distinct points on the first run here. Counting those would show the union
    front "shrinking" from 15 to 14 while nothing was actually lost, which reads as a
    regression and is not one.
    """
    union = np.atleast_2d(np.asarray(union, dtype=np.float64))
    baseline = np.atleast_2d(np.asarray(baseline, dtype=np.float64))
    union_rows = {tuple(r) for r in union}
    baseline_rows = {tuple(r) for r in baseline}
    return {
        "n_union": len(union_rows),
        "n_baseline": len(baseline_rows),
        "n_new_points": len(union_rows - baseline_rows),
        "n_baseline_displaced": len(baseline_rows - union_rows),
        "weakly_dominates_baseline": bool(weakly_dominates_all(union, baseline)),
    }


# ── Driver ──────────────────────────────────────────────────────────────────

def build_multi_algorithm_front(
    model: StaticPlacementModel, *, seed: int = 42,
    pop_size: int = 40, generations: int = 40,
    nsga2_pop: int = 100, nsga2_gen: int = 80,
    algorithms=ALL_ALGORITHMS, verbose: bool = False, strict: bool = False,
) -> dict:
    """Run every solver on ``model`` and return the union front plus per-solver detail.

    ``strict`` re-raises a solver failure instead of recording it and carrying on; tests
    use it to assert that a specific failure really happened.
    """
    per_solver: dict[str, np.ndarray] = {}
    detail: dict[str, dict] = {}

    for name in algorithms:
        if name == "nsga2":
            t0 = time.perf_counter()
            res = build_reference_front(model, pop_size=nsga2_pop, n_gen=nsga2_gen,
                                        seed=seed, verbose=verbose)
            per_solver[name] = res["F"]
            detail[name] = {"n_points": int(len(res["F"])),
                            "seconds": time.perf_counter() - t0,
                            "n_eval": res["n_eval"]}
            anchors = {"spread": res["spread"], "packed": res["packed"]}
            continue

        # One solver blowing up must not destroy the whole reference front. SMS-EMOA in
        # particular divides by the population's objective range during normalisation, so a
        # population that collapses onto one objective value yields NaN and pagmo's
        # hypervolume refuses it. A union built from three of four solvers is still a valid
        # union — losing the two that worked because the third crashed is not. The failure
        # is recorded, not swallowed: it appears in `detail` and in the saved JSON.
        try:
            res = run_pmo_algorithm(name, model, pop_size=pop_size,
                                    generations=generations, seed=seed, verbose=verbose)
        except SolverUnavailable:
            raise
        except Exception as e:
            if strict:
                raise
            # pagmo raises with a multi-line, leading-newline message; flatten it or the
            # report prints "FAILED: ValueError:" and says nothing at all.
            msg = " ".join(str(e).split()) or "(no message)"
            detail[name] = {"n_points": 0, "seconds": 0.0, "n_eval": 0,
                            "error": f"{type(e).__name__}: {msg}"}
            per_solver[name] = np.empty((0, 2))
            print(f"[ref-front] WARNING: {name} failed and contributed nothing "
                  f"({type(e).__name__}: {msg[:120]})")
            continue

        per_solver[name] = res["F"]
        detail[name] = {"n_points": int(len(res["F"])), "seconds": res["seconds"],
                        "n_eval": res["evaluations"],
                        "cache_hits": res["cache_hits"]}

    merged = union_front(per_solver)
    baseline = per_solver.get("nsga2", np.empty((0, 2)))
    return {
        "F": merged["F"],
        "contributors": merged["contributors"],
        "per_solver": {k: v for k, v in per_solver.items()},
        "detail": detail,
        "gain_over_nsga2": front_gain(merged["F"], baseline),
        "anchors": anchors if "nsga2" in per_solver else {},
    }


# ── Re-scoring the campaign against the richer front (Lưu ý #8) ─────────────

def rescore_against_fronts(points_by_method: dict[str, np.ndarray],
                           old_front: np.ndarray, new_front: np.ndarray,
                           backend: str | None = None) -> dict:
    """Score every method against BOTH reference fronts, so the change is visible.

    CLAUDE.md Lưu ý #8 forbids mixing indicator values computed under different reference
    geometry. Applying that warning correctly here needs one distinction the warning does
    not make, and getting it backwards would produce a confidently wrong table:

    * **HV does not move.** The reference point is derived from the *methods'* own points
      (``ideal``/``nadir`` over ``method_points``), and a reference front is only ever used
      for IGD+. So swapping in a richer front cannot change any hypervolume. This function
      asserts that rather than assuming it — ``hv_invariant_holds``.
    * **IGD+ moves, and it should get WORSE (larger).** A richer front sits closer to the
      true front, so every method is now measured against a harder target. An IGD+ that
      *improved* would mean the added solvers found points that are easier to reach than
      the ones they replaced — i.e. a worse reference — and is a reason to investigate, not
      a success.

    Both columns come back so a report can never quote one as though it were the other.
    """
    try:
        from . import pareto_metrics as pm
    except ImportError:  # pragma: no cover - direct-script fallback
        from eval import pareto_metrics as pm

    old = pm.evaluate_methods(points_by_method, reference_front=old_front,
                              backend=backend)
    new = pm.evaluate_methods(points_by_method, reference_front=new_front,
                              backend=backend)

    rows = {}
    for name in points_by_method:
        o, n = old["methods"][name], new["methods"][name]
        rows[name] = {
            "hypervolume": n["hypervolume"],
            "hypervolume_old_ref": o["hypervolume"],
            "hv_unchanged": bool(abs(n["hypervolume"] - o["hypervolume"]) <= 1e-12),
            "igd_plus_old_ref": o["igd_plus"],
            "igd_plus_new_ref": n["igd_plus"],
            "igd_plus_delta": n["igd_plus"] - o["igd_plus"],
            "n_points": n["n_points"],
        }
    return {
        "ideal": new["ideal"], "nadir": new["nadir"], "backend": new["backend"],
        "methods": rows,
        "hv_invariant_holds": all(r["hv_unchanged"] for r in rows.values()),
        "igd_plus_never_improved": all(r["igd_plus_delta"] >= -1e-12
                                       for r in rows.values()),
    }


def solver_igd_report(per_solver: dict[str, np.ndarray], union: np.ndarray,
                      baseline: np.ndarray, backend: str | None = None) -> dict:
    """IGD+ of each *solver's own* front against the union and against NSGA-II alone.

    This is where the reference front legitimately changes an IGD+ number. It is **not**
    the DES campaign: the static front and the CloudSim-measured points come from
    different evaluators on different instances, and ``run_campaign.nsga2_commensurable``
    keeps them off shared axes for exactly that reason (Lưu ý #9). Scoring the campaign
    against this front would be the head-to-head comparison the plan forbids elsewhere, so
    the comparison stays inside the static domain, where all four solvers optimise the
    identical objective on the identical instance.

    Reading it: a solver with a large IGD+ against the union is one the union rescued the
    reference from. NSGA-II's own value is the headline — it says how much a single-solver
    reference was missing.
    """
    try:
        from . import pareto_metrics as pm
    except ImportError:  # pragma: no cover - direct-script fallback
        from eval import pareto_metrics as pm

    fronts = {k: np.atleast_2d(np.asarray(v, dtype=np.float64))
              for k, v in per_solver.items() if len(np.atleast_2d(v)) > 0}
    if not fronts:
        raise ValueError("no solver fronts to score")

    # Both reference sets MUST be deduplicated. IGD+ averages over the reference points, so
    # a repeated point is counted twice and drags the mean toward itself. NSGA-II returns
    # duplicates (93 raw rows for 7 distinct points on HIGH) while the union is already
    # deduplicated, and comparing the two as-is produced deltas of ±0.05 — including a
    # *negative* one — on scenarios where the two fronts are the identical point set and
    # the true delta is exactly 0.
    union = _unique_rows(union)
    baseline = _unique_rows(baseline)

    # One fixed geometry over everything involved, so the two IGD+ columns are on the same
    # axes (Lưu ý #8) — including the union, which may extend beyond any single solver.
    all_pts = np.vstack([*fronts.values(), union, baseline])
    ideal = pm.ideal_point(all_pts)
    nadir = pm.nadir_point(all_pts)

    rows = {}
    for name, F in fronts.items():
        vs_union = pm.igd_plus(F, union, ideal, nadir, backend=backend)
        vs_base = pm.igd_plus(F, baseline, ideal, nadir, backend=backend)
        rows[name] = {
            "igd_plus_vs_union": vs_union,
            "igd_plus_vs_nsga2_only": vs_base,
            "delta": vs_union - vs_base,
            "n_points": int(len(F)),
        }
    return {"ideal": ideal.tolist(), "nadir": nadir.tolist(), "solvers": rows,
            "harder_against_union": all(r["delta"] >= -1e-12 for r in rows.values())}


# ── CLI ─────────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser(
        description="Multi-algorithm static reference Pareto front (W5.3)")
    ap.add_argument("--scenario", default="LOW")
    ap.add_argument("--trace", default=None)
    ap.add_argument("--topology", default=os.environ.get("TOPOLOGY_CONFIG") or None)
    ap.add_argument("--num-hosts", type=int, default=10)
    ap.add_argument("--max-tasks", type=int, default=120,
                    help="subsample size; pymultiobjective is pure Python, so the "
                         "instance has to stay modest (documented idealisation)")
    ap.add_argument("--pop-size", type=int, default=40, help="pymultiobjective solvers")
    ap.add_argument("--generations", type=int, default=40)
    ap.add_argument("--nsga2-pop", type=int, default=100)
    ap.add_argument("--nsga2-gen", type=int, default=80)
    ap.add_argument("--algorithms", default=",".join(ALL_ALGORITHMS))
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--output", default="/data/results")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    # R5/R6: refuse to write WM-1 output onto the LEGACY results.
    paths.guard_results_root(args.output, what="the reference front")

    seed = args.seed if args.seed is not None else int(os.environ.get("RANDOM_SEED", "42"))
    hosts = topo_mod.load_hosts(args.topology, args.num_hosts)

    trace_path = trace_loader.resolve_trace_path(args.scenario, seed, trace=args.trace)
    patterned = trace_loader.uses_trace_pattern() and not args.trace
    label = trace_loader.PASSTHROUGH if patterned else args.scenario
    tasks = trace_loader.filter_scenario(trace_loader.read_trace(trace_path), label)
    if args.max_tasks and len(tasks) > args.max_tasks:
        idx = np.linspace(0, len(tasks) - 1, args.max_tasks).astype(int)
        tasks = [tasks[i] for i in np.unique(idx)]

    algorithms = tuple(a.strip() for a in args.algorithms.split(",") if a.strip())
    print(f"[ref-front] trace={trace_path} "
          f"{'(WM-1, whole)' if patterned else f'(legacy slice {args.scenario})'}")
    print(f"[ref-front] hosts={len(hosts)} tasks={len(tasks)} seed={seed} "
          f"solvers={list(algorithms)}")

    model = StaticPlacementModel(hosts, tasks)
    res = build_multi_algorithm_front(
        model, seed=seed, pop_size=args.pop_size, generations=args.generations,
        nsga2_pop=args.nsga2_pop, nsga2_gen=args.nsga2_gen,
        algorithms=algorithms, verbose=args.verbose)

    print("\n[ref-front] per solver:")
    for name, d in res["detail"].items():
        suffix = f"   FAILED: {d['error']}" if "error" in d else ""
        print(f"    {name:<10} {d['n_points']:>3} pts  {d['seconds']:>7.1f}s  "
              f"{d['n_eval']:>6} evals{suffix}")
    print("    (evals are the budget each solver actually spent — the union is only a "
          "fair\n     'no solver found more' claim if these are comparable)")

    g = res["gain_over_nsga2"]
    print(f"\n[ref-front] union front: {g['n_union']} points "
          f"(NSGA-II alone: {g['n_baseline']})")
    print(f"    new points contributed by the extra solvers : {g['n_new_points']}")
    print(f"    NSGA-II points displaced (were dominated)   : {g['n_baseline_displaced']}")
    print(f"    union weakly dominates the NSGA-II front    : "
          f"{g['weakly_dominates_baseline']}")

    credit: dict[str, int] = {}
    for cs in res["contributors"]:
        for c in cs:
            credit[c] = credit.get(c, 0) + 1
    print("    points on the union front by solver         : "
          + ", ".join(f"{k}={v}" for k, v in sorted(credit.items())))

    igd = None
    if "nsga2" in res["per_solver"]:
        igd = solver_igd_report(res["per_solver"], res["F"], res["per_solver"]["nsga2"])
        print("\n[ref-front] IGD+ of each solver's own front (static domain only):")
        print(f"    {'solver':<10} {'vs union':>10} {'vs nsga2-only':>14} {'delta':>9}")
        for name, r in sorted(igd["solvers"].items()):
            print(f"    {name:<10} {r['igd_plus_vs_union']:>10.5f} "
                  f"{r['igd_plus_vs_nsga2_only']:>14.5f} {r['delta']:>+9.5f}")
        print(f"    a richer reference is a harder target (delta >= 0 for all): "
              f"{igd['harder_against_union']}")
        print("    NOTE: the DES campaign is NOT rescored against this front — different "
              "evaluator\n          and instance, see run_campaign.nsga2_commensurable "
              "(Lưu ý #9).")

    out_dir = Path(args.output) / f"reffront-{args.scenario}"
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "scenario": args.scenario, "trace": trace_path,
        "trace_mode": "wm1" if patterned else "legacy-slice",
        "topology": args.topology or "homogeneous",
        "num_hosts": len(hosts), "num_tasks": len(tasks), "seed": seed,
        "algorithms": list(algorithms),
        "objectives": ["energy_kwh", "sla_cost"], "sense": ["min", "min"],
        "front": [[float(a), float(b)] for a, b in res["F"]],
        "contributors": res["contributors"],
        "per_solver": {k: [[float(a), float(b)] for a, b in np.atleast_2d(v)]
                       for k, v in res["per_solver"].items()},
        "detail": res["detail"],
        "gain_over_nsga2": g,
        "solver_igd_plus": igd,
        "anchors": {k: [float(x) for x in v] for k, v in res["anchors"].items()},
        "caveat": ("static solver with full trace foreknowledge and no temporal "
                   "dynamics — an idealised reference, not a competitor to the online "
                   "scheduler (CLAUDE.md Lưu ý #9)"),
    }
    out_file = out_dir / "reference_front_multi.json"
    out_file.write_text(json.dumps(payload, indent=2))
    print(f"\n[ref-front] saved → {out_file}")


if __name__ == "__main__":
    main()
