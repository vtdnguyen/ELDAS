"""W5.2 — Compare two stochastic multi-objective methods with the Empirical Attainment
Function (EAF), as a second opinion on the hypervolume bootstrap.

Why a second opinion at all
---------------------------
``hv_bootstrap.py`` answers "is CMDP-PID's hypervolume different from fixed-weight PPO's,
or is the gap seed noise?" — and it is the only answer the campaign currently has. That is
uncomfortable, because a hypervolume verdict depends on two choices that the EAF does not
need:

* **a reference point.** HV is only defined against a nadir, and moving it changes not just
  the magnitude but occasionally the ordering of two fronts. The EAF uses dominance alone,
  so it is invariant to any monotone rescaling of either objective — no reference point, no
  normalisation, nothing to tune.
* **collapsing runs before scoring.** The bootstrap builds one front per family by
  averaging each operating point over seeds, then scores that single front. The EAF keeps
  every run separate and asks, for each region of objective space, *what fraction of runs
  reached it*. A method that is excellent on three seeds and poor on two looks identical to
  a consistently mediocre one after averaging; under the EAF it does not.

So the two are genuinely different questions, and agreement between them is evidence rather
than tautology. **Disagreement is a finding, not a bug to paper over** — it would say the
verdict rests on the reference point or on the averaging, which is exactly what a thesis
committee should be told.

What is computed
----------------
A **run** is one seed. That run's front is its operating points (each budget ``d`` for
CMDP-PID, each weight ``w`` for fixed-weight PPO, the single point for a heuristic).

The EAF of a method at a point ``z`` is the fraction of its runs that *attain* ``z`` — that
is, runs holding at least one solution that weakly dominates ``z``::

    EAF_A(z) = |{ run r in A : exists p in front(r) with p <= z componentwise }| / |A|

The comparison statistic is the two-sided Kolmogorov-Smirnov-type maximum deviation::

    T = max_z | EAF_A(z) - EAF_B(z) |

with the two one-sided pieces reported separately so the *direction* is explicit. Because
both EAFs are step functions that only change at the observed coordinate values, evaluating
on the cross-product grid of those values is **exact**, not an approximation (see
:func:`probe_grid`).

Significance comes from an **exact permutation test** over run labels: under the null the
runs are exchangeable between the two methods, so every way of splitting the pooled runs
into groups of the original sizes is equally likely. With 5 seeds each there are only
C(10,5) = 252 splits, so the null distribution is enumerated exactly rather than sampled.
That also fixes the resolution floor: the smallest attainable p-value is 1/252 = 0.004, so
"p = 0.004" means "as extreme as anything possible here", not "p < 0.001".

Run (offline — needs only numpy and the point files)::

    docker compose run --rm --no-deps \
        -v "$PWD/rl-agent/src:/app/src:ro" --entrypoint python rl-agent \
        src/eval/eaf_compare.py --scenario HIGH \
        --compare "cmdp-pid,ppo-fixed;cmdp-pid,random"
"""

from __future__ import annotations

import argparse
import itertools
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

try:
    from .points import PointRecord, load_points
    from .run_campaign import metric_family
    from . import paths
except ImportError:  # pragma: no cover - direct-script fallback
    from eval.points import PointRecord, load_points
    from eval.run_campaign import metric_family
    from eval import paths

#: Above this many label splits the permutation test samples instead of enumerating.
#: 5v5 seeds gives 252, so the campaign always takes the exact path.
EXACT_PERMUTATION_LIMIT = 20_000


# ── EAF core ────────────────────────────────────────────────────────────────

def attains(front: np.ndarray, probes: np.ndarray) -> np.ndarray:
    """``(n_probes,)`` bool: does this run hold a solution weakly dominating each probe?

    Weak dominance (``<=`` on every objective) is the definition of attainment, so a run
    whose solution sits exactly on the probe counts as attaining it.
    """
    front = np.atleast_2d(np.asarray(front, dtype=np.float64))
    probes = np.atleast_2d(np.asarray(probes, dtype=np.float64))
    if len(front) == 0:
        return np.zeros(len(probes), dtype=bool)
    # (n_probes, n_front, m) -> all objectives satisfied -> any solution qualifies.
    le = front[None, :, :] <= probes[:, None, :]
    return le.all(axis=2).any(axis=1)


def eaf_values(runs: list[np.ndarray], probes: np.ndarray) -> np.ndarray:
    """``(n_probes,)`` attainment frequency in [0, 1] over the runs of one method."""
    if not runs:
        raise ValueError("no runs supplied")
    hits = np.zeros(len(np.atleast_2d(probes)), dtype=np.float64)
    for front in runs:
        hits += attains(front, probes)
    return hits / len(runs)


def probe_grid(runs_a: list[np.ndarray], runs_b: list[np.ndarray]) -> np.ndarray:
    """Cross-product of every observed coordinate value — an EXACT probe set.

    Both EAFs are piecewise constant on the cells cut by the observed x- and y-values, and
    each cell's lower-left corner is one of these grid points, so the maximum deviation is
    attained at a grid point. Nothing is missed and nothing is approximated; a random or
    uniform probe set would silently under-report the statistic.

    Only defined for 2 objectives, which is what Phase 2 uses throughout.
    """
    pts = [np.atleast_2d(np.asarray(f, dtype=np.float64))
           for f in (*runs_a, *runs_b) if len(f) > 0]
    if not pts:
        raise ValueError("no points in either method")
    allp = np.vstack(pts)
    if allp.shape[1] != 2:
        raise ValueError("probe_grid supports 2 objectives only "
                         f"(got {allp.shape[1]})")
    xs = np.unique(allp[:, 0])
    ys = np.unique(allp[:, 1])
    gx, gy = np.meshgrid(xs, ys, indexing="ij")
    return np.column_stack([gx.ravel(), gy.ravel()])


def eaf_statistic(runs_a: list[np.ndarray], runs_b: list[np.ndarray],
                  probes: np.ndarray | None = None) -> dict:
    """Two-sided KS-type EAF deviation plus its two one-sided components.

    ``d_plus``  = max_z (EAF_A - EAF_B): the region where A attains more often than B.
    ``d_minus`` = max_z (EAF_B - EAF_A): the reverse.
    ``T`` = max of the two — the statistic the permutation test works on.

    Direction is read off ``d_plus`` vs ``d_minus``, not off ``T``, which is deliberately
    sign-blind so the null distribution is symmetric under relabelling.
    """
    probes = probe_grid(runs_a, runs_b) if probes is None else probes
    ea = eaf_values(runs_a, probes)
    eb = eaf_values(runs_b, probes)
    diff = ea - eb
    # `+ 0.0` normalises negative zero, which max() propagates and which prints as "-0.000".
    d_plus = float(max(diff.max(), 0.0)) + 0.0
    d_minus = float(max((-diff).max(), 0.0)) + 0.0
    return {
        "d_plus": d_plus,
        "d_minus": d_minus,
        "T": float(max(d_plus, d_minus)),
        "n_probes": int(len(probes)),
        "mean_signed_diff": float(diff.mean()),
    }


def permutation_test(runs_a: list[np.ndarray], runs_b: list[np.ndarray], *,
                     n_perm: int = EXACT_PERMUTATION_LIMIT,
                     rng: np.random.Generator | None = None) -> dict:
    """Exact (or sampled) permutation test on the two-sided EAF statistic.

    Under H0 the runs are exchangeable, so relabelling them gives the null distribution of
    ``T``. The observed labelling is included in the count — omitting it can produce
    ``p = 0``, which is never a defensible claim from a finite permutation set.
    """
    pooled = [*runs_a, *runs_b]
    na, nb = len(runs_a), len(runs_b)
    if na == 0 or nb == 0:
        raise ValueError(f"both methods need runs (got {na} and {nb})")

    probes = probe_grid(runs_a, runs_b)          # fixed across permutations
    observed = eaf_statistic(runs_a, runs_b, probes)

    n_total = na + nb
    n_splits = 1
    for k in range(na):
        n_splits = n_splits * (n_total - k) // (k + 1)

    exact = n_splits <= n_perm
    if exact:
        splits = itertools.combinations(range(n_total), na)
    else:                                        # pragma: no cover - not hit at 5v5
        rng = rng or np.random.default_rng(0)
        splits = (tuple(rng.permutation(n_total)[:na]) for _ in range(n_perm))

    ge = 0
    total = 0
    for idx in splits:
        sel = set(idx)
        a = [pooled[i] for i in range(n_total) if i in sel]
        b = [pooled[i] for i in range(n_total) if i not in sel]
        t = eaf_statistic(a, b, probes)["T"]
        total += 1
        if t >= observed["T"] - 1e-12:
            ge += 1

    return {
        **observed,
        "p_value": ge / total,
        "n_permutations": total,
        "exact": exact,
        # The floor is worth carrying into the report: at 5v5 no test can resolve below
        # 1/252, so a p at the floor means "maximally extreme here", not "vanishingly small".
        "p_resolution": 1.0 / total,
        "n_runs_a": na,
        "n_runs_b": nb,
    }


#: ``better`` values that are not a method name.
NO_DIFFERENCE = "none"       #: the test could not distinguish the two
CROSSING = "crossing"        #: significantly different, but neither is uniformly better

#: Below this the two one-sided deviations count as equal, i.e. each method reaches regions
#: the other misses. EAF values are multiples of 1/n_runs, so anything under half a run's
#: worth of advantage is not a direction.
_DIRECTION_EPS = 1e-9


def compare(runs_a: list[np.ndarray], runs_b: list[np.ndarray], *,
            name_a: str = "A", name_b: str = "B", alpha: float = 0.05,
            n_perm: int = EXACT_PERMUTATION_LIMIT) -> dict:
    """Full verdict for one pair: statistic, p-value, direction, significance.

    Three outcomes, not two. ``d_plus`` and ``d_minus`` measure *different* regions of
    objective space, so a significant result with ``d_plus ~= d_minus`` means each method
    attains ground the other cannot — the fronts **cross**. Reporting that as a "tie" would
    merge it with "we found no difference", which is close to the opposite claim: one says
    the methods are interchangeable, the other says they are distinguishable but not
    ordered. Both occur in the campaign data (LOW crosses; HIGH does not).
    """
    res = permutation_test(runs_a, runs_b, n_perm=n_perm)
    significant = res["p_value"] <= alpha

    # Direction is decided by the LOSER's deviation, not by the gap between the two.
    #
    # ``d_plus`` and ``d_minus`` are maxima over *disjoint* regions of objective space,
    # so ranking them does not order the methods: d_plus=0.80 alongside d_minus=1.00
    # says A reaches ground B never reaches AND B reaches ground A never reaches — two
    # crossing fronts — yet a gap test reads the 0.20 difference as "B is better".
    # Measured on homo/LOW, where it declared ppo-fixed the winner over cmdp-pid while
    # the paired comparison had cmdp-pid using 119 kWh less on 5 seeds out of 5 and the
    # hypervolume bootstrap said the opposite with a CI excluding zero. The tool then
    # reported that as a CONTRADICTION against HV, when the two were not in conflict at
    # all — only this rule was wrong.
    #
    # A method is uniformly better only when the other has essentially no exclusive
    # ground. EAF values are multiples of 1/n_runs, so the smallest visible deviation is
    # one whole run; that much is a single seed poking out and is not distinguishable
    # from sampling noise. Two or more runs consistently reaching ground the other never
    # reaches is a crossing. Hence the line sits AT one run, not below it — putting it at
    # half a run would make every non-zero deviation a crossing and the verdict useless.
    eps = max(_DIRECTION_EPS, 1.0 / max(1, min(res["n_runs_a"], res["n_runs_b"])))
    gap = res["d_plus"] - res["d_minus"]
    if not significant:
        direction = NO_DIFFERENCE
    elif min(res["d_plus"], res["d_minus"]) > eps:
        direction = CROSSING
    elif gap > _DIRECTION_EPS:
        direction = name_a
    elif gap < -_DIRECTION_EPS:
        direction = name_b
    else:
        direction = CROSSING
    return {**res, "a": name_a, "b": name_b, "alpha": alpha,
            "significant": bool(significant), "better": direction}


# ── moocore cross-check (independent implementation) ────────────────────────

def moocore_surface_check(runs: list[np.ndarray],
                          percentiles=(25, 50, 75, 100)) -> dict:
    """Verify our EAF against moocore's exact attainment surfaces.

    moocore is the reference C implementation from the group that introduced the EAF, and
    it shares no code with the function above. Every point on its ``p``-percentile surface
    must be attained by at least ``p`` % of runs under our definition; a wrong inequality
    direction or an off-by-one in the run count fails this immediately.

    Returns ``{"available": False}`` when moocore is not installed, so the check degrades
    to a skip rather than an error.
    """
    try:
        import moocore
    except ImportError:
        return {"available": False}

    pts, sets = [], []
    for i, front in enumerate(runs, start=1):
        f = np.atleast_2d(np.asarray(front, dtype=np.float64))
        if len(f) == 0:
            continue
        pts.append(f)
        sets.append(np.full(len(f), i))
    if not pts:
        return {"available": False}
    points = np.vstack(pts)
    set_ids = np.concatenate(sets)

    worst = 0.0
    checked = 0
    for p in percentiles:
        surface = moocore.eaf(points, sets=set_ids, percentiles=[p])
        if len(surface) == 0:
            continue
        z = np.asarray(surface)[:, :2]
        got = eaf_values(runs, z)
        # Our value may exceed p/100 (the surface is the p-th percentile boundary), but it
        # must never be below it.
        worst = max(worst, float((p / 100.0 - got).max()))
        checked += len(z)
    return {"available": True, "max_shortfall": worst, "n_surface_points": checked,
            "ok": worst <= 1e-9}


# ── Campaign wiring ─────────────────────────────────────────────────────────

def load_all_points(results_dir: Path, scenario: str) -> list[PointRecord]:
    """The same three point files ``hv_bootstrap`` reads, so both verdicts use one dataset.

    Reading a different set would make any disagreement uninterpretable — it could be the
    method or it could be the data.
    """
    files = [
        results_dir / f"campaign-{scenario}" / "points.jsonl",   # heuristics + random
        results_dir / f"sweep-{scenario}" / "points.jsonl",       # cmdp-d*
        results_dir / f"ppo-fixed-{scenario}" / "points.jsonl",   # ppo-w*
    ]
    points: list[PointRecord] = []
    for f in files:
        if f.exists():
            points += load_points(f)
    if not points:
        raise SystemExit(
            f"[eaf] no points for {scenario} under {results_dir}. Run the campaign "
            f"(and sweep / ppo-fixed) first.")
    return points


def runs_by_family(points: list[PointRecord]) -> dict[str, dict[int, np.ndarray]]:
    """``{family: {seed: (n_operating_points x 2) front}}``.

    One run = one seed, and its front is every operating point that seed produced. This is
    the step that differs from ``hv_bootstrap``, which averages each operating point across
    seeds *before* scoring — see the module docstring for why that matters.
    """
    tree: dict[str, dict[int, list[tuple[float, float]]]] = \
        defaultdict(lambda: defaultdict(list))
    for p in points:
        fam = metric_family(p.method)
        if fam == "nsga2":                       # static reference, not a competitor
            continue
        tree[fam][int(p.seed)].append((float(p.energy_kwh), float(p.sla_cost)))
    return {fam: {s: np.asarray(v, dtype=np.float64) for s, v in by_seed.items()}
            for fam, by_seed in tree.items()}


def hv_bootstrap_verdicts(results_dir: Path, scenario: str) -> dict[tuple[str, str], dict]:
    """Read ``hv_bootstrap.json`` so the EAF report can state agreement explicitly."""
    path = results_dir / f"campaign-{scenario}" / "hv_bootstrap.json"
    if not path.exists():
        return {}
    data = json.loads(path.read_text())
    out = {}
    for g in data.get("gaps", []):
        if "error" in g:
            continue
        out[(g["a"], g["b"])] = g
    return out


def _hv_direction(gap: dict) -> str:
    if not gap.get("significant_95"):
        return NO_DIFFERENCE
    return gap["a"] if gap["delta_point"] > 0 else gap["b"]


def agreement(eaf_result: dict, hv_gap: dict | None) -> dict:
    """Do the EAF and the HV bootstrap point the same way?

    The only outcome that stops the work is the two naming **opposite winners** — that would
    mean the verdict is an artefact of the reference point or of averaging runs before
    scoring, and neither number could be reported. Everything else is a difference in what
    the two indicators can resolve:

    * one inconclusive — expected, they have different power at 5 runs;
    * EAF says the fronts *cross* while HV names a winner — also expected and informative:
      HV integrates area against a fixed nadir, so a method can enclose more volume while
      still failing to reach regions the other reaches. Not a contradiction, but it does
      mean "method X is better" must be stated as "X has larger hypervolume", not as "X
      dominates".
    """
    if hv_gap is None:
        return {"status": "no-hv-baseline"}
    hv_dir = _hv_direction(hv_gap)
    eaf_dir = eaf_result["better"]
    if hv_dir == eaf_dir:
        status = "agree"
    elif eaf_dir == CROSSING:
        status = "partial (EAF: fronts cross, no uniform winner)"
    elif NO_DIFFERENCE in (hv_dir, eaf_dir):
        status = "partial (one is inconclusive)"
    else:
        status = "CONTRADICTION"
    return {"status": status, "hv_direction": hv_dir, "eaf_direction": eaf_dir,
            "hv_delta": hv_gap.get("delta_point"),
            "hv_significant": hv_gap.get("significant_95")}


# ── Reporting ───────────────────────────────────────────────────────────────

def print_report(scenario: str, results: list[dict], cross: dict, seeds_by_family: dict,
                 alpha: float) -> None:
    print("\n" + "=" * 78)
    print(f"  EAF comparison - {scenario}   (one run = one seed; no reference point)")
    print("=" * 78)
    for fam, seeds in sorted(seeds_by_family.items()):
        print(f"  {fam:>12}: {len(seeds)} runs, seeds {sorted(seeds)}")
    if cross.get("available"):
        verdict = "OK" if cross.get("ok") else f"MISMATCH ({cross['max_shortfall']:.3g})"
        print(f"\n  moocore cross-check on {cross['n_surface_points']} EAF surface "
              f"points: {verdict}")
    else:
        print("\n  moocore not installed - EAF cross-check skipped")

    for r in results:
        e, agr = r["eaf"], r["agreement"]
        print("-" * 78)
        print(f"  {e['a']}  vs  {e['b']}")
        print(f"    max EAF advantage  {e['a']}: {e['d_plus']:.3f}   "
              f"{e['b']}: {e['d_minus']:.3f}")
        print(f"    T = {e['T']:.3f}   p = {e['p_value']:.4f} "
              f"({'exact' if e['exact'] else 'sampled'}, {e['n_permutations']} splits, "
              f"resolution {e['p_resolution']:.4f})")
        if e["better"] == CROSSING:
            verdict = ("SIGNIFICANT but the fronts CROSS - each reaches ground the "
                       "other misses, so neither is uniformly better")
        elif e["significant"]:
            verdict = f"SIGNIFICANT at alpha={alpha}; better: {e['better']}"
        else:
            verdict = f"not significant at alpha={alpha} - no difference detected"
        print(f"    => {verdict}")
        if agr["status"] != "no-hv-baseline":
            print(f"    HV bootstrap said: {agr['hv_direction']} "
                  f"(dHV={agr['hv_delta']:+.4f}, sig={agr['hv_significant']})")
            print(f"    AGREEMENT: {agr['status']}")
    print("=" * 78 + "\n")


def main() -> None:
    ap = argparse.ArgumentParser(
        description="EAF comparison of two method families (W5.2)")
    ap.add_argument("--scenario", default="HIGH")
    ap.add_argument("--results", default="/data/results")
    ap.add_argument("--compare", default="cmdp-pid,ppo-fixed;cmdp-pid,random",
                    help="pairs 'a,b' joined by ';'")
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--out", default=None,
                    help="JSON output (default: campaign-<sc>/eaf_compare.json)")
    args = ap.parse_args()

    # R5/R6: refuse to write WM-1 output onto the LEGACY results.
    paths.guard_results_root(args.results, what="the EAF comparison")

    results_dir = Path(args.results)
    points = load_all_points(results_dir, args.scenario)
    tree = runs_by_family(points)
    hv_gaps = hv_bootstrap_verdicts(results_dir, args.scenario)

    seeds_by_family = {f: list(d.keys()) for f, d in tree.items()}
    cross: dict = {"available": False}
    for fam in ("cmdp-pid", "ppo-fixed"):
        if fam in tree:
            cross = moocore_surface_check(list(tree[fam].values()))
            break

    results = []
    contradictions = 0
    for pair in args.compare.split(";"):
        pair = pair.strip()
        if not pair:
            continue
        a, _, b = pair.partition(",")
        a, b = a.strip(), b.strip()
        if a not in tree or b not in tree:
            print(f"[eaf] skipping {a} vs {b}: family missing (have {sorted(tree)})")
            continue
        res = compare(list(tree[a].values()), list(tree[b].values()),
                      name_a=a, name_b=b, alpha=args.alpha)
        agr = agreement(res, hv_gaps.get((a, b)))
        if agr["status"] == "CONTRADICTION":
            contradictions += 1
        results.append({"eaf": res, "agreement": agr})

    print_report(args.scenario, results, cross, seeds_by_family, args.alpha)

    out = Path(args.out) if args.out else \
        results_dir / f"campaign-{args.scenario}" / "eaf_compare.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "scenario": args.scenario, "alpha": args.alpha,
        "seeds_by_family": {k: sorted(v) for k, v in seeds_by_family.items()},
        "moocore_cross_check": cross,
        "comparisons": results,
    }, indent=2))
    print(f"[eaf] written -> {out}")

    if contradictions:
        print(f"[eaf] {contradictions} CONTRADICTION(s) against the HV bootstrap — "
              f"investigate before reporting either (PLAN W5.2).")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
