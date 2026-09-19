"""Bootstrap 95 % confidence intervals for hypervolume (HV) — and a *paired*
CI for the HV gap between two method families (e.g. CMDP-PID vs fixed-weight
PPO).

Why this exists
---------------
The campaign table (``run_campaign.py``) reports ONE hypervolume number per
method family. With ≥5 seeds we can ask the harder, honest question a thesis
committee will ask: *is CMDP-PID's HV actually different from fixed-weight
PPO's, or is the gap within seed noise?* A single point estimate cannot answer
that; a resampling CI can.

Method (nonparametric bootstrap over the SEED dimension)
--------------------------------------------------------
1. Load the same per-seed points the campaign uses (heuristics + random from
   ``campaign-{sc}/points.jsonl``, CMDP from ``sweep-{sc}/points.jsonl``,
   fixed-weight PPO from ``ppo-fixed-{sc}/points.jsonl``).
2. Reuse the **fixed (ideal, nadir)** from ``campaign-{sc}/campaign_summary.json``
   — the very same normalisation the table used — so every resampled HV is
   comparable to the table and to each other (CLAUDE.md Lưu ý #8). We do NOT
   recompute the reference geometry per resample; that would make HVs
   incommensurable across samples.
3. For ``B`` iterations: draw the seed set **with replacement**; for each family
   rebuild its front exactly as the campaign does (seed-mean point per operating
   point — per budget ``d`` / per weight ``w``), then score HV against the fixed
   geometry. The spread of those HVs is the family's 95 % CI (2.5/97.5 pct).
4. **Paired** gap: in each iteration use the *same* seed draw for both families
   and record ΔHV = HV(A) − HV(B). The CI of Δ (and the fraction of draws with
   Δ>0) says whether one family is significantly ahead. Pairing on the seed draw
   removes the shared-seed noise, giving a tighter, fairer comparison than
   differencing two independent CIs.

Caveat (stated, not hidden): with only 5 seeds the bootstrap CI is wide and
approximate — it is a guard against over-claiming, not a precise interval. Report
it as such.

Run (offline — no gateway, no sb3; only numpy + the point files)::

    docker compose run --rm --no-deps \
        -v "$PWD/rl-agent/src:/app/src:ro" --entrypoint python rl-agent \
        src/eval/hv_bootstrap.py --scenario HIGH --n-boot 2000 \
        --compare cmdp-pid,ppo-fixed
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

try:
    from .points import PointRecord, load_points
    from . import pareto_metrics as pm
    from .run_campaign import metric_family
    from . import paths
except ImportError:  # pragma: no cover - direct-script fallback
    from eval.points import PointRecord, load_points
    from eval import pareto_metrics as pm
    from eval.run_campaign import metric_family
    from eval import paths


# ── Load the same points the campaign aggregates ─────────────────────────────

def load_all_points(results_dir: Path, scenario: str) -> list[PointRecord]:
    """Heuristics + random (campaign) + CMDP sweep + fixed-weight PPO front.

    This mirrors ``run_campaign.run_campaign`` exactly so the bootstrap operates
    on the identical point set the table was built from. NSGA-II is intentionally
    absent (it is excluded from the shared HV axes — Lưu ý #9).
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
            f"[hv-boot] no points for {scenario} under {results_dir}. Run the "
            f"campaign (and sweep/ppo-fixed) first."
        )
    return points


def load_fixed_geometry(results_dir: Path, scenario: str) -> tuple[np.ndarray, np.ndarray, dict]:
    """Read the campaign's fixed (ideal, nadir) + reported per-family HV.

    Reusing the campaign's geometry (rather than recomputing) is what keeps every
    bootstrap HV on the SAME axes as the published table (Lưu ý #8).
    """
    path = results_dir / f"campaign-{scenario}" / "campaign_summary.json"
    if not path.exists():
        raise SystemExit(
            f"[hv-boot] {path} missing — run run_campaign.py for {scenario} first "
            f"(the bootstrap reuses its fixed reference point)."
        )
    s = json.loads(path.read_text())
    ideal = np.asarray(s["ideal"], dtype=np.float64)
    nadir = np.asarray(s["nadir"], dtype=np.float64)
    reported_hv = {m: v.get("hypervolume") for m, v in s.get("methods", {}).items()}
    return ideal, nadir, reported_hv


# ── Reshape into family → operating-point → seed → (energy, sla) ─────────────

def index_points(points: list[PointRecord]) -> tuple[dict, list[int]]:
    """``{family: {method: {seed: (energy, sla)}}}`` + the sorted seed list.

    ``family`` collapses ``cmdp-d*`` → ``cmdp-pid`` and ``ppo-w*`` → ``ppo-fixed``
    (via ``run_campaign.metric_family``); ``method`` stays the operating point
    (each budget d / weight w / heuristic) so a family's front has one point per
    operating point — exactly the set HV is meant to score.
    """
    tree: dict[str, dict[str, dict[int, tuple[float, float]]]] = \
        defaultdict(lambda: defaultdict(dict))
    seeds: set[int] = set()
    for p in points:
        fam = metric_family(p.method)
        if fam == "nsga2":                       # never on the shared HV axes
            continue
        tree[fam][p.method][int(p.seed)] = (float(p.energy_kwh), float(p.sla_cost))
        seeds.add(int(p.seed))
    return tree, sorted(seeds)


def family_front(methods: dict, seed_sample) -> np.ndarray:
    """Front for one family under a seed resample: seed-mean point per operating
    point (missing seeds for an operating point are simply skipped)."""
    rows = []
    for _method, by_seed in methods.items():
        vals = [by_seed[s] for s in seed_sample if s in by_seed]
        if vals:
            rows.append(np.asarray(vals, dtype=np.float64).mean(axis=0))
    return np.asarray(rows, dtype=np.float64)


def family_hv(methods: dict, seed_sample, ideal, nadir) -> float:
    front = family_front(methods, seed_sample)
    if len(front) == 0:
        return 0.0
    return pm.hypervolume(front, ideal, nadir)


# ── Bootstrap ────────────────────────────────────────────────────────────────

def _pct_ci(samples: np.ndarray, lo: float = 2.5, hi: float = 97.5) -> tuple[float, float]:
    return float(np.percentile(samples, lo)), float(np.percentile(samples, hi))


def bootstrap_hv(
    tree: dict, seeds: list[int], ideal: np.ndarray, nadir: np.ndarray,
    n_boot: int, rng: np.random.Generator,
) -> dict:
    """Per-family HV point estimate (all seeds once) + bootstrap 95 % CI."""
    families = list(tree.keys())
    # Point estimate = every seed exactly once (reproduces the campaign HV).
    point = {f: family_hv(tree[f], seeds, ideal, nadir) for f in families}

    draws = {f: np.empty(n_boot) for f in families}
    n = len(seeds)
    seeds_arr = np.asarray(seeds)
    for b in range(n_boot):
        sample = list(seeds_arr[rng.integers(0, n, size=n)])
        for f in families:
            draws[f][b] = family_hv(tree[f], sample, ideal, nadir)

    out = {}
    for f in families:
        lo, hi = _pct_ci(draws[f])
        out[f] = {
            "hv_point": point[f],
            "hv_boot_mean": float(draws[f].mean()),
            "ci95_low": lo, "ci95_high": hi,
            "n_operating_points": len(tree[f]),
        }
    return out, draws


def paired_gap(
    tree: dict, seeds: list[int], ideal: np.ndarray, nadir: np.ndarray,
    a: str, b: str, n_boot: int, rng: np.random.Generator,
) -> dict:
    """Paired bootstrap of ΔHV = HV(a) − HV(b): same seed draw scores both."""
    if a not in tree or b not in tree:
        return {"error": f"family not found: need both '{a}' and '{b}' "
                         f"(have {sorted(tree)})"}
    n = len(seeds)
    seeds_arr = np.asarray(seeds)
    deltas = np.empty(n_boot)
    for i in range(n_boot):
        sample = list(seeds_arr[rng.integers(0, n, size=n)])
        deltas[i] = (family_hv(tree[a], sample, ideal, nadir)
                     - family_hv(tree[b], sample, ideal, nadir))
    lo, hi = _pct_ci(deltas)
    point = (family_hv(tree[a], seeds, ideal, nadir)
             - family_hv(tree[b], seeds, ideal, nadir))
    p_a_gt_b = float((deltas > 0).mean())
    return {
        "a": a, "b": b,
        "delta_point": float(point),
        "delta_ci95_low": lo, "delta_ci95_high": hi,
        "p_a_gt_b": p_a_gt_b,
        "significant_95": bool(lo > 0 or hi < 0),   # CI excludes 0
    }


# ── Reporting ────────────────────────────────────────────────────────────────

def print_report(scenario: str, per_family: dict, reported_hv: dict,
                 gaps: list[dict], n_boot: int) -> None:
    print("\n" + "=" * 74)
    print(f"  HV bootstrap 95% CI - {scenario}  (B={n_boot} resamples over seeds)")
    print("=" * 74)
    print(f"{'family':>14} {'HV (point)':>11} {'95% CI':>22} {'table HV':>10}")
    for fam in sorted(per_family):
        r = per_family[fam]
        rep = reported_hv.get(fam)
        rep_s = f"{rep:.4f}" if rep is not None else "  –"
        ci = f"[{r['ci95_low']:.4f}, {r['ci95_high']:.4f}]"
        print(f"{fam:>14} {r['hv_point']:>11.4f} {ci:>22} {rep_s:>10}")
    print("-" * 74)
    print("  'table HV' should match 'HV (point)' - confirms same geometry.")
    for g in gaps:
        if "error" in g:
            print(f"\n  gap: {g['error']}")
            continue
        verdict = ("SIGNIFICANT (CI excludes 0)" if g["significant_95"]
                   else "NOT significant (CI includes 0) - gap within seed noise")
        print(f"\n  dHV = HV[{g['a']}] - HV[{g['b']}]")
        print(f"    point estimate : {g['delta_point']:+.4f}")
        print(f"    95% CI         : [{g['delta_ci95_low']:+.4f}, "
              f"{g['delta_ci95_high']:+.4f}]")
        print(f"    P({g['a']} > {g['b']}) = {g['p_a_gt_b']:.2f}")
        print(f"    => {verdict}")
    print("=" * 74 + "\n")


def main() -> None:
    ap = argparse.ArgumentParser(description="Bootstrap HV CIs + paired family gap")
    ap.add_argument("--scenario", default="HIGH", help="LOW|HIGH|BURST slice the configured trace; with TRACE_PATTERN set, any generated scenario (incl. OVERLOAD, REPLAY) selects its own file")
    ap.add_argument("--results", default="/data/results")
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0, help="RNG seed for the bootstrap")
    ap.add_argument("--compare", default="cmdp-pid,ppo-fixed",
                    help="comma-separated family PAIRS to gap-test, pairs joined "
                         "by ';' e.g. 'cmdp-pid,ppo-fixed;cmdp-pid,random'")
    ap.add_argument("--out", default=None,
                    help="JSON output path (default: campaign-<sc>/hv_bootstrap.json)")
    args = ap.parse_args()

    # R5/R6: refuse to write WM-1 output onto the LEGACY results.
    paths.guard_results_root(args.results, what="the HV bootstrap")

    results_dir = Path(args.results)
    points = load_all_points(results_dir, args.scenario)
    ideal, nadir, reported_hv = load_fixed_geometry(results_dir, args.scenario)
    tree, seeds = index_points(points)

    if len(seeds) < 2:
        raise SystemExit(f"[hv-boot] need ≥2 seeds to bootstrap; found {seeds}")
    print(f"[hv-boot] {args.scenario}: {len(seeds)} seeds {seeds}, "
          f"families={sorted(tree)}")

    rng = np.random.default_rng(args.seed)
    per_family, _draws = bootstrap_hv(tree, seeds, ideal, nadir, args.n_boot, rng)

    gaps = []
    for pair in args.compare.split(";"):
        pair = pair.strip()
        if not pair:
            continue
        a, _, b = pair.partition(",")
        rng_g = np.random.default_rng(args.seed + 1)   # own stream, reproducible
        gaps.append(paired_gap(tree, seeds, ideal, nadir,
                               a.strip(), b.strip(), args.n_boot, rng_g))

    print_report(args.scenario, per_family, reported_hv, gaps, args.n_boot)

    out = Path(args.out) if args.out else \
        results_dir / f"campaign-{args.scenario}" / "hv_bootstrap.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "scenario": args.scenario, "n_boot": args.n_boot, "seeds": seeds,
        "ideal": ideal.tolist(), "nadir": nadir.tolist(),
        "per_family": per_family, "gaps": gaps,
    }, indent=2))
    print(f"[hv-boot] written -> {out}")


if __name__ == "__main__":
    main()
