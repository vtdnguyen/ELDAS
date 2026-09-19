"""G2.6 — Campaign: every method × scenario × seeds → table (mean ± 95 % CI),
hypervolume / IGD+, and the energy↔SLA Pareto figure.

Methods scored on ONE common objective pair (both minimised, same units):
``energy_kwh`` and ``sla_cost`` (= C_SLA). Sources:

  * 5 classical heuristics + fixed-weight PPO — one episode per (scenario, seed);
  * CMDP-PID — the G2.5 budget sweep (``sweep-{scenario}/points.jsonl``);
  * NSGA-II — the G2.3 static reference front (``nsga2-{scenario}/reference_front.json``).

Scientific guardrails:
  * **≥5 seeds** before any number is reportable; a smaller run is labelled a
    smoke check and its CI is withheld (Lưu ý #10).
  * **One fixed reference point** for hypervolume across every method (Lưu ý #8),
    handled inside ``pareto_metrics``.
  * **NSGA-II is a static idealisation** with full trace foreknowledge — it is a
    reference/upper bound, NOT a head-to-head competitor (Lưu ý #9). The table
    and the figure both mark it as such.

Offline (no gateway) — aggregate/plot from already-collected points::

    docker compose run --rm --no-deps rl-agent \
        python src/eval/run_campaign.py --scenario LOW --from-points

Full run (needs the gateway)::

    docker compose run --rm rl-agent python src/eval/run_campaign.py \
        --scenario LOW --seeds 42,43,44,45,46 --run-baselines
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

try:
    from .aggregate import aggregate_by_method
    from .points import PointRecord, load_points, method_points_dict, save_points
    from . import pareto_metrics as pm
    from . import paths
except ImportError:  # pragma: no cover - direct-script fallback
    from eval.aggregate import aggregate_by_method
    from eval.points import PointRecord, load_points, method_points_dict, save_points
    from eval import pareto_metrics as pm
    from eval import paths


HEURISTICS = ["roundrobin", "random", "k8s", "firstfit", "bestfit"]
NSGA2_METHOD = "nsga2"


# ── Collectors ──────────────────────────────────────────────────────────────

def collect_baseline_points(
    scenario: str, seeds: list[int], policies: list[str] | None = None,
    output_dir: Path | None = None,
) -> list[PointRecord]:
    """Run each heuristic once per seed and return points (needs the gateway)."""
    from baseline_eval import evaluate_baseline           # lazy: needs py4j
    from environment import CloudSimEnv

    policies = policies or HEURISTICS
    points: list[PointRecord] = []
    env = CloudSimEnv(scenario=scenario, seed=seeds[0])
    try:
        for seed in seeds:
            for policy in policies:
                r = evaluate_baseline(env, policy, scenario=scenario, seed=seed)
                points.append(PointRecord(
                    method=policy, scenario=scenario, seed=seed,
                    energy_kwh=float(r["total_energy_kwh"]),
                    sla_cost=float(r["total_sla_cost"]),
                    extra={"steps": r["steps"],
                           "dropped_tasks": int(r.get("dropped_tasks", 0))},
                ))
                print(f"[campaign] {policy:>11} seed={seed}: "
                      f"E={r['total_energy_kwh']:.2f} kWh  "
                      f"C_SLA={r['total_sla_cost']:.4e}")
                if output_dir is not None:
                    env.export_metrics(str(output_dir / f"baseline-{scenario}" / policy))
    finally:
        env.close()
    return points


def load_nsga2_front(results_dir: Path, scenario: str) -> tuple[np.ndarray, dict] | tuple[None, None]:
    """Load the G2.3 static reference front (+ its provenance), if it exists."""
    path = results_dir / f"nsga2-{scenario}" / "reference_front.json"
    if not path.exists():
        return None, None
    payload = json.loads(path.read_text())
    if payload.get("objectives") != ["energy_kwh", "sla_cost"]:
        raise ValueError(f"unexpected objectives in {path}: {payload.get('objectives')}")
    return np.array(payload["front"], dtype=np.float64), payload


def nsga2_commensurable(payload: dict, des_points: list[PointRecord]) -> tuple[bool, str]:
    """Decide whether the NSGA-II front may share HV/IGD+ axes with DES methods.

    **Default answer is NO, and that is the scientifically correct default.**
    Two independent reasons the static front is not commensurable with the
    DES-measured points:

    1. **Different evaluator.** NSGA-II objectives come from the static
       surrogate (``eval.static_model``), not from CloudSim. Its energy is a
       different *measurement* of energy, not a better score on the same one.
    2. **Different instance.** The reference front is solved on a ``--max-tasks``
       subset, while the online methods run the full scenario. Fewer tasks ⇒
       trivially less energy.

    Mixing them under one fixed nadir produces a meaningless HV≈1 / IGD+≈0 for
    NSGA-II — it would look like the static solver "wins", which is exactly the
    false head-to-head claim Lưu ý #9 forbids. To make the front genuinely
    comparable, its *assignments* must be replayed through the DES so CloudSim
    measures them (see C7 in Tracking_detail.md).

    Returns ``(commensurable, reason)``.
    """
    n_front = int(payload.get("num_tasks", -1))
    des_tasks = {int((p.extra or {}).get("steps", -1)) for p in des_points
                 if (p.extra or {}).get("steps") is not None}
    des_tasks.discard(-1)

    if des_tasks and n_front > 0 and n_front not in des_tasks:
        return False, (
            f"instance mismatch: reference front solved on {n_front} task(s), "
            f"online methods ran {sorted(des_tasks)} task(s)"
        )
    return False, (
        "static surrogate evaluator ≠ CloudSim DES evaluator "
        "(objectives are not the same measurement)"
    )


def load_sweep_points(results_dir: Path, scenario: str) -> list[PointRecord]:
    """Load the G2.5 CMDP budget-sweep points, if present."""
    path = results_dir / f"sweep-{scenario}" / "points.jsonl"
    return load_points(path) if path.exists() else []


def load_ppo_min_points(results_dir: Path, scenario: str) -> list[PointRecord]:
    """Load the fixed-weight PPO (ppo-min) point from ``baseline_results.json``.

    ``train_min`` records ``total_sla_cost`` (from ``getSlaCost`` — the same
    C_SLA axis as everything else, C6). Emit it as one point so the campaign can
    place the Phase-1 fixed-weight baseline next to the CMDP-PID front. It is a
    single trained reference (not a ≥5-seed method), so it shows as n=1 and the
    table's own "< min_seeds" guard flags it — deliberately, not silently.
    """
    path = results_dir / f"baseline-{scenario}" / "baseline_results.json"
    if not path.exists():
        return []
    try:
        data = json.loads(path.read_text())
    except Exception:
        return []
    row = data.get("ppo-min")
    if not row or "total_sla_cost" not in row:
        return []
    return [PointRecord(
        method="ppo-min", scenario=scenario, seed=int(row.get("seed", 42)),
        energy_kwh=float(row["total_energy_kwh"]),
        sla_cost=float(row["total_sla_cost"]),
        extra={"weights": row.get("weights")},
    )]


def load_ppo_fixed_points(results_dir: Path, scenario: str) -> list[PointRecord]:
    """Load the fixed-weight PPO *front* (``eval/ppo_fixed_sweep.py``), if present.

    ``ppo-fixed-{scenario}/points.jsonl`` holds one point per (weight, seed) with
    the same schema as the CMDP sweep. Each weight is a ``ppo-w{w}`` method, so
    ``aggregate_by_method`` gives it mean ± CI over seeds, and ``metric_family``
    collapses ``ppo-w*`` into ONE ``ppo-fixed`` family — hypervolume/IGD+ then
    score the fixed-weight front against the CMDP-PID front (front-to-front, the
    fair comparison). When this file exists it SUPERSEDES the single-point
    ``ppo-min`` reference (which stays as the fallback for older runs).
    """
    path = results_dir / f"ppo-fixed-{scenario}" / "points.jsonl"
    return load_points(path) if path.exists() else []


# ── Grouping for the metrics ────────────────────────────────────────────────

def metric_family(method: str) -> str:
    """Collapse the per-budget CMDP rows into ONE method for HV/IGD+.

    The CMDP-PID *method* produces a whole front (one policy per budget d), so
    its achievable set — not each budget in isolation — is what should be scored
    against the heuristics' single operating points. Symmetrically, the
    fixed-weight PPO produces a front by sweeping the scalarisation weight, so its
    ``ppo-w*`` operating points collapse into one ``ppo-fixed`` family — the
    front-to-front comparison against CMDP-PID (manual weight tuning vs.
    principled budget tuning).
    """
    m = method.lower()
    if m.startswith("cmdp"):
        return "cmdp-pid"
    if m.startswith("ppo-w"):
        return "ppo-fixed"
    return method


def family_points_dict(
    points: list[PointRecord], scenario: str | None = None,
    nsga2_front: np.ndarray | None = None,
    average_seeds: bool = True,
) -> dict[str, np.ndarray]:
    """Group points by metric family (cmdp-d* → cmdp-pid) for HV/IGD+.

    ``average_seeds=True`` (the default) first collapses each *method* to its
    **seed-mean point**, then groups methods into families. This matters for
    correctness: hypervolume scores a trade-off *front*, so a method must not
    gain HV merely by being noisy across seeds. A stochastic policy (``random``)
    otherwise scatters 5 points and covers more of the objective box than a
    deterministic one that collapses to a single point — rewarding variance, not
    scheduling quality. Seed spread belongs in the CI column, not in HV.

    After averaging, a single-operating-point method (each heuristic, fixed-weight
    PPO) contributes exactly one point, while CMDP-PID contributes one point per
    budget ``d`` — i.e. its actual front, which is what HV is meant to measure.
    """
    by_method: dict[str, list[tuple[float, float]]] = {}
    for p in points:
        if scenario is not None and p.scenario != scenario:
            continue
        by_method.setdefault(p.method, []).append(p.objectives())

    groups: dict[str, list[tuple[float, float]]] = {}
    for method, objs in by_method.items():
        arr = np.array(objs, dtype=np.float64)
        rows = [tuple(arr.mean(axis=0))] if average_seeds else [tuple(o) for o in arr]
        groups.setdefault(metric_family(method), []).extend(rows)

    out = {m: np.array(v, dtype=np.float64) for m, v in groups.items()}
    if nsga2_front is not None and len(nsga2_front) > 0:
        out[NSGA2_METHOD] = np.asarray(nsga2_front, dtype=np.float64)
    return out


# ── Table ───────────────────────────────────────────────────────────────────

def _fmt_ci(mean: float, ci: float, fmt: str) -> str:
    if ci != ci:                       # NaN ⇒ single seed, no CI
        return f"{mean:{fmt}} (n=1)"
    return f"{mean:{fmt}} ± {ci:{fmt}}"


def build_table(agg: dict, metrics: dict, scenario: str, min_seeds: int = 5,
                nsga2_note: str | None = None) -> str:
    """Markdown summary table — one row per method (per-budget CMDP rows kept)."""
    lines = [
        f"### Campaign summary — {scenario}",
        "",
        "| Method | seeds | Energy kWh (mean ± 95% CI) | C_SLA (mean ± 95% CI) | dropped | HV ↑ | IGD+ ↓ |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for name in sorted(agg):
        a = agg[name]
        m = metrics["methods"].get(metric_family(name), {})
        hv = m.get("hypervolume")
        igd = m.get("igd_plus")
        hv_cell = f"{hv:.4f}" if hv is not None else "–"
        igd_cell = f"{igd:.4f}" if igd is not None else "–"
        energy_cell = _fmt_ci(a["energy"].mean, a["energy"].ci95, ".2f")
        sla_cell = _fmt_ci(a["sla"].mean, a["sla"].ci95, ".4g")
        # W6.1/W3: a method that meets its SLA budget by leaving tasks unplaced has
        # changed the problem rather than solved it, and energy computed over fewer
        # tasks is not comparable with energy that placed them all (R10). Without this
        # column that failure reads as a clean win in every other cell of the row.
        drops = a.get("dropped")
        drop_cell = "–" if drops is None else f"{drops:.0f}"
        lines.append(
            f"| {name} | {a['n']} | {energy_cell} | {sla_cell} | {drop_cell} "
            f"| {hv_cell} | {igd_cell} |"
        )
    if NSGA2_METHOD in metrics["methods"]:
        m = metrics["methods"][NSGA2_METHOD]
        lines.append(
            f"| {NSGA2_METHOD} (static ref) | – | – | – "
            f"| {m['hypervolume']:.4f} | {m['igd_plus']:.4f} |"
        )
    lines += [
        "",
        f"- Objectives: `energy_kwh` (min), `sla_cost` = C_SLA (min). "
        f"Fixed reference point (nadir) = {np.round(metrics['nadir'], 4).tolist()}, "
        f"ideal = {np.round(metrics['ideal'], 4).tolist()} — shared by every method "
        f"so hypervolumes are comparable (Lưu ý #8).",
        "- HV higher = better; IGD+ lower = better. Both Pareto-compliant.",
    ]
    if nsga2_note:
        lines.append(
            f"- **NSGA-II EXCLUDED from this table's HV/IGD+** — {nsga2_note}. "
            f"The static front is reported separately in `nsga2-{scenario}/"
            f"reference_front.json`. Scoring it on these axes would fabricate a "
            f"head-to-head win for a solver that measures energy with a different "
            f"model on a different instance (Lưu ý #9)."
        )
    else:
        lines.append(
            "- **NSGA-II is a STATIC idealisation** (full trace foreknowledge, no "
            "temporal dynamics): a reference/upper bound, NOT a fair head-to-head "
            "opponent for the online schedulers (Lưu ý #9)."
        )
    seed_counts = {a["n"] for a in agg.values()}
    if seed_counts and min(seed_counts) < min_seeds:
        lines.append(
            f"- ⚠ **Not reportable**: some methods have < {min_seeds} seeds "
            f"(Lưu ý #10). This is a smoke/wiring run, not a thesis result."
        )
    return "\n".join(lines)


# ── Driver ──────────────────────────────────────────────────────────────────

def run_campaign(
    scenario: str,
    seeds: list[int],
    results_dir: Path,
    *,
    run_baselines: bool,
    min_seeds: int = 5,
    force_nsga2: bool = False,
) -> dict:
    campaign_dir = results_dir / f"campaign-{scenario}"
    campaign_dir.mkdir(parents=True, exist_ok=True)
    points_file = campaign_dir / "points.jsonl"

    points: list[PointRecord] = []
    if run_baselines:
        points += collect_baseline_points(scenario, seeds, output_dir=results_dir)
        save_points(points, points_file)
    elif points_file.exists():
        points = load_points(points_file)
        print(f"[campaign] loaded {len(points)} baseline point(s) from {points_file}")

    sweep = load_sweep_points(results_dir, scenario)
    if sweep:
        print(f"[campaign] + {len(sweep)} CMDP sweep point(s)")
    points += sweep

    ppo_fixed = load_ppo_fixed_points(results_dir, scenario)
    if ppo_fixed:
        n_w = len({p.method for p in ppo_fixed})
        print(f"[campaign] + fixed-weight PPO FRONT: {len(ppo_fixed)} point(s) "
              f"across {n_w} weight(s) → 'ppo-fixed' family (front-to-front)")
        points += ppo_fixed
    else:
        ppo_min = load_ppo_min_points(results_dir, scenario)
        if ppo_min:
            print(f"[campaign] + fixed-weight PPO (ppo-min) reference point (n=1 "
                  f"fallback; run eval/ppo_fixed_sweep.py for a ≥5-seed front)")
        points += ppo_min

    nsga2_front, nsga2_payload = load_nsga2_front(results_dir, scenario)
    nsga2_note = None
    if nsga2_front is not None:
        print(f"[campaign] + NSGA-II reference front ({len(nsga2_front)} points)")
        ok, reason = nsga2_commensurable(nsga2_payload, points)
        if not ok and not force_nsga2:
            nsga2_note = reason
            print(f"[campaign] ⚠ NSGA-II EXCLUDED from shared HV/IGD+ and figure: "
                  f"{reason}.\n"
                  f"[campaign]   It stays a separate static reference (Lưu ý #9); "
                  f"its assignments must be replayed through the DES to be "
                  f"comparable. Override with --force-nsga2-metrics (not advised).")
            nsga2_front = None

    if not points:
        raise SystemExit(
            f"[campaign] no points for {scenario}. Run with --run-baselines, or "
            f"produce {points_file} / sweep-{scenario}/points.jsonl first."
        )

    agg = aggregate_by_method(points, scenario=scenario)
    fam_points = family_points_dict(points, scenario=scenario, nsga2_front=nsga2_front)
    metrics = pm.evaluate_methods(fam_points)

    table = build_table(agg, metrics, scenario, min_seeds=min_seeds,
                        nsga2_note=nsga2_note)
    print("\n" + table + "\n")

    (campaign_dir / "table.md").write_text(table, encoding="utf-8")
    with open(campaign_dir / "campaign_summary.json", "w") as f:
        json.dump({
            "scenario": scenario, "seeds": seeds,
            "objectives": metrics["objectives"], "sense": metrics["sense"],
            "ideal": metrics["ideal"], "nadir": metrics["nadir"],
            "nsga2_excluded_reason": nsga2_note,
            "methods": metrics["methods"],
            "aggregate": {
                m: {"energy_mean": a["energy"].mean, "energy_ci95": a["energy"].ci95,
                    "sla_mean": a["sla"].mean, "sla_ci95": a["sla"].ci95,
                    "n": a["n"], "seeds": a["seeds"]}
                for m, a in agg.items()
            },
        }, f, indent=2)

    fig_path = None
    if agg:
        from eval.pareto_plot import plot_pareto      # lazy: needs matplotlib
        fig_path = plot_pareto(agg, campaign_dir / f"pareto-{scenario}.png",
                               scenario, nsga2_front=nsga2_front)
        print(f"[campaign] figure  → {fig_path}")
    print(f"[campaign] table   → {campaign_dir / 'table.md'}")
    print(f"[campaign] summary → {campaign_dir / 'campaign_summary.json'}")

    return {"aggregate": agg, "metrics": metrics, "table": table,
            "figure": str(fig_path) if fig_path else None}


def main() -> None:
    ap = argparse.ArgumentParser(description="G2.6 — full campaign table + Pareto figure")
    ap.add_argument("--scenario", default="LOW", help="LOW|HIGH|BURST slice the configured trace; with TRACE_PATTERN set, any generated scenario (incl. OVERLOAD, REPLAY) selects its own file")
    ap.add_argument("--seeds", default="42,43,44,45,46",
                    help="comma-separated seeds (≥5 for reportable CI)")
    ap.add_argument("--results", default="/data/results")
    ap.add_argument("--run-baselines", action="store_true",
                    help="execute heuristic episodes (needs the gateway)")
    ap.add_argument("--from-points", action="store_true",
                    help="offline: aggregate/plot from existing point files only")
    ap.add_argument("--force-nsga2-metrics", action="store_true",
                    help="score the static NSGA-II front on the shared HV/IGD+ axes "
                         "anyway (NOT advised: different evaluator/instance — see "
                         "nsga2_commensurable)")
    args = ap.parse_args()

    # R5/R6: refuse to write WM-1 output onto the LEGACY results.
    paths.guard_results_root(args.results, what="the campaign")

    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    run_campaign(args.scenario, seeds, Path(args.results),
                 run_baselines=args.run_baselines and not args.from_points,
                 force_nsga2=args.force_nsga2_metrics)


if __name__ == "__main__":
    main()
