"""Energy-oriented baselines vs. CMDP-PID — supplementary comparison for the defence.

The reviewer's point: FirstFit/BestFit/K8s never read the power model, so beating
them says little about an *energy-aware* opponent. This tool adds two that do:

  * ``pabfd``    — Power-Aware Best Fit Decreasing (Beloglazov et al., FGCS 2012),
                   extended with the host state machine (VmAllocationPolicyPowerAware).
  * ``k8s-most`` — Kubernetes ``NodeResourcesFit`` with ``MostAllocated`` scoring,
                   the documented bin-packing mode (VmAllocationPolicyK8sMostAllocated).

It writes to its OWN results root and never touches a campaign directory: adding
methods to a campaign moves its ideal/nadir box and changes every hypervolume
already reported (Lưu ý #26). HV computed here lives in a new box and is labelled so.

Two sub-commands:

  run     (needs the gateway) — run the policies on one (arm, scenario) for the given
          seeds; ``firstfit``/``bestfit`` are re-run as a CONTROL: they must reproduce
          the stored campaign numbers to the last digit, otherwise this run is not on
          the same workload/simulator as the thesis and nothing here is comparable.
  report  (offline) — merge with the stored campaign + sweep + ppo-fixed points of
          the source root, aggregate, run paired tests against every CMDP budget,
          and write ``report-{arm}-{scenario}.json`` + a markdown summary.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from eval.aggregate import aggregate_by_method, mean_ci95          # noqa: E402
from eval.points import PointRecord, load_points, save_points      # noqa: E402
from eval import pareto_metrics as pm                              # noqa: E402
from eval.run_campaign import family_points_dict, metric_family    # noqa: E402

NEW_POLICIES = ["pabfd", "k8s-most"]
CONTROL_POLICIES = ["firstfit", "bestfit"]


# ── run ─────────────────────────────────────────────────────────────────────

def cmd_run(args) -> None:
    from eval.run_campaign import collect_baseline_points   # lazy: needs py4j
    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    policies = [p for p in args.policies.split(",") if p.strip()]
    pts = collect_baseline_points(args.scenario, seeds, policies=policies)
    out = Path(args.output) / f"points-{args.scenario}.jsonl"
    save_points(pts, out, merge=True)
    print(f"[energy-baselines] {len(pts)} point(s) → {out}")


# ── report ──────────────────────────────────────────────────────────────────

def _load(path: Path) -> list[PointRecord]:
    return load_points(path) if path.exists() else []


def _by(points, method):
    return {p.seed: p for p in points if p.method == method}


def control_check(new_pts, ref_pts) -> dict:
    """Re-run heuristics must match the stored campaign exactly (same sim + workload)."""
    out = {}
    for m in CONTROL_POLICIES:
        a, b = _by(new_pts, m), _by(ref_pts, m)
        common = sorted(set(a) & set(b))
        if not common:
            out[m] = {"n": 0, "ok": None}
            continue
        rel = max(max(abs(a[s].energy_kwh - b[s].energy_kwh) / b[s].energy_kwh,
                      abs(a[s].sla_cost - b[s].sla_cost) / max(b[s].sla_cost, 1.0))
                  for s in common)
        out[m] = {"n": len(common), "max_rel_diff": rel, "ok": rel < 1e-9}
    return out


def paired(cmdp_by_seed: dict, base_by_seed: dict) -> dict:
    """Paired Δ = cmdp − baseline per seed (negative = cmdp better), both axes."""
    seeds = sorted(set(cmdp_by_seed) & set(base_by_seed))
    dE = [cmdp_by_seed[s].energy_kwh - base_by_seed[s].energy_kwh for s in seeds]
    dC = [cmdp_by_seed[s].sla_cost - base_by_seed[s].sla_cost for s in seeds]
    e, c = mean_ci95(dE), mean_ci95(dC)

    def sig(m):            # −1 cmdp better, +1 baseline better, 0 CI contains 0
        if m.ci95 != m.ci95:
            return 0
        if m.mean + m.ci95 < 0:
            return -1
        if m.mean - m.ci95 > 0:
            return 1
        return 0

    se, sc = sig(e), sig(c)
    if se == -1 and sc == -1:
        verdict = "cmdp DOMINATES"
    elif se == 1 and sc == 1:
        verdict = "baseline DOMINATES"
    elif {se, sc} == {-1, 1}:
        verdict = "trade-off"
    elif se == 0 and sc == 0:
        verdict = "no significant difference"
    else:
        verdict = "one-sided"            # one axis significant, other within noise
    both_wins = sum(1 for x, y in zip(dE, dC) if x < 0 and y < 0)
    both_loss = sum(1 for x, y in zip(dE, dC) if x > 0 and y > 0)
    return {"n": len(seeds), "dE_mean": e.mean, "dE_ci95": e.ci95,
            "dC_mean": c.mean, "dC_ci95": c.ci95, "sig_E": se, "sig_C": sc,
            "verdict": verdict, "seeds_cmdp_wins_both": both_wins,
            "seeds_baseline_wins_both": both_loss}


def cmd_report(args) -> None:
    src = Path(args.source)
    sc = args.scenario
    ref = _load(src / f"campaign-{sc}" / "points.jsonl")
    sweep = _load(src / f"sweep-{sc}" / "points.jsonl")
    ppo = _load(src / f"ppo-fixed-{sc}" / "points.jsonl")
    new = _load(Path(args.output) / f"points-{sc}.jsonl")
    if not new:
        raise SystemExit(f"no new points for {sc} under {args.output}")

    ctrl = control_check(new, ref)
    seeds_ref = sorted({p.seed for p in ref})
    include = [m for m in args.include.split(",") if m.strip()]
    added = [p for p in new if p.method in include and p.seed in seeds_ref]
    allp = ref + sweep + ppo + added

    agg = aggregate_by_method(allp, scenario=sc)
    fam = family_points_dict(allp, scenario=sc)
    metrics = pm.evaluate_methods(fam)

    budgets = sorted({p.method for p in sweep}, key=lambda m: float(m.split("-d")[1]))
    comps = {}
    for b in include + ["firstfit", "bestfit", "k8s"]:
        base = _by(allp, b)
        comps[b] = {m: paired(_by(sweep, m), base) for m in budgets}

    rep = {
        "arm": args.arm, "scenario": sc, "source_root": str(src),
        "seeds": seeds_ref, "control": ctrl,
        "aggregate": {m: {"n": a["n"], "energy_mean": a["energy"].mean,
                          "energy_ci95": a["energy"].ci95, "sla_mean": a["sla"].mean,
                          "sla_ci95": a["sla"].ci95, "dropped": a.get("dropped")}
                      for m, a in agg.items()},
        "hv_new_box": {k: v.get("hypervolume") for k, v in metrics["methods"].items()},
        "igd_new_box": {k: v.get("igd_plus") for k, v in metrics["methods"].items()},
        "box": {"ideal": list(map(float, metrics["ideal"])),
                "nadir": list(map(float, metrics["nadir"]))},
        "paired_cmdp_minus_baseline": comps,
    }
    out = Path(args.output) / f"report-{args.arm}-{sc}{args.tag}.json"
    out.write_text(json.dumps(rep, indent=2, default=float), encoding="utf-8")

    print(f"\n### {args.arm}/{sc}  (seeds n={len(seeds_ref)}, source {src})")
    for m, r in ctrl.items():
        print(f"  control {m}: n={r['n']} ok={r['ok']} "
              f"max_rel_diff={r.get('max_rel_diff', float('nan')):.2e}")
    for m in sorted(agg, key=lambda k: agg[k]["energy"].mean):
        a = agg[m]
        print(f"  {m:>14}  E={a['energy'].mean:10.2f} ± {a['energy'].ci95:7.2f}  "
              f"C={a['sla'].mean:.3e} ± {a['sla'].ci95:.2e}  drop={a.get('dropped')}")
    print("  HV (new box): " + ", ".join(
        f"{k}={v:.4f}" for k, v in sorted(rep["hv_new_box"].items(),
                                          key=lambda kv: -(kv[1] or 0))))
    for b in include:
        for m, r in comps[b].items():
            print(f"  {m:>13} − {b:<8} ΔE={r['dE_mean']:+9.2f} ± {r['dE_ci95']:7.2f}  "
                  f"ΔC={r['dC_mean']:+.3e} ± {r['dC_ci95']:.2e}  → {r['verdict']}"
                  f"  (cmdp wins both {r['seeds_cmdp_wins_both']}/{r['n']},"
                  f" loses both {r['seeds_baseline_wins_both']}/{r['n']})")
    print(f"  → {out}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--scenario", required=True)
    r.add_argument("--seeds", required=True)
    r.add_argument("--policies", default=",".join(NEW_POLICIES + CONTROL_POLICIES))
    r.add_argument("--output", required=True)
    p = sub.add_parser("report")
    p.add_argument("--arm", required=True)
    p.add_argument("--scenario", required=True)
    p.add_argument("--source", required=True, help="campaign root, e.g. /data/results/wm1-v3/homo")
    p.add_argument("--output", required=True)
    p.add_argument("--include", default=",".join(NEW_POLICIES),
                   help="new policies to add to the comparison (the HV box depends on this)")
    p.add_argument("--tag", default="", help="suffix for the report file name")
    args = ap.parse_args()
    cmd_run(args) if args.cmd == "run" else cmd_report(args)


if __name__ == "__main__":
    main()
