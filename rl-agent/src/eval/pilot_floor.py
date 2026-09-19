"""W6.1 — Pilot: find the *feasible SLA floor* so the budget grid can be placed.

The budget ``d`` constrains ``J`` = the episode mean of the **normalised** cost, not
raw ``C_SLA``.  That scale is set by the running normaliser, so it moves whenever the
cost definition moves — and it moved twice: the absolute deadline floor (W1.5) and the
charge for dropped tasks (W3).  The old grid ``0.02…0.10`` was calibrated before both,
so it now sits at an arbitrary place on a different axis.  Sweeping it would very
likely land entirely inside the slack region, where every budget converges to the same
unconstrained policy and the "Pareto front" is one point drawn five times (§14.3).

So measure the two ends first:

* **loose** ``d`` (default 9.9, far above any achievable J) — the constraint never
  binds, λ stays at 0, and the run converges to the unconstrained optimum.  Its tail
  ``J`` is ``J_free``: the *most* SLA cost the sweep will ever see.
* **tight** ``d = 0`` — λ grows until it dominates, pushing J as low as the policy can
  drive it.  Its tail ``J`` is ``J_floor``: the *least*.

Any budget outside ``[J_floor, J_free]`` is wasted compute: above it the constraint is
slack, below it is infeasible and λ diverges.  The grid this prints spans the interior.

Also reported, because W6.1 asks for them by name: episode length, number of dual
updates actually performed, final ``value_loss``/``approx_kl``, and the dropped-task
count next to the best heuristic's — an agent that meets its budget by leaving tasks
unplaced has not solved the problem, it has changed it (W3).

Usage
-----
  python src/eval/pilot_floor.py --scenario LOW --output /data/results/wm1/homo
  python src/eval/pilot_floor.py --scenario HIGH --episodes 40 --loose-budget 9.9
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

try:
    from . import paths
except ImportError:                                      # pragma: no cover
    from eval import paths                               # type: ignore[no-redef]

#: The grid is deliberately NOT symmetric inside [J_floor, J_free].
#:
#: The pilot trains for ~40 episodes and the campaign for ~200, so the interval it
#: measures is a *shrunken* version of the campaign's: more training pushes the
#: unconstrained policy to pack harder (J_free rises) and gives the constrained one
#: more room to comply (J_floor falls).  The two errors point in opposite directions,
#: which is what makes the pilot usable at all — but they are not symmetric in cost:
#:
#:   * a budget that ends up ABOVE the campaign's J_free is slack, λ stays 0, and the
#:     run reproduces the unconstrained policy — a duplicate point, wasted compute;
#:   * a budget BELOW the campaign's J_floor is infeasible, λ grows, and the run
#:     converges to the most SLA-conservative policy — the extreme point of the front,
#:     which is a point worth having.
#:
#: So overshoot downward and stay clear of the top.
#: Budgets as fractions of the unconstrained operating point: ``d = alpha * J_free``.
#:
#: The obvious alternative — spread the grid over the measured ``[J_floor, J_free]`` —
#: was tried first and does not survive contact with the pilot.  On 9 of 10 arm x
#: scenario combinations a 40-episode pilot returns the two endpoints within a fraction
#: of a percent of each other (and on BURST it returns them inverted), so the interval
#: is noise and the grid derived from it collapses onto five near-identical budgets:
#: one point drawn five times, the §14.3 failure the pilot exists to prevent.
#:
#: J_free does not have that problem.  It comes from a run with lambda pinned at 0, so
#: it is a plain unconstrained-PPO measurement, taken through exactly the same
#: normaliser path the campaign uses.  Every alpha < 1 is then a budget below where the
#: unconstrained policy sits, i.e. binding by construction, and the spacing is a design
#: choice rather than an estimate of something the pilot cannot see.
#:
#: Weighted toward the tight end: that is where lambda is large and the trade-off
#: actually bends. Near alpha=1 the constraint barely bites and points bunch up.
#:
#: FOUR points, not five (§20.7). The five-point grid was measured on the corrected
#: reward and its two interior points came back inside the seed noise: 0.0903 and
#: 0.1237 differ by 4.7% in J but their energies (874.7 +- 95.1 and 879.1 +- 65.7)
#: overlap completely. Points that cannot be told apart do not make the front denser,
#: they make it look non-monotone and cost a fifth of the night. Spending the same
#: compute on four budgets spread wider buys resolution instead of repetition.
DEFAULT_BUDGET_FRACTIONS = (0.45, 0.65, 0.85, 0.97)


def _fractions_from_env(
        default: tuple[float, ...] = DEFAULT_BUDGET_FRACTIONS) -> tuple[float, ...]:
    """``ELDAS_BUDGET_FRACTIONS="0.45,0.65,0.85,0.97"`` overrides the grid shape.

    Env rather than a flag because the grid is read back by ``--print-grid`` from a
    driver script, and a flag would have to be threaded identically through every
    caller. A malformed value raises: silently falling back to the default would run a
    whole night on a grid nobody chose.
    """
    raw = os.environ.get("ELDAS_BUDGET_FRACTIONS", "").strip()
    if not raw:
        return default
    fr = tuple(sorted(float(x) for x in raw.replace(" ", ",").split(",") if x))
    if not fr or fr[0] <= 0 or fr[-1] >= 1.0:
        raise ValueError(
            f"ELDAS_BUDGET_FRACTIONS={raw!r}: every fraction must be in (0, 1). "
            "A fraction >= 1 puts the budget at or above the unconstrained operating "
            "point, where the constraint is slack and the run is a duplicate.")
    return fr


BUDGET_FRACTIONS = _fractions_from_env()


def suggest_grid(j_free: float, n: int | None = None,
                 fractions: tuple[float, ...] | None = None) -> list[float]:
    """``n`` binding budgets, as fractions of the unconstrained cost ``j_free``."""
    if not j_free or j_free <= 0 or j_free != j_free:      # None / 0 / NaN
        return []
    fractions = fractions if fractions is not None else BUDGET_FRACTIONS
    n = n if n else len(fractions)
    fr = list(fractions[:n])
    if len(fr) < n:                                        # asked for more than we name
        fr = [0.5 + 0.47 * i / (n - 1) for i in range(n)]
    return [round(f * j_free, 4) for f in fr]


def _run(scenario: str, seed: int, d: float, timesteps: int, out_dir: Path,
         label: str, k_p: float, k_i: float,
         cost_freeze_after: int | None = None) -> dict:
    # Imported here, not at module scope: `--print-grid` and the grid tests must work
    # without the RL stack installed, and this is the only path that needs it.
    from train_cmdp import evaluate_cmdp, train_cmdp

    model_path = Path("/data/models") / f"pilot-{scenario}-{label}.zip"
    _, pid, conv = train_cmdp(
        scenario=scenario, seed=seed, total_timesteps=timesteps, budget_d=d,
        k_p=k_p, k_i=k_i, cost_freeze_after=cost_freeze_after, out_path=model_path,
    )
    # evaluate_cmdp merges its row into <parent>/baseline_results.json, so keep the
    # pilot one directory deep: its rows must not land in baseline-<SC>, where the
    # campaign would pick them up as if they were tuned results.
    row = evaluate_cmdp(model_path, scenario, seed, d, pid, out_dir / label,
                        cost_freeze_after=cost_freeze_after)
    return {
        "label": label, "budget_d": d,
        "J_tail_mean": conv.get("J_tail_mean"),
        "lambda_final": conv.get("lambda_final"),
        "lambda_rel_std": conv.get("lambda_rel_std"),
        "n_dual_updates": conv.get("n_updates"),
        "value_loss_final": conv.get("value_loss_final"),
        "approx_kl_final": conv.get("approx_kl_final"),
        "energy_kwh": row.get("total_energy_kwh"),
        "sla_cost": row.get("total_sla_cost"),
        "dropped_tasks": row.get("dropped_tasks"),
        "eval_steps": row.get("steps"),
    }


def heuristic_reference(scenario: str, seed: int) -> dict:
    """Each heuristic's energy, C_SLA, drop count **and J**, on one shared normaliser.

    The J column is the one that decides whether a budget sweep can work at all, and
    it is not obtainable from a training run.

    ``d`` constrains ``J`` = the episode mean of ``C_SLA / std(C_SLA)``, where the std
    is estimated from the run's own cost stream.  That self-normalisation is what keeps
    the Lagrangian stable (Lưu ý #1), but it also means a policy that halves its raw
    cost partly shrinks its own denominator, so J moves less than the physical quantity
    does.  How much less is an empirical property of the workload, and if the answer is
    "not at all", then no grid of ``d`` can separate policies and the whole
    budget-sweep construction fails on that scenario — quietly, by producing five
    near-identical points.

    Five fixed policies spanning the trade-off, pushed through **one** wrapper so they
    share a normaliser, answer it directly and cost five episodes.
    """
    from baseline_eval import BASELINE_POLICIES
    from cmdp import CMDPRewardWrapper
    from environment import CloudSimEnv

    base = CloudSimEnv(scenario=scenario, seed=seed, normalize_reward=False)

    rows = {}
    try:
        for policy in BASELINE_POLICIES:
            # A FRESH wrapper per policy. Sharing one across all five looks tidier and
            # is wrong: the normaliser accumulates, so the first policy is divided by a
            # std estimated from almost no data and the last by a well-estimated one.
            # Measured, that artifact alone reordered the policies — roundrobin came
            # out with the highest J on LOW despite having the second-LOWEST raw cost.
            # pid=None + set_lambda(0): nothing learns from the shaped reward here, but
            # λ must be defined, and 0 keeps R_eff purely energy.
            env = CMDPRewardWrapper(base, pid=None, normalize=True)
            env.set_lambda(0.0)
            env.reset(seed=seed, options={"scenario": scenario})
            steps = 0
            while True:
                action = int(base._ep.selectBaselineAction(policy))
                _obs, _r, term, trunc, info = env.step(action)
                steps += 1
                if term or trunc:
                    break
            ep = info.get("episode", {})
            rows[policy] = {"energy_kwh": ep.get("total_energy_kwh", 0.0),
                            "sla_cost": ep.get("total_sla_cost", 0.0),
                            "dropped_tasks": int(ep.get("dropped_tasks", 0)),
                            "J": info.get("episode_cost"),
                            "steps": steps}
    finally:
        base.close()
    return rows


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--print-grid", metavar="PILOT_JSON", default=None,
                   help="read a finished pilot_floor.json and print the budget grid "
                        "derived from its J_floor/J_free with the CURRENT grid policy, "
                        "then exit. Keeps one definition of the grid: changing the "
                        "policy does not mean re-running a pilot to pick it up.")
    p.add_argument("--scenario", default=None)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--episodes", type=int, default=40,
                   help="episodes per endpoint run; timesteps = episodes x tasks")
    p.add_argument("--tasks", type=int, default=None,
                   help="tasks per episode; default: measured by running the heuristics")
    p.add_argument("--loose-budget", type=float, default=9.9,
                   help="a budget far above any achievable J, so lambda stays 0")
    p.add_argument("--tight-budget", type=float, default=0.0)
    p.add_argument("--k-p", type=float, default=0.3,
                   help="pilot gains are deliberately fast: the point is to reach the "
                        "floor within 40 episodes, not to produce a reportable policy")
    p.add_argument("--k-i", type=float, default=0.3)
    p.add_argument("--grid-size", type=int, default=None,
                   help="number of budgets; default = however many fractions are "
                        "configured (see ELDAS_BUDGET_FRACTIONS)")
    p.add_argument("--cost-freeze-after", type=int, default=None, metavar="N",
                   help="Freeze the constraint scale after N samples. MUST match what "
                        "the sweep will use: J is measured here to place the budget "
                        "grid, so measuring it on a different scale puts the grid in "
                        "the wrong units. Default: 5 episodes' worth of steps.")
    p.add_argument("--output", default="/data/results",
                   help="results root; the pilot writes <root>/pilot-<scenario>/")
    p.add_argument("--skip-heuristics", action="store_true")
    args = p.parse_args(argv)

    if args.print_grid:
        blob = json.loads(Path(args.print_grid).read_text())
        grid = suggest_grid(blob.get("J_free"), args.grid_size)
        if not grid:
            print("", file=sys.stderr)
            return 3
        print(",".join(format(x, "g") for x in grid))
        return 0

    if not args.scenario:
        p.error("--scenario is required unless --print-grid is used")

    out_root = Path(args.output)
    paths.guard_results_root(out_root, what="pilot (W6.1)")
    out_dir = out_root / f"pilot-{args.scenario}"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'=' * 72}\n  W6.1 PILOT - {args.scenario} (seed {args.seed})\n{'=' * 72}")

    heur = {} if args.skip_heuristics else heuristic_reference(args.scenario, args.seed)
    tasks = args.tasks
    if tasks is None and heur:
        tasks = max(int(r["steps"]) for r in heur.values())
    if tasks is None:
        print("[pilot] --tasks is required when --skip-heuristics is set", file=sys.stderr)
        return 2
    timesteps = args.episodes * tasks
    print(f"[pilot] episode length = {tasks} steps -> {args.episodes} episodes "
          f"= {timesteps:,} timesteps per endpoint")

    freeze = args.cost_freeze_after
    if freeze is None:
        freeze = 5 * tasks
    print(f"[pilot] cost scale frozen after {freeze} samples (5 episodes)")
    loose = _run(args.scenario, args.seed, args.loose_budget, timesteps, out_dir,
                 "loose", args.k_p, args.k_i, freeze)
    tight = _run(args.scenario, args.seed, args.tight_budget, timesteps, out_dir,
                 "tight", args.k_p, args.k_i, freeze)

    j_free = loose["J_tail_mean"]
    j_floor = tight["J_tail_mean"]
    grid = suggest_grid(j_free, args.grid_size)

    # ── verdicts ────────────────────────────────────────────────────────────
    verdicts: list[dict] = []

    def verdict(name, ok, detail):
        verdicts.append({"check": name, "verdict": "PASS" if ok else "FAIL",
                         "detail": detail})

    verdict("lambda responds to a tight budget",
            (tight["lambda_final"] or 0.0) > (loose["lambda_final"] or 0.0),
            f"lambda(tight d={args.tight_budget}) = {tight['lambda_final']:.4g} vs "
            f"lambda(loose d={args.loose_budget}) = {loose['lambda_final']:.4g}")
    verdict("J_free measurable, so a budget grid exists", bool(grid),
            f"J_free = {j_free} -> grid {grid}")
    # NOT a pass/fail: J_floor is what a 40-episode run with fast gains reaches, and
    # that is not the feasible floor - it is mostly a statement about how far PPO got.
    # Measured across 10 arm x scenario combinations the two endpoints came back within
    # a fraction of a percent (BURST inverted them), which is why the grid is anchored
    # on J_free alone. Recorded because the sign and size are still worth seeing.
    span = (j_free - j_floor) if (j_free is not None and j_floor is not None) else None
    verdicts.append({"check": "J span between endpoints (informational)",
                     "verdict": "INFO",
                     "detail": f"J_free {j_free} - J_floor {j_floor} = {span}; "
                               f"negative or ~0 means this pilot could not resolve the "
                               f"floor, not that no trade-off exists (the heuristics "
                               f"below show the physical spread)"})
    for r in (loose, tight):
        vl = r["value_loss_final"]
        verdict(f"value_loss bounded ({r['label']})",
                vl is not None and vl == vl and vl < 1e6,
                f"value_loss = {vl} (explosion ~1e10 means reward normalisation broke)")

    # The question a drop count has to answer is "is the POLICY shedding work", and the
    # only thing that separates that from "this workload does not fit" is what a fixed
    # policy does on the same trace. Requiring a literal 0 on LOW/REPLAY conflates the
    # two: on hetero/LOW the agent drops 1 task and so does every heuristic — 18 GPU
    # cards and affinity, not placement. Judge against the best heuristic everywhere,
    # and say which of the two situations a non-zero count is.
    drop_ref = min((r["dropped_tasks"] for r in heur.values()), default=None)
    for r in (loose, tight):
        d_agent = r["dropped_tasks"]
        if drop_ref is None:
            continue
        floor_note = ("this scenario is calibrated below capacity, so any drop at all is "
                      "worth explaining" if args.scenario in ("LOW", "REPLAY") else "")
        verdict(f"drops not worse than best heuristic ({r['label']})",
                d_agent <= drop_ref,
                f"agent dropped {d_agent}, best heuristic dropped {drop_ref}"
                + (f" (both > 0 => the workload does not fit, not the policy)"
                   if d_agent > 0 and drop_ref > 0 else "")
                + f" - dropping MORE than the best fixed policy is buying energy by "
                  f"throwing work away (W3)"
                + (f". {floor_note}" if floor_note else ""))

    report = {
        "scenario": args.scenario, "seed": args.seed,
        "episodes": args.episodes, "tasks_per_episode": tasks,
        "timesteps_per_endpoint": timesteps,
        "gains": {"k_p": args.k_p, "k_i": args.k_i},
        "loose": loose, "tight": tight,
        "J_free": j_free, "J_floor": j_floor,
        "suggested_budgets": grid,
        "heuristics": heur,
        "verdicts": verdicts,
    }
    (out_dir / "pilot_floor.json").write_text(json.dumps(report, indent=2))

    print(f"\n{'=' * 72}\n  RESULT - {args.scenario}\n{'=' * 72}")
    print(f"  episode length      : {tasks} steps")
    print(f"  dual updates        : loose {loose['n_dual_updates']}, "
          f"tight {tight['n_dual_updates']}")
    print(f"  J_free  (d loose)   : {j_free}")
    print(f"  J_floor (d=0)       : {j_floor}")
    # J is the mean of the NORMALISED cost, and each run fits its own normaliser to
    # its own cost stream, so J_free and J_floor are not strictly the same unit. The
    # raw pair below is: it comes from getSlaCost() and is what the Pareto front is
    # actually drawn on. Read the J span as "is there room to trade", and this one as
    # "how much".
    print(f"  raw C_SLA           : {loose['sla_cost']:.4g} (loose) -> "
          f"{tight['sla_cost']:.4g} (tight)")
    print(f"  raw energy kWh      : {loose['energy_kwh']:.4g} (loose) -> "
          f"{tight['energy_kwh']:.4g} (tight)")
    print(f"  dropped (loose/tight/best-heuristic): "
          f"{loose['dropped_tasks']}/{tight['dropped_tasks']}/{drop_ref}")
    if heur:
        print("\n  fixed policies (the physical spread the agent is competing with):")
        print(f"    {'policy':<11s} {'E kWh':>9s} {'C_SLA':>12s} {'J':>8s}  drop")
        for pol, r in heur.items():
            jj = r.get("J")
            print(f"    {pol:<11s} {r['energy_kwh']:9.1f} {r['sla_cost']:12.4e} "
                  f"{(jj if jj is not None else float('nan')):8.4f}  {r['dropped_tasks']}")
        cs = [r["sla_cost"] for r in heur.values()]
        print(f"    raw C_SLA spans {max(cs) / max(min(cs), 1e-9):.2f}x across fixed "
              f"policies - that is the room the agent has to trade")
    for v in verdicts:
        print(f"  [{v['verdict']}] {v['check']} - {v['detail']}")
    print(f"\n  SUGGESTED BUDGET GRID: {','.join(f'{g:g}' for g in grid)}")
    print(f"  written -> {out_dir / 'pilot_floor.json'}\n")
    # INFO entries are observations, not gates.
    return 0 if all(v["verdict"] in ("PASS", "INFO") for v in verdicts) else 1


if __name__ == "__main__":
    raise SystemExit(main())
