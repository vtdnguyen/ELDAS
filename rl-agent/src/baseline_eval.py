"""
T4.4 — Baseline evaluation: run full episodes with K8s-default and
Random schedulers, then compare against MORL results.

This script uses the same ``CloudSimEnv`` as the RL agent but delegates
action selection to ``GatewayEntryPoint.selectBaselineAction()`` on the
Java side.  This ensures the baselines operate under identical simulation
conditions (same trace, same scenario, same hosts).

Usage (inside Docker)::

    python src/baseline_eval.py                         # defaults
    python src/baseline_eval.py --scenario HIGH --seed 42 --output /data/results

Usage (standalone)::

    from baseline_eval import evaluate_baseline, run_all_baselines
    results = run_all_baselines(scenario="HIGH", seed=42)
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np

from environment import CloudSimEnv
from tracker import ExperimentTracker


# ── Single baseline episode ───────────────────────────────────────────────

def evaluate_baseline(
    env: CloudSimEnv,
    policy_name: str,
    scenario: str = "HIGH",
    seed: int = 42,
) -> dict[str, Any]:
    """Run one full episode using a baseline scheduler.

    Parameters
    ----------
    env : CloudSimEnv
        The environment instance (must NOT be wrapped with ScalarRewardWrapper).
    policy_name : str
        ``"k8s"`` or ``"random"`` — forwarded to Java's
        ``GatewayEntryPoint.selectBaselineAction()``.
    scenario : str
        Scenario name for the episode reset.
    seed : int
        Random seed.

    Returns
    -------
    dict
        Episode summary with energy, SLA, and scheduling metrics.
    """
    obs, info = env.reset(seed=seed, options={"scenario": scenario})

    total_energy_reward = 0.0
    total_sla_reward = 0.0
    steps = 0

    while True:
        # Ask Java to select the action using the baseline policy
        action = int(env._ep.selectBaselineAction(policy_name))

        obs, reward_vec, terminated, truncated, info = env.step(action)

        total_energy_reward += float(reward_vec[0])
        total_sla_reward += float(reward_vec[1])
        steps += 1

        if terminated or truncated:
            break

    episode_info = info.get("episode", {})

    return {
        "scheduler": policy_name,
        "scenario": scenario,
        "seed": seed,
        "steps": steps,
        "total_energy_reward": total_energy_reward,
        "total_sla_reward": total_sla_reward,
        "mean_energy_reward": total_energy_reward / max(1, steps),
        "mean_sla_reward": total_sla_reward / max(1, steps),
        "total_energy_kwh": episode_info.get("total_energy_kwh", 0.0),
    }


# ── Run all baselines ─────────────────────────────────────────────────────

BASELINE_POLICIES = ["k8s", "random"]


def run_all_baselines(
    scenario: str = "HIGH",
    seed: int = 42,
    tracker: ExperimentTracker | None = None,
    gateway_host: str | None = None,
    gateway_port: int | None = None,
    output_dir: str | None = None,
) -> dict[str, dict[str, Any]]:
    """Run all baseline schedulers and return a comparison dict.

    Parameters
    ----------
    scenario : str
        Load scenario.
    seed : int
        Random seed.
    tracker : ExperimentTracker or None
        If provided, log episode results and comparison table.
    gateway_host, gateway_port : str, int or None
        Py4J connection overrides.
    output_dir : str or None
        If provided, export per-policy ``metrics.csv`` + ``summary.json``
        under ``{output_dir}/{policy}/`` via Java's MetricsExporter.

    Returns
    -------
    dict
        ``{"k8s": {...}, "random": {...}}`` — per-scheduler metrics.
    """
    env = CloudSimEnv(
        scenario=scenario,
        seed=seed,
        gateway_host=gateway_host,
        gateway_port=gateway_port,
    )

    results: dict[str, dict[str, Any]] = {}

    for policy in BASELINE_POLICIES:
        print(f"\n[baseline_eval] Running baseline: {policy} "
              f"(scenario={scenario}, seed={seed})")

        result = evaluate_baseline(env, policy, scenario=scenario, seed=seed)
        results[policy] = result

        print(f"[baseline_eval] {policy}: {result['steps']} steps, "
              f"energy={result['total_energy_kwh']:.4f} kWh, "
              f"r_energy={result['total_energy_reward']:.2f}, "
              f"r_sla={result['total_sla_reward']:.2f}")

        # Export per-policy metrics.csv + summary.json via Java MetricsExporter
        if output_dir is not None:
            policy_dir = str(Path(output_dir) / policy)
            env.export_metrics(policy_dir)
            print(f"[baseline_eval] Metrics exported → {policy_dir}/")

        if tracker is not None:
            tracker.log_episode(
                episode=0,
                info={"episode": result},
                scheduler=policy,
            )

    if tracker is not None:
        comparison = {
            name: {
                "energy_kwh": r["total_energy_kwh"],
                "total_energy_reward": r["total_energy_reward"],
                "total_sla_reward": r["total_sla_reward"],
                "steps": r["steps"],
            }
            for name, r in results.items()
        }
        tracker.log_comparison(comparison)

    env.close()
    return results


# ── Results export ─────────────────────────────────────────────────────────

def save_results(results: dict[str, dict[str, Any]], output_dir: str) -> None:
    """Save baseline results to a JSON file."""
    path = Path(output_dir)
    path.mkdir(parents=True, exist_ok=True)
    out_file = path / "baseline_results.json"

    # Convert numpy types to native Python for JSON serialisation
    clean = {}
    for name, metrics in results.items():
        clean[name] = {
            k: float(v) if isinstance(v, (np.floating, float)) else v
            for k, v in metrics.items()
        }

    with open(out_file, "w") as f:
        json.dump(clean, f, indent=2)

    print(f"[baseline_eval] Results saved to {out_file}")


# ── CLI entry point ───────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run baseline scheduler evaluations"
    )
    parser.add_argument(
        "--scenario", default="HIGH", choices=["LOW", "HIGH", "BURST"],
        help="Load scenario (default: HIGH)"
    )
    parser.add_argument(
        "--seed", type=int, default=None,
        help="Random seed (default: from RANDOM_SEED env var or 42)"
    )
    parser.add_argument(
        "--output", default="/data/results",
        help="Output directory for results JSON (default: /data/results)"
    )
    parser.add_argument(
        "--no-wandb", action="store_true",
        help="Disable WandB logging"
    )
    args = parser.parse_args()

    seed = args.seed or int(os.environ.get("RANDOM_SEED", "42"))

    tracker = ExperimentTracker(
        project="ELDAS",
        run_name=f"baseline-{args.scenario.lower()}-s{seed}",
        config={
            "scenario": args.scenario,
            "seed": seed,
            "type": "baseline",
        },
        tags=["baseline", args.scenario.lower()],
        enabled=False if args.no_wandb else None,
    )

    # Output goes into a scenario-named subfolder so multiple scenarios
    # can coexist under the same --output root (e.g. /data/results).
    scenario_dir = str(Path(args.output) / f"baseline-{args.scenario}")

    results = run_all_baselines(
        scenario=args.scenario,
        seed=seed,
        tracker=tracker,
        output_dir=scenario_dir,
    )

    save_results(results, scenario_dir)
    tracker.finish()


if __name__ == "__main__":
    main()
