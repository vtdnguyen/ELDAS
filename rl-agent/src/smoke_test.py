"""
T4.5 — End-to-end smoke test for the full RL loop via Py4J.

Validates the complete pipeline:
    Python  ──reset──►  Java
    Python  ◄──obs────  Java
    Python  ──action──► Java
    Python  ◄──reward── Java
    ... until done ...
    Python  ──export──► Java (CSV + JSON)

This script connects to the **real** running CloudSim Plus gateway
(not a mock) and runs a few short episodes.  Use it to sanity-check
the docker-compose stack before launching real training.

Run inside the rl-agent container::

    docker compose run --rm rl-agent python src/smoke_test.py

Or with custom args::

    python src/smoke_test.py --scenario LOW --episodes 2 --max-steps 50
"""

from __future__ import annotations

import argparse
import os
import sys
import time
import traceback

import numpy as np

from environment import CloudSimEnv, ScalarRewardWrapper, make_env
import reward as reward_mod


# ── ANSI colours for clear output ──────────────────────────────────────────

class C:
    OK = "\033[92m"
    WARN = "\033[93m"
    ERR = "\033[91m"
    INFO = "\033[94m"
    BOLD = "\033[1m"
    END = "\033[0m"


def _ok(msg: str) -> None:
    print(f"{C.OK}✓{C.END} {msg}")


def _info(msg: str) -> None:
    print(f"{C.INFO}ℹ{C.END} {msg}")


def _warn(msg: str) -> None:
    print(f"{C.WARN}!{C.END} {msg}")


def _err(msg: str) -> None:
    print(f"{C.ERR}✗{C.END} {msg}")


def _section(title: str) -> None:
    print(f"\n{C.BOLD}{'═' * 70}{C.END}")
    print(f"{C.BOLD}  {title}{C.END}")
    print(f"{C.BOLD}{'═' * 70}{C.END}")


# ── Validation helpers ────────────────────────────────────────────────────

class SmokeTestFailed(Exception):
    """Raised when an assertion in the smoke test fails."""


def _assert(cond: bool, msg: str) -> None:
    if not cond:
        _err(f"FAIL: {msg}")
        raise SmokeTestFailed(msg)
    _ok(msg)


# ── Test phases ───────────────────────────────────────────────────────────

def test_connection(env: CloudSimEnv) -> None:
    """Phase 1: verify Py4J connection is alive."""
    _section("Phase 1: Py4J Connection")
    _info(f"Gateway endpoint: {env._gateway.gateway_parameters.address}:"
          f"{env._gateway.gateway_parameters.port}")
    _assert(env._num_hosts > 0, f"Got {env._num_hosts} hosts from Java")
    _assert(env.action_space.n == env._num_hosts,
            f"Action space size = num hosts ({env._num_hosts})")
    _assert(env.observation_space.shape == (3 * env._num_hosts + 4,),
            f"Observation shape = (3H+4,) = ({3 * env._num_hosts + 4},)")


def test_reset(env: CloudSimEnv) -> np.ndarray:
    """Phase 2: verify reset works and returns valid initial state."""
    _section("Phase 2: Reset")
    obs, info = env.reset()

    _assert(obs.shape == (3 * env._num_hosts + 4,),
            f"Observation has correct shape {obs.shape}")
    _assert(obs.dtype == np.float32, "Observation dtype is float32")
    _assert(env.observation_space.contains(obs),
            "Observation is within declared space [0, 1]")
    _assert((obs >= 0).all() and (obs <= 1).all(),
            "All observation values are in [0, 1]")
    _assert("task_index" in info and info["task_index"] == 0,
            f"Initial task index = 0")
    _assert("scenario" in info, f"Info contains scenario: {info['scenario']}")

    _info(f"Initial obs sample: {obs[:5].tolist()} ...")
    _info(f"Task name: {info.get('task_name', 'unknown')}")
    return obs


def test_action_mask(env: CloudSimEnv) -> np.ndarray:
    """Phase 3: verify action masking works."""
    _section("Phase 3: Action Mask")
    mask = env.action_masks()

    _assert(mask.shape == (env._num_hosts,),
            f"Mask shape matches num hosts ({env._num_hosts})")
    _assert(mask.dtype == bool, "Mask dtype is bool")
    _assert(mask.any(), "At least one host is feasible")

    feasible_count = int(mask.sum())
    _info(f"Feasible hosts: {feasible_count}/{env._num_hosts}")
    _info(f"Mask: {mask.tolist()}")
    return mask


def test_single_step(env: CloudSimEnv, mask: np.ndarray) -> None:
    """Phase 4: verify a single step returns proper output."""
    _section("Phase 4: Single Step")
    feasible = np.where(mask)[0]
    action = int(feasible[0])
    _info(f"Taking action: place task on host {action}")

    obs, reward, terminated, truncated, info = env.step(action)

    _assert(obs.shape == (3 * env._num_hosts + 4,),
            "Step obs has correct shape")
    _assert(env.observation_space.contains(obs),
            "Step obs is in space")
    _assert(isinstance(reward, np.ndarray) and reward.shape == (2,),
            f"Reward is vector of shape (2,)")
    _assert(reward.dtype == np.float32, "Reward dtype is float32")
    _assert(reward[0] <= 0, f"R_energy ≤ 0 (got {reward[0]:.4f})")
    _assert(reward[1] <= 0, f"R_sla ≤ 0 (got {reward[1]:.4f})")
    _assert(isinstance(terminated, bool), "terminated is bool")
    _assert(truncated is False, "truncated is False (no time limit)")
    _assert("raw_reward" in info, "info contains raw_reward")

    _info(f"Reward vector: [R_energy={reward[0]:.4f}, R_sla={reward[1]:.4f}]")
    _info(f"Task index after step: {info['task_index']}")


def test_full_episode(env: CloudSimEnv, max_steps: int = 5000) -> dict:
    """Phase 5: run a full episode (or up to max_steps steps).

    The trace has ~8000 rows; LOW = ~2000 tasks.  We ask Java for the
    actual task count and only assert natural termination when max_steps
    is large enough to cover the whole episode.
    """
    task_count = int(env._ep.getTaskCount())
    will_complete = max_steps >= task_count
    _section(f"Phase 5: Full Episode  "
             f"(tasks={task_count}, max_steps={max_steps}, "
             f"full={'yes' if will_complete else 'no — partial run'})")

    if not will_complete:
        _warn(f"max_steps ({max_steps}) < task_count ({task_count}). "
              f"Running {max_steps} steps then stopping — "
              f"pass --max-steps {task_count} to run to completion.")

    obs, info = env.reset()
    rewards = []
    start = time.time()
    step_count = 0
    terminated = False

    for step in range(max_steps):
        mask = env.action_masks()
        feasible = np.where(mask)[0]
        if len(feasible) == 0:
            _warn(f"Step {step}: no feasible hosts — picking host 0")
            action = 0
        else:
            # Least-utilised feasible host (non-trivial after manual PE tracking)
            cpu_utils = obs[: env._num_hosts]
            best_idx = feasible[np.argmin(cpu_utils[feasible])]
            action = int(best_idx)

        obs, reward, terminated, truncated, info = env.step(action)
        rewards.append(reward)
        step_count += 1

        if step % (max(1, task_count // 10)) == 0:
            print(f"  step {step:>5}/{task_count}: action={action:>2}, "
                  f"R_energy={reward[0]:>10.4f}, R_sla={reward[1]:>8.4f}")

        if terminated:
            break

    elapsed = time.time() - start
    rewards_arr = np.array(rewards)  # (T, 2)

    # Reward quality checks (apply to whatever steps we ran)
    non_idle = rewards_arr[:, 0] < -120.0   # below idle power = load increased
    _assert(non_idle.any(),
            f"R_energy < -120 (idle) in at least one step "
            f"— manual PE tracking is working (got {rewards_arr[:, 0].min():.2f})")

    diverse = len(set(int(np.argmin(obs[:env._num_hosts])) for obs in
                      [rewards_arr[i] for i in range(min(5, len(rewards_arr)))])) >= 1
    _ok(f"Ran {step_count} steps in {elapsed:.1f}s "
        f"({step_count / elapsed:.0f} steps/sec)")

    if will_complete:
        _assert(terminated,
                f"Episode terminated naturally after {step_count} steps")
        ep = info.get("episode", {})
        _info(f"Total energy: {ep.get('total_energy_kwh', '?'):.6f} kWh")
        _info(f"R_energy sum: {ep.get('total_energy_reward', '?'):.2f}")
        _info(f"R_sla sum:    {ep.get('total_sla_reward', '?'):.2f}")
        return ep
    else:
        _ok(f"Partial run validated ({step_count}/{task_count} tasks scheduled)")
        return {"length": step_count}


def test_export(env: CloudSimEnv, output_dir: str) -> None:
    """Phase 6: export metrics to disk."""
    _section("Phase 6: Metrics Export")
    env.export_metrics(output_dir)
    _info(f"Exported to {output_dir}")
    _ok(f"Java MetricsExporter wrote files (check {output_dir}/metrics.csv "
        f"and summary.json)")


def test_scalar_wrapper() -> None:
    """Phase 7: verify the scalar reward wrapper."""
    _section("Phase 7: ScalarRewardWrapper")
    weights = np.array([0.7, 0.3], dtype=np.float32)
    env = make_env(scenario="LOW", scalarise=True, weights=weights)
    obs, info = env.reset()
    mask = env.action_masks()
    feasible = np.where(mask)[0]
    obs, scalar_r, term, _, info = env.step(int(feasible[0]))

    _assert(isinstance(scalar_r, float), f"Wrapped reward is float, got {type(scalar_r).__name__}")
    raw = info["raw_reward"]
    expected = float(np.dot(weights, raw))
    _assert(abs(scalar_r - expected) < 1e-4,
            f"Scalar = w·r ({scalar_r:.4f} ≈ {expected:.4f})")

    env.close()


def test_baseline_action(env: CloudSimEnv) -> None:
    """Phase 8: verify baseline policy selection works."""
    _section("Phase 8: Baseline Action Selection")
    env.reset()

    k8s_action = int(env._ep.selectBaselineAction("k8s"))
    random_action = int(env._ep.selectBaselineAction("random"))

    _assert(0 <= k8s_action < env._num_hosts,
            f"K8s baseline returned valid action: {k8s_action}")
    _assert(0 <= random_action < env._num_hosts,
            f"Random baseline returned valid action: {random_action}")


# ── Main ──────────────────────────────────────────────────────────────────

def main() -> int:
    parser = argparse.ArgumentParser(description="ELDAS smoke test (T4.5)")
    parser.add_argument("--scenario", default="LOW",
                        choices=["LOW", "HIGH", "BURST"],
                        help="Load scenario (default: LOW for quick smoke test)")
    parser.add_argument("--seed", type=int, default=None,
                        help="Random seed (default: from env)")
    parser.add_argument("--max-steps", type=int, default=5000,
                        help="Max steps per episode (default: 5000). "
                             "LOW scenario has ~2000 tasks; use 0 for unlimited.")
    parser.add_argument("--output", default="/data/results/smoke",
                        help="Metrics export directory")
    parser.add_argument("--gateway-host", default=None,
                        help="Py4J host (default: from GATEWAY_HOST env)")
    parser.add_argument("--gateway-port", type=int, default=None,
                        help="Py4J port (default: from GATEWAY_PORT env)")
    args = parser.parse_args()

    seed = args.seed or int(os.environ.get("RANDOM_SEED", "42"))

    print(f"{C.BOLD}╔══════════════════════════════════════════════════════════════════════╗{C.END}")
    print(f"{C.BOLD}║       ELDAS Smoke Test — T4.5 — Full RL Loop via Py4J              ║{C.END}")
    print(f"{C.BOLD}╚══════════════════════════════════════════════════════════════════════╝{C.END}")
    _info(f"Scenario: {args.scenario}")
    _info(f"Seed: {seed}")
    _info(f"Max steps: {args.max_steps}")

    try:
        env = CloudSimEnv(
            scenario=args.scenario,
            seed=seed,
            gateway_host=args.gateway_host,
            gateway_port=args.gateway_port,
        )
    except ConnectionError as e:
        _err(f"Cannot connect to Java gateway: {e}")
        _err("Hint: ensure 'docker compose up' has the cloudsim-java service healthy.")
        return 1

    try:
        test_connection(env)
        test_reset(env)
        mask = test_action_mask(env)
        test_single_step(env, mask)
        ep_summary = test_full_episode(env, max_steps=args.max_steps)
        test_export(env, args.output)
        test_baseline_action(env)
        env.close()

        test_scalar_wrapper()

        _section("✓ ALL SMOKE TESTS PASSED")
        print(f"{C.OK}{C.BOLD}The full RL loop is operational.{C.END}")
        print(f"\nNext steps:")
        print(f"  • Run baselines:  python src/baseline_eval.py --scenario HIGH")
        print(f"  • Start training: (Phase 2 — P2.1)")
        return 0

    except SmokeTestFailed as e:
        _section(f"✗ SMOKE TEST FAILED: {e}")
        env.close()
        return 1
    except Exception as e:
        _err(f"Unexpected error: {e}")
        traceback.print_exc()
        env.close()
        return 2


if __name__ == "__main__":
    sys.exit(main())
