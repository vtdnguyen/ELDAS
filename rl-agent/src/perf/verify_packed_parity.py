"""SYS.2 — Prove the packed transport changes speed and NOTHING else.

The packed single-RPC path (``StepCodec`` / ``decode_packed``) is only
acceptable if it is a pure transport optimisation: every number an episode
produces must be identical to the reference per-element path, bit for bit. A
transport that quietly perturbed an observation or a cost would corrupt every
Phase-2 result while looking like a speed-up.

This driver runs the **same action sequence** through both transports and
compares everything the RL loop consumes:

  * per-step observation (exact equality — same doubles, same float32 cast)
  * per-step reward vector ``[R_energy, R_sla]`` and cost ``C_SLA``
  * per-step action mask, done flag, task index and task name
  * episode totals: ``getTotalEnergyKwh()`` and ``getSlaCost()``

Determinism is what makes this a valid comparison: the scenario filter and the
trace are seed-fixed, so replaying an identical action sequence from an
identical reset must retrace the identical trajectory. Any difference is a real
codec bug, not noise.

Run (needs the gateway)::

    docker compose up -d cloudsim-java
    docker compose run --rm rl-agent python src/perf/verify_packed_parity.py \
        --scenario LOW --steps 200
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from environment import CloudSimEnv


def _rollout(env: CloudSimEnv, actions: list[int]) -> dict:
    """Replay a fixed action sequence, recording everything the RL loop sees."""
    obs, _ = env.reset()
    trace = {
        "obs": [obs.copy()],
        "mask": [env.action_masks().copy()],
        "reward": [],
        "cost": [],
        "done": [],
        "task_index": [],
        "task_name": [],
        "dropped_tasks": [],
    }
    for a in actions:
        obs, reward, terminated, truncated, info = env.step(a)
        trace["obs"].append(obs.copy())
        trace["mask"].append(env.action_masks().copy())
        trace["reward"].append(np.asarray(reward, dtype=np.float64).copy())
        trace["cost"].append(float(info["cost"]))
        trace["done"].append(bool(terminated))
        trace["task_index"].append(int(info["task_index"]))
        trace["task_name"].append(str(info["task_name"]))
        # W3.1 — droppedTasks entered the header in wire v2. Compared per step so
        # a codec that mis-parsed the new field is caught here, not by a campaign
        # whose SLA cost is quietly wrong.
        trace["dropped_tasks"].append(int(info["dropped_tasks"]))
        if terminated or truncated:
            break
    trace["energy_kwh"] = float(env._ep.getTotalEnergyKwh())
    trace["sla_cost"] = float(env._ep.getSlaCost())
    trace["dropped_total"] = int(env._ep.getDroppedTasks())
    return trace


def _plan_actions(env: CloudSimEnv, steps: int, seed: int) -> list[int]:
    """Pick a feasible, varied action sequence by replaying from a fresh reset.

    Actions must be feasible or the two runs would diverge for reasons unrelated
    to the codec, so they are chosen by actually walking the env once.
    """
    rng = np.random.default_rng(seed)
    env.reset()
    actions: list[int] = []
    for _ in range(steps):
        mask = env.action_masks()
        feasible = np.flatnonzero(mask)
        a = int(rng.choice(feasible)) if feasible.size else 0
        actions.append(a)
        _, _, terminated, truncated, _ = env.step(a)
        if terminated or truncated:
            break
    return actions


def compare(ref: dict, packed: dict) -> list[str]:
    """Return a list of mismatch descriptions (empty ⇒ exact parity)."""
    problems: list[str] = []

    if len(ref["obs"]) != len(packed["obs"]):
        problems.append(
            f"episode length differs: reference {len(ref['obs'])} vs "
            f"packed {len(packed['obs'])} observations"
        )
        return problems

    for i, (a, b) in enumerate(zip(ref["obs"], packed["obs"])):
        if not np.array_equal(a, b):
            problems.append(f"observation[{i}] differs (max |Δ| = {np.abs(a - b).max():.3e})")
    for i, (a, b) in enumerate(zip(ref["mask"], packed["mask"])):
        if not np.array_equal(a, b):
            problems.append(f"action_mask[{i}] differs: {a} vs {b}")
    for i, (a, b) in enumerate(zip(ref["reward"], packed["reward"])):
        if not np.array_equal(a, b):
            problems.append(f"reward[{i}] differs: {a} vs {b}")
    for key in ("cost", "done", "task_index", "task_name", "dropped_tasks"):
        for i, (a, b) in enumerate(zip(ref[key], packed[key])):
            if a != b:
                problems.append(f"{key}[{i}] differs: {a!r} vs {b!r}")
    for key in ("energy_kwh", "sla_cost", "dropped_total"):
        if ref[key] != packed[key]:
            problems.append(
                f"episode {key} differs: {ref[key]!r} vs {packed[key]!r} "
                f"(Δ = {packed[key] - ref[key]:.6e})"
            )
    return problems


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    # Free-form: under TRACE_PATTERN the label selects a WM-1 file, so OVERLOAD —
    # the one scenario that exercises the drop path end to end — must be sayable.
    ap.add_argument("--scenario", default="LOW")
    ap.add_argument("--seed", type=int, default=int(os.environ.get("RANDOM_SEED", "42")))
    ap.add_argument("--steps", type=int, default=200)
    args = ap.parse_args()

    env = CloudSimEnv(scenario=args.scenario, seed=args.seed, normalize_reward=False)
    try:
        if not env._packed:
            print("[parity] FAIL: gateway does not advertise the packed transport "
                  "— rebuild cloudsim-java.", file=sys.stderr)
            return 2

        actions = _plan_actions(env, args.steps, args.seed)
        print(f"[parity] replaying {len(actions)} feasible actions on both transports")

        packed_trace = _rollout(env, actions)

        # Force the reference path on the SAME env/gateway so the only factor
        # that changes is the transport, not a different JVM or a different
        # episode.
        env._packed = False
        ref_trace = _rollout(env, actions)
        env._packed = True

        problems = compare(ref_trace, packed_trace)
    finally:
        env.close()

    print("\n" + "=" * 72)
    print("  SYS.2 — packed vs reference transport parity")
    print("=" * 72)
    print(f"  steps compared      : {len(ref_trace['reward'])}")
    print(f"  energy kWh (ref)    : {ref_trace['energy_kwh']!r}")
    print(f"  energy kWh (packed) : {packed_trace['energy_kwh']!r}")
    print(f"  C_SLA (ref)         : {ref_trace['sla_cost']!r}")
    print(f"  C_SLA (packed)      : {packed_trace['sla_cost']!r}")
    print(f"  dropped (ref/packed): {ref_trace['dropped_total']} / "
          f"{packed_trace['dropped_total']}")
    if ref_trace["dropped_total"] == 0:
        print("  NOTE: no task was dropped, so the W3.1 droppedTasks field was")
        print("        never exercised beyond 0. Re-run against a cluster or")
        print("        scenario that forces drops (e.g. NUM_HOSTS=1) to cover it.")
    print("-" * 72)
    if problems:
        print(f"  VERDICT: FAIL — {len(problems)} mismatch(es):")
        for p in problems[:20]:
            print(f"    - {p}")
        print("=" * 72 + "\n")
        return 1
    print("  VERDICT: PASS — every observation, reward, cost, mask, done flag,")
    print("           task id/name and episode total is EXACTLY equal.")
    print("           The packed transport changes speed only.")
    print("=" * 72 + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
