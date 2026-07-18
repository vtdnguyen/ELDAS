"""T8.9 — Minimum MORL PPO training.

Goal: produce ONE trained policy (single weight w_energy=0.5, w_sla=0.5)
sufficient to populate the 6-row comparison table in T8.10. This is NOT
a tuned model — no hyperparameter sweep, no architecture search, no
weight sweep. Those belong to Phase 2A.

The expectation is modest: the trained policy should land somewhere on
the Pareto front established by the 5 baselines (see CLAUDE.md Phase
1.8 Block B table). If PPO-min produces results that are obviously
worse than every baseline (e.g. higher energy AND worse SLA than
RoundRobin), the training is broken — investigate before drawing
conclusions in T8.10.

Usage::

    docker compose run --rm rl-agent python src/train_min.py
    # Customisable:
    docker compose run --rm rl-agent python src/train_min.py \\
        --total-timesteps 200000 --weight-energy 0.5 --weight-sla 0.5
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

# sb3-contrib hosts MaskablePPO; SB3 itself does not.
from sb3_contrib import MaskablePPO
from sb3_contrib.common.maskable.callbacks import MaskableEvalCallback
from sb3_contrib.common.maskable.utils import get_action_masks
from sb3_contrib.common.wrappers import ActionMasker
from stable_baselines3.common.callbacks import CallbackList
from stable_baselines3.common.monitor import Monitor

from environment import CloudSimEnv, ScalarRewardWrapper
import reward as reward_mod


# ── Environment construction ─────────────────────────────────────────────

def _action_mask_fn(env):
    """ActionMasker requires a callable that takes the env and returns the mask.

    The wrapper traversal stops at the first class exposing ``action_masks``.
    """
    return env.action_masks()


def build_env(scenario: str, seed: int, weights: np.ndarray,
              normalize_reward: bool = True) -> "Monitor":
    """Construct the training env: CloudSim → Scalar → ActionMasker → Monitor.

    Wrapping order matters:
      1. ``CloudSimEnv``     — raw multi-objective env (with Welford reward
         normalisation when ``normalize_reward=True``)
      2. ``ScalarRewardWrapper`` — collapses ``[R_energy, R_sla]`` to scalar
         via linear scalarisation with the configured weights
      3. ``ActionMasker``    — exposes ``action_masks()`` for MaskablePPO
      4. ``Monitor``         — logs episode-level reward/length for SB3

    Reward normalisation is ON by default for T8.9: the raw reward scale
    (energy ~−700/step, SLA can hit −50k/step on violations) is far too
    wide for PPO's value head — without normalisation the value loss
    explodes and approx_kl collapses to ~1e-9 (verified empirically in
    the 5k smoke run). ``info["raw_reward"]`` still carries the original
    vector so eval-time reporting is unaffected.
    """
    base = CloudSimEnv(scenario=scenario, seed=seed,
                       normalize_reward=normalize_reward)
    scalar = ScalarRewardWrapper(base, weights=weights)
    masked = ActionMasker(scalar, action_mask_fn=_action_mask_fn)
    return Monitor(masked)


# ── Training driver ──────────────────────────────────────────────────────

def train(
    scenario: str,
    seed: int,
    total_timesteps: int,
    weights: np.ndarray,
    out_path: Path,
    log_dir: Path,
) -> Path:
    """Train one MaskablePPO model with the given scalarisation weights.

    Returns the path to the saved model zip. The caller is responsible
    for evaluating it (see T8.10 / baseline_eval-style harness).
    """
    print(f"[train_min] scenario={scenario}, seed={seed}, "
          f"weights={weights.tolist()}, total_timesteps={total_timesteps:,}")

    env = build_env(scenario, seed, weights)

    # Hyperparameters — kept close to SB3 defaults. We tune only n_steps
    # because the trace has ~7k steps per episode, so 2048 (the default)
    # collects less than half an episode per rollout; 512 makes update
    # cadence reasonable without starving the policy of trajectories.
    # NOTE: tensorboard_log intentionally omitted. The tensorboard pip extra
    # is not in requirements.txt and we don't need it for the "minimum" PPO
    # (WandB tracker covers logging needs).
    model = MaskablePPO(
        policy="MlpPolicy",
        env=env,
        learning_rate=3e-4,
        n_steps=512,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        vf_coef=0.5,
        max_grad_norm=0.5,
        verbose=1,
        seed=seed,
        device="cpu",   # the policy is tiny; GPU is wasted overhead here.
    )

    model.learn(total_timesteps=total_timesteps, progress_bar=False)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    model.save(str(out_path))
    print(f"[train_min] model saved -> {out_path}")

    env.close()
    return out_path


# ── Evaluation: one full episode under the trained policy ────────────────

def evaluate(model_path: Path, scenario: str, seed: int,
             weights: np.ndarray, output_dir: Path) -> dict:
    """Run one greedy episode with the trained model, export metrics.

    The output mirrors what ``baseline_eval.py`` produces for the 5 baselines,
    so T8.10's comparison table can ingest it without special-casing.
    """
    print(f"[train_min] evaluating {model_path} on {scenario}/seed={seed}")
    env = build_env(scenario, seed, weights)
    base_env = env.env.env  # Monitor -> ActionMasker -> ScalarRewardWrapper
    cloud_env = base_env.env  # ScalarRewardWrapper -> CloudSimEnv

    model = MaskablePPO.load(str(model_path), device="cpu")

    obs, _ = env.reset()
    total_energy_r = 0.0
    total_sla_r = 0.0
    steps = 0
    while True:
        masks = get_action_masks(env)
        action, _ = model.predict(obs, action_masks=masks, deterministic=True)
        obs, _reward, terminated, truncated, info = env.step(int(action))
        # We need the RAW vector reward (the underlying CloudSimEnv tracks it).
        raw = info.get("raw_reward")
        if raw is not None:
            total_energy_r += float(raw[0])
            total_sla_r += float(raw[1])
        steps += 1
        if terminated or truncated:
            break

    energy_kwh = float(cloud_env._ep.getTotalEnergyKwh())
    # C6 — read C_SLA straight from Java (getSlaCost), the SAME source the
    # heuristics and CMDP-PID use, so ppo-min lands on the campaign's SLA axis
    # (sla_cost = Σ κ·max(0, tardiness)) rather than the reward-magnitude
    # (total_sla_reward). Without this a ppo-min row cannot be compared to the
    # others under one metric (Lưu ý #5).
    sla_cost = float(cloud_env._ep.getSlaCost())
    output_dir.mkdir(parents=True, exist_ok=True)
    cloud_env.export_metrics(str(output_dir))

    # T8.10 — merge the eval result into baseline_results.json so the
    # comparison script picks up total_sla_reward without special-casing.
    parent = output_dir.parent  # e.g. /data/results/baseline-HIGH
    baseline_json = parent / "baseline_results.json"
    existing: dict = {}
    if baseline_json.exists():
        try:
            existing = json.loads(baseline_json.read_text())
        except Exception:
            existing = {}
    existing["ppo-min"] = {
        "scheduler": "ppo-min",
        "scenario": scenario,
        "seed": seed,
        "steps": steps,
        "total_energy_reward": total_energy_r,
        "total_sla_reward": total_sla_r,
        "total_energy_kwh": energy_kwh,
        "total_sla_cost": sla_cost,
        "weights": weights.tolist(),
    }
    baseline_json.write_text(json.dumps(existing, indent=2))
    print(f"[train_min] merged eval result -> {baseline_json}")

    summary = {
        "scheduler": "ppo-min",
        "scenario": scenario,
        "seed": seed,
        "steps": steps,
        "total_energy_kwh": energy_kwh,
        "total_sla_cost": sla_cost,
        "total_energy_reward": total_energy_r,
        "total_sla_reward": total_sla_r,
        "weights": weights.tolist(),
    }
    print(f"[train_min] eval: {summary}")
    env.close()
    return summary


# ── CLI ──────────────────────────────────────────────────────────────────

def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--scenario", default="HIGH", choices=["LOW", "HIGH", "BURST"])
    p.add_argument("--seed", type=int,
                   default=int(os.environ.get("RANDOM_SEED", "42")))
    p.add_argument("--total-timesteps", type=int, default=100_000,
                   help="PPO learning timesteps (default 100k, recommended 100k–300k for T8.9)")
    p.add_argument("--weight-energy", type=float, default=0.5)
    p.add_argument("--weight-sla",    type=float, default=0.5)
    p.add_argument("--model-out", default="/data/models/ppo-min.zip", type=Path)
    p.add_argument("--log-dir",   default="/data/models/ppo-min-tb", type=Path)
    p.add_argument("--eval-out",  default="/data/results/baseline-HIGH/ppo-min", type=Path)
    p.add_argument("--skip-train", action="store_true",
                   help="Load existing model and only evaluate (useful for re-runs)")
    args = p.parse_args()

    weights = np.array([args.weight_energy, args.weight_sla], dtype=np.float32)

    if not args.skip_train:
        train(args.scenario, args.seed, args.total_timesteps,
              weights, args.model_out, args.log_dir)
    elif not args.model_out.exists():
        print(f"[train_min] --skip-train set but {args.model_out} missing",
              file=sys.stderr)
        return 2

    evaluate(args.model_out, args.scenario, args.seed, weights, args.eval_out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
