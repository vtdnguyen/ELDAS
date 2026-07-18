"""SYS — Where does a *training* step's time go, now that the env is fast?

SYS.2 cut the env step from ~22 ms to ~0.45 ms (55×), which means the old
assumption "the simulator is the bottleneck" is dead and the next bottleneck has
to be *measured*, not guessed. This profiler decomposes ``MaskablePPO.learn``
into the only two things it does:

    total_per_step = env_step + learner_overhead
                     └ CloudSim over Py4J   └ policy forward + PPO backward

by timing (a) raw env stepping with no policy at all, and (b) a real
``model.learn`` of the same length. The learner's share is the remainder.

It then sweeps ``torch.set_num_threads``. This is not a micro-optimisation on a
hunch: PPO here trains a tiny MLP (default ``[64, 64]``) on batches of 64, and
for matrices that small the cost of fanning work out to N threads and joining
them back can exceed the arithmetic itself — so the default (half the container's
cores) may well be *slower* than one thread. The point is to measure which.

Run (needs the gateway)::

    docker compose up -d cloudsim-java
    docker compose run --rm rl-agent python src/perf/profile_train.py \
        --scenario LOW --timesteps 6000
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch


def _raw_env_fps(scenario: str, seed: int, steps: int) -> float:
    """Steps/sec with no policy — the pure env ceiling (SYS.2's result)."""
    from environment import CloudSimEnv

    env = CloudSimEnv(scenario=scenario, seed=seed, normalize_reward=False)
    try:
        rng = np.random.default_rng(0)
        env.reset()
        t0 = time.perf_counter()
        for _ in range(steps):
            mask = env.action_masks()
            feasible = np.flatnonzero(mask)
            a = int(rng.choice(feasible)) if feasible.size else 0
            _, _, term, trunc, _ = env.step(a)
            if term or trunc:
                env.reset()
        dt = time.perf_counter() - t0
    finally:
        env.close()
    return steps / dt


def _learn_fps(scenario: str, seed: int, timesteps: int, n_threads: int,
               n_steps: int = 512) -> float:
    """Steps/sec through a real MaskablePPO.learn at a given torch thread count."""
    from sb3_contrib import MaskablePPO

    from optim.pid_lagrangian import PIDLagrangian
    from train_cmdp import build_cmdp_env

    torch.set_num_threads(n_threads)

    pid = PIDLagrangian(k_p=0.05, k_i=0.05, k_d=0.0, lambda_init=0.0)
    env = build_cmdp_env(scenario, seed, pid)
    model = MaskablePPO(
        policy="MlpPolicy", env=env, learning_rate=3e-4, n_steps=n_steps,
        batch_size=64, n_epochs=10, gamma=0.99, gae_lambda=0.95, clip_range=0.2,
        ent_coef=0.01, vf_coef=0.5, max_grad_norm=0.5, verbose=0, seed=seed,
        device="cpu",
    )
    t0 = time.perf_counter()
    model.learn(total_timesteps=timesteps, progress_bar=False)
    dt = time.perf_counter() - t0
    env.close()
    return timesteps / dt


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scenario", default="LOW", choices=["LOW", "HIGH", "BURST"])
    ap.add_argument("--seed", type=int, default=int(os.environ.get("RANDOM_SEED", "42")))
    ap.add_argument("--timesteps", type=int, default=6000,
                    help="timesteps per learn() measurement")
    ap.add_argument("--threads", default="1,2,6",
                    help="comma-separated torch thread counts to compare")
    ap.add_argument("--json-out", default=None)
    args = ap.parse_args()

    default_threads = torch.get_num_threads()
    thread_opts = [int(t) for t in args.threads.split(",") if t.strip()]

    env_fps = _raw_env_fps(args.scenario, args.seed, min(1500, args.timesteps))
    results = {t: _learn_fps(args.scenario, args.seed, args.timesteps, t)
               for t in thread_opts}

    best_t = max(results, key=lambda t: results[t])
    best_fps = results[best_t]
    # Per-step budget: what learn() costs minus what the env costs.
    env_ms = 1e3 / env_fps
    total_ms = 1e3 / best_fps
    learner_ms = total_ms - env_ms

    report = {
        "scenario": args.scenario,
        "cpu_count": os.cpu_count(),
        "torch_default_threads": default_threads,
        "raw_env_fps": env_fps,
        "learn_fps_by_threads": {str(k): v for k, v in results.items()},
        "best_threads": best_t,
        "best_learn_fps": best_fps,
        "env_ms_per_step": env_ms,
        "learner_ms_per_step": learner_ms,
        "learner_share": learner_ms / total_ms if total_ms > 0 else float("nan"),
    }

    print("\n" + "=" * 74)
    print(f"  SYS — training-loop profile (scenario={args.scenario}, "
          f"{os.cpu_count()} CPUs, torch default={default_threads} threads)")
    print("=" * 74)
    print(f"  raw env ceiling (no policy) : {env_fps:8.1f} steps/s  "
          f"({env_ms:.3f} ms/step)")
    print("-" * 74)
    print(f"{'torch threads':<16}{'learn steps/s':>16}{'vs default':>14}")
    base = results.get(default_threads)
    for t in thread_opts:
        rel = f"{results[t] / base:.2f}×" if base else "—"
        star = "  ←best" if t == best_t else ""
        print(f"{t:<16}{results[t]:>16.1f}{rel:>14}{star}")
    print("-" * 74)
    print(f"  per-step budget at best     : env {env_ms:.3f} ms + learner "
          f"{learner_ms:.3f} ms = {total_ms:.3f} ms")
    print(f"  learner share of a step     : {report['learner_share']:.1%}")
    print("=" * 74 + "\n")

    if args.json_out:
        out = Path(args.json_out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, indent=2))
        print(f"[profile_train] report -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
