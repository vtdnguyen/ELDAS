"""SYS — Soak test: do repeated episode resets leak JVM resources?

A converged budget sweep runs hundreds of episodes per policy across dozens of
runs, so anything that leaks *per reset* — a stepping thread that outlives its
episode, a snapshot buffer that is never cleared — turns into an OOM or a thread
explosion hours in, long after the short smoke tests have all passed. Unit tests
cannot see this: each one resets a handful of times.

``SimulationManager.resetSimulation()`` calls ``terminateExisting()``, which
interrupts the old stepping thread and ``join``s it with a 2 s timeout. If that
join ever times out the thread stays alive and the next reset adds another. This
driver exercises the path hard and lets the caller compare JVM thread/heap counts
before and after (via ``jcmd``), which is the only way to actually know.

Run::

    docker compose up -d cloudsim-java
    docker exec cloudsim jcmd 1 Thread.print | grep -c cloudsim-step   # before
    docker compose run --rm rl-agent python src/perf/soak_resets.py --resets 60
    docker exec cloudsim jcmd 1 Thread.print | grep -c cloudsim-step   # after
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from environment import CloudSimEnv


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scenario", default="LOW",
                    help="LOW|HIGH|BURST slice the configured trace; with "
                         "TRACE_PATTERN set, any generated scenario works")
    ap.add_argument("--seed", type=int, default=int(os.environ.get("RANDOM_SEED", "42")))
    ap.add_argument("--resets", type=int, default=60, help="episode resets to perform")
    ap.add_argument("--steps-per-episode", type=int, default=25,
                    help="steps before abandoning the episode and resetting — "
                         "abandoning mid-episode is the harsher case for thread "
                         "cleanup than letting the episode finish naturally")
    args = ap.parse_args()

    env = CloudSimEnv(scenario=args.scenario, seed=args.seed, normalize_reward=False)
    rng = np.random.default_rng(0)
    t0 = time.perf_counter()
    try:
        for i in range(args.resets):
            env.reset()
            for _ in range(args.steps_per_episode):
                mask = env.action_masks()
                feasible = np.flatnonzero(mask)
                a = int(rng.choice(feasible)) if feasible.size else 0
                _, _, term, trunc, _ = env.step(a)
                if term or trunc:
                    break
            if (i + 1) % 10 == 0:
                print(f"[soak] {i + 1}/{args.resets} resets "
                      f"({time.perf_counter() - t0:.1f}s)")
    finally:
        env.close()

    print(f"[soak] done: {args.resets} resets × ≤{args.steps_per_episode} steps "
          f"in {time.perf_counter() - t0:.1f}s")
    print("[soak] now compare JVM threads/heap with the pre-run reading:")
    print("        docker exec cloudsim jcmd 1 Thread.print | grep -c cloudsim-step")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
