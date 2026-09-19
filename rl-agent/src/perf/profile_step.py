"""SYS.2 / SYS.3 — Measure where an RL step's wall-clock time actually goes.

Both SYS.2 and SYS.3 are explicitly gated on *measuring before optimising*, so
this script produces the evidence rather than assuming it.

What the first run of this profiler found
-----------------------------------------
The naive reading — "``step()`` takes 21 ms, so the CloudSim simulation is
slow" — was **wrong**, and the correction is the whole point of SYS.2:

  * ``ep.ping(0)``  — a no-op RPC                     ≈ 0.18 ms
  * ``ep.step(a)``  — the RPC that runs the simulation ≈ 0.49 ms
  * ``list(obs)``   — reading the 64-element result    ≈ 18.3 ms  ← 0.29 ms/elem

Py4J passes Java arrays **by reference**: every element of a ``double[]`` costs
its own round trip. So ~76 RPCs (64 observation + 10 mask + a few scalars) were
being paid per step, while the JVM's actual work was ~0.5 ms. Transport was
~96% of the step budget, not the ~2% a "1 RPC per call" model predicts, and it
grew with the host count — which is what made 50 hosts (SYS.3) disproportionately
slow.

The fix is not batching *steps* (which would sacrifice per-step action masking
for no reason). It is batching each step's *payload* into one RPC: Java encodes
observation+reward+cost+done+mask into a single ``byte[]`` (the one array type
Py4J passes by value) — see ``sim.StepCodec`` and ``state_builder.decode_packed``.
``perf/verify_packed_parity.py`` proves the numbers are bit-identical.

This profiler measures BOTH transports in one run and reports the real speed-up.

Run (needs the gateway)::

    docker compose up -d cloudsim-java
    docker compose run --rm rl-agent python src/perf/profile_step.py \
        --scenario LOW --steps 300

    # SYS.3 — same probe at 50 hosts (restart Java with NUM_HOSTS=50 first)
    NUM_HOSTS=50 docker compose up -d --force-recreate cloudsim-java
    docker compose run --rm rl-agent python src/perf/profile_step.py \
        --scenario LOW --steps 300 --json-out /data/results/perf/hosts50.json
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

# Support both `python src/perf/profile_step.py` and `python -m perf.profile_step`.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from environment import CloudSimEnv


@dataclass
class Stats:
    """Latency summary in milliseconds.

    p50 rather than mean is the headline: RPC latency is right-skewed, so a mean
    over a few GC pauses would overstate the typical step.
    """

    n: int
    p50_ms: float
    p95_ms: float
    mean_ms: float
    min_ms: float
    max_ms: float

    @staticmethod
    def of(samples_sec: list[float]) -> "Stats":
        ms = sorted(s * 1e3 for s in samples_sec)
        if not ms:
            nan = float("nan")
            return Stats(0, nan, nan, nan, nan, nan)
        return Stats(
            n=len(ms),
            p50_ms=float(statistics.median(ms)),
            p95_ms=float(ms[min(len(ms) - 1, int(0.95 * len(ms)))]),
            mean_ms=float(statistics.fmean(ms)),
            min_ms=float(ms[0]),
            max_ms=float(ms[-1]),
        )


# ── Probes ─────────────────────────────────────────────────────────────────

def measure_rtt(ep, samples: int = 200, warmup: int = 50) -> Stats:
    """Pure Py4J round-trip time via the no-op ``ping``.

    The warm-up matters: the JVM interprets bytecode before the JIT compiles the
    Py4J reflection path, so the first calls are several times slower than steady
    state. Profiling those would overstate transport cost.
    """
    for i in range(warmup):
        ep.ping(i)
    out: list[float] = []
    for i in range(samples):
        t0 = time.perf_counter()
        ep.ping(i)
        out.append(time.perf_counter() - t0)
    return Stats.of(out)


def measure_transport(env: CloudSimEnv, steps: int, packed: bool) -> tuple[Stats, Stats]:
    """Time ``step`` + ``action_masks`` over a real episode on one transport.

    Actions are drawn uniformly from the *feasible* hosts so the simulation
    follows a realistic trajectory: always picking host 0 would pack everything
    onto one host and exercise an unrepresentative code path (and mask). The RNG
    is re-seeded per transport so both walk the same distribution of states.
    """
    was_packed = env._packed
    env._packed = packed
    try:
        rng = np.random.default_rng(0)
        step_times: list[float] = []
        mask_times: list[float] = []

        env.reset()
        for _ in range(steps):
            t0 = time.perf_counter()
            mask = env.action_masks()
            mask_times.append(time.perf_counter() - t0)

            feasible = np.flatnonzero(mask)
            action = int(rng.choice(feasible)) if feasible.size else 0

            t0 = time.perf_counter()
            _, _, terminated, truncated, _ = env.step(action)
            step_times.append(time.perf_counter() - t0)

            if terminated or truncated:
                env.reset()
        return Stats.of(step_times), Stats.of(mask_times)
    finally:
        env._packed = was_packed


# ── Report ─────────────────────────────────────────────────────────────────

def _budget(step: Stats, mask: Stats, rtt: Stats, num_hosts: int) -> dict:
    """Decompose one transport's per-step budget.

    ``jvm_simulation`` is measured by subtraction against the no-op ping: the
    packed step is exactly one RPC, so ``step_p50 − rtt_p50`` is the JVM's work.
    ``implied_rpcs`` divides the total by the cost of one round trip — the
    diagnostic that exposed the per-element array proxying.
    """
    total = step.p50_ms + mask.p50_ms
    return {
        "step_p50_ms": step.p50_ms,
        "mask_p50_ms": mask.p50_ms,
        "total_per_step_ms": total,
        "implied_rpcs": total / rtt.p50_ms if rtt.p50_ms > 0 else float("nan"),
        "steps_per_sec": 1e3 / total if total > 0 else float("nan"),
    }


def build_report(rtt: Stats, packed: tuple[Stats, Stats],
                 legacy: tuple[Stats, Stats] | None, *,
                 scenario: str, num_hosts: int) -> dict:
    p = _budget(packed[0], packed[1], rtt, num_hosts)
    # The packed path is a single RPC, so what remains after transport is the
    # simulation itself — the honest measure of how heavy CloudSim actually is.
    jvm_ms = packed[0].p50_ms - rtt.p50_ms

    report = {
        "scenario": scenario,
        "num_hosts": num_hosts,
        "rtt_ping": asdict(rtt),
        "packed": {"step": asdict(packed[0]), "mask": asdict(packed[1]), **p},
        "jvm_simulation_per_step_ms": jvm_ms,
    }
    if legacy is not None:
        l = _budget(legacy[0], legacy[1], rtt, num_hosts)
        report["legacy"] = {"step": asdict(legacy[0]), "mask": asdict(legacy[1]), **l}
        report["speedup"] = l["total_per_step_ms"] / p["total_per_step_ms"]
        report["transport_share_legacy"] = (
            1.0 - jvm_ms / l["total_per_step_ms"] if l["total_per_step_ms"] > 0 else float("nan")
        )
    return report


def print_report(r: dict) -> None:
    print("\n" + "=" * 74)
    print(f"  SYS.2/SYS.3 — step latency profile "
          f"(scenario={r['scenario']}, hosts={r['num_hosts']})")
    print("=" * 74)
    print(f"  pure RTT (ping, 1 RPC) : {r['rtt_ping']['p50_ms']:.3f} ms (p50)")
    print(f"  JVM simulation / step  : {r['jvm_simulation_per_step_ms']:.3f} ms "
          f"(packed step − ping)")
    print("-" * 74)
    print(f"{'transport':<12}{'step ms':>10}{'mask ms':>10}{'total ms':>11}"
          f"{'~RPCs':>9}{'steps/s':>10}")
    for label, key in (("legacy", "legacy"), ("packed", "packed")):
        if key not in r:
            continue
        b = r[key]
        print(f"{label:<12}{b['step_p50_ms']:>10.3f}{b['mask_p50_ms']:>10.3f}"
              f"{b['total_per_step_ms']:>11.3f}{b['implied_rpcs']:>9.1f}"
              f"{b['steps_per_sec']:>10.1f}")
    print("-" * 74)
    if "speedup" in r:
        print(f"  SYS.2 RESULT: packed transport is {r['speedup']:.1f}× faster per step.")
        print(f"    Transport was {r['transport_share_legacy']:.1%} of the legacy step "
              f"budget — Py4J proxies")
        print(f"    double[]/boolean[] one element per RPC, so the legacy path pays")
        print(f"    ~{r['legacy']['implied_rpcs']:.0f} round trips per step vs "
              f"~{r['packed']['implied_rpcs']:.0f} for packed.")
        print(f"    Numbers are bit-identical (see verify_packed_parity.py).")
    print("=" * 74 + "\n")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scenario", default="LOW", help="LOW|HIGH|BURST slice the configured trace; with TRACE_PATTERN set, any generated scenario (incl. OVERLOAD, REPLAY) selects its own file")
    ap.add_argument("--seed", type=int, default=int(os.environ.get("RANDOM_SEED", "42")))
    ap.add_argument("--steps", type=int, default=300, help="timed env steps per transport")
    ap.add_argument("--rtt-samples", type=int, default=200)
    ap.add_argument("--skip-legacy", action="store_true",
                    help="profile only the packed transport (legacy is slow at H=50)")
    ap.add_argument("--json-out", default=None, help="write the report as JSON")
    args = ap.parse_args()

    env = CloudSimEnv(scenario=args.scenario, seed=args.seed, normalize_reward=False)
    try:
        if not env._packed:
            print("[profile_step] WARNING: gateway has no packed transport — "
                  "rebuild cloudsim-java to measure SYS.2.", file=sys.stderr)

        rtt = measure_rtt(env._ep, samples=args.rtt_samples)
        packed = measure_transport(env, args.steps, packed=True) if env._packed else None
        legacy = None if args.skip_legacy else measure_transport(env, args.steps, packed=False)

        if packed is None:
            print("[profile_step] no packed transport to profile.", file=sys.stderr)
            return 2

        report = build_report(rtt, packed, legacy, scenario=args.scenario,
                              num_hosts=env._num_hosts)
    finally:
        env.close()

    print_report(report)

    if args.json_out:
        out = Path(args.json_out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, indent=2))
        print(f"[profile_step] report -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
