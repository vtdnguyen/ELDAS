"""Command-line entry point for WM-1.

    python -m workload.cli generate --source /data/trace/openb_pod_list_default.csv \
                                    --out /data/wm1
    python -m workload.cli plan                     # print the calibration table
    python -m workload.cli adapters                 # what source traces are wired up
"""

from __future__ import annotations

import argparse
import sys

from . import adapters, arrivals, wm1
from .calibrate import ARMS, plan_load
from .jobsize import JobSizePool

DEFAULT_SOURCE = "/data/trace/openb_pod_list_default.csv"
DEFAULT_OUT = "/data/wm1"


def _cmd_generate(args) -> int:
    arms = args.arms.split(",") if args.arms else list(ARMS)
    scenarios = args.scenarios.split(",") if args.scenarios else list(wm1.ALL_SCENARIOS)
    seeds = [int(s) for s in args.seeds.split(",")] if args.seeds else list(wm1.DEFAULT_SEEDS)

    for arm in arms:
        if arm not in ARMS:
            print(f"[ERROR] unknown arm {arm!r}; known: {sorted(ARMS)}", file=sys.stderr)
            return 2
    for s in scenarios:
        if s not in wm1.ALL_SCENARIOS:
            print(f"[ERROR] unknown scenario {s!r}; known: {list(wm1.ALL_SCENARIOS)}",
                  file=sys.stderr)
            return 2

    for arm in arms:
        wm1.generate_arm(args.source, arm, args.out,
                         scenarios=scenarios, seeds=seeds,
                         source_adapter=args.adapter)
    return 0


def _cmd_plan(args) -> int:
    """Print the calibration table without writing anything — cheap sanity check."""
    tasks = adapters.load(args.adapter, args.source)
    print(f"{'arm':<7} {'scenario':<9} {'rho*':>5} {'IDC*':>5} | "
          f"{'p1':>7} {'N':>6} {'lam/h':>7} | {'rho_cpu':>7} {'rho_gpu':>7} {'binds':>6}")
    for arm, (capacity, horizon) in ARMS.items():
        pool = JobSizePool.from_trace(tasks, horizon)
        m = pool.moments
        print(f"--- {arm}: T={horizon / 86400:g}d, {capacity.total_pes} PE / "
              f"{capacity.total_gpus} GPU, E[work]={m.e_work:.0f} PE-s, "
              f"truncated {100 * m.truncated_fraction:.2f}% ---")
        for scenario, (rho, idc_target) in wm1.SCENARIOS.items():
            p = plan_load(pool, capacity, rho, horizon)
            duty = (arrivals.fit_duty_cycle(p.n_task, idc_target, horizon)
                    if args.fit else float("nan"))
            print(f"{arm:<7} {scenario:<9} {rho:>5.2f} {idc_target:>5.0f} | "
                  f"{duty:>7.4f} {p.n_task:>6d} {p.lambda_per_hour:>7.2f} | "
                  f"{p.rho_cpu:>7.3f} {p.rho_gpu:>7.3f} {p.bottleneck:>6}")
    return 0


def _cmd_adapters(args) -> int:
    for name in adapters.list_adapters():
        a = adapters.get(name)
        print(f"{name:<12} capabilities: {sorted(a.capabilities())}")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="workload.cli", description="WM-1 workload model")
    ap.add_argument("--source", default=DEFAULT_SOURCE, help="source trace CSV")
    ap.add_argument("--adapter", default="openb", help="source adapter name")
    sub = ap.add_subparsers(dest="cmd", required=True)

    g = sub.add_parser("generate", help="write the (arm x scenario x seed) matrix")
    g.add_argument("--out", default=DEFAULT_OUT, help="output root directory")
    g.add_argument("--arms", default="", help="comma list (default: all)")
    g.add_argument("--scenarios", default="", help="comma list (default: all)")
    g.add_argument("--seeds", default="", help="comma list (default: 42..46)")
    g.set_defaults(func=_cmd_generate)

    p = sub.add_parser("plan", help="print the calibration table without generating")
    p.add_argument("--fit", action="store_true",
                   help="also fit the duty cycle (slower, a few seconds per row)")
    p.set_defaults(func=_cmd_plan)

    a = sub.add_parser("adapters", help="list registered source adapters")
    a.set_defaults(func=_cmd_adapters)

    args = ap.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
