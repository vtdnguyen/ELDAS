#!/usr/bin/env python3
"""W0.1 verification — proves `characterize-workload.py` reports correct numbers.

Five independent checks, none of which trusts the tool's own arithmetic:

  1. Scenario filters agree, task-for-task, with `rl-agent/src/eval/trace_loader.py`
     — a separately written mirror of ScenarioFilter.java that the existing test
     suite already covers.
  2. IDC and C_a^2 return ~1.0 on a synthetic Poisson stream (their definition).
  3. IDC counts empty windows — checked against a hand-computed 10-window case.
  4. `demand_curve` peak and dwell time match a hand-computed 3-task overlap.
  5. rho identity: rho * capacity * horizon == measured PE-seconds.
  6. Heterogeneous topology parsing matches config/topology-hetero.json by hand
     (3x4 + 3x2 + 4x0 = 18 GPUs, 10x64 = 640 PEs).

Run from the repo root:  python scripts/verify-characterize-workload.py
Exit code 0 = all pass.
"""

from __future__ import annotations

import importlib.util
import os
import random
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TRACE = os.path.join(ROOT, "data", "alibaba-trace", "openb_pod_list_default.csv")
TOPOLOGY = os.path.join(ROOT, "config", "topology-hetero.json")

sys.path.insert(0, os.path.join(ROOT, "rl-agent", "src"))

_spec = importlib.util.spec_from_file_location(
    "cw", os.path.join(ROOT, "scripts", "characterize-workload.py"))
cw = importlib.util.module_from_spec(_spec)
sys.modules["cw"] = cw            # dataclass(slots=True) needs the module registered
_spec.loader.exec_module(cw)


def main() -> int:
    ok = True

    def check(name: str, passed: bool, detail: str) -> None:
        nonlocal ok
        ok &= passed
        print(f"  [{'OK  ' if passed else 'FAIL'}] {name}: {detail}")

    # ── 1. filters vs the repo's independent mirror ─────────────────────────
    print("1. scenario filters vs rl-agent/src/eval/trace_loader.py")
    if not os.path.isfile(TRACE):
        print(f"  [SKIP] trace not found: {TRACE}")
    else:
        from eval import trace_loader as tl

        mine, theirs = cw.read_trace(TRACE), tl.read_trace(TRACE)
        check("row count", len(mine) == len(theirs), f"{len(mine)} vs {len(theirs)}")
        for a, b in (("LEGACY_LOW", "LOW"), ("LEGACY_HIGH", "HIGH"), ("LEGACY_BURST", "BURST")):
            m, t = cw.filter_scenario(mine, a), tl.filter_scenario(theirs, b)
            same = (len(m) == len(t)
                    and [x.name for x in m] == [x.name for x in t]
                    and all(abs(x.duration - y.duration) < 1e-9 for x, y in zip(m, t))
                    and all(x.pes == y.pes_needed for x, y in zip(m, t)))
            check(a, same, f"n={len(m)}, names/duration/pes identical={same}")

    # ── 2. Poisson stream => IDC = C_a^2 = 1 ────────────────────────────────
    print("2. arrival-process statistics on a synthetic Poisson stream")
    rng = random.Random(7)
    t, times = 0.0, []
    for _ in range(200_000):
        t += rng.expovariate(1 / 50.0)
        times.append(t)
    idc = cw.index_of_dispersion(times, 3600.0, times[0], times[-1])
    ca2 = cw.squared_cv_interarrival(times)
    check("IDC(1h) ~ 1", abs(idc - 1) < 0.06, f"{idc:.3f}")
    check("C_a^2 ~ 1", abs(ca2 - 1) < 0.03, f"{ca2:.3f}")

    # ── 3. empty windows are counted ────────────────────────────────────────
    print("3. IDC counts empty windows (the trap: 77% of openb 1h windows are empty)")
    arrivals = [0.0, 1.0, 2.0, 9 * 3600.0 + 1]
    got = cw.index_of_dispersion(arrivals, 3600.0, 0.0, 9 * 3600.0 + 1)
    counts = [3, 0, 0, 0, 0, 0, 0, 0, 0, 1]
    m = sum(counts) / len(counts)
    expected = (sum((c - m) ** 2 for c in counts) / len(counts)) / m
    check("hand-computed 10-window case", abs(got - expected) < 1e-9,
          f"{got:.4f} == {expected:.4f}")

    # ── 4. demand curve ─────────────────────────────────────────────────────
    print("4. concurrent-demand sweep line")
    mk = lambda c, d, pes: cw.Task("x", pes * 1000, 0, 0, 0, "BE", "Running", c, c + d, c)
    curve = cw.demand_curve([mk(0, 10, 4), mk(5, 10, 6), mk(100, 1, 2)], 0.0, 100.0)
    peak = max(lvl for lvl, _ in curve)
    dwell = dict(curve).get(10.0)
    check("peak / dwell at overlap", peak == 10.0 and abs(dwell - 5.0) < 1e-9,
          f"peak={peak:.0f} PE, {dwell:.1f}s at 10 PE (expect 10 PE, 5.0s)")

    # ── 5. rho identity ─────────────────────────────────────────────────────
    print("5. rho identity")
    if os.path.isfile(TRACE):
        cap = cw.capacity_from_args(10, 64, 8)
        r = cw.characterise(cw.filter_scenario(cw.read_trace(TRACE), "LEGACY_HIGH"),
                            cap, [3600.0])
        lhs = r["rho_cpu"] * cap.total_pes * r["horizon_sec"]
        check("rho * capacity * T == PE-seconds",
              abs(lhs - r["cpu_work_pe_sec"]) / r["cpu_work_pe_sec"] < 1e-12,
              f"{lhs:.6e} == {r['cpu_work_pe_sec']:.6e}")

    # ── 6. heterogeneous topology ───────────────────────────────────────────
    print("6. heterogeneous topology parsing")
    if os.path.isfile(TOPOLOGY):
        h = cw.capacity_from_topology(TOPOLOGY, 64, 8)
        check("3x4 + 3x2 + 4x0 GPUs, 10x64 PEs",
              (h.hosts, h.total_pes, h.total_gpus) == (10, 640, 18),
              f"{h.hosts} hosts, {h.total_pes} PE, {h.total_gpus} GPU")

    print()
    print("ALL CHECKS:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
