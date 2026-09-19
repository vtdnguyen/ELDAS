#!/usr/bin/env python3
"""W2.1/W2.2 verification — the gateway loads the trace TRACE_PATTERN points at.

Runs against a live gateway and checks the property that matters: for every
(scenario, seed) the simulator ends up with exactly the workload WM-1 generated for
that combination, and never silently with something else.

Checked per (scenario, seed):
  * task count matches the arm's wm1-manifest.json
  * OVERLOAD and REPLAY work at all — they are not members of the legacy Scenario enum,
    so they exercise the label/filter decoupling
  * a scenario with no generated file raises instead of falling back to the legacy trace

Usage (inside the rl-agent container, gateway already up with TRACE_PATTERN set):
    python scripts/verify-trace-pattern.py --arm homo
"""

from __future__ import annotations

import argparse
import json
import os
import sys

from py4j.java_gateway import GatewayParameters, JavaGateway
from py4j.protocol import Py4JJavaError


def connect(host: str, port: int) -> JavaGateway:
    gw = JavaGateway(gateway_parameters=GatewayParameters(
        address=host, port=port, auto_convert=True))
    # ping() is the only no-op the gateway answers before a reset; getActionSize()
    # and friends deliberately refuse until an episode exists.
    gw.entry_point.ping(0)
    return gw


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--arm", default="homo")
    ap.add_argument("--manifest-root", default="/data/wm1")
    ap.add_argument("--host", default=os.environ.get("GATEWAY_HOST", "cloudsim-java"))
    ap.add_argument("--port", type=int, default=int(os.environ.get("GATEWAY_PORT", "25333")))
    ap.add_argument("--seeds", default="42,43")
    args = ap.parse_args(argv)

    manifest_path = os.path.join(args.manifest_root, args.arm, "wm1-manifest.json")
    if not os.path.isfile(manifest_path):
        print(f"[ERROR] manifest not found: {manifest_path}", file=sys.stderr)
        return 2
    with open(manifest_path, encoding="utf-8") as f:
        manifest = json.load(f)

    expected = {(t["scenario"], t["seed"]): t["n_task"] for t in manifest["traces"]}
    seeds = [int(s) for s in args.seeds.split(",")]

    gw = connect(args.host, args.port)
    ep = gw.entry_point

    ok = True
    print(f"{'scenario':<10} {'seed':>5} {'expected':>9} {'loaded':>8}  verdict")
    for scenario in ("LOW", "HIGH", "BURST", "OVERLOAD", "REPLAY"):
        for seed in seeds:
            want = expected.get((scenario, seed))
            if want is None:
                continue
            ep.reset(scenario, seed)
            got = ep.getTaskCount()
            good = got == want
            ok &= good
            print(f"{scenario:<10} {seed:>5} {want:>9} {got:>8}  "
                  f"{'OK' if good else 'MISMATCH'}")

    # A scenario with no generated file must raise, not quietly load the legacy trace.
    print()
    try:
        ep.reset("NO_SUCH_SCENARIO", 42)
        print("[FAIL] an unresolvable scenario was accepted — silent fallback is possible")
        ok = False
    except Py4JJavaError as e:
        msg = str(e)
        good = "not readable" in msg or "IllegalStateException" in msg
        ok &= good
        print(f"[{'OK' if good else 'FAIL'}] unresolvable scenario raises "
              f"({msg.splitlines()[0][:90]}...)")

    print()
    print("TRACE PATTERN VERIFICATION:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
