"""Skeptical verification of PPO-min result.

Run after T8.9 training completes. Checks:
  1. Pareto dominance over Random (genuine improvement?)
  2. Hosts visited diversity (not degenerate "always host 0")
  3. State machine signature consistent across episode (not stuck in 1 phase)

Outputs PASS / WARN / FAIL with reasoning. Designed to surface unexpected
behaviour rather than confirm a desired narrative.
"""
from __future__ import annotations
import json
import sys
from pathlib import Path

import pandas as pd

HIGH_DIR = Path("data/results/baseline-HIGH")


def main() -> int:
    results = json.loads((HIGH_DIR / "baseline_results.json").read_text())
    ppo = results.get("ppo-min")
    rnd = results.get("random")
    if not ppo or not rnd:
        print("[verify] Missing ppo-min or random entry in baseline_results.json")
        return 1

    print("=" * 60)
    print("Skeptical audit of PPO-min @ 100k steps (HIGH scenario)")
    print("=" * 60)

    # ── Check 1: Pareto dominance over Random ──────────────────────────
    e_ppo, e_rnd = ppo["total_energy_kwh"], rnd["total_energy_kwh"]
    sla_ppo, sla_rnd = ppo["total_sla_reward"], rnd["total_sla_reward"]
    e_better = e_ppo < e_rnd
    sla_better = sla_ppo > sla_rnd
    print(f"\n[1] Pareto dominance over Random")
    print(f"    Energy: PPO {e_ppo:.0f} vs Random {e_rnd:.0f} "
          f"({'PPO LOWER' if e_better else 'PPO HIGHER'})")
    print(f"    R_sla:  PPO {sla_ppo:.0f} vs Random {sla_rnd:.0f} "
          f"({'PPO BETTER' if sla_better else 'PPO WORSE'})")
    if e_better and sla_better:
        print("    -> PASS: PPO-min Pareto-dominates Random on both axes")
    elif e_better or sla_better:
        print("    -> NEUTRAL: PPO-min trades one axis for another vs Random")
    else:
        print("    -> FAIL: PPO-min dominated BY Random — training failed")
        return 2

    # ── Check 2: Host diversity from metrics.csv ───────────────────────
    csv = HIGH_DIR / "ppo-min" / "metrics.csv"
    if csv.exists():
        df = pd.read_csv(csv)
        host_cpu_cols = [c for c in df.columns if c.endswith("_cpu_util")
                         and c.startswith("h")]
        peak_util = {c: df[c].max() for c in host_cpu_cols}
        avg_util = {c: df[c].mean() for c in host_cpu_cols}
        max_peak = max(peak_util.values())
        min_peak = min(peak_util.values())
        print(f"\n[2] Host utilisation diversity (PPO-min episode)")
        print(f"    Per-host peak util range: [{min_peak:.3f}, {max_peak:.3f}]")
        print(f"    Per-host avg util range: "
              f"[{min(avg_util.values()):.3f}, {max(avg_util.values()):.3f}]")
        zero_peak = sum(1 for v in peak_util.values() if v == 0)
        print(f"    Hosts never used: {zero_peak}/{len(peak_util)}")
        if zero_peak >= len(peak_util) - 1:
            print("    -> FAIL: only 1 host ever used — policy is degenerate")
            return 3
        elif zero_peak > 0:
            print(f"    -> WARN: {zero_peak} hosts unused, policy biased")
        else:
            print("    -> PASS: every host received at least one task")

    # ── Check 3: state-machine signature ───────────────────────────────
    summary = json.loads((HIGH_DIR / "ppo-min" / "summary.json").read_text())
    avg_active = summary.get("avg_active_hosts", 0)
    avg_susp = summary.get("avg_suspended_hosts", 0)
    avg_idle = summary.get("avg_idle_hosts", 0)
    wakeups = summary.get("total_wakeups", 0)
    print(f"\n[3] State-machine signature")
    print(f"    avg_active_hosts:    {avg_active:.4f}")
    print(f"    avg_idle_hosts:      {avg_idle:.4f}")
    print(f"    avg_suspended_hosts: {avg_susp:.4f}")
    print(f"    total_wakeups:       {wakeups}")
    if abs(avg_active - 10) < 0.05 and wakeups < 5:
        print("    -> WARN: behaves like full-spread (similar to K8s)")
    elif wakeups > 200 and avg_susp > 0.1:
        print("    -> WARN: behaves like pack-tight (similar to BestFit)")
    else:
        print(f"    -> NEUTRAL: middle-of-the-road behaviour "
              f"(wakeups {wakeups} between K8s {9} and Random {147})")

    # ── Cross-reference with all 5 baselines ───────────────────────────
    print(f"\n[4] Pareto positioning vs all 5 baselines")
    rows = []
    for name in ["bestfit", "firstfit", "ppo-min", "random", "k8s", "roundrobin"]:
        s = json.loads((HIGH_DIR / name / "summary.json").read_text())
        r = results.get(name, {})
        rows.append((name, s["total_energy_kwh"], r.get("total_sla_reward", 0),
                     s.get("total_wakeups", 0)))
    rows.sort(key=lambda r: r[1])  # sort by energy ascending
    print(f"    {'rank':<6}{'sched':<12}{'energy kWh':>14}{'R_sla':>20}{'wake':>8}")
    for i, (name, e, sla, w) in enumerate(rows, 1):
        marker = "  <--" if name == "ppo-min" else ""
        print(f"    {i:<6}{name:<12}{e:>14.2f}{sla:>20.0f}{w:>8}{marker}")

    # Check: does PPO-min strictly dominate ANY baseline?
    dominates = []
    for name, e, sla, _w in rows:
        if name == "ppo-min":
            continue
        if e_ppo < e and sla_ppo > sla:
            dominates.append(name)
    if dominates:
        print(f"\n    PPO-min strictly dominates: {dominates}")
        print("    -> PASS: at least one baseline is Pareto-dominated")
    else:
        print(f"\n    -> WARN: PPO-min does NOT strictly dominate any baseline")

    print("\n" + "=" * 60)
    print("Audit complete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
