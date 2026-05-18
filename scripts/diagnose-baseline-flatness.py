#!/usr/bin/env python3
"""T8.0 — Diagnose the energy-gradient flatness of the original (always-on,
linear-power) baseline and contrast it with the new state-machine model.

The script reads ``data/results/baseline-{SCENARIO}-flat/{policy}/metrics.csv``
(the pre-fix archive) and ``data/results/baseline-{SCENARIO}/{policy}/metrics.csv``
(post-fix), and produces a multi-panel figure showing:

1. Cumulative total_energy_kwh over sim-time for K8s vs Random — flat
   parallel lines pre-fix, divergent lines post-fix.
2. Per-host CPU utilisation timeline for K8s vs Random (post-fix only;
   pre-fix snapshots did not record per-host data) — visualises the
   "spread vs accidental-pack" story directly.
3. Host-state distribution over time (post-fix) — how many hosts are
   ACTIVE / IDLE / SUSPENDED at each snapshot.

Output: ``assets/report/figures/diagnostic-baseline-flatness.pdf``.

Usage (host)::

    python scripts/diagnose-baseline-flatness.py [--scenario HIGH]

Why this script exists: the original report claimed K8s ≈ Random because
of the "always-on linear power" assumption. With the state machine in
place (Phase 1.8 Block A), the two policies now diverge by ≥5 % across
all three scenarios. This script generates the evidence figure for the
report's "Energy-Gradient Diagnostic" subsection.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# State code → human label + colour. Matches SimulationManager.HostState codes.
STATE_LABEL = {0: "SUSPENDED", 1: "IDLE", 2: "ACTIVE"}
STATE_COLOR = {0: "#cccccc", 1: "#ffcc66", 2: "#1f77b4"}

POLICIES = ["k8s", "random"]


def _load_csv(path: Path) -> Optional[pd.DataFrame]:
    if not path.exists():
        print(f"[diag] missing: {path}", file=sys.stderr)
        return None
    return pd.read_csv(path)


def _summary(scenario_dir: Path, policy: str) -> Optional[dict]:
    path = scenario_dir / policy / "summary.json"
    if not path.exists():
        return None
    return json.loads(path.read_text())


def _diff_pct(a: float, b: float) -> float:
    """Symmetric % difference: 100·|a−b| / mean(a, b)."""
    if a + b == 0:
        return 0.0
    return 100.0 * abs(a - b) / ((a + b) / 2.0)


def diagnose(scenario: str, results_root: Path, out_pdf: Path) -> int:
    flat_dir = results_root / f"baseline-{scenario}-flat"
    new_dir  = results_root / f"baseline-{scenario}"

    print(f"[diag] scenario={scenario}")
    print(f"[diag] flat archive: {flat_dir}")
    print(f"[diag] post-fix:     {new_dir}")

    # Load CSVs
    flat = {p: _load_csv(flat_dir / p / "metrics.csv") for p in POLICIES}
    new  = {p: _load_csv(new_dir  / p / "metrics.csv") for p in POLICIES}

    if not all(df is not None for df in new.values()):
        print("[diag] post-fix metrics missing — aborting.", file=sys.stderr)
        return 1

    fig = plt.figure(figsize=(11, 13))
    gs = fig.add_gridspec(4, 2, hspace=0.55, wspace=0.25)

    # ── Panel 1: cumulative energy over sim-time (FLAT vs NEW) ────────────
    ax1 = fig.add_subplot(gs[0, :])
    for p, df in flat.items():
        if df is None:
            continue
        ax1.plot(df["timestamp"] / 86400.0, df["total_energy_kwh"],
                 linestyle="--", alpha=0.55,
                 label=f"{p} (flat — pre-state-machine)")
    for p, df in new.items():
        ax1.plot(df["timestamp"] / 86400.0, df["total_energy_kwh"],
                 linewidth=1.6, label=f"{p} (post-state-machine)")
    ax1.set_xlabel("Sim-time (days)")
    ax1.set_ylabel("Cumulative energy (kWh)")
    ax1.set_title(
        f"Energy-gradient diagnostic — scenario {scenario}\n"
        "FLAT lines (dashed): K8s vs Random virtually identical · "
        "POST-FIX (solid): visible separation"
    )
    ax1.grid(alpha=0.3)
    ax1.legend(loc="lower right", fontsize=9)

    # ── Panel 2: end-of-sim energy bar chart with % diff ──────────────────
    ax2 = fig.add_subplot(gs[1, 0])
    labels, flat_vals, new_vals = [], [], []
    for p in POLICIES:
        s_flat = _summary(flat_dir, p)
        s_new  = _summary(new_dir,  p)
        if s_flat and s_new:
            labels.append(p)
            flat_vals.append(s_flat["total_energy_kwh"])
            new_vals.append(s_new["total_energy_kwh"])
    x = np.arange(len(labels))
    w = 0.38
    ax2.bar(x - w/2, flat_vals, w, label="flat model", color="#aaaaaa")
    ax2.bar(x + w/2, new_vals,  w, label="state-machine", color="#1f77b4")
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels)
    ax2.set_ylabel("Total energy (kWh)")
    flat_gap = _diff_pct(*flat_vals) if len(flat_vals) == 2 else 0
    new_gap  = _diff_pct(*new_vals)  if len(new_vals)  == 2 else 0
    ax2.set_title(f"End-of-sim energy\nK8s↔Random gap: "
                  f"flat {flat_gap:.2f}% → new {new_gap:.2f}%")
    ax2.grid(alpha=0.3, axis="y")
    ax2.legend(fontsize=9)

    # ── Panel 3: host-state stack over time (NEW) ─────────────────────────
    ax3 = fig.add_subplot(gs[1, 1])
    ref = new["k8s"]
    if {"active_hosts", "idle_hosts", "suspended_hosts"} <= set(ref.columns):
        t_days = ref["timestamp"] / 86400.0
        ax3.stackplot(
            t_days,
            ref["suspended_hosts"], ref["idle_hosts"], ref["active_hosts"],
            labels=["SUSPENDED", "IDLE", "ACTIVE"],
            colors=[STATE_COLOR[0], STATE_COLOR[1], STATE_COLOR[2]],
            alpha=0.85,
        )
        ax3.set_xlabel("Sim-time (days)")
        ax3.set_ylabel("# hosts")
        ax3.set_title("Host-state mix over time (K8s)")
        ax3.legend(loc="lower right", fontsize=9)
        ax3.grid(alpha=0.3)

    # ── Panel 4 & 5: per-host CPU util timelines (K8s + Random) ───────────
    for col, p in enumerate(POLICIES):
        ax = fig.add_subplot(gs[2, col])
        df = new[p]
        host_cols = sorted(
            [c for c in df.columns if c.endswith("_cpu_util") and c.startswith("h")],
            key=lambda c: int(c[1:].split("_")[0]),
        )
        if host_cols:
            t_days = df["timestamp"] / 86400.0
            for c in host_cols:
                ax.plot(t_days, df[c], linewidth=0.6, alpha=0.6,
                        label=c.split("_")[0])
            ax.set_xlabel("Sim-time (days)")
            ax.set_ylabel("Per-host CPU util")
            ax.set_title(f"Per-host CPU utilisation — {p}")
            ax.set_ylim(-0.05, 1.05)
            ax.grid(alpha=0.3)
            if len(host_cols) <= 12:
                ax.legend(fontsize=6, ncol=2, loc="upper right")

    # ── Panel 6 & 7: host-state heat-strip (K8s + Random) ────────────────
    for col, p in enumerate(POLICIES):
        ax = fig.add_subplot(gs[3, col])
        df = new[p]
        state_cols = sorted(
            [c for c in df.columns if c.endswith("_state") and c.startswith("h")],
            key=lambda c: int(c[1:].split("_")[0]),
        )
        if state_cols:
            mat = df[state_cols].to_numpy().T  # hosts × snapshots
            # Build a custom colour map matching STATE_COLOR.
            from matplotlib.colors import ListedColormap
            cmap = ListedColormap([STATE_COLOR[0], STATE_COLOR[1], STATE_COLOR[2]])
            im = ax.imshow(
                mat, aspect="auto", origin="lower", cmap=cmap,
                vmin=-0.5, vmax=2.5,
                extent=[df["timestamp"].iloc[0] / 86400.0,
                        df["timestamp"].iloc[-1] / 86400.0,
                        -0.5, mat.shape[0] - 0.5],
                interpolation="nearest",
            )
            ax.set_xlabel("Sim-time (days)")
            ax.set_ylabel("Host #")
            ax.set_yticks(range(mat.shape[0]))
            ax.set_title(f"Host-state timeline — {p}")
            # Colour-bar legend
            cbar = fig.colorbar(im, ax=ax, ticks=[0, 1, 2], shrink=0.6)
            cbar.set_ticklabels([STATE_LABEL[0], STATE_LABEL[1], STATE_LABEL[2]])

    # ── Bottom annotation: numeric summary ────────────────────────────────
    text_lines = ["Numeric summary (kWh)\n"]
    for p in POLICIES:
        s_flat = _summary(flat_dir, p)
        s_new  = _summary(new_dir,  p)
        if s_flat and s_new:
            text_lines.append(
                f"  {p:6s}  flat={s_flat['total_energy_kwh']:9.2f}  "
                f"new={s_new['total_energy_kwh']:9.2f}  "
                f"Δ={s_new['total_energy_kwh']-s_flat['total_energy_kwh']:+8.2f}  "
                f"wakeups={s_new.get('total_wakeups', '–')}"
            )
    fig.suptitle(
        f"Phase 1.8 / T8.0 — Energy gradient diagnostic ({scenario})",
        fontsize=13, y=0.995,
    )
    fig.text(0.02, 0.005, "\n".join(text_lines), family="monospace",
             fontsize=8, va="bottom")

    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_pdf, bbox_inches="tight")
    print(f"[diag] figure -> {out_pdf}")
    return 0


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--scenario", default="HIGH", choices=["LOW", "HIGH", "BURST"])
    p.add_argument("--results-root", default="data/results", type=Path)
    p.add_argument("--out", default=None, type=Path,
                   help="output PDF path "
                        "(default: assets/report/figures/diagnostic-baseline-flatness.pdf)")
    args = p.parse_args()

    out = args.out or Path("assets/report/figures/diagnostic-baseline-flatness.pdf")
    return diagnose(args.scenario, args.results_root.resolve(), out.resolve())


if __name__ == "__main__":
    raise SystemExit(main())
