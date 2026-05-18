#!/usr/bin/env python3
"""T8.10 - Phase 1.8 closing comparison: 6 schedulers x 5 metrics.

Reads ``data/results/baseline-HIGH/{policy}/summary.json`` for each of:
  FirstFit, BestFit, RoundRobin, K8s, Random, ppo-min

Builds a single JSON + a single PDF figure suitable for the report's
"Phase 1 closing comparison" section.

Metrics surfaced (columns of the table):
  - total_energy_kwh
  - makespan_sec
  - sla_violation_rate
  - avg_cpu_utilization
  - total_wakeups

NOTE on neutrality:
  Each row is read straight from Java's ``summary.json`` - the numbers
  are NOT massaged. If a row is missing (e.g. ppo-min never trained),
  it shows up as ``"--"`` in the printed table and the JSON omits it.
  We do NOT fabricate values for absent runs.

Usage::

    python scripts/phase1-comparison.py [--scenario HIGH]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np


# Ordering chosen to read "energy-cheapest -> energy-priciest" in the
# baseline empirics (HIGH scenario, Phase 1.8 energy model).
SCHEDULER_ORDER = ["bestfit", "firstfit", "ppo-min",
                   "random", "k8s", "roundrobin"]

METRIC_COLS = [
    ("total_energy_kwh",      "Energy (kWh)",    "{:>12.2f}"),
    ("makespan_sec",          "Makespan (s)",    "{:>12.0f}"),
    ("sla_violation_rate",    "SLA viol. rate",  "{:>14.4f}"),
    ("avg_cpu_utilization",   "Avg CPU util",    "{:>12.4f}"),
    ("total_wakeups",         "# Wakeups",       "{:>10d}"),
]


def _load_summary(path: Path) -> Optional[dict]:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except Exception as e:
        print(f"[phase1-cmp] failed to load {path}: {e}", file=sys.stderr)
        return None


def _format_cell(value, fmt: str) -> str:
    if value is None:
        return "--"
    try:
        return fmt.format(value).strip()
    except (TypeError, ValueError):
        return str(value)


def build_table(scenario: str, results_root: Path) -> dict:
    """Read 6 summaries, return a structured dict + print the table."""
    base = results_root / f"baseline-{scenario}"

    rows: dict[str, dict] = {}
    for sched in SCHEDULER_ORDER:
        rows[sched] = _load_summary(base / sched / "summary.json") or {}

    # -- Pretty-print --------------------------------------------------
    header = ["Scheduler"] + [label for _, label, _ in METRIC_COLS]
    widths = [12] + [16] * len(METRIC_COLS)

    print("\nPhase 1.8 closing comparison - scenario", scenario)
    print("=" * sum(widths))
    print("  ".join(f"{h:<{w}}" for h, w in zip(header, widths)))
    print("-" * sum(widths))
    for sched in SCHEDULER_ORDER:
        row = rows[sched]
        if not row:
            cells = [sched] + ["--"] * len(METRIC_COLS)
        else:
            cells = [sched] + [
                _format_cell(row.get(key), fmt)
                for key, _label, fmt in METRIC_COLS
            ]
        print("  ".join(f"{c:<{w}}" for c, w in zip(cells, widths)))
    print("=" * sum(widths))

    # -- Skeptical sanity check ----------------------------------------
    energies = [
        rows[s].get("total_energy_kwh") for s in SCHEDULER_ORDER
        if rows[s].get("total_energy_kwh") is not None
    ]
    if len(energies) >= 2:
        gap = (max(energies) - min(energies)) / max(energies) * 100
        print(f"\nEnergy spread across {len(energies)} schedulers: "
              f"{min(energies):.2f} -> {max(energies):.2f} kWh "
              f"(delta = {gap:.2f}%)")
        if gap < 5:
            print("  WARNING: spread < 5% - state machine may not be "
                  "delivering the expected gradient. Re-examine.")
    return {"scenario": scenario, "schedulers": rows}


def plot_pareto(scenario: str, rows: dict, out_pdf: Path) -> None:
    """One scatter: energy on x, SLA reward on y. Each point is a scheduler.

    Reads ``total_sla_reward`` from ``baseline_results.json`` when present
    (some schedulers report it there but not in their own summary.json).
    """
    base = Path("data/results") / f"baseline-{scenario}"
    eval_json = base / "baseline_results.json"

    sla_lookup: dict[str, float] = {}
    if eval_json.exists():
        try:
            data = json.loads(eval_json.read_text())
            for name, body in data.items():
                if isinstance(body, dict) and "total_sla_reward" in body:
                    sla_lookup[name] = float(body["total_sla_reward"])
        except Exception:
            pass

    fig, ax = plt.subplots(figsize=(8, 6))
    palette = {
        "bestfit":    "#1f77b4",
        "firstfit":   "#aec7e8",
        "ppo-min":    "#d62728",
        "random":     "#7f7f7f",
        "k8s":        "#2ca02c",
        "roundrobin": "#bcbd22",
    }

    plotted = 0
    for sched in SCHEDULER_ORDER:
        summary = rows.get(sched) or {}
        energy = summary.get("total_energy_kwh")
        # SLA: prefer the eval JSON's reward sum; fall back to violation
        # count (negative because lower = better -> flip sign).
        sla_r = sla_lookup.get(sched)
        if energy is None or sla_r is None:
            continue
        ax.scatter(energy, sla_r, s=140, color=palette.get(sched, "k"),
                   edgecolor="black", linewidth=0.7, label=sched)
        ax.annotate(sched, (energy, sla_r), textcoords="offset points",
                    xytext=(7, 5), fontsize=9)
        plotted += 1

    ax.set_xlabel("Total energy (kWh) - lower is better")
    ax.set_ylabel("R_sla cumulative - higher (less negative) is better")
    ax.set_title(f"Phase 1.8 Pareto view - scenario {scenario}\n"
                 f"({plotted}/{len(SCHEDULER_ORDER)} schedulers with both axes)")
    ax.grid(alpha=0.3)
    if plotted > 0:
        ax.legend(loc="best", fontsize=9)

    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_pdf, bbox_inches="tight")
    print(f"\n[phase1-cmp] figure -> {out_pdf}")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--scenario", default="HIGH", choices=["LOW", "HIGH", "BURST"])
    p.add_argument("--results-root", default="data/results", type=Path)
    p.add_argument("--out-json", default=None, type=Path)
    p.add_argument("--out-pdf", default=None, type=Path)
    args = p.parse_args()

    rows = build_table(args.scenario, args.results_root.resolve())

    out_json = args.out_json or Path(
        f"data/results/phase1-comparison/summary-{args.scenario}.json"
    )
    out_pdf = args.out_pdf or Path(
        f"assets/report/figures/phase1-comparison-{args.scenario}.pdf"
    )

    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(rows, indent=2))
    print(f"\n[phase1-cmp] JSON -> {out_json}")

    plot_pareto(args.scenario, rows["schedulers"], out_pdf.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
