#!/usr/bin/env python3
"""
collect_all_data.py — One-command data collection and figure generation for ELDAS.

Run from the ELDAS root directory:
    python assets/report/scripts/collect_all_data.py

Prerequisites:
    pip install pandas numpy matplotlib          # host-side deps (one-time)
    docker compose up -d                         # both containers must be healthy
    bash scripts/download-trace.sh               # trace must exist

What it does:
    1. Smoke test       (LOW, 3000 steps) — verifies the full RL loop
    2. Baseline runs    (3 scenarios × 2 policies) — exports metrics.csv + summary.json
    3. Trace analysis   — computes Table 5.5 stats + QoS/GPU distributions
    4. Figure generation — saves PDF figures to assets/report/figures/
    5. Number printout  — prints all \\TODOnum replacement values for Chapter 5

Flags:
    --skip-docker    Skip steps 1-2; reuse whatever is in data/results/
    --skip-smoke     Skip step 1 only (run baselines but not smoke test)
    --seed N         Override random seed (default: 42)
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

# ── Paths (all relative to ELDAS root, resolved at script load time) ──────────

ROOT    = Path(__file__).resolve().parent.parent.parent.parent   # ELDAS/
DATA    = ROOT / "data"
TRACE   = DATA / "alibaba-trace" / "openb_pod_list_default.csv"
RESULTS = DATA / "results"
FIGURES = ROOT / "assets" / "report" / "figures"

SCENARIOS = ["LOW", "HIGH", "BURST"]
POLICIES  = ["k8s", "random"]


# ── Dependency check ──────────────────────────────────────────────────────────

def _check_deps() -> None:
    missing = []
    for pkg in ("pandas", "numpy", "matplotlib"):
        try:
            __import__(pkg)
        except ImportError:
            missing.append(pkg)
    if missing:
        print(f"[collect] Missing Python packages: {', '.join(missing)}")
        print(f"[collect] Fix:  pip install {' '.join(missing)}")
        sys.exit(1)


# ── Docker helpers ────────────────────────────────────────────────────────────

def _run(cmd: str, label: str) -> None:
    print(f"\n{'─'*60}")
    print(f"[collect] {label}")
    print(f"  $ {cmd}")
    print(f"{'─'*60}")
    result = subprocess.run(cmd, shell=True, cwd=ROOT)
    if result.returncode != 0:
        print(f"[collect] FAILED (exit {result.returncode}): {label}")
        sys.exit(1)


def run_smoke(seed: int) -> None:
    _run(
        f"docker compose run --rm rl-agent "
        f"python src/smoke_test.py --scenario LOW --max-steps 3000 "
        f"--output /data/results/smoke",
        "Step 1 — Smoke test (LOW, 3000 steps)",
    )


def run_baselines(seed: int) -> None:
    for scenario in SCENARIOS:
        _run(
            f"docker compose run --rm rl-agent "
            f"python src/baseline_eval.py "
            f"--scenario {scenario} --seed {seed} "
            f"--output /data/results",
            f"Step 2 — Baseline: {scenario}",
        )


# ── Baseline summary loader ───────────────────────────────────────────────────

def _load_summary(scenario: str, policy: str) -> dict:
    """Try per-policy summary.json first, fallback to baseline_results.json."""
    p1 = RESULTS / f"baseline-{scenario}" / policy / "summary.json"
    if p1.exists():
        with open(p1) as f:
            return json.load(f)

    p2 = RESULTS / f"baseline-{scenario}" / "baseline_results.json"
    if p2.exists():
        with open(p2) as f:
            data = json.load(f)
        return data.get(policy, {})

    return {}


# ── Step 3: Trace analysis ────────────────────────────────────────────────────

def analyze_trace() -> tuple[object, dict]:
    """Return (DataFrame, stats_dict)."""
    import pandas as pd

    print(f"\n[collect] Step 3 — Analyzing trace: {TRACE.name}")
    df = pd.read_csv(TRACE)
    df = df[df["pod_phase"] != "Pending"].copy()
    df["duration"] = (
        df["deletion_time"].fillna(0) - df["scheduled_time"].clip(lower=0)
    ).clip(lower=0)

    stats: dict = {}
    for col in ["cpu_milli", "memory_mib", "num_gpu", "duration"]:
        s = df[col]
        stats[col] = {
            "min":    float(s.min()),
            "p25":    float(s.quantile(0.25)),
            "median": float(s.median()),
            "mean":   float(s.mean()),
            "p75":    float(s.quantile(0.75)),
            "max":    float(s.max()),
        }

    stats["total_tasks"] = len(df)
    stats["qos"]         = (df["qos"].value_counts(normalize=True) * 100).to_dict()
    stats["gpu_dist"]    = (
        df["num_gpu"].value_counts(normalize=True).sort_index() * 100
    ).to_dict()

    print(f"[collect] Trace: {len(df):,} tasks after Pending filter")
    return df, stats


# ── Step 4: Figure generation ─────────────────────────────────────────────────

def generate_figures(trace_df: object) -> None:
    import numpy as np
    import matplotlib
    matplotlib.use("Agg")          # headless — no display required
    import matplotlib.pyplot as plt
    import pandas as pd

    FIGURES.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.size": 11, "font.family": "DejaVu Sans"})

    x = np.arange(len(SCENARIOS))
    w = 0.35

    # ── Fig 5.3: Total energy bar chart ───────────────────────────────────────
    e_k8s    = [_load_summary(s, "k8s").get("total_energy_kwh", 0)    for s in SCENARIOS]
    e_random = [_load_summary(s, "random").get("total_energy_kwh", 0) for s in SCENARIOS]

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(x - w/2, e_k8s,    w, label="K8s LRP", color="#1f77b4")
    ax.bar(x + w/2, e_random, w, label="Random",   color="#ff7f0e")
    ax.set_xticks(x); ax.set_xticklabels(SCENARIOS)
    ax.set_ylabel("Total energy (kWh)")
    ax.set_title("Baseline energy comparison")
    ax.legend(); plt.tight_layout()
    plt.savefig(FIGURES / "baseline-energy.pdf", bbox_inches="tight"); plt.close()
    print(f"[collect] Saved baseline-energy.pdf")

    # ── Fig 5.4: SLA violation rate bar chart ─────────────────────────────────
    sla_k8s    = [_load_summary(s, "k8s").get("sla_violation_rate", 0) * 100    for s in SCENARIOS]
    sla_random = [_load_summary(s, "random").get("sla_violation_rate", 0) * 100 for s in SCENARIOS]

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(x - w/2, sla_k8s,    w, label="K8s LRP", color="#1f77b4")
    ax.bar(x + w/2, sla_random, w, label="Random",   color="#ff7f0e")
    ax.set_xticks(x); ax.set_xticklabels(SCENARIOS)
    ax.set_ylabel("SLA violation rate (%)")
    ax.set_title("Baseline SLA comparison")
    ax.legend(); plt.tight_layout()
    plt.savefig(FIGURES / "baseline-sla.pdf", bbox_inches="tight"); plt.close()
    print(f"[collect] Saved baseline-sla.pdf")

    # ── Fig 5.5: Cumulative energy time-series ────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(14, 4), sharey=True)
    any_csv = False
    for ax, s in zip(axes, SCENARIOS):
        for policy, color in [("k8s", "#1f77b4"), ("random", "#ff7f0e")]:
            csv_p = RESULTS / f"baseline-{s}" / policy / "metrics.csv"
            if csv_p.exists():
                df_m = pd.read_csv(csv_p)
                ax.plot(df_m["timestamp"], df_m["total_energy_kwh"],
                        label=policy.upper(), color=color, linewidth=1.5)
                any_csv = True
        ax.set_title(f"Scenario: {s}")
        ax.set_xlabel("Sim time (s)")
        ax.legend()
    axes[0].set_ylabel("Cumulative energy (kWh)")
    plt.tight_layout()
    if any_csv:
        plt.savefig(FIGURES / "baseline-energy-time.pdf", bbox_inches="tight")
        print(f"[collect] Saved baseline-energy-time.pdf")
    else:
        print(f"[collect] SKIP baseline-energy-time.pdf — metrics.csv not found")
    plt.close()

    # ── Fig 5.7: num_gpu histogram ────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(6, 3.5))
    trace_df["num_gpu"].value_counts().sort_index().plot(
        kind="bar", ax=ax, color="#2ca02c"
    )
    ax.set_yscale("log")
    ax.set_xlabel("num_gpu"); ax.set_ylabel("Tasks (log scale)")
    ax.set_title("GPU demand distribution")
    plt.tight_layout()
    plt.savefig(FIGURES / "trace-numgpu-hist.pdf", bbox_inches="tight"); plt.close()
    print(f"[collect] Saved trace-numgpu-hist.pdf")

    # ── Fig 5.8: QoS pie ──────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(5, 5))
    trace_df["qos"].value_counts().plot(
        kind="pie", ax=ax, autopct="%1.1f%%",
        colors=["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"],
    )
    ax.set_ylabel(""); ax.set_title("Task QoS distribution")
    plt.tight_layout()
    plt.savefig(FIGURES / "trace-qos-pie.pdf", bbox_inches="tight"); plt.close()
    print(f"[collect] Saved trace-qos-pie.pdf")


# ── Step 5: Print TODOnum replacements ────────────────────────────────────────

def print_numbers(stats: dict) -> None:
    SEP = "=" * 70

    print(f"\n{SEP}")
    print("  \\TODOnum REPLACEMENTS — copy into 05-ket-qua.tex")
    print(SEP)

    # Table 5.5 — trace stats
    print("\n── TABLE 5.5 (tab:trace-stats) ─────────────────────────────────")
    for col in ["cpu_milli", "memory_mib", "num_gpu", "duration"]:
        s = stats[col]
        print(f"  {col:<14}: min={s['min']:.1f}  p25={s['p25']:.1f}  "
              f"median={s['median']:.1f}  mean={s['mean']:.1f}  "
              f"p75={s['p75']:.1f}  max={s['max']:.1f}")
    print(f"  Total tasks (non-Pending): {stats['total_tasks']:,}")

    # QoS breakdown
    print("\n── QoS distribution (Fig 5.8) ───────────────────────────────────")
    for qos, pct in sorted(stats["qos"].items(), key=lambda kv: -kv[1]):
        print(f"  {qos}: {pct:.1f}%")

    # GPU demand
    print("\n── GPU demand (Fig 5.7) ─────────────────────────────────────────")
    for gpus, pct in sorted(stats["gpu_dist"].items()):
        print(f"  num_gpu={gpus}: {pct:.1f}%")

    # Table 5.3 — baseline results
    print("\n── TABLE 5.3 (tab:baseline-results) ────────────────────────────")
    hdr = f"  {'Scenario':<7} {'Policy':<8} {'Energy kWh':<13} {'SLA viol':<10} {'Viol%':<8} {'CPU util'}"
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))
    for scen in SCENARIOS:
        for policy in POLICIES:
            s = _load_summary(scen, policy)
            if s:
                print(f"  {scen:<7} {policy:<8} "
                      f"{s.get('total_energy_kwh', 0):<13.4f} "
                      f"{s.get('sla_violations', 0):<10} "
                      f"{s.get('sla_violation_rate', 0) * 100:<8.2f} "
                      f"{s.get('avg_cpu_utilization', 0):.4f}")
            else:
                print(f"  {scen:<7} {policy:<8} [no data — run baselines first]")

    # Ratios for §5.3 narrative
    print("\n── Ratios for §5.3 narrative ────────────────────────────────────")
    for scen in SCENARIOS:
        k = _load_summary(scen, "k8s")
        r = _load_summary(scen, "random")
        if not (k and r):
            continue
        e_k, e_r = k.get("total_energy_kwh", 0), r.get("total_energy_kwh", 0)
        v_k, v_r = k.get("sla_violation_rate", 1e-9), r.get("sla_violation_rate", 1e-9)
        if e_r > 0:
            ratio_e   = (1 - e_k / e_r) * 100
            direction = "less" if ratio_e > 0 else "more"
            ratio_sla = v_r / max(v_k, 1e-9)
            print(f"  {scen}: K8s uses {abs(ratio_e):.1f}% {direction} energy; "
                  f"SLA ratio Random/K8s = {ratio_sla:.2f}×")

    # Per-scenario task stats (Table tab:scenario-stats, lines 355-357)
    # Derived from baseline summary.json (total_tasks) for each scenario.
    # CPU mean/GPU mean approximated from trace data filtered to scenario size.
    print("\n── TABLE tab:scenario-stats (lines 355-357) ─────────────────────")
    print(f"  {'Scenario':<8} {'#Tasks':<8} {'Energy(kWh)':<12} {'SimTime(s)':<12}")
    for scen in SCENARIOS:
        s = _load_summary(scen, "k8s")
        if s:
            print(f"  {scen:<8} {s.get('total_tasks', 0):<8} "
                  f"{s.get('total_energy_kwh', 0):<12.4f} "
                  f"{s.get('makespan_sec', 0):<12.1f}")

    # Smoke test summary
    smoke_p = RESULTS / "smoke" / "summary.json"
    if smoke_p.exists():
        with open(smoke_p) as f:
            smoke = json.load(f)
        print("\n── TABLE 5.2 (tab:smoke-results) ───────────────────────────────")
        for k, v in smoke.items():
            print(f"  {k}: {v}")

    print(f"\n{SEP}\n")


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="ELDAS: collect all data + generate figures")
    parser.add_argument("--skip-docker", action="store_true",
                        help="Skip Docker simulation steps; reuse existing data/results/")
    parser.add_argument("--skip-smoke", action="store_true",
                        help="Skip smoke test (run baselines only)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    args = parser.parse_args()

    _check_deps()

    print(f"\n{'='*60}")
    print(f"  ELDAS collect_all_data.py")
    print(f"  Root:    {ROOT}")
    print(f"  Results: {RESULTS}")
    print(f"  Figures: {FIGURES}")
    print(f"{'='*60}")

    if not TRACE.exists():
        print(f"\n[collect] ERROR: Trace file not found: {TRACE}")
        print("[collect] Fix:  bash scripts/download-trace.sh")
        sys.exit(1)

    RESULTS.mkdir(parents=True, exist_ok=True)

    if not args.skip_docker:
        if not args.skip_smoke:
            run_smoke(args.seed)
        run_baselines(args.seed)
    else:
        print("\n[collect] --skip-docker: reusing existing results in data/results/")

    trace_df, stats = analyze_trace()

    print("\n[collect] Step 4 — Generating figures...")
    generate_figures(trace_df)

    print_numbers(stats)

    print("[collect] Done. Next steps:")
    print("  1. Copy the numbers above into \\TODOnum{...} in 05-ket-qua.tex")
    print("  2. Take screenshots: docker compose ps, smoke test terminal")
    print("  3. Run pdflatex to rebuild the PDF")


if __name__ == "__main__":
    main()
