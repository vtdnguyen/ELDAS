import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np

results = Path("../../../data/results")
figures = Path("../figures")
figures.mkdir(exist_ok=True)
plt.rcParams["font.size"] = 11

# ── Fig 5.3: bar energy ───────────────────────────────────────────
scens = ["LOW", "HIGH", "BURST"]
fig, ax = plt.subplots(figsize=(7, 4))
x = np.arange(len(scens))
w = 0.35
e_k8s    = [pd.read_json(results / f"baseline-{s}/baseline_results.json")["k8s"]["total_energy_kwh"]    for s in scens]
e_random = [pd.read_json(results / f"baseline-{s}/baseline_results.json")["random"]["total_energy_kwh"] for s in scens]
ax.bar(x - w/2, e_k8s,    w, label="K8s LRP",  color="#1f77b4")
ax.bar(x + w/2, e_random, w, label="Random",    color="#ff7f0e")
ax.set_xticks(x); ax.set_xticklabels(scens)
ax.set_ylabel("Total energy (kWh)")
ax.legend(); plt.tight_layout()
plt.savefig(figures / "baseline-energy.pdf"); plt.close()

# ── Fig 5.4: bar SLA rate (similar logic) ─────────────────────────
# ── Fig 5.5: cumulative energy time-series ────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(14, 4), sharey=True)
for ax, s in zip(axes, scens):
    for policy, color in [("k8s", "#1f77b4"), ("random", "#ff7f0e")]:
        df = pd.read_csv(results / f"baseline-{s}/{policy}/metrics.csv")
        ax.plot(df["timestamp"], df["total_energy_kwh"], label=policy.upper(), color=color)
    ax.set_title(f"Scenario: {s}")
    ax.set_xlabel("Sim time (s)"); ax.legend()
axes[0].set_ylabel("Cumulative energy (kWh)")
plt.tight_layout()
plt.savefig(figures / "baseline-energy-time.pdf"); plt.close()

# ── Fig 5.7: num_gpu histogram ────────────────────────────────────
df = pd.read_csv("../../../data/alibaba-trace/openb_pod_list_default.csv")
df = df[df["pod_phase"] != "Pending"]
fig, ax = plt.subplots(figsize=(6, 3.5))
df["num_gpu"].value_counts().sort_index().plot(kind="bar", ax=ax, color="#2ca02c")
ax.set_yscale("log")
ax.set_xlabel("num_gpu"); ax.set_ylabel("Số tác vụ (log)")
plt.tight_layout()
plt.savefig(figures / "trace-numgpu-hist.pdf"); plt.close()

# ── Fig 5.8: QoS pie ──────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(5, 5))
df["qos"].value_counts().plot(kind="pie", ax=ax, autopct="%1.1f%%",
                               colors=["#1f77b4","#ff7f0e","#2ca02c","#d62728"])
ax.set_ylabel("")
plt.tight_layout()
plt.savefig(figures / "trace-qos-pie.pdf"); plt.close()

print("All figures saved to", figures)