import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

results = Path("../../../data/results")  # adjust if needed
fig, axes = plt.subplots(1, 3, figsize=(15, 4), sharey=True)

for ax, scen in zip(axes, ["LOW", "HIGH", "BURST"]):
    for policy in ["k8s", "random"]:
        df = pd.read_csv(results / f"baseline-{scen.lower()}-{policy}/metrics.csv")
        ax.plot(df["timestamp"], df["total_energy_kwh"],
                label=policy.upper(), linewidth=1.5)
    ax.set_title(f"Scenario: {scen}")
    ax.set_xlabel("Simulation time (s)")
    ax.legend()
axes[0].set_ylabel("Cumulative energy (kWh)")
plt.tight_layout()
plt.savefig("../figures/baseline-energy.pdf", bbox_inches="tight")