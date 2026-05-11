import pandas as pd
import numpy as np
from pathlib import Path

# 1. Trace stats
df = pd.read_csv("../../../data/alibaba-trace/openb_pod_list_default.csv")
df = df[df["pod_phase"] != "Pending"]
df["duration"] = df["deletion_time"] - df["scheduled_time"].clip(lower=0)
print("=== TABLE 5.5 ===")
for col in ["cpu_milli", "memory_mib", "num_gpu", "duration"]:
    s = df[col]
    print(f"{col}: min={s.min()}, median={s.median()}, "
          f"mean={s.mean():.1f}, max={s.max()}")

# 2. QoS pie data
qos_counts = df["qos"].value_counts(normalize=True) * 100
print("\n=== QoS distribution (Fig 5.8) ===")
print(qos_counts)

# 3. num_gpu histogram data
gpu_dist = df["num_gpu"].value_counts(normalize=True).sort_index() * 100
print("\n=== num_gpu distribution (Fig 5.7) ===")
print(gpu_dist)

# 4. Smoke metrics analysis
smoke = pd.read_csv("../../../data/results/smoke/metrics.csv")
r_energy_per_step = smoke["total_energy_kwh"].diff() * 3_600_000  # to Watt-seconds
print(f"\n=== Smoke reward shape ===")
print(f"r_energy range: [{r_energy_per_step.min():.0f}, "
      f"{r_energy_per_step.max():.0f}] Ws")
print(f"SLA violations: {smoke['sla_violations'].iloc[-1]} / "
      f"{smoke['tasks_scheduled'].iloc[-1]}")