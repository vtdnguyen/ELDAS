"""Host topology for the static baseline — a Python view of the same
``config/topology-hetero.json`` the Java side loads (G2.1), or the homogeneous
default when no config is given.

The defaults mirror ``SimulationConfig.DEFAULT_HOST`` / ``DEFAULT_POWER`` so a
homogeneous static run and a homogeneous DES run share the same host model.
"""

from __future__ import annotations

import json
from dataclasses import dataclass


# ── Homogeneous defaults (mirror Java SimulationConfig.DEFAULT_*) ────────────

_DEF = dict(
    vcpu=64,
    ram_gb=256,
    gpu=8,
    gpu_memory_mb=32_768,
    cpu_max_watt=400.0,
    cpu_idle_watt=120.0,
    gpu_max_watt=300.0,
    gpu_idle_watt=30.0,
    idle_threshold_sec=30.0,
    suspended_power_watt=10.0,
    wake_energy_kwh=0.0005,
    wake_latency_sec=5.0,
)


@dataclass(frozen=True, slots=True)
class Host:
    """One physical host (resolved from a SKU or the homogeneous default)."""

    sku: str
    pes: int
    ram_mib: int
    gpu_count: int
    cpu_idle_watt: float
    cpu_max_watt: float
    gpu_idle_watt: float
    gpu_max_watt: float
    suspended_power_watt: float


def homogeneous(num_hosts: int = 10) -> list[Host]:
    """Return ``num_hosts`` identical hosts using the Java default spec."""
    return [
        Host(
            sku="default",
            pes=_DEF["vcpu"],
            ram_mib=_DEF["ram_gb"] * 1024,
            gpu_count=_DEF["gpu"],
            cpu_idle_watt=_DEF["cpu_idle_watt"],
            cpu_max_watt=_DEF["cpu_max_watt"],
            gpu_idle_watt=_DEF["gpu_idle_watt"],
            gpu_max_watt=_DEF["gpu_max_watt"],
            suspended_power_watt=_DEF["suspended_power_watt"],
        )
        for _ in range(num_hosts)
    ]


def from_json(json_path: str) -> list[Host]:
    """Expand ``config/topology-hetero.json`` into a flat per-host list.

    Absent numeric fields inherit the homogeneous default — matching the
    lenient Java ``TopologyConfig`` loader.
    """
    with open(json_path) as f:
        cfg = json.load(f)
    skus = cfg.get("skus")
    if not skus:
        raise ValueError(f"topology JSON has no 'skus' array: {json_path}")

    hosts: list[Host] = []
    for i, s in enumerate(skus):
        count = int(s.get("count", 0))
        if count <= 0:
            raise ValueError(f"SKU {i} ('{s.get('name')}') has count <= 0")
        name = s.get("name") or f"sku-{i}"
        host = Host(
            sku=name,
            pes=int(s.get("vcpu", _DEF["vcpu"])),
            ram_mib=int(s.get("ramGb", _DEF["ram_gb"])) * 1024,
            gpu_count=int(s.get("gpu", _DEF["gpu"])),
            cpu_idle_watt=float(s.get("cpuIdleWatt", _DEF["cpu_idle_watt"])),
            cpu_max_watt=float(s.get("cpuMaxWatt", _DEF["cpu_max_watt"])),
            gpu_idle_watt=float(s.get("gpuIdleWatt", _DEF["gpu_idle_watt"])),
            gpu_max_watt=float(s.get("gpuMaxWatt", _DEF["gpu_max_watt"])),
            suspended_power_watt=float(
                s.get("suspendedPowerWatt", _DEF["suspended_power_watt"])
            ),
        )
        hosts.extend([host] * count)
    return hosts


def load_hosts(topology_config: str | None, num_hosts: int = 10) -> list[Host]:
    """Load hosts from ``topology_config`` JSON if given, else homogeneous."""
    if topology_config:
        return from_json(topology_config)
    return homogeneous(num_hosts)
