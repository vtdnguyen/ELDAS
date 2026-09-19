"""G2.3 tests — static placement model, affinity repair, trace/topology
loaders, and (guarded) the pymoo NSGA-II reference-front driver.

The static-model tests are the *scientific correctness* checks: they pin the
energy/SLA formulas to hand-computed numbers and assert the energy↔SLA Pareto
tension (packing ⇒ less energy, more SLA) that the whole G2.3 baseline relies on.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest

from eval import topology as topo_mod
from eval import trace_loader
from eval.qos import qos_slack_factor, qos_weight
from eval.static_model import StaticPlacementModel
from eval.trace_loader import Task


# ── Helpers ─────────────────────────────────────────────────────────────────

def _task(name, cpu_milli, dur, qos="LS", num_gpu=0, gpu_milli=0, creation=0.0):
    return Task(
        name=name, cpu_milli=cpu_milli, memory_mib=1024,
        num_gpu=num_gpu, gpu_milli=gpu_milli, qos=qos, pod_phase="Running",
        creation_time=creation, deletion_time=creation + dur, scheduled_time=creation,
    )


_CONFIG = Path(__file__).resolve().parents[2] / "config" / "topology-hetero.json"


# ── Task derived fields mirror Java ─────────────────────────────────────────

def test_task_derived_fields_match_java_formulas():
    t = _task("t", cpu_milli=32000, dur=100.0, qos="LS")
    assert t.pes_needed == 32                      # max(1, cpu_milli//1000)
    assert t.duration == 100.0                     # deletion - max(creation, scheduled)
    assert t.qos_weight == 3.0                     # κ(LS)
    # W1.5: creation + dur*slack(LS=1.1) + floor(LS)=106 -> 0 + 110 + 106
    assert t.deadline == pytest.approx(216.0)
    assert not t.needs_gpu


def test_needs_gpu_covers_fractional_share():
    assert _task("g", 1000, 10, num_gpu=1).needs_gpu
    assert _task("f", 1000, 10, num_gpu=0, gpu_milli=500).needs_gpu   # fractional
    assert not _task("c", 1000, 10, num_gpu=0, gpu_milli=0).needs_gpu


# ── Energy + SLA pinned to a hand computation ───────────────────────────────

def test_evaluate_matches_hand_computation():
    hosts = topo_mod.homogeneous(10)               # 64 PE, 8 GPU, cpu[120/400], gpu[30/300]
    task = _task("t", cpu_milli=32000, dur=100.0, qos="LS")   # pes=32, deadline=110
    model = StaticPlacementModel(hosts, [task])

    energy_kwh, sla_cost = model.evaluate(np.array([0]))

    # Energy (Watt-seconds), host0 active over span=100:
    #   cpu = 120*100 + (400-120)*(100*0.5) = 12000 + 14000 = 26000
    #   gpu = 8*30*100 + (300-30)*0          = 24000
    #   9 suspended hosts = 9 * 10 * horizon(100) = 9000
    #   total = 59000 Ws → 0.0163889 kWh
    assert energy_kwh == pytest.approx(59000.0 / 3_600_000.0, rel=1e-9)

    # SLA: load=32/64=0.5 → congestion=1.5 → completion=150; deadline=110+106=216.
    # No violation: the W1.5 absolute floor absorbs a 50 s slowdown on a 100 s job,
    # which is exactly what it exists to do — before the floor this job was charged
    # 120 units of C_SLA for a delay smaller than a host wake-up.
    assert sla_cost == pytest.approx(0.0, abs=1e-12)


def test_sla_cost_charges_once_the_slowdown_exceeds_the_floor():
    """The floor delays the onset of C_SLA; it must not remove it."""
    hosts = topo_mod.homogeneous(10)
    task = _task("t", cpu_milli=32000, dur=10_000.0, qos="LS")
    model = StaticPlacementModel(hosts, [task])

    _, sla_cost = model.evaluate(np.array([0]))

    # load=32/64=0.5 → congestion=1.5 → completion=15000
    # deadline = 10000*1.1 + 106 = 11106 → tardiness = 3894, κ=3 → 11682
    assert sla_cost == pytest.approx(11682.0, rel=1e-9)


def test_sla_cost_uses_same_units_as_java_c_sla():
    # C_SLA = Σ κ·max(0, completion − deadline); reproduce for 2 tasks by hand.
    hosts = topo_mod.homogeneous(2)
    t1 = _task("a", cpu_milli=64000, dur=10_000, qos="LS")   # pes=64 → load=1 → cong=2
    t2 = _task("b", cpu_milli=64000, dur=10_000, qos="BE")   # BE slack 3.0 → generous
    model = StaticPlacementModel(hosts, [t1, t2])
    # Both onto host 0: load=min(1,128/64)=1 → congestion 2.
    _, sla = model.evaluate(np.array([0, 0]))
    # t1: completion=20000, deadline=10000*1.1+106=11106 → tard=8894, κ=3 → 26682
    # t2: completion=20000, deadline=10000*3.0+2120=32120 → tard=0 → 0
    assert sla == pytest.approx(26682.0, rel=1e-9)


# ── The Pareto tension: pack ⇒ less energy, more SLA ─────────────────────────

def test_packing_trades_energy_for_sla():
    hosts = topo_mod.homogeneous(10)
    # Durations must exceed the W1.5 absolute floor for the tension to be visible at
    # all: with 100 s jobs both placements now sit at zero SLA cost, which is correct
    # behaviour but tests nothing.
    tasks = [_task(f"t{i}", cpu_milli=8000, dur=10_000, qos="LS", creation=i * 1.0)
             for i in range(30)]
    model = StaticPlacementModel(hosts, tasks)

    e_spread, s_spread = model.evaluate(model.spread_assignment())
    e_packed, s_packed = model.evaluate(model.packed_assignment())

    # Packing suspends 9 hosts ⇒ strictly less energy…
    assert e_packed < e_spread
    # …but concentrates load ⇒ strictly more SLA cost.
    assert s_packed > s_spread
    for v in (e_spread, e_packed, s_spread, s_packed):
        assert math.isfinite(v) and v >= 0.0


# ── GPU affinity (G2.2) ─────────────────────────────────────────────────────

def test_affinity_repair_moves_gpu_tasks_to_gpu_hosts():
    hosts = topo_mod.from_json(str(_CONFIG))       # 3×4gpu, 3×2gpu, 4×0gpu
    tasks = [_task("g", 4000, 100, num_gpu=2), _task("c", 4000, 100, num_gpu=0)]
    model = StaticPlacementModel(hosts, tasks)

    # Force the GPU task onto a CPU-only host (index 6) → infeasible.
    bad = np.array([6, 6])
    assert not model.is_feasible(bad)

    rng = np.random.default_rng(0)
    repaired = model.repair_affinity(bad, rng)
    assert model.is_feasible(repaired)
    # The GPU task now sits on a host with GPUs; the CPU task is untouched.
    assert hosts[repaired[0]].gpu_count > 0
    assert repaired[1] == 6


def test_affinity_repair_raises_without_gpu_hosts():
    # A CPU-only cluster cannot host a GPU task → repair must fail loudly.
    cpu_only = [topo_mod.Host("cpu", 64, 262144, 0, 60, 200, 30, 300, 10) for _ in range(3)]
    model = StaticPlacementModel(cpu_only, [_task("g", 4000, 100, num_gpu=1)])
    with pytest.raises(ValueError):
        model.repair_affinity(np.array([0]), np.random.default_rng(0))


def test_spread_and_packed_are_affinity_feasible():
    hosts = topo_mod.from_json(str(_CONFIG))
    tasks = [_task(f"g{i}", 4000, 100, num_gpu=1, creation=i) for i in range(5)] + \
            [_task(f"c{i}", 4000, 100, num_gpu=0, creation=i) for i in range(5)]
    model = StaticPlacementModel(hosts, tasks)
    assert model.is_feasible(model.spread_assignment())
    assert model.is_feasible(model.packed_assignment())


# ── Non-dominated filter ────────────────────────────────────────────────────

def test_non_dominated_filter():
    from eval.nsga2_baseline import non_dominated
    F = np.array([
        [1.0, 5.0],   # non-dominated
        [2.0, 4.0],   # non-dominated
        [3.0, 6.0],   # dominated by [2,4] and [1,5]
        [2.0, 4.0],   # duplicate of a non-dominated point (kept: not strictly dominated)
    ])
    mask = non_dominated(F)
    assert mask[0] and mask[1]
    assert not mask[2]


# ── Topology + trace loaders ────────────────────────────────────────────────

def test_topology_from_json_matches_hetero_spec():
    hosts = topo_mod.from_json(str(_CONFIG))
    assert len(hosts) == 10
    assert [h.gpu_count for h in hosts] == [4, 4, 4, 2, 2, 2, 0, 0, 0, 0]
    # CPU power differentiates the SKUs (energy axis).
    assert hosts[0].cpu_max_watt == 500 and hosts[0].cpu_idle_watt == 200
    assert hosts[9].cpu_max_watt == 200 and hosts[9].cpu_idle_watt == 60


def test_qos_maps_match_java():
    assert qos_weight("LS") == 3.0 and qos_weight("BE") == 0.5
    assert qos_weight("unknown") == 1.0
    assert qos_slack_factor("LS") == 1.1 and qos_slack_factor("BE") == 3.0


# ── pymoo driver (guarded — skips cleanly when pymoo is absent) ──────────────

try:
    import pymoo  # noqa: F401
    _HAS_PYMOO = True
except ImportError:
    _HAS_PYMOO = False


@pytest.mark.skipif(not _HAS_PYMOO, reason="pymoo not installed")
def test_nsga2_front_is_feasible_and_non_dominated():
    from eval.nsga2_baseline import build_reference_front, non_dominated

    hosts = topo_mod.from_json(str(_CONFIG))
    tasks = [_task(f"g{i}", 4000, 100, num_gpu=1, creation=i) for i in range(8)] + \
            [_task(f"c{i}", 6000, 100, num_gpu=0, creation=i, qos="Guaranteed")
             for i in range(12)]
    model = StaticPlacementModel(hosts, tasks)

    result = build_reference_front(model, pop_size=40, n_gen=15, seed=42)
    F, X = result["F"], result["X"]

    assert len(F) >= 1
    # Every reported assignment must respect GPU affinity.
    for row in X:
        assert model.is_feasible(row)
    # The reported front is genuinely non-dominated.
    assert non_dominated(F).all()
    # A reference front must not be dominated by a trivial heuristic anchor.
    #
    # This replaces an earlier bound, `front_min_energy >= packed_energy`, which was
    # never a real invariant: `packed` is a heuristic (everything onto the first
    # feasible host), not the energy optimum, and on the heterogeneous topology packing
    # onto the cheapest SKU beats it. The old assertion only held incidentally because
    # SLA cost used to push solutions away from that region; once the W1.5 deadline
    # floor stopped charging short jobs, NSGA-II was free to find the better packing and
    # the assertion fired on a *correct* result.
    for label in ("packed", "spread"):
        e_a, s_a = result[label]
        for row in F:
            dominates = (e_a <= row[0] + 1e-12 and s_a <= row[1] + 1e-12
                         and (e_a < row[0] - 1e-12 or s_a < row[1] - 1e-12))
            assert not dominates, (
                f"the {label} anchor ({e_a:.6g}, {s_a:.6g}) dominates a reference-front "
                f"point ({row[0]:.6g}, {row[1]:.6g})")
    assert F[:, 0].min() > 0.0
