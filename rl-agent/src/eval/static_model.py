"""Deterministic static energy / SLA surrogate for a task→host assignment.

This is the objective function NSGA-II optimises (G2.3). It is a *static*
idealisation — no discrete-event scheduling, full trace foreknowledge — chosen
to reproduce the same **Pareto tension** the CloudSim DES exhibits, in the same
**units**, so its points live on the same axes as the online results:

    energy_kwh  (lower = better)   —  Σ host energy over the horizon
    sla_cost    (lower = better)   —  Σ κ·max(0, completion − deadline)  (= Java C_SLA)

Energy model (per host h, mirroring the DES power model):
  * host with NO task            → SUSPENDED for the horizon:  P_sus · T
  * host with ≥1 task ("on" over its busy span  span_h = max_end − min_start):
        CPU idle bill            :  P_cpu_idle · span_h
        CPU dynamic bill         :  (P_cpu_max − P_cpu_idle) · Σ_i dur_i·cpuFrac_i
        GPU idle bill (all cards):  gpu_count · P_gpu_idle · span_h
        GPU dynamic bill         :  (P_gpu_max − P_gpu_idle) · Σ_i num_gpu_i·dur_i
  Packing ⇒ fewer hosts "on" ⇒ more suspended ⇒ **less energy**.

SLA model (static congestion proxy for the DES per-task contention factor):
        load_h      = min(1, Σ_i pes_i / host.pes)          (aggregate CPU pressure)
        congestion_h= 1 + load_h                            (∈ [1, 2], as in the DES)
        completion_i= creation_i + dur_i · congestion_{h(i)}
        tardiness_i = max(0, completion_i − deadline_i)
        sla_cost    = Σ_i κ_i · tardiness_i
  Packing ⇒ higher load ⇒ higher congestion ⇒ later completion ⇒ **more SLA cost**.

The two effects pull in opposite directions ⇒ a genuine energy↔SLA Pareto front.
GPU affinity (G2.2) is a hard constraint enforced by ``repair_affinity`` — a GPU
task may only sit on a host with GPUs.
"""

from __future__ import annotations

import numpy as np

from .topology import Host
from .trace_loader import Task

_WS_PER_KWH = 3_600_000.0


class StaticPlacementModel:
    """Vectorised static evaluator over a fixed (hosts, tasks) instance."""

    def __init__(self, hosts: list[Host], tasks: list[Task]) -> None:
        if not hosts:
            raise ValueError("no hosts")
        if not tasks:
            raise ValueError("no tasks")
        self.hosts = hosts
        self.tasks = tasks
        self.n_hosts = len(hosts)
        self.n_tasks = len(tasks)

        # Pre-extract per-task numpy arrays (immutable across evaluations).
        self.pes = np.array([t.pes_needed for t in tasks], dtype=np.float64)
        self.gpus = np.array([t.num_gpu for t in tasks], dtype=np.float64)
        self.dur = np.array([t.duration for t in tasks], dtype=np.float64)
        self.creation = np.array([t.creation_time for t in tasks], dtype=np.float64)
        self.deadline = np.array([t.deadline for t in tasks], dtype=np.float64)
        self.kappa = np.array([t.qos_weight for t in tasks], dtype=np.float64)
        self.end = self.creation + self.dur
        self.needs_gpu = np.array([t.needs_gpu for t in tasks], dtype=bool)

        # Per-host spec arrays.
        self.host_pes = np.array([h.pes for h in hosts], dtype=np.float64)
        self.host_gpu = np.array([h.gpu_count for h in hosts], dtype=np.float64)
        self.cpu_idle = np.array([h.cpu_idle_watt for h in hosts], dtype=np.float64)
        self.cpu_span = np.array(
            [h.cpu_max_watt - h.cpu_idle_watt for h in hosts], dtype=np.float64
        )
        self.gpu_idle = np.array([h.gpu_idle_watt for h in hosts], dtype=np.float64)
        self.gpu_span = np.array(
            [h.gpu_max_watt - h.gpu_idle_watt for h in hosts], dtype=np.float64
        )
        self.p_sus = np.array([h.suspended_power_watt for h in hosts], dtype=np.float64)
        self.host_has_gpu = self.host_gpu > 0

        # Horizon = latest completion across the whole trace.
        self.horizon = float(self.end.max() - self.creation.min())

    # ── Affinity feasibility (G2.2) ────────────────────────────────────────

    def gpu_host_indices(self) -> np.ndarray:
        """Indices of hosts that physically have GPUs."""
        return np.nonzero(self.host_has_gpu)[0]

    def is_feasible(self, assignment: np.ndarray) -> bool:
        """True iff every GPU task sits on a GPU host (affinity)."""
        assigned_gpu = self.host_has_gpu[assignment]
        return bool(np.all(assigned_gpu[self.needs_gpu]))

    def repair_affinity(
        self, assignment: np.ndarray, rng: np.random.Generator
    ) -> np.ndarray:
        """Move any GPU task off a CPU-only host to a random GPU host.

        Raises if the topology has no GPU host at all but GPU tasks exist.
        """
        a = np.asarray(assignment, dtype=np.int64).copy()
        bad = self.needs_gpu & ~self.host_has_gpu[a]
        if not bad.any():
            return a
        gpu_hosts = self.gpu_host_indices()
        if gpu_hosts.size == 0:
            raise ValueError("GPU tasks present but topology has no GPU host")
        a[bad] = rng.choice(gpu_hosts, size=int(bad.sum()))
        return a

    # ── Objective evaluation ───────────────────────────────────────────────

    def evaluate(self, assignment: np.ndarray) -> tuple[float, float]:
        """Return ``(energy_kwh, sla_cost)`` for one assignment.

        ``assignment[i]`` = host index for task ``i``. Assumes affinity has
        been repaired; a GPU task on a CPU-only host simply draws no GPU energy
        (it should not occur after ``repair_affinity``).
        """
        a = np.asarray(assignment, dtype=np.int64)

        # Aggregate per-host demand via bincount (fast, vectorised).
        nh = self.n_hosts
        cpu_dyn_ws = np.bincount(
            a, weights=self.dur * np.minimum(1.0, self.pes / self.host_pes[a]),
            minlength=nh,
        )
        gpu_dyn_ws = np.bincount(a, weights=self.gpus * self.dur, minlength=nh)

        # Busy span per host = max_end − min_start among its tasks (0 if empty).
        has_task = np.bincount(a, minlength=nh) > 0
        min_start = np.full(nh, np.inf)
        max_end = np.full(nh, -np.inf)
        np.minimum.at(min_start, a, self.creation)
        np.maximum.at(max_end, a, self.end)
        span = np.where(has_task, max_end - min_start, 0.0)

        # ── Energy (Watt-seconds → kWh) ──
        cpu_idle_ws = np.where(has_task, self.cpu_idle * span, 0.0)
        cpu_ws = cpu_idle_ws + self.cpu_span * cpu_dyn_ws
        gpu_idle_ws = np.where(has_task, self.host_gpu * self.gpu_idle * span, 0.0)
        gpu_ws = gpu_idle_ws + self.gpu_span * gpu_dyn_ws
        # Empty hosts sit SUSPENDED for the whole horizon.
        sus_ws = np.where(has_task, 0.0, self.p_sus * self.horizon)
        energy_ws = cpu_ws + gpu_ws + sus_ws
        energy_kwh = float(energy_ws.sum() / _WS_PER_KWH)

        # ── SLA cost (κ · seconds of tardiness) ──
        # Congestion proxy = per-host TIME-AVERAGED CPU pressure over the busy
        # window: work / (capacity · span) = mean utilisation while the host is
        # on. Using the busy span (not an instantaneous sum) is what keeps the
        # measure from saturating to 1 for both spread and packed on a real
        # trace — packing concentrates work into fewer host-seconds ⇒ higher
        # pressure ⇒ higher congestion, exactly the DES tension. Clipped to
        # [0,1] so congestion stays in [1,2] like the DES congestionFactor.
        work_pes_s = np.bincount(a, weights=self.dur * self.pes, minlength=nh)
        safe_span = np.where(span > 0.0, span, 1.0)
        pressure = work_pes_s / (self.host_pes * safe_span)     # per host
        load = np.minimum(1.0, pressure)
        congestion = 1.0 + load[a]                              # per task ∈ [1, 2]
        completion = self.creation + self.dur * congestion
        tardiness = np.maximum(0.0, completion - self.deadline)
        sla_cost = float((self.kappa * tardiness).sum())

        return energy_kwh, sla_cost

    # ── Canonical extreme assignments (for tests / anchors) ────────────────

    def spread_assignment(self) -> np.ndarray:
        """Round-robin over affinity-feasible hosts — the low-SLA anchor."""
        gpu_hosts = self.gpu_host_indices()
        all_hosts = np.arange(self.n_hosts)
        a = np.empty(self.n_tasks, dtype=np.int64)
        cpu_ptr = gpu_ptr = 0
        for i in range(self.n_tasks):
            if self.needs_gpu[i]:
                a[i] = gpu_hosts[gpu_ptr % gpu_hosts.size]
                gpu_ptr += 1
            else:
                a[i] = all_hosts[cpu_ptr % all_hosts.size]
                cpu_ptr += 1
        return a

    def packed_assignment(self) -> np.ndarray:
        """Everything onto one GPU-capable host — the low-energy anchor."""
        target = int(self.gpu_host_indices()[0]) if self.gpu_host_indices().size \
            else 0
        return np.full(self.n_tasks, target, dtype=np.int64)
