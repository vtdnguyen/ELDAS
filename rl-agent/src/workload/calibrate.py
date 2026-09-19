"""Load calibration — bottleneck (dominant-resource) offered load (PLAN §3.1, §3.2).

Offered load on a single resource is the textbook utilisation law,

    rho_r = lambda * E[demand_r x duration] / capacity_r

but a cluster has two capacities here and they do not bind together. On the homogeneous
topology (640 PE, 80 GPU) CPU binds; on the 3-SKU heterogeneous one (640 PE, **18** GPU)
GPU binds, and hard: calibrating hetero to rho_cpu = 0.85 implies rho_gpu = 2.6, a
workload the cluster physically cannot carry. So the calibrated quantity is the
**dominant resource share**,

    rho = max_r (rho_r)

following Dominant Resource Fairness (Ghodsi et al., NSDI'11). One definition, feasible
on both topologies, and the reported rho is always the one that actually constrains.

Solving for the arrival rate and count:

    lambda = rho / max(E[work]/C_pe, E[gpu work]/C_gpu)
    N      = round(lambda * T)

``N`` is what the arrival model then places exactly (PLAN §3.4), which is what makes the
realised load exact rather than a draw.
"""

from __future__ import annotations

from dataclasses import dataclass

from .jobsize import JobSizePool


class CalibrationError(ValueError):
    """Raised for an infeasible or ill-posed calibration request."""


@dataclass(frozen=True, slots=True)
class HostShape:
    """One host's usable size — what a *single* task must fit inside (W3.1)."""

    name: str
    pes: int
    ram_mib: int
    gpus: int


@dataclass(frozen=True, slots=True)
class Capacity:
    """Cluster capacity along the two axes the simulator enforces.

    ``host_shapes`` carries the per-host sizes as well, because aggregate capacity is not
    sufficient to tell whether a workload is runnable: the simulator places whole tasks on
    single hosts, so a pod asking for 88 vCPU is unplaceable on a 640-PE cluster of
    64-PE nodes no matter how idle it is.
    """

    name: str
    hosts: int
    total_pes: int
    total_gpus: int
    #: Distinct host sizes in this cluster. Empty ⇒ per-host limits unknown, and
    #: :meth:`fits_any_host` degrades to "everything fits" (used by tests that only
    #: exercise the aggregate calibration maths).
    host_shapes: tuple[HostShape, ...] = ()

    def __post_init__(self):
        if self.total_pes <= 0:
            raise CalibrationError(f"{self.name}: total_pes must be positive")
        if self.total_gpus < 0:
            raise CalibrationError(f"{self.name}: total_gpus must not be negative")

    def fits_any_host(self, task) -> bool:
        """Could an *empty* cluster run this task at all?

        Mirrors ``SimulationManager.canHost`` against an idle host, GPU affinity included:
        a task wanting any GPU share needs a host that physically has cards. A task failing
        this is dropped by every policy in every scenario, so it contributes a constant,
        seed-dependent floor to ``C_SLA`` that no scheduling decision can move — which
        breaks the CMDP reading of the constraint (the cost must respond to the policy).
        """
        if not self.host_shapes:
            return True
        for h in self.host_shapes:
            if task.pes > h.pes or task.memory_mib > h.ram_mib:
                continue
            if task.needs_gpu and (h.gpus <= 0 or task.num_gpu > h.gpus):
                continue
            return True
        return False


#: Per-host size shared by both arms: 64 vCPU / 256 GiB (SimulationConfig.DEFAULT_HOST and
#: config/topology-hetero.json, whose three SKUs differ only in GPU count and power).
_HOST_PES = 64
_HOST_RAM_MIB = 256 * 1024

#: The two experiment arms (PLAN §3.2). Horizons are chosen so the HIGH scenario carries
#: roughly two thousand arrivals, which keeps episode length - and therefore the number
#: of dual updates at a fixed step budget - comparable across arms.
ARMS = {
    "homo": (Capacity("homogeneous 10x(64 PE, 8 GPU)", 10, 640, 80,
                      (HostShape("homogeneous", _HOST_PES, _HOST_RAM_MIB, 8),)),
             4 * 86400.0),
    "hetero": (Capacity("hetero-3sku (18 GPU)", 10, 640, 18,
                        (HostShape("gpu-heavy", _HOST_PES, _HOST_RAM_MIB, 4),
                         HostShape("balanced", _HOST_PES, _HOST_RAM_MIB, 2),
                         HostShape("cpu-only", _HOST_PES, _HOST_RAM_MIB, 0))),
               12 * 86400.0),
}


@dataclass(frozen=True, slots=True)
class LoadPlan:
    """The calibrated arrival plan for one (arm, scenario)."""

    rho_target: float
    n_task: int
    lambda_per_sec: float
    horizon_sec: float
    rho_cpu: float          #: realised CPU load at this lambda
    rho_gpu: float          #: realised GPU load at this lambda
    bottleneck: str         #: "cpu" or "gpu" - which resource sets the load

    @property
    def lambda_per_hour(self) -> float:
        return self.lambda_per_sec * 3600.0


def bottleneck_load(pool: JobSizePool, capacity: Capacity,
                    lambda_per_sec: float) -> tuple[float, float, str]:
    """Realised (rho_cpu, rho_gpu, bottleneck) at a given arrival rate."""
    m = pool.moments
    rho_cpu = lambda_per_sec * m.e_work / capacity.total_pes
    rho_gpu = (lambda_per_sec * m.e_gpu_work / capacity.total_gpus
               if capacity.total_gpus > 0 else 0.0)
    return rho_cpu, rho_gpu, ("gpu" if rho_gpu > rho_cpu else "cpu")


def plan_load(pool: JobSizePool, capacity: Capacity, rho_target: float,
              horizon_sec: float) -> LoadPlan:
    """Solve for the arrival rate and count that put the dominant resource at ``rho``."""
    if rho_target <= 0:
        raise CalibrationError(f"rho target must be positive, got {rho_target}")
    if horizon_sec <= 0:
        raise CalibrationError(f"horizon must be positive, got {horizon_sec}")
    if pool.horizon_sec != horizon_sec:
        # Truncation is applied at pool construction, so a pool built for a different
        # horizon carries the wrong E[work] and would silently mis-calibrate.
        raise CalibrationError(
            f"pool was built for a {pool.horizon_sec / 86400:g}-day horizon but "
            f"calibration asks for {horizon_sec / 86400:g} days; rebuild the pool")

    m = pool.moments
    per_task = max(m.e_work / capacity.total_pes,
                   (m.e_gpu_work / capacity.total_gpus) if capacity.total_gpus > 0 else 0.0)
    if per_task <= 0:
        raise CalibrationError(
            f"{capacity.name}: pool has no demand on either capacity axis")

    lam = rho_target / per_task
    n = round(lam * horizon_sec)
    if n <= 0:
        raise CalibrationError(
            f"calibration yields {n} arrivals for rho={rho_target} over "
            f"{horizon_sec / 86400:g} days; the horizon is too short for this load")

    rho_cpu, rho_gpu, which = bottleneck_load(pool, capacity, lam)
    return LoadPlan(rho_target=rho_target, n_task=n, lambda_per_sec=lam,
                    horizon_sec=horizon_sec, rho_cpu=rho_cpu, rho_gpu=rho_gpu,
                    bottleneck=which)


def measured_load(tasks, capacity: Capacity, horizon_sec: float) -> tuple[float, float]:
    """Offered load actually present in a generated trace — the acceptance check.

    Deliberately computed from the emitted tasks rather than from the plan, so a bug
    between calibration and generation shows up as a mismatch instead of agreeing with
    itself.
    """
    if horizon_sec <= 0:
        raise CalibrationError(f"horizon must be positive, got {horizon_sec}")
    cpu = sum(t.pes * t.duration for t in tasks) / (capacity.total_pes * horizon_sec)
    gpu = (sum(t.num_gpu * t.duration for t in tasks) / (capacity.total_gpus * horizon_sec)
           if capacity.total_gpus > 0 else 0.0)
    return cpu, gpu
