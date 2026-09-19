"""Job-size model — empirical bootstrap with horizon truncation (PLAN §3.1, §3.3).

WM-1 resamples **whole rows** from the source trace rather than fitting marginals. That
keeps the correlation structure between CPU, memory, GPU, QoS and duration for free —
correlations a parametric model would have to reproduce explicitly with a copula, and
which matter here because GPU-hungry jobs are not a random subset of the trace.

Durations are truncated at the arm horizon ``T``. This is horizon-consistency, not
trace-fixing: a job longer than the entire simulated window cannot be represented inside
that window, it is background load. The fraction affected is small and is reported so it
can be stated in the thesis (PLAN §3.3):

    T = 4 days   ->  0.93 % of jobs truncated, E[work] =  92 572 PE-s
    T = 12 days  ->  0.61 % of jobs truncated, E[work] = 170 708 PE-s

The heavy tail that makes truncation necessary is not an openb artefact — Philly's
run_time mean/median ratio is 260x, Helios's 104x, openb's 47x (PLAN §1.3 note on F).
"""

from __future__ import annotations

import random
from dataclasses import dataclass

from .schema import PHASE_FAILED, PHASE_PENDING, SchemaError, Task

#: Phases excluded from the WM-1 job-size pool. ``Pending`` never ran; ``Failed`` pods
#: have a lifetime that is not a service demand (PLAN §1.3 issue b).
DEFAULT_EXCLUDED_PHASES = frozenset({PHASE_PENDING, PHASE_FAILED})

#: Relative tolerance the conditioned bootstrap drives realised offered load to.
#: 0.5 % leaves comfortable margin under the +/-2 % acceptance criterion while keeping
#: the number of swaps - and therefore the deviation from an unconditioned sample - small.
LOAD_TOLERANCE = 0.005

#: Swap budget per task. Every measured case converges within ~1 000 swaps total, so this
#: is a runaway guard, not a working limit.
MAX_SWAPS_PER_TASK = 200


@dataclass(frozen=True, slots=True)
class PoolMoments:
    """First moments of the truncated pool — the inputs to load calibration (W1.4)."""

    e_work: float             #: E[pes x duration] in PE-seconds
    e_gpu_work: float         #: E[num_gpu x duration] in card-seconds
    truncated_fraction: float #: share of pool jobs whose duration was clipped at T
    size: int                 #: number of jobs in the pool
    #: share of candidate jobs excluded because no single host could ever run them (W3.1)
    unplaceable_fraction: float = 0.0


class JobSizePool:
    """A bootstrap sampler over trace jobs, with durations truncated at ``horizon_sec``.

    The pool is immutable once built; sampling is a pure function of the supplied RNG so
    the same seed always yields the same jobs (required by W1.6's byte-identical
    regeneration guarantee).
    """

    __slots__ = ("_jobs", "_horizon", "_moments", "_excluded")

    def __init__(self, jobs: list[Task], horizon_sec: float,
                 excluded_phases: frozenset[str] = DEFAULT_EXCLUDED_PHASES,
                 capacity=None):
        if horizon_sec <= 0:
            raise SchemaError(f"horizon must be positive, got {horizon_sec}")
        if not jobs:
            raise SchemaError("job-size pool is empty after filtering")

        self._horizon = float(horizon_sec)
        self._excluded = excluded_phases

        # W3.1 — Exclude jobs no single host in this cluster could ever run. openb's pods
        # were sized for larger machines than the 64-vCPU / 256-GiB node modelled here, so
        # ~0.1 % of rows ask for more than one host holds (measured: 88 vCPU, 257 GiB).
        # Sampling them produces tasks that EVERY policy drops, which puts a constant floor
        # under C_SLA that no decision can move — the constraint would then be partly a
        # property of the trace rather than of the policy, and the budget grid `d` would be
        # calibrated against that floor.
        #
        # Excluded, not clamped: a 88-vCPU pod resized to 64 is a demand that was never
        # observed. Duration truncation at T is a different case — there the job is real and
        # only its tail falls outside the observation window.
        placeable: list[Task] = jobs
        unplaceable = 0
        if capacity is not None:
            placeable = [t for t in jobs if capacity.fits_any_host(t)]
            unplaceable = len(jobs) - len(placeable)
            if not placeable:
                raise SchemaError(
                    f"no job in the pool fits any host of {capacity.name}; "
                    f"all {len(jobs)} candidates exceed a single host's capacity")

        # Truncation happens once, at construction. Sampling then just picks rows, so
        # a sample can never disagree with the moments reported for the pool.
        truncated = 0
        cut: list[Task] = []
        for t in placeable:
            d = t.duration
            if d > self._horizon:
                truncated += 1
                cut.append(t.with_duration(self._horizon))
            else:
                cut.append(t)
        self._jobs = tuple(cut)

        n = len(self._jobs)
        self._moments = PoolMoments(
            e_work=sum(t.pes * t.duration for t in self._jobs) / n,
            e_gpu_work=sum(t.num_gpu * t.duration for t in self._jobs) / n,
            truncated_fraction=truncated / n,
            size=n,
            unplaceable_fraction=unplaceable / max(1, unplaceable + n),
        )

    # ── construction ────────────────────────────────────────────────────────

    @classmethod
    def from_trace(cls, tasks: list[Task], horizon_sec: float, *,
                   excluded_phases: frozenset[str] = DEFAULT_EXCLUDED_PHASES,
                   capacity=None) -> "JobSizePool":
        """Filter by pod phase (and, given a ``capacity``, by placeability), then build."""
        kept = [t for t in tasks if t.pod_phase not in excluded_phases]
        if not kept:
            raise SchemaError(
                f"no jobs left after excluding phases {sorted(excluded_phases)} "
                f"from {len(tasks)} tasks")
        return cls(kept, horizon_sec, excluded_phases, capacity=capacity)

    # ── introspection ───────────────────────────────────────────────────────

    @property
    def moments(self) -> PoolMoments:
        return self._moments

    @property
    def horizon_sec(self) -> float:
        return self._horizon

    @property
    def excluded_phases(self) -> frozenset[str]:
        return self._excluded

    @property
    def jobs(self) -> tuple[Task, ...]:
        return self._jobs

    def __len__(self) -> int:
        return len(self._jobs)

    # ── sampling ────────────────────────────────────────────────────────────

    def sample(self, n: int, rng: random.Random) -> list[Task]:
        """Draw ``n`` whole job rows with replacement.

        Returned tasks still carry their *original* timestamps; the caller pairs them
        with generated arrival times via :meth:`Task.with_arrival`. Keeping the two steps
        separate is what makes the job-size model and the arrival model independently
        testable (PLAN §2).
        """
        if n < 0:
            raise SchemaError(f"sample size must be non-negative, got {n}")
        size = len(self._jobs)
        return [self._jobs[rng.randrange(size)] for _ in range(n)]

    def sample_for_load(self, n: int, rng: random.Random, *,
                        pe_capacity: int, gpu_capacity: int, horizon_sec: float,
                        rho_target: float, tol: float = LOAD_TOLERANCE,
                        max_attempts_per_task: int = MAX_SWAPS_PER_TASK) -> list[Task]:
        """Draw ``n`` jobs whose realised offered load is ``rho_target`` within ``tol``.

        A plain bootstrap pins the arrival *count* but not the *work*, and with this
        trace's tail — the top 1 % of jobs carry 92 % of all CPU-seconds — that is not
        nearly enough. Measured over five seeds, an unconditioned draw of the calibrated
        count realises loads of 0.77-1.05 on the homogeneous arm and **0.57-1.13** on the
        heterogeneous one against a 0.85 target. At that spread the LOW and HIGH scenarios
        overlap across seeds and the load axis stops being a controlled factor at all.

        So the draw is *conditioned* on the load: sample ``n`` jobs, then repeatedly
        propose replacing a random one with a fresh draw and keep the swap only when it
        moves realised load closer to target. This is a conditional bootstrap - the sample
        stays representative of the pool, conditioned on the quantity the experiment
        controls - and it keeps ``n`` exact, so episode length stays comparable.

        Convergence is fast: every one of 30 (arm, scenario, seed) combinations settles
        inside 0.2 % within a thousand swaps.

        Raises:
            SchemaError: if the repair cannot reach ``tol``, which would mean the pool
                cannot express the requested load at this count.
        """
        if n <= 0:
            raise SchemaError(f"sample size must be positive, got {n}")
        if rho_target <= 0:
            raise SchemaError(f"rho target must be positive, got {rho_target}")
        if pe_capacity <= 0 or horizon_sec <= 0:
            raise SchemaError("pe_capacity and horizon_sec must be positive")

        pe_sec = pe_capacity * horizon_sec
        gpu_sec = gpu_capacity * horizon_sec if gpu_capacity > 0 else 0.0

        def load(w_cpu: float, w_gpu: float) -> float:
            return max(w_cpu / pe_sec, (w_gpu / gpu_sec) if gpu_sec > 0 else 0.0)

        size = len(self._jobs)
        jobs = [self._jobs[rng.randrange(size)] for _ in range(n)]
        w_cpu = sum(j.pes * j.duration for j in jobs)
        w_gpu = sum(j.num_gpu * j.duration for j in jobs)
        current = load(w_cpu, w_gpu)

        budget = max_attempts_per_task * n
        attempts = 0
        while abs(current - rho_target) / rho_target > tol and attempts < budget:
            attempts += 1
            i = rng.randrange(n)
            cand = self._jobs[rng.randrange(size)]
            old = jobs[i]
            new_cpu = w_cpu - old.pes * old.duration + cand.pes * cand.duration
            new_gpu = w_gpu - old.num_gpu * old.duration + cand.num_gpu * cand.duration
            new_load = load(new_cpu, new_gpu)
            if abs(new_load - rho_target) < abs(current - rho_target):
                jobs[i] = cand
                w_cpu, w_gpu, current = new_cpu, new_gpu, new_load

        if abs(current - rho_target) / rho_target > tol:
            raise SchemaError(
                f"could not condition {n} jobs to rho={rho_target:g} within {tol:.1%} "
                f"after {attempts} swaps (reached {current:.4f}); the pool may not be "
                f"able to express this load at this arrival count")
        return jobs

    def sample_at(self, arrivals: list[float], rng: random.Random, *,
                  name_prefix: str = "wm1") -> list[Task]:
        """Draw one job per arrival time and place it there.

        Names are positional (``wm1-000000``) so a generated trace has stable, unique
        identifiers — :func:`schema.validate_tasks` rejects duplicates, and bootstrap
        sampling reuses source rows by design.
        """
        drawn = self.sample(len(arrivals), rng)
        return [job.with_arrival(t, name=f"{name_prefix}-{i:06d}")
                for i, (t, job) in enumerate(zip(arrivals, drawn))]
