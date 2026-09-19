"""W1.2 — job-size pool: horizon truncation, moments, determinism.

Acceptance targets from PLAN-Workload-Model.md W1.2, as revised by W3.1 (the pool now
also excludes jobs no single host can run — see POOL_MOMENTS for why the numbers moved):
    T = 4 days   -> E[work] =  73 627 +/- 1 PE-s,  0.9 % of jobs truncated
    T = 12 days  -> E[work] = 128 585 +/- 1 PE-s,  0.6 % of jobs truncated
    same seed -> identical sample
"""

from __future__ import annotations

import dataclasses
import random

import pytest

from workload import schema
from workload.calibrate import ARMS as WM1_ARMS, Capacity, HostShape
from workload.jobsize import DEFAULT_EXCLUDED_PHASES, JobSizePool
from workload.schema import SchemaError, Task

from _workload_fixtures import ARMS, DAY, POOL_MOMENTS, openb_tasks  # noqa: F401


def mk(name, cpu_milli, num_gpu, duration, phase="Running"):
    return Task(name=name, cpu_milli=cpu_milli, memory_mib=512, num_gpu=num_gpu,
                gpu_milli=0, gpu_spec="", qos="LS", pod_phase=phase,
                creation_time=0.0, deletion_time=duration, scheduled_time=0.0)


# ── Hand-computable behaviour ───────────────────────────────────────────────

def test_moments_are_hand_checkable_without_truncation():
    jobs = [mk("a", 4000, 2, 10.0), mk("b", 1500, 0, 20.0)]   # pes 4 and 1
    pool = JobSizePool(jobs, horizon_sec=1000.0)
    m = pool.moments
    assert m.size == 2
    assert m.e_work == (4 * 10 + 1 * 20) / 2
    assert m.e_gpu_work == (2 * 10 + 0 * 20) / 2
    assert m.truncated_fraction == 0.0


def test_truncation_clips_duration_and_is_counted():
    jobs = [mk("short", 1000, 1, 50.0), mk("long", 1000, 1, 500.0)]
    pool = JobSizePool(jobs, horizon_sec=100.0)
    assert pool.moments.truncated_fraction == 0.5
    assert pool.moments.e_work == (1 * 50 + 1 * 100) / 2       # long clipped to 100
    assert sorted(j.duration for j in pool.jobs) == [50.0, 100.0]


def test_truncation_is_applied_once_at_construction():
    """Sampling can never disagree with the reported moments."""
    pool = JobSizePool([mk("x", 1000, 0, 10_000.0)], horizon_sec=100.0)
    drawn = pool.sample(5, random.Random(0))
    assert all(t.duration == 100.0 for t in drawn)


def test_from_trace_excludes_pending_and_failed_by_default():
    jobs = [mk("r", 1000, 0, 10, "Running"), mk("p", 1000, 0, 10, "Pending"),
            mk("f", 1000, 0, 10, "Failed"), mk("s", 1000, 0, 10, "Succeeded")]
    pool = JobSizePool.from_trace(jobs, horizon_sec=100.0)
    assert {j.name for j in pool.jobs} == {"r", "s"}
    assert pool.excluded_phases == DEFAULT_EXCLUDED_PHASES


def test_from_trace_honours_a_custom_phase_filter():
    jobs = [mk("r", 1000, 0, 10, "Running"), mk("f", 1000, 0, 10, "Failed")]
    pool = JobSizePool.from_trace(jobs, horizon_sec=100.0,
                                  excluded_phases=frozenset({"Pending"}))
    assert {j.name for j in pool.jobs} == {"r", "f"}


# ── Guard rails ─────────────────────────────────────────────────────────────

def test_non_positive_horizon_is_rejected():
    with pytest.raises(SchemaError, match="horizon must be positive"):
        JobSizePool([mk("a", 1000, 0, 10)], horizon_sec=0.0)


def test_empty_pool_is_rejected():
    with pytest.raises(SchemaError, match="empty"):
        JobSizePool([], horizon_sec=100.0)


def test_from_trace_reports_which_filter_emptied_the_pool():
    with pytest.raises(SchemaError, match="no jobs left after excluding"):
        JobSizePool.from_trace([mk("p", 1000, 0, 10, "Pending")], horizon_sec=100.0)


def test_negative_sample_size_is_rejected():
    pool = JobSizePool([mk("a", 1000, 0, 10)], horizon_sec=100.0)
    with pytest.raises(SchemaError, match="non-negative"):
        pool.sample(-1, random.Random(0))


# ── Determinism (required by W1.6's byte-identical regeneration) ────────────

def test_same_seed_gives_an_identical_sample():
    jobs = [mk(f"j{i}", 1000 * (i + 1), i % 3, 10.0 * (i + 1)) for i in range(50)]
    pool = JobSizePool(jobs, horizon_sec=1e6)
    a = pool.sample(200, random.Random(42))
    b = pool.sample(200, random.Random(42))
    assert a == b


def test_different_seeds_give_different_samples():
    jobs = [mk(f"j{i}", 1000 * (i + 1), i % 3, 10.0 * (i + 1)) for i in range(50)]
    pool = JobSizePool(jobs, horizon_sec=1e6)
    assert pool.sample(200, random.Random(42)) != pool.sample(200, random.Random(43))


def test_sample_at_places_jobs_and_names_them_uniquely():
    jobs = [mk(f"j{i}", 1000, 1, 30.0) for i in range(5)]
    pool = JobSizePool(jobs, horizon_sec=1e6)
    arrivals = [0.0, 5.0, 5.0, 90.0]
    placed = pool.sample_at(arrivals, random.Random(7))

    assert [t.creation_time for t in placed] == arrivals
    assert all(t.scheduled_time == t.creation_time for t in placed)
    assert all(t.duration == 30.0 for t in placed)
    assert [t.name for t in placed] == ["wm1-000000", "wm1-000001", "wm1-000002", "wm1-000003"]
    schema.validate_tasks(placed)          # sorted, unique, non-negative


def test_sample_at_output_survives_a_csv_round_trip(tmp_path):
    jobs = [mk(f"j{i}", 1000 * (i + 1), i % 2, 12.5 * (i + 1)) for i in range(10)]
    pool = JobSizePool(jobs, horizon_sec=1e6)
    placed = pool.sample_at([i * 3.7 for i in range(40)], random.Random(3))
    p = str(tmp_path / "wm1.csv")
    schema.write_csv(p, placed)
    assert schema.read_csv(p) == placed


# ── Locked values against the real trace (PLAN §3.3) ────────────────────────

@pytest.mark.parametrize("arm", ["homo", "hetero"])
def test_pool_moments_reproduce_the_design_reference(arm, openb_tasks):
    _, _, days = ARMS[arm]
    e_work, e_gpu, trunc, unplaceable = POOL_MOMENTS[arm]
    capacity, _ = WM1_ARMS[arm]
    pool = JobSizePool.from_trace(openb_tasks, horizon_sec=days * DAY, capacity=capacity)

    assert pool.moments.e_work == pytest.approx(e_work, abs=1.0)
    assert pool.moments.e_gpu_work == pytest.approx(e_gpu, abs=1.0)
    assert pool.moments.truncated_fraction == pytest.approx(trunc, abs=1e-6)
    assert pool.moments.unplaceable_fraction == pytest.approx(unplaceable, abs=1e-6)


def test_pool_size_is_the_wm1_filtered_trace(openb_tasks):
    pool = JobSizePool.from_trace(openb_tasks, horizon_sec=4 * DAY)
    assert pool.moments.size == 5385


# ── W3.1 — jobs no single host can run are excluded ─────────────────────────

def test_jobs_larger_than_any_host_are_excluded(openb_tasks):
    """openb pods were sized for bigger machines than the 64-vCPU / 256-GiB node here.

    Sampling them yields tasks that every policy drops, so C_SLA gains a constant the
    agent cannot influence — and the CMDP constraint stops being a function of the policy.
    """
    capacity, _ = WM1_ARMS["homo"]
    plain = JobSizePool.from_trace(openb_tasks, horizon_sec=4 * DAY)
    filtered = JobSizePool.from_trace(openb_tasks, horizon_sec=4 * DAY, capacity=capacity)

    assert filtered.moments.size < plain.moments.size
    assert all(capacity.fits_any_host(j) for j in filtered.jobs)
    # The excluded rows are few but enormous: ~0.15 % of jobs holding ~20 % of the work.
    assert filtered.moments.unplaceable_fraction < 0.01
    assert filtered.moments.e_work < 0.85 * plain.moments.e_work


def test_capacity_without_host_shapes_filters_nothing(openb_tasks):
    """A Capacity that does not know its host sizes must not silently drop jobs."""
    bare = Capacity("aggregate-only", 10, 640, 80)
    plain = JobSizePool.from_trace(openb_tasks, horizon_sec=4 * DAY)
    same = JobSizePool.from_trace(openb_tasks, horizon_sec=4 * DAY, capacity=bare)
    assert same.moments.size == plain.moments.size
    assert same.moments.unplaceable_fraction == 0.0


def test_gpu_affinity_counts_as_unplaceable_on_a_cpu_only_cluster():
    """Mirrors canHost: a fractional GPU share still needs a host with cards."""
    cpu_only = Capacity("cpu-only", 1, 64, 0, (HostShape("cpu-only", 64, 1024, 0),))
    frac = mk("frac", 1000, 0, 10.0)
    frac = dataclasses.replace(frac, gpu_milli=500, memory_mib=1)
    plain = mk("plain", 1000, 0, 10.0)
    plain = dataclasses.replace(plain, memory_mib=1)

    assert cpu_only.fits_any_host(plain)
    assert not cpu_only.fits_any_host(frac)
    pool = JobSizePool([frac, plain], horizon_sec=1e6, capacity=cpu_only)
    assert pool.moments.size == 1
    assert pool.jobs[0].name == "plain"


def test_truncation_touches_under_one_percent_of_jobs(openb_tasks):
    """The claim the thesis makes about horizon truncation being a small intervention."""
    for arm in ("homo", "hetero"):
        _, _, days = ARMS[arm]
        pool = JobSizePool.from_trace(openb_tasks, horizon_sec=days * DAY)
        assert pool.moments.truncated_fraction < 0.01


def test_a_longer_horizon_raises_expected_work(openb_tasks):
    """Sanity on the direction: truncation removes work, so relaxing it must add work.

    This is why the arm horizon cannot be chosen independently of the arrival count -
    N = lambda*T does not scale linearly with T (PLAN §3.2).
    """
    short = JobSizePool.from_trace(openb_tasks, horizon_sec=4 * DAY).moments
    long = JobSizePool.from_trace(openb_tasks, horizon_sec=12 * DAY).moments
    assert long.e_work > short.e_work
    assert long.truncated_fraction < short.truncated_fraction
