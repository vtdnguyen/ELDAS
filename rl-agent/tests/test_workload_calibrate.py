"""W1.4 — load calibration on the dominant resource.

Acceptance target from PLAN-Workload-Model.md W1.4: reproduce the N column of §3.6 for
all eight (arm, scenario) rows, and have the load measured back off a generated trace
land within 2 % of the target.
"""

from __future__ import annotations

import pytest

from workload.calibrate import (ARMS, CalibrationError, Capacity, bottleneck_load,
                                measured_load, plan_load)
from workload.jobsize import JobSizePool
from workload.schema import Task

from _workload_fixtures import DAY, SCENARIOS, openb_tasks  # noqa: F401


def mk(name, cpu_milli, num_gpu, duration):
    return Task(name=name, cpu_milli=cpu_milli, memory_mib=1, num_gpu=num_gpu,
                gpu_milli=0, gpu_spec="", qos="LS", pod_phase="Running",
                creation_time=0.0, deletion_time=duration, scheduled_time=0.0)


# ── Capacity ────────────────────────────────────────────────────────────────

def test_capacity_rejects_nonsense():
    with pytest.raises(CalibrationError):
        Capacity("bad", 1, 0, 8)
    with pytest.raises(CalibrationError):
        Capacity("bad", 1, 64, -1)


def test_the_two_arms_are_the_documented_ones():
    homo, t_homo = ARMS["homo"]
    het, t_het = ARMS["hetero"]
    assert (homo.total_pes, homo.total_gpus, t_homo / DAY) == (640, 80, 4)
    assert (het.total_pes, het.total_gpus, t_het / DAY) == (640, 18, 12)


# ── Which resource binds ────────────────────────────────────────────────────

def test_cpu_binds_when_gpus_are_plentiful():
    pool = JobSizePool([mk("a", 8000, 1, 100.0)], horizon_sec=1000.0)
    _, _, which = bottleneck_load(pool, Capacity("c", 1, 8, 64), 0.01)
    assert which == "cpu"


def test_gpu_binds_when_cards_are_scarce():
    pool = JobSizePool([mk("a", 1000, 4, 100.0)], horizon_sec=1000.0)
    _, _, which = bottleneck_load(pool, Capacity("c", 1, 640, 2), 0.01)
    assert which == "gpu"


def test_a_gpuless_cluster_reports_zero_gpu_load():
    pool = JobSizePool([mk("a", 1000, 0, 100.0)], horizon_sec=1000.0)
    cpu, gpu, which = bottleneck_load(pool, Capacity("c", 1, 64, 0), 0.01)
    assert gpu == 0.0 and which == "cpu" and cpu > 0


# ── The calibration itself ──────────────────────────────────────────────────

def test_hand_computable_calibration():
    """One job type: 4 PE for 100 s on a 8-PE cluster over 1000 s.

    E[work] = 400 PE-s, so rho = lambda * 400 / 8. For rho = 0.5, lambda = 0.01/s,
    which over 1000 s is 10 arrivals.
    """
    pool = JobSizePool([mk("a", 4000, 0, 100.0)], horizon_sec=1000.0)
    plan = plan_load(pool, Capacity("c", 1, 8, 0), 0.5, 1000.0)
    assert plan.lambda_per_sec == pytest.approx(0.01)
    assert plan.n_task == 10
    assert plan.rho_cpu == pytest.approx(0.5)
    assert plan.bottleneck == "cpu"


def test_the_dominant_resource_gets_the_target_and_the_other_stays_below():
    pool = JobSizePool([mk("a", 1000, 4, 100.0)], horizon_sec=1000.0)
    plan = plan_load(pool, Capacity("c", 1, 640, 2), 0.5, 1000.0)
    assert plan.rho_gpu == pytest.approx(0.5)
    assert plan.rho_cpu < plan.rho_gpu
    assert plan.bottleneck == "gpu"


def test_a_pool_built_for_a_different_horizon_is_refused():
    """Truncation happens at pool construction, so a mismatched pool carries the wrong
    E[work] and would mis-calibrate silently."""
    pool = JobSizePool([mk("a", 4000, 0, 100.0)], horizon_sec=1000.0)
    with pytest.raises(CalibrationError, match="rebuild the pool"):
        plan_load(pool, Capacity("c", 1, 8, 0), 0.5, 2000.0)


def test_invalid_targets_are_refused():
    pool = JobSizePool([mk("a", 4000, 0, 100.0)], horizon_sec=1000.0)
    cap = Capacity("c", 1, 8, 0)
    with pytest.raises(CalibrationError, match="rho target must be positive"):
        plan_load(pool, cap, 0.0, 1000.0)
    with pytest.raises(CalibrationError, match="horizon must be positive"):
        plan_load(pool, cap, 0.5, 0.0)


def test_a_horizon_too_short_for_the_load_is_refused():
    """Rounding to zero arrivals must be an error, not an empty trace."""
    pool = JobSizePool([mk("a", 64000, 0, 1e6)], horizon_sec=1.0)
    with pytest.raises(CalibrationError, match="too short"):
        plan_load(pool, Capacity("c", 1, 64, 0), 1e-9, 1.0)


def test_load_scales_linearly_with_rho():
    pool = JobSizePool([mk("a", 4000, 0, 100.0)], horizon_sec=1000.0)
    cap = Capacity("c", 1, 8, 0)
    assert (plan_load(pool, cap, 0.8, 1000.0).n_task
            == pytest.approx(2 * plan_load(pool, cap, 0.4, 1000.0).n_task, abs=1))


# ── measured_load is an independent check, not a restatement ───────────────

def test_measured_load_is_hand_checkable():
    tasks = [mk("a", 4000, 2, 100.0), mk("b", 1000, 0, 200.0)]
    cpu, gpu = measured_load(tasks, Capacity("c", 1, 8, 4), 1000.0)
    assert cpu == pytest.approx((4 * 100 + 1 * 200) / (8 * 1000))
    assert gpu == pytest.approx((2 * 100) / (4 * 1000))


def test_measured_load_rejects_a_non_positive_horizon():
    with pytest.raises(CalibrationError):
        measured_load([mk("a", 1000, 0, 1.0)], Capacity("c", 1, 8, 0), 0.0)


# ── Locked values against the real trace (PLAN §3.6) ───────────────────────

@pytest.mark.parametrize("arm,scenario", sorted(SCENARIOS))
def test_arrival_count_reproduces_the_design_reference(arm, scenario, openb_tasks):
    rho, _, _, want_n, _ = SCENARIOS[(arm, scenario)]
    capacity, horizon = ARMS[arm]
    # capacity= matters here, not just in generation: excluding unplaceable jobs (W3.1)
    # changes E[work] and therefore the arrival count the design reference expects.
    pool = JobSizePool.from_trace(openb_tasks, horizon, capacity=capacity)
    plan = plan_load(pool, capacity, rho, horizon)
    assert plan.n_task == want_n


@pytest.mark.parametrize("arm,expected", [("homo", "cpu"), ("hetero", "gpu")])
def test_each_arm_binds_on_the_resource_the_plan_says_it_does(arm, expected, openb_tasks):
    """The reason calibration is on the dominant resource at all: hetero has 18 GPUs
    against 80, so calibrating it on CPU would demand rho_gpu ~ 2.6."""
    capacity, horizon = ARMS[arm]
    pool = JobSizePool.from_trace(openb_tasks, horizon)
    assert plan_load(pool, capacity, 0.85, horizon).bottleneck == expected


def test_calibrating_hetero_on_cpu_would_be_infeasible(openb_tasks):
    """Pin the number that forced the dominant-resource design."""
    capacity, horizon = ARMS["hetero"]
    pool = JobSizePool.from_trace(openb_tasks, horizon)
    m = pool.moments
    lam_cpu = 0.85 * capacity.total_pes / m.e_work        # naive CPU-only calibration
    _, rho_gpu, _ = bottleneck_load(pool, capacity, lam_cpu)
    assert rho_gpu > 2.0, "hetero must be GPU-infeasible under CPU-only calibration"
