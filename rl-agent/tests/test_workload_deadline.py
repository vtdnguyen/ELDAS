"""W1.5 — deadline model: bounded slowdown plus an absolute floor.

Acceptance targets from PLAN-Workload-Model.md W1.5:
    Java B16 still passes (checked in Docker, see Tracking)
    absolute slack >= 100 s for every QoS class
    share of tasks with slack < 60 s falls from 24.6 % to 0 %
    Python and Java compute the same deadline

The last one is a genuine cross-language check, not a restatement: ValidationRunner B20
sums every deadline in the schedulable trace and prints the total, and the value is
pinned here. Any divergence in a constant, a formula, or the ordering of the trace shows
up as a mismatch.
"""

from __future__ import annotations

import pytest

from eval import qos as eval_qos
from workload import deadline as dl
from workload import schema
from workload.schema import Task

from _workload_fixtures import openb_tasks  # noqa: F401

#: Printed by ValidationRunner B20 ("deadline checksum") on the openb trace,
#: HIGH scenario (Pending excluded), homogeneous topology.
JAVA_DEADLINE_CHECKSUM = 83847610799.300020
JAVA_MIN_BUDGET_SEC = 107.3
JAVA_FLOORS = {"LS": 106.0, "Guaranteed": 212.0, "Burstable": 530.0, "BE": 2120.0}

QOS_CLASSES = ("LS", "Guaranteed", "Burstable", "BE")


def mk(qos="LS", duration=600.0, creation=0.0):
    return Task(name="t", cpu_milli=1000, memory_mib=1, num_gpu=0, gpu_milli=0,
                gpu_spec="", qos=qos, pod_phase="Running",
                creation_time=creation, deletion_time=creation + duration,
                scheduled_time=creation)


# ── The floor itself ────────────────────────────────────────────────────────

@pytest.mark.parametrize("qos", QOS_CLASSES)
def test_floor_matches_java(qos):
    assert dl.slack_floor(qos) == pytest.approx(JAVA_FLOORS[qos])


def test_floors_are_ordered_by_strictness():
    """A stricter class must tolerate LESS absolute lateness.

    The per-class-percentile design this replaced inverted exactly this ordering
    (Guaranteed 2 s against LS 107 s) because Guaranteed has only 7 samples.
    """
    floors = [dl.slack_floor(q) for q in QOS_CLASSES]
    assert floors == sorted(floors)


@pytest.mark.parametrize("qos", QOS_CLASSES)
def test_every_class_tolerates_at_least_100s(qos):
    assert dl.slack_floor(qos) >= 100.0


def test_ls_floor_dwarfs_the_wake_latency():
    """The whole point: a 5 s wake must not be able to violate an SLO on its own."""
    assert dl.slack_floor("LS") > 10 * 5.0


def test_unknown_qos_falls_back_without_crashing():
    assert dl.slack_floor("SomethingElse") == pytest.approx(106.0 * 1.5)
    assert dl.qos_slack_factor("SomethingElse") == 1.5


# ── The deadline formula ────────────────────────────────────────────────────

def test_deadline_is_slowdown_allowance_plus_floor():
    t = mk(qos="LS", duration=600.0, creation=1000.0)
    assert dl.deadline(t) == pytest.approx(1000.0 + 600.0 * 1.1 + 106.0)


def test_slack_budget_is_the_absolute_lateness_allowed():
    t = mk(qos="LS", duration=600.0, creation=1000.0)
    assert dl.slack_budget(t) == pytest.approx(dl.deadline(t) - t.creation_time - t.duration)
    assert dl.slack_budget(t) == pytest.approx(600.0 * 0.1 + 106.0)


def test_a_zero_length_job_still_gets_the_full_floor():
    """The degenerate case the floor exists for."""
    assert dl.slack_budget(mk(qos="LS", duration=0.0)) == pytest.approx(106.0)


def test_budget_grows_with_duration_and_with_looseness():
    assert dl.slack_budget(mk("LS", 6000.0)) > dl.slack_budget(mk("LS", 600.0))
    assert dl.slack_budget(mk("BE", 600.0)) > dl.slack_budget(mk("LS", 600.0))


# ── Mirrors must agree ──────────────────────────────────────────────────────

@pytest.mark.parametrize("qos", QOS_CLASSES + ("Unknown",))
def test_eval_qos_mirror_agrees(qos):
    """`eval/qos.py` feeds the static NSGA-II baseline; if it drifts, the static and
    DES SLA costs stop being in the same units (CLAUDE.md Lưu ý #5)."""
    assert eval_qos.qos_slack_floor_sec(qos) == dl.slack_floor(qos)
    assert eval_qos.qos_slack_factor(qos) == dl.qos_slack_factor(qos)


def test_trace_loader_deadline_matches(openb_tasks):
    from eval import trace_loader as tl
    from _workload_fixtures import TRACE_PATH
    theirs = tl.read_trace(TRACE_PATH)
    mine = {t.name: t for t in openb_tasks}
    checked = 0
    for t in theirs[:500]:
        if t.pod_phase == "Pending":
            continue
        assert t.deadline == pytest.approx(dl.deadline(mine[t.name]), rel=1e-12)
        checked += 1
    assert checked > 100


# ── Cross-language contract against Java ────────────────────────────────────

def test_deadline_checksum_matches_validationrunner_b20(openb_tasks):
    sched = schema.schedulable(openb_tasks)          # Pending excluded, like B20
    total = 0.0
    for t in sched:
        total += dl.deadline(t)
    assert len(sched) == 7255
    assert total == pytest.approx(JAVA_DEADLINE_CHECKSUM, rel=1e-12), (
        "Python and Java disagree on the deadline; a constant or the formula drifted "
        "in one of SimulationConfig.java / workload/deadline.py / eval/qos.py")


def test_minimum_budget_matches_java(openb_tasks):
    budgets = [dl.slack_budget(t) for t in schema.schedulable(openb_tasks)]
    assert min(budgets) == pytest.approx(JAVA_MIN_BUDGET_SEC, abs=0.1)


# ── The degeneracy this change exists to remove ────────────────────────────

def test_no_task_is_left_with_a_sub_minute_budget(openb_tasks):
    budgets = [dl.slack_budget(t) for t in schema.schedulable(openb_tasks)]
    assert sum(1 for b in budgets if b < 60.0) == 0


def test_the_old_multiplicative_only_model_really_was_degenerate(openb_tasks):
    """Pin the 'before' number so the improvement is auditable, not asserted."""
    sched = schema.schedulable(openb_tasks)
    old = [t.duration * (dl.qos_slack_factor(t.qos) - 1.0) for t in sched]
    share = sum(1 for b in old if b < 60.0) / len(old)
    assert share == pytest.approx(0.246, abs=0.005)


def test_median_budget_improves_by_more_than_a_factor_of_two(openb_tasks):
    sched = schema.schedulable(openb_tasks)
    old = sorted(t.duration * (dl.qos_slack_factor(t.qos) - 1.0) for t in sched)
    new = sorted(dl.slack_budget(t) for t in sched)
    assert new[len(new) // 2] > 2 * old[len(old) // 2]


# ── The measurement behind the constant ────────────────────────────────────

def test_measured_lag_p90_reproduces_the_hardcoded_constant(openb_tasks):
    """The constant is measured, not chosen - so re-measuring must reproduce it."""
    assert dl.measure_scheduling_lag_p90(openb_tasks) == pytest.approx(
        dl.OPENB_SCHEDULING_LAG_P90_SEC, abs=1.0)


def test_pending_pods_are_excluded_from_the_lag_measurement():
    """Pending pods have no scheduled_time; parsed as 0 it would read as a huge
    negative lag and drag the percentile down."""
    tasks = [Task("a", 1000, 1, 0, 0, "", "LS", "Running", 100.0, 200.0, 150.0),
             Task("b", 1000, 1, 0, 0, "", "LS", "Pending", 100.0, 0.0, 0.0)]
    assert dl.measure_scheduling_lag_p90(tasks) == 50.0


def test_measuring_with_no_schedulable_tasks_raises():
    with pytest.raises(ValueError, match="no schedulable"):
        dl.measure_scheduling_lag_p90(
            [Task("b", 1000, 1, 0, 0, "", "LS", "Pending", 0.0, 0.0, 0.0)])
