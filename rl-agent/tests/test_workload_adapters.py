"""W1.1 — adapter registry, capability interlock, and the openb adapter.

Acceptance targets from PLAN-Workload-Model.md W1.1:
    adapters.load("openb", <trace>) -> 8 152 tasks
    after dropping Pending + Failed -> 5 385
    sum(pes x duration) matches scripts/characterize-workload.py to 1e-12
"""

from __future__ import annotations

import pytest

from workload import adapters, schema
from workload.adapters import AdapterError
from workload.adapters.openb import OpenbAdapter

from _workload_fixtures import OPENB_CPU_WORK_SCHEDULABLE, openb_tasks  # noqa: F401


# ── Registry ────────────────────────────────────────────────────────────────

def test_openb_is_registered():
    assert "openb" in adapters.list_adapters()
    assert isinstance(adapters.get("openb"), OpenbAdapter)


def test_unknown_adapter_names_the_available_ones():
    with pytest.raises(AdapterError, match="unknown adapter 'nope'"):
        adapters.get("nope")


def test_registering_a_duplicate_name_is_an_error_not_a_silent_swap():
    class Clash:
        name = "openb"

        def capabilities(self):
            return frozenset()

        def load(self, path):
            return []

        def native_arrivals(self, tasks):
            return None

    with pytest.raises(AdapterError, match="already registered"):
        adapters.register(Clash())


def test_registering_the_same_object_twice_is_idempotent():
    same = adapters.get("openb")
    assert adapters.register(same) is same


def test_unknown_capability_strings_are_rejected():
    class Weird:
        name = "weird-caps"

        def capabilities(self):
            return frozenset({"telepathy"})

        def load(self, path):
            return []

        def native_arrivals(self, tasks):
            return None

    with pytest.raises(AdapterError, match="unknown capabilities"):
        adapters.register(Weird())


def test_adapter_without_a_name_is_rejected():
    class Anon:
        name = ""

        def capabilities(self):
            return frozenset()

        def load(self, path):
            return []

        def native_arrivals(self, tasks):
            return None

    with pytest.raises(AdapterError, match="no name"):
        adapters.register(Anon())


def test_openb_satisfies_the_protocol():
    assert isinstance(adapters.get("openb"), adapters.TraceAdapter)


# ── Capability interlock (PLAN §7/§8) ───────────────────────────────────────

def test_openb_declares_everything_wm1_needs():
    caps = adapters.get("openb").capabilities()
    assert {adapters.CAP_JOBSIZE, adapters.CAP_ARRIVALS,
            adapters.CAP_QOS, adapters.CAP_MEMORY, adapters.CAP_GPU} <= caps


def test_require_passes_when_capabilities_are_present():
    assert adapters.require("openb", adapters.CAP_JOBSIZE, adapters.CAP_QOS).name == "openb"


def test_require_blocks_a_trace_that_cannot_supply_job_sizes():
    """The interlock that keeps Philly/Helios out of the job-size path.

    Those traces have no memory and no QoS column, and without this guard WM-1 would
    silently emit a trace with memory_mib = 0 everywhere.
    """
    class ArrivalsOnly:
        name = "arrivals-only-fixture"

        def capabilities(self):
            return frozenset({adapters.CAP_ARRIVALS, adapters.CAP_GPU})

        def load(self, path):
            return []

        def native_arrivals(self, tasks):
            return [0.0]

    adapters.register(ArrivalsOnly())
    adapters.require("arrivals-only-fixture", adapters.CAP_ARRIVALS)   # fine
    with pytest.raises(AdapterError, match="cannot supply"):
        adapters.require("arrivals-only-fixture", adapters.CAP_JOBSIZE, adapters.CAP_QOS)


# ── openb adapter against the real trace ────────────────────────────────────

def test_openb_loads_the_expected_task_count(openb_tasks):
    assert len(openb_tasks) == 8152


def test_phase_counts_match_the_measured_trace(openb_tasks):
    counts = {}
    for t in openb_tasks:
        counts[t.pod_phase] = counts.get(t.pod_phase, 0) + 1
    assert counts == {"Running": 5193, "Failed": 1870, "Pending": 897, "Succeeded": 192}


def test_scenariofilter_parity_then_wm1_pool_size(openb_tasks):
    # ScenarioFilter.java drops Pending only -> LEGACY_HIGH
    assert len(schema.schedulable(openb_tasks)) == 8152 - 897 == 7255
    # WM-1 additionally drops Failed (PLAN W1.1 acceptance: 5 385)
    assert len(schema.schedulable(openb_tasks, drop_failed=True)) == 5385


def test_cpu_work_matches_characterize_workload_exactly(openb_tasks):
    """Cross-tool agreement on the quantity every load figure is built from."""
    got = schema.cpu_work(schema.schedulable(openb_tasks))
    assert abs(got - OPENB_CPU_WORK_SCHEDULABLE) <= 1e-12 * OPENB_CPU_WORK_SCHEDULABLE


def test_tasks_come_back_sorted(openb_tasks):
    times = [t.creation_time for t in openb_tasks]
    assert times == sorted(times)


def test_native_arrivals_are_the_creation_times(openb_tasks):
    arrivals = adapters.get("openb").native_arrivals(openb_tasks)
    assert arrivals == [t.creation_time for t in openb_tasks]


def test_the_snapshot_caveat_is_recorded_on_the_adapter():
    """The diagnosis that motivated GĐ 2.4 must live in code, not only in a document."""
    from workload.adapters import openb as openb_mod
    assert "snapshot" in openb_mod.ARRIVAL_CAVEAT.lower()


def test_openb_snapshot_property_still_holds(openb_tasks):
    """Guard the premise of PLAN §1.3-C: if a future trace refresh makes deletion times
    start near zero, the 'snapshot' diagnosis no longer applies and the plan needs
    revisiting rather than silently carrying on."""
    sched = schema.schedulable(openb_tasks)
    assert min(t.deletion_time for t in sched) > 9.0e6
    assert max(t.creation_time for t in sched) > 1.2e7
