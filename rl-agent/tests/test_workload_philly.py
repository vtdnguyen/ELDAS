"""W4.2 — Philly/Helios adapter + the burstiness anchors it must reproduce.

Acceptance from PLAN-Workload-Model.md W4.2/W4.3:

    the adapter reconstructs arrivals from ``interval`` by cumulative sum
    its measured C_a^2 / IDC(1h) / within-3d median match the W4.3 anchors (< 5 %)
    it REFUSES to act as a job-size source, loudly

The anchors in ``_workload_fixtures`` come from ``scripts/wm1-design-reference.py
--anchors``, which parses the CSV directly and shares no code with ``workload/``. Agreement
between the two is what makes the number in §3.5 an independent measurement rather than
this module agreeing with itself.
"""

from __future__ import annotations

import os

import pytest

from workload import adapters, burstiness, schema, wm1
from workload.adapters import AdapterError
from workload.adapters.philly import EXPECTED_HEADER, JOBSIZE_REFUSAL

from _workload_fixtures import REFERENCE_ANCHORS, REFERENCE_TRACES

ANCHOR_TOL = 0.05          # PLAN W4.3: "adapter cho số lệch > 5% ⇒ adapter sai"


def _trace(name):
    path = REFERENCE_TRACES.get(name)
    if path is None:
        pytest.skip(f"{name} reference trace not found; "
                    f"run scripts/fetch-reference-traces.sh")
    return path


# ── Registration ────────────────────────────────────────────────────────────

def test_both_reference_traces_are_registered():
    assert {"philly", "helios"} <= set(adapters.list_adapters())


@pytest.mark.parametrize("name", ["philly", "helios"])
def test_capabilities_declare_arrivals_and_gpu_only(name):
    caps = adapters.get(name).capabilities()
    assert adapters.CAP_ARRIVALS in caps
    assert adapters.CAP_GPU in caps
    # The withheld ones are the point of the interlock.
    assert adapters.CAP_JOBSIZE not in caps
    assert adapters.CAP_QOS not in caps
    assert adapters.CAP_MEMORY not in caps


# ── The refusal (the reason this adapter is fenced at all) ──────────────────

@pytest.mark.parametrize("name", ["philly", "helios"])
def test_requiring_jobsize_raises_with_the_reason(name):
    with pytest.raises(AdapterError, match="cannot supply"):
        adapters.require(name, adapters.CAP_JOBSIZE)


@pytest.mark.parametrize("name", ["philly", "helios"])
def test_refuse_jobsize_names_the_missing_columns(name):
    """A bare "unsupported" would send the reader back to the source files."""
    with pytest.raises(AdapterError) as e:
        adapters.get(name).refuse_jobsize()
    msg = str(e.value)
    assert "memory" in msg and "QoS" in msg and "wall_time" in msg


def test_generation_refuses_a_reference_trace_as_its_source(tmp_path):
    """The failure this prevents is silent, not loud.

    Without the interlock WM-1 would emit a trace with memory_mib = 0 and qos = "" for
    every task; the simulator schedules that without complaint, every task fits every
    host, and the campaign produces normal-looking numbers for a workload nobody
    characterised.
    """
    with pytest.raises(AdapterError, match="cannot supply"):
        wm1.generate_arm(_trace("philly"), "homo", str(tmp_path),
                         source_adapter="philly", scenarios=("LOW",), seeds=(42,),
                         verbose=False)


# ── Parsing ─────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("name", ["philly", "helios"])
def test_arrivals_are_the_cumulative_sum_of_interval(name):
    import csv

    path = _trace(name)
    tasks = adapters.load(name, path)
    with open(path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    assert len(tasks) == len(rows)
    running = 0.0
    for task, row in zip(tasks[:200], rows[:200]):
        running += float(row["interval"])
        assert task.creation_time == pytest.approx(running, abs=1e-9)


@pytest.mark.parametrize("name", ["philly", "helios"])
def test_duration_is_run_time_and_dispatch_is_instant(name):
    import csv

    path = _trace(name)
    tasks = adapters.load(name, path)
    with open(path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    for task, row in zip(tasks[:200], rows[:200]):
        assert task.duration == pytest.approx(float(row["run_time"]), abs=1e-6)
        # No queue-wait column exists, so the adapter must not invent one.
        assert task.scheduled_time == task.creation_time


def test_failed_jobs_carry_the_openb_phase_so_the_pool_filter_would_see_them():
    tasks = adapters.load("philly", _trace("philly"))
    assert any(t.pod_phase == schema.PHASE_FAILED for t in tasks)
    # Killed jobs still occupied their GPUs for run_time, so they are not "failed".
    assert any(t.pod_phase == "Running" for t in tasks)


def test_philly_has_no_cpu_and_says_so():
    """Verified limitation, pinned: cpu_num is 0 in every Philly row."""
    tasks = adapters.load("philly", _trace("philly"))
    assert all(t.cpu_milli == 0 for t in tasks)
    assert adapters.get("philly").has_cpu is False
    assert adapters.get("helios").has_cpu is True


def test_helios_does_have_cpu_values():
    tasks = adapters.load("helios", _trace("helios"))
    assert any(t.cpu_milli > 0 for t in tasks)


def test_names_are_unique_so_validate_tasks_would_accept_them():
    tasks = adapters.load("helios", _trace("helios"))
    assert len({t.name for t in tasks}) == len(tasks)


# ── Loading refuses the wrong file rather than mapping columns at random ────

def test_a_changed_header_is_rejected(tmp_path):
    p = tmp_path / "bad.csv"
    p.write_text("u_id,user,gpu_num\n0,1,1\n", encoding="utf-8")
    with pytest.raises(schema.SchemaError, match="unexpected header"):
        adapters.load("philly", str(p))


def test_a_negative_interval_is_refused_not_clamped(tmp_path):
    """Clamping would fabricate a simultaneous arrival — i.e. invent a burst."""
    p = tmp_path / "neg.csv"
    p.write_text(",".join(EXPECTED_HEADER) + "\n"
                 "0,1,1,0,1,0,100,0,Pass\n"
                 "1,1,1,0,1,-5,100,0,Pass\n", encoding="utf-8")
    with pytest.raises(schema.SchemaError, match="negative interval"):
        adapters.load("philly", str(p))


def test_a_missing_file_points_at_the_fetch_script(tmp_path):
    with pytest.raises(schema.SchemaError, match="fetch-reference-traces"):
        adapters.load("philly", str(tmp_path / "nope.csv"))


# ── W4.3 anchors: the numbers §3.5 rests on ────────────────────────────────

@pytest.mark.parametrize("name", ["philly", "helios"])
def test_adapter_reproduces_the_design_reference_anchors(name):
    want = REFERENCE_ANCHORS[name]
    got = burstiness.profile_adapter(name, _trace(name))

    assert got.squared_cv == pytest.approx(want["ca2"], rel=ANCHOR_TOL)
    assert got.idc_whole == pytest.approx(want["idc_whole"], rel=ANCHOR_TOL)
    assert got.idc_window_median == pytest.approx(want["idc_3d_median"], rel=ANCHOR_TOL)
    assert got.idc_window_q1 == pytest.approx(want["idc_3d_q1"], rel=ANCHOR_TOL)
    assert got.idc_window_q3 == pytest.approx(want["idc_3d_q3"], rel=ANCHOR_TOL)
    assert got.n_windows == want["n_windows"]


@pytest.mark.parametrize("name", ["philly", "helios"])
def test_within_window_burstiness_is_below_whole_trace_burstiness(name):
    """The validity argument behind §3.5, stated as a check.

    Whole-trace IDC is inflated by week-to-week rate drift over ~2 months, which an agent
    inside a 4-day episode never experiences as burstiness. If this ever inverted, the
    reasoning for targeting 40 instead of 160 would need revisiting.
    """
    got = burstiness.profile_adapter(name, _trace(name))
    assert got.idc_window_median < got.idc_whole


def test_the_burst_target_is_bracketed_by_the_two_clusters():
    """IDC(1h) = 40 must sit between the two anchors, or it is not an anchored choice."""
    from workload.wm1 import SCENARIOS

    _, burst_target = SCENARIOS["BURST"]
    philly = burstiness.profile_adapter("philly", _trace("philly")).idc_window_median
    helios = burstiness.profile_adapter("helios", _trace("helios")).idc_window_median
    assert min(philly, helios) < burst_target < max(philly, helios)


def test_the_low_high_target_is_anchored_on_openb_itself():
    from workload.wm1 import SCENARIOS

    _, high_target = SCENARIOS["HIGH"]
    trace = REFERENCE_TRACES.get("openb")
    if trace is None:
        pytest.skip("openb trace not found")
    openb = burstiness.profile_adapter(
        "openb", trace, exclude_phases=burstiness.SCHEDULABLE_ONLY).idc_whole
    # "burstiness like the trace we already use": 10 against a measured 12.0.
    assert openb == pytest.approx(12.0, rel=0.05)
    assert high_target == pytest.approx(openb, rel=0.25)


def test_the_openb_population_matters_and_is_pinned():
    """Three defensible populations, three different numbers — so it must be stated.

    Quoting 13.5 (all rows) next to a Philly figure computed over real submissions only
    would read as a difference in burstiness when it is a difference in who was counted.
    """
    trace = REFERENCE_TRACES.get("openb")
    if trace is None:
        pytest.skip("openb trace not found")
    whole = burstiness.profile_adapter("openb", trace)
    sched = burstiness.profile_adapter("openb", trace,
                                       exclude_phases=burstiness.SCHEDULABLE_ONLY)
    assert whole.n_arrivals == 8152 and whole.idc_whole == pytest.approx(13.5, rel=0.05)
    assert sched.n_arrivals == 7255 and sched.idc_whole == pytest.approx(12.0, rel=0.05)


@pytest.mark.parametrize("name", ["philly", "helios"])
def test_few_windows_produce_an_explicit_caveat(name):
    """The IQR comes from 6-8 disjoint windows; the report must not present it as more."""
    got = burstiness.profile_adapter(name, _trace(name))
    assert got.n_windows < 10
    assert "windows" in (got.caveat() or "")


# ── The measurement helper itself ──────────────────────────────────────────

def test_window_idcs_are_disjoint_not_sliding():
    """Overlapping windows share arrivals, so their IDCs correlate and the IQR would look
    tighter than the data supports."""
    times = [float(i) for i in range(0, 100)]
    got = burstiness.window_idcs(times, window_sec=25.0, counting_window_sec=5.0,
                                 min_arrivals=1)
    assert len(got) == 3          # [0,25) [25,50) [50,75); [75,100) needs t+W <= t1


def test_a_perfectly_regular_stream_has_zero_dispersion():
    times = [float(i) for i in range(0, 1000)]
    got = burstiness.profile("regular", times)
    # One arrival per second exactly: every counting window holds the same number.
    assert got.idc_whole == pytest.approx(0.0, abs=1e-9)
    assert got.squared_cv == pytest.approx(0.0, abs=1e-9)


def test_profile_needs_at_least_two_arrivals():
    with pytest.raises(ValueError, match="at least 2"):
        burstiness.profile("x", [1.0])


def test_windows_below_the_arrival_floor_are_excluded():
    # 3 arrivals in day 1, 60 in day 2. The trailing point at 172800 is what makes day 2
    # a *complete* window (the loop only evaluates windows that fit before the last
    # arrival), which is also why the real traces yield 8 and 6 windows rather than 19/6.
    times = [0.0, 1.0, 2.0] + [86400.0 + i for i in range(60)] + [172800.0]
    got = burstiness.window_idcs(times, window_sec=86400.0, min_arrivals=50)
    assert len(got) == 1
