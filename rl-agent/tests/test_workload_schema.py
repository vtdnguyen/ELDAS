"""W1.1 — canonical schema: derived quantities, I/O round-trip, validation."""

from __future__ import annotations

import os

import pytest

from workload import schema
from workload.schema import SchemaError, Task


def mk(name="t", cpu_milli=4000, memory_mib=1024, num_gpu=1, gpu_milli=0,
       creation=0.0, deletion=100.0, scheduled=None, qos="LS", phase="Running"):
    return Task(name=name, cpu_milli=cpu_milli, memory_mib=memory_mib, num_gpu=num_gpu,
                gpu_milli=gpu_milli, gpu_spec="", qos=qos, pod_phase=phase,
                creation_time=creation, deletion_time=deletion,
                scheduled_time=creation if scheduled is None else scheduled)


# ── Derived quantities must mirror the Java record exactly ─────────────────

def test_duration_uses_max_of_creation_and_scheduled():
    # AlibabaTraceReader.TaskRecord.duration(): deletion - max(creation, scheduled)
    assert mk(creation=10, scheduled=50, deletion=200).duration == 150
    assert mk(creation=50, scheduled=10, deletion=200).duration == 150


def test_duration_never_negative():
    assert mk(creation=100, deletion=50, scheduled=100).duration == 0.0


@pytest.mark.parametrize("cpu_milli,expected", [
    (0, 1),        # max(1, ...) floor
    (999, 1),      # integer division rounds down, floor applies
    (1000, 1),
    (12000, 12),
    (12999, 12),   # truncation, not rounding - matches Java integer division
])
def test_pes_matches_java_integer_division(cpu_milli, expected):
    assert mk(cpu_milli=cpu_milli).pes == expected


@pytest.mark.parametrize("num_gpu,gpu_milli,expected", [
    (0, 0, False),
    (1, 0, True),
    (0, 500, True),    # fractional share still needs a GPU host (affinity)
    (2, 1000, True),
])
def test_needs_gpu_counts_fractional_shares(num_gpu, gpu_milli, expected):
    assert mk(num_gpu=num_gpu, gpu_milli=gpu_milli).needs_gpu is expected


def test_gpu_capacity_uses_whole_cards_only():
    """A fractional-share task needs a GPU host but consumes zero whole cards.

    This asymmetry is deliberate and mirrors SimulationManager.canHost(), which tests
    affinity on (num_gpu > 0 or gpu_milli > 0) but capacity on num_gpu alone.
    """
    frac = mk(num_gpu=0, gpu_milli=500, deletion=100.0)
    assert frac.needs_gpu is True
    assert schema.gpu_work([frac]) == 0.0


# ── Transforms ──────────────────────────────────────────────────────────────

def test_with_arrival_preserves_duration_and_demand():
    t = mk(cpu_milli=8000, memory_mib=2048, num_gpu=2, creation=1000, scheduled=1500,
           deletion=3000)
    moved = t.with_arrival(77.0, name="wm1-000001")
    assert moved.name == "wm1-000001"
    assert moved.creation_time == moved.scheduled_time == 77.0
    assert moved.duration == t.duration == 1500
    assert moved.deletion_time == 77.0 + 1500
    assert (moved.cpu_milli, moved.memory_mib, moved.num_gpu) == (8000, 2048, 2)


def test_with_duration_is_measured_from_the_start_not_creation():
    t = mk(creation=10, scheduled=50, deletion=200)   # duration 150, start 50
    assert t.with_duration(20).duration == 20
    assert t.with_duration(20).deletion_time == 70    # start (50) + 20


def test_with_duration_rejects_negative():
    with pytest.raises(SchemaError):
        mk().with_duration(-1.0)


# ── I/O ─────────────────────────────────────────────────────────────────────

def test_write_read_round_trip_is_exact(tmp_path):
    tasks = [mk(name=f"t{i}", creation=i * 1.5, deletion=i * 1.5 + 0.1 * i,
                cpu_milli=1000 * i + 7, num_gpu=i % 3) for i in range(1, 20)]
    p = str(tmp_path / "out.csv")
    schema.write_csv(p, tasks)
    back = schema.read_csv(p)
    assert len(back) == len(tasks)
    for a, b in zip(tasks, back):
        assert a == b, "float formatting must round-trip bit-exactly"


def test_write_is_byte_deterministic(tmp_path):
    tasks = [mk(name=f"t{i}", creation=i * 0.3333333333333, deletion=i * 7.7 + 1)
             for i in range(1, 30)]
    a, b = str(tmp_path / "a.csv"), str(tmp_path / "b.csv")
    schema.write_csv(a, tasks)
    schema.write_csv(b, tasks)
    assert open(a, "rb").read() == open(b, "rb").read()


def test_written_header_matches_the_java_contract(tmp_path):
    p = str(tmp_path / "h.csv")
    schema.write_csv(p, [mk()])
    assert open(p, encoding="utf-8").readline().strip() == ",".join(schema.HEADER)


def test_read_skips_malformed_rows_but_rejects_a_bad_header(tmp_path):
    good = str(tmp_path / "g.csv")
    with open(good, "w", encoding="utf-8", newline="\n") as f:
        f.write(",".join(schema.HEADER) + "\n")
        f.write("a,1000,1,0,0,,LS,Running,0,10,0\n")
        f.write("truncated,row\n")                       # too few columns -> skipped
        f.write("b,nan,1,0,0,,LS,Running,5,15,5\n")      # nan tolerated as 0
    tasks = schema.read_csv(good)
    assert [t.name for t in tasks] == ["a", "b"]
    assert tasks[1].cpu_milli == 0

    bad = str(tmp_path / "b.csv")
    with open(bad, "w", encoding="utf-8", newline="\n") as f:
        f.write("name,cpu_milli\n1,2\n")
    with pytest.raises(SchemaError, match="expected 11 columns"):
        schema.read_csv(bad)


def test_read_missing_file_raises():
    with pytest.raises(SchemaError, match="not found"):
        schema.read_csv(os.path.join("nope", "missing.csv"))


def test_read_sorts_by_creation_time(tmp_path):
    p = str(tmp_path / "u.csv")
    with open(p, "w", encoding="utf-8", newline="\n") as f:
        f.write(",".join(schema.HEADER) + "\n")
        for name, c in (("late", 900), ("early", 5), ("mid", 100)):
            f.write(f"{name},1000,1,0,0,,LS,Running,{c},{c + 10},{c}\n")
    assert [t.name for t in schema.read_csv(p)] == ["early", "mid", "late"]


# ── Validation catches the failures that would otherwise be silent ──────────

def test_validate_rejects_unsorted_tasks():
    with pytest.raises(SchemaError, match="sorted by creation_time"):
        schema.validate_tasks([mk(name="a", creation=10), mk(name="b", creation=5)])


def test_validate_rejects_duplicate_names():
    with pytest.raises(SchemaError, match="duplicate"):
        schema.validate_tasks([mk(name="x", creation=0), mk(name="x", creation=1)])


def test_validate_rejects_completion_before_start():
    bad = Task("x", 1000, 1, 0, 0, "", "LS", "Running", 100.0, 50.0, 100.0)
    with pytest.raises(SchemaError, match="precedes start"):
        schema.validate_tasks([bad])


def test_validate_rejects_negative_demand():
    with pytest.raises(SchemaError, match="negative resource demand"):
        schema.validate_tasks([mk(cpu_milli=-1)])


def test_write_validates_by_default(tmp_path):
    with pytest.raises(SchemaError):
        schema.write_csv(str(tmp_path / "x.csv"),
                         [mk(name="a", creation=10), mk(name="b", creation=5)])


# ── Aggregates ──────────────────────────────────────────────────────────────

def test_cpu_and_gpu_work_are_hand_checkable():
    tasks = [mk(name="a", cpu_milli=4000, num_gpu=2, creation=0, deletion=10),
             mk(name="b", cpu_milli=1500, num_gpu=0, creation=0, deletion=20)]
    assert schema.cpu_work(tasks) == 4 * 10 + 1 * 20      # pes 4 and 1
    assert schema.gpu_work(tasks) == 2 * 10 + 0 * 20


def test_schedulable_reproduces_the_java_filter_then_the_wm1_filter():
    tasks = [mk(name="r", phase="Running"), mk(name="p", phase="Pending"),
             mk(name="f", phase="Failed"), mk(name="s", phase="Succeeded")]
    # ScenarioFilter.java drops only Pending
    assert [t.name for t in schema.schedulable(tasks)] == ["r", "f", "s"]
    # WM-1 additionally drops Failed
    assert [t.name for t in schema.schedulable(tasks, drop_failed=True)] == ["r", "s"]
