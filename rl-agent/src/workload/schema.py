"""Canonical task record for the WM-1 workload pipeline (PLAN-Workload-Model.md §4.2).

This module is the **single source of truth** for the trace format ELDAS consumes: the
11-column Alibaba openb schema. Everything downstream — `jobsize`, `arrivals`,
`calibrate`, `wm1` — works on :class:`Task` and never touches a CSV directly, which is
what lets a new source trace be added as one adapter (PLAN §7).

Derived quantities mirror the Java side **exactly** so a generated trace measures the
same in Python and in the simulator:

    duration = max(0, deletion_time - max(creation_time, scheduled_time))
                                        -> AlibabaTraceReader.TaskRecord.duration()
    pes      = max(1, cpu_milli // 1000)
                                        -> AlibabaTraceReader.TaskRecord.pesNeeded()
    needs_gpu = num_gpu > 0 or gpu_milli > 0
                                        -> SimulationManager.canHost() affinity test

Note the asymmetry, which is not a bug: ``needs_gpu`` (affinity) counts a fractional
share, but GPU *capacity* is consumed in whole cards (``num_gpu``) because that is what
``canHost`` checks.
"""

from __future__ import annotations

import csv
import os
from dataclasses import dataclass, replace

# ── CSV contract ────────────────────────────────────────────────────────────

HEADER = (
    "name", "cpu_milli", "memory_mib", "num_gpu", "gpu_milli",
    "gpu_spec", "qos", "pod_phase", "creation_time", "deletion_time", "scheduled_time",
)
MIN_COLUMNS = len(HEADER)          # 11 — matches AlibabaTraceReader.MIN_COLUMNS

#: Pod phases that never ran and carry no meaningful duration. ScenarioFilter.java
#: drops only ``Pending``; WM-1 additionally drops ``Failed`` (PLAN §1.3-B / §3.1).
PHASE_PENDING = "Pending"
PHASE_FAILED = "Failed"


class SchemaError(ValueError):
    """Raised when a trace violates the openb contract in a way that must not pass."""


# ── Task ────────────────────────────────────────────────────────────────────

@dataclass(frozen=True, slots=True)
class Task:
    name: str
    cpu_milli: int
    memory_mib: int
    num_gpu: int
    gpu_milli: int
    gpu_spec: str
    qos: str
    pod_phase: str
    creation_time: float
    deletion_time: float
    scheduled_time: float

    # ── derived (mirrors of the Java record) ────────────────────────────────

    @property
    def duration(self) -> float:
        return max(0.0, self.deletion_time - max(self.creation_time, self.scheduled_time))

    @property
    def pes(self) -> int:
        return max(1, self.cpu_milli // 1000)

    @property
    def needs_gpu(self) -> bool:
        """Affinity signal — whole cards OR a fractional share."""
        return self.num_gpu > 0 or self.gpu_milli > 0

    # ── transforms ──────────────────────────────────────────────────────────

    def with_arrival(self, t: float, *, name: str | None = None) -> "Task":
        """Move this job to arrival time ``t``, preserving demand and duration.

        The generated trace assumes instant dispatch (``scheduled_time == creation_time``)
        so that ``duration`` round-trips unchanged: the simulator, not the trace, decides
        when a task actually starts.
        """
        d = self.duration
        return replace(self, name=name if name is not None else self.name,
                       creation_time=t, scheduled_time=t, deletion_time=t + d)

    def with_duration(self, d: float) -> "Task":
        """Return a copy whose duration is ``d`` (used by horizon truncation)."""
        if d < 0:
            raise SchemaError(f"negative duration for {self.name!r}: {d}")
        base = max(self.creation_time, self.scheduled_time)
        return replace(self, deletion_time=base + d)


# ── Parsing helpers (tolerant, matching AlibabaTraceReader) ─────────────────

def _as_int(s: str, fallback: int = 0) -> int:
    s = s.strip()
    if not s or s.lower() == "nan":
        return fallback
    return int(float(s))


def _as_float(s: str, fallback: float = 0.0) -> float:
    s = s.strip()
    if not s or s.lower() == "nan":
        return fallback
    return float(s)


# ── I/O ─────────────────────────────────────────────────────────────────────

def read_csv(path: str) -> list[Task]:
    """Parse an openb-schema CSV, sorted by ``creation_time``.

    Malformed rows are skipped rather than fatal, matching the Java reader's tolerance
    (a handful of bad rows in a large trace must not abort an experiment). A file whose
    *header* is wrong is a different matter and raises, because that means the caller
    handed us the wrong kind of file.
    """
    if not os.path.isfile(path):
        raise SchemaError(f"trace not found: {path}")

    tasks: list[Task] = []
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader, None)
        if header is None:
            raise SchemaError(f"trace is empty: {path}")
        if len(header) < MIN_COLUMNS:
            raise SchemaError(
                f"{path}: expected {MIN_COLUMNS} columns {HEADER}, got {len(header)}: {header}")

        for row in reader:
            if len(row) < MIN_COLUMNS:
                continue
            try:
                tasks.append(Task(
                    name=row[0].strip(),
                    cpu_milli=_as_int(row[1]),
                    memory_mib=_as_int(row[2]),
                    num_gpu=_as_int(row[3]),
                    gpu_milli=_as_int(row[4]),
                    gpu_spec=row[5].strip(),
                    qos=row[6].strip(),
                    pod_phase=row[7].strip(),
                    creation_time=_as_float(row[8]),
                    deletion_time=_as_float(row[9]),
                    scheduled_time=_as_float(row[10]),
                ))
            except (ValueError, IndexError):
                continue

    tasks.sort(key=lambda t: t.creation_time)
    return tasks


def write_csv(path: str, tasks: list[Task], *, validate: bool = True) -> None:
    """Write tasks as an openb-schema CSV that the Java reader accepts unchanged.

    Floats use ``repr`` so a read-back round-trips bit-exactly — the "same seed produces
    an identical file" guarantee (W1.6) depends on this being deterministic.
    """
    if validate:
        validate_tasks(tasks)
    parent = os.path.dirname(os.path.abspath(path))
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(HEADER)
        for t in tasks:
            w.writerow([
                t.name, t.cpu_milli, t.memory_mib, t.num_gpu, t.gpu_milli,
                t.gpu_spec, t.qos, t.pod_phase,
                repr(float(t.creation_time)), repr(float(t.deletion_time)),
                repr(float(t.scheduled_time)),
            ])


def validate_tasks(tasks: list[Task]) -> None:
    """Fail loudly on the mistakes that would otherwise surface as wrong physics.

    An unsorted trace silently breaks the simulator's energy integration (it advances
    time to each arrival in order), and a negative duration produces a task that
    completes before it starts — both are the kind of bug that yields plausible-looking
    numbers rather than a crash.
    """
    prev = float("-inf")
    seen: set[str] = set()
    for t in tasks:
        if t.creation_time < prev:
            raise SchemaError(
                f"tasks must be sorted by creation_time; {t.name!r} at {t.creation_time} "
                f"follows {prev}")
        prev = t.creation_time
        if t.name in seen:
            raise SchemaError(f"duplicate task name: {t.name!r}")
        seen.add(t.name)
        if t.deletion_time < max(t.creation_time, t.scheduled_time):
            raise SchemaError(
                f"{t.name!r}: deletion_time {t.deletion_time} precedes start "
                f"{max(t.creation_time, t.scheduled_time)}")
        if t.cpu_milli < 0 or t.memory_mib < 0 or t.num_gpu < 0 or t.gpu_milli < 0:
            raise SchemaError(f"{t.name!r}: negative resource demand")


# ── Aggregates (shared by jobsize, calibrate and the acceptance tests) ──────

def cpu_work(tasks) -> float:
    """Total CPU demand in PE-seconds."""
    return sum(t.pes * t.duration for t in tasks)


def gpu_work(tasks) -> float:
    """Total GPU demand in card-seconds — whole cards, matching ``canHost``."""
    return sum(t.num_gpu * t.duration for t in tasks)


def schedulable(tasks, *, drop_failed: bool = False) -> list[Task]:
    """Apply the phase filter. ``drop_failed=False`` reproduces ScenarioFilter.java."""
    drop = {PHASE_PENDING, PHASE_FAILED} if drop_failed else {PHASE_PENDING}
    return [t for t in tasks if t.pod_phase not in drop]
