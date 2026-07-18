"""Alibaba trace loader + scenario filter (Python mirror of Java
``AlibabaTraceReader`` + ``ScenarioFilter``) for the static NSGA-II baseline.

Only the fields the static model needs are kept. The derived quantities
(``duration``, ``deadline``, ``pes_needed``, ``qos_weight``) are computed with
the *same* formulas as the Java side so the static SLA cost is comparable in
units to the DES ``C_SLA`` (CLAUDE.md Lưu ý #5).
"""

from __future__ import annotations

import csv
from dataclasses import dataclass

from . import qos as qos_mod


# ── Task record ─────────────────────────────────────────────────────────────

@dataclass(frozen=True, slots=True)
class Task:
    name: str
    cpu_milli: int
    memory_mib: int
    num_gpu: int
    gpu_milli: int
    qos: str
    pod_phase: str
    creation_time: float
    deletion_time: float
    scheduled_time: float

    @property
    def duration(self) -> float:
        """Estimated execution duration (s) — matches Java TaskRecord.duration()."""
        return max(0.0, self.deletion_time - max(self.creation_time, self.scheduled_time))

    @property
    def pes_needed(self) -> int:
        """CPU PEs (≥ 1) — matches Java TaskRecord.pesNeeded()."""
        return max(1, self.cpu_milli // 1000)

    @property
    def qos_weight(self) -> float:
        return qos_mod.qos_weight(self.qos)

    @property
    def deadline(self) -> float:
        """creation + duration × slack(qos) — matches Java derivation."""
        return self.creation_time + self.duration * qos_mod.qos_slack_factor(self.qos)

    @property
    def needs_gpu(self) -> bool:
        """G2.2 affinity signal — whole cards OR a fractional share."""
        return self.num_gpu > 0 or self.gpu_milli > 0


# ── CSV parsing ─────────────────────────────────────────────────────────────

def _parse_int(s: str, fallback: int = 0) -> int:
    s = s.strip()
    if not s or s.lower() == "nan":
        return fallback
    return int(float(s))


def _parse_float(s: str, fallback: float = 0.0) -> float:
    s = s.strip()
    if not s or s.lower() == "nan":
        return fallback
    return float(s)


def read_trace(csv_path: str) -> list[Task]:
    """Parse the trace CSV into tasks sorted by creation time.

    Bad rows are skipped (never abort), matching the Java reader's tolerance.
    """
    tasks: list[Task] = []
    with open(csv_path, newline="") as f:
        reader = csv.reader(f)
        next(reader, None)  # skip header
        for row in reader:
            if len(row) < 11:
                continue
            try:
                tasks.append(
                    Task(
                        name=row[0].strip(),
                        cpu_milli=_parse_int(row[1]),
                        memory_mib=_parse_int(row[2]),
                        num_gpu=_parse_int(row[3]),
                        gpu_milli=_parse_int(row[4]),
                        qos=row[6].strip(),
                        pod_phase=row[7].strip(),
                        creation_time=_parse_float(row[8]),
                        deletion_time=_parse_float(row[9]),
                        scheduled_time=_parse_float(row[10]),
                    )
                )
            except (ValueError, IndexError):
                continue
    tasks.sort(key=lambda t: t.creation_time)
    return tasks


# ── Scenario filter (mirror of Java ScenarioFilter) ─────────────────────────

_LOW_FRACTION = 0.25
_BURST_WINDOW_SEC = 3600.0
_BURST_PERCENTILE = 0.80


def filter_scenario(tasks: list[Task], scenario: str) -> list[Task]:
    """Apply the LOW / HIGH / BURST filter (Pending tasks always excluded)."""
    schedulable = [t for t in tasks if t.pod_phase != "Pending"]
    scenario = scenario.upper()
    if scenario == "HIGH":
        return list(schedulable)
    if scenario == "LOW":
        if not schedulable:
            return []
        count = max(1, int(len(schedulable) * _LOW_FRACTION))
        return list(schedulable[:count])
    if scenario == "BURST":
        return _filter_burst(schedulable)
    raise ValueError(f"unknown scenario: {scenario}")


def _filter_burst(tasks: list[Task]) -> list[Task]:
    if not tasks:
        return []
    min_t = tasks[0].creation_time
    windows: dict[int, list[Task]] = {}
    for t in tasks:
        bucket = int((t.creation_time - min_t) / _BURST_WINDOW_SEC)
        windows.setdefault(bucket, []).append(t)
    sizes = sorted(len(w) for w in windows.values())
    idx = int(len(sizes) * _BURST_PERCENTILE)
    threshold = sizes[min(idx, len(sizes) - 1)]
    result = [t for w in windows.values() if len(w) >= threshold for t in w]
    result.sort(key=lambda t: t.creation_time)
    return result
