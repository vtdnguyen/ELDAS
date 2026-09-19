"""Alibaba ``cluster-trace-gpu-v2023`` (openb pod list) — the current source trace.

This is the trace ELDAS has used since Phase 1: 8 152 pods with CPU, memory, GPU (whole
cards and fractional shares), QoS class and creation/deletion/scheduled timestamps.

**What it is good for:** the job-size distribution. It is the only trace in reach that
carries CPU *and* memory *and* GPU *and* QoS together, which is exactly what WM-1's
bootstrap needs (PLAN §3.1).

**What to be careful about:** the arrival stream. The file is a *snapshot plus an
observation window*, not a submission log — the earliest ``deletion_time`` is 9 964 972 s
while ``creation_time`` spans 0…12 901 761 s, so the oldest pods are the long-lived
service population that was already running when collection began (PLAN §1.3-C). The
adapter still exposes :meth:`native_arrivals` because the timestamps are real and the
REPLAY scenario legitimately uses them, but :data:`ARRIVAL_CAVEAT` records why they must
not be treated as a stationary arrival process to fit a model to.

Source: https://github.com/alibaba/clusterdata/tree/master/cluster-trace-gpu-v2023
"""

from __future__ import annotations

from .. import schema
from ..schema import Task
from . import (CAP_ARRIVALS, CAP_GPU, CAP_JOBSIZE, CAP_MEMORY, CAP_QOS, register)

#: Recorded on the adapter (and surfaced by tooling) rather than buried in a docstring,
#: because the whole point of GĐ 2.4 is that this caveat was previously invisible.
ARRIVAL_CAVEAT = (
    "openb is a snapshot + observation window, not a submission log: the earliest "
    "deletion_time is 9.96e6 s while creation_time spans 0..1.29e7 s, so ordering by "
    "creation_time selects the long-lived service population first. Usable for REPLAY "
    "over a chosen window; NOT usable as a stationary arrival process to fit."
)


class OpenbAdapter:
    """Adapter for the openb pod-list CSV."""

    name = "openb"

    def capabilities(self) -> frozenset[str]:
        return frozenset({CAP_JOBSIZE, CAP_ARRIVALS, CAP_QOS, CAP_MEMORY, CAP_GPU})

    def load(self, path: str) -> list[Task]:
        """The canonical schema *is* the openb schema, so this is a straight parse."""
        return schema.read_csv(path)

    def native_arrivals(self, tasks: list[Task]) -> list[float]:
        return [t.creation_time for t in tasks]

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"<OpenbAdapter caps={sorted(self.capabilities())}>"


ADAPTER = register(OpenbAdapter())
