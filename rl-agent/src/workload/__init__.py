"""WM-1 — parametric workload model for ELDAS (PLAN-Workload-Model.md).

Three independent blocks, each testable on its own:

    schema     canonical Task (11-column openb CSV) + I/O + validation
    adapters   the only modules that know a *specific* source trace (P2 foundation)
    jobsize    empirical bootstrap over whole job rows, truncated at the arm horizon
    arrivals   MMPP-2 ON/OFF modulator with a fitted duty cycle, fixed-N placement

Still to come: ``calibrate`` (bottleneck load -> lambda -> N, W1.4), ``deadline``
(W1.5), ``wm1`` orchestrator and ``cli`` (W1.6).

The expected values every block is checked against are produced independently by
``scripts/wm1-design-reference.py``; if the two disagree, one of them is wrong.
"""

from __future__ import annotations

from . import adapters, arrivals, jobsize, schema
from .jobsize import JobSizePool, PoolMoments
from .schema import SchemaError, Task

__all__ = [
    "adapters", "arrivals", "jobsize", "schema",
    "Task", "SchemaError", "JobSizePool", "PoolMoments",
]
