"""Philly (Microsoft) and Helios (SenseTime) GPU-cluster traces — **arrivals only**.

PLAN-Workload-Model.md §8 / W4.2. These two traces exist in this project for exactly one
job: to be an *independent* anchor for the burstiness parameter of the WM-1 ``BURST``
scenario. The thesis claims IDC(1h) = 40 is not an invented number but a value bracketed
by two production GPU clusters measured at the experiment's own horizon; that claim needs
the traces to be readable through the same code path as everything else, which is what
this adapter provides.

**What they can supply**

    interval   inter-arrival seconds -> cumulative sum -> creation_time  (a real
               submission log, unlike openb's snapshot — see openb.ARRIVAL_CAVEAT)
    run_time   seconds -> duration
    gpu_num    whole GPU cards requested
    cpu_num    CPU cores (Helios only — see below)

**What they cannot supply — verified against the actual files, not assumed**

    wall_time  0 in EVERY row of BOTH traces. An earlier reading of the plan assumed this
               was a requested time limit that could serve as a deadline. It is not
               present, so nothing here can produce an SLA target.
    cpu_num    0 in EVERY row of Philly. Helios has real values.
    memory     no column at all.
    QoS        no column at all.

That is why :meth:`PhillyFamilyAdapter.capabilities` withholds :data:`CAP_JOBSIZE`. Feeding
WM-1 from one of these traces would emit a workload with ``memory_mib = 0`` and no QoS
class everywhere; the simulator would schedule it without a word of complaint, every task
would fit every host, and the resulting "results" would look entirely normal. The
capability interlock (:func:`adapters.require`) turns that into an exception at the call
site instead — see :data:`JOBSIZE_REFUSAL` for the message.

Source: DIR-LAB/Gen-Parallel-Workloads (JSSPP 2024). Files, checksums and the verified
limitations above are recorded in ``data/reference-traces/MANIFEST.json``.
"""

from __future__ import annotations

import csv
import os

from ..schema import PHASE_FAILED, SchemaError, Task
from . import CAP_ARRIVALS, CAP_GPU, AdapterError, register

#: Columns the Gen-Parallel-Workloads CSVs carry. Checked on load: a header mismatch means
#: upstream changed the format, and silently mapping the wrong column into ``gpu_num``
#: would corrupt the one measurement these traces exist to provide.
EXPECTED_HEADER = (
    "u_id", "user", "gpu_num", "cpu_num", "node_num",
    "interval", "run_time", "wall_time", "new_status",
)

#: Surfaced on the adapter rather than buried in a docstring, for the same reason
#: ``openb.ARRIVAL_CAVEAT`` is: the limitation was previously invisible to callers.
JOBSIZE_REFUSAL = (
    "Philly/Helios have no memory column, no QoS column, and wall_time = 0 in every row; "
    "Philly additionally has cpu_num = 0 in every row. Using them as a WM-1 job-size "
    "source would generate tasks with zero memory and no QoS class — which the simulator "
    "accepts silently. They are arrival-process anchors only (PLAN §8, W4.2)."
)

#: The traces are anonymised submission logs with no absolute epoch: ``interval`` is the
#: gap from the previous job. Arrival times are therefore relative to the first job, which
#: is all the IDC/C_a² measurements need.
_T0 = 0.0


class PhillyFamilyAdapter:
    """Adapter for one Gen-Parallel-Workloads CSV (Philly or Helios — same schema)."""

    def __init__(self, name: str, cluster: str, *, has_cpu: bool):
        self.name = name
        self.cluster = cluster
        self.has_cpu = has_cpu

    def capabilities(self) -> frozenset[str]:
        # Deliberately NOT CAP_JOBSIZE / CAP_QOS / CAP_MEMORY. CAP_GPU is honest: gpu_num
        # is real and populated, and it is what makes these traces GPU-cluster anchors
        # rather than generic HPC ones.
        return frozenset({CAP_ARRIVALS, CAP_GPU})

    def load(self, path: str) -> list[Task]:
        """Parse into canonical tasks, arrival times reconstructed by cumulative sum.

        Rows are kept in file order: ``interval`` is defined relative to the previous row,
        so the file order *is* the arrival order and re-sorting would be meaningless.
        """
        if not os.path.isfile(path):
            raise SchemaError(f"reference trace not found: {path} "
                              f"— run scripts/fetch-reference-traces.sh")

        tasks: list[Task] = []
        with open(path, newline="", encoding="utf-8") as f:
            reader = csv.reader(f)
            header = next(reader, None)
            if header is None or tuple(h.strip() for h in header) != EXPECTED_HEADER:
                raise SchemaError(
                    f"{path}: unexpected header {header}; expected {list(EXPECTED_HEADER)}. "
                    f"Upstream changed the format — re-run scripts/fetch-reference-traces.sh "
                    f"(it verifies the sha256) before trusting any measurement from it.")

            t = _T0
            for i, row in enumerate(reader):
                if len(row) < len(EXPECTED_HEADER):
                    continue
                try:
                    interval = float(row[5])
                    run_time = float(row[6])
                    gpu_num = int(float(row[2]))
                    cpu_num = int(float(row[3]))
                except ValueError:
                    continue

                # Guard the one thing that would silently ruin an arrival measurement.
                # A negative gap cannot be recovered from, and clamping it would fabricate
                # a burst; refusing is the only honest option.
                if interval < 0:
                    raise SchemaError(
                        f"{path} row {i + 2}: negative interval {interval}; the arrival "
                        f"stream cannot be reconstructed from this file")
                t += interval

                tasks.append(Task(
                    name=f"{self.name}-{row[0].strip()}",
                    # 0 for Philly by construction; `pes` floors at 1, so never treat this
                    # as a CPU demand without checking `has_cpu` first.
                    cpu_milli=max(0, cpu_num) * 1000,
                    memory_mib=0,          # absent from the source
                    num_gpu=max(0, gpu_num),
                    gpu_milli=0,           # no fractional-share concept in these traces
                    gpu_spec="",
                    qos="",                # absent — CAP_QOS is withheld for this reason
                    pod_phase=_phase(row[8].strip()),
                    creation_time=t,
                    deletion_time=t + max(0.0, run_time),
                    scheduled_time=t,      # no queue-wait column; assume instant dispatch
                ))
        if not tasks:
            raise SchemaError(f"{path}: no usable rows")
        return tasks

    def native_arrivals(self, tasks: list[Task]) -> list[float]:
        return [t.creation_time for t in tasks]

    def refuse_jobsize(self) -> None:
        """Raise with the reason. Called by tooling that wants the message, not the check."""
        raise AdapterError(f"adapter {self.name!r} cannot supply job sizes. {JOBSIZE_REFUSAL}")

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"<PhillyFamilyAdapter {self.name} caps={sorted(self.capabilities())}>"


def _phase(status: str) -> str:
    """Map ``new_status`` onto the openb phase vocabulary.

    ``Failed`` carries over directly (WM-1 excludes failed jobs from the pool because their
    lifetime is not a service demand). ``Pass`` and ``Killed`` both ran to some length, so
    both map to ``Running`` — a killed job still occupied its GPUs for ``run_time``, which
    is exactly what an arrival/duration measurement should count.
    """
    return PHASE_FAILED if status == "Failed" else "Running"


PHILLY = register(PhillyFamilyAdapter("philly", "Philly (Microsoft, 2 490 GPU)",
                                      has_cpu=False))
HELIOS = register(PhillyFamilyAdapter("helios", "Helios (SenseTime, 6 416 GPU)",
                                      has_cpu=True))
