"""Shared fixtures and locked expected values for the WM-1 tests.

Every constant here is a number produced **independently** of `workload/` — either by
`scripts/characterize-workload.py` (W0.1, itself cross-validated against
`rl-agent/src/eval/trace_loader.py`) or by `scripts/wm1-design-reference.py` (the design
prototype that fixed PLAN-Workload-Model.md §3). If `workload/` disagrees with one of
them, one of the two is wrong and the work stops — that is the whole point of keeping a
second implementation around.
"""

from __future__ import annotations

import os

import pytest

# ── Locating the trace ──────────────────────────────────────────────────────

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
_TRACE_NAME = "openb_pod_list_default.csv"

_CANDIDATES = (
    os.environ.get("ELDAS_TRACE", ""),
    os.path.join("/data", "trace", _TRACE_NAME),                       # in-container
    os.path.join(_REPO_ROOT, "data", "alibaba-trace", _TRACE_NAME),    # on the host
)


def find_trace() -> str | None:
    for p in _CANDIDATES:
        if p and os.path.isfile(p):
            return p
    return None


TRACE_PATH = find_trace()
requires_trace = pytest.mark.skipif(
    TRACE_PATH is None,
    reason=f"{_TRACE_NAME} not found; set ELDAS_TRACE or run scripts/download-trace.sh")


# ── Expected values from scripts/characterize-workload.py ───────────────────

#: sum(pes * duration) over Pending-excluded tasks (= LEGACY_HIGH), PE-seconds.
OPENB_CPU_WORK_SCHEDULABLE = 2502940756.0
#: sum(num_gpu * duration) over the same set, card-seconds.
OPENB_GPU_WORK_SCHEDULABLE = 214603958.0


# ── Expected values from scripts/wm1-design-reference.py (PLAN §3.3, §3.6) ──

DAY = 86400.0

#: arm -> (total PEs, total GPUs, horizon in days)
ARMS = {
    "homo":   (640, 80, 4),
    "hetero": (640, 18, 12),
}

#: arm -> (E[work] PE-s, E[gpu work] card-s, truncated fraction, unplaceable fraction)
#: at that arm's horizon.
#:
#: W3.1 changed these: the pool now also excludes jobs that no *single* host could run
#: (openb pods sized for larger machines — 88 vCPU, 257 GiB). Only 0.15 % of rows, but
#: they carried ~20 % of all CPU-seconds, so E[work] fell 92 572 → 73 627 PE-s and every
#: arrival count rose to compensate. Keeping them would have put a floor under C_SLA that
#: no policy could move, since every scheduler drops them alike (PLAN §3.9).
POOL_MOMENTS = {
    "homo":   (73626.87613911103, 6979.875581179096, 0.0089269109, 0.0014856082),
    "hetero": (128584.79493859323, 11522.47004093785, 0.0059545962, 0.0020427112),
}

#: (arm, scenario) -> (rho, IDC target, fitted p1, N, mean IDC(1h) over seeds 42..46)
SCENARIOS = {
    ("homo", "LOW"):       (0.30, 10.0, 0.393286,  901,  8.5535),
    ("homo", "HIGH"):      (0.85, 10.0, 0.610390, 2554,  8.1938),
    ("homo", "BURST"):     (0.85, 40.0, 0.315607, 2554, 31.8773),
    ("homo", "OVERLOAD"):  (1.25, 10.0, 0.679459, 3755, 10.1965),
    ("hetero", "LOW"):     (0.30, 10.0, 0.117299,  486,  9.5092),
    ("hetero", "HIGH"):    (0.85, 10.0, 0.257778, 1377,  9.9666),
    ("hetero", "BURST"):   (0.85, 40.0, 0.073592, 1377, 43.2671),
    ("hetero", "OVERLOAD"):(1.25, 10.0, 0.336771, 2025,  9.3108),
}

SEEDS = (42, 43, 44, 45, 46)


# ── W4.3 burstiness anchors (scripts/wm1-design-reference.py --anchors) ─────

_REF_TRACE_DIRS = (
    os.environ.get("ELDAS_REFERENCE_TRACES", ""),
    "/data/reference-traces",                                    # in-container
    os.path.join(_REPO_ROOT, "data", "reference-traces"),        # on the host
)


def _find_reference_traces() -> dict[str, str]:
    """Locate the Philly/Helios CSVs, plus openb for the third row of the §3.5 table."""
    found: dict[str, str] = {}
    for d in _REF_TRACE_DIRS:
        if not d or not os.path.isdir(d):
            continue
        for key in ("philly", "helios"):
            p = os.path.join(d, f"{key}_data_training.csv")
            if key not in found and os.path.isfile(p):
                found[key] = p
        if len(found) == 2:
            break
    if TRACE_PATH is not None:
        found["openb"] = TRACE_PATH
    return found


REFERENCE_TRACES = _find_reference_traces()

#: Measured by the design prototype, which parses the CSVs directly and shares no code
#: with ``workload/``. The adapter must land within 5 % of every entry (PLAN W4.3); in
#: practice it reproduces them exactly, because both call the same IDC estimator.
REFERENCE_ANCHORS = {
    "philly": {"ca2": 332.7, "idc_whole": 160.3, "idc_3d_median": 69.0,
               "idc_3d_q1": 7.1, "idc_3d_q3": 99.2, "n_windows": 8},
    "helios": {"ca2": 20.1, "idc_whole": 105.8, "idc_3d_median": 23.1,
               "idc_3d_q1": 16.0, "idc_3d_q3": 141.3, "n_windows": 6},
}


# ── Fixtures ────────────────────────────────────────────────────────────────

@pytest.fixture(scope="session")
def openb_tasks():
    """The real openb trace, parsed once per session (8 152 rows)."""
    if TRACE_PATH is None:
        pytest.skip(f"{_TRACE_NAME} not found; set ELDAS_TRACE")
    from workload import adapters
    return adapters.load("openb", TRACE_PATH)
