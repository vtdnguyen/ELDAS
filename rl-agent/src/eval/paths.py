"""Where results are allowed to be written (PLAN-Workload-Model.md §4.1, risks R5/R6).

Two rules, one function each:

**R5 — WM-1 results must never land on LEGACY results.** The Phase-1 numbers in
``/data/results`` are the comparison baseline for the whole thesis; a WM-1 run that writes
``campaign-HIGH/`` over them destroys the thing the new work is measured against, and does
it silently — the files look normal afterwards.

**R6 — the two arms must never overwrite each other.** ``homo`` and ``hetero`` produce a
``campaign-HIGH/`` each. Without the arm somewhere in the path the second run replaces the
first, again silently. This already happened once in this project (C16, "LOW ppo-min
stale"), which is why the plan calls it out by name.

§4.1 writes the layout as ``campaign-{arm}-{SCENARIO}/``. This module puts the arm one
level up instead — ``/data/results/wm1/{arm}/campaign-{SCENARIO}/`` — for a reason worth
stating, because it is a deliberate deviation:

    Every reader and writer already shares one ``--results`` / ``--output`` argument.
    Moving the arm into the ROOT means they stay consistent by construction. Renaming the
    twelve ``f"campaign-{scenario}"`` constructions across seven modules would require
    every reader to be changed in lockstep with every writer, and a single missed reader
    does not crash — it reads last week's file and reports it as this week's result.

The anti-collision guarantee §4.1 asks for is met either way; this spelling cannot drift.
"""

from __future__ import annotations

import os
from pathlib import Path

#: Where Phase-1 / LEGACY results live. Writing WM-1 output here is refused.
LEGACY_ROOT = "/data/results"

#: Root for everything produced from generated WM-1 workloads.
WM1_ROOT = "/data/results/wm1"

KNOWN_ARMS = ("homo", "hetero")


class ResultsPathError(RuntimeError):
    """The requested output location would overwrite results it must not touch."""


def wm1_mode(pattern: str | None = None) -> bool:
    """True when this process is running on a generated WM-1 workload."""
    p = pattern if pattern is not None else os.environ.get("TRACE_PATTERN", "")
    return bool(p and p.strip())


def arm_from_pattern(pattern: str | None = None) -> str | None:
    """Infer ``homo`` / ``hetero`` from the trace pattern, if it names one.

    Used only to *suggest* the right root in an error message. It is never used to pick a
    path silently: guessing the arm from a string and then writing results under it is how
    a mislabelled campaign happens.
    """
    p = pattern if pattern is not None else os.environ.get("TRACE_PATTERN", "")
    parts = Path(p.replace("\\", "/")).parts if p else ()
    for arm in KNOWN_ARMS:
        if arm in parts:
            return arm
    return None


def wm1_results_root(arm: str) -> str:
    """``/data/results/wm1/<arm>`` — the root a WM-1 run for ``arm`` should write to."""
    if arm not in KNOWN_ARMS:
        raise ResultsPathError(f"unknown arm {arm!r}; expected one of {KNOWN_ARMS}")
    return f"{WM1_ROOT}/{arm}"


def guard_results_root(root: str | os.PathLike, *, pattern: str | None = None,
                       what: str = "this run") -> str:
    """Refuse a WM-1 run that is pointed at the LEGACY results root.

    Call this in every CLI that writes results, right after parsing arguments. It is a
    validation step only — it never rewrites the path, because a helpful auto-correct here
    would put results somewhere the operator did not ask for and did not check.
    """
    root_s = str(root)
    if not wm1_mode(pattern):
        return root_s

    normalised = str(Path(root_s).as_posix()).rstrip("/")
    if normalised != LEGACY_ROOT.rstrip("/"):
        return root_s

    arm = arm_from_pattern(pattern)
    suggestion = wm1_results_root(arm) if arm else f"{WM1_ROOT}/<arm>"
    raise ResultsPathError(
        f"TRACE_PATTERN is set, so {what} is running a generated WM-1 workload, but the "
        f"output root is the LEGACY results directory {LEGACY_ROOT!r}. Writing there "
        f"would overwrite the Phase-1 results this work is compared against, and it would "
        f"look like nothing happened (PLAN §4.1 risk R5).\n"
        f"  Use: --results {suggestion}   (or --output, depending on the tool)\n"
        f"  The arm belongs in the path so the two arms cannot overwrite each other "
        f"either (risk R6)."
    )
