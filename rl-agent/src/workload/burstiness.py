"""Burstiness measurement over a real arrival stream (PLAN §3.5, W4.3/W4.4).

The WM-1 ``BURST`` scenario targets IDC(1h) = 40. That number is only defensible if it is
*bracketed by measurement*, and specifically by measurement **at the experiment's own
horizon** — which is the whole subtlety this module exists to handle.

IDC depends on the observation window as well as the counting scale. Philly's IDC(1h)
measured over its full 57 days is 160, but most of that dispersion is week-to-week rate
change: an agent inside a 4-day episode never experiences it as burstiness, it experiences
it as "this episode happens to be busier than the last one". So the anchor that matters is
IDC(1h) measured **inside windows the length of an episode**, which for Philly is 69 —
less than half the whole-trace figure.

Kept separate from :mod:`workload.arrivals` on purpose: that module *generates* streams and
must stay dependency-free and deterministic; this one *measures* streams that already
exist. They share :func:`arrivals.idc` so the generated and the observed numbers are
produced by literally the same estimator — otherwise "we matched the anchor" would be a
statement about two estimators agreeing, not about two streams.
"""

from __future__ import annotations

import statistics
from dataclasses import dataclass

from .arrivals import IDC_WINDOW_SEC, idc, squared_cv

#: Episode-length window used for the within-window anchors. Three days sits just under
#: the homogeneous arm's 4-day horizon; it is short enough that both reference traces
#: yield several disjoint windows and long enough to be the same order as an episode.
ANCHOR_WINDOW_SEC = 3 * 86400.0

#: A window with only a handful of arrivals gives an IDC dominated by the few non-empty
#: bins. Below this the estimate is noise, so such windows are excluded and counted.
MIN_ARRIVALS_PER_WINDOW = 50


@dataclass(frozen=True, slots=True)
class BurstinessProfile:
    """Everything §3.5 reports about one arrival stream."""

    name: str
    n_arrivals: int
    span_days: float
    squared_cv: float           #: C_a^2 over inter-arrival gaps
    idc_whole: float            #: IDC(1h) over the entire trace
    idc_window_median: float | None   #: IDC(1h) within ANCHOR_WINDOW_SEC windows
    idc_window_q1: float | None
    idc_window_q3: float | None
    n_windows: int              #: windows that cleared MIN_ARRIVALS_PER_WINDOW
    idc_window_values: tuple[float, ...] = ()

    @property
    def has_window_estimate(self) -> bool:
        return self.n_windows > 0

    def caveat(self) -> str | None:
        """The honesty line that belongs next to the number in the report.

        With 6-8 windows the interquartile range is estimated from very few points, so it
        describes the spread of *these* windows rather than a population IQR. Stating that
        is the difference between an anchor and a decoration.
        """
        if self.n_windows == 0:
            return f"{self.name}: no window carried >= {MIN_ARRIVALS_PER_WINDOW} arrivals"
        if self.n_windows < 10:
            return (f"{self.name}: IQR estimated from only {self.n_windows} disjoint "
                    f"windows — read it as the spread of those windows, not a population "
                    f"interval")
        return None


def window_idcs(times, *, window_sec: float = ANCHOR_WINDOW_SEC,
                counting_window_sec: float = IDC_WINDOW_SEC,
                min_arrivals: int = MIN_ARRIVALS_PER_WINDOW) -> list[float]:
    """IDC(``counting_window_sec``) inside each disjoint ``window_sec`` slice.

    Disjoint rather than sliding: overlapping windows share arrivals, so their IDCs are
    correlated and a median or IQR over them would understate the true spread while
    looking more precise.
    """
    if not times:
        return []
    t0, t1 = times[0], times[-1]
    out: list[float] = []
    start = t0
    while start + window_sec <= t1:
        stop = start + window_sec
        # times is sorted, but a linear scan is fine at these sizes (15 000 rows) and
        # keeps this readable next to the reference prototype it must agree with.
        n = sum(1 for x in times if start <= x < stop)
        if n >= min_arrivals:
            out.append(idc(times, counting_window_sec, start, stop))
        start = stop
    return out


def profile(name: str, times) -> BurstinessProfile:
    """Measure one arrival stream. ``times`` must be sorted ascending."""
    times = list(times)
    if len(times) < 2:
        raise ValueError(f"{name}: need at least 2 arrivals, got {len(times)}")

    within = window_idcs(times)
    q1 = q3 = median = None
    if within:
        median = statistics.median(within)
        if len(within) > 3:
            qs = statistics.quantiles(within, n=4)
            q1, q3 = qs[0], qs[2]

    t0, t1 = times[0], times[-1]
    return BurstinessProfile(
        name=name,
        n_arrivals=len(times),
        span_days=(t1 - t0) / 86400.0,
        squared_cv=squared_cv(times),
        idc_whole=idc(times, IDC_WINDOW_SEC, t0, t1),
        idc_window_median=median,
        idc_window_q1=q1,
        idc_window_q3=q3,
        n_windows=len(within),
        idc_window_values=tuple(within),
    )


#: Pods that never ran. Excluded when profiling openb, because a ``Pending`` pod is an
#: arrival the simulator never sees — the §3.5 figure of IDC(1h) = 12.0 is measured on the
#: 7 255 schedulable pods, not on all 8 152 rows (which would read 13.5). Philly and Helios
#: have no equivalent: every row there is a real submission.
SCHEDULABLE_ONLY = frozenset({"Pending"})


def profile_adapter(adapter_name: str, path: str, *,
                    exclude_phases: frozenset[str] | None = None) -> BurstinessProfile:
    """Load a trace through its adapter and profile its arrivals.

    Goes through the adapter — and through :func:`adapters.require` with ``CAP_ARRIVALS`` —
    so a trace without a usable arrival stream is refused here rather than silently
    measured. openb would pass that check but is a snapshot, not a submission log
    (``openb.ARRIVAL_CAVEAT``); its number belongs in §3.5 as context, not as an anchor.

    ``exclude_phases`` selects the *population*, which is not a detail: on openb the whole
    file reads IDC(1h) = 13.5, the schedulable subset 12.0, and the WM-1 pool (also
    dropping ``Failed``) 9.5. Comparing one of those against a Philly number computed over
    every row would be comparing different things and calling it a difference in
    burstiness.
    """
    from . import adapters

    adapter = adapters.require(adapter_name, adapters.CAP_ARRIVALS)
    tasks = adapter.load(path)
    if exclude_phases:
        tasks = [t for t in tasks if t.pod_phase not in exclude_phases]
        if len(tasks) < 2:
            raise ValueError(
                f"{adapter_name}: excluding phases {sorted(exclude_phases)} left "
                f"{len(tasks)} tasks")
    times = adapter.native_arrivals(tasks)
    if times is None:
        raise ValueError(f"adapter {adapter_name!r} has no usable arrival stream")
    return profile(adapter_name, sorted(times))
