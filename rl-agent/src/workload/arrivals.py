"""Arrival model — MMPP-2 ON/OFF with a fitted duty cycle (PLAN §3.1, §3.4, §3.5).

Burstiness is a *parameter* here, not a side effect of how the trace was sliced. The
modulator is a two-state Markov-modulated Poisson process: arrivals happen only while
the chain is ON, mean ON sojourn is pinned at one hour, and the stationary ON
probability ``p1`` is fitted so the measured index of dispersion hits its target.

Two design choices are load-bearing and both were forced by measurement, not taste:

**Fit numerically, not from the asymptotic formula.** ``IDC(w)`` only approaches its
asymptote for windows far above the burst timescale. At w = 1 h on a multi-day horizon
we sit squarely in the transition region, so the closed-form MMPP expression is not a
usable predictor of what the generated stream will actually measure. Bisecting on the
measured value sidesteps that entirely.

**Place exactly N arrivals (fixed-N).** Conditioning on the count makes the realised
offered load exact by construction and removes a large source of cross-seed variance,
which matters when the whole campaign has 5 seeds. Without it the prototype produced
387 arrivals for hetero/BURST against 497 for hetero/HIGH at the *same* mean rate — a
23 % gap that would have been read as a load difference.

Targets in use (PLAN §3.5):

    LOW / HIGH / OVERLOAD   IDC(1h) = 10    (openb itself measures 12.0)
    BURST                   IDC(1h) = 40    (Philly 69.0, Helios 23.1 within a 3-day window)

The BURST anchor is measured **inside a window the length of the experiment horizon**,
not over the whole source trace. This is a validity argument, not a feasibility one:
this parameterisation can reach far higher values (IDC ~ 900 at n = 2031 over 4 days),
but Philly's whole-trace IDC(1h) of 160 largely reflects week-to-week rate changes over
57 days. An agent inside a 4-day episode never experiences that as burstiness — it would
see it as a different mean rate per episode. Generating at 160 would make *every* episode
burstier than Philly's own upper-quartile 4-day slice (99.2).
"""

from __future__ import annotations

import functools
import math
import random

# ── Locked design constants (PLAN §3.5/§3.6) ────────────────────────────────

E_ON_SEC = 3600.0        #: mean ON sojourn of the modulator
IDC_WINDOW_SEC = 3600.0  #: scale at which IDC targets are defined

FIT_REPS = 12            #: replicate streams averaged inside each bisection step
FIT_ITERS = 32           #: bisection iterations
FIT_SEED_BASE = 9000     #: fixed seeds -> the fit is a deterministic function of p1
FIT_P1_LO = 0.004        #: burstiest duty cycle the bracket considers
FIT_P1_HI = 0.98         #: smoothest duty cycle the bracket considers
FIT_REL_TOL = 0.10       #: max |measured - target| / target accepted at convergence


class ArrivalError(ValueError):
    """Raised for invalid arrival-model parameters."""


class BurstinessNotAchievable(ArrivalError):
    """The requested IDC cannot be produced on this horizon with this arrival count.

    Almost always means the target was measured over a much longer observation window
    than the horizon being generated — see the module docstring and PLAN §3.5.
    """


# ── Measurement ─────────────────────────────────────────────────────────────

def idc(times, window: float, t0: float, t1: float) -> float:
    """Index of dispersion for counts, ``Var(N_w) / E(N_w)``, over ``[t0, t1)``.

    Empty windows are counted. Dropping them is the classic mistake: the openb trace
    leaves 77 % of its one-hour windows empty, and excluding those collapses the
    variance and understates burstiness by an order of magnitude.

    Poisson gives 1.0; bursty gives >> 1.
    """
    if window <= 0:
        raise ArrivalError(f"window must be positive, got {window}")
    span = max(t1 - t0, 0.0)
    n_win = max(1, int(math.ceil(span / window)) if span > 0 else 1)
    counts = [0] * n_win
    for t in times:
        if t0 <= t < t1:
            counts[min(n_win - 1, int((t - t0) / window))] += 1
    mean = sum(counts) / n_win
    if mean <= 0:
        return float("nan")
    var = sum((c - mean) ** 2 for c in counts) / n_win
    return var / mean


def squared_cv(times) -> float:
    """``C_a^2 = Var(inter-arrival) / mean(inter-arrival)^2``. Poisson gives 1.0."""
    if len(times) < 3:
        return float("nan")
    gaps = [times[i + 1] - times[i] for i in range(len(times) - 1)]
    m = sum(gaps) / len(gaps)
    if m <= 0:
        return float("nan")
    return (sum((g - m) ** 2 for g in gaps) / len(gaps)) / (m * m)


# ── Modulator ───────────────────────────────────────────────────────────────

def on_intervals(p1: float, horizon: float, rng: random.Random) -> list[tuple[float, float]]:
    """Sample the ON intervals of the two-state modulator over ``[0, horizon)``.

    ``p1`` is the stationary ON probability. With the mean ON sojourn pinned at
    :data:`E_ON_SEC`, the OFF rate follows as ``r2 = r1 * p1 / (1 - p1)``, so ``p1``
    alone controls burstiness: smaller ``p1`` means the same arrivals are squeezed into
    rarer, more intense bursts.
    """
    if horizon <= 0:
        raise ArrivalError(f"horizon must be positive, got {horizon}")
    if not 0 < p1 <= 1:
        raise ArrivalError(f"duty cycle must be in (0, 1], got {p1}")
    if p1 >= 1.0:
        # Degenerate always-ON case. Combined with fixed-N placement this is exactly a
        # Poisson process conditioned on its count, which is the IDC ~ 1 reference.
        return [(0.0, horizon)]

    r1 = 1.0 / E_ON_SEC                 # ON -> OFF
    r2 = r1 * p1 / (1.0 - p1)           # OFF -> ON, chosen so P(ON) = p1
    t = 0.0
    on = rng.random() < p1              # start in the stationary state
    out: list[tuple[float, float]] = []
    while t < horizon:
        end = min(t + rng.expovariate(r1 if on else r2), horizon)
        if on and end > t:
            out.append((t, end))
        t, on = end, not on
    return out


def place_arrivals(n: int, intervals: list[tuple[float, float]],
                   rng: random.Random) -> list[float]:
    """Place exactly ``n`` arrivals uniformly over the union of ``intervals``.

    Uniform placement on a fixed set is the order-statistics construction of a Poisson
    process conditioned on its count, so the result is a genuine MMPP realisation with
    the count pinned — the burstiness still comes entirely from the ON/OFF structure.

    The interval scan is linear rather than a bisection over cumulative offsets: the
    modulator produces only tens of intervals per horizon, and the running subtraction
    keeps this bit-identical to ``scripts/wm1-design-reference.py``, which is the
    independent source of the expected values these numbers are checked against.
    """
    if n < 0:
        raise ArrivalError(f"arrival count must be non-negative, got {n}")
    total = sum(b - a for a, b in intervals)
    if n == 0:
        return []
    if total <= 0:
        raise ArrivalError(
            "modulator produced no ON time; the duty cycle is too small for this horizon")

    out: list[float] = []
    for _ in range(n):
        u = rng.random() * total
        for a, b in intervals:
            width = b - a
            if u <= width:
                out.append(a + u)
                break
            u -= width
        else:  # pragma: no cover - only reachable through float drift at the top end
            # Must stay strictly below the horizon: the half-open IDC window and the
            # simulator's arrival loop both assume no task lands exactly at T.
            out.append(math.nextafter(intervals[-1][1], -math.inf))
    out.sort()
    return out


def generate(n: int, p1: float, horizon: float, rng: random.Random) -> list[float]:
    """Sample the modulator, then place exactly ``n`` arrivals on its ON set."""
    return place_arrivals(n, on_intervals(p1, horizon, rng), rng)


# ── Fitting ─────────────────────────────────────────────────────────────────

def _mean_idc(n: int, p1: float, horizon: float, window: float, reps: int) -> float:
    """Deterministic estimator of IDC at ``p1`` — fixed seeds make bisection stable.

    A duty cycle small enough that the modulator never switches ON on some horizon is
    reported as ``inf`` rather than raising: within the bisection that is simply
    "burstier than anything achievable", and the search recovers by widening the duty
    cycle. Only if the *converged* value is still non-finite does the caller give up.
    """
    total = 0.0
    for k in range(reps):
        try:
            value = idc(generate(n, p1, horizon, random.Random(FIT_SEED_BASE + k)),
                        window, 0.0, horizon)
        except ArrivalError:
            return math.inf
        if not math.isfinite(value):
            return math.inf
        total += value
    return total / reps


@functools.lru_cache(maxsize=64)
def fit_duty_cycle(n: int, target_idc: float, horizon: float, *,
                   window: float = IDC_WINDOW_SEC,
                   reps: int = FIT_REPS,
                   iters: int = FIT_ITERS,
                   rel_tol: float = FIT_REL_TOL) -> float:
    """Find the duty cycle whose generated stream measures ``target_idc`` at ``window``.

    Bisects on ``p1`` in log space over ``[FIT_P1_LO, FIT_P1_HI]``. Because the estimator
    uses fixed seeds it is a deterministic function of ``p1``, so the search converges
    tightly and any residual gap means the target lies outside the achievable range.

    Memoised: the fit depends on ``(n, target, horizon)`` but not on the trace seed, so
    generating 5 seeds of one scenario must not pay for 5 identical bisections. Safe
    because the function is pure — the estimator's seeds are fixed constants.

    Raises:
        BurstinessNotAchievable: if the converged stream misses the target by more than
            ``rel_tol``. The ceiling is set by the horizon and the arrival count — there
            are only ``horizon / window`` windows to concentrate arrivals into, so a
            target far above what that allows cannot be met no matter how rare the
            bursts (PLAN §3.5, risk R3).
    """
    if n <= 0:
        raise ArrivalError(f"arrival count must be positive to fit, got {n}")
    if target_idc < 1.0:
        raise ArrivalError(
            f"IDC target must be >= 1 (Poisson); got {target_idc}")

    lo, hi = FIT_P1_LO, FIT_P1_HI
    p1 = math.sqrt(lo * hi)
    for _ in range(iters):
        p1 = math.sqrt(lo * hi)
        if _mean_idc(n, p1, horizon, window, reps) > target_idc:
            lo = p1        # burstier than asked -> smooth it by raising the duty cycle
        else:
            hi = p1
    p1 = math.sqrt(lo * hi)

    got = _mean_idc(n, p1, horizon, window, reps)
    if not math.isfinite(got) or abs(got - target_idc) / target_idc > rel_tol:
        best = ("the modulator runs out of ON time before getting there"
                if not math.isfinite(got) else f"the closest reachable value is {got:.1f}")
        raise BurstinessNotAchievable(
            f"cannot reach IDC({window / 3600:g}h) = {target_idc:g} with n={n} over "
            f"{horizon / 86400:g} days: {best} (duty cycle {p1:.5f}, bracket "
            f"[{FIT_P1_LO}, {FIT_P1_HI}]). There are only {int(horizon / window)} windows "
            f"to concentrate {n} arrivals into, which caps the achievable dispersion. If "
            f"the target came from a trace observed over a much longer period than this "
            f"horizon, re-measure it inside a window of the same length "
            f"(PLAN-Workload-Model.md §3.5).")
    return p1
