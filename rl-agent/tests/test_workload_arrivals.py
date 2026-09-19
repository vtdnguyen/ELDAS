"""W1.3 — arrival model: IDC measurement, MMPP-2 modulator, fixed-N, duty-cycle fit.

Acceptance targets from PLAN-Workload-Model.md W1.3:
    (a) a pure Poisson stream measures IDC = 1.0 +/- 0.06
    (b) the fitted duty cycle matches PLAN §3.6 within +/- 0.03
    (c) mean IDC(1h) over seeds 42..46 lands in [0.75x, 1.30x] of target
    (d) len(arrivals) == N exactly
"""

from __future__ import annotations

import importlib.util
import math
import os
import random

import pytest

from workload import arrivals
from workload.arrivals import (ArrivalError, BurstinessNotAchievable, E_ON_SEC,
                               IDC_WINDOW_SEC)

from _workload_fixtures import ARMS, DAY, SCENARIOS, SEEDS  # noqa: F401

HOUR = 3600.0


# ── IDC: definition and the empty-window trap ──────────────────────────────

def test_idc_hand_computed_case_counts_empty_windows():
    """Three arrivals in window 0, one in window 9, eight empty windows between.

    counts = [3,0,0,0,0,0,0,0,0,1] -> mean 0.4, var 0.84 -> IDC 2.1.
    Dropping the empty windows would give a completely different answer, which is the
    mistake this function exists to avoid (77 % of openb's 1 h windows are empty).
    """
    times = [0.0, 1.0, 2.0, 9 * HOUR + 1]
    assert arrivals.idc(times, HOUR, 0.0, 10 * HOUR) == pytest.approx(2.1)


def test_idc_of_a_perfectly_regular_stream_is_zero():
    times = [i * HOUR + 100.0 for i in range(24)]     # exactly one per window
    assert arrivals.idc(times, HOUR, 0.0, 24 * HOUR) == pytest.approx(0.0)


def test_idc_is_nan_when_there_are_no_arrivals():
    assert math.isnan(arrivals.idc([], HOUR, 0.0, 10 * HOUR))


def test_idc_window_range_is_half_open():
    """Arrivals at exactly t1 are outside the measured window."""
    assert math.isnan(arrivals.idc([10 * HOUR], HOUR, 0.0, 10 * HOUR))
    assert not math.isnan(arrivals.idc([10 * HOUR - 1], HOUR, 0.0, 10 * HOUR))


def test_idc_rejects_a_non_positive_window():
    with pytest.raises(ArrivalError, match="window must be positive"):
        arrivals.idc([1.0], 0.0, 0.0, 10.0)


def test_squared_cv_of_a_regular_stream_is_zero_and_of_poisson_is_one():
    assert arrivals.squared_cv([i * 10.0 for i in range(50)]) == pytest.approx(0.0)
    rng = random.Random(11)
    t, times = 0.0, []
    for _ in range(60_000):
        t += rng.expovariate(1 / 50.0)
        times.append(t)
    assert arrivals.squared_cv(times) == pytest.approx(1.0, abs=0.05)


def test_squared_cv_needs_enough_samples():
    assert math.isnan(arrivals.squared_cv([0.0, 1.0]))


def test_idc_agrees_with_characterize_workload():
    """Cross-module agreement with the W0.1 tool that will validate generated traces.

    Both implementations must measure the same burstiness, otherwise a trace could pass
    generation and fail characterisation (or worse, the reverse).
    """
    root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    path = os.path.join(root, "scripts", "characterize-workload.py")
    if not os.path.isfile(path):
        pytest.skip("scripts/characterize-workload.py not reachable from here")
    spec = importlib.util.spec_from_file_location("cw_for_test", path)
    cw = importlib.util.module_from_spec(spec)
    import sys
    sys.modules["cw_for_test"] = cw
    spec.loader.exec_module(cw)

    T = 4 * DAY
    stream = arrivals.generate(500, 0.3, T, random.Random(5))
    mine = arrivals.idc(stream, HOUR, 0.0, T)
    theirs = cw.index_of_dispersion(stream, HOUR, 0.0, T)
    assert mine == pytest.approx(theirs, rel=1e-12)


# ── Modulator ───────────────────────────────────────────────────────────────

def test_on_intervals_are_ordered_disjoint_and_inside_the_horizon():
    T = 4 * DAY
    iv = arrivals.on_intervals(0.3, T, random.Random(1))
    assert iv, "a 4-day horizon at p1=0.3 must contain some ON time"
    prev_end = 0.0
    for a, b in iv:
        assert 0.0 <= a < b <= T
        assert a >= prev_end
        prev_end = b


def test_on_fraction_tracks_the_duty_cycle():
    """P(ON) = p1 is the point of the r2 = r1*p1/(1-p1) construction."""
    T = 400 * DAY                       # long horizon so the average is tight
    for p1 in (0.1, 0.3, 0.7):
        iv = arrivals.on_intervals(p1, T, random.Random(int(p1 * 100)))
        assert sum(b - a for a, b in iv) / T == pytest.approx(p1, abs=0.03)


def test_mean_on_sojourn_is_pinned_to_one_hour():
    T = 400 * DAY
    iv = arrivals.on_intervals(0.3, T, random.Random(2))
    interior = [b - a for a, b in iv][1:-1]      # drop horizon-clipped ends
    assert sum(interior) / len(interior) == pytest.approx(E_ON_SEC, rel=0.10)


def test_duty_cycle_of_one_is_the_degenerate_always_on_case():
    T = 3 * DAY
    assert arrivals.on_intervals(1.0, T, random.Random(0)) == [(0.0, T)]


@pytest.mark.parametrize("p1", [0.0, -0.1, 1.5])
def test_invalid_duty_cycles_are_rejected(p1):
    with pytest.raises(ArrivalError, match="duty cycle"):
        arrivals.on_intervals(p1, DAY, random.Random(0))


def test_non_positive_horizon_is_rejected():
    with pytest.raises(ArrivalError, match="horizon must be positive"):
        arrivals.on_intervals(0.3, 0.0, random.Random(0))


# ── Fixed-N placement (acceptance d) ───────────────────────────────────────

@pytest.mark.parametrize("n", [0, 1, 17, 2031])
def test_exactly_n_arrivals_are_produced(n):
    out = arrivals.generate(n, 0.3, 4 * DAY, random.Random(42))
    assert len(out) == n


def test_arrivals_are_sorted_and_inside_the_on_set():
    T = 4 * DAY
    rng = random.Random(9)
    iv = arrivals.on_intervals(0.25, T, rng)
    placed = arrivals.place_arrivals(300, iv, rng)
    assert placed == sorted(placed)
    for t in placed:
        assert any(a <= t <= b for a, b in iv), f"{t} fell outside every ON interval"


def test_arrivals_stay_strictly_inside_the_horizon():
    """WM-1 relies on this: the half-open IDC window would otherwise drop the last job."""
    T = 4 * DAY
    for seed in SEEDS:
        out = arrivals.generate(2031, 0.27, T, random.Random(seed))
        assert 0.0 <= out[0] and out[-1] < T


def test_placement_is_deterministic_for_a_given_seed():
    a = arrivals.generate(500, 0.3, 4 * DAY, random.Random(42))
    b = arrivals.generate(500, 0.3, 4 * DAY, random.Random(42))
    assert a == b


def test_negative_count_is_rejected():
    with pytest.raises(ArrivalError, match="non-negative"):
        arrivals.place_arrivals(-1, [(0.0, 1.0)], random.Random(0))


def test_placement_without_any_on_time_raises():
    with pytest.raises(ArrivalError, match="no ON time"):
        arrivals.place_arrivals(5, [], random.Random(0))


# ── Poisson reference (acceptance a) ───────────────────────────────────────

def test_always_on_fixed_n_is_a_poisson_process_with_idc_one():
    """p1 = 1 makes placement uniform on [0, T), i.e. Poisson conditioned on its count."""
    T = 40 * DAY
    vals = [arrivals.idc(arrivals.generate(20_000, 1.0, T, random.Random(s)),
                         HOUR, 0.0, T)
            for s in SEEDS]
    assert sum(vals) / len(vals) == pytest.approx(1.0, abs=0.06)


def test_burstier_duty_cycles_raise_idc_monotonically():
    T = 4 * DAY
    got = [arrivals._mean_idc(2031, p1, T, IDC_WINDOW_SEC, reps=6)
           for p1 in (0.9, 0.6, 0.4, 0.25, 0.12)]
    assert got == sorted(got), f"IDC must increase as the duty cycle falls, got {got}"


# ── Duty-cycle fit (acceptance b and c) ────────────────────────────────────

@pytest.mark.parametrize("arm,scenario", sorted(SCENARIOS))
def test_fitted_duty_cycle_matches_the_design_reference(arm, scenario):
    _, idc_target, want_p1, n, _ = SCENARIOS[(arm, scenario)]
    _, _, days = ARMS[arm]
    got = arrivals.fit_duty_cycle(n, idc_target, days * DAY)
    assert got == pytest.approx(want_p1, abs=0.03)


@pytest.mark.parametrize("arm,scenario", sorted(SCENARIOS))
def test_realised_burstiness_lands_on_target_across_seeds(arm, scenario):
    """Acceptance (c). The band is wide on purpose: BURST's cross-seed sd is 6.5-11.8
    on a mean of 36-44, so a tight per-seed assertion would fail at random (risk R3).
    """
    _, idc_target, p1, n, want_mean = SCENARIOS[(arm, scenario)]
    _, _, days = ARMS[arm]
    T = days * DAY
    vals = [arrivals.idc(arrivals.generate(n, p1, T, random.Random(s)), HOUR, 0.0, T)
            for s in SEEDS]
    mean = sum(vals) / len(vals)
    assert mean == pytest.approx(want_mean, rel=0.02), "drifted from the design reference"
    assert 0.75 * idc_target <= mean <= 1.30 * idc_target


def test_fit_is_deterministic():
    T = 4 * DAY
    assert arrivals.fit_duty_cycle(717, 10.0, T) == arrivals.fit_duty_cycle(717, 10.0, T)


def test_burst_is_measurably_burstier_than_high_at_the_same_load():
    """The property that makes BURST vs HIGH a controlled comparison (PLAN §3.6)."""
    T = 4 * DAY
    _, _, p1_high, n, _ = SCENARIOS[("homo", "HIGH")]
    _, _, p1_burst, n_burst, _ = SCENARIOS[("homo", "BURST")]
    assert n == n_burst, "HIGH and BURST must carry the same arrival count"
    high = arrivals.idc(arrivals.generate(n, p1_high, T, random.Random(42)), HOUR, 0.0, T)
    burst = arrivals.idc(arrivals.generate(n, p1_burst, T, random.Random(42)), HOUR, 0.0, T)
    assert burst > 3 * high


# ── The guard that caught the original design error ────────────────────────

def test_unreachable_burstiness_raises_instead_of_returning_junk():
    """A horizon offers only `horizon / window` windows to concentrate arrivals into,
    which caps the achievable dispersion. Past that ceiling the modulator runs out of ON
    time and the fit must give up loudly rather than return a stream with no arrivals.
    """
    with pytest.raises(BurstinessNotAchievable, match="cannot reach"):
        arrivals.fit_duty_cycle(2031, 1100.0, 4 * DAY)


def test_the_unreachable_error_points_at_the_real_cause():
    try:
        arrivals.fit_duty_cycle(2031, 1100.0, 4 * DAY)
    except BurstinessNotAchievable as e:
        msg = str(e)
        assert "96 windows" in msg, "the message must name the actual limiting quantity"
        assert "3.5" in msg
    else:  # pragma: no cover
        pytest.fail("expected BurstinessNotAchievable")


def test_a_degenerate_duty_cycle_recovers_during_the_search():
    """Mid-search a candidate duty cycle can be so small the modulator never switches ON.

    That must be treated as "burstier than anything achievable" so the bisection widens
    and recovers - not propagate as a low-level error. Philly's whole-trace IDC of 160 is
    reachable on this horizon and only passes through such candidates on the way.
    """
    T = 4 * DAY
    p1 = arrivals.fit_duty_cycle(2031, 160.0, T)
    vals = [arrivals.idc(arrivals.generate(2031, p1, T, random.Random(s)), HOUR, 0.0, T)
            for s in SEEDS]
    assert sum(vals) / len(vals) > 80.0, f"stream is not actually bursty: {vals}"
    assert all(len(arrivals.generate(2031, p1, T, random.Random(s))) == 2031 for s in SEEDS)


def test_extreme_burstiness_targets_blow_up_cross_seed_variance():
    """A third, independent reason the BURST anchor is 40 and not Philly's 160.

    At the whole-trace target the realised IDC ranges over 64..202 across five seeds -
    a spread far larger than any effect the campaign is trying to measure. The
    within-horizon anchor keeps it to roughly +/- 6.5 on a mean of 36 (PLAN §3.6), which
    is what makes a 5-seed comparison meaningful at all.
    """
    T = 4 * DAY
    def spread(target):
        p1 = arrivals.fit_duty_cycle(2031, target, T)
        vals = [arrivals.idc(arrivals.generate(2031, p1, T, random.Random(s)), HOUR, 0.0, T)
                for s in SEEDS]
        mean = sum(vals) / len(vals)
        sd = (sum((v - mean) ** 2 for v in vals) / len(vals)) ** 0.5
        return sd / mean

    assert spread(40.0) < 0.30
    assert spread(160.0) > 0.30


def test_sub_poisson_targets_are_rejected():
    with pytest.raises(ArrivalError, match="must be >= 1"):
        arrivals.fit_duty_cycle(100, 0.5, DAY)


def test_fitting_with_no_arrivals_is_rejected():
    with pytest.raises(ArrivalError, match="must be positive to fit"):
        arrivals.fit_duty_cycle(0, 10.0, DAY)
