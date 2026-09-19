"""W5.2 — Empirical Attainment Function comparison.

The EAF is the second opinion on the hypervolume bootstrap, so its own correctness cannot
rest on agreeing with that bootstrap. These tests pin it three ways:

  * hand-enumerated attainment on tiny run sets (the definition itself)
  * the probe grid is EXACT — a deliberately coarse grid must not find a larger deviation
  * moocore's exact EAF surfaces, an independent C implementation, as a cross-check
"""

from __future__ import annotations

import itertools

import numpy as np
import pytest

from eval import eaf_compare as ec


def front(*points) -> np.ndarray:
    return np.asarray(points, dtype=np.float64)


# ── Attainment: the definition ──────────────────────────────────────────────

def test_attainment_is_weak_dominance():
    f = front((2.0, 5.0))
    probes = np.array([
        [2.0, 5.0],    # exactly on the point — weakly dominated, so attained
        [3.0, 6.0],    # worse on both — attained
        [2.0, 4.0],    # better on obj 1 — NOT attained
        [1.0, 9.0],    # better on obj 0 — NOT attained
    ])
    assert ec.attains(f, probes).tolist() == [True, True, False, False]


def test_a_run_attains_a_probe_if_any_of_its_solutions_does():
    f = front((1.0, 9.0), (9.0, 1.0))
    assert ec.attains(f, np.array([[9.0, 2.0]])).tolist() == [True]
    assert ec.attains(f, np.array([[5.0, 5.0]])).tolist() == [False]


def test_an_empty_run_attains_nothing():
    assert ec.attains(np.empty((0, 2)), np.array([[1e9, 1e9]])).tolist() == [False]


def test_eaf_is_the_fraction_of_runs_attaining():
    runs = [front((1.0, 1.0)), front((2.0, 2.0)), front((3.0, 3.0))]
    probes = np.array([[1.0, 1.0], [2.0, 2.0], [3.0, 3.0], [0.5, 0.5]])
    np.testing.assert_allclose(ec.eaf_values(runs, probes),
                               [1 / 3, 2 / 3, 1.0, 0.0])


# ── The statistic, hand-computed ────────────────────────────────────────────

def test_identical_methods_have_zero_deviation():
    runs = [front((1.0, 4.0)), front((2.0, 3.0))]
    s = ec.eaf_statistic(runs, [r.copy() for r in runs])
    assert s["d_plus"] == 0.0 and s["d_minus"] == 0.0 and s["T"] == 0.0


def test_one_method_dominating_everywhere_gives_deviation_one():
    a = [front((1.0, 1.0)), front((1.0, 1.0))]
    b = [front((9.0, 9.0)), front((9.0, 9.0))]
    s = ec.eaf_statistic(a, b)
    # At z = (1,1) every A run attains and no B run does.
    assert s["d_plus"] == pytest.approx(1.0)
    assert s["d_minus"] == pytest.approx(0.0)


def test_partial_advantage_is_the_fraction_of_runs():
    # A: 1 of 2 runs reaches (1,1); B: neither does. Everywhere else they match.
    a = [front((1.0, 1.0)), front((5.0, 5.0))]
    b = [front((5.0, 5.0)), front((5.0, 5.0))]
    s = ec.eaf_statistic(a, b)
    assert s["d_plus"] == pytest.approx(0.5)
    assert s["d_minus"] == pytest.approx(0.0)


def test_the_statistic_is_symmetric_under_swapping_the_arguments():
    a = [front((1.0, 4.0)), front((2.0, 3.0))]
    b = [front((3.0, 2.0)), front((4.0, 1.0))]
    ab = ec.eaf_statistic(a, b)
    ba = ec.eaf_statistic(b, a)
    assert ab["T"] == pytest.approx(ba["T"])
    assert ab["d_plus"] == pytest.approx(ba["d_minus"])


def test_the_statistic_ignores_monotone_rescaling_of_an_objective():
    """The property that makes the EAF a genuine second opinion.

    Hypervolume changes when an objective is rescaled (the reference point moves with it);
    the EAF depends only on dominance, so it cannot. If this ever failed, the EAF verdict
    would inherit exactly the sensitivity it exists to avoid.
    """
    a = [front((1.0, 400.0)), front((2.0, 300.0))]
    b = [front((3.0, 200.0)), front((4.0, 100.0))]
    base = ec.eaf_statistic(a, b)

    def scale(runs):
        return [np.column_stack([f[:, 0] * 1000.0, np.log(f[:, 1])]) for f in runs]

    scaled = ec.eaf_statistic(scale(a), scale(b))
    assert scaled["T"] == pytest.approx(base["T"])
    assert scaled["d_plus"] == pytest.approx(base["d_plus"])


# ── The probe grid is exact, not a sample ──────────────────────────────────

def test_probe_grid_is_the_cross_product_of_observed_coordinates():
    a = [front((1.0, 4.0))]
    b = [front((3.0, 2.0))]
    grid = ec.probe_grid(a, b)
    rows = {tuple(r) for r in grid}
    assert rows == {(1.0, 4.0), (1.0, 2.0), (3.0, 4.0), (3.0, 2.0)}


def test_a_denser_random_probe_set_finds_no_larger_deviation():
    """The exactness claim, checked rather than asserted.

    The EAF only steps at observed coordinate values, so the cross-product grid must
    already contain a maximiser. A dense random probe cloud is allowed to tie it — never to
    beat it.
    """
    rng = np.random.default_rng(7)
    a = [rng.uniform(0, 10, size=(3, 2)) for _ in range(4)]
    b = [rng.uniform(0, 10, size=(3, 2)) for _ in range(4)]
    exact = ec.eaf_statistic(a, b)["T"]
    dense = ec.eaf_statistic(a, b, probes=rng.uniform(-1, 11, size=(20000, 2)))["T"]
    assert dense <= exact + 1e-12


def test_probe_grid_rejects_three_objectives():
    a = [np.array([[1.0, 2.0, 3.0]])]
    with pytest.raises(ValueError, match="2 objectives"):
        ec.probe_grid(a, a)


# ── Permutation test ────────────────────────────────────────────────────────

def test_permutation_test_is_exact_at_campaign_sizes():
    a = [front((float(i), 10.0 - i)) for i in range(1, 6)]
    b = [front((float(i) + 0.5, 10.5 - i)) for i in range(1, 6)]
    res = ec.permutation_test(a, b)
    assert res["exact"] is True
    # C(10,5) = 252 splits.
    assert res["n_permutations"] == 252
    assert res["p_resolution"] == pytest.approx(1 / 252)


def test_p_value_can_never_be_zero():
    """The observed labelling is one of the permutations, so p >= 1/n.

    Omitting it is a classic slip that yields "p = 0", a claim no finite permutation set
    can support.
    """
    a = [front((1.0, 1.0))] * 4
    b = [front((9.0, 9.0))] * 4
    res = ec.permutation_test(a, b)
    assert res["p_value"] >= res["p_resolution"] - 1e-12
    assert res["p_value"] > 0.0


def test_identical_methods_are_never_significant():
    runs = [front((float(i), 10.0 - i)) for i in range(1, 6)]
    res = ec.compare(runs, [r.copy() for r in runs], name_a="x", name_b="y")
    assert res["T"] == 0.0
    assert res["p_value"] == pytest.approx(1.0)
    assert res["significant"] is False
    assert res["better"] == ec.NO_DIFFERENCE


def test_a_clearly_dominating_method_wins_and_names_itself():
    a = [front((1.0, 1.0)) for _ in range(5)]
    b = [front((9.0, 9.0)) for _ in range(5)]
    res = ec.compare(a, b, name_a="good", name_b="bad")
    assert res["significant"] is True
    assert res["better"] == "good"
    assert res["d_plus"] > res["d_minus"]


def test_equal_advantages_are_reported_as_crossing_not_as_a_tie():
    """The distinction the campaign data actually needs (LOW crosses).

    A and B each attain a region the other cannot: A reaches low-energy points, B reaches
    low-SLA points. Calling that a "tie" would say the two are interchangeable, which is
    close to the opposite of what the data shows.
    """
    a = [front((1.0, 9.0)) for _ in range(5)]
    b = [front((9.0, 1.0)) for _ in range(5)]
    res = ec.compare(a, b, name_a="A", name_b="B")
    assert res["d_plus"] == pytest.approx(res["d_minus"])
    assert res["significant"] is True
    assert res["better"] == ec.CROSSING
    assert res["better"] not in ("A", "B", ec.NO_DIFFERENCE)


def test_unequal_but_both_substantial_advantages_are_still_crossing():
    """The homo/LOW case: d_plus=0.8 vs d_minus=1.0 is crossing, not "B wins".

    Ranking the two maxima orders regions, not methods. Four of A's five runs reach a
    corner B never reaches; calling B uniformly better because its own exclusive corner
    scores 1.0 instead of 0.8 contradicted both the paired per-seed comparison and the
    hypervolume bootstrap on real campaign data.
    """
    # A reaches a low-energy corner in 4 runs out of 5; B reaches a low-SLA corner in all
    # 5. Neither ordering holds everywhere.
    a = [front((1.0, 9.0)) for _ in range(4)] + [front((9.0, 9.0))]
    b = [front((9.0, 1.0)) for _ in range(5)]
    res = ec.compare(a, b, name_a="A", name_b="B")
    assert res["significant"] is True
    assert res["d_plus"] == pytest.approx(0.8)
    assert res["d_minus"] == pytest.approx(1.0)
    assert res["better"] == ec.CROSSING, "unequal maxima over disjoint regions do not order the methods"


def test_a_single_outlier_run_does_not_block_a_clear_direction():
    """One run's worth of exclusive ground is noise, not a crossing front."""
    a = [front((1.0, 1.0)) for _ in range(5)]
    b = [front((9.0, 9.0)) for _ in range(4)] + [front((0.5, 20.0))]
    res = ec.compare(a, b, name_a="good", name_b="bad")
    assert res["significant"] is True
    assert 0 < res["d_minus"] <= 0.2
    assert res["better"] == "good"


def test_crossing_against_an_hv_winner_is_partial_not_a_contradiction():
    """HV integrates area against a fixed nadir, so it can name a winner where the EAF
    finds no uniform one. Informative, but not grounds to stop."""
    eaf = {"better": ec.CROSSING}
    hv = {"a": "cmdp-pid", "b": "random", "delta_point": -0.12,
          "significant_95": True}
    assert ec.agreement(eaf, hv)["status"].startswith("partial (EAF: fronts cross")


def test_negative_zero_never_reaches_the_report():
    """max(-0.0, 0.0) returns -0.0 in Python, which prints as "-0.000"."""
    runs = [front((1.0, 1.0)) for _ in range(3)]
    s = ec.eaf_statistic(runs, [r.copy() for r in runs])
    assert str(s["d_minus"]) == "0.0" and str(s["d_plus"]) == "0.0"


def test_direction_flips_with_the_argument_order():
    a = [front((1.0, 1.0)) for _ in range(5)]
    b = [front((9.0, 9.0)) for _ in range(5)]
    assert ec.compare(a, b, name_a="g", name_b="d")["better"] == "g"
    assert ec.compare(b, a, name_a="d", name_b="g")["better"] == "g"


def test_permutation_null_distribution_is_uniform_enough_under_h0():
    """Exchangeable runs must not be flagged as different.

    Ten runs drawn from ONE distribution, split arbitrarily: a test that called this
    significant would manufacture findings out of seed noise, which is precisely the failure
    the campaign is trying to avoid.
    """
    rng = np.random.default_rng(11)
    runs = [rng.uniform(0, 10, size=(3, 2)) for _ in range(10)]
    flagged = 0
    trials = 0
    for idx in itertools.islice(itertools.combinations(range(10), 5), 12):
        sel = set(idx)
        a = [runs[i] for i in range(10) if i in sel]
        b = [runs[i] for i in range(10) if i not in sel]
        trials += 1
        flagged += ec.compare(a, b)["significant"]
    assert flagged <= 1, f"{flagged}/{trials} false positives under H0"


def test_empty_method_is_refused():
    with pytest.raises(ValueError, match="both methods need runs"):
        ec.permutation_test([front((1.0, 1.0))], [])


# ── moocore cross-check ─────────────────────────────────────────────────────

def test_moocore_surfaces_agree_with_our_eaf():
    moocore = pytest.importorskip("moocore")  # noqa: F841
    rng = np.random.default_rng(3)
    runs = [rng.uniform(0, 10, size=(4, 2)) for _ in range(5)]
    out = ec.moocore_surface_check(runs)
    assert out["available"] is True
    assert out["n_surface_points"] > 0
    assert out["ok"] is True, f"max shortfall {out['max_shortfall']}"


def test_cross_check_degrades_to_unavailable_without_moocore(monkeypatch):
    import builtins
    real_import = builtins.__import__

    def fake(name, *a, **k):
        if name == "moocore":
            raise ImportError("simulated")
        return real_import(name, *a, **k)

    monkeypatch.setattr(builtins, "__import__", fake)
    assert ec.moocore_surface_check([front((1.0, 1.0))]) == {"available": False}


# ── Agreement bookkeeping against the HV bootstrap ─────────────────────────

def test_agreement_matches_when_both_pick_the_same_winner():
    eaf = {"better": "cmdp-pid"}
    hv = {"a": "cmdp-pid", "b": "ppo-fixed", "delta_point": 0.07,
          "significant_95": True}
    assert ec.agreement(eaf, hv)["status"] == "agree"


def test_agreement_matches_when_both_find_nothing():
    eaf = {"better": ec.NO_DIFFERENCE}
    hv = {"a": "cmdp-pid", "b": "ppo-fixed", "delta_point": 0.03,
          "significant_95": False}
    assert ec.agreement(eaf, hv)["status"] == "agree"


def test_opposite_winners_are_flagged_as_a_contradiction():
    """The case W5.2 exists to catch: one test says A wins, the other says B does."""
    eaf = {"better": "ppo-fixed"}
    hv = {"a": "cmdp-pid", "b": "ppo-fixed", "delta_point": 0.07,
          "significant_95": True}
    assert ec.agreement(eaf, hv)["status"] == "CONTRADICTION"


def test_one_inconclusive_test_is_partial_not_a_contradiction():
    eaf = {"better": "cmdp-pid"}
    hv = {"a": "cmdp-pid", "b": "ppo-fixed", "delta_point": 0.03,
          "significant_95": False}
    assert ec.agreement(eaf, hv)["status"].startswith("partial")


def test_hv_direction_reads_the_sign_of_the_gap():
    assert ec._hv_direction({"a": "x", "b": "y", "delta_point": -0.2,
                             "significant_95": True}) == "y"
    assert ec._hv_direction({"a": "x", "b": "y", "delta_point": 0.2,
                             "significant_95": True}) == "x"
    assert ec._hv_direction({"a": "x", "b": "y", "delta_point": 0.2,
                             "significant_95": False}) == ec.NO_DIFFERENCE


def test_missing_hv_baseline_is_reported_not_assumed():
    assert ec.agreement({"better": "x"}, None)["status"] == "no-hv-baseline"


# ── Runs are per-seed, unlike the bootstrap's averaged front ───────────────

def test_runs_by_family_keeps_each_seed_separate():
    from eval.points import PointRecord

    pts = [
        PointRecord(scenario="LOW", method="cmdp-d0.02", seed=42,
                    energy_kwh=10.0, sla_cost=100.0),
        PointRecord(scenario="LOW", method="cmdp-d0.04", seed=42,
                    energy_kwh=11.0, sla_cost=80.0),
        PointRecord(scenario="LOW", method="cmdp-d0.02", seed=43,
                    energy_kwh=12.0, sla_cost=90.0),
    ]
    tree = ec.runs_by_family(pts)
    assert set(tree) == {"cmdp-pid"}
    assert sorted(tree["cmdp-pid"]) == [42, 43]
    # Seed 42 ran two budgets ⇒ its front has two points, not one averaged point.
    assert tree["cmdp-pid"][42].shape == (2, 2)
    assert tree["cmdp-pid"][43].shape == (1, 2)


def test_nsga2_is_excluded_from_the_comparison():
    from eval.points import PointRecord

    pts = [PointRecord(scenario="LOW", method="nsga2", seed=42,
                       energy_kwh=1.0, sla_cost=1.0)]
    assert ec.runs_by_family(pts) == {}
