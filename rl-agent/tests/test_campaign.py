"""G2.5/G2.6 tests — seed aggregation (mean ± 95 % CI), budget-front
monotonicity, point IO, metric family grouping, table build, and the Pareto
figure. All deterministic: no gateway, no training, no pymoo required.
"""

from __future__ import annotations

import json
import math

import numpy as np
import pytest

from eval import pareto_metrics as pm
from eval import run_campaign as rc
from eval import sweep_budget as sb
from eval.aggregate import (
    aggregate_by_method,
    check_budget_front_monotone,
    mean_ci95,
    t_critical_95,
)
from eval.points import PointRecord, load_points, method_points_dict, save_points


def _p(method, seed, e, s, scenario="LOW", **extra):
    return PointRecord(method=method, scenario=scenario, seed=seed,
                       energy_kwh=e, sla_cost=s, extra=extra or None)


# ── mean ± CI (Lưu ý #10) ───────────────────────────────────────────────────

def test_mean_ci95_hand_computed():
    # values 2,4,4,4,5,5,7,9 → mean 5, std(ddof=1) = 2.138089935
    vals = [2, 4, 4, 4, 5, 5, 7, 9]
    r = mean_ci95(vals)
    assert r.mean == pytest.approx(5.0)
    assert r.std == pytest.approx(2.1380899353, rel=1e-8)
    assert r.n == 8
    # sem = std/sqrt(8) = 0.755928946; t95(df=7) = 2.365 → ci = 1.78777...
    assert r.ci95 == pytest.approx(2.365 * 2.1380899353 / math.sqrt(8), rel=1e-9)


def test_single_seed_has_no_ci():
    # A lone sample must not pretend to a CI — it reports NaN, not ±0.
    r = mean_ci95([42.0])
    assert r.mean == 42.0
    assert math.isnan(r.ci95)
    assert r.n == 1


def test_empty_is_nan():
    r = mean_ci95([])
    assert math.isnan(r.mean) and r.n == 0


def test_t_critical_table_and_large_df_fallback():
    assert t_critical_95(4) == 2.776       # df=4 (n=5) — the ≥5-seed case
    assert t_critical_95(7) == 2.365
    assert t_critical_95(500) == 1.96      # normal approx for large df


# ── Budget-front monotonicity (Lưu ý #7) ────────────────────────────────────

def test_monotone_front_accepted():
    # d ascending (looser): SLA cost rises, energy falls → monotone.
    front = [(0.02, 100.0, 10.0), (0.04, 90.0, 20.0), (0.06, 80.0, 30.0)]
    rep = check_budget_front_monotone(front)
    assert rep.monotone
    assert rep.violations == []


def test_non_monotone_sla_inversion_flagged():
    # Loosening d from 0.04→0.06 LOWERS SLA cost ⇒ unconverged policy.
    front = [(0.02, 100.0, 10.0), (0.04, 90.0, 30.0), (0.06, 80.0, 20.0)]
    rep = check_budget_front_monotone(front)
    assert not rep.monotone
    assert (0.04, 0.06) in rep.violations
    assert "SLA cost fell" in " ".join(rep.detail)


def test_non_monotone_energy_inversion_flagged():
    # Loosening d RAISES energy ⇒ unconverged.
    front = [(0.02, 80.0, 10.0), (0.04, 95.0, 20.0)]
    rep = check_budget_front_monotone(front)
    assert not rep.monotone
    assert "energy rose" in " ".join(rep.detail)


def test_monotonicity_is_order_independent():
    # Input order must not matter — it sorts by d first.
    front = [(0.06, 80.0, 30.0), (0.02, 100.0, 10.0), (0.04, 90.0, 20.0)]
    assert check_budget_front_monotone(front).monotone


# ── Sweep front assembly (averages seeds before checking) ───────────────────

def test_build_budget_front_averages_seeds():
    points = [
        _p("cmdp-d0.02", 42, 100.0, 10.0, budget_d=0.02),
        _p("cmdp-d0.02", 43, 102.0, 12.0, budget_d=0.02),
        _p("cmdp-d0.04", 42, 90.0, 20.0, budget_d=0.04),
        _p("cmdp-d0.04", 43, 88.0, 22.0, budget_d=0.04),
    ]
    front = sb.build_budget_front(points)
    assert front == [(0.02, 101.0, 11.0), (0.04, 89.0, 21.0)]


def test_summarise_sweep_reports_monotone_verdict():
    points = [
        _p("cmdp-d0.02", 42, 100.0, 10.0, budget_d=0.02),
        _p("cmdp-d0.04", 42, 90.0, 20.0, budget_d=0.04),
    ]
    s = sb.summarise_sweep(points)
    assert s["monotone"] is True
    assert len(s["front"]) == 2
    assert s["per_budget"]["cmdp-d0.02"]["energy_mean"] == 100.0


def test_summarise_sweep_detects_unconverged_budget():
    points = [
        _p("cmdp-d0.02", 42, 100.0, 30.0, budget_d=0.02),
        _p("cmdp-d0.04", 42, 90.0, 20.0, budget_d=0.04),   # SLA fell → bad
    ]
    s = sb.summarise_sweep(points)
    assert s["monotone"] is False
    assert s["monotonicity_detail"]


# ── Point IO round-trip ─────────────────────────────────────────────────────

def test_points_roundtrip(tmp_path):
    pts = [_p("bestfit", 42, 1.0, 2.0), _p("k8s", 43, 3.0, 4.0, budget_d=0.1)]
    f = tmp_path / "sub" / "points.jsonl"
    save_points(pts, f)
    back = load_points(f)
    assert back == pts


def test_method_points_dict_groups_and_filters():
    pts = [
        _p("bestfit", 42, 1.0, 2.0, scenario="LOW"),
        _p("bestfit", 43, 3.0, 4.0, scenario="LOW"),
        _p("bestfit", 44, 9.0, 9.0, scenario="HIGH"),
    ]
    d = method_points_dict(pts, scenario="LOW")
    assert set(d) == {"bestfit"}
    assert d["bestfit"].shape == (2, 2)
    np.testing.assert_allclose(d["bestfit"], [[1.0, 2.0], [3.0, 4.0]])


# ── Metric family grouping (cmdp-d* → one method) ───────────────────────────

def test_metric_family_collapses_cmdp_budgets():
    assert rc.metric_family("cmdp-d0.04") == "cmdp-pid"
    assert rc.metric_family("cmdp-d0") == "cmdp-pid"
    assert rc.metric_family("bestfit") == "bestfit"
    assert rc.metric_family("ppo-min") == "ppo-min"


def test_family_points_dict_merges_budgets_and_adds_nsga2():
    pts = [
        _p("cmdp-d0.02", 42, 100.0, 10.0, budget_d=0.02),
        _p("cmdp-d0.04", 42, 90.0, 20.0, budget_d=0.04),
        _p("bestfit", 42, 80.0, 50.0),
    ]
    front = np.array([[70.0, 40.0], [75.0, 30.0]])
    d = rc.family_points_dict(pts, scenario="LOW", nsga2_front=front)
    assert set(d) == {"cmdp-pid", "bestfit", "nsga2"}
    assert d["cmdp-pid"].shape == (2, 2)      # both budgets = one method's front
    assert d["nsga2"].shape == (2, 2)


def test_family_points_averages_seeds_so_noise_is_not_rewarded():
    # A stochastic method must collapse to ONE mean point, not a 3-point cloud —
    # otherwise HV would reward seed variance instead of trade-off coverage.
    pts = [
        _p("random", 42, 90.0, 10.0),
        _p("random", 43, 100.0, 20.0),
        _p("random", 44, 110.0, 30.0),
    ]
    d = rc.family_points_dict(pts, scenario="LOW")
    assert d["random"].shape == (1, 2)
    np.testing.assert_allclose(d["random"], [[100.0, 20.0]])   # the seed mean

    # Opting out keeps the raw cloud (diagnostics only).
    raw = rc.family_points_dict(pts, scenario="LOW", average_seeds=False)
    assert raw["random"].shape == (3, 2)


def test_cmdp_front_keeps_one_point_per_budget_after_averaging():
    # Averaging is per-method, so the sweep still contributes its whole front.
    pts = [
        _p("cmdp-d0.02", 42, 100.0, 10.0, budget_d=0.02),
        _p("cmdp-d0.02", 43, 102.0, 12.0, budget_d=0.02),
        _p("cmdp-d0.04", 42, 90.0, 20.0, budget_d=0.04),
        _p("cmdp-d0.04", 43, 88.0, 22.0, budget_d=0.04),
    ]
    d = rc.family_points_dict(pts, scenario="LOW")
    assert d["cmdp-pid"].shape == (2, 2)       # 2 budgets, seeds averaged
    rows = sorted(tuple(r) for r in d["cmdp-pid"])
    assert rows == [(89.0, 21.0), (101.0, 11.0)]


# ── Table ───────────────────────────────────────────────────────────────────

def test_nsga2_is_not_commensurable_by_default():
    # Same task count, but the static surrogate is still a different evaluator
    # than the DES ⇒ must NOT share HV axes.
    des = [_p("bestfit", 42, 100.0, 10.0, steps=1813)]
    ok, reason = rc.nsga2_commensurable({"num_tasks": 1813}, des)
    assert ok is False
    assert "evaluator" in reason


def test_nsga2_instance_mismatch_is_reported():
    # Front solved on 80 tasks vs online methods' 1813 ⇒ mismatch called out.
    des = [_p("bestfit", 42, 100.0, 10.0, steps=1813)]
    ok, reason = rc.nsga2_commensurable({"num_tasks": 80}, des)
    assert ok is False
    assert "instance mismatch" in reason
    assert "80" in reason and "1813" in reason


def test_table_documents_nsga2_exclusion():
    pts = [_p("bestfit", 42, 100.0, 10.0), _p("k8s", 42, 110.0, 8.0)]
    agg = aggregate_by_method(pts, scenario="LOW")
    metrics = pm.evaluate_methods(rc.family_points_dict(pts, "LOW"), use_pymoo=False)
    table = rc.build_table(agg, metrics, "LOW", nsga2_note="instance mismatch: 80 vs 1813")
    assert "EXCLUDED from this table's HV/IGD+" in table
    assert "instance mismatch" in table
    # The bogus "nsga2 wins" row must not appear when it was excluded.
    assert "nsga2 (static ref)" not in table


def test_build_table_marks_under_seeded_run_and_nsga2_caveat():
    pts = [_p("bestfit", 42, 100.0, 10.0), _p("k8s", 42, 110.0, 8.0)]
    agg = aggregate_by_method(pts, scenario="LOW")
    metrics = pm.evaluate_methods(rc.family_points_dict(pts, "LOW"), use_pymoo=False)
    table = rc.build_table(agg, metrics, "LOW", min_seeds=5)
    assert "| bestfit |" in table and "| k8s |" in table
    assert "(n=1)" in table                      # no fake CI for 1 seed
    assert "Not reportable" in table             # <5 seeds ⇒ flagged
    assert "STATIC idealisation" in table        # Lưu ý #9 caveat present
    assert "Fixed reference point" in table      # Lưu ý #8 documented


def test_build_table_reports_ci_with_five_seeds():
    pts = [_p("bestfit", s, 100.0 + s, 10.0) for s in range(42, 47)]   # 5 seeds
    agg = aggregate_by_method(pts, scenario="LOW")
    metrics = pm.evaluate_methods(rc.family_points_dict(pts, "LOW"), use_pymoo=False)
    table = rc.build_table(agg, metrics, "LOW", min_seeds=5)
    assert "±" in table
    assert "Not reportable" not in table


# ── Figure ──────────────────────────────────────────────────────────────────

def test_plot_pareto_writes_figure(tmp_path):
    from eval.pareto_plot import method_family, plot_pareto

    pts = [_p(m, s, 100.0 + s + i * 10, 1e6 * (i + 1))
           for i, m in enumerate(["bestfit", "k8s", "cmdp-d0.04", "ppo-min"])
           for s in range(42, 47)]
    agg = aggregate_by_method(pts, scenario="LOW")
    front = np.array([[90.0, 5e5], [95.0, 3e5]])
    out = plot_pareto(agg, tmp_path / "p.png", "LOW", nsga2_front=front)
    assert out.exists() and out.stat().st_size > 0

    # Colour encodes FAMILY (≤4 slots for an all-pairs scatter), not rank.
    assert method_family("cmdp-d0.04") == "cmdp"
    assert method_family("nsga2") == "nsga2"
    assert method_family("ppo-min") == "ppo"
    assert method_family("bestfit") == "heuristic"


def test_plot_handles_single_seed_without_ci_whiskers(tmp_path):
    from eval.pareto_plot import plot_pareto
    agg = aggregate_by_method([_p("bestfit", 42, 100.0, 1e6)], scenario="LOW")
    assert math.isnan(agg["bestfit"]["energy"].ci95)
    out = plot_pareto(agg, tmp_path / "single.png", "LOW")
    assert out.exists()
