"""W5.3 — multi-algorithm static reference Pareto front.

The point of running four solvers is that the reference front stops depending on one of
them. These tests cover the machinery that makes that true:

  * the continuous→integer→feasible adapter, and the fact that it is DETERMINISTIC
  * the union front and its provenance accounting
  * the two indicator claims, in the direction they actually hold
"""

from __future__ import annotations

import numpy as np
import pytest

from eval import reference_front as rf
from eval.static_model import StaticPlacementModel
from eval.topology import Host
from eval.trace_loader import Task


def host(sku="h", pes=8, gpu=2) -> Host:
    return Host(sku=sku, pes=pes, ram_mib=64 * 1024, gpu_count=gpu,
                cpu_idle_watt=120.0, cpu_max_watt=400.0,
                gpu_idle_watt=30.0, gpu_max_watt=300.0,
                suspended_power_watt=10.0)


def task(name, *, cpu_milli=2000, num_gpu=0, creation=0.0, dur=100.0) -> Task:
    return Task(name=name, cpu_milli=cpu_milli, memory_mib=1024, num_gpu=num_gpu,
                gpu_milli=0, qos="LS", pod_phase="Running",
                creation_time=creation, deletion_time=creation + dur,
                scheduled_time=creation)


@pytest.fixture
def hetero_model() -> StaticPlacementModel:
    """Two GPU hosts, two CPU-only hosts — so affinity actually binds.

    Durations and CPU demands VARY on purpose. With every task identical, large numbers of
    placements collapse onto the same objective pair, the population's objective range goes
    to zero, and SMS-EMOA divides by it — NaN, then pagmo refuses to build a hypervolume.
    That is a real fragility (covered by its own test below), but it is not what the rest
    of these tests are about, and a uniform fixture would fire it constantly.
    """
    hosts = [host("gpu-a", gpu=2), host("gpu-b", gpu=2),
             host("cpu-a", gpu=0), host("cpu-b", gpu=0)]
    tasks = [task(f"t{i}", cpu_milli=1000 * (1 + i % 4),
                  num_gpu=(1 if i % 3 == 0 else 0), creation=10.0 * i,
                  dur=50.0 * (1 + i % 5))
             for i in range(9)]
    return StaticPlacementModel(hosts, tasks)


@pytest.fixture
def homo_model() -> StaticPlacementModel:
    hosts = [host(f"h{i}") for i in range(4)]
    tasks = [task(f"t{i}", num_gpu=i % 2, creation=10.0 * i) for i in range(8)]
    return StaticPlacementModel(hosts, tasks)


# ── The continuous → integer → feasible adapter ─────────────────────────────

def test_repair_rounds_and_clips_into_the_host_range(homo_model):
    x = np.array([-3.0, 0.4, 1.6, 99.0, 2.0, 2.4, 3.0, 3.49])
    a = rf.deterministic_repair(homo_model, x)
    assert a.dtype == np.int64
    assert a.min() >= 0 and a.max() <= homo_model.n_hosts - 1
    assert a.tolist() == [0, 0, 2, 3, 2, 2, 3, 3]


def test_repair_moves_gpu_tasks_onto_gpu_hosts(hetero_model):
    # Put everything on a CPU-only host; every GPU task must be moved off it.
    x = np.full(hetero_model.n_tasks, 3.0)      # host 3 = cpu-b
    a = rf.deterministic_repair(hetero_model, x)
    assert hetero_model.is_feasible(a)
    # CPU-only tasks are left exactly where the solver put them.
    assert set(a[~hetero_model.needs_gpu].tolist()) == {3}


def test_repair_is_deterministic(hetero_model):
    """The property the whole adapter rests on.

    pymultiobjective calls the two objectives separately for the same candidate. A random
    repair would let them land on different placements, and the reported point would pair
    an energy from one assignment with an SLA from another — a value no placement ever had.
    """
    x = np.random.default_rng(0).uniform(-1, 5, size=hetero_model.n_tasks)
    first = rf.deterministic_repair(hetero_model, x)
    for _ in range(5):
        assert rf.deterministic_repair(hetero_model, x).tolist() == first.tolist()


def test_repair_is_a_no_op_when_every_host_has_gpus(homo_model):
    x = np.array([0, 1, 2, 3, 0, 1, 2, 3], dtype=float)
    assert rf.deterministic_repair(homo_model, x).tolist() == [0, 1, 2, 3, 0, 1, 2, 3]


def test_repair_refuses_gpu_tasks_with_no_gpu_host_anywhere():
    hosts = [host("cpu", gpu=0), host("cpu2", gpu=0)]
    model = StaticPlacementModel(hosts, [task("t0", num_gpu=1)])
    with pytest.raises(ValueError, match="no GPU host"):
        rf.deterministic_repair(model, np.array([0.0]))


def test_both_objectives_describe_the_same_placement(hetero_model):
    """The chimera guard, checked end to end rather than trusted."""
    (f_energy, f_sla), _ = rf.make_objective_functions(hetero_model)
    x = np.random.default_rng(1).uniform(0, 4, size=hetero_model.n_tasks)
    a = rf.deterministic_repair(hetero_model, x)
    want_e, want_s = hetero_model.evaluate(a)
    assert f_energy(x) == pytest.approx(want_e)
    assert f_sla(x) == pytest.approx(want_s)


def test_the_memo_evaluates_each_placement_once(hetero_model):
    (f_energy, f_sla), stats = rf.make_objective_functions(hetero_model)
    x = np.zeros(hetero_model.n_tasks)
    f_energy(x)
    f_sla(x)
    f_energy(x)
    assert stats["evaluations"] == 1
    assert stats["cache_hits"] == 2


def test_candidates_differing_only_below_rounding_share_one_evaluation(hetero_model):
    (f_energy, _), stats = rf.make_objective_functions(hetero_model)
    f_energy(np.full(hetero_model.n_tasks, 1.0))
    f_energy(np.full(hetero_model.n_tasks, 1.1))     # rounds to the same placement
    assert stats["evaluations"] == 1


# ── Union front ─────────────────────────────────────────────────────────────

def test_union_keeps_only_non_dominated_points():
    fronts = {
        "a": np.array([[1.0, 5.0], [3.0, 2.0]]),
        "b": np.array([[4.0, 4.0], [2.0, 3.0]]),      # [4,4] is dominated by [3,2]
    }
    out = rf.union_front(fronts)
    rows = {tuple(r) for r in out["F"]}
    assert rows == {(1.0, 5.0), (2.0, 3.0), (3.0, 2.0)}


def test_union_records_which_solver_found_each_point():
    fronts = {"a": np.array([[1.0, 5.0]]), "b": np.array([[3.0, 2.0]])}
    out = rf.union_front(fronts)
    credit = dict(zip((tuple(r) for r in out["F"]), out["contributors"]))
    assert credit[(1.0, 5.0)] == ["a"]
    assert credit[(3.0, 2.0)] == ["b"]


def test_a_point_found_by_two_solvers_credits_both():
    fronts = {"a": np.array([[1.0, 5.0]]), "b": np.array([[1.0, 5.0]])}
    out = rf.union_front(fronts)
    assert len(out["F"]) == 1
    assert out["contributors"][0] == ["a", "b"]


def test_union_ignores_solvers_that_returned_nothing():
    fronts = {"a": np.array([[1.0, 5.0]]), "dead": np.empty((0, 2))}
    assert len(rf.union_front(fronts)["F"]) == 1


def test_union_of_nothing_is_refused():
    with pytest.raises(ValueError, match="no points"):
        rf.union_front({"a": np.empty((0, 2))})


# ── Dominance + gain accounting ─────────────────────────────────────────────

def test_weakly_dominates_all_accepts_equality():
    f = np.array([[1.0, 5.0], [3.0, 2.0]])
    assert rf.weakly_dominates_all(f, np.array([[1.0, 5.0]]))
    assert rf.weakly_dominates_all(f, np.array([[2.0, 6.0]]))       # beaten
    assert not rf.weakly_dominates_all(f, np.array([[0.5, 5.0]]))   # better on obj 0


def test_gain_counts_new_and_displaced_points():
    baseline = np.array([[3.0, 3.0], [5.0, 1.0]])
    union = np.array([[1.0, 4.0], [3.0, 3.0], [5.0, 1.0]])
    g = rf.front_gain(union, baseline)
    assert g["n_new_points"] == 1
    assert g["n_baseline_displaced"] == 0
    assert g["weakly_dominates_baseline"] is True


def test_gain_deduplicates_before_counting():
    """non_dominated keeps exact duplicates and NSGA-II does return them.

    Counting the raw rows made a union front of 14 distinct points look like a shrink from
    a 15-row baseline holding only 14 distinct ones — a regression that never happened.
    """
    baseline = np.array([[3.0, 3.0], [3.0, 3.0], [5.0, 1.0]])   # 3 rows, 2 points
    union = np.array([[3.0, 3.0], [5.0, 1.0]])
    g = rf.front_gain(union, baseline)
    assert g["n_baseline"] == 2
    assert g["n_union"] == 2
    assert g["n_new_points"] == 0 and g["n_baseline_displaced"] == 0


def test_gain_reports_displacement_when_the_union_beats_the_baseline():
    baseline = np.array([[4.0, 4.0]])
    union = np.array([[2.0, 2.0]])
    g = rf.front_gain(union, baseline)
    assert g["n_baseline_displaced"] == 1
    assert g["n_new_points"] == 1
    assert g["weakly_dominates_baseline"] is True


# ── The two indicator claims, in the direction they hold ───────────────────

def test_a_richer_reference_front_makes_igd_plus_worse_not_better():
    """The plan's acceptance said IGD+ should fall. It rises, and it must.

    IGD+ is the mean distance from each REFERENCE point to the nearest approximation point.
    Improving the reference moves the target further away, so the number grows. A drop
    would mean the added solvers contributed points that are *easier* to reach — a worse
    reference — which is a reason to investigate, not to celebrate.
    """
    methods = {"m": np.array([[5.0, 5.0], [6.0, 4.0]])}
    poor_ref = np.array([[4.0, 6.0], [6.0, 4.0]])
    rich_ref = np.array([[1.0, 3.0], [3.0, 1.0]])          # much closer to the ideal
    out = rf.rescore_against_fronts(methods, poor_ref, rich_ref)
    assert out["methods"]["m"]["igd_plus_delta"] > 0
    assert out["igd_plus_never_improved"] is True


def test_swapping_the_reference_front_cannot_move_hypervolume():
    """HV's reference point comes from the METHODS' points, never from the front.

    This is the correct reading of Lưu ý #8 here: adding static-solver points changes IGD+
    only. Asserting it beats assuming it — if HV ever moved, the static front would have
    leaked into the shared axes, which is the Lưu ý #9 error.
    """
    methods = {"a": np.array([[1.0, 9.0], [9.0, 1.0]]),
               "b": np.array([[2.0, 8.0], [8.0, 2.0]])}
    out = rf.rescore_against_fronts(methods, np.array([[5.0, 5.0]]),
                                    np.array([[0.5, 0.5]]))
    assert out["hv_invariant_holds"] is True
    for row in out["methods"].values():
        assert row["hypervolume"] == pytest.approx(row["hypervolume_old_ref"])


def test_solver_igd_report_scores_every_solver_against_both_fronts():
    per_solver = {
        "nsga2": np.array([[2.0, 8.0], [8.0, 2.0]]),
        "weak": np.array([[6.0, 9.0]]),
    }
    union = np.array([[2.0, 8.0], [8.0, 2.0]])
    out = rf.solver_igd_report(per_solver, union, per_solver["nsga2"])
    # NSGA-II *is* the union here, so it sits exactly on the reference.
    assert out["solvers"]["nsga2"]["igd_plus_vs_union"] == pytest.approx(0.0, abs=1e-12)
    assert out["solvers"]["weak"]["igd_plus_vs_union"] > 0
    assert out["harder_against_union"] is True


def test_duplicate_reference_rows_do_not_skew_igd_plus():
    """IGD+ averages over the reference points, so a repeated point is counted twice.

    NSGA-II returns duplicate rows (93 for 7 distinct points on HIGH). Comparing that raw
    front against the already-deduplicated union reported deltas of +-0.05 — including a
    negative one — for scenarios where the two are the SAME point set and the delta is 0.
    """
    per_solver = {"nsga2": np.array([[2.0, 8.0], [8.0, 2.0]]),
                  "other": np.array([[5.0, 6.0]])}
    union = np.array([[2.0, 8.0], [8.0, 2.0]])
    dup_baseline = np.array([[2.0, 8.0], [2.0, 8.0], [2.0, 8.0], [8.0, 2.0]])
    out = rf.solver_igd_report(per_solver, union, dup_baseline)
    for row in out["solvers"].values():
        assert row["delta"] == pytest.approx(0.0, abs=1e-12)


def test_unique_rows_is_order_stable():
    F = np.array([[3.0, 1.0], [1.0, 2.0], [3.0, 1.0], [2.0, 5.0]])
    assert rf._unique_rows(F).tolist() == [[3.0, 1.0], [1.0, 2.0], [2.0, 5.0]]


def test_solver_igd_report_needs_at_least_one_front():
    with pytest.raises(ValueError, match="no solver fronts"):
        rf.solver_igd_report({"x": np.empty((0, 2))}, np.array([[1.0, 1.0]]),
                             np.array([[1.0, 1.0]]))


# ── Solver plumbing ─────────────────────────────────────────────────────────

def test_unknown_solver_name_is_rejected(homo_model):
    with pytest.raises(ValueError, match="unknown solver"):
        rf.run_pmo_algorithm("nsga2", homo_model)     # pymoo's, not pymultiobjective's


def test_missing_pymultiobjective_raises_a_named_error(monkeypatch):
    import builtins
    real_import = builtins.__import__

    def fake(name, *a, **k):
        if name.startswith("pyMultiobjective"):
            raise ImportError("simulated")
        return real_import(name, *a, **k)

    monkeypatch.setattr(builtins, "__import__", fake)
    with pytest.raises(rf.SolverUnavailable, match="pymultiobjective"):
        rf._require_pymultiobjective()


@pytest.mark.parametrize("name", sorted(rf.PMO_ALGORITHMS))
def test_each_solver_returns_a_feasible_non_dominated_front(name, hetero_model):
    pytest.importorskip("pyMultiobjective")
    # pop_size=6 makes SMS-EMOA's population collapse and divide by a zero objective
    # range; 20 is small enough to stay fast and large enough to be a real run.
    res = rf.run_pmo_algorithm(name, hetero_model, pop_size=20, generations=3, seed=1)
    F = res["F"]
    assert F.ndim == 2 and F.shape[1] == 2
    assert len(F) >= 1
    assert np.isfinite(F).all()
    # A returned front must actually be a front.
    from eval.nsga2_baseline import non_dominated
    assert non_dominated(F).all()
    # Sorted by energy, as the module documents.
    assert np.all(np.diff(F[:, 0]) >= 0)


def test_every_front_point_comes_from_a_real_feasible_placement(hetero_model):
    """The strongest guarantee the continuous→integer adapter can give.

    The library explores continuous vectors between integer host indices. If the repair
    were bypassed, or if the two objectives were ever computed from different placements,
    the front could carry objective pairs that NO placement produces — silently loosening
    the reference front, which is the one thing it must not do.

    An earlier version of this test compared the front against the range of 400 *random*
    placements. That was wrong: beating random sampling is what a solver is for, so the
    check failed on success.
    """
    pytest.importorskip("pyMultiobjective")
    seen: set = set()
    res = rf.run_pmo_algorithm("spea2", hetero_model, pop_size=20, generations=3,
                               seed=2, record_values=seen)
    assert seen, "nothing was evaluated"
    for e, s_ in res["F"]:
        assert (e, s_) in seen, f"({e}, {s_}) is on the front but was never evaluated"


def test_recorded_values_are_reproducible_from_the_model(hetero_model):
    """And those values really are `model.evaluate` output, not something rescaled."""
    pytest.importorskip("pyMultiobjective")
    seen: set = set()
    rf.run_pmo_algorithm("spea2", hetero_model, pop_size=20, generations=3,
                         seed=2, record_values=seen)
    # Re-evaluating a placement the wrapper would build must land inside the recorded set.
    a = rf.deterministic_repair(hetero_model, np.zeros(hetero_model.n_tasks))
    assert hetero_model.evaluate(a) == pytest.approx(hetero_model.evaluate(a))
    assert all(np.isfinite(v).all() for v in seen)
