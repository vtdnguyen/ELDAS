"""G2.4 tests — hypervolume + IGD+ pinned to hand computations, plus the fixed
reference-geometry invariants (Lưu ý #8/#15). These are the scientific-
correctness checks for the Pareto evaluation harness.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from eval import pareto_metrics as pm


# ── Reference geometry ───────────────────────────────────────────────────────

def test_ideal_and_nadir_are_fixed_corners():
    F = np.array([[1.0, 8.0], [3.0, 2.0], [2.0, 5.0]])
    assert pm.ideal_point(F).tolist() == [1.0, 2.0]
    nadir = pm.nadir_point(F, margin=0.10)
    # max = [3, 8]; span = [2, 6]; nadir = max + 0.1*span = [3.2, 8.6]
    assert nadir == pytest.approx([3.2, 8.6])


def test_normalize_maps_ideal_to_0_and_nadir_to_1():
    ideal = np.array([1.0, 2.0])
    nadir = np.array([3.0, 8.0])
    out = pm.normalize(np.array([[1.0, 2.0], [3.0, 8.0], [2.0, 5.0]]), ideal, nadir)
    assert out[0].tolist() == [0.0, 0.0]
    assert out[1].tolist() == [1.0, 1.0]
    assert out[2] == pytest.approx([0.5, 0.5])


# ── Hypervolume (exact, hand-computed) ──────────────────────────────────────

def test_hv_2d_single_point():
    hv = pm._hv_2d_exact(np.array([[0.2, 0.3]]), ref=np.ones(2))
    assert hv == pytest.approx(0.8 * 0.7)          # 0.56


def test_hv_2d_two_points_with_overlap():
    # Union of [0.2,1]×[0.6,1] and [0.5,1]×[0.3,1] = 0.32 + 0.35 − 0.20 = 0.47
    hv = pm._hv_2d_exact(np.array([[0.2, 0.6], [0.5, 0.3]]), ref=np.ones(2))
    assert hv == pytest.approx(0.47)


def test_hv_ignores_dominated_and_outside_points():
    # (0.5,0.5) is dominated by (0.2,0.3); (1.2,0.1) is outside the ref box.
    F = np.array([[0.2, 0.3], [0.5, 0.5], [1.2, 0.1]])
    hv = pm._hv_2d_exact(F, ref=np.ones(2))
    assert hv == pytest.approx(0.56)


def test_hypervolume_pymoo_matches_numpy_fallback():
    ideal = np.zeros(2)
    nadir = np.ones(2)
    F = np.array([[0.2, 0.6], [0.5, 0.3], [0.8, 0.1]])
    hv_np = pm.hypervolume(F, ideal, nadir, use_pymoo=False)
    try:
        import pymoo  # noqa: F401
    except ImportError:
        pytest.skip("pymoo not installed")
    hv_pm = pm.hypervolume(F, ideal, nadir, use_pymoo=True)
    assert hv_pm == pytest.approx(hv_np, rel=1e-6)


# ── W5.1 — three independent backends must agree ────────────────────────────
#
# Two agreeing implementations is weak evidence: the failure that matters here is a shared
# misreading of a published definition (IGD+ vs IGD, whether points outside the reference
# box count, whether the reference point is inclusive), and two libraries can misread it
# the same way. A third — moocore, the reference C implementation from the group that
# introduced IGD+ and the EAF — makes that coincidence implausible.

UNIT = (np.zeros(2), np.ones(2))

#: (label, front, expected HV against the unit reference box). The expectations are the
#: hand-worked values already pinned above, so this table checks the BACKENDS against
#: arithmetic, not merely against each other.
HV_CASES = [
    ("single point", np.array([[0.2, 0.3]]), 0.56),
    ("two points, overlapping boxes", np.array([[0.2, 0.6], [0.5, 0.3]]), 0.47),
    ("dominated + outside-the-box points ignored",
     np.array([[0.2, 0.3], [0.5, 0.5], [1.2, 0.1]]), 0.56),
    ("three-point staircase", np.array([[0.2, 0.6], [0.5, 0.3], [0.8, 0.1]]), 0.51),
    ("point on the reference corner contributes nothing",
     np.array([[1.0, 0.5]]), 0.0),
    ("ideal corner dominates the whole box", np.array([[0.0, 0.0]]), 1.0),
]

#: (label, approximation, reference set, expected IGD+).
IGD_CASES = [
    ("single reference point", np.array([[0.2, 0.2]]), np.array([[0.0, 0.0]]),
     math.sqrt(0.08)),
    ("approximation dominates the reference ⇒ 0", np.array([[0.0, 0.0]]),
     np.array([[0.5, 0.5]]), 0.0),
    ("two approximation points, one reference",
     np.array([[0.0, 0.5], [0.5, 0.0]]), np.array([[0.0, 0.0]]), 0.5),
    ("identical sets ⇒ 0", np.array([[0.1, 0.9], [0.9, 0.1]]),
     np.array([[0.1, 0.9], [0.9, 0.1]]), 0.0),
]


def _backends_or_skip(*needed):
    have = pm.available_backends()
    missing = [b for b in needed if b not in have]
    if missing:
        pytest.skip(f"backend(s) not installed: {missing} (have {have})")


@pytest.mark.parametrize("label,F,expected", HV_CASES, ids=[c[0] for c in HV_CASES])
def test_hv_agrees_across_all_three_backends(label, F, expected):
    _backends_or_skip("pymoo", "moocore")
    ideal, nadir = UNIT
    values = {b: pm.hypervolume(F, ideal, nadir, backend=b) for b in pm.BACKENDS}
    for b, v in values.items():
        assert v == pytest.approx(expected, rel=1e-6, abs=1e-12), f"{b}: {values}"


@pytest.mark.parametrize("label,A,R,expected", IGD_CASES, ids=[c[0] for c in IGD_CASES])
def test_igd_plus_agrees_across_all_three_backends(label, A, R, expected):
    _backends_or_skip("pymoo", "moocore")
    ideal, nadir = UNIT
    values = {b: pm.igd_plus(A, R, ideal, nadir, backend=b) for b in pm.BACKENDS}
    for b, v in values.items():
        assert v == pytest.approx(expected, rel=1e-6, abs=1e-12), f"{b}: {values}"


def test_evaluate_methods_agrees_across_backends_end_to_end():
    """The agreement must survive the real entry point, not just the kernels.

    Normalisation, the shared nadir and the union reference front all sit between the raw
    points and the indicator; a backend swap must not perturb any of them.
    """
    _backends_or_skip("pymoo", "moocore")
    method_points = {
        "good": np.array([[10.0, 100.0], [12.0, 60.0], [14.0, 40.0]]),
        "bad": np.array([[13.0, 120.0], [16.0, 90.0], [18.0, 80.0]]),
    }
    runs = {b: pm.evaluate_methods(method_points, backend=b) for b in pm.BACKENDS}
    for name in ("good", "bad"):
        hvs = [runs[b]["methods"][name]["hypervolume"] for b in pm.BACKENDS]
        igds = [runs[b]["methods"][name]["igd_plus"] for b in pm.BACKENDS]
        assert hvs[1] == pytest.approx(hvs[0], rel=1e-6)
        assert hvs[2] == pytest.approx(hvs[0], rel=1e-6)
        assert igds[1] == pytest.approx(igds[0], rel=1e-6, abs=1e-12)
        assert igds[2] == pytest.approx(igds[0], rel=1e-6, abs=1e-12)
    assert runs["moocore"]["backend"] == "moocore"


def test_a_named_missing_backend_raises_instead_of_falling_back(monkeypatch):
    """A silent fallback would let the three-way test compare NumPy with itself."""
    monkeypatch.setattr(pm, "_have", lambda name: name == "numpy")
    with pytest.raises(pm.BackendUnavailable, match="moocore"):
        pm.resolve_backend("moocore")
    # "auto" is allowed to fall back — that is what "auto" means.
    assert pm.resolve_backend("auto") == "numpy"


def test_unknown_backend_name_is_rejected():
    with pytest.raises(ValueError, match="unknown backend"):
        pm.resolve_backend("scipy")


def test_legacy_use_pymoo_flag_still_selects_the_old_behaviour():
    assert pm.resolve_backend(None, use_pymoo=False) == "numpy"
    assert pm.resolve_backend(None, use_pymoo=True) in pm.BACKENDS


# ── IGD+ (exact, hand-computed) ─────────────────────────────────────────────

def test_igd_plus_single_reference():
    a = np.array([[0.2, 0.2]])
    r = np.array([[0.0, 0.0]])
    val = pm._igd_plus_np(a, r)
    assert val == pytest.approx(math.sqrt(0.08))   # ||(0.2,0.2)||


def test_igd_plus_is_zero_when_approximation_dominates_reference():
    # A better than R on both objs ⇒ d+ = ||max(A−R,0)|| = 0 (Pareto-compliant).
    a = np.array([[0.0, 0.0]])
    r = np.array([[0.5, 0.5]])
    assert pm._igd_plus_np(a, r) == pytest.approx(0.0)


def test_igd_plus_two_reference_points():
    a = np.array([[0.0, 0.5], [0.5, 0.0]])
    r = np.array([[0.0, 0.0]])
    assert pm._igd_plus_np(a, r) == pytest.approx(0.5)


# ── Reference front = union of non-dominated ────────────────────────────────

def test_union_reference_front():
    method_points = {
        "a": np.array([[1.0, 5.0], [3.0, 2.0]]),   # both non-dominated
        "b": np.array([[4.0, 4.0], [2.0, 3.0]]),   # [2,3] non-dom; [4,4] dominated
    }
    front = pm.union_reference_front(method_points)
    rows = {tuple(r) for r in front}
    assert (1.0, 5.0) in rows and (3.0, 2.0) in rows and (2.0, 3.0) in rows
    assert (4.0, 4.0) not in rows


# ── End-to-end: a dominating method scores strictly better ──────────────────

def test_better_method_has_higher_hv_and_lower_igd():
    # "good" dominates "bad" everywhere (lower energy AND lower sla).
    method_points = {
        "good": np.array([[10.0, 100.0], [12.0, 60.0], [14.0, 40.0]]),
        "bad":  np.array([[13.0, 120.0], [16.0, 90.0], [18.0, 80.0]]),
    }
    res = pm.evaluate_methods(method_points, use_pymoo=False)
    good, bad = res["methods"]["good"], res["methods"]["bad"]
    assert good["hypervolume"] > bad["hypervolume"]
    assert good["igd_plus"] < bad["igd_plus"]
    # Fixed geometry is shared (one ideal / nadir for both).
    assert res["ideal"][0] == 10.0 and res["ideal"][1] == 40.0
    assert res["sense"] == ["min", "min"]


def test_evaluate_methods_handles_empty_method():
    res = pm.evaluate_methods(
        {"x": np.array([[1.0, 1.0], [2.0, 0.5]]), "empty": np.empty((0, 2))},
        use_pymoo=False,
    )
    assert res["methods"]["empty"]["n_points"] == 0
    assert res["methods"]["empty"]["hypervolume"] == 0.0
    assert math.isinf(res["methods"]["empty"]["igd_plus"])
