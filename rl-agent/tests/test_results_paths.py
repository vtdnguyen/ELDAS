"""Pre-flight guards on where results may be written (PLAN §4.1, risks R5/R6).

Both failures these prevent are *silent*. A WM-1 run pointed at the LEGACY results root
overwrites the Phase-1 baseline the whole thesis compares against, and the directory looks
perfectly normal afterwards. Two arms sharing one root overwrite each other the same way —
which has already happened in this project once (C16, "LOW ppo-min stale").

An overnight campaign is exactly when nobody is watching, so the check has to be a hard
failure at start-up rather than something to notice in the morning.
"""

from __future__ import annotations

import pytest

from eval import paths


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    monkeypatch.delenv("TRACE_PATTERN", raising=False)


# ── Mode detection ──────────────────────────────────────────────────────────

def test_no_trace_pattern_means_legacy_mode():
    assert paths.wm1_mode() is False
    assert paths.wm1_mode("") is False
    assert paths.wm1_mode("   ") is False


def test_a_trace_pattern_means_wm1_mode(monkeypatch):
    monkeypatch.setenv("TRACE_PATTERN", "/data/wm1/homo/{scenario}/seed{seed}.csv")
    assert paths.wm1_mode() is True


@pytest.mark.parametrize("pattern,arm", [
    ("/data/wm1/homo/{scenario}/seed{seed}.csv", "homo"),
    ("/data/wm1/hetero/{scenario}/seed{seed}.csv", "hetero"),
    ("/somewhere/else/{scenario}.csv", None),
])
def test_arm_is_inferred_only_for_reporting(pattern, arm):
    assert paths.arm_from_pattern(pattern) == arm


# ── The guard ───────────────────────────────────────────────────────────────

def test_legacy_run_may_write_to_the_legacy_root():
    assert paths.guard_results_root("/data/results") == "/data/results"


def test_wm1_run_is_refused_the_legacy_root(monkeypatch):
    monkeypatch.setenv("TRACE_PATTERN", "/data/wm1/homo/{scenario}/seed{seed}.csv")
    with pytest.raises(paths.ResultsPathError, match="LEGACY results directory"):
        paths.guard_results_root("/data/results")


def test_the_refusal_names_the_root_to_use_instead(monkeypatch):
    monkeypatch.setenv("TRACE_PATTERN", "/data/wm1/hetero/{scenario}/seed{seed}.csv")
    with pytest.raises(paths.ResultsPathError) as e:
        paths.guard_results_root("/data/results")
    assert "/data/results/wm1/hetero" in str(e.value)


def test_an_unrecognised_arm_still_refuses_but_cannot_guess(monkeypatch):
    monkeypatch.setenv("TRACE_PATTERN", "/tmp/custom/{scenario}.csv")
    with pytest.raises(paths.ResultsPathError, match=r"wm1/<arm>"):
        paths.guard_results_root("/data/results")


@pytest.mark.parametrize("root", [
    "/data/results/wm1/homo",
    "/data/results/wm1/hetero",
    "/data/results/wm1/homo/",
    "/somewhere/scratch",
])
def test_any_root_outside_the_legacy_one_is_allowed(root, monkeypatch):
    monkeypatch.setenv("TRACE_PATTERN", "/data/wm1/homo/{scenario}/seed{seed}.csv")
    assert paths.guard_results_root(root) == root


def test_a_trailing_slash_does_not_smuggle_the_legacy_root_past_the_guard(monkeypatch):
    """"/data/results/" is the same directory as "/data/results"."""
    monkeypatch.setenv("TRACE_PATTERN", "/data/wm1/homo/{scenario}/seed{seed}.csv")
    with pytest.raises(paths.ResultsPathError):
        paths.guard_results_root("/data/results/")


def test_the_guard_never_rewrites_the_path():
    """Auto-correcting would put results somewhere the operator never checked."""
    assert paths.guard_results_root("/somewhere/odd") == "/somewhere/odd"


# ── Root construction ───────────────────────────────────────────────────────

def test_wm1_root_puts_the_arm_in_the_path():
    assert paths.wm1_results_root("homo") == "/data/results/wm1/homo"
    assert paths.wm1_results_root("hetero") == "/data/results/wm1/hetero"


def test_the_two_arms_get_different_roots():
    """R6 in one line: this is what stops hetero overwriting homo."""
    assert paths.wm1_results_root("homo") != paths.wm1_results_root("hetero")


def test_an_unknown_arm_is_refused():
    with pytest.raises(paths.ResultsPathError, match="unknown arm"):
        paths.wm1_results_root("homogeneous")
