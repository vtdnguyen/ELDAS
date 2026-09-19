"""W2.3/W2.4 — Python mirror of the scenario filter and trace resolution.

The Java side is checked by ValidationRunner B21/B22. These tests cover the Python
half and, where it matters, that the two halves agree:

  * the LEGACY_ rename does not change the wire protocol — "LOW" still works
  * a WM-1 trace is passed through, never re-sliced
  * OVERLOAD / REPLAY are rejected by the legacy filter with a message that says why
  * trace resolution mirrors SimulationConfig.resolveTracePath, including refusing to
    fall back silently
"""

from __future__ import annotations

import os

import pytest

from eval import trace_loader as tl
from eval.trace_loader import Task

from _workload_fixtures import TRACE_PATH, requires_trace


def mk(name, phase="Running", creation=0.0):
    return Task(name=name, cpu_milli=1000, memory_mib=1, num_gpu=0, gpu_milli=0,
                qos="LS", pod_phase=phase, creation_time=creation,
                deletion_time=creation + 10.0, scheduled_time=creation)


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    """Resolution reads the environment; never inherit the caller's."""
    monkeypatch.delenv("TRACE_PATTERN", raising=False)
    monkeypatch.delenv("TRACE_FILE", raising=False)


# ── W2.3: the rename keeps the wire protocol ────────────────────────────────

@pytest.mark.parametrize("historical,explicit", [
    ("LOW", "LEGACY_LOW"), ("HIGH", "LEGACY_HIGH"), ("BURST", "LEGACY_BURST"),
])
def test_both_spellings_select_the_same_slice(historical, explicit):
    tasks = [mk(f"t{i}", creation=float(i)) for i in range(40)]
    assert tl.filter_scenario(tasks, historical) == tl.filter_scenario(tasks, explicit)


@pytest.mark.parametrize("label", ["low", "Low", "  HIGH  ", "legacy_burst"])
def test_labels_are_case_and_whitespace_insensitive(label):
    tasks = [mk(f"t{i}", creation=float(i)) for i in range(40)]
    tl.filter_scenario(tasks, label)      # must not raise


def test_legacy_low_still_takes_the_first_quarter():
    tasks = [mk(f"t{i}", creation=float(i)) for i in range(40)]
    got = tl.filter_scenario(tasks, "LOW")
    assert [t.name for t in got] == [f"t{i}" for i in range(10)]


def test_legacy_high_is_everything_schedulable():
    tasks = [mk("a"), mk("p", phase="Pending"), mk("f", phase="Failed")]
    assert {t.name for t in tl.filter_scenario(tasks, "HIGH")} == {"a", "f"}


# ── W2.2/W2.4: passthrough ──────────────────────────────────────────────────

def test_passthrough_keeps_every_schedulable_task():
    tasks = [mk(f"t{i}", creation=float(i)) for i in range(40)]
    assert tl.filter_scenario(tasks, tl.PASSTHROUGH) == tasks


def test_passthrough_still_drops_pending():
    """Same rule on both paths: the simulator never schedules a Pending pod."""
    tasks = [mk("a"), mk("p", phase="Pending")]
    assert [t.name for t in tl.filter_scenario(tasks, "NONE")] == ["a"]


def test_passthrough_and_legacy_high_agree_on_a_legacy_trace():
    """Mirrors ValidationRunner B21d, so the two languages cannot drift apart."""
    tasks = [mk(f"t{i}", creation=float(i)) for i in range(40)] + [mk("p", phase="Pending")]
    assert tl.filter_scenario(tasks, "NONE") == tl.filter_scenario(tasks, "HIGH")


def test_passthrough_does_not_reslice_a_wm1_trace():
    """The failure this exists to prevent: slicing a file that is already a scenario."""
    tasks = [mk(f"wm1-{i:06d}", creation=float(i)) for i in range(2031)]
    assert len(tl.filter_scenario(tasks, tl.PASSTHROUGH)) == 2031
    assert len(tl.filter_scenario(tasks, "LOW")) == 507      # what would have happened


# ── Rejections carry an actionable message ─────────────────────────────────

@pytest.mark.parametrize("label", ["OVERLOAD", "REPLAY"])
def test_wm1_only_scenarios_are_rejected_with_a_pointer(label):
    with pytest.raises(ValueError, match="only as a generated WM-1 trace"):
        tl.filter_scenario([mk("a")], label)


def test_unknown_scenario_lists_the_valid_ones():
    with pytest.raises(ValueError, match="unknown scenario"):
        tl.filter_scenario([mk("a")], "NOPE")


# ── W2.4: trace resolution mirrors the Java rules ──────────────────────────

def test_no_pattern_returns_the_default():
    assert tl.resolve_trace_path("HIGH", 42) == "/data/trace/openb_pod_list_default.csv"
    assert tl.uses_trace_pattern() is False


def test_trace_file_env_overrides_the_default(monkeypatch, tmp_path):
    monkeypatch.setenv("TRACE_FILE", "/somewhere/else.csv")
    assert tl.resolve_trace_path("HIGH", 42) == "/somewhere/else.csv"


def test_explicit_argument_wins_over_everything(monkeypatch):
    monkeypatch.setenv("TRACE_PATTERN", "/x/{scenario}/seed{seed}.csv")
    assert tl.resolve_trace_path("HIGH", 42, trace="/given.csv") == "/given.csv"


def test_pattern_substitutes_scenario_and_seed(tmp_path):
    d = tmp_path / "HIGH"
    d.mkdir()
    (d / "seed42.csv").write_text("x", encoding="utf-8")
    pattern = str(tmp_path / "{scenario}" / "seed{seed}.csv")
    assert tl.resolve_trace_path("HIGH", 42, pattern=pattern) == str(d / "seed42.csv")
    assert tl.uses_trace_pattern(pattern) is True


def test_pattern_without_scenario_placeholder_is_rejected():
    """Otherwise LOW, HIGH and BURST would be the same experiment, silently."""
    with pytest.raises(ValueError, match="no .*scenario.* placeholder"):
        tl.resolve_trace_path("HIGH", 42, pattern="/x/seed{seed}.csv")


def test_unresolvable_pattern_raises_instead_of_falling_back():
    """A silent fallback yields a complete, plausible, entirely wrong campaign."""
    with pytest.raises(ValueError, match="does not exist"):
        tl.resolve_trace_path("HIGH", 42, pattern="/nowhere/{scenario}/seed{seed}.csv")


# ── Agreement with the generated traces ────────────────────────────────────

@requires_trace
def test_resolution_finds_every_generated_trace():
    """Same 10 combinations ValidationRunner B22 and verify-trace-pattern.py check."""
    root = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                        "..", "data", "wm1", "homo")
    root = os.path.normpath(root)
    if not os.path.isdir(root):
        pytest.skip("data/wm1 not generated; run scripts/gen-workloads.sh")
    pattern = os.path.join(root, "{scenario}", "seed{seed}.csv")
    for scenario in ("LOW", "HIGH", "BURST", "OVERLOAD", "REPLAY"):
        for seed in (42, 46):
            p = tl.resolve_trace_path(scenario, seed, pattern=pattern)
            assert os.path.isfile(p)
            tasks = tl.filter_scenario(tl.read_trace(p), tl.PASSTHROUGH)
            assert tasks, f"{scenario}/seed{seed} loaded empty"
