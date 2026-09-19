"""W1.6 — WM-1 orchestrator: generation, determinism, manifest, REPLAY.

Acceptance targets from PLAN-Workload-Model.md W1.6:
    every generated file parses back through the openb adapter
    the same seed regenerates a byte-identical file
    the manifest carries rho, idc target, p1, N, T, truncation, seed, git sha, source sha
"""

from __future__ import annotations

import json
import os

import pytest

from workload import adapters, schema, wm1
from workload.calibrate import ARMS, measured_load
from workload.wm1 import GenerationError

from _workload_fixtures import SCENARIOS, openb_tasks  # noqa: F401


# ── Guard rails ─────────────────────────────────────────────────────────────

def test_unknown_arm_and_scenario_are_refused(openb_tasks):
    with pytest.raises(GenerationError, match="unknown arm"):
        wm1.generate_trace(openb_tasks, "nope", "HIGH", 42)
    with pytest.raises(GenerationError, match="unknown scenario"):
        wm1.generate_trace(openb_tasks, "homo", "NOPE", 42)


# ── Generated traces ────────────────────────────────────────────────────────

@pytest.mark.parametrize("arm,scenario", sorted(SCENARIOS))
def test_generated_trace_matches_the_design_reference(arm, scenario, openb_tasks):
    rho, idc_target, want_p1, want_n, _ = SCENARIOS[(arm, scenario)]
    tasks, spec = wm1.generate_trace(openb_tasks, arm, scenario, 42)

    assert len(tasks) == want_n == spec.n_task
    assert spec.duty_cycle == pytest.approx(want_p1, abs=0.03)
    assert spec.rho_target == rho
    assert spec.idc_target == idc_target


@pytest.mark.parametrize("arm,scenario", sorted(SCENARIOS))
def test_realised_load_lands_within_two_percent_of_target(arm, scenario, openb_tasks):
    """Acceptance: the load measured back off the emitted tasks, not the plan."""
    rho, _, _, _, _ = SCENARIOS[(arm, scenario)]
    capacity, horizon = ARMS[arm]
    tasks, spec = wm1.generate_trace(openb_tasks, arm, scenario, 42)
    cpu, gpu = measured_load(tasks, capacity, horizon)
    assert max(cpu, gpu) == pytest.approx(rho, rel=0.02)
    assert spec.bottleneck == ("gpu" if gpu > cpu else "cpu")


@pytest.mark.parametrize("arm", ["homo", "hetero"])
def test_generated_traces_are_well_formed(arm, openb_tasks):
    tasks, _ = wm1.generate_trace(openb_tasks, arm, "HIGH", 42)
    schema.validate_tasks(tasks)                 # sorted, unique names, sane times
    _, horizon = ARMS[arm]
    assert all(0.0 <= t.creation_time < horizon for t in tasks)
    assert all(t.scheduled_time == t.creation_time for t in tasks)
    # Durations are stored as (creation, deletion) so recomputing them costs a ULP or
    # two at these magnitudes; the truncation bound holds to within that.
    assert all(t.duration <= horizon * (1 + 1e-9) for t in tasks)


def test_burst_and_high_differ_only_in_burstiness(openb_tasks):
    """The controlled comparison the whole redesign is for.

    Split RNG streams make this a *paired* contrast: same seed, same rho target, so the
    job multiset is identical and only the arrival times move.
    """
    high, hspec = wm1.generate_trace(openb_tasks, "homo", "HIGH", 42)
    burst, bspec = wm1.generate_trace(openb_tasks, "homo", "BURST", 42)
    assert hspec.n_task == bspec.n_task
    assert hspec.horizon_sec == bspec.horizon_sec
    assert hspec.rho_target == bspec.rho_target
    assert bspec.duty_cycle < hspec.duty_cycle       # burstier = rarer, denser bursts
    assert len(high) == len(burst)

    # identical jobs, different arrival times
    def demand(ts):
        return sorted((t.cpu_milli, t.memory_mib, t.num_gpu, round(t.duration, 6))
                      for t in ts)
    assert demand(high) == demand(burst)
    assert [t.creation_time for t in high] != [t.creation_time for t in burst]
    assert hspec.rho_cpu == pytest.approx(bspec.rho_cpu, rel=1e-9)


def test_low_high_overload_separate_on_load(openb_tasks):
    """PLAN §2.4 check 4: every pair must be at least 30 % apart in offered load."""
    capacity, horizon = ARMS["homo"]
    rhos = {}
    for scenario in ("LOW", "HIGH", "OVERLOAD"):
        tasks, _ = wm1.generate_trace(openb_tasks, "homo", scenario, 42)
        cpu, gpu = measured_load(tasks, capacity, horizon)
        rhos[scenario] = max(cpu, gpu)
    names = list(rhos)
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            sep = abs(rhos[a] - rhos[b]) / max(rhos[a], rhos[b])
            assert sep >= 0.30, f"{a} vs {b} only {sep:.1%} apart: {rhos}"


# ── Determinism ─────────────────────────────────────────────────────────────

def test_same_seed_regenerates_identical_tasks(openb_tasks):
    a, _ = wm1.generate_trace(openb_tasks, "homo", "BURST", 42)
    b, _ = wm1.generate_trace(openb_tasks, "homo", "BURST", 42)
    assert a == b


def test_different_seeds_differ(openb_tasks):
    a, _ = wm1.generate_trace(openb_tasks, "homo", "BURST", 42)
    b, _ = wm1.generate_trace(openb_tasks, "homo", "BURST", 43)
    assert a != b
    assert len(a) == len(b)          # ...but the arrival count is pinned (fixed-N)


def test_files_are_byte_identical_across_runs(tmp_path, openb_tasks):
    tasks, _ = wm1.generate_trace(openb_tasks, "homo", "LOW", 42)
    a, b = str(tmp_path / "a.csv"), str(tmp_path / "b.csv")
    schema.write_csv(a, tasks)
    schema.write_csv(b, tasks)
    assert open(a, "rb").read() == open(b, "rb").read()


def test_written_trace_round_trips_through_the_openb_adapter(tmp_path, openb_tasks):
    """The generated file must be readable by the same path the simulator uses."""
    tasks, _ = wm1.generate_trace(openb_tasks, "homo", "LOW", 42)
    p = str(tmp_path / "wm1.csv")
    schema.write_csv(p, tasks)
    assert adapters.load("openb", p) == tasks


# ── REPLAY ──────────────────────────────────────────────────────────────────

def test_replay_takes_a_real_window_rebased_to_zero(openb_tasks):
    tasks, spec = wm1.generate_trace(openb_tasks, "homo", "REPLAY", 42)
    _, horizon = ARMS["homo"]
    assert spec.rho_target is None and spec.duty_cycle is None
    assert tasks
    assert tasks[0].creation_time == 0.0
    assert tasks[-1].creation_time < horizon
    schema.validate_tasks(tasks)


def test_replay_is_busy_on_both_axes_not_just_work(openb_tasks):
    """The window must be a real busy period, not one whale.

    Maximising work alone picks a 5-task window at rho = 2.44; matching the HIGH load
    target alone picks a 7-task window. Both would replay a degenerate instance.
    """
    capacity, horizon = ARMS["homo"]
    tasks, spec = wm1.generate_trace(openb_tasks, "homo", "REPLAY", 42)

    assert len(tasks) > 500, f"REPLAY window is degenerate: {len(tasks)} tasks"
    cpu, gpu = measured_load(tasks, capacity, horizon)
    # Lower bound relaxed in W3.1: once pods no host can run are excluded, the busiest
    # placeable 4-day window sits at 0.232, not 0.454. The old figure counted load the
    # cluster could never have served.
    assert 0.15 < max(cpu, gpu) < 2.0, f"REPLAY load is implausible: {cpu}/{gpu}"
    assert spec.n_task == len(tasks)


def test_replay_is_lighter_than_the_modelled_high_load(openb_tasks):
    """Pin the finding: no real 4-day window of openb is as loaded as HIGH.

    The busiest realistic window sits near rho = 0.23 against HIGH's 0.85, so HIGH is a
    stress level this trace does not naturally exhibit. Legitimate for a stress scenario,
    but it belongs in the report rather than being quietly assumed away.
    """
    capacity, horizon = ARMS["homo"]
    replay, _ = wm1.generate_trace(openb_tasks, "homo", "REPLAY", 42)
    high, _ = wm1.generate_trace(openb_tasks, "homo", "HIGH", 42)
    assert max(measured_load(replay, capacity, horizon)) < 0.7 * max(
        measured_load(high, capacity, horizon))


def test_replay_excludes_failed_pods_like_the_generated_scenarios(openb_tasks):
    """Otherwise REPLAY and the model would be measured on different populations."""
    tasks, _ = wm1.generate_trace(openb_tasks, "homo", "REPLAY", 42)
    assert all(t.pod_phase not in ("Pending", "Failed") for t in tasks)


def test_replay_truncates_durations_at_the_horizon(openb_tasks):
    tasks, _ = wm1.generate_trace(openb_tasks, "homo", "REPLAY", 42)
    _, horizon = ARMS["homo"]
    assert all(t.duration <= horizon * (1 + 1e-9) for t in tasks)


def test_replay_is_seed_independent(openb_tasks):
    a, _ = wm1.generate_trace(openb_tasks, "homo", "REPLAY", 42)
    b, _ = wm1.generate_trace(openb_tasks, "homo", "REPLAY", 46)
    assert a == b


def test_replay_on_an_empty_trace_raises():
    with pytest.raises(GenerationError, match="non-empty"):
        wm1._busiest_window([], 100.0, ARMS["homo"][0])


# ── Batch generation + manifest ─────────────────────────────────────────────

def test_generate_arm_writes_the_matrix_and_a_complete_manifest(tmp_path):
    from _workload_fixtures import TRACE_PATH
    if TRACE_PATH is None:
        pytest.skip("trace not available")
    out = str(tmp_path / "wm1")
    manifest = wm1.generate_arm(TRACE_PATH, "homo", out,
                                scenarios=("LOW", "BURST", "REPLAY"), seeds=(42, 43),
                                verbose=False)

    assert len(manifest["traces"]) == 6
    for entry in manifest["traces"]:
        # Paths are relative to the manifest's own directory (<out>/<arm>), so a reader
        # never has to know how deep the manifest sits.
        p = os.path.join(out, "homo", entry["path"])
        assert os.path.isfile(p), p
        assert not entry["path"].startswith("homo/"), (
            "path must not repeat the arm — that is the double-<arm> bug B22 caught")
        assert entry["sha256"]
        for key in ("arm", "scenario", "seed", "horizon_sec", "n_task",
                    "rho_cpu", "rho_gpu", "bottleneck", "truncated_fraction",
                    "source_sha256"):
            assert key in entry

    on_disk = json.load(open(os.path.join(out, "homo", "wm1-manifest.json"),
                             encoding="utf-8"))
    assert on_disk["arm"] == "homo"
    assert on_disk["capacity"]["total_gpus"] == 80
    assert on_disk["arrival_model"]["mean_on_sec"] == 3600.0
    assert on_disk["source"]["sha256"] == manifest["traces"][0]["source_sha256"]


def test_paths_keep_the_arms_apart():
    """Risk R6: homo and hetero writing to the same path is exactly the bug that hit
    ppo-min LOW before (Tracking C16)."""
    a = wm1.trace_path("/data/wm1", "homo", "HIGH", 42)
    b = wm1.trace_path("/data/wm1", "hetero", "HIGH", 42)
    assert a != b
    assert a.endswith(os.path.join("homo", "HIGH", "seed42.csv"))


def test_regenerating_an_arm_is_reproducible(tmp_path):
    from _workload_fixtures import TRACE_PATH
    if TRACE_PATH is None:
        pytest.skip("trace not available")
    a = str(tmp_path / "a")
    b = str(tmp_path / "b")
    for out in (a, b):
        wm1.generate_arm(TRACE_PATH, "homo", out, scenarios=("LOW",), seeds=(42,),
                         verbose=False)
    pa = wm1.trace_path(a, "homo", "LOW", 42)
    pb = wm1.trace_path(b, "homo", "LOW", 42)
    assert open(pa, "rb").read() == open(pb, "rb").read()
