"""Unit tests for cmdp.py (G1.3 — Constrained-MDP reward shaping).

Uses a self-contained stub env (no Java/Py4J) so the Lagrangian shaping logic
is verified deterministically:
  - effective reward = R_energy − λ·C_SLA;
  - λ is read LIVE from the shared controller (dual on a slower timescale);
  - R_energy and C_SLA are normalised SEPARATELY (Lưu ý #1);
  - the episodic constraint statistic J is exposed on the terminal step.
"""

import os
import sys

import gymnasium as gym
import numpy as np
import pytest
from gymnasium import spaces

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from cmdp import CMDPRewardWrapper


# ── Stubs ───────────────────────────────────────────────────────────────────


class _StubEnv(gym.Env):
    """Fixed-length episode emitting prescribed (R_energy, C_SLA) per step."""

    def __init__(self, energies, costs, emit_cost_info=True):
        self.observation_space = spaces.Box(0.0, 1.0, (2,), np.float32)
        self.action_space = spaces.Discrete(2)
        self._energies = list(energies)
        self._costs = list(costs)
        self._emit = emit_cost_info
        self._t = 0

    def reset(self, *, seed=None, options=None):
        self._t = 0
        return np.zeros(2, np.float32), {}

    def step(self, action):
        e = float(self._energies[self._t])
        c = float(self._costs[self._t])
        self._t += 1
        done = self._t >= len(self._energies)
        reward_vec = np.array([e, -c], dtype=np.float32)  # R_sla = −C_SLA
        info = {"raw_reward": reward_vec}
        if self._emit:
            info["cost"] = c
        return np.zeros(2, np.float32), reward_vec, done, False, info

    def action_masks(self):
        return np.array([True, True])


class _StubPID:
    """Minimal dual controller exposing a mutable ``lambda_``."""

    def __init__(self, lam=0.0):
        self.lambda_ = float(lam)


# ── Effective reward (no normalisation) ─────────────────────────────────────


class TestEffectiveReward:
    def test_lagrangian_combination(self):
        pid = _StubPID(lam=2.0)
        env = CMDPRewardWrapper(
            _StubEnv(energies=[-10.0, -20.0], costs=[1.0, 3.0]),
            pid=pid, normalize=False,
        )
        env.reset()
        _, r0, _, _, info0 = env.step(0)
        assert r0 == pytest.approx(-10.0 - 2.0 * 1.0)   # −12
        assert info0["reward_energy"] == pytest.approx(-10.0)
        assert info0["cost"] == pytest.approx(1.0)
        assert info0["lambda"] == pytest.approx(2.0)
        assert info0["effective_reward"] == pytest.approx(-12.0)

        _, r1, _, _, _ = env.step(0)
        assert r1 == pytest.approx(-20.0 - 2.0 * 3.0)   # −26

    def test_lambda_zero_is_pure_energy(self):
        pid = _StubPID(lam=0.0)
        env = CMDPRewardWrapper(
            _StubEnv(energies=[-5.0, -7.0], costs=[4.0, 9.0]),
            pid=pid, normalize=False,
        )
        env.reset()
        _, r, _, _, _ = env.step(0)
        assert r == pytest.approx(-5.0)  # cost ignored when λ=0

    def test_lambda_is_read_live(self):
        """Raising λ between steps must change the effective penalty."""
        pid = _StubPID(lam=0.0)
        env = CMDPRewardWrapper(
            _StubEnv(energies=[-1.0, -1.0], costs=[10.0, 10.0]),
            pid=pid, normalize=False,
        )
        env.reset()
        _, r0, _, _, _ = env.step(0)   # λ=0 → −1
        pid.lambda_ = 0.5              # dual update happens between steps
        _, r1, _, _, _ = env.step(0)   # λ=0.5 → −1 − 0.5·10 = −6
        assert r0 == pytest.approx(-1.0)
        assert r1 == pytest.approx(-6.0)


# ── Cost sourcing ───────────────────────────────────────────────────────────


class TestCostSource:
    def test_uses_info_cost(self):
        env = CMDPRewardWrapper(
            _StubEnv(energies=[-2.0], costs=[7.0], emit_cost_info=True),
            pid=_StubPID(1.0), normalize=False,
        )
        env.reset()
        _, _, _, _, info = env.step(0)
        assert info["cost"] == pytest.approx(7.0)

    def test_falls_back_to_neg_rsla(self):
        """When the env omits info['cost'], use −R_sla (= magnitude ≥ 0)."""
        env = CMDPRewardWrapper(
            _StubEnv(energies=[-2.0], costs=[7.0], emit_cost_info=False),
            pid=_StubPID(1.0), normalize=False,
        )
        env.reset()
        _, r, _, _, info = env.step(0)
        assert info["cost"] == pytest.approx(7.0)
        assert r == pytest.approx(-2.0 - 1.0 * 7.0)


# ── Episodic constraint statistic ───────────────────────────────────────────


class TestEpisodeStatistic:
    def test_episode_cost_is_mean_no_norm(self):
        costs = [2.0, 4.0, 6.0]
        env = CMDPRewardWrapper(
            _StubEnv(energies=[-1.0] * 3, costs=costs),
            pid=_StubPID(0.0), normalize=False,
        )
        env.reset()
        info = {}
        for _ in range(3):
            _, _, term, _, info = env.step(0)
        assert term is True
        assert info["episode_cost"] == pytest.approx(np.mean(costs))      # 4.0
        assert info["episode_cost_raw"] == pytest.approx(sum(costs))      # 12.0
        assert env.last_episode_cost == pytest.approx(np.mean(costs))
        assert env.last_episode_cost_raw == pytest.approx(sum(costs))

    def test_no_episode_cost_before_done(self):
        env = CMDPRewardWrapper(
            _StubEnv(energies=[-1.0, -1.0], costs=[2.0, 2.0]),
            pid=_StubPID(0.0), normalize=False,
        )
        env.reset()
        _, _, term, _, info = env.step(0)
        assert term is False
        assert "episode_cost" not in info

    def test_reset_clears_accumulators(self):
        env = CMDPRewardWrapper(
            _StubEnv(energies=[-1.0, -1.0], costs=[5.0, 5.0]),
            pid=_StubPID(0.0), normalize=False,
        )
        env.reset()
        env.step(0)
        env.reset()
        _, _, _, _, info = env.step(0)  # first step of new episode, not terminal
        assert "episode_cost" not in info
        assert env._ep_len == 1


# ── Separate normalisation (Lưu ý #1) ───────────────────────────────────────


class TestSeparateNormalization:
    def test_cost_normalised_nonnegative(self):
        """Cost uses std-only scaling → normalised cost stays ≥ 0."""
        rng = np.random.default_rng(0)
        costs = list(np.abs(rng.normal(50.0, 10.0, size=200)))
        env = CMDPRewardWrapper(
            _StubEnv(energies=[-100.0] * 200, costs=costs),
            pid=_StubPID(1.0), normalize=True,
        )
        env.reset()
        seen = []
        for _ in range(200):
            _, _, term, _, info = env.step(0)
            seen.append(info["cost_norm"])
            if term:
                break
        # All positive costs → all normalised costs ≥ 0 (no centering).
        assert min(seen) >= 0.0
        # And scaled to O(1), not the raw ~50 magnitude.
        assert np.mean(seen) < 10.0

    def test_energy_centered_around_zero(self):
        """Energy uses full standardisation → varied energy centres near 0."""
        rng = np.random.default_rng(1)
        energies = list(rng.normal(-700.0, 50.0, size=300))
        env = CMDPRewardWrapper(
            _StubEnv(energies=energies, costs=[1.0] * 300),
            pid=_StubPID(0.0), normalize=True,
        )
        env.reset()
        seen = []
        for _ in range(300):
            _, _, term, _, info = env.step(0)
            seen.append(info["reward_energy_norm"])
            if term:
                break
        # Centered standardisation → running mean of normalised energy ≈ 0.
        assert abs(np.mean(seen)) < 0.5

    def test_independent_scales(self):
        """Energy and cost normalisers are independent objects/scales."""
        env = CMDPRewardWrapper(
            _StubEnv(energies=[-1000.0, -1000.0], costs=[1.0, 1.0]),
            pid=_StubPID(0.0), normalize=True,
        )
        assert env._energy_norm is not env._cost_norm
        assert env._cost_norm._center is False
        assert env._energy_norm._center is True


# ── action_masks passthrough ────────────────────────────────────────────────


def test_action_masks_passthrough():
    env = CMDPRewardWrapper(
        _StubEnv(energies=[-1.0], costs=[1.0]),
        pid=_StubPID(0.0), normalize=False,
    )
    np.testing.assert_array_equal(env.action_masks(), [True, True])
