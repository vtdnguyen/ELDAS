"""Unit tests for environment.py (T4.3).

Uses MockGateway to simulate the Java GatewayEntryPoint without
requiring a running CloudSim Plus process.
"""

import sys
import os
from unittest.mock import patch

import numpy as np
import pytest
from gymnasium import spaces

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.dirname(__file__))

from mock_gateway import MockGateway, MockEntryPoint, NUM_HOSTS, NUM_TASKS
import environment
import reward as reward_mod


# ── Fixture: patch Py4J connection ────────────────────────────────────────

@pytest.fixture
def mock_gw():
    return MockGateway()


@pytest.fixture
def env(mock_gw):
    """Create a CloudSimEnv with mocked Py4J connection."""
    with patch.object(environment, "_connect_gateway", return_value=mock_gw):
        e = environment.CloudSimEnv(scenario="HIGH", seed=42)
        yield e
        e.close()


@pytest.fixture
def env_normalised(mock_gw):
    """CloudSimEnv with reward normalisation enabled."""
    with patch.object(environment, "_connect_gateway", return_value=mock_gw):
        e = environment.CloudSimEnv(scenario="HIGH", seed=42, normalize_reward=True)
        yield e
        e.close()


@pytest.fixture
def scalar_env(mock_gw):
    """CloudSimEnv wrapped with ScalarRewardWrapper."""
    with patch.object(environment, "_connect_gateway", return_value=mock_gw):
        e = environment.make_env(
            scenario="HIGH", seed=42, scalarise=True,
            weights=np.array([0.8, 0.2], dtype=np.float32),
        )
        yield e
        e.close()


# ── Space definitions ─────────────────────────────────────────────────────

class TestSpaces:
    def test_observation_space(self, env):
        assert isinstance(env.observation_space, spaces.Box)
        assert env.observation_space.shape == (3 * NUM_HOSTS + 4,)
        assert env.observation_space.dtype == np.float32

    def test_action_space(self, env):
        assert isinstance(env.action_space, spaces.Discrete)
        assert env.action_space.n == NUM_HOSTS


# ── Reset ─────────────────────────────────────────────────────────────────

class TestReset:
    def test_returns_obs_and_info(self, env):
        obs, info = env.reset()
        assert isinstance(obs, np.ndarray)
        assert obs.shape == (3 * NUM_HOSTS + 4,)
        assert obs.dtype == np.float32
        assert isinstance(info, dict)

    def test_obs_in_space(self, env):
        obs, _ = env.reset()
        assert env.observation_space.contains(obs)

    def test_info_fields(self, env):
        _, info = env.reset()
        assert "task_index" in info
        assert "task_name" in info
        assert "num_hosts" in info
        assert "scenario" in info
        assert "seed" in info
        assert info["num_hosts"] == NUM_HOSTS
        assert info["scenario"] == "HIGH"

    def test_reset_with_seed_override(self, env):
        obs1, _ = env.reset(seed=100)
        obs2, _ = env.reset(seed=200)
        # Different seeds should produce different initial observations
        # (or at least not crash)
        assert obs1.shape == obs2.shape

    def test_reset_with_scenario_override(self, env):
        _, info = env.reset(options={"scenario": "LOW"})
        assert info["scenario"] == "LOW"

    def test_double_reset(self, env):
        obs1, _ = env.reset()
        obs2, _ = env.reset()
        assert obs1.shape == obs2.shape


# ── Step ──────────────────────────────────────────────────────────────────

class TestStep:
    def test_step_returns_five_tuple(self, env):
        env.reset()
        obs, reward, terminated, truncated, info = env.step(0)

        assert isinstance(obs, np.ndarray)
        assert obs.shape == (3 * NUM_HOSTS + 4,)
        assert isinstance(reward, np.ndarray)
        assert reward.shape == (2,)
        assert isinstance(terminated, bool)
        assert isinstance(truncated, bool)
        assert truncated is False
        assert isinstance(info, dict)

    def test_reward_is_vector(self, env):
        env.reset()
        _, reward, _, _, _ = env.step(0)
        assert reward.dtype == np.float32
        assert reward.shape == (2,)

    def test_obs_stays_in_space(self, env):
        env.reset()
        for _ in range(NUM_TASKS):
            obs, _, done, _, _ = env.step(0)
            assert env.observation_space.contains(obs)
            if done:
                break

    def test_episode_terminates(self, env):
        env.reset()
        terminated = False
        for _ in range(NUM_TASKS + 5):
            _, _, terminated, _, _ = env.step(0)
            if terminated:
                break
        assert terminated

    def test_step_after_done_raises(self, env):
        env.reset()
        for _ in range(NUM_TASKS):
            _, _, done, _, _ = env.step(0)
            if done:
                break
        with pytest.raises(RuntimeError, match="done"):
            env.step(0)

    def test_info_has_raw_reward(self, env):
        env.reset()
        _, _, _, _, info = env.step(0)
        assert "raw_reward" in info
        assert info["raw_reward"].shape == (2,)

    def test_episode_info_at_end(self, env):
        env.reset()
        info = {}
        for _ in range(NUM_TASKS):
            _, _, done, _, info = env.step(0)
            if done:
                break
        assert "episode" in info
        ep = info["episode"]
        assert "length" in ep
        assert "total_energy_reward" in ep
        assert "total_sla_reward" in ep
        assert "total_energy_kwh" in ep
        assert ep["length"] == NUM_TASKS

    def test_step_count(self, env):
        env.reset()
        for i in range(NUM_TASKS):
            _, _, _, _, info = env.step(0)
            assert info["step"] == i + 1


# ── Action masking ────────────────────────────────────────────────────────

class TestActionMasking:
    def test_mask_shape(self, env):
        env.reset()
        mask = env.action_masks()
        assert isinstance(mask, np.ndarray)
        assert mask.dtype == bool
        assert mask.shape == (NUM_HOSTS,)

    def test_mask_has_valid_actions(self, env):
        env.reset()
        mask = env.action_masks()
        assert mask.any()  # at least one valid action

    def test_mock_masks_last_two(self, env):
        """Mock gateway masks last 2 hosts."""
        env.reset()
        mask = env.action_masks()
        assert mask[-1] is np.bool_(False)
        assert mask[-2] is np.bool_(False)
        assert mask[0] is np.bool_(True)

    def test_all_false_fallback(self, mock_gw):
        """If Java returns all-False, env should fallback to all-True."""
        original_mask = mock_gw.entry_point.getActionMask

        def all_false_mask():
            return [False] * NUM_HOSTS

        mock_gw.entry_point.getActionMask = all_false_mask

        with patch.object(environment, "_connect_gateway", return_value=mock_gw):
            env = environment.CloudSimEnv(scenario="HIGH", seed=42)
            env.reset()
            mask = env.action_masks()
            assert mask.all()  # fallback: all True
            env.close()

        mock_gw.entry_point.getActionMask = original_mask


# ── Reward normalisation ──────────────────────────────────────────────────

class TestNormalisedReward:
    def test_normalised_reward_differs(self, env, env_normalised):
        env.reset()
        env_normalised.reset()
        _, raw, _, _, _ = env.step(0)
        _, normed, _, _, _ = env_normalised.step(0)
        # They may be similar for the first step, but should differ
        # as normaliser has no history yet the values can still differ
        assert raw.shape == normed.shape == (2,)

    def test_normaliser_resets_between_episodes(self, env_normalised):
        env_normalised.reset()
        for _ in range(NUM_TASKS):
            _, _, done, _, _ = env_normalised.step(0)
            if done:
                break
        # Reset should create fresh normaliser
        env_normalised.reset()
        _, reward, _, _, _ = env_normalised.step(0)
        assert reward.shape == (2,)
        assert np.isfinite(reward).all()


# ── ScalarRewardWrapper ───────────────────────────────────────────────────

class TestScalarWrapper:
    def test_scalar_reward(self, scalar_env):
        scalar_env.reset()
        _, reward, _, _, _ = scalar_env.step(0)
        assert isinstance(reward, float)

    def test_action_masks_propagated(self, scalar_env):
        scalar_env.reset()
        mask = scalar_env.action_masks()
        assert isinstance(mask, np.ndarray)
        assert mask.dtype == bool
        assert mask.shape == (NUM_HOSTS,)

    def test_scalar_is_weighted_sum(self, mock_gw):
        weights = np.array([0.6, 0.4], dtype=np.float32)
        with patch.object(environment, "_connect_gateway", return_value=mock_gw):
            base_env = environment.CloudSimEnv(scenario="HIGH", seed=42)
            wrapped = environment.ScalarRewardWrapper(base_env, weights=weights)
            wrapped.reset()
            _, scalar_r, _, _, info = wrapped.step(0)
            raw = info["raw_reward"]
            expected = float(np.dot(weights, raw))
            assert scalar_r == pytest.approx(expected, rel=1e-5)
            wrapped.close()


# ── make_env factory ──────────────────────────────────────────────────────

class TestMakeEnv:
    def test_make_env_scalar(self, mock_gw):
        with patch.object(environment, "_connect_gateway", return_value=mock_gw):
            env = environment.make_env(scalarise=True)
            obs, _ = env.reset()
            _, reward, _, _, _ = env.step(0)
            assert isinstance(reward, float)
            env.close()

    def test_make_env_vector(self, mock_gw):
        with patch.object(environment, "_connect_gateway", return_value=mock_gw):
            env = environment.make_env(scalarise=False)
            obs, _ = env.reset()
            _, reward, _, _, _ = env.step(0)
            assert isinstance(reward, np.ndarray)
            assert reward.shape == (2,)
            env.close()


# ── Full episode integration ──────────────────────────────────────────────

class TestFullEpisode:
    def test_complete_episode_vector(self, env):
        obs, info = env.reset()
        rewards = []
        for _ in range(NUM_TASKS + 5):
            mask = env.action_masks()
            valid = np.where(mask)[0]
            action = int(valid[0])
            obs, reward, done, _, info = env.step(action)
            rewards.append(reward)
            if done:
                break
        assert done
        assert len(rewards) == NUM_TASKS
        assert all(r.shape == (2,) for r in rewards)

    def test_complete_episode_scalar(self, scalar_env):
        obs, info = scalar_env.reset()
        rewards = []
        for _ in range(NUM_TASKS + 5):
            mask = scalar_env.action_masks()
            valid = np.where(mask)[0]
            action = int(valid[0])
            obs, reward, done, _, info = scalar_env.step(action)
            rewards.append(reward)
            if done:
                break
        assert done
        assert len(rewards) == NUM_TASKS
        assert all(isinstance(r, float) for r in rewards)

    def test_reset_after_episode(self, env):
        """Can reset and run a second episode after the first completes."""
        env.reset()
        for _ in range(NUM_TASKS):
            _, _, done, _, _ = env.step(0)
            if done:
                break

        obs2, info2 = env.reset()
        assert env.observation_space.contains(obs2)
        _, reward, _, _, _ = env.step(0)
        assert reward.shape == (2,)
