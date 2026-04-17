"""Unit tests for state_builder.py (T4.1)."""

import sys
import os

import numpy as np
import pytest
from gymnasium import spaces

# Add src to path so we can import the module directly
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import state_builder


# ── Fixtures ───────────────────────────────────────────────────────────────

NUM_HOSTS = 10  # default datacenter has 10 hosts
OBS_DIM = 3 * NUM_HOSTS + state_builder.TASK_FEATURE_DIM  # 34


def _make_obs(num_hosts: int = NUM_HOSTS, fill: float = 0.5) -> list[float]:
    """Create a synthetic observation list (simulating Java double[])."""
    return [fill] * (3 * num_hosts + state_builder.TASK_FEATURE_DIM)


# ── observation_space ──────────────────────────────────────────────────────


class TestObservationSpace:
    def test_shape(self):
        space = state_builder.observation_space(NUM_HOSTS)
        assert space.shape == (OBS_DIM,)

    def test_dtype(self):
        space = state_builder.observation_space(NUM_HOSTS)
        assert space.dtype == np.float32

    def test_bounds(self):
        space = state_builder.observation_space(NUM_HOSTS)
        assert (space.low == 0.0).all()
        assert (space.high == 1.0).all()

    def test_is_box(self):
        space = state_builder.observation_space(NUM_HOSTS)
        assert isinstance(space, spaces.Box)

    def test_sample_in_bounds(self):
        space = state_builder.observation_space(NUM_HOSTS)
        sample = space.sample()
        assert space.contains(sample)

    def test_different_host_counts(self):
        for h in [1, 5, 20, 100]:
            space = state_builder.observation_space(h)
            assert space.shape == (3 * h + 4,)


# ── from_java ──────────────────────────────────────────────────────────────


class TestFromJava:
    def test_basic_conversion(self):
        raw = _make_obs(fill=0.3)
        obs = state_builder.from_java(raw, NUM_HOSTS)
        assert obs.dtype == np.float32
        assert obs.shape == (OBS_DIM,)
        np.testing.assert_allclose(obs, 0.3, atol=1e-6)

    def test_clips_above_one(self):
        raw = _make_obs()
        raw[0] = 1.5   # CPU util > 1 (can happen with time-shared scheduling)
        raw[5] = 2.0   # another out-of-range
        obs = state_builder.from_java(raw, NUM_HOSTS)
        assert obs[0] == pytest.approx(1.0)
        assert obs[5] == pytest.approx(1.0)

    def test_clips_below_zero(self):
        raw = _make_obs()
        raw[0] = -0.1
        obs = state_builder.from_java(raw, NUM_HOSTS)
        assert obs[0] == pytest.approx(0.0)

    def test_zero_observation(self):
        raw = _make_obs(fill=0.0)
        obs = state_builder.from_java(raw, NUM_HOSTS)
        np.testing.assert_array_equal(obs, 0.0)

    def test_full_observation(self):
        raw = _make_obs(fill=1.0)
        obs = state_builder.from_java(raw, NUM_HOSTS)
        np.testing.assert_array_equal(obs, 1.0)

    def test_length_mismatch_raises(self):
        raw = [0.5] * 10  # wrong length
        with pytest.raises(ValueError, match="mismatch"):
            state_builder.from_java(raw, NUM_HOSTS)

    def test_empty_array_raises(self):
        with pytest.raises(ValueError, match="mismatch"):
            state_builder.from_java([], NUM_HOSTS)

    def test_result_in_observation_space(self):
        """The converted obs must be contained in the gymnasium space."""
        raw = _make_obs(fill=0.7)
        obs = state_builder.from_java(raw, NUM_HOSTS)
        space = state_builder.observation_space(NUM_HOSTS)
        assert space.contains(obs)

    def test_accepts_python_list(self):
        """Py4J returns Java arrays as iterables — list should work."""
        raw = list(_make_obs())
        obs = state_builder.from_java(raw, NUM_HOSTS)
        assert obs.shape == (OBS_DIM,)

    def test_accepts_tuple(self):
        raw = tuple(_make_obs())
        obs = state_builder.from_java(raw, NUM_HOSTS)
        assert obs.shape == (OBS_DIM,)

    def test_single_host(self):
        raw = _make_obs(num_hosts=1, fill=0.5)
        obs = state_builder.from_java(raw, 1)
        assert obs.shape == (7,)  # 3*1 + 4


# ── parse ──────────────────────────────────────────────────────────────────


class TestParse:
    def test_decomposition(self):
        raw = np.zeros(OBS_DIM, dtype=np.float32)
        # Set distinct values per section
        raw[0:NUM_HOSTS] = 0.1          # CPU
        raw[NUM_HOSTS:2*NUM_HOSTS] = 0.2  # MEM
        raw[2*NUM_HOSTS:3*NUM_HOSTS] = 0.3  # GPU
        raw[3*NUM_HOSTS] = 0.4          # task CPU
        raw[3*NUM_HOSTS+1] = 0.5        # task MEM
        raw[3*NUM_HOSTS+2] = 0.6        # task GPU
        raw[3*NUM_HOSTS+3] = 0.7        # task QoS

        p = state_builder.parse(raw, NUM_HOSTS)

        assert p.num_hosts == NUM_HOSTS
        np.testing.assert_allclose(p.host_cpu_util, 0.1, atol=1e-6)
        np.testing.assert_allclose(p.host_mem_util, 0.2, atol=1e-6)
        np.testing.assert_allclose(p.host_gpu_util, 0.3, atol=1e-6)
        assert p.task_cpu_norm == pytest.approx(0.4)
        assert p.task_mem_norm == pytest.approx(0.5)
        assert p.task_gpu_norm == pytest.approx(0.6)
        assert p.task_qos_norm == pytest.approx(0.7)

    def test_mean_utilities(self):
        raw = np.zeros(OBS_DIM, dtype=np.float32)
        raw[0:NUM_HOSTS] = np.linspace(0.0, 1.0, NUM_HOSTS)
        raw[NUM_HOSTS:2*NUM_HOSTS] = 0.5
        raw[2*NUM_HOSTS:3*NUM_HOSTS] = 0.25

        p = state_builder.parse(raw, NUM_HOSTS)

        assert p.mean_cpu_util() == pytest.approx(0.5, abs=0.05)
        assert p.mean_mem_util() == pytest.approx(0.5)
        assert p.mean_gpu_util() == pytest.approx(0.25)

    def test_parse_is_copy(self):
        """Modifying parsed arrays must not affect the original."""
        raw = np.ones(OBS_DIM, dtype=np.float32)
        p = state_builder.parse(raw, NUM_HOSTS)
        p.host_cpu_util[0] = 999.0
        assert raw[0] == 1.0

    def test_roundtrip_from_java_then_parse(self):
        """Integration: from_java → parse should decompose correctly."""
        raw = [0.0] * OBS_DIM
        raw[0] = 0.8  # host 0 CPU
        raw[3 * NUM_HOSTS + 3] = 0.9  # task QoS

        obs = state_builder.from_java(raw, NUM_HOSTS)
        p = state_builder.parse(obs, NUM_HOSTS)

        assert p.host_cpu_util[0] == pytest.approx(0.8)
        assert p.task_qos_norm == pytest.approx(0.9)
