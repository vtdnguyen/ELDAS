"""Unit tests for reward.py (T4.2)."""

import sys
import os

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import reward


# ── from_java ──────────────────────────────────────────────────────────────


class TestFromJava:
    def test_basic_conversion(self):
        raw = [-100.0, -0.5]
        vec = reward.from_java(raw)
        assert vec.dtype == np.float32
        assert vec.shape == (2,)
        assert vec[0] == pytest.approx(-100.0)
        assert vec[1] == pytest.approx(-0.5)

    def test_zero_reward(self):
        vec = reward.from_java([0.0, 0.0])
        np.testing.assert_array_equal(vec, [0.0, 0.0])

    def test_wrong_length_raises(self):
        with pytest.raises(ValueError, match="mismatch"):
            reward.from_java([1.0, 2.0, 3.0])

    def test_empty_raises(self):
        with pytest.raises(ValueError, match="mismatch"):
            reward.from_java([])

    def test_accepts_tuple(self):
        vec = reward.from_java((-10.0, -1.0))
        assert vec.shape == (2,)

    def test_negative_values_preserved(self):
        """Rewards are penalties (≤ 0) — they must not be clipped."""
        vec = reward.from_java([-5000.0, -999.0])
        assert vec[0] == pytest.approx(-5000.0, rel=1e-4)
        assert vec[1] == pytest.approx(-999.0, rel=1e-4)


# ── load_weights ───────────────────────────────────────────────────────────


class TestLoadWeights:
    def test_default_weights(self, monkeypatch):
        monkeypatch.delenv("ENERGY_WEIGHT", raising=False)
        monkeypatch.delenv("SLA_WEIGHT", raising=False)
        w = reward.load_weights()
        assert w.shape == (2,)
        assert w[0] == pytest.approx(0.8)
        assert w[1] == pytest.approx(0.2)

    def test_custom_weights(self, monkeypatch):
        monkeypatch.setenv("ENERGY_WEIGHT", "0.3")
        monkeypatch.setenv("SLA_WEIGHT", "0.7")
        w = reward.load_weights()
        assert w[0] == pytest.approx(0.3)
        assert w[1] == pytest.approx(0.7)

    def test_weights_normalised(self, monkeypatch):
        monkeypatch.setenv("ENERGY_WEIGHT", "3")
        monkeypatch.setenv("SLA_WEIGHT", "7")
        w = reward.load_weights()
        assert w.sum() == pytest.approx(1.0)
        assert w[0] == pytest.approx(0.3)
        assert w[1] == pytest.approx(0.7)

    def test_zero_weights_fallback(self, monkeypatch):
        monkeypatch.setenv("ENERGY_WEIGHT", "0")
        monkeypatch.setenv("SLA_WEIGHT", "0")
        w = reward.load_weights()
        np.testing.assert_allclose(w, [0.5, 0.5])

    def test_equal_weights(self, monkeypatch):
        monkeypatch.setenv("ENERGY_WEIGHT", "0.5")
        monkeypatch.setenv("SLA_WEIGHT", "0.5")
        w = reward.load_weights()
        assert w[0] == pytest.approx(0.5)
        assert w[1] == pytest.approx(0.5)


# ── scalarise ──────────────────────────────────────────────────────────────


class TestScalarise:
    def test_basic(self):
        r = np.array([-100.0, -2.0], dtype=np.float32)
        w = np.array([0.8, 0.2], dtype=np.float32)
        s = reward.scalarise(r, w)
        expected = 0.8 * (-100.0) + 0.2 * (-2.0)
        assert s == pytest.approx(expected)

    def test_energy_only(self):
        r = np.array([-50.0, -10.0], dtype=np.float32)
        w = np.array([1.0, 0.0], dtype=np.float32)
        assert reward.scalarise(r, w) == pytest.approx(-50.0)

    def test_sla_only(self):
        r = np.array([-50.0, -10.0], dtype=np.float32)
        w = np.array([0.0, 1.0], dtype=np.float32)
        assert reward.scalarise(r, w) == pytest.approx(-10.0)

    def test_zero_reward(self):
        r = np.array([0.0, 0.0], dtype=np.float32)
        w = np.array([0.5, 0.5], dtype=np.float32)
        assert reward.scalarise(r, w) == pytest.approx(0.0)

    def test_uses_env_weights_when_none(self, monkeypatch):
        monkeypatch.setenv("ENERGY_WEIGHT", "0.6")
        monkeypatch.setenv("SLA_WEIGHT", "0.4")
        r = np.array([-10.0, -5.0], dtype=np.float32)
        s = reward.scalarise(r, weights=None)
        expected = 0.6 * (-10.0) + 0.4 * (-5.0)
        assert s == pytest.approx(expected)

    def test_returns_float(self):
        r = np.array([-1.0, -1.0], dtype=np.float32)
        w = np.array([0.5, 0.5], dtype=np.float32)
        result = reward.scalarise(r, w)
        assert isinstance(result, float)


# ── RewardNormalizer ───────────────────────────────────────────────────────


class TestRewardNormalizer:
    def test_initial_state(self):
        norm = reward.RewardNormalizer()
        assert norm.count == 0
        np.testing.assert_array_equal(norm.mean, [0.0, 0.0])
        # std should be 1.0 when n < 2 (avoid division by zero)
        np.testing.assert_array_equal(norm.std, [1.0, 1.0])

    def test_single_update(self):
        norm = reward.RewardNormalizer()
        norm.update(np.array([-10.0, -2.0]))
        assert norm.count == 1
        np.testing.assert_allclose(norm.mean, [-10.0, -2.0])

    def test_normalize_after_updates(self):
        norm = reward.RewardNormalizer()
        # Feed constant rewards
        for _ in range(100):
            norm.update(np.array([-5.0, -1.0]))

        result = norm.normalize(np.array([-5.0, -1.0]))
        # Constant input → mean = input → normalised ≈ 0
        np.testing.assert_allclose(result, [0.0, 0.0], atol=0.01)

    def test_normalize_variance(self):
        norm = reward.RewardNormalizer()
        rng = np.random.default_rng(42)
        samples = rng.normal(loc=[-100, -5], scale=[20, 2], size=(1000, 2))
        for s in samples:
            norm.update(s)

        # After 1000 samples, mean should be close to true mean
        np.testing.assert_allclose(norm.mean, [-100, -5], atol=2.0)
        # std should be close to true std
        np.testing.assert_allclose(norm.std, [20, 2], atol=2.0)

    def test_update_and_normalize(self):
        norm = reward.RewardNormalizer()
        r = np.array([-10.0, -2.0], dtype=np.float32)
        result = norm.update_and_normalize(r)
        assert result.shape == (2,)
        assert result.dtype == np.float32
        assert norm.count == 1

    def test_normalize_returns_float32(self):
        norm = reward.RewardNormalizer()
        norm.update(np.array([-10.0, -2.0]))
        norm.update(np.array([-20.0, -4.0]))
        result = norm.normalize(np.array([-15.0, -3.0]))
        assert result.dtype == np.float32

    def test_different_magnitudes(self):
        """Energy can be ~100s, SLA can be ~0.1 — normaliser should handle this."""
        norm = reward.RewardNormalizer()
        for _ in range(500):
            norm.update(np.array([-500.0, -0.1]))
        for _ in range(500):
            norm.update(np.array([-100.0, -3.0]))

        # Both components should normalise to roughly the same scale
        r1 = norm.normalize(np.array([-500.0, -0.1]))
        r2 = norm.normalize(np.array([-100.0, -3.0]))
        # Neither should be orders of magnitude larger than the other
        assert abs(r1[0]) < 5.0
        assert abs(r1[1]) < 5.0
        assert abs(r2[0]) < 5.0
        assert abs(r2[1]) < 5.0


# ── RunningScalarNormalizer (G1.3) ──────────────────────────────────────────


class TestRunningScalarNormalizer:
    def test_initial_std_is_one(self):
        norm = reward.RunningScalarNormalizer()
        assert norm.count == 0
        assert norm.std == 1.0
        assert norm.mean == 0.0

    def test_centered_constant_normalises_to_zero(self):
        norm = reward.RunningScalarNormalizer(center=True)
        for _ in range(100):
            norm.update(-700.0)
        assert norm.normalize(-700.0) == pytest.approx(0.0, abs=1e-6)

    def test_uncentered_keeps_sign(self):
        """center=False (cost mode): positive input → positive output."""
        norm = reward.RunningScalarNormalizer(center=False)
        for x in np.abs(np.random.default_rng(0).normal(50, 10, size=200)):
            norm.update(float(x))
        out = norm.normalize(50.0)
        assert out > 0.0  # std-only scaling preserves the ≥0 nature of cost

    def test_scales_to_order_one(self):
        norm = reward.RunningScalarNormalizer(center=False)
        rng = np.random.default_rng(1)
        for x in np.abs(rng.normal(1e5, 2e4, size=500)):
            norm.update(float(x))
        # Raw ~1e5 → normalised ~O(few), not 1e5.
        assert abs(norm.normalize(1e5)) < 10.0

    def test_update_and_normalize_advances_state(self):
        norm = reward.RunningScalarNormalizer(center=True)
        for v in (-1.0, -2.0, -3.0):
            norm.update(v)
        # Probe a value that is NOT the running mean (−2) so the extra sample
        # in update_and_normalize visibly shifts mean/std.
        manual = norm.normalize(0.0)
        combined = reward.RunningScalarNormalizer(center=True)
        for v in (-1.0, -2.0, -3.0):
            combined.update(v)
        out = combined.update_and_normalize(0.0)
        assert isinstance(out, float)
        assert combined.count == 4
        assert manual != pytest.approx(out)  # extra update changed the stats


# ── Integration: from_java → normalise → scalarise ────────────────────────


class TestIntegration:
    def test_full_pipeline(self, monkeypatch):
        monkeypatch.setenv("ENERGY_WEIGHT", "0.5")
        monkeypatch.setenv("SLA_WEIGHT", "0.5")

        norm = reward.RewardNormalizer()

        # Simulate a few steps
        raw_rewards = [
            [-200.0, -1.0],
            [-180.0, -0.5],
            [-220.0, -2.0],
            [-190.0, -0.0],
        ]

        scalars = []
        for raw in raw_rewards:
            vec = reward.from_java(raw)
            normalised = norm.update_and_normalize(vec)
            s = reward.scalarise(normalised, weights=reward.load_weights())
            scalars.append(s)

        # All scalars should be finite
        assert all(np.isfinite(s) for s in scalars)
        # Should be floats
        assert all(isinstance(s, float) for s in scalars)
