"""
T4.2 — Reward processing: converts the raw Java reward vector into
numpy arrays for the MO-Gymnasium RL environment.

The Java side (SimulationManager.computeReward) returns a two-component
vector per step:

    reward[0] = R_energy = −ΔE          (negative energy delta, Watt-seconds)
    reward[1] = R_sla    = −λ × max(0, completion − deadline)

Both components are ≤ 0 (penalties).

This module:
  1. Converts Java double[] → numpy float32.
  2. Normalises each component to a stable range via running statistics,
     preventing one objective from dominating early training.
  3. Provides linear scalarization for single-objective PPO (Phase 2).
  4. Keeps the raw vector available for MO-Gymnasium's multi-objective API.
"""

from __future__ import annotations

import os

import numpy as np


# ── Constants ──────────────────────────────────────────────────────────────

REWARD_DIM = 2  # [R_energy, R_sla]

# Default scalarization weights — overridden by env vars or sweep config.
_DEFAULT_ENERGY_WEIGHT = 0.8
_DEFAULT_SLA_WEIGHT = 0.2


# ── Weight loading ─────────────────────────────────────────────────────────

def load_weights() -> np.ndarray:
    """Load scalarization weights from environment variables.

    Returns
    -------
    np.ndarray
        Shape ``(2,)``, ``[w_energy, w_sla]``, summing to 1.0.
    """
    w_energy = float(os.environ.get("ENERGY_WEIGHT", _DEFAULT_ENERGY_WEIGHT))
    w_sla = float(os.environ.get("SLA_WEIGHT", _DEFAULT_SLA_WEIGHT))
    total = w_energy + w_sla
    if total == 0:
        return np.array([0.5, 0.5], dtype=np.float32)
    return np.array([w_energy / total, w_sla / total], dtype=np.float32)


# ── Raw conversion ─────────────────────────────────────────────────────────

def from_java(java_array) -> np.ndarray:
    """Convert a Java double[] reward to a numpy float32 vector.

    Parameters
    ----------
    java_array : Java double[] or list-like
        The raw reward from ``StepResult.reward()``.

    Returns
    -------
    np.ndarray
        Shape ``(2,)``, dtype ``float32``.

    Raises
    ------
    ValueError
        If the array length is not 2.
    """
    vec = np.array(list(java_array), dtype=np.float32)
    if vec.shape[0] != REWARD_DIM:
        raise ValueError(
            f"Reward vector length mismatch: got {vec.shape[0]}, expected {REWARD_DIM}"
        )
    return vec


# ── Running normaliser ────────────────────────────────────────────────────

class RewardNormalizer:
    """Welford online normaliser for multi-objective reward vectors.

    Each component is independently normalised to approximately zero mean
    and unit variance.  This prevents one objective (e.g. energy, which
    can be orders of magnitude larger than SLA penalty in early episodes)
    from dominating the scalarised signal.

    Normalisation is optional — the environment can bypass it by passing
    raw rewards directly to ``scalarise()``.
    """

    def __init__(self, dim: int = REWARD_DIM, epsilon: float = 1e-8) -> None:
        self._n = 0
        self._mean = np.zeros(dim, dtype=np.float64)
        self._m2 = np.zeros(dim, dtype=np.float64)
        self._epsilon = epsilon

    @property
    def count(self) -> int:
        return self._n

    @property
    def mean(self) -> np.ndarray:
        return self._mean.copy().astype(np.float32)

    @property
    def std(self) -> np.ndarray:
        if self._n < 2:
            return np.ones_like(self._mean, dtype=np.float32)
        var = self._m2 / (self._n - 1)
        return np.sqrt(var + self._epsilon).astype(np.float32)

    def update(self, reward: np.ndarray) -> None:
        """Incorporate a new reward sample (Welford's algorithm)."""
        self._n += 1
        delta = reward.astype(np.float64) - self._mean
        self._mean += delta / self._n
        delta2 = reward.astype(np.float64) - self._mean
        self._m2 += delta * delta2

    def normalize(self, reward: np.ndarray) -> np.ndarray:
        """Normalise a reward vector using running statistics.

        Parameters
        ----------
        reward : np.ndarray
            Raw reward vector of shape ``(2,)``.

        Returns
        -------
        np.ndarray
            Normalised reward, shape ``(2,)``, dtype ``float32``.
        """
        return ((reward.astype(np.float64) - self._mean) / self.std).astype(
            np.float32
        )

    def update_and_normalize(self, reward: np.ndarray) -> np.ndarray:
        """Update statistics and return the normalised reward in one call."""
        self.update(reward)
        return self.normalize(reward)


# ── Scalarisation ──────────────────────────────────────────────────────────

def scalarise(
    reward_vec: np.ndarray,
    weights: np.ndarray | None = None,
) -> float:
    """Linear scalarisation of the reward vector.

    Parameters
    ----------
    reward_vec : np.ndarray
        Shape ``(2,)`` — ``[R_energy, R_sla]``.
    weights : np.ndarray or None
        Shape ``(2,)`` — ``[w_energy, w_sla]``.
        If ``None``, loads from environment variables.

    Returns
    -------
    float
        Scalar reward ``w_energy * R_energy + w_sla * R_sla``.
    """
    if weights is None:
        weights = load_weights()
    return float(np.dot(weights, reward_vec))
