"""
T4.2 — Reward processing: converts the raw Java reward vector into
numpy arrays for the MO-Gymnasium RL environment.

The Java side (SimulationManager.computeReward) returns a two-component
vector per step:

    reward[0] = R_energy = −ΔE          (negative energy delta, Watt-seconds)
    reward[1] = R_sla    = −κ × max(0, completion − deadline)

(κ is the per-task QoS weight — Phase 2 G1.0 renamed it from λ so that λ is
reserved for the Lagrangian multiplier of the Constrained-MDP formulation.
The non-negative magnitude ``−R_sla`` is the CMDP constraint cost ``C_SLA``.)

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


# ── Scalar running normaliser (Phase 2, G1.3) ──────────────────────────────

class RunningScalarNormalizer:
    """Welford online normaliser for a single SCALAR signal.

    The Phase-2 Constrained-MDP core scales ``R_energy`` and ``C_SLA``
    **separately** before forming the effective reward ``R_energy − λ·C_SLA``
    (CLAUDE.md Lưu ý #1).  Normalising the *fused* scalar is wrong because the
    Lagrange multiplier λ drifts over training, so a fused running scale would
    keep changing meaning.  Two independent instances of this class give each
    objective its own stable scale for PPO's value head.

    Parameters
    ----------
    center : bool
        If ``True`` (default) subtract the running mean — full standardisation,
        appropriate for the energy reward.  For the constraint cost ``C_SLA``
        pass ``center=False`` so the normalised value stays ≥ 0 and ``λ·C_SLA``
        remains a genuine non-negative penalty.
    epsilon : float
        Numerical floor added under the square-root of the variance.
    """

    def __init__(self, center: bool = True, epsilon: float = 1e-8,
                 freeze_after: int | None = None) -> None:
        self._n = 0
        self._mean = 0.0
        self._m2 = 0.0
        self._center = center
        self._epsilon = epsilon
        self._freeze_after = freeze_after
        self._frozen_std: float | None = None

    @property
    def count(self) -> int:
        return self._n

    @property
    def mean(self) -> float:
        return self._mean

    @property
    def std(self) -> float:
        if self._frozen_std is not None:
            return self._frozen_std
        if self._n < 2:
            return 1.0
        var = self._m2 / (self._n - 1)
        return float(np.sqrt(var + self._epsilon))

    @property
    def frozen(self) -> bool:
        """True once the scale has stopped tracking the incoming stream."""
        return self._frozen_std is not None

    def update(self, x: float) -> None:
        """Incorporate a new scalar sample (Welford's algorithm).

        Stops updating once ``freeze_after`` samples have been seen, if that
        was requested. Freezing matters for the CONSTRAINT signal and is a
        correctness issue, not a tuning knob:

        A running scale divides the cost by a std estimated from the same
        stream, so ``J = mean(c)/std(c)`` measures the *shape* of the cost
        distribution rather than its level. A policy that halves its raw cost
        also roughly halves that std, so J barely moves and the budget ``d``
        has nothing to grip. Measured on a full campaign: at LOW, driving
        lambda from 0 to 3.61 cut raw C_SLA by 10.6% while J went UP by 2.2%;
        at REPLAY the two moved in opposite directions again. The dual loop
        was pushing on a quantity decoupled from the objective it was meant
        to constrain.

        With the scale frozen, ``c_n`` is a fixed multiple of the raw cost, so
        ``J`` is proportional to mean raw cost per step and ``d`` becomes a
        (rescaled) physical target. Freeze late enough that the estimate is
        stable: a few episodes of samples is plenty.
        """
        if self._frozen_std is not None:
            return
        self._n += 1
        delta = float(x) - self._mean
        self._mean += delta / self._n
        delta2 = float(x) - self._mean
        self._m2 += delta * delta2
        if self._freeze_after is not None and self._n >= self._freeze_after:
            self._frozen_std = self.std

    def normalize(self, x: float) -> float:
        """Normalise a scalar using running statistics (no state update)."""
        shift = self._mean if self._center else 0.0
        return (float(x) - shift) / self.std

    def update_and_normalize(self, x: float) -> float:
        """Update statistics and return the normalised scalar in one call."""
        self.update(x)
        return self.normalize(x)


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
