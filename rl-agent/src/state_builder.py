"""
T4.1 / T8.9 — Observation builder: converts raw Java observation arrays
into normalised numpy vectors for the MO-Gymnasium RL environment.

The Java side (SimulationManager.buildObservation) produces a flat double[]
with layout:

    [0    .. H-1   ]    host CPU utilisation       (0.0–1.0)
    [H    .. 2H-1  ]    host memory utilisation    (0.0–1.0)
    [2H   .. 3H-1  ]    host GPU utilisation        (0.0–1.0)
    [3H   .. 6H-1  ]    host state one-hot (3 per host):
                            SUSPENDED, IDLE, ACTIVE
    [6H   .. 6H+3  ]    current task features:
                            cpu_norm, mem_norm, gpu_norm, qos_lambda_norm

Total length = 6H + 4.

This module:
  1. Converts the Java array to a numpy float32 vector.
  2. Clips values to [0, 1] for numerical safety (the one-hot block is
     already in {0, 1} but utilisation can occasionally exceed 1.0).
  3. Provides the Gymnasium observation_space (Box) definition.
  4. Offers a structured parse for debugging / logging.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from gymnasium import spaces


# ── Constants ──────────────────────────────────────────────────────────────

TASK_FEATURE_DIM = 4   # cpu_norm, mem_norm, gpu_norm, qos_lambda_norm
PER_HOST_DIM     = 6   # cpu_util, mem_util, gpu_util, state_onehot[3]

# Slice helpers for the host blocks: feature i ∈ {0..5} starts at i*H.
# Layout is feature-major (all H CPU utils, then all H mem utils, …).
_CPU   = 0
_MEM   = 1
_GPU   = 2
_SUS   = 3   # one-hot: SUSPENDED
_IDLE  = 4   # one-hot: IDLE
_ACT   = 5   # one-hot: ACTIVE


# ── Observation space factory ──────────────────────────────────────────────

def observation_space(num_hosts: int) -> spaces.Box:
    """Return a Gymnasium Box for the observation vector (6H + 4)."""
    dim = PER_HOST_DIM * num_hosts + TASK_FEATURE_DIM
    return spaces.Box(low=0.0, high=1.0, shape=(dim,), dtype=np.float32)


# ── Raw conversion ─────────────────────────────────────────────────────────

def from_java(java_array, num_hosts: int) -> np.ndarray:
    """Convert a Java double[] (via Py4J) to a clipped float32 numpy array.

    Parameters
    ----------
    java_array : Java double[] or list-like
        The raw observation from ``StepResult.observation()``.
    num_hosts : int
        Number of hosts in the datacenter (H).

    Returns
    -------
    np.ndarray
        Shape ``(6H + 4,)``, dtype ``float32``, values clipped to [0, 1].

    Raises
    ------
    ValueError
        If the array length does not match the expected ``6H + 4``.
    """
    obs = np.array(list(java_array), dtype=np.float32)
    expected = PER_HOST_DIM * num_hosts + TASK_FEATURE_DIM
    if obs.shape[0] != expected:
        raise ValueError(
            f"Observation length mismatch: got {obs.shape[0]}, "
            f"expected {expected} ({PER_HOST_DIM}×{num_hosts} + {TASK_FEATURE_DIM})"
        )
    np.clip(obs, 0.0, 1.0, out=obs)
    return obs


# ── Structured parse (for debugging / logging) ────────────────────────────

@dataclass(frozen=True, slots=True)
class ParsedObservation:
    """Decomposed observation vector for inspection."""

    host_cpu_util: np.ndarray   # (H,) CPU utilisation per host
    host_mem_util: np.ndarray   # (H,) memory utilisation per host
    host_gpu_util: np.ndarray   # (H,) GPU utilisation per host
    host_state_onehot: np.ndarray  # (H, 3) — [suspended, idle, active]
    task_cpu_norm: float        # current task CPU demand (normalised)
    task_mem_norm: float        # current task memory demand (normalised)
    task_gpu_norm: float        # current task GPU demand (normalised)
    task_qos_norm: float        # current task QoS lambda (normalised)

    @property
    def num_hosts(self) -> int:
        return self.host_cpu_util.shape[0]

    def mean_cpu_util(self) -> float:
        return float(self.host_cpu_util.mean())

    def mean_mem_util(self) -> float:
        return float(self.host_mem_util.mean())

    def mean_gpu_util(self) -> float:
        return float(self.host_gpu_util.mean())

    def host_state_codes(self) -> np.ndarray:
        """Decoded per-host state index ∈ {0=SUSPENDED, 1=IDLE, 2=ACTIVE}."""
        return self.host_state_onehot.argmax(axis=1)


def parse(obs: np.ndarray, num_hosts: int) -> ParsedObservation:
    """Split a flat observation vector into named components.

    Parameters
    ----------
    obs : np.ndarray
        Flat observation of shape ``(6H + 4,)``.
    num_hosts : int
        Number of hosts (H).

    Returns
    -------
    ParsedObservation
    """
    h = num_hosts
    # Java emits the one-hot as 3 contiguous floats per host (host-major),
    # so [3H .. 6H-1] is laid out (host0[sus,idle,act], host1[sus,idle,act], …).
    onehot = obs[3 * h : 6 * h].reshape(h, 3).copy()
    return ParsedObservation(
        host_cpu_util=obs[0:h].copy(),
        host_mem_util=obs[h : 2 * h].copy(),
        host_gpu_util=obs[2 * h : 3 * h].copy(),
        host_state_onehot=onehot,
        task_cpu_norm=float(obs[6 * h]),
        task_mem_norm=float(obs[6 * h + 1]),
        task_gpu_norm=float(obs[6 * h + 2]),
        task_qos_norm=float(obs[6 * h + 3]),
    )
