"""
T4.1 — Observation builder: converts raw Java observation arrays into
normalised numpy vectors for the MO-Gymnasium RL environment.

The Java side (SimulationManager.buildObservation) already produces a
flat double[] with layout:

    [0 .. H-1]        host CPU utilisation       (0.0–1.0)
    [H .. 2H-1]       host memory utilisation     (0.0–1.0)
    [2H .. 3H-1]      host GPU utilisation         (0.0–1.0)
    [3H .. 3H+3]      current task features:
                         cpu_norm, mem_norm, gpu_norm, qos_lambda_norm

Total length = 3H + 4.

This module:
  1. Converts the Java array to a numpy float32 vector.
  2. Clips values to [0, 1] for numerical safety (Java util values can
     occasionally exceed 1.0 under time-shared VM scheduling).
  3. Provides the Gymnasium observation_space (Box) definition.
  4. Offers a structured parse for debugging / logging.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from gymnasium import spaces


# ── Constants ──────────────────────────────────────────────────────────────

TASK_FEATURE_DIM = 4  # cpu_norm, mem_norm, gpu_norm, qos_lambda_norm


# ── Observation space factory ──────────────────────────────────────────────

def observation_space(num_hosts: int) -> spaces.Box:
    """Return a Gymnasium Box for the observation vector."""
    dim = 3 * num_hosts + TASK_FEATURE_DIM
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
        Shape ``(3H + 4,)``, dtype ``float32``, values clipped to [0, 1].

    Raises
    ------
    ValueError
        If the array length does not match the expected ``3H + 4``.
    """
    obs = np.array(list(java_array), dtype=np.float32)
    expected = 3 * num_hosts + TASK_FEATURE_DIM
    if obs.shape[0] != expected:
        raise ValueError(
            f"Observation length mismatch: got {obs.shape[0]}, "
            f"expected {expected} (3×{num_hosts} + {TASK_FEATURE_DIM})"
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


def parse(obs: np.ndarray, num_hosts: int) -> ParsedObservation:
    """Split a flat observation vector into named components.

    Parameters
    ----------
    obs : np.ndarray
        Flat observation of shape ``(3H + 4,)``.
    num_hosts : int
        Number of hosts (H).

    Returns
    -------
    ParsedObservation
    """
    h = num_hosts
    return ParsedObservation(
        host_cpu_util=obs[0:h].copy(),
        host_mem_util=obs[h : 2 * h].copy(),
        host_gpu_util=obs[2 * h : 3 * h].copy(),
        task_cpu_norm=float(obs[3 * h]),
        task_mem_norm=float(obs[3 * h + 1]),
        task_gpu_norm=float(obs[3 * h + 2]),
        task_qos_norm=float(obs[3 * h + 3]),
    )
