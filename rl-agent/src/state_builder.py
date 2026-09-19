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
                            cpu_norm, mem_norm, gpu_norm, qos_weight_norm

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

import struct
from dataclasses import dataclass

import numpy as np
from gymnasium import spaces


# ── Constants ──────────────────────────────────────────────────────────────

TASK_FEATURE_DIM = 4   # cpu_norm, mem_norm, gpu_norm, qos_weight_norm
PER_HOST_DIM     = 6   # cpu_util, mem_util, gpu_util, state_onehot[3]

# SYS.2 — wire-format version of the packed step blob. Must match
# StepCodec.VERSION on the Java side; a mismatch means the jar and the Python
# source drifted apart, which would silently misread every field.
#
# v2 (W3.1) added droppedTasks to the header, moving the observation from byte 14
# to byte 18. Reading a v2 blob with the v1 header would take that int32 as the
# top half of observation[0] and shift every value after it — a corrupt state
# vector that still decodes without error, which is exactly what the version
# check exists to stop (CLAUDE.md Lưu ý #17).
PACKED_VERSION = 2
# version, done, taskIndex, H, obsLen, droppedTasks
_PACKED_HEADER = struct.Struct(">bbiiii")

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


# ── SYS.2 — packed single-RPC transport ────────────────────────────────────

@dataclass(frozen=True)
class PackedStep:
    """One decoded ``StepCodec`` blob — the full result of a step.

    Carries exactly what the per-element Py4J path yields, including the action
    mask for the resulting state (which the legacy path fetches with a separate
    ``getActionMask()`` call and H more round trips).
    """

    observation: np.ndarray   # (6H+4,) float32, clipped to [0, 1]
    reward: np.ndarray        # (2,) float64 — [R_energy, R_sla]
    cost: float               # C_SLA ≥ 0 for this step (G1.1)
    done: bool
    task_index: int
    task_name: str
    # (H,) bool on the packed transport; None on the legacy per-element
    # transport, which fetches the mask through a separate getActionMask() call.
    action_mask: np.ndarray | None
    # W3.1 — episode-cumulative count of tasks no host could accept. Cumulative
    # rather than a per-step flag so the final step carries the episode total
    # without a further round trip; the per-step event is the delta.
    dropped_tasks: int = 0


def decode_packed(blob, num_hosts: int) -> PackedStep:
    """Decode a ``StepCodec`` blob (see the Java class for the layout).

    Parameters
    ----------
    blob : bytes-like
        The ``byte[]`` returned by ``stepPacked`` / ``resetPacked``. Py4J hands
        Java ``byte[]`` to Python as ``bytes`` **by value** — a single round
        trip, unlike ``double[]``/``boolean[]`` which are proxied per element.
    num_hosts : int
        Expected H, cross-checked against the blob's own header.

    Raises
    ------
    ValueError
        On a version mismatch, a host-count mismatch, or a truncated blob —
        all of which mean the two sides disagree about the format and must
        fail loudly rather than decode garbage into an observation.
    """
    buf = bytes(blob)
    if len(buf) < _PACKED_HEADER.size:
        raise ValueError(
            f"Packed step blob is truncated: {len(buf)} bytes, "
            f"header alone needs {_PACKED_HEADER.size}"
        )

    version, done, task_index, h, obs_len, dropped = _PACKED_HEADER.unpack_from(buf, 0)
    if version != PACKED_VERSION:
        raise ValueError(
            f"Packed step wire-format mismatch: blob says v{version}, this "
            f"build expects v{PACKED_VERSION} — the Java jar and the Python "
            f"source are out of sync (rebuild cloudsim-java)."
        )
    if h != num_hosts:
        raise ValueError(f"Host count mismatch: blob says H={h}, env expects {num_hosts}")

    expected_obs = PER_HOST_DIM * num_hosts + TASK_FEATURE_DIM
    if obs_len != expected_obs:
        raise ValueError(
            f"Observation length mismatch: blob says {obs_len}, expected "
            f"{expected_obs} ({PER_HOST_DIM}×{num_hosts} + {TASK_FEATURE_DIM})"
        )

    off = _PACKED_HEADER.size
    # '>f8' — big-endian float64, matching the JVM's natural byte order, so the
    # decode does not depend on this machine's endianness.
    obs = np.frombuffer(buf, dtype=">f8", count=obs_len, offset=off).astype(np.float32)
    off += 8 * obs_len

    reward = np.frombuffer(buf, dtype=">f8", count=2, offset=off).astype(np.float64)
    off += 16

    cost = float(np.frombuffer(buf, dtype=">f8", count=1, offset=off)[0])
    off += 8

    mask = np.frombuffer(buf, dtype=np.uint8, count=h, offset=off).astype(bool)
    off += h

    (name_len,) = struct.unpack_from(">i", buf, off)
    off += 4
    if off + name_len > len(buf):
        raise ValueError(
            f"Packed step blob is truncated: taskName needs {name_len} bytes "
            f"at offset {off} but only {len(buf) - off} remain"
        )
    task_name = buf[off:off + name_len].decode("utf-8")

    # Clip on a writable copy — np.frombuffer views immutable bytes.
    obs = np.clip(obs, 0.0, 1.0)

    return PackedStep(
        observation=obs,
        reward=reward,
        cost=cost,
        done=bool(done),
        task_index=int(task_index),
        task_name=task_name,
        action_mask=mask,
        dropped_tasks=int(dropped),
    )


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
