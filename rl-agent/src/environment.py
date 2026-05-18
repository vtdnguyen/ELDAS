"""
T4.3 — MO-Gymnasium environment wrapping the CloudSim Plus simulation.

Connects to the Java GatewayEntryPoint via Py4J and exposes a standard
Gymnasium / MO-Gymnasium interface for RL training:

    env = CloudSimEnv(scenario="HIGH", seed=42)
    obs, info = env.reset()
    obs, reward, terminated, truncated, info = env.step(action)

Key features:
  - **Vector reward**: returns ``np.ndarray([R_energy, R_sla])`` — keeps
    multi-objective signal intact for MO-Gymnasium / MORL algorithms.
  - **Action masking**: ``env.action_masks()`` returns a bool array for
    MaskablePPO (sb3-contrib).  Invalid hosts are masked out.
  - **Linear scalarization wrapper**: ``ScalarRewardWrapper`` converts
    the vector reward to a scalar for standard PPO / MaskablePPO training.
  - **Py4J retry**: tolerates transient connection failures during Docker
    startup (configurable retries + backoff).
"""

from __future__ import annotations

import os
import time
from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from py4j.java_gateway import JavaGateway, GatewayParameters

import reward as reward_mod
import state_builder


# ── Py4J connection ────────────────────────────────────────────────────────

_DEFAULT_HOST = "cloudsim-java"
_DEFAULT_PORT = 25333
_CONNECT_RETRIES = 10
_CONNECT_BACKOFF_SEC = 3.0


def _connect_gateway(
    host: str | None = None,
    port: int | None = None,
    retries: int = _CONNECT_RETRIES,
    backoff: float = _CONNECT_BACKOFF_SEC,
) -> JavaGateway:
    """Open a Py4J connection to the Java gateway with retry logic.

    Docker Compose starts both containers concurrently — the healthcheck
    ensures Java is up before Python starts, but a brief window may remain.
    """
    host = host or os.environ.get("GATEWAY_HOST", _DEFAULT_HOST)
    port = port or int(os.environ.get("GATEWAY_PORT", str(_DEFAULT_PORT)))

    last_err: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            gw = JavaGateway(
                gateway_parameters=GatewayParameters(
                    address=host, port=port, auto_convert=True
                )
            )
            # Smoke-test: call a lightweight method on the entry point
            _ = gw.entry_point.toString()
            print(f"[environment] Connected to Py4J gateway {host}:{port} "
                  f"(attempt {attempt})")
            return gw
        except Exception as e:
            last_err = e
            if attempt < retries:
                print(f"[environment] Py4J connection failed (attempt {attempt}/"
                      f"{retries}): {e} — retrying in {backoff}s")
                time.sleep(backoff)
    raise ConnectionError(
        f"Cannot connect to Py4J gateway at {host}:{port} "
        f"after {retries} attempts"
    ) from last_err


# ── Core environment ───────────────────────────────────────────────────────

class CloudSimEnv(gym.Env):
    """Gymnasium environment backed by CloudSim Plus via Py4J.

    **Observation space**: ``Box(0, 1, shape=(6H+4,), float32)``
        H host-level features (CPU, MEM, GPU util) + 4 task features.

    **Action space**: ``Discrete(H)``
        Index of the host to schedule the current task on.

    **Reward**: ``np.ndarray([R_energy, R_sla], dtype=float32)``
        Multi-objective vector.  Use ``ScalarRewardWrapper`` to scalarise.

    Parameters
    ----------
    scenario : str
        Load scenario: ``"LOW"``, ``"HIGH"``, or ``"BURST"``.
    seed : int or None
        Random seed.  ``None`` reads ``RANDOM_SEED`` from env.
    gateway_host, gateway_port : str, int or None
        Py4J connection overrides (default: from env vars).
    normalize_reward : bool
        If ``True``, apply Welford running normalisation to the reward
        vector (useful when combining with scalarisation).
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        scenario: str = "HIGH",
        seed: int | None = None,
        gateway_host: str | None = None,
        gateway_port: int | None = None,
        normalize_reward: bool = False,
    ) -> None:
        super().__init__()

        self._scenario = scenario.upper()
        self._seed = seed or int(os.environ.get("RANDOM_SEED", "42"))
        self._normalize = normalize_reward
        self._reward_normalizer = reward_mod.RewardNormalizer() if normalize_reward else None

        # Connect to Java
        self._gateway = _connect_gateway(gateway_host, gateway_port)
        self._ep = self._gateway.entry_point

        # Do an initial reset to discover action/observation sizes
        self._initial_result = self._ep.reset(self._scenario, self._seed)
        self._num_hosts: int = int(self._ep.getActionSize())

        # Spaces
        self.observation_space = state_builder.observation_space(self._num_hosts)
        self.action_space = spaces.Discrete(self._num_hosts)

        # Episode state
        self._current_obs: np.ndarray | None = None
        self._done = False
        self._step_count = 0
        self._episode_rewards: list[np.ndarray] = []
        self._needs_initial_reset = True

    # ── Gymnasium API ──────────────────────────────────────────────────

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        """Reset the simulation and return the initial observation.

        Parameters
        ----------
        seed : int, optional
            Override seed for this episode.
        options : dict, optional
            - ``"scenario"``: override scenario for this episode.

        Returns
        -------
        observation : np.ndarray
        info : dict
        """
        ep_seed = seed if seed is not None else self._seed
        ep_scenario = (options or {}).get("scenario", self._scenario)

        # If we just created the env and haven't stepped, reuse the initial reset
        if self._needs_initial_reset and ep_seed == self._seed and ep_scenario == self._scenario:
            result = self._initial_result
            self._needs_initial_reset = False
        else:
            result = self._ep.reset(ep_scenario, ep_seed)

        self._current_obs = state_builder.from_java(
            result.observation(), self._num_hosts
        )
        self._done = False
        self._step_count = 0
        self._episode_rewards = []

        if self._reward_normalizer is not None:
            self._reward_normalizer = reward_mod.RewardNormalizer()

        info = {
            "task_index": int(result.taskIndex()),
            "task_name": str(result.taskName()),
            "num_hosts": self._num_hosts,
            "scenario": ep_scenario,
            "seed": ep_seed,
        }
        return self._current_obs, info

    def step(
        self, action: int
    ) -> tuple[np.ndarray, np.ndarray, bool, bool, dict[str, Any]]:
        """Execute one scheduling decision.

        Parameters
        ----------
        action : int
            Index of the host to place the current task on.

        Returns
        -------
        observation : np.ndarray  — shape ``(6H+4,)``
        reward : np.ndarray       — shape ``(2,)`` ``[R_energy, R_sla]``
        terminated : bool         — ``True`` when all tasks are scheduled
        truncated : bool          — always ``False`` (no time limit)
        info : dict
        """
        if self._done:
            raise RuntimeError("Episode is done — call reset() first")

        result = self._ep.step(int(action))

        self._current_obs = state_builder.from_java(
            result.observation(), self._num_hosts
        )
        raw_reward = reward_mod.from_java(result.reward())

        if self._reward_normalizer is not None:
            reward_vec = self._reward_normalizer.update_and_normalize(raw_reward)
        else:
            reward_vec = raw_reward

        terminated = bool(result.done())
        self._done = terminated
        self._step_count += 1
        self._episode_rewards.append(raw_reward)

        info: dict[str, Any] = {
            "task_index": int(result.taskIndex()),
            "task_name": str(result.taskName()),
            "step": self._step_count,
            "raw_reward": raw_reward,
        }

        if terminated:
            info["episode"] = self._build_episode_info()

        return self._current_obs, reward_vec, terminated, False, info

    # ── Action masking (for MaskablePPO) ──────────────────────────────

    def action_masks(self) -> np.ndarray:
        """Return a boolean mask over the action space.

        ``True`` at index ``i`` means host ``i`` can accept the current task.
        MaskablePPO calls this automatically each step.
        """
        java_mask = self._ep.getActionMask()
        mask = np.array(list(java_mask), dtype=bool)

        # Safety: if no host is feasible, allow all (prevent stuck episode)
        if not mask.any():
            mask[:] = True

        return mask

    # ── Episode summary ───────────────────────────────────────────────

    def _build_episode_info(self) -> dict[str, Any]:
        """Build end-of-episode summary for logging."""
        rewards = np.array(self._episode_rewards)  # (T, 2)
        return {
            "length": self._step_count,
            "total_energy_reward": float(rewards[:, 0].sum()),
            "total_sla_reward": float(rewards[:, 1].sum()),
            "mean_energy_reward": float(rewards[:, 0].mean()),
            "mean_sla_reward": float(rewards[:, 1].mean()),
            "total_energy_kwh": float(self._ep.getTotalEnergyKwh()),
        }

    # ── Metrics export ────────────────────────────────────────────────

    def export_metrics(self, output_dir: str = "/data/results") -> None:
        """Delegate CSV/JSON export to the Java side."""
        self._ep.exportMetrics(output_dir)

    # ── Cleanup ───────────────────────────────────────────────────────

    def close(self) -> None:
        """Close this client's Py4J connection without killing the Java server.

        Passes send_shutdown=False so that the Java GatewayServer keeps running
        and other Python clients (or the next test phase) can still connect.
        Call ep.shutdown() explicitly if you want to stop the Java side.
        """
        try:
            self._gateway.shutdown(send_shutdown=False)
        except Exception:
            pass


# ── Scalar reward wrapper ─────────────────────────────────────────────────

class ScalarRewardWrapper(gym.RewardWrapper):
    """Convert the vector reward to a scalar via linear scalarisation.

    Wraps ``CloudSimEnv`` for use with standard PPO / MaskablePPO which
    expect a float reward.

    Parameters
    ----------
    env : CloudSimEnv
        The base environment.
    weights : np.ndarray or None
        ``[w_energy, w_sla]``.  If ``None``, reads from env vars.
    """

    def __init__(
        self,
        env: CloudSimEnv,
        weights: np.ndarray | None = None,
    ) -> None:
        super().__init__(env)
        self._weights = weights if weights is not None else reward_mod.load_weights()

    def reward(self, reward: np.ndarray) -> float:  # type: ignore[override]
        return reward_mod.scalarise(reward, self._weights)

    # Propagate action_masks through the wrapper
    def action_masks(self) -> np.ndarray:
        return self.env.action_masks()  # type: ignore[union-attr]


# ── Factory helpers ────────────────────────────────────────────────────────

def make_env(
    scenario: str = "HIGH",
    seed: int | None = None,
    scalarise: bool = True,
    weights: np.ndarray | None = None,
    normalize_reward: bool = False,
    gateway_host: str | None = None,
    gateway_port: int | None = None,
) -> gym.Env:
    """Create and optionally wrap the CloudSim environment.

    Parameters
    ----------
    scenario : str
        ``"LOW"``, ``"HIGH"``, or ``"BURST"``.
    seed : int or None
        Random seed.
    scalarise : bool
        If ``True``, wrap with ``ScalarRewardWrapper`` for PPO training.
    weights : np.ndarray or None
        Scalarisation weights (only used if ``scalarise=True``).
    normalize_reward : bool
        Apply running reward normalisation inside the base env.
    gateway_host, gateway_port : str, int or None
        Py4J connection overrides.

    Returns
    -------
    gym.Env
        Ready-to-use environment.
    """
    env: gym.Env = CloudSimEnv(
        scenario=scenario,
        seed=seed,
        gateway_host=gateway_host,
        gateway_port=gateway_port,
        normalize_reward=normalize_reward,
    )
    if scalarise:
        env = ScalarRewardWrapper(env, weights=weights)
    return env
