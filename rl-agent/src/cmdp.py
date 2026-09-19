"""
G1.3 — Constrained-MDP reward shaping for the two-timescale PID-Lagrangian core.

The Phase-2 problem is a Constrained MDP:

    maximise   E[ Σ R_energy ]      subject to   E[ C_SLA ] ≤ d

solved by two-timescale primal-dual: ``MaskablePPO`` optimises the policy against
the Lagrangian *effective* reward

    R_eff = R_energy − λ · C_SLA                                       (primal)

while a :class:`~optim.pid_lagrangian.PIDLagrangian` advances the multiplier λ on
a slower timescale (dual; driven by ``LambdaUpdateCallback`` in ``train_cmdp.py``).

:class:`CMDPRewardWrapper` performs the primal shaping.  It reads the env's
``R_energy`` (= ``reward[0]``) and ``C_SLA`` (= ``info["cost"]`` from the G1.1
env), normalises the **two signals separately** (CLAUDE.md Lưu ý #1 — never
normalise the fused ``R_energy − λ·C_SLA`` because λ drifts), reads λ live from a
shared PID controller, and returns the scalar ``R_eff`` that MaskablePPO trains
on.  It also aggregates the per-episode constraint statistic ``J`` that the dual
callback consumes, keeping a single ``C_SLA`` definition across reward, λ-update
and budget ``d`` (Lưu ý #5).

This module deliberately imports only ``gymnasium`` + ``numpy`` (no sb3) so it
stays unit-testable without the heavy RL stack.
"""

from __future__ import annotations

from typing import Any

import gymnasium as gym
import numpy as np

from reward import RunningScalarNormalizer


class CMDPRewardWrapper(gym.Wrapper):
    """Collapse ``(R_energy, C_SLA)`` into the Lagrangian effective reward.

    Parameters
    ----------
    env : gym.Env
        Base env whose ``step`` returns a 2-vector reward ``[R_energy, R_sla]``
        and ``info["cost"] = C_SLA ≥ 0`` (the G1.1 ``CloudSimEnv``).  Build the
        base env with ``normalize_reward=False`` — this wrapper owns the
        (separate) normalisation.
    pid : object or None
        Shared dual controller exposing a ``lambda_`` attribute (read live each
        step).  Typically :class:`optim.pid_lagrangian.PIDLagrangian`.  May be
        ``None`` when λ is delivered by :meth:`set_lambda` instead — the case
        under ``SubprocVecEnv`` (SYS.1), where this wrapper lives in a worker
        process and cannot share the controller's memory.
    normalize : bool
        Apply separate Welford normalisation to ``R_energy`` and ``C_SLA``.
        Strongly recommended (Lưu ý #1); ``False`` only for tests/diagnostics.
    center_energy : bool
        Full-standardise the energy reward (subtract running mean).  The cost is
        always scaled std-only so it stays ≥ 0.

    Notes
    -----
    The per-episode constraint statistic exposed to the dual update is the
    **episode mean of the (normalised) per-step cost**, i.e. ``J = (Σ c_n) / T``.
    Using the mean (rather than the raw sum, which is ~1e8 in physical units —
    see ValidationRunner B16) keeps the budget ``d`` an O(1), sweepable quantity
    while remaining the *same* ``C_SLA`` signal that the reward penalises.
    """

    def __init__(
        self,
        env: gym.Env,
        pid: Any,
        normalize: bool = True,
        center_energy: bool = True,
        cost_freeze_after: int | None = None,
    ) -> None:
        super().__init__(env)
        self._pid = pid
        self._lambda_override: float | None = None
        self._normalize = normalize
        self._energy_norm = (
            RunningScalarNormalizer(center=center_energy) if normalize else None
        )
        # cost_freeze_after: stop the CONSTRAINT scale from tracking the policy.
        # See RunningScalarNormalizer.update for the measurement that motivates
        # it; without freezing, J is a shape statistic and the budget d cannot
        # steer the run. None keeps the original running-scale behaviour.
        self._cost_norm = (
            RunningScalarNormalizer(center=False, freeze_after=cost_freeze_after)
            if normalize else None
        )

        # Per-episode accumulators (reset in reset()).
        self._ep_cost_norm_sum = 0.0
        self._ep_cost_raw_sum = 0.0
        self._ep_energy_reward_sum = 0.0
        self._ep_len = 0

        # Last completed-episode statistics — read by the dual callback / eval.
        self.last_episode_cost: float | None = None        # mean normalised cost J
        self.last_episode_cost_raw: float | None = None     # Σ raw C_SLA (physical)

    # ── Helpers ────────────────────────────────────────────────────────────

    @property
    def lambda_(self) -> float:
        """Current Lagrange multiplier.

        Read live from the shared PID controller in-process.  Under
        ``SubprocVecEnv`` (SYS.1) the controller lives in the learner process and
        this wrapper only has a pickled copy of it, so a value pushed by
        :meth:`set_lambda` takes precedence — otherwise every worker would train
        against a frozen λ from construction time while the real dual advanced.
        """
        if self._lambda_override is not None:
            return self._lambda_override
        if self._pid is None:
            raise RuntimeError(
                "CMDPRewardWrapper has neither a pid nor a broadcast λ — "
                "pass pid=..., or call set_lambda() before stepping."
            )
        return float(self._pid.lambda_)

    def set_lambda(self, value: float) -> float:
        """Push a λ from the dual controller into this (possibly remote) wrapper.

        Called through ``VecEnv.env_method`` after each dual update.  λ is a
        slow-timescale variable updated once every ``dual_every`` episodes
        (Lưu ý #6), so workers using the previous λ for the remainder of an
        in-flight rollout is consistent with the two-timescale scheme rather
        than a race.
        """
        self._lambda_override = float(value)
        return self._lambda_override

    def _reset_episode_stats(self) -> None:
        self._ep_cost_norm_sum = 0.0
        self._ep_cost_raw_sum = 0.0
        self._ep_energy_reward_sum = 0.0
        self._ep_len = 0

    # ── Gymnasium API ──────────────────────────────────────────────────────

    def reset(self, **kwargs: Any) -> tuple[np.ndarray, dict[str, Any]]:
        obs, info = self.env.reset(**kwargs)
        self._reset_episode_stats()
        return obs, info

    def step(
        self, action: int
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        obs, reward_vec, terminated, truncated, info = self.env.step(action)

        r_energy = float(reward_vec[0])
        # Prefer the authoritative cost from the env (G1.1); fall back to the
        # SLA reward magnitude if an older env did not surface it.
        c_sla = float(info["cost"]) if "cost" in info else max(0.0, -float(reward_vec[1]))

        if self._normalize:
            e_n = self._energy_norm.update_and_normalize(r_energy)
            c_n = self._cost_norm.update_and_normalize(c_sla)
        else:
            e_n, c_n = r_energy, c_sla

        lam = self.lambda_
        effective = e_n - lam * c_n

        self._ep_cost_norm_sum += c_n
        self._ep_cost_raw_sum += c_sla
        self._ep_energy_reward_sum += r_energy
        self._ep_len += 1

        # Surface the CMDP decomposition for logging without losing the raw
        # vector reward (still in info["raw_reward"] from the base env).
        info["reward_energy"] = r_energy
        info["reward_energy_norm"] = e_n
        info["cost"] = c_sla
        info["cost_norm"] = c_n
        info["lambda"] = lam
        info["effective_reward"] = effective

        if terminated or truncated:
            T = max(1, self._ep_len)
            j = self._ep_cost_norm_sum / T
            self.last_episode_cost = j
            self.last_episode_cost_raw = self._ep_cost_raw_sum
            info["episode_cost"] = j
            info["episode_cost_raw"] = self._ep_cost_raw_sum
            info["episode_reward_energy"] = self._ep_energy_reward_sum

        return obs, effective, terminated, truncated, info

    # ── Pass-through for MaskablePPO ───────────────────────────────────────

    def action_masks(self) -> np.ndarray:
        return self.env.action_masks()  # type: ignore[attr-defined]
