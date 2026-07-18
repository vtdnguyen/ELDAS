"""SYS.1 — Parallel CloudSim environments via ``SubprocVecEnv``.

Rollout collection is the wall-clock cost of a CMDP run, and each step blocks on
a JVM simulation. Running N environments in N worker processes overlaps those
waits, which is the correct speed lever for a black-box discrete-event simulator
(CLAUDE.md Lưu ý #13 — the engine stays CloudSim).

Topology
--------
One JVM serves N *independent* gateways on consecutive ports (``NUM_GATEWAYS`` in
``docker-compose.yml``); worker ``i`` connects to ``PY4J_PORT + i``::

    SubprocVecEnv
      ├─ worker 0 ──Py4J:25333──┐
      ├─ worker 1 ──Py4J:25334──┤   one JVM,
      └─ worker i ──Py4J:2533x──┘   N × (GatewayEntryPoint → SimulationManager)

This is safe because ``SimulationManager`` holds **no static state**: each
gateway owns its own CloudSim instance, hosts, energy accumulators and stepping
thread. (``MetricsRegistry`` *is* static — its gauges are unlabelled per gateway,
so Prometheus output is not interpretable during a parallel run. Java's ``Main``
warns; monitoring is off the critical path, Lưu ý #16.)

λ across the process boundary
-----------------------------
``CMDPRewardWrapper`` normally reads λ live from the shared ``PIDLagrangian``.
Under ``SubprocVecEnv`` each worker holds a *pickled copy* of that controller, so
a live read would freeze λ at its construction value and silently disable the
dual update. :class:`LambdaBroadcastWrapper` is therefore mounted as the
**outermost** wrapper — ``VecEnv.env_method`` resolves attributes on the
outermost env — and ``train_cmdp`` pushes λ to every worker after each dual
update.

Reproducibility caveat (matters for the thesis)
-----------------------------------------------
``n_envs > 1`` changes rollout composition, so results are **not** bit-identical
to ``n_envs=1`` even at the same seed. The default stays 1 so every existing
Phase-1/Phase-2 number is reproduced exactly; opt in per run, and keep a fixed
``n_envs`` across the seeds of any comparison you report (Lưu ý #10).
"""

from __future__ import annotations

import os
from typing import Any, Callable

import gymnasium as gym
import numpy as np

from cmdp import CMDPRewardWrapper
from environment import CloudSimEnv

DEFAULT_BASE_PORT = int(os.environ.get("GATEWAY_PORT", "25333"))


class LambdaBroadcastWrapper(gym.Wrapper):
    """Outermost wrapper exposing λ control to ``VecEnv.env_method``.

    Why a dedicated wrapper rather than calling the CMDP wrapper directly:
    SB3's ``env_method`` does a plain ``getattr(env, name)`` on the outermost
    env, and Gymnasium 1.x removed ``Wrapper.__getattr__`` attribute forwarding.
    So a ``set_lambda`` buried under ``Monitor``/``ActionMasker`` is unreachable.
    Holding a direct reference and sitting on top makes the hook resolvable
    regardless of the wrapper stack beneath it.

    The same reasoning forces ``action_masks`` to delegate through a direct
    reference rather than ``self.env``: the env directly below is ``Monitor``,
    which does **not** forward unknown attributes down to the ``ActionMasker``.
    """

    def __init__(self, env: gym.Env, cmdp: CMDPRewardWrapper,
                 masker: gym.Env | None = None) -> None:
        super().__init__(env)
        self._cmdp = cmdp
        # Fall back to the CMDP wrapper, which also exposes action_masks by
        # delegating to the base env — equivalent masks, one less hop.
        self._masker = masker if masker is not None else cmdp

    def set_lambda(self, value: float) -> float:
        """Push λ down to the CMDP shaper (called once per dual update)."""
        return self._cmdp.set_lambda(value)

    def get_lambda(self) -> float:
        return self._cmdp.lambda_

    def action_masks(self) -> np.ndarray:
        # Direct reference, NOT self.env: Monitor sits between us and the
        # ActionMasker and does not forward attribute lookups (Gymnasium 1.x).
        return self._masker.action_masks()  # type: ignore[attr-defined]


def _action_mask_fn(env: gym.Env) -> np.ndarray:
    return env.action_masks()  # type: ignore[attr-defined]


def make_cmdp_env_fn(
    scenario: str,
    seed: int,
    *,
    lambda_init: float = 0.0,
    gateway_port: int | None = None,
    normalize: bool = True,
) -> Callable[[], gym.Env]:
    """Build a picklable thunk constructing one CMDP env bound to one gateway.

    The thunk defers construction to the worker process: the Py4J connection and
    its threads must be created *inside* the worker, never inherited across the
    fork/spawn boundary.

    ``pid=None`` is deliberate — a worker must not carry a copy of the dual
    controller that it would then read stale values from. λ arrives only via
    ``set_lambda``, seeded here with ``lambda_init`` so step 0 is well defined.
    """
    from sb3_contrib.common.wrappers import ActionMasker
    from stable_baselines3.common.monitor import Monitor

    def _init() -> gym.Env:
        base = CloudSimEnv(
            scenario=scenario,
            seed=seed,
            gateway_port=gateway_port,
            normalize_reward=False,   # the CMDP wrapper owns separate normalisation
        )
        shaped = CMDPRewardWrapper(base, pid=None, normalize=normalize)
        shaped.set_lambda(lambda_init)
        masked = ActionMasker(shaped, action_mask_fn=_action_mask_fn)
        monitored = Monitor(masked)
        return LambdaBroadcastWrapper(monitored, shaped, masker=masked)

    return _init


def make_cmdp_vec_env(
    scenario: str,
    seed: int,
    n_envs: int = 1,
    *,
    lambda_init: float = 0.0,
    base_port: int | None = None,
    normalize: bool = True,
    force_subproc: bool = False,
):
    """Build a vectorised CMDP env across ``n_envs`` gateways.

    Parameters
    ----------
    n_envs : int
        Number of parallel environments. Requires the Java side to serve at
        least this many gateways (``NUM_GATEWAYS``); worker ``i`` uses port
        ``base_port + i``.
    seed : int
        Base seed. Worker ``i`` gets ``seed + i`` so the workers explore
        different trajectories instead of stepping through identical episodes in
        lockstep — which would waste the parallelism entirely.
    force_subproc : bool
        Use ``SubprocVecEnv`` even for ``n_envs == 1`` (testing/diagnostics).

    Returns
    -------
    VecEnv
        ``DummyVecEnv`` when ``n_envs == 1`` (no process overhead, and the path
        every existing result was produced on), else ``SubprocVecEnv``.
    """
    from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv

    if n_envs < 1:
        raise ValueError(f"n_envs must be ≥ 1, got {n_envs}")

    port0 = DEFAULT_BASE_PORT if base_port is None else base_port
    fns = [
        make_cmdp_env_fn(
            scenario, seed + i, lambda_init=lambda_init,
            gateway_port=port0 + i, normalize=normalize,
        )
        for i in range(n_envs)
    ]

    if n_envs == 1 and not force_subproc:
        return DummyVecEnv(fns)

    # "fork" is the cheapest start method on Linux and avoids re-importing the
    # heavy RL stack per worker. Each worker builds its own Py4J connection
    # inside _init (post-fork), so no socket or JVM thread is inherited.
    return SubprocVecEnv(fns, start_method="fork")


def broadcast_lambda(vec_env, value: float) -> None:
    """Push λ to every worker's CMDP wrapper.

    No-op-safe: raises loudly if the hook is missing, because silently failing to
    propagate λ would leave the policy training against a stale multiplier while
    the dual controller's logs claimed otherwise — a wrong result that looks fine.
    """
    vec_env.env_method("set_lambda", float(value))
