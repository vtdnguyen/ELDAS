"""SYS.1 — tests for parallel envs and the λ broadcast across process boundaries.

The failure mode these guard against is the quiet one: under ``SubprocVecEnv``
each worker holds a *pickled copy* of the PID controller, so a wrapper that
reads ``pid.lambda_`` live would freeze λ at its construction value. Training
would look healthy — the dual logs would show λ climbing — while the policy
optimised against λ=0 the whole time, i.e. an unconstrained run reported as a
converged CMDP one.

These tests use a stub env (no gateway, no sb3 vec machinery) so they run
offline; the process-boundary semantics are exercised by pickling the wrapper
stack rather than by spawning real workers.
"""

from __future__ import annotations

import pickle
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import gymnasium as gym
from gymnasium import spaces

from cmdp import CMDPRewardWrapper


class StubEnv(gym.Env):
    """Minimal stand-in for CloudSimEnv: 2-vector reward + info['cost']."""

    def __init__(self, num_hosts: int = 3, episode_len: int = 4):
        self.observation_space = spaces.Box(0.0, 1.0, shape=(6 * num_hosts + 4,),
                                            dtype=np.float32)
        self.action_space = spaces.Discrete(num_hosts)
        self._n = num_hosts
        self._len = episode_len
        self._t = 0

    def reset(self, *, seed=None, options=None):
        self._t = 0
        return np.zeros(self.observation_space.shape, dtype=np.float32), {}

    def step(self, action):
        self._t += 1
        obs = np.zeros(self.observation_space.shape, dtype=np.float32)
        # R_energy = −1 per step, C_SLA = +2 per step: fixed so any change in the
        # effective reward is attributable to λ alone.
        return obs, np.array([-1.0, -2.0], dtype=np.float32), self._t >= self._len, \
            False, {"cost": 2.0}

    def action_masks(self):
        return np.ones(self._n, dtype=bool)


class FakePID:
    def __init__(self, lambda_: float = 0.0):
        self.lambda_ = lambda_


def make_stack(pid=None, lambda_init: float | None = None):
    """CMDP wrapper over the stub, normalisation off so arithmetic is exact."""
    shaped = CMDPRewardWrapper(StubEnv(), pid=pid, normalize=False)
    if lambda_init is not None:
        shaped.set_lambda(lambda_init)
    return shaped


class TestLambdaSource:
    def test_reads_pid_live_when_no_override(self):
        pid = FakePID(0.0)
        w = make_stack(pid=pid)
        assert w.lambda_ == 0.0
        pid.lambda_ = 0.75
        assert w.lambda_ == 0.75, "in-process wrapper must track the shared pid"

    def test_set_lambda_takes_precedence_over_pid(self):
        # This is the SubprocVecEnv case: the pid copy is stale, the broadcast
        # value is authoritative.
        pid = FakePID(0.0)
        w = make_stack(pid=pid)
        w.set_lambda(0.5)
        pid.lambda_ = 99.0  # a stale copy advancing in the worker must not win
        assert w.lambda_ == 0.5

    def test_no_pid_and_no_lambda_raises_rather_than_defaulting(self):
        # Silently defaulting to λ=0 would train an unconstrained policy while
        # reporting it as constrained.
        w = CMDPRewardWrapper(StubEnv(), pid=None, normalize=False)
        with pytest.raises(RuntimeError, match="neither a pid nor a broadcast"):
            _ = w.lambda_

    def test_set_lambda_returns_the_applied_value(self):
        w = make_stack(lambda_init=0.0)
        assert w.set_lambda(1.25) == 1.25


class TestLambdaAffectsEffectiveReward:
    """λ must actually reach the reward arithmetic, not just be stored."""

    def test_effective_reward_uses_broadcast_lambda(self):
        w = make_stack(lambda_init=0.0)
        w.reset()
        _, r0, *_ = w.step(0)
        assert r0 == pytest.approx(-1.0)  # R_eff = −1 − 0·2

        w.set_lambda(0.5)
        _, r1, *_ = w.step(0)
        assert r1 == pytest.approx(-2.0)  # R_eff = −1 − 0.5·2

    def test_broadcast_mid_episode_is_picked_up_on_next_step(self):
        w = make_stack(lambda_init=0.0)
        w.reset()
        w.step(0)
        w.set_lambda(2.0)
        _, r, *_ = w.step(0)
        assert r == pytest.approx(-5.0)  # −1 − 2·2

    def test_episode_cost_statistic_is_lambda_independent(self):
        # J is the constraint measurement fed BACK to the PID; if λ leaked into
        # it the dual loop would be measuring its own output.
        js = []
        for lam in (0.0, 3.0):
            w = make_stack(lambda_init=lam)
            w.reset()
            for _ in range(4):
                _, _, term, _, info = w.step(0)
            js.append(info["episode_cost"])
        assert js[0] == pytest.approx(js[1])


class TestPicklability:
    """SubprocVecEnv pickles the env thunk; a live pid must not be required."""

    def test_wrapper_stack_without_pid_is_picklable(self):
        w = make_stack(lambda_init=0.25)
        restored = pickle.loads(pickle.dumps(w))
        assert restored.lambda_ == 0.25

    def test_broadcast_lambda_survives_pickling(self):
        w = make_stack(lambda_init=0.0)
        w.set_lambda(1.5)
        restored = pickle.loads(pickle.dumps(w))
        assert restored.lambda_ == 1.5
        restored.reset()
        _, r, *_ = restored.step(0)
        assert r == pytest.approx(-4.0)  # −1 − 1.5·2


class OpaqueWrapper(gym.Wrapper):
    """A wrapper that does NOT forward unknown attributes — like ``Monitor``.

    Gymnasium 1.x removed ``Wrapper.__getattr__`` forwarding, so a plain
    ``gym.Wrapper`` genuinely hides ``action_masks``/``set_lambda`` defined
    further down the stack. Standing in for Monitor here keeps these tests
    honest: an earlier version of this file wrapped the CMDP wrapper *directly*,
    which forwarded by accident and let a real AttributeError reach production.
    """


class TestBroadcastWrapper:
    """LambdaBroadcastWrapper must expose its hooks on the OUTERMOST env.

    SB3's env_method and sb3-contrib's get_action_masks both do a plain getattr
    on the outermost env, so anything buried under Monitor/ActionMasker is
    unreachable from the learner process.
    """

    @staticmethod
    def _realistic_stack():
        """Mirror the production layering: CMDP → (opaque, i.e. Monitor) → outer."""
        from vec_env import LambdaBroadcastWrapper

        shaped = make_stack(lambda_init=0.0)
        opaque = OpaqueWrapper(shaped)
        return shaped, LambdaBroadcastWrapper(opaque, shaped, masker=shaped)

    def test_opaque_wrapper_really_hides_attributes(self):
        # Guards the guard: if this ever starts forwarding, the tests below stop
        # proving anything and must be revisited.
        shaped = make_stack(lambda_init=0.0)
        with pytest.raises(AttributeError):
            OpaqueWrapper(shaped).action_masks

    def test_set_lambda_resolves_by_plain_getattr(self):
        shaped, outer = self._realistic_stack()
        # Exactly what SubprocVecEnv's worker does for env_method.
        getattr(outer, "set_lambda")(0.8)
        assert shaped.lambda_ == 0.8
        assert getattr(outer, "get_lambda")() == 0.8

    def test_action_masks_resolves_through_an_opaque_middle_wrapper(self):
        # The regression: delegating via self.env hits Monitor and raises
        # AttributeError, which killed every SubprocVecEnv worker at startup.
        _, outer = self._realistic_stack()
        np.testing.assert_array_equal(
            getattr(outer, "action_masks")(), np.ones(3, dtype=bool))

    def test_set_lambda_reaches_reward_through_the_full_stack(self):
        _, outer = self._realistic_stack()
        outer.reset()
        getattr(outer, "set_lambda")(1.0)
        _, r, *_ = outer.step(0)
        assert r == pytest.approx(-3.0)  # −1 − 1·2


class TestVecEnvFactory:
    def test_rejects_zero_envs(self):
        from vec_env import make_cmdp_vec_env

        with pytest.raises(ValueError, match="n_envs must be ≥ 1"):
            make_cmdp_vec_env("LOW", 42, n_envs=0)

    def test_env_fn_is_a_deferred_thunk(self):
        # Construction must happen inside the worker: building a Py4J connection
        # in the parent and inheriting it across fork would break.
        from vec_env import make_cmdp_env_fn

        fn = make_cmdp_env_fn("LOW", 42, gateway_port=25333)
        assert callable(fn)  # no gateway contacted yet

    def test_workers_get_distinct_seeds_and_ports(self, monkeypatch):
        # Identical seeds would make every worker replay the same episode in
        # lockstep, wasting the parallelism entirely.
        import vec_env

        captured = []

        def spy(scenario, seed, *, lambda_init=0.0, gateway_port=None, normalize=True):
            captured.append((seed, gateway_port))
            return lambda: None

        monkeypatch.setattr(vec_env, "make_cmdp_env_fn", spy)

        class FakeVec:
            def __init__(self, fns):
                self.fns = fns

        monkeypatch.setattr(
            "stable_baselines3.common.vec_env.SubprocVecEnv", FakeVec, raising=False)

        try:
            vec_env.make_cmdp_vec_env("LOW", 42, n_envs=3, base_port=25333)
        except Exception:
            pass  # only the port/seed plan matters here

        assert captured == [(42, 25333), (43, 25334), (44, 25335)]
