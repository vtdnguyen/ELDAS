"""Unit tests for optim/pid_lagrangian.py (G1.2).

Covers the correctness properties that the Phase-2 CMDP core relies on:
  - λ ≥ 0 always (dual feasibility);
  - integral anti-windup (no negative accumulation);
  - λ_init seeds the controller (Δ = 0 is a no-op);
  - pure-integral mode reproduces vanilla Lagrangian ascent;
  - the controller drives a simple linear cost plant to the budget d
    (the behaviour G1.4 cross-validates against OmniSafe CPPOPID).
"""

import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from optim.pid_lagrangian import PIDLagrangian


# ── Construction ────────────────────────────────────────────────────────────


class TestConstruction:
    def test_defaults(self):
        pid = PIDLagrangian()
        assert pid.k_p == pytest.approx(1e-4)
        assert pid.k_i == pytest.approx(1e-4)
        assert pid.k_d == pytest.approx(0.0)
        assert pid.lambda_ == 0.0
        assert pid.integral == 0.0
        assert pid.prev_delta == 0.0

    def test_lambda_init_seeds_state(self):
        pid = PIDLagrangian(lambda_init=2.5)
        assert pid.lambda_ == pytest.approx(2.5)
        assert pid.integral == pytest.approx(2.5)

    def test_negative_gain_raises(self):
        with pytest.raises(ValueError, match="gains"):
            PIDLagrangian(k_p=-1.0)
        with pytest.raises(ValueError, match="gains"):
            PIDLagrangian(k_i=-1e-3)
        with pytest.raises(ValueError, match="gains"):
            PIDLagrangian(k_d=-0.5)

    def test_negative_lambda_init_raises(self):
        with pytest.raises(ValueError, match="lambda_init"):
            PIDLagrangian(lambda_init=-0.1)

    def test_lambda_max_below_init_raises(self):
        with pytest.raises(ValueError, match="lambda_max"):
            PIDLagrangian(lambda_init=3.0, lambda_max=1.0)


# ── Basic update semantics ──────────────────────────────────────────────────


class TestUpdate:
    def test_no_violation_is_noop_at_init(self):
        """update(J_cost = d) with Δ = 0 keeps λ at its initial value."""
        pid = PIDLagrangian(lambda_init=1.5)
        lam = pid.update(j_cost=4.0, budget_d=4.0)
        assert lam == pytest.approx(1.5)
        assert pid.prev_delta == pytest.approx(0.0)

    def test_violation_raises_lambda(self):
        pid = PIDLagrangian(k_p=0.1, k_i=0.1)
        lam0 = pid.lambda_
        lam1 = pid.update(j_cost=10.0, budget_d=4.0)  # Δ = +6
        assert lam1 > lam0

    def test_lambda_is_nonnegative(self):
        pid = PIDLagrangian(k_p=0.5, k_i=0.5, lambda_init=0.2)
        # Hammer with constraint satisfaction (Δ < 0) — λ must floor at 0.
        for _ in range(100):
            lam = pid.update(j_cost=0.0, budget_d=5.0)
            assert lam >= 0.0
        assert pid.lambda_ == pytest.approx(0.0)

    def test_returns_stored_lambda(self):
        pid = PIDLagrangian(k_p=0.1, k_i=0.1)
        lam = pid.update(j_cost=8.0, budget_d=2.0)
        assert lam == pytest.approx(pid.lambda_)


# ── Integral anti-windup ────────────────────────────────────────────────────


class TestAntiWindup:
    def test_integral_never_negative(self):
        pid = PIDLagrangian(k_i=0.3)
        for _ in range(50):
            pid.update(j_cost=0.0, budget_d=10.0)  # large negative Δ each call
            assert pid.integral >= 0.0

    def test_integral_recovers_quickly_after_windup(self):
        """A satisfied-then-violated sequence must respond without lag.

        Because the integral is clamped at 0 (not allowed to go deeply
        negative while the constraint is satisfied), a later violation
        immediately starts re-accumulating from 0.
        """
        pid = PIDLagrangian(k_p=0.0, k_i=0.2)
        for _ in range(20):
            pid.update(j_cost=0.0, budget_d=5.0)  # satisfied — integral pinned 0
        assert pid.integral == pytest.approx(0.0)

        lam = pid.update(j_cost=10.0, budget_d=5.0)  # Δ = +5
        # Pure-integral: λ = max(0, 0 + k_i·Δ) = 0.2 · 5 = 1.0
        assert lam == pytest.approx(1.0)


# ── Equivalence to vanilla Lagrangian (pure integral) ───────────────────────


class TestPureIntegral:
    def test_matches_vanilla_lagrangian_recursion(self):
        """With k_p = k_d = 0, the update is λ ← max(0, λ + k_i·Δ)."""
        k_i = 0.05
        pid = PIDLagrangian(k_p=0.0, k_i=k_i, k_d=0.0, lambda_init=0.0)

        ref_lambda = 0.0
        costs = [9.0, 7.0, 6.5, 5.0, 4.0, 4.5]
        d = 4.0
        for c in costs:
            ref_lambda = max(0.0, ref_lambda + k_i * (c - d))
            pid.update(j_cost=c, budget_d=d)
            assert pid.lambda_ == pytest.approx(ref_lambda, abs=1e-9)


# ── Derivative term ─────────────────────────────────────────────────────────


class TestDerivative:
    def test_derivative_adds_on_rising_violation(self):
        """A K_D > 0 controller reacts more strongly when Δ jumps up."""
        d = 4.0
        no_d = PIDLagrangian(k_p=0.1, k_i=0.0, k_d=0.0)
        with_d = PIDLagrangian(k_p=0.1, k_i=0.0, k_d=0.1)

        # First step establishes prev_delta.
        no_d.update(j_cost=5.0, budget_d=d)    # Δ = 1
        with_d.update(j_cost=5.0, budget_d=d)  # Δ = 1
        # Second step: Δ jumps to 6 (rising) → derivative term lifts λ.
        lam_no = no_d.update(j_cost=10.0, budget_d=d)
        lam_with = with_d.update(j_cost=10.0, budget_d=d)
        assert lam_with > lam_no


# ── lambda_max clamp ────────────────────────────────────────────────────────


class TestLambdaMax:
    def test_upper_clamp(self):
        pid = PIDLagrangian(k_p=1.0, k_i=1.0, lambda_max=2.0)
        for _ in range(50):
            lam = pid.update(j_cost=100.0, budget_d=0.0)
            assert lam <= 2.0 + 1e-12
        assert pid.lambda_ == pytest.approx(2.0)


# ── Closed-loop convergence ─────────────────────────────────────────────────


class TestConvergence:
    def test_drives_linear_plant_to_budget(self):
        """Closed-loop check: a monotone plant J(λ) = c0 − g·λ (clamped ≥ 0)
        should settle at the budget d, with λ → λ* = (c0 − d)/g.

        This is the smoke-level correctness behaviour referenced by G1.4.
        """
        c0, g, d = 10.0, 2.0, 4.0
        lam_star = (c0 - d) / g  # = 3.0

        pid = PIDLagrangian(k_p=0.02, k_i=0.02, k_d=0.0)
        cost = c0
        for _ in range(5000):
            pid.update(j_cost=cost, budget_d=d)
            cost = max(0.0, c0 - g * pid.lambda_)

        assert cost == pytest.approx(d, abs=0.1)
        assert pid.lambda_ == pytest.approx(lam_star, abs=0.1)

    def test_loose_budget_drives_lambda_to_zero(self):
        """If the unconstrained plant already satisfies the budget, λ → 0
        (the constraint is inactive — energy is optimised freely)."""
        c0, g, d = 3.0, 2.0, 5.0  # plant min cost 3 < budget 5 ⇒ slack
        pid = PIDLagrangian(k_p=0.05, k_i=0.05, lambda_init=1.0)
        cost = c0
        for _ in range(2000):
            pid.update(j_cost=cost, budget_d=d)
            cost = max(0.0, c0 - g * pid.lambda_)
        assert pid.lambda_ == pytest.approx(0.0, abs=1e-6)


# ── Checkpoint round-trip ───────────────────────────────────────────────────


class TestStateDict:
    def test_round_trip(self):
        pid = PIDLagrangian(k_p=0.1, k_i=0.1, k_d=0.05)
        for c in (8.0, 6.0, 7.0):
            pid.update(j_cost=c, budget_d=4.0)
        snap = pid.state_dict()

        restored = PIDLagrangian(k_p=0.1, k_i=0.1, k_d=0.05)
        restored.load_state_dict(snap)
        assert restored.lambda_ == pytest.approx(pid.lambda_)
        assert restored.integral == pytest.approx(pid.integral)
        assert restored.prev_delta == pytest.approx(pid.prev_delta)

        # Continuing from the restored state matches the original trajectory.
        a = pid.update(j_cost=5.0, budget_d=4.0)
        b = restored.update(j_cost=5.0, budget_d=4.0)
        assert a == pytest.approx(b)
