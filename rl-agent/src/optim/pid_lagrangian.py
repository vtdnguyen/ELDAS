"""
G1.2 — PID-Lagrangian dual update for the Constrained-MDP core.

The Phase-2 problem is a Constrained MDP:

    maximise   E[ Σ R_energy ]
    subject to E[ C_SLA ] ≤ d

solved by two-timescale primal-dual: ``MaskablePPO`` optimises the policy
against the effective reward ``R_energy − λ·C_SLA`` (primal), while this class
updates the Lagrange multiplier ``λ ≥ 0`` (dual).  Plain Lagrangian ascent
(``λ ← max(0, λ + η·(J_cost − d))``) is just the integral term of a PID
controller and tends to overshoot/oscillate.  Following Stooke, Achiam &
Abbeel (2020), *"Responsive Safety in RL by PID Lagrangian Methods"*, we add
proportional and derivative terms that damp those oscillations:

    Δ        = J_cost − d                       (constraint violation)
    I        ← max(0, I + K_I·Δ)                (integral, anti-windup clamp)
    λ        = max(0, K_P·Δ + I + K_D·(Δ − Δ_prev))
    Δ_prev   ← Δ

* The integral term ``I`` is the classical dual ascent (it alone recovers
  vanilla Lagrangian).  ``K_P`` and ``K_D`` only shape the transient.
* ``λ = max(0, ·)`` enforces the dual feasibility constraint λ ≥ 0
  (CLAUDE.md Lưu ý #4).
* The integral is clamped to ≥ 0 (anti-windup): with a one-sided inequality
  constraint a persistently-satisfied constraint must not accumulate negative
  integral that would later delay the response to a violation.  This matches
  Stooke (2020) and the OmniSafe ``PIDLagrangian`` reference used for the G1.4
  cross-validation.

Defaults ``K_P = K_I = 1e-4``, ``K_D = 0`` follow Stooke (2020) / Spoor (2026).
``λ`` is seeded from ``lambda_init`` (the integral is initialised to the same
value so that ``update`` with ``Δ = 0`` reproduces ``lambda_init``).

The class is a pure, framework-free scalar controller: one ``update`` call per
dual step.  Call it on a *slower* timescale than the policy (e.g. once per N
PPO rollouts) so the primal sees a quasi-static λ — a correctness condition of
primal-dual convergence, not an option (CLAUDE.md Lưu ý #6).
"""

from __future__ import annotations


class PIDLagrangian:
    """PID controller for the Lagrange multiplier λ of a single constraint.

    Parameters
    ----------
    k_p, k_i, k_d : float
        Proportional / integral / derivative gains.  Defaults
        ``k_p = k_i = 1e-4``, ``k_d = 0.0`` (Stooke 2020).  All must be ≥ 0.
    lambda_init : float
        Initial multiplier (also seeds the integral term).  Must be ≥ 0.
    lambda_max : float or None
        Optional upper clamp on λ to bound the effective penalty during early,
        noisy cost estimates.  ``None`` (default) leaves λ unbounded above.

    Attributes
    ----------
    lambda_ : float
        Current multiplier — read this each dual step to form the effective
        reward ``R_energy − lambda_·C_SLA``.
    """

    def __init__(
        self,
        k_p: float = 1e-4,
        k_i: float = 1e-4,
        k_d: float = 0.0,
        lambda_init: float = 0.0,
        lambda_max: float | None = None,
    ) -> None:
        if k_p < 0 or k_i < 0 or k_d < 0:
            raise ValueError(
                f"PID gains must be ≥ 0, got k_p={k_p}, k_i={k_i}, k_d={k_d}"
            )
        if lambda_init < 0:
            raise ValueError(f"lambda_init must be ≥ 0, got {lambda_init}")
        if lambda_max is not None and lambda_max < lambda_init:
            raise ValueError(
                f"lambda_max ({lambda_max}) must be ≥ lambda_init ({lambda_init})"
            )

        self.k_p = float(k_p)
        self.k_i = float(k_i)
        self.k_d = float(k_d)
        self.lambda_max = float(lambda_max) if lambda_max is not None else None

        # Integral seeded with λ_0 so that update(J_cost=d) is a no-op at start.
        self._integral = float(lambda_init)
        self._prev_delta = 0.0
        self.lambda_ = float(lambda_init)

    # ── Dual update ────────────────────────────────────────────────────────

    def update(self, j_cost: float, budget_d: float) -> float:
        """Run one PID step and return the updated multiplier λ.

        Parameters
        ----------
        j_cost : float
            Measured expected constraint cost ``E[C_SLA]`` (e.g. the mean
            episodic ``total_sla_cost`` over the last batch of rollouts).
        budget_d : float
            The SLA budget ``d`` (same units as ``j_cost``).  Sweeping ``d``
            from tight → loose traces the Pareto front (CLAUDE.md G2.5).

        Returns
        -------
        float
            The new ``λ ≥ 0`` (also stored in ``self.lambda_``).
        """
        delta = float(j_cost) - float(budget_d)

        # Integral term with anti-windup clamp (Stooke 2020).
        self._integral = max(0.0, self._integral + self.k_i * delta)

        derivative = delta - self._prev_delta
        raw = self.k_p * delta + self._integral + self.k_d * derivative

        new_lambda = max(0.0, raw)
        if self.lambda_max is not None:
            new_lambda = min(new_lambda, self.lambda_max)

        self.lambda_ = new_lambda
        self._prev_delta = delta
        return self.lambda_

    # ── Introspection ──────────────────────────────────────────────────────

    @property
    def integral(self) -> float:
        """Current integral accumulator (the vanilla-Lagrangian component)."""
        return self._integral

    @property
    def prev_delta(self) -> float:
        """Constraint violation Δ from the previous ``update`` call."""
        return self._prev_delta

    def state_dict(self) -> dict[str, float]:
        """Serialisable controller state (for checkpoint/restore)."""
        return {
            "lambda_": self.lambda_,
            "integral": self._integral,
            "prev_delta": self._prev_delta,
        }

    def load_state_dict(self, state: dict[str, float]) -> None:
        """Restore controller state produced by :meth:`state_dict`."""
        self.lambda_ = float(state["lambda_"])
        self._integral = float(state["integral"])
        self._prev_delta = float(state["prev_delta"])

    def __repr__(self) -> str:
        return (
            f"PIDLagrangian(k_p={self.k_p}, k_i={self.k_i}, k_d={self.k_d}, "
            f"lambda_={self.lambda_:.6g}, integral={self._integral:.6g})"
        )
