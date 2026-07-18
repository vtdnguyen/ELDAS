"""
G1.5 (mathematical validation) — verify the PID-Lagrangian dual realises the
Constrained-MDP KKT / optimality conditions.

Theory (Paternain et al., 2019, "Constrained RL has Zero Duality Gap"):
for the CMDP

    max_π  E[Σ R_energy]   s.t.   J(π) := E[C_SLA] ≤ d ,

strong duality holds, so the saddle point (π*, λ*) of the Lagrangian
``L(π,λ)=E[ΣR_energy] − λ(J(π)−d)`` is **Pareto-optimal**, and at the optimum
the KKT conditions hold:

    (i)   dual feasibility        λ* ≥ 0
    (ii)  primal feasibility      J(π*) ≤ d
    (iii) complementary slackness λ*·(J(π*) − d) = 0
          ⇒ either λ*=0 (constraint slack) or J(π*)=d (constraint active).

The PID-Lagrangian is the dual ascent that *reaches* this saddle point.  We
cannot run a full RL optimum here, so we validate against a **monotone cost
plant** that stands in for the inner ``max_π`` solution as a function of the
penalty: a larger penalty λ buys a lower achievable cost,

    J(λ) = max(j_floor, c0 − g·λ),        c0 > j_floor ≥ 0, g > 0,

where ``c0`` is the cost of the unconstrained (energy-only, λ=0) policy and
``j_floor`` is the cost of the most SLA-friendly policy (λ→∞).  Driving the PID
controller to convergence and sweeping the budget ``d`` must reproduce the KKT
structure exactly, and the resulting ``(J*(d), λ*(d))`` must be **monotone** —
the parametric Pareto front the budget sweep traces in G2.5 (CLAUDE.md Lưu ý #7).

This is deterministic and fast; it is the formal correctness gate for GĐ 2.1's
dual mechanism, complementing ``validate_lambda_vs_omnisafe.py`` (which checks
the update *law*) and the real-system convergence run (which checks the
end-to-end pipeline).
"""

from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from optim.pid_lagrangian import PIDLagrangian


def solve_dual(c0, g, j_floor, d, k_p, k_i, steps=20000):
    """Run PID-Lagrangian to convergence on the plant J(λ)=max(j_floor,c0−g·λ).

    Returns the converged ``(J*, λ*)``.
    """
    pid = PIDLagrangian(k_p=k_p, k_i=k_i, k_d=0.0, lambda_init=0.0)
    cost = c0
    for _ in range(steps):
        lam = pid.update(cost, d)
        cost = max(j_floor, c0 - g * lam)
    return cost, pid.lambda_


def main() -> int:
    # Plant: unconstrained cost c0=10, SLA-floor j_floor=2, slope g=2.
    c0, g, j_floor = 10.0, 2.0, 2.0
    # Gains tuned to the plant scale (cost ~O(10)); see G1.5 note on gain scaling.
    k_p, k_i = 0.05, 0.05
    tol = 0.05

    print("=" * 70)
    print("  G1.5 - CMDP KKT / zero-duality-gap validation (deterministic plant)")
    print("=" * 70)
    print(f"  plant J(lam)=max({j_floor}, {c0}-{g}*lam)   gains K_P=K_I={k_i}")
    print(f"  feasible region: d >= j_floor={j_floor};  binding: {j_floor} <= d < c0={c0}")
    print("-" * 70)
    print(f"  {'d':>6} | {'J*':>8} {'lam*':>8} | {'regime':<10} | KKT checks")
    print("-" * 70)

    budgets = [12.0, 10.0, 8.0, 6.0, 4.0, 2.0, 1.0]  # loose → binding → infeasible
    rows = []
    all_ok = True
    for d in budgets:
        j_star, lam_star = solve_dual(c0, g, j_floor, d, k_p, k_i)
        feasible = d >= j_floor - 1e-9

        # KKT condition checks.
        dual_feas = lam_star >= -1e-9                       # (i)
        prim_feas = (j_star <= d + tol) if feasible else (j_star <= c0 + tol)  # (ii)
        compl_slack = abs(lam_star * (j_star - d)) < tol * max(1.0, abs(c0))    # (iii)

        if d >= c0:
            regime = "SLACK"      # λ*=0 expected, J*=c0 ≤ d
            expect_ok = (lam_star < tol) and (abs(j_star - c0) < tol)
        elif feasible:
            regime = "ACTIVE"     # λ*=(c0−d)/g, J*=d
            lam_analytic = (c0 - d) / g
            expect_ok = (abs(j_star - d) < tol) and (abs(lam_star - lam_analytic) < 10 * tol)
        else:
            regime = "INFEAS"     # λ grows; J*→floor > d; (iii) cannot hold
            expect_ok = (j_star <= j_floor + tol) and (lam_star > 0)
            compl_slack = True  # not applicable when infeasible — don't fail on it

        ok = dual_feas and prim_feas and compl_slack and expect_ok
        all_ok &= ok
        rows.append((d, j_star, lam_star, regime))
        checks = (f"dual={'Y' if dual_feas else 'N'} "
                  f"prim={'Y' if prim_feas else 'N'} "
                  f"compl={'Y' if compl_slack else 'N'} "
                  f"regime={'Y' if expect_ok else 'N'}")
        print(f"  {d:>6.2f} | {j_star:>8.4f} {lam_star:>8.4f} | {regime:<10} | "
              f"{checks}  {'OK' if ok else 'FAIL'}")

    # ── Monotonicity of the parametric Pareto front (Lưu ý #7) ─────────────
    # Restrict to the feasible budgets; tighter d ⇒ higher λ*, lower J*.
    feas = [(d, j, l) for (d, j, l, r) in rows if d >= j_floor - 1e-9]
    feas.sort(key=lambda t: t[0])  # ascending d
    js = [j for (_, j, _) in feas]
    ls = [l for (_, _, l) in feas]
    # ascending d ⇒ J* non-decreasing and λ* non-increasing.
    j_mono = all(js[i] <= js[i + 1] + tol for i in range(len(js) - 1))
    l_mono = all(ls[i] >= ls[i + 1] - tol for i in range(len(ls) - 1))

    print("-" * 70)
    print(f"  Pareto monotonicity (feasible d): "
          f"J*(d) non-decreasing={'Y' if j_mono else 'N'}, "
          f"lam*(d) non-increasing={'Y' if l_mono else 'N'}")
    mono_ok = j_mono and l_mono
    all_ok &= mono_ok

    print("=" * 70)
    verdict = ("PASS - dual mechanism realises CMDP KKT + monotone Pareto front"
               if all_ok else "FAIL - KKT/monotonicity violated, investigate")
    print(f"  VERDICT: {verdict}")
    print("=" * 70)
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
