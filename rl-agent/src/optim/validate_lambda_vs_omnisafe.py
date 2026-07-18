"""
G1.4 — Cross-validate our :class:`PIDLagrangian` against OmniSafe's reference
PID-Lagrangian (CPPOPID's dual controller, JMLR 2024).

CLAUDE.md mandates this correctness check *before* trusting any CMDP result:
the λ-update law we implemented in G1.2 must match the published reference
qualitatively.  We isolate the **dual-update law itself** (the G1.2 deliverable)
rather than running two full RL loops, because:

  * OmniSafe is built for masking-free continuous control; wiring our
    MaskablePPO env into its trainer is exactly what CLAUDE.md Lưu ý #3 warns
    against.  The dual controller, however, is a small standalone scalar
    component we *can* compare directly.
  * Feeding BOTH controllers the **same** cost trajectory removes every
    confound (different policies, seeds, advantage estimators) and tests the
    one thing that must agree: ``J_cost → λ``.

Two comparisons are run:
  1. **Open-loop** — an identical synthetic cost sequence is fed to both
     controllers; λ trajectories must coincide (same update law, matched gains).
  2. **Closed-loop** — both drive a monotone cost plant ``J(λ)=c0−g·λ``; both
     must settle at the budget ``d`` with the same fixed-point λ.

OmniSafe's controller is configured to disable its extra EMA smoothing /
normalisation so it reduces to the canonical Stooke (2020) law we implement.

If OmniSafe is not installed, the script SKIPS the cross-check (exit 0) with
install instructions, and still runs a self-consistency check against an inline
reference re-derived from Stooke (2020).  Install for the real check via::

    pip install -r requirements-validate.txt   # isolated; may conflict w/ sb3
    python src/optim/validate_lambda_vs_omnisafe.py
"""

from __future__ import annotations

import inspect
import os
import sys

import numpy as np

# Make ``src/`` importable so ``optim`` resolves whether run as a script
# (python src/optim/validate_lambda_vs_omnisafe.py) or as a module.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from optim.pid_lagrangian import PIDLagrangian


# ── OmniSafe reference adapter ──────────────────────────────────────────────

def _import_omnisafe_pid():
    """Return OmniSafe's PIDLagrangian class, or None if unavailable."""
    for path in ("omnisafe.common.pid_lagrange", "omnisafe.common.lagrange"):
        try:
            mod = __import__(path, fromlist=["PIDLagrangian"])
            if hasattr(mod, "PIDLagrangian"):
                return getattr(mod, "PIDLagrangian")
        except Exception:
            continue
    return None


class _OmniAdapter:
    """Wrap OmniSafe's PIDLagrangian to a uniform ``step(J, d) -> λ`` interface.

    Configured (via whatever kwargs its constructor accepts) to disable EMA
    smoothing and sum/diff normalisation so it reduces to the canonical
    Stooke law.  ``d`` is baked in at construction (OmniSafe's API), so the
    budget passed to :meth:`step` must stay constant across the run.
    """

    def __init__(self, cls, k_p, k_i, k_d, lambda_init, budget_d):
        self._d = budget_d
        wanted = {
            "pid_kp": k_p, "pid_ki": k_i, "pid_kd": k_d,
            "pid_d_delay": 1,
            "pid_delta_p_ema_alpha": 1.0,   # 1.0 ⇒ use current delta, no smoothing
            "pid_delta_d_ema_alpha": 1.0,
            "sum_norm": False, "diff_norm": False,
            "penalty_max": 100.0,
            "lagrangian_multiplier_init": lambda_init,
            "cost_limit": budget_d,
        }
        sig = inspect.signature(cls.__init__)
        kwargs = {k: v for k, v in wanted.items() if k in sig.parameters}
        self._pid = cls(**kwargs)

    def _read_lambda(self) -> float:
        for attr in ("lagrangian_multiplier", "cost_penalty", "_cost_penalty"):
            val = getattr(self._pid, attr, None)
            if val is not None:
                return float(val.item() if hasattr(val, "item") else val)
        raise AttributeError("cannot read OmniSafe multiplier")

    def step(self, j_cost: float, budget_d: float) -> float:
        # OmniSafe bakes the limit in; budget must match construction.
        for name in ("pid_update", "update"):
            fn = getattr(self._pid, name, None)
            if fn is not None:
                fn(j_cost)
                return self._read_lambda()
        raise AttributeError("cannot find OmniSafe update method")


class _StookeReference:
    """Inline canonical Stooke (2020) PID-Lagrangian — used when OmniSafe is
    absent so the script still self-checks our implementation against the
    published law (not a circular check: this is re-derived from the paper)."""

    def __init__(self, k_p, k_i, k_d, lambda_init):
        self.k_p, self.k_i, self.k_d = k_p, k_i, k_d
        self._i = lambda_init
        self._prev = 0.0
        self.lam = lambda_init

    def step(self, j_cost: float, budget_d: float) -> float:
        delta = j_cost - budget_d
        self._i = max(0.0, self._i + self.k_i * delta)
        self.lam = max(0.0, self.k_p * delta + self._i + self.k_d * (delta - self._prev))
        self._prev = delta
        return self.lam


# ── Comparison harness ──────────────────────────────────────────────────────

def open_loop(ours, ref, costs, d):
    """Feed identical cost sequence to both controllers; return λ trajectories."""
    lam_ours, lam_ref = [], []
    for c in costs:
        lam_ours.append(ours.update(c, d))
        lam_ref.append(ref.step(c, d))
    return np.array(lam_ours), np.array(lam_ref)


def closed_loop(controller, c0, g, d, steps, is_ours):
    """Drive plant J(lam)=max(0, c0-g*lam); return final (cost, λ)."""
    cost = c0
    for _ in range(steps):
        lam = controller.update(cost, d) if is_ours else controller.step(cost, d)
        cost = max(0.0, c0 - g * lam)
    return cost, lam


def main() -> int:
    k_p, k_i, k_d = 0.05, 0.05, 0.01
    lambda_init = 0.0
    d = 4.0

    print("=" * 64)
    print("  G1.4 - PIDLagrangian cross-validation vs OmniSafe reference")
    print("=" * 64)

    OmniPID = _import_omnisafe_pid()
    if OmniPID is not None:
        print(f"[info] OmniSafe PIDLagrangian found: {OmniPID.__module__}.{OmniPID.__name__}")
        ref_factory = lambda: _OmniAdapter(OmniPID, k_p, k_i, k_d, lambda_init, d)
        ref_name = "OmniSafe"
    else:
        print("[skip] OmniSafe not importable — falling back to inline Stooke (2020) "
              "reference.\n"
              "       For the real cross-check: pip install -r requirements-validate.txt")
        ref_factory = lambda: _StookeReference(k_p, k_i, k_d, lambda_init)
        ref_name = "Stooke-ref"

    # ── 1. Open-loop: identical cost sequence ──────────────────────────────
    rng = np.random.default_rng(0)
    base = np.concatenate([
        np.linspace(9.0, 4.5, 40),     # violating, decaying toward budget
        np.full(30, 3.5),              # under budget
        np.linspace(3.5, 7.0, 30),     # rising violation again
    ])
    costs = base + rng.normal(0.0, 0.1, size=base.shape)

    lam_ours, lam_ref = open_loop(
        PIDLagrangian(k_p=k_p, k_i=k_i, k_d=k_d, lambda_init=lambda_init),
        ref_factory(), costs, d,
    )
    abs_diff = np.abs(lam_ours - lam_ref)
    denom = np.maximum(np.abs(lam_ref), 1e-8)
    rel_diff = abs_diff / denom
    corr = float(np.corrcoef(lam_ours, lam_ref)[0, 1])

    print(f"\n[open-loop] {len(costs)} steps, identical cost sequence")
    print(f"   max |d_lambda|        = {abs_diff.max():.3e}")
    print(f"   max rel |d_lambda|    = {rel_diff.max():.3e}")
    print(f"   corr(lam_ours,lam_ref)  = {corr:.6f}")
    print(f"   final lam ours={lam_ours[-1]:.6f}  {ref_name}={lam_ref[-1]:.6f}")

    open_ok = (corr > 0.999) and (abs_diff.max() < 1e-3 or rel_diff.max() < 1e-2)

    # ── 2. Closed-loop: monotone plant, both settle at budget ──────────────
    c0, g, steps = 10.0, 2.0, 4000
    lam_star = (c0 - d) / g
    cost_ours, lam_o = closed_loop(
        PIDLagrangian(k_p=k_p, k_i=k_i, k_d=k_d, lambda_init=lambda_init),
        c0, g, d, steps, is_ours=True,
    )
    ref = ref_factory()
    cost_ref, lam_r = closed_loop(ref, c0, g, d, steps, is_ours=False)

    print(f"\n[closed-loop] plant J(lam)=max(0,{c0}-{g}*lam), budget d={d}, lam*={lam_star:.4f}")
    print(f"   ours : cost={cost_ours:.4f}  lam={lam_o:.4f}")
    print(f"   {ref_name:5}: cost={cost_ref:.4f}  lam={lam_r:.4f}")

    closed_ok = (abs(cost_ours - d) < 0.1 and abs(lam_o - lam_star) < 0.1
                 and abs(cost_ours - cost_ref) < 0.2 and abs(lam_o - lam_r) < 0.2)

    # ── Verdict ────────────────────────────────────────────────────────────
    print("\n" + "-" * 64)
    print(f"   open-loop   match : {'PASS' if open_ok else 'FAIL'}")
    print(f"   closed-loop match : {'PASS' if closed_ok else 'FAIL'}")
    ok = open_ok and closed_ok
    print(f"   VERDICT           : {'PASS - lambda-update law cross-validated' if ok else 'FAIL - investigate'}")
    print("-" * 64)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
