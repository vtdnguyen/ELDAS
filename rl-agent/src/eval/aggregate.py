"""Seed aggregation (mean ± 95 % CI) and Pareto-front monotonicity checks.

Used by the budget sweep (G2.5) and the campaign table (G2.6).

* **≥5 seeds + CI** is mandatory for every reported number (CLAUDE.md Lưu ý #10)
  — ``mean_ci95`` returns the half-width so callers can print ``mean ± ci``.
* **Front monotonicity** (Lưu ý #7): sweeping the SLA budget ``d`` from tight to
  loose must trace a monotone front — a tighter budget buys lower SLA cost at
  the price of higher energy. A non-monotone point means that budget's policy
  has not converged; it must be retrained, not plotted.

Self-contained (NumPy only): the Student-t critical values are a small table so
the CI does not depend on SciPy being importable.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# Two-sided 95 % Student-t critical values, indexed by degrees of freedom.
# df ≥ 30 is well approximated by the normal value 1.96.
_T95 = {
    1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365,
    8: 2.306, 9: 2.262, 10: 2.228, 11: 2.201, 12: 2.179, 13: 2.160, 14: 2.145,
    15: 2.131, 16: 2.120, 17: 2.110, 18: 2.101, 19: 2.093, 20: 2.086,
    21: 2.080, 22: 2.074, 23: 2.069, 24: 2.064, 25: 2.060, 26: 2.056,
    27: 2.052, 28: 2.048, 29: 2.045,
}
_T95_LARGE = 1.96


def t_critical_95(df: int) -> float:
    """Two-sided 95 % t critical value for ``df`` degrees of freedom."""
    if df <= 0:
        return float("nan")
    return _T95.get(df, _T95_LARGE)


@dataclass(frozen=True, slots=True)
class MeanCI:
    mean: float
    ci95: float      # half-width; report as mean ± ci95
    std: float
    n: int

    def as_tuple(self) -> tuple[float, float]:
        return (self.mean, self.ci95)


def mean_ci95(values) -> MeanCI:
    """Mean and 95 % CI half-width (Student-t, ddof=1) over seeds.

    A single sample has no CI (``ci95 = nan``) — that is deliberate: it makes
    "1 seed" visibly unreportable rather than silently showing ±0 (Lưu ý #10).
    """
    v = np.asarray(list(values), dtype=np.float64)
    n = v.size
    if n == 0:
        return MeanCI(float("nan"), float("nan"), float("nan"), 0)
    mean = float(v.mean())
    if n == 1:
        return MeanCI(mean, float("nan"), 0.0, 1)
    std = float(v.std(ddof=1))
    sem = std / np.sqrt(n)
    return MeanCI(mean, float(t_critical_95(n - 1) * sem), std, n)


# ── Per-method aggregation across seeds ─────────────────────────────────────

def aggregate_by_method(points, scenario: str | None = None) -> dict[str, dict]:
    """Aggregate point records into per-method mean ± CI on both objectives.

    Returns ``{method: {"energy": MeanCI, "sla": MeanCI, "seeds": [...]}}``.
    """
    groups: dict[str, list] = {}
    for p in points:
        if scenario is not None and p.scenario != scenario:
            continue
        groups.setdefault(p.method, []).append(p)

    out: dict[str, dict] = {}
    for method, recs in groups.items():
        # W3/W6.1 — mean tasks dropped, when the producer recorded it. Kept as None
        # (not 0) when absent: "no drops" and "nobody measured drops" must not print
        # the same, or an old points file silently claims a clean run.
        drops = [float(r.extra["dropped_tasks"]) for r in recs
                 if r.extra and r.extra.get("dropped_tasks") is not None]
        out[method] = {
            "energy": mean_ci95([r.energy_kwh for r in recs]),
            "sla": mean_ci95([r.sla_cost for r in recs]),
            "seeds": sorted({r.seed for r in recs}),
            "n": len(recs),
            "dropped": (sum(drops) / len(drops)) if drops else None,
        }
    return out


# ── Front monotonicity (Lưu ý #7) ───────────────────────────────────────────

@dataclass(frozen=True, slots=True)
class MonotonicityReport:
    monotone: bool
    violations: list[tuple[float, float]]   # (budget_i, budget_j) offending pairs
    detail: list[str]

    def summary(self) -> str:
        if self.monotone:
            return "front is monotone in d (tighter d ⇒ lower SLA cost, higher energy)"
        return (f"NON-MONOTONE at {len(self.violations)} pair(s): "
                + "; ".join(self.detail))


def check_budget_front_monotone(
    budget_points: list[tuple[float, float, float]],
    energy_tol: float = 0.0,
    sla_tol: float = 0.0,
) -> MonotonicityReport:
    """Verify the d-sweep traces a monotone energy↔SLA front.

    Parameters
    ----------
    budget_points : list of ``(d, energy_kwh, sla_cost)``
    energy_tol, sla_tol : slack (absolute) tolerated before flagging.

    Expectation (Lưu ý #7): as ``d`` **increases** (looser SLA budget) the
    converged policy should be allowed *more* SLA cost and spend *less* energy.
    So sorting by d ascending, ``sla_cost`` must be non-decreasing and
    ``energy_kwh`` non-increasing. Any inversion beyond tolerance ⇒ that budget's
    policy likely has not converged and must be retrained before it is plotted.
    """
    pts = sorted(budget_points, key=lambda t: t[0])
    violations: list[tuple[float, float]] = []
    detail: list[str] = []

    for i in range(len(pts) - 1):
        d_i, e_i, s_i = pts[i]
        d_j, e_j, s_j = pts[i + 1]
        if s_j < s_i - sla_tol:
            violations.append((d_i, d_j))
            detail.append(
                f"d={d_i:g}→{d_j:g}: SLA cost fell {s_i:.4g}→{s_j:.4g} "
                f"(looser budget should not lower SLA cost)"
            )
        if e_j > e_i + energy_tol:
            violations.append((d_i, d_j))
            detail.append(
                f"d={d_i:g}→{d_j:g}: energy rose {e_i:.4g}→{e_j:.4g} "
                f"(looser budget should not raise energy)"
            )

    return MonotonicityReport(
        monotone=not violations, violations=violations, detail=detail
    )
