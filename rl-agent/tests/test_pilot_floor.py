"""W6.1 — tests for the budget grid the whole campaign is placed on.

The grid is the single most expensive thing to get wrong in W6: a bad one does not
crash, it produces five policies that are all the same policy and a "Pareto front"
that is one point drawn five times (§14.3). These lock the properties that make that
impossible.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

SRC = Path(__file__).resolve().parent.parent / "src"
sys.path.insert(0, str(SRC))


def _load():
    """Import pilot_floor directly.

    It must import without the RL stack present — sb3 is pulled in inside the one
    function that trains — so this doubles as a guard against someone hoisting that
    import back to module scope and quietly making `--print-grid` unusable offline.
    """
    spec = importlib.util.spec_from_file_location(
        "pilot_floor_under_test", SRC / "eval" / "pilot_floor.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


pf = _load()


class TestSuggestGrid:
    def test_every_budget_binds(self):
        """Each budget must sit strictly below the unconstrained cost.

        A budget at or above J_free is slack: λ stays 0, the run reproduces the
        unconstrained policy, and the point is a duplicate.
        """
        j_free = 0.183
        grid = pf.suggest_grid(j_free)
        assert grid, "a positive J_free must yield a grid"
        assert all(0 < d < j_free for d in grid), grid

    def test_strictly_increasing(self):
        grid = pf.suggest_grid(0.42)
        assert grid == sorted(grid)
        assert len(set(grid)) == len(grid), "duplicate budgets waste a whole run"

    def test_scales_with_j_free(self):
        """Doubling the operating point doubles the grid: it is a pure fraction."""
        a = pf.suggest_grid(0.1)
        b = pf.suggest_grid(0.2)
        assert all(y == pytest.approx(2 * x, rel=1e-3) for x, y in zip(a, b))

    def test_reaches_well_below_the_operating_point(self):
        """The tight end must be far enough down to make λ grow.

        A grid hugging J_free (say 0.95..0.99) would bind so weakly that all five
        policies land on top of each other — the collapse in a different disguise.
        """
        grid = pf.suggest_grid(1.0)
        assert min(grid) <= 0.6

    @pytest.mark.parametrize("bad", [None, 0.0, -0.1, float("nan")])
    def test_unusable_j_free_yields_no_grid(self, bad):
        """No grid beats a fabricated one: the caller skips the scenario loudly."""
        assert pf.suggest_grid(bad) == []

    def test_respects_requested_size(self):
        assert len(pf.suggest_grid(0.2, n=3)) == 3
        assert len(pf.suggest_grid(0.2, n=5)) == 5

    def test_more_points_than_named_fractions_still_spans(self):
        grid = pf.suggest_grid(1.0, n=8)
        assert len(grid) == 8
        assert grid == sorted(grid)
        assert all(0 < d < 1.0 for d in grid)
