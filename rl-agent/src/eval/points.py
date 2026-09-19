"""Canonical Pareto-point schema shared by the sweep (G2.5) and campaign (G2.6).

A *point* is one method's outcome for one (scenario, seed): the two minimised
objectives ``energy_kwh`` and ``sla_cost`` (= C_SLA, the same quantity the CMDP
budget ``d`` constrains and the NSGA-II static model reports). Keeping a single
schema means every producer (heuristics, fixed-weight PPO, CMDP sweep, NSGA-II)
feeds the same metric/plot code with the same units (Lưu ý #5/#15).

Stored as JSON Lines (one record per line) so runs can be appended
incrementally without rewriting the whole file.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True, slots=True)
class PointRecord:
    method: str          # e.g. "bestfit", "ppo-min", "cmdp-d0.04", "nsga2"
    scenario: str        # LOW | HIGH | BURST
    seed: int
    energy_kwh: float    # objective 0 (min)
    sla_cost: float      # objective 1 (min) = C_SLA
    extra: dict | None = None   # optional: budget d, viol_rate, wakeups, ...

    def objectives(self) -> tuple[float, float]:
        return (self.energy_kwh, self.sla_cost)


def save_points(points: list[PointRecord], path: str | Path, *,
                merge: bool = False) -> None:
    """Write points as JSON Lines (creates parent dirs).

    ``merge=False`` (default) replaces the file with exactly ``points``.

    ``merge=True`` folds ``points`` into whatever is already there, keyed by
    ``(method, seed)``. Use it whenever a run produces a **subset** of the seeds the
    file is meant to hold — extending a campaign with more seeds, or re-running a few
    that failed. Replacing rather than appending keeps it idempotent: re-running the
    same (method, seed) overwrites its row instead of adding a duplicate that would
    silently double-weight that seed in every mean and confidence interval.

    This parameter exists because its absence cost real data. The docstring used to say
    "Append/write" while the code opened the file ``"w"``, and a run that added seeds
    47–56 to a sweep **erased** seeds 42–46 — no error, no warning, and the resulting
    table looked entirely normal apart from the learned methods quietly resting on ten
    seeds while the baselines rested on fifteen.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    rows = [asdict(p) for p in points]
    if merge and path.exists():
        by_key: dict[tuple, dict] = {}
        for existing in load_points(path):
            d = asdict(existing)
            by_key[(d.get("method"), d.get("seed"))] = d
        for d in rows:                       # cái mới ghi đè cái cũ cùng khoá
            by_key[(d.get("method"), d.get("seed"))] = d
        rows = [by_key[k] for k in sorted(by_key, key=lambda k: (str(k[0]), k[1]))]

    with open(path, "w") as f:
        for d in rows:
            f.write(json.dumps(d) + "\n")


def load_points(path: str | Path) -> list[PointRecord]:
    """Load points from a JSON Lines file."""
    out: list[PointRecord] = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            out.append(PointRecord(
                method=d["method"], scenario=d["scenario"], seed=int(d["seed"]),
                energy_kwh=float(d["energy_kwh"]), sla_cost=float(d["sla_cost"]),
                extra=d.get("extra"),
            ))
    return out


def method_points_dict(
    points: list[PointRecord], scenario: str | None = None
) -> dict[str, np.ndarray]:
    """Group points into ``{method: (n×2) array of (energy, sla)}`` for the
    metrics module. Optionally restrict to one scenario.
    """
    groups: dict[str, list[tuple[float, float]]] = {}
    for p in points:
        if scenario is not None and p.scenario != scenario:
            continue
        groups.setdefault(p.method, []).append(p.objectives())
    return {m: np.array(v, dtype=np.float64) for m, v in groups.items()}
