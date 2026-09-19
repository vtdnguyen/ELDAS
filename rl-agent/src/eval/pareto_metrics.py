"""G2.4 — Pareto quality metrics: hypervolume (HV) and IGD+.

Both indicators are **Pareto-compliant** (IGD+ unlike plain IGD; HV always) —
CLAUDE.md Lưu ý #8. The two objectives are, throughout Phase 2, **both
minimised** and in the same units as the DES / NSGA-II front:

    obj[0] = energy_kwh   (lower better)
    obj[1] = sla_cost      (= C_SLA = Σ κ·max(0, completion − deadline), lower better)

Key invariants (Lưu ý #8 / #15):
  * **ONE fixed reference point** (a common nadir) for every method — otherwise
    hypervolumes are not comparable. We derive it once from the union of all
    method points and reuse it everywhere.
  * Objectives are **normalised to [0,1]** against a fixed (ideal, nadir) so the
    two very-different scales (energy ~1e4 vs C_SLA ~1e8) contribute comparably.
    In normalised space the reference point is (1,1).
  * **Reference front for IGD+** = the non-dominated union of all methods'
    points (or a supplied NSGA-II front). All methods are scored against the
    same reference front.
  * Consistent min/min orientation before any dominance / HV computation
    (Lưu ý #15).

This module is pure NumPy for the geometry; HV / IGD+ delegate to a Pareto-compliant
indicator library when available, with a small exact NumPy implementation for the
2-objective case so the metric is testable — and auditable — without any library.

**Three backends** (W5.1): ``pymoo``, ``moocore`` and ``numpy``. Three rather than two on
purpose. Both libraries implement the same published definitions, so if all three agree to
1e-6 on the hand-worked cases, a bug would have to be present *identically* in an
independent C implementation (moocore is the reference implementation from the group that
introduced IGD+ and the EAF), in pymoo, and in our own sweep-line — which is not a
plausible coincidence. Two agreeing is much weaker evidence, because a shared
misunderstanding of the definition is exactly the failure mode that matters here.
"""

from __future__ import annotations

import numpy as np

try:
    from .nsga2_baseline import non_dominated
except ImportError:  # pragma: no cover - direct-script fallback
    from nsga2_baseline import non_dominated


# ── Backends (W5.1) ─────────────────────────────────────────────────────────

#: Preference order for ``backend="auto"``. pymoo first only because it is the pinned
#: dependency the campaign already runs on; the numbers are identical.
BACKENDS = ("pymoo", "moocore", "numpy")


class BackendUnavailable(RuntimeError):
    """An explicitly requested backend is not installed."""


def _have(name: str) -> bool:
    if name == "numpy":
        return True
    try:
        if name == "pymoo":
            import pymoo.indicators.hv  # noqa: F401
        elif name == "moocore":
            import moocore  # noqa: F401
        else:
            return False
    except ImportError:
        return False
    return True


def available_backends() -> tuple[str, ...]:
    """Which backends this environment can actually run."""
    return tuple(b for b in BACKENDS if _have(b))


def resolve_backend(backend: str | None, use_pymoo: bool = True) -> str:
    """Pick the backend to run, honouring the legacy ``use_pymoo`` flag.

    An **explicitly named** backend that is missing raises instead of falling back. That
    matters more than it looks: the whole value of the three-way cross-check is that the
    three names run three different implementations, and a silent fallback would let the
    agreement test pass while comparing NumPy against itself.
    """
    if backend is None:
        backend = "auto" if use_pymoo else "numpy"
    if backend == "auto":
        for b in BACKENDS:
            if _have(b):
                return b
        return "numpy"                                  # pragma: no cover - always true
    if backend not in BACKENDS:
        raise ValueError(f"unknown backend {backend!r}; expected one of {BACKENDS}")
    if not _have(backend):
        raise BackendUnavailable(
            f"backend {backend!r} requested but not installed "
            f"(available: {available_backends()})")
    return backend


# ── Fixed reference geometry (Lưu ý #8) ─────────────────────────────────────

def stack_points(method_points: dict[str, np.ndarray]) -> np.ndarray:
    """Vertically stack every method's (n_i × 2) point array into one (N × 2)."""
    arrays = [np.atleast_2d(np.asarray(p, dtype=np.float64))
              for p in method_points.values() if len(p) > 0]
    if not arrays:
        raise ValueError("no points supplied")
    return np.vstack(arrays)


def ideal_point(all_F: np.ndarray) -> np.ndarray:
    """Best-case corner = per-objective minimum over ALL points (min/min)."""
    return np.asarray(all_F, dtype=np.float64).min(axis=0)


def nadir_point(all_F: np.ndarray, margin: float = 0.10) -> np.ndarray:
    """Fixed reference/nadir = per-objective max over ALL points, pushed out by
    ``margin`` so the worst point still encloses positive hypervolume.

    Using the *same* nadir for every method is what makes hypervolumes
    comparable (Lưu ý #8). ``margin`` is relative to the ideal→max span.
    """
    F = np.asarray(all_F, dtype=np.float64)
    hi = F.max(axis=0)
    lo = F.min(axis=0)
    span = np.where(hi > lo, hi - lo, np.abs(hi) + 1.0)
    return hi + margin * span


def normalize(F: np.ndarray, ideal: np.ndarray, nadir: np.ndarray) -> np.ndarray:
    """Map objectives into [0,1] via a FIXED (ideal, nadir); ideal→0, nadir→1."""
    F = np.atleast_2d(np.asarray(F, dtype=np.float64))
    denom = np.where(nadir > ideal, nadir - ideal, 1.0)
    return (F - ideal) / denom


# ── Hypervolume ─────────────────────────────────────────────────────────────

def _hv_2d_exact(F_norm: np.ndarray, ref: np.ndarray) -> float:
    """Exact 2-objective hypervolume (minimisation) dominated by ``F_norm``
    up to reference ``ref``. Sweep-line over the non-dominated staircase.

    Independent of pymoo so HV is testable and auditable.
    """
    F = np.atleast_2d(F_norm).astype(np.float64)
    # Keep only points that dominate the reference (strictly inside the box).
    inside = np.all(F < ref, axis=1)
    F = F[inside]
    if len(F) == 0:
        return 0.0
    F = F[non_dominated(F)]
    # Sort by x ascending (tie-break y). Non-dominated ⇒ y strictly decreasing,
    # so the dominated region splits into vertical strips [x_i, x_{i+1}) whose
    # height is (ref_y − y_i). See the two hand-worked cases in the tests.
    order = np.lexsort((F[:, 1], F[:, 0]))
    F = F[order]
    n = len(F)
    hv = 0.0
    for i in range(n):
        x_next = F[i + 1, 0] if i + 1 < n else ref[0]
        hv += (x_next - F[i, 0]) * (ref[1] - F[i, 1])
    return float(hv)


def hypervolume(
    F: np.ndarray, ideal: np.ndarray, nadir: np.ndarray, use_pymoo: bool = True,
    backend: str | None = None,
) -> float:
    """Normalised hypervolume of front ``F`` against the fixed (ideal, nadir).

    Reference point in normalised space is (1,1). Returns 0 for an empty or
    fully-dominated (outside-the-box) front.

    ``backend`` selects the implementation (:data:`BACKENDS`); ``None`` keeps the legacy
    ``use_pymoo`` behaviour, so every existing call site is unchanged.
    """
    F_norm = normalize(F, ideal, nadir)
    ref = np.ones(F_norm.shape[1])
    chosen = resolve_backend(backend, use_pymoo)

    if chosen == "pymoo":
        from pymoo.indicators.hv import HV
        return float(HV(ref_point=ref)(F_norm))
    if chosen == "moocore":
        import moocore
        # moocore minimises by default and ignores points outside the reference box,
        # matching both the pymoo indicator and _hv_2d_exact.
        return float(moocore.hypervolume(F_norm, ref=ref))
    if F_norm.shape[1] != 2:  # pragma: no cover - only 2-obj in Phase 2
        raise ValueError("NumPy HV fallback supports 2 objectives only")
    return _hv_2d_exact(F_norm, ref)


# ── IGD+ ─────────────────────────────────────────────────────────────────────

def _igd_plus_np(A_norm: np.ndarray, R_norm: np.ndarray) -> float:
    """Exact IGD+ (minimisation): mean over reference points r of the minimum
    modified distance d+(a, r) = ||max(a − r, 0)|| to the approximation A.

    d+ ignores the components where ``a`` is already better than ``r`` — this is
    what makes IGD+ weakly Pareto-compliant (unlike plain IGD).
    """
    A = np.atleast_2d(A_norm).astype(np.float64)
    R = np.atleast_2d(R_norm).astype(np.float64)
    total = 0.0
    for r in R:
        diff = np.maximum(A - r, 0.0)          # (|A|, m)
        d = np.sqrt((diff ** 2).sum(axis=1))   # (|A|,)
        total += d.min()
    return float(total / len(R))


def igd_plus(
    F: np.ndarray,
    reference_front: np.ndarray,
    ideal: np.ndarray,
    nadir: np.ndarray,
    use_pymoo: bool = True,
    backend: str | None = None,
) -> float:
    """Normalised IGD+ of ``F`` against ``reference_front`` (lower = better)."""
    A_norm = normalize(F, ideal, nadir)
    R_norm = normalize(reference_front, ideal, nadir)
    chosen = resolve_backend(backend, use_pymoo)

    if chosen == "pymoo":
        from pymoo.indicators.igd_plus import IGDPlus
        return float(IGDPlus(R_norm)(A_norm))
    if chosen == "moocore":
        import moocore
        # moocore's `ref` is the reference SET (same role as pymoo's IGDPlus(R)).
        return float(moocore.igd_plus(A_norm, ref=R_norm))
    return _igd_plus_np(A_norm, R_norm)


# ── Reference front (union of all methods) ──────────────────────────────────

def union_reference_front(method_points: dict[str, np.ndarray]) -> np.ndarray:
    """Reference front for IGD+ = non-dominated points of the union of all
    methods (Lưu ý #8). An external NSGA-II front may be passed as one method.
    """
    all_F = stack_points(method_points)
    return all_F[non_dominated(all_F)]


# ── Top-level scoring ────────────────────────────────────────────────────────

def evaluate_methods(
    method_points: dict[str, np.ndarray],
    reference_front: np.ndarray | None = None,
    nadir_margin: float = 0.10,
    use_pymoo: bool = True,
    backend: str | None = None,
) -> dict:
    """Compute HV + IGD+ for every method under ONE fixed reference geometry.

    Parameters
    ----------
    method_points : {name: (n×2) array of (energy_kwh, sla_cost)}
    reference_front : optional external reference front (e.g. NSGA-II). If
        ``None``, the union non-dominated front of all methods is used.

    Returns
    -------
    dict with ``ideal``, ``nadir``, ``reference_front`` and a per-method
    ``{hypervolume, igd_plus, n_points}`` block. Higher HV = better, lower
    IGD+ = better.
    """
    all_F = stack_points(method_points)
    ideal = ideal_point(all_F)
    nadir = nadir_point(all_F, margin=nadir_margin)
    chosen = resolve_backend(backend, use_pymoo)

    ref_front = (union_reference_front(method_points)
                 if reference_front is None
                 else np.atleast_2d(np.asarray(reference_front, dtype=np.float64)))

    results = {}
    for name, pts in method_points.items():
        pts = np.atleast_2d(np.asarray(pts, dtype=np.float64))
        if len(pts) == 0:
            results[name] = {"hypervolume": 0.0, "igd_plus": float("inf"),
                             "n_points": 0}
            continue
        results[name] = {
            "hypervolume": hypervolume(pts, ideal, nadir, backend=chosen),
            "igd_plus": igd_plus(pts, ref_front, ideal, nadir, backend=chosen),
            "n_points": int(len(pts)),
        }

    return {
        "ideal": ideal.tolist(),
        "nadir": nadir.tolist(),
        "reference_front": ref_front.tolist(),
        "objectives": ["energy_kwh", "sla_cost"],
        "sense": ["min", "min"],
        # Recorded so a results file says which implementation produced its numbers.
        "backend": chosen,
        "methods": results,
    }
