"""G2.6 — Pareto front figure: energy (kWh) vs SLA cost (C_SLA), with 95 % CI.

Static figure for the thesis/LaTeX report, so it commits to the **light print
surface** on purpose (no dark mode: the output is a PNG/PDF embedded in a
document, not a themed web chart).

Design decisions and why (dataviz method):
  * **Form** — a scatter of (energy, SLA): both objectives are minimised, so the
    reader's job is "which method sits closest to the bottom-left, and which
    trades along a front". Lower-left = better on both.
  * **One axis pair only** — never a second y-scale (the #1 chart mistake).
  * **Colour = method FAMILY, not rank.** A scatter is scored on the *all-pairs*
    CVD pairlist, where only the first four categorical slots clear the floors.
    So the four validated slots encode the four families (CMDP-PID, NSGA-II,
    heuristics, fixed-weight PPO); individual methods inside a family are
    separated by **marker shape + a direct label**, i.e. secondary encoding —
    never by inventing a 5th..8th hue. This is also the documented relief for
    the two slots that sit below 3:1 on the light surface: every point is
    directly labelled and the campaign additionally emits a table view.
  * **Legend always present** (≥2 series) and text wears ink tokens, never the
    series colour.
  * Recessive solid hairline grid (never dashed); log SLA axis only when the
    values span more than a decade (C_SLA ranges over orders of magnitude).
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")   # headless container
import matplotlib.pyplot as plt  # noqa: E402

# ── Validated palette (see dataviz references/palette.md; slots 1–4 pass the
#    all-pairs gate in light mode: worst CVD ΔE 13.0, worst normal ΔE 19.6). ──
FAMILY_COLOR = {
    "cmdp":      "#2a78d6",   # slot 1 blue    — the proposed CMDP-PID sweep
    "nsga2":     "#008300",   # slot 2 green   — static NSGA-II reference front
    "heuristic": "#e87ba4",   # slot 3 magenta — classical baselines
    "ppo":       "#eda100",   # slot 4 yellow  — Phase-1 fixed-weight PPO
}
FAMILY_LABEL = {
    "cmdp":      "CMDP-PID (budget sweep)",
    "nsga2":     "NSGA-II static reference",
    "heuristic": "Heuristic baselines",
    "ppo":       "Fixed-weight PPO",
}

# Chart chrome & ink (light surface).
SURFACE   = "#fcfcfb"
INK       = "#0b0b0b"
INK_2     = "#52514e"
MUTED     = "#898781"
GRIDLINE  = "#e1e0d9"
BASELINE  = "#c3c2b7"

# Marker shapes separate methods WITHIN a family (secondary encoding).
_HEURISTIC_MARKERS = {
    "bestfit": "o", "firstfit": "s", "k8s": "^", "random": "D", "roundrobin": "v",
}


def method_family(name: str) -> str:
    """Map a method name to its colour family."""
    n = name.lower()
    if n.startswith("cmdp"):
        return "cmdp"
    if n.startswith("nsga"):
        return "nsga2"
    if n.startswith("ppo"):
        return "ppo"
    return "heuristic"


def _marker_for(name: str) -> str:
    n = name.lower()
    if n.startswith("cmdp"):
        return "*"
    if n.startswith("ppo"):
        return "P"
    return _HEURISTIC_MARKERS.get(n, "o")


def _place_labels(fig, ax, anchors: list[tuple[float, float, str]]) -> None:
    """Direct-label every point, nudging labels so they never overlap.

    Necessary because methods can be near-coincident (bestfit and firstfit sit
    ~2 kWh apart on a ~19 000 kWh axis): a fixed offset would stack their labels
    into an unreadable blot, which is the "label overflows/collides" anti-pattern
    and would also break the direct-label relief the palette's sub-3:1 slots rely
    on. Greedy first-fit over a few candidate offsets, in display space.
    """
    if not anchors:
        return
    fig.canvas.draw()                      # fix transforms before measuring
    renderer = fig.canvas.get_renderer()

    # Candidate offsets in points: right, right-below, left, left-below, further out.
    candidates = [(9, 5), (9, -13), (-9, 5), (-9, -13), (9, 18), (9, -26)]
    placed: list[tuple[float, float, float, float]] = []   # display-space bboxes

    def overlaps(box) -> bool:
        x0, y0, x1, y1 = box
        for px0, py0, px1, py1 in placed:
            if x0 < px1 and px0 < x1 and y0 < py1 and py0 < y1:
                return True
        return False

    # Deterministic order: left-to-right, so output is reproducible.
    for x, y, name in sorted(anchors, key=lambda a: (a[0], a[1])):
        chosen = None
        for dx, dy in candidates:
            ha = "left" if dx >= 0 else "right"
            ann = ax.annotate(
                name, (x, y), textcoords="offset points", xytext=(dx, dy),
                fontsize=8, color=INK_2, ha=ha,
            )
            bb = ann.get_window_extent(renderer=renderer)
            box = (bb.x0 - 1, bb.y0 - 1, bb.x1 + 1, bb.y1 + 1)
            if not overlaps(box):
                placed.append(box)
                chosen = ann
                break
            ann.remove()
        if chosen is None:
            # Every candidate collided — keep the last resort rather than drop
            # the label (identity must never rest on colour alone).
            ann = ax.annotate(
                name, (x, y), textcoords="offset points", xytext=(9, -26),
                fontsize=8, color=INK_2,
            )
            bb = ann.get_window_extent(renderer=renderer)
            placed.append((bb.x0 - 1, bb.y0 - 1, bb.x1 + 1, bb.y1 + 1))


def plot_pareto(
    agg: dict,
    out_path: str | Path,
    scenario: str,
    nsga2_front=None,
    title: str | None = None,
) -> Path:
    """Render the energy↔SLA Pareto figure.

    Parameters
    ----------
    agg : ``{method: {"energy": MeanCI, "sla": MeanCI, ...}}`` (from
        ``aggregate.aggregate_by_method``) — one marker per method, with 95 % CI
        whiskers on both axes when >1 seed.
    nsga2_front : optional (n×2) array of the static reference front, drawn as a
        connected line (it is one deterministic front, not a seed distribution).
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(7.6, 5.4), dpi=200)
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)

    # Recessive solid hairline grid, drawn under the marks.
    ax.grid(True, which="major", color=GRIDLINE, linewidth=0.6, linestyle="-")
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(BASELINE)
        ax.spines[side].set_linewidth(1.0)
    ax.tick_params(colors=MUTED, labelsize=9, length=0)

    seen_families: set[str] = set()
    label_anchors: list[tuple[float, float, str]] = []   # (x, y, name) for pass 2

    # ── NSGA-II reference front as a connected line ──
    if nsga2_front is not None and len(nsga2_front) > 0:
        import numpy as np
        F = np.atleast_2d(np.asarray(nsga2_front, dtype=float))
        F = F[F[:, 0].argsort()]
        ax.plot(F[:, 0], F[:, 1], color=FAMILY_COLOR["nsga2"], linewidth=2.0,
                marker="o", markersize=4, alpha=0.9, zorder=2,
                label=FAMILY_LABEL["nsga2"])
        seen_families.add("nsga2")

    # ── One marker per method, 95 % CI whiskers on both axes ──
    for name in sorted(agg):
        stats = agg[name]
        fam = method_family(name)
        color = FAMILY_COLOR[fam]
        e, s = stats["energy"], stats["sla"]

        # A NaN CI (single seed) must not draw a whisker — it would imply a
        # precision we do not have (Lưu ý #10).
        xerr = None if (e.ci95 != e.ci95) else e.ci95
        yerr = None if (s.ci95 != s.ci95) else s.ci95

        label = FAMILY_LABEL[fam] if fam not in seen_families else None
        seen_families.add(fam)

        ax.errorbar(
            e.mean, s.mean, xerr=xerr, yerr=yerr,
            fmt=_marker_for(name), color=color, markersize=9,
            markeredgecolor=SURFACE, markeredgewidth=1.2,   # 2px-equivalent ring
            ecolor=color, elinewidth=1.4, capsize=3, alpha=0.95,
            zorder=3, label=label,
        )
        # Direct labels are placed in a SECOND pass: near-coincident methods
        # (e.g. bestfit vs firstfit differ by ~2 kWh out of ~19 000) would
        # otherwise stack their labels into an unreadable blot.
        label_anchors.append((e.mean, s.mean, name))

    # Log SLA axis only when the spread genuinely warrants it.
    sla_vals = [a["sla"].mean for a in agg.values() if a["sla"].mean > 0]
    if nsga2_front is not None and len(nsga2_front) > 0:
        sla_vals += [float(v) for v in nsga2_front[:, 1] if v > 0]
    if sla_vals and max(sla_vals) / max(min(sla_vals), 1e-12) > 10:
        ax.set_yscale("log")

    ax.set_xlabel("Energy (kWh)  — lower is better", fontsize=10, color=INK_2)
    ax.set_ylabel("SLA cost  $C_{SLA}$ (κ·s)  — lower is better",
                  fontsize=10, color=INK_2)
    ax.set_title(
        title or f"Energy ↔ SLA Pareto front — {scenario}  (mean ± 95 % CI)",
        fontsize=12, color=INK, pad=12, loc="left",
    )
    # "Better" cue points at the GOOD corner (lower-left): both objectives are
    # minimised, so the arrow head must sit at the origin side, not away from it.
    ax.annotate("", xy=(0.02, 0.02), xytext=(0.11, 0.11),
                xycoords="axes fraction",
                arrowprops=dict(arrowstyle="->", color=MUTED, lw=1.2))
    ax.annotate("better", xy=(0.115, 0.115), xycoords="axes fraction",
                fontsize=8, color=MUTED, ha="left", va="bottom")

    _place_labels(fig, ax, label_anchors)

    leg = ax.legend(frameon=False, fontsize=9, loc="upper right")
    for txt in leg.get_texts():
        txt.set_color(INK_2)      # text wears ink, not the series colour

    fig.tight_layout()            # include the axis band, no clipped labels
    fig.savefig(out_path, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)
    return out_path
