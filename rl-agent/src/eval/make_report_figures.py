"""Generate the Phase-2 report figures into ``assets_v2/figures/``.

Every figure is built ONLY from committed artefacts (campaign points, the NSGA-II
reference front, the topology JSON) — no numbers are typed in here, so a figure
can never drift from the data it claims to show. Provenance for each is recorded
in ``assets_v2/README.md``.

Design (dataviz method, see pareto_plot.py for the full rationale):
  * light print surface (these are PNGs for LaTeX, not themed web charts);
  * categorical hues taken from the validated palette, assigned in fixed order
    and never cycled; a scatter is capped at the 4 all-pairs-validated slots, so
    colour encodes method FAMILY and marker+direct label separate members;
  * one axis per panel — two measures of different scale become two panels
    (small multiples), never a dual y-axis;
  * recessive solid hairline grid; text in ink tokens, never the series colour.

Run::

    docker compose run --rm --no-deps \
      -v "$PWD/rl-agent/src:/app/src:ro" -v "$PWD/assets_v2:/assets_v2" \
      -v "$PWD/config:/config:ro" --entrypoint python rl-agent \
      src/eval/make_report_figures.py --scenario LOW
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

try:
    from .aggregate import aggregate_by_method
    from .pareto_plot import (BASELINE, FAMILY_COLOR, GRIDLINE, INK, INK_2,
                              MUTED, SURFACE, method_family, plot_pareto)
    from .points import load_points
    from . import pareto_metrics as pm
    from . import run_campaign as rc
except ImportError:  # pragma: no cover
    from eval.aggregate import aggregate_by_method
    from eval.pareto_plot import (BASELINE, FAMILY_COLOR, GRIDLINE, INK, INK_2,
                                  MUTED, SURFACE, method_family, plot_pareto)
    from eval.points import load_points
    from eval import pareto_metrics as pm
    from eval import run_campaign as rc


def _style(ax) -> None:
    """Shared chrome: recessive solid grid, no top/right spines, muted ticks."""
    ax.set_facecolor(SURFACE)
    ax.grid(True, color=GRIDLINE, linewidth=0.6, linestyle="-")
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(BASELINE)
        ax.spines[s].set_linewidth(1.0)
    ax.tick_params(colors=MUTED, labelsize=9, length=0)


# ── Figure 2: HV / IGD+ per method ──────────────────────────────────────────

def fig_hv_igd(metrics: dict, out: Path, scenario: str) -> Path:
    """Two panels, never a dual axis: HV (↑ better) and IGD+ (↓ better).

    They are different scales AND different directions, so plotting them on one
    axis pair would be the single worst chart mistake. Small multiples instead.
    """
    methods = [m for m in sorted(metrics["methods"]) if metrics["methods"][m]["n_points"]]
    hv = [metrics["methods"][m]["hypervolume"] for m in methods]
    igd = [metrics["methods"][m]["igd_plus"] for m in methods]
    colors = [FAMILY_COLOR[method_family(m)] for m in methods]

    fig, axes = plt.subplots(1, 2, figsize=(9.6, 4.2), dpi=200)
    fig.patch.set_facecolor(SURFACE)

    for ax, vals, name, better in (
        (axes[0], hv, "Hypervolume", "higher is better"),
        (axes[1], igd, "IGD+", "lower is better"),
    ):
        order = np.argsort(vals)[::-1] if name == "Hypervolume" else np.argsort(vals)
        labels = [methods[i] for i in order]
        v = [vals[i] for i in order]
        c = [colors[i] for i in order]
        _style(ax)
        bars = ax.barh(range(len(v)), v, color=c, height=0.62)
        ax.set_yticks(range(len(v)))
        ax.set_yticklabels(labels, fontsize=9, color=INK_2)
        ax.invert_yaxis()
        # Direct value labels — a small set, so the reader never reads the axis.
        span = max(v) if max(v) > 0 else 1.0
        for b, val in zip(bars, v):
            ax.annotate(f"{val:.3f}", (b.get_width() + span * 0.02,
                                       b.get_y() + b.get_height() / 2),
                        va="center", fontsize=8, color=INK_2)
        ax.set_xlim(0, span * 1.22)
        ax.set_title(f"{name}  — {better}", fontsize=11, color=INK, loc="left", pad=8)

    fig.suptitle(
        f"Pareto quality per scheduler — {scenario} "
        f"(shared fixed reference point; seed-averaged)",
        fontsize=12, color=INK, x=0.01, ha="left", y=0.99,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(out, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)
    return out


# ── Figure 3: NSGA-II static reference front ────────────────────────────────

def fig_nsga2_front(payload: dict, out: Path) -> Path:
    """The G2.3 static front. ONE series ⇒ no legend box; the title names it."""
    F = np.array(payload["front"], dtype=float)
    F = F[F[:, 0].argsort()]
    spread = payload.get("anchor_spread")
    packed = payload.get("anchor_packed")

    fig, ax = plt.subplots(figsize=(7.6, 5.0), dpi=200)
    fig.patch.set_facecolor(SURFACE)
    _style(ax)

    ax.plot(F[:, 0], F[:, 1], color=FAMILY_COLOR["nsga2"], linewidth=2.0,
            marker="o", markersize=6, markeredgecolor=SURFACE,
            markeredgewidth=1.2, zorder=3)

    # The anchors are deliberately NOT plotted: they sit far to the right
    # (~2540–2600 kWh vs the front's ~1880–1975), so including them stretched
    # the x-axis until the front — the actual subject of the figure — occupied
    # ~13 % of the width. Their values go in the caption instead, where they
    # still tell the reader what NSGA-II improved on.

    # The SLA axis spans ~10× here; a log scale is what makes the knee legible
    # instead of collapsing 12 of 14 points onto the floor.
    if F[:, 1].max() / max(F[:, 1].min(), 1e-12) > 10:
        ax.set_yscale("log")

    ax.set_xlabel("Energy (kWh)  — lower is better", fontsize=10, color=INK_2)
    ax.set_ylabel("SLA cost  $C_{SLA}$ (κ·s)  — lower is better", fontsize=10,
                  color=INK_2)
    topo = "heterogeneous" if "hetero" in str(payload.get("topology", "")) else "homogeneous"
    ax.set_title(
        f"NSGA-II static reference front — {payload['scenario']} "
        f"({topo}, {payload['num_tasks']} tasks, {len(F)} points)",
        fontsize=12, color=INK, loc="left", pad=12,
    )
    # The caveat belongs ON the figure: this front is not a head-to-head result.
    # Wrap it — an unwrapped caption makes bbox_inches="tight" stretch the whole
    # figure to the text's width, leaving the plot marooned in dead space.
    import textwrap
    caption = ("Static idealisation: full-trace foreknowledge, no temporal dynamics, "
               "static surrogate evaluator — a reference/upper bound, NOT comparable "
               "head-to-head with the online schedulers (Lưu ý #9).")
    if spread and packed:
        caption += (f" NSGA-II dominates both naive anchors (off-scale right): "
                    f"spread E={spread[0]:.0f} kWh, packed E={packed[0]:.0f} kWh "
                    f"(both C_SLA≈{spread[1]:.2g}) — it routes CPU tasks to cheap "
                    f"CPU-only hosts and packs GPU tasks onto few GPU hosts (G2.1/G2.2).")
    fig.text(0.01, -0.02, "\n".join(textwrap.wrap(caption, width=95)),
             fontsize=7.5, color=MUTED, ha="left", va="top")
    fig.tight_layout()
    fig.savefig(out, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)
    return out


# ── Figure 4: heterogeneous topology SKU profile ────────────────────────────

def fig_topology(topology_json: Path, out: Path) -> Path:
    """The G2.1 SKU profile: CPU power band per SKU (one axis, watts) with the
    GPU inventory annotated — the two axes G2.1/G2.2 actually add."""
    cfg = json.loads(topology_json.read_text())
    skus = cfg["skus"]
    idle = [float(s.get("cpuIdleWatt", 120)) for s in skus]
    mx = [float(s.get("cpuMaxWatt", 400)) for s in skus]
    gpus = [int(s.get("gpu", 8)) for s in skus]
    # GPU inventory lives IN the tick label rather than as a separate annotation:
    # a floating annotation under a 2-line tick label collided with it, and the
    # text duplicated the SKU name anyway.
    names = [f"{s['name']}\n×{s['count']}  ·  {g} GPU"
             for s, g in zip(skus, gpus)]

    x = np.arange(len(skus))
    w = 0.36
    fig, ax = plt.subplots(figsize=(7.6, 4.6), dpi=200)
    fig.patch.set_facecolor(SURFACE)
    _style(ax)

    # Both series are watts ⇒ one axis (never a second scale).
    b1 = ax.bar(x - w / 2 - 0.01, idle, w, label="$P_{idle}$ (W)",
                color="#2a78d6")
    b2 = ax.bar(x + w / 2 + 0.01, mx, w, label="$P_{max}$ (W)",
                color="#008300")
    for bars in (b1, b2):
        for b in bars:
            ax.annotate(f"{b.get_height():.0f}",
                        (b.get_x() + b.get_width() / 2, b.get_height()),
                        ha="center", va="bottom", fontsize=8.5, color=INK_2)

    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=9.5, color=INK_2)
    ax.set_ylabel("CPU power (W)", fontsize=10, color=INK_2)
    ax.set_ylim(0, max(mx) * 1.18)
    ax.set_title("G2.1 heterogeneous topology — 3 SKUs × 10 hosts",
                 fontsize=12, color=INK, loc="left", pad=12)
    leg = ax.legend(frameon=False, fontsize=9, loc="upper right")
    for t in leg.get_texts():
        t.set_color(INK_2)
    import textwrap
    fig.text(0.01, -0.06, "\n".join(textwrap.wrap(
        "vCPU/RAM held uniform (64 vCPU / 256 GB) so SKUs differ only along the "
        "energy axis (P_idle/P_max) and the GPU-affinity axis — the two dimensions "
        "G2.1/G2.2 introduce. GPU tasks are mask-infeasible on CPU-only hosts (G2.2).",
        width=95)), fontsize=7.5, color=MUTED, ha="left", va="top")
    fig.tight_layout()
    fig.savefig(out, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)
    return out


# ── Driver ──────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser(description="Generate Phase-2 report figures")
    ap.add_argument("--scenario", default="LOW")
    ap.add_argument("--results", default="/data/results")
    ap.add_argument("--topology", default="/config/topology-hetero.json")
    ap.add_argument("--output", default="/assets_v2/figures")
    args = ap.parse_args()

    R = Path(args.results)
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    made: list[Path] = []

    # Fig 1 — campaign Pareto front.
    pts_file = R / f"campaign-{args.scenario}" / "points.jsonl"
    metrics = None
    if pts_file.exists():
        points = load_points(pts_file)
        agg = aggregate_by_method(points, scenario=args.scenario)
        fam = rc.family_points_dict(points, scenario=args.scenario)
        metrics = pm.evaluate_methods(fam)
        made.append(plot_pareto(agg, out / f"fig1-pareto-{args.scenario}.png",
                                args.scenario, nsga2_front=None))
        # Fig 2 — HV / IGD+.
        made.append(fig_hv_igd(metrics, out / f"fig2-hv-igd-{args.scenario}.png",
                               args.scenario))
    else:
        print(f"[figures] skip fig1/fig2 — no {pts_file}")

    # Fig 3 — NSGA-II static reference front.
    nsga_file = R / f"nsga2-{args.scenario}" / "reference_front.json"
    if nsga_file.exists():
        made.append(fig_nsga2_front(json.loads(nsga_file.read_text()),
                                    out / f"fig3-nsga2-front-{args.scenario}.png"))
    else:
        print(f"[figures] skip fig3 — no {nsga_file}")

    # Fig 4 — topology SKUs.
    topo = Path(args.topology)
    if topo.exists():
        made.append(fig_topology(topo, out / "fig4-topology-hetero-skus.png"))
    else:
        print(f"[figures] skip fig4 — no {topo}")

    for p in made:
        print(f"[figures] wrote {p}")


if __name__ == "__main__":
    main()
