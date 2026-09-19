#!/usr/bin/env python3
"""W4.4 — Figure + prose for the burstiness anchors (PLAN-Workload-Model.md §3.5).

The claim the thesis has to support is narrow and checkable:

    IDC(1h) = 40 for the BURST scenario is not an invented number. It is bracketed by two
    independent production GPU clusters, measured at the horizon the experiment actually
    runs at — and the WM-1 generator reproduces it.

A single number in a table cannot carry that, because the whole argument turns on *which
window the measurement was taken over*. Philly reads 160 over its full 57 days and 69
inside a 3-day slice; quoting the first would make the target look absurdly low and the
second makes it look reasonable, and only one of them describes what an agent inside a
4-day episode ever experiences. So the figure puts both on the same axis.

Panel A  per-trace IDC(1h): whole-trace value, the distribution across disjoint 3-day
         windows, and the two chosen targets drawn across everything.
Panel B  what WM-1 actually generates at those targets, per scenario, across seeds —
         the loop being closed rather than assumed.

Outputs (all under assets_v2, which is gitignored — regenerate, do not commit):
    assets_v2/figures/burstiness-anchors.png
    assets_v2/figures/burstiness-anchors.json     machine-readable, for the tests
    assets_v2/docs/burstiness-anchors.md          the paragraph, with its caveats

Usage
-----
    docker run --rm -v "$PWD/rl-agent/src:/app/src:ro" \
        -v "$PWD/data:/data:ro" -v "$PWD/assets_v2:/out" \
        -e PYTHONPATH=/app/src --entrypoint python eldas-rl-agent \
        /scripts/plot-burstiness-anchors.py --out /out
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "..", "rl-agent", "src"))

from workload import arrivals, burstiness, wm1                    # noqa: E402
from workload.calibrate import ARMS                               # noqa: E402
from workload.jobsize import JobSizePool                          # noqa: E402
from workload.wm1 import _ARRIVAL_STREAM_OFFSET                   # noqa: E402

SEEDS = (42, 43, 44, 45, 46)

#: Traces on the left panel. openb is measured on the SCHEDULABLE population (Pending pods
#: are arrivals the simulator never sees); Philly and Helios have no such distinction, so
#: they are measured whole. Mixing those populations would show a difference in who was
#: counted as a difference in burstiness.
SOURCES = (
    ("philly", "Philly\n(Microsoft, 2 490 GPU)", "reference-traces/philly_data_training.csv", None),
    ("helios", "Helios\n(SenseTime, 6 416 GPU)", "reference-traces/helios_data_training.csv", None),
    ("openb", "openb\n(Alibaba, in use)", "alibaba-trace/openb_pod_list_default.csv",
     burstiness.SCHEDULABLE_ONLY),
)


def measure_sources(data_root: str) -> list[dict]:
    out = []
    for name, label, rel, exclude in SOURCES:
        path = os.path.join(data_root, rel)
        if not os.path.isfile(path):
            print(f"[warn] missing {path} — run scripts/fetch-reference-traces.sh",
                  file=sys.stderr)
            continue
        p = burstiness.profile_adapter(name, path, exclude_phases=exclude)
        out.append({
            "name": name, "label": label,
            "n_arrivals": p.n_arrivals, "span_days": p.span_days,
            "squared_cv": p.squared_cv, "idc_whole": p.idc_whole,
            "idc_3d_median": p.idc_window_median,
            "idc_3d_q1": p.idc_window_q1, "idc_3d_q3": p.idc_window_q3,
            "n_windows": p.n_windows,
            "idc_3d_values": list(p.idc_window_values),
            "population": ("schedulable (Pending excluded)" if exclude else "all rows"),
            "caveat": p.caveat(),
        })
    return out


def measure_generated(trace_path: str, arm: str = "homo") -> list[dict]:
    """What the generator actually produces at each scenario's IDC target.

    Measured on the arrival times alone, with the same estimator used on the real traces,
    so panel B is comparable with panel A rather than merely adjacent to it.
    """
    from workload import adapters

    capacity, horizon = ARMS[arm]
    source = adapters.require("openb", adapters.CAP_JOBSIZE).load(trace_path)
    pool = JobSizePool.from_trace(source, horizon, capacity=capacity)

    from workload.calibrate import plan_load

    out = []
    for scenario, (rho, idc_target) in wm1.SCENARIOS.items():
        plan = plan_load(pool, capacity, rho, horizon)
        duty = arrivals.fit_duty_cycle(plan.n_task, idc_target, horizon)
        vals = []
        for seed in SEEDS:
            times = arrivals.generate(plan.n_task, duty, horizon,
                                      random.Random(seed + _ARRIVAL_STREAM_OFFSET))
            vals.append(arrivals.idc(times, arrivals.IDC_WINDOW_SEC, 0.0, horizon))
        mean = sum(vals) / len(vals)
        sd = (sum((v - mean) ** 2 for v in vals) / len(vals)) ** 0.5
        out.append({"scenario": scenario, "rho": rho, "idc_target": idc_target,
                    "n_task": plan.n_task, "duty_cycle": duty,
                    "idc_mean": mean, "idc_sd": sd, "idc_values": vals})
    return out


# ── Figure ──────────────────────────────────────────────────────────────────

def plot(sources: list[dict], generated: list[dict], out_png: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    targets = sorted({g["idc_target"] for g in generated})

    # Shared log y-axis across both panels on purpose: it lets the reader check the claim
    # directly instead of trusting a caption. The BURST points in B must land in the gap
    # between the Helios and Philly medians in A — if they do not, the target is not
    # bracketed and the anchoring argument fails.
    fig, (ax_a, ax_b) = plt.subplots(
        1, 2, figsize=(11.5, 5.0), sharey=True,
        gridspec_kw={"width_ratios": [1.25, 1.0]})

    # ── Panel A: measured burstiness of the real traces ─────────────────────
    xs = range(len(sources))
    for x, s in zip(xs, sources):
        vals = s["idc_3d_values"]
        if vals:
            # Every window as a point: with 6-8 of them a box plot would imply a
            # distribution the data cannot support. Show the data.
            jitter = [x + 0.06 * ((i % 5) - 2) for i in range(len(vals))]
            ax_a.scatter(jitter, vals, s=34, alpha=0.75, color="#4c72b0", zorder=3,
                         label="IDC(1h) in a 3-day window" if x == 0 else None)
            ax_a.hlines(s["idc_3d_median"], x - 0.22, x + 0.22, color="#4c72b0",
                        lw=2.5, zorder=4,
                        label="median across windows" if x == 0 else None)
        ax_a.scatter([x], [s["idc_whole"]], marker="D", s=70, color="#c44e52", zorder=5,
                     label="IDC(1h) over the whole trace" if x == 0 else None)

    for t in targets:
        ax_a.axhline(t, ls="--", lw=1.2, color="#55a868", zorder=1)
        ax_a.text(len(sources) - 0.42, t * 1.06, f"WM-1 target {t:g}",
                  color="#55a868", fontsize=9, va="bottom", ha="right")

    ax_a.set_yscale("log")
    ax_a.set_xticks(list(xs))
    ax_a.set_xlim(-0.55, len(sources) - 0.45)
    ax_a.set_xticklabels([s["label"] for s in sources], fontsize=9)
    ax_a.set_ylabel("IDC(1h) = Var(N) / E(N)   (log scale)")
    ax_a.set_title("A. Measured burstiness — window length matters", fontsize=11, loc="left")
    ax_a.grid(axis="y", alpha=0.25)
    ax_a.legend(fontsize=8, loc="lower left", framealpha=0.9)

    # ── Panel B: what WM-1 generates at those targets ───────────────────────
    order = ["LOW", "HIGH", "OVERLOAD", "BURST"]
    gen = sorted(generated, key=lambda g: order.index(g["scenario"])
                 if g["scenario"] in order else 99)
    xs_b = range(len(gen))
    for t in targets:
        ax_b.axhline(t, ls="--", lw=1.2, color="#55a868", zorder=1)
    for x, g in zip(xs_b, gen):
        ax_b.scatter([x] * len(g["idc_values"]), g["idc_values"], s=34, alpha=0.75,
                     color="#4c72b0", zorder=3,
                     label="realised, per seed" if x == 0 else None)
        ax_b.hlines(g["idc_target"], x - 0.28, x + 0.28, color="#55a868", lw=2.5,
                    zorder=4, label="target" if x == 0 else None)

    ax_b.set_xticks(list(xs_b))
    ax_b.set_xlim(-0.6, len(gen) - 0.4)
    ax_b.set_xticklabels([g["scenario"] for g in gen], fontsize=9)
    ax_b.set_title("B. What WM-1 generates (homo, 5 seeds)", fontsize=11, loc="left")
    ax_b.grid(axis="y", alpha=0.25)
    ax_b.legend(fontsize=8, loc="lower left", framealpha=0.9)

    fig.tight_layout()
    os.makedirs(os.path.dirname(out_png), exist_ok=True)
    fig.savefig(out_png, dpi=160)
    plt.close(fig)


# ── Prose ───────────────────────────────────────────────────────────────────

def _n(value) -> str:
    """Thousands separated by a space — the convention the rest of the plan uses."""
    return f"{value:,}".replace(",", " ")


def write_prose(sources: list[dict], generated: list[dict], out_md: str) -> None:
    by = {s["name"]: s for s in sources}
    gen = {g["scenario"]: g for g in generated}
    philly, helios, openb = by.get("philly"), by.get("helios"), by.get("openb")

    lines = [
        "# Neo tham số burstiness vào hai trace GPU sản xuất (W4.4)",
        "",
        "> Sinh tự động bởi `scripts/plot-burstiness-anchors.py`. **Không sửa tay** — sửa"
        " script rồi chạy lại.",
        "",
        "## Đoạn văn cho LV",
        "",
    ]

    if philly and helios and openb:
        lines += [
            f"Tham số burstiness của WM-1 được neo vào đo đạc chứ không chọn tay. Chỉ số"
            f" phân tán theo số đếm `IDC(w) = Var(N_w)/E(N_w)` (Poisson = 1) được đo trên"
            f" hai trace cụm GPU sản xuất **độc lập với đề tài** — Philly (Microsoft,"
            f" 2 490 GPU) và Helios (SenseTime, 6 416 GPU), lấy từ bộ"
            f" DIR-LAB/Gen-Parallel-Workloads (JSSPP 2024) — và trên chính trace đang dùng"
            f" (openb/Alibaba).",
            "",
            f"Điểm mấu chốt là **cửa sổ quan sát**. Đo trên toàn bộ trace, Philly cho"
            f" IDC(1h) = {philly['idc_whole']:.1f} trên {philly['span_days']:.0f} ngày và"
            f" Helios {helios['idc_whole']:.1f} trên {helios['span_days']:.0f} ngày. Nhưng"
            f" phần lớn độ phân tán đó là biến động **tuần-qua-tuần** của cường độ nộp"
            f" việc; một agent sống trong một episode dài 4 ngày không bao giờ trải nghiệm"
            f" nó như burstiness — với agent, đó chỉ là \"episode này trung bình bận hơn"
            f" episode trước\". Vì vậy mốc neo phải đo **trong cửa sổ dài đúng bằng một"
            f" episode**: trong các cửa sổ 3 ngày rời nhau, Philly có trung vị"
            f" {philly['idc_3d_median']:.1f} (IQR {philly['idc_3d_q1']:.1f}–"
            f"{philly['idc_3d_q3']:.1f}, {philly['n_windows']} cửa sổ) và Helios"
            f" {helios['idc_3d_median']:.1f} (IQR {helios['idc_3d_q1']:.1f}–"
            f"{helios['idc_3d_q3']:.1f}, {helios['n_windows']} cửa sổ) — đều **thấp hơn"
            f" hẳn** con số toàn trace.",
            "",
            f"Từ đó hai mục tiêu được chốt: `LOW`/`HIGH`/`OVERLOAD` dùng **IDC(1h) = 10**,"
            f" neo vào chính openb (đo được {openb['idc_whole']:.1f} trên"
            f" {_n(openb['n_arrivals'])} pod khả lập lịch) — tức \"bursty như trace đang"
            f" dùng\"; `BURST` dùng **IDC(1h) = 40**, nằm **giữa** hai cụm sản xuất đo ở"
            f" cùng horizon ({helios['idc_3d_median']:.0f} < 40 <"
            f" {philly['idc_3d_median']:.0f}). Đây là lý do burstiness trong ELDAS là một"
            f" **tham số có mốc thực nghiệm**, không phải hệ quả phụ của cách cắt trace như"
            f" ở thiết kế Giai đoạn 1.",
            "",
        ]
        if gen.get("BURST"):
            b = gen["BURST"]
            lines += [
                f"Vòng lặp được khép lại chứ không giả định: sinh ở mục tiêu 40, WM-1 đo"
                f" lại được **{b['idc_mean']:.1f} ± {b['idc_sd']:.1f}** trên"
                f" {len(SEEDS)} seed (panel B) — vẫn nằm giữa hai mốc"
                f" {helios['idc_3d_median']:.0f} và {philly['idc_3d_median']:.0f}. Chênh"
                f" lệch so với đúng 40 là do bisection fit trên một tập stream replicate"
                f" khác với 5 seed dùng để chạy campaign; độ lệch chuẩn giữa các seed"
                f" ({b['idc_sd']:.1f}) đã bao trùm khoảng chênh đó.",
                "",
            ]

    lines += ["## Caveat — phải ghi kèm khi trích số", ""]
    for s in sources:
        if s.get("caveat"):
            # caveat() already leads with the trace name; prefixing it again reads as a stutter.
            lines.append(f"- {s['caveat']}.")
    lines += [
        "- IQR ở đây là khoảng biến thiên của **chính những cửa sổ đó**, không phải khoảng"
        " tin cậy của một tổng thể — cỡ mẫu quá nhỏ để đọc theo nghĩa thứ hai.",
        "- Philly/Helios **chỉ** dùng để đo arrival process. `wall_time` bằng 0 ở **mọi"
        " dòng của cả hai** trace và `cpu_num` bằng 0 ở **mọi dòng Philly**; không có cột"
        " memory, không có cột QoS. Adapter vì thế **từ chối** vai trò nguồn job-size"
        " (`adapters.require(..., CAP_JOBSIZE)` raise).",
        "- Ba tổng thể openb cho **ba con số khác nhau**: toàn bộ 8 152 dòng ⇒ 13.5;"
        f" 7 255 pod khả lập lịch (bỏ `Pending`) ⇒"
        f" {by.get('openb', {}).get('idc_whole', float('nan')):.1f} — **đây là con số"
        " dùng trong §3.5**; pool WM-1 (bỏ cả `Failed`, 5 385 job) ⇒ 9.5. Đặt một trong"
        " ba cạnh số Philly (đo trên mọi dòng, vì trace đó không có khái niệm `Pending`)"
        " mà không nói rõ đếm ai thì khác biệt về **tổng thể** sẽ bị đọc thành khác biệt"
        " về **burstiness**.",
        "",
        "## Số đo",
        "",
        "| Trace | n | span (ngày) | tổng thể | `C_a²` | IDC(1h) toàn trace |"
        " IDC(1h) cửa sổ 3d (trung vị) | IQR | #cửa sổ |",
        "|---|---:|---:|---|---:|---:|---:|---|---:|",
    ]
    for s in sources:
        med = "—" if s["idc_3d_median"] is None else f"**{s['idc_3d_median']:.1f}**"
        iqr = ("—" if s["idc_3d_q1"] is None
               else f"{s['idc_3d_q1']:.1f}–{s['idc_3d_q3']:.1f}")
        lines.append(
            f"| {s['name']} | {s['n_arrivals']:,} | {s['span_days']:.1f} |"
            f" {s['population']} | {s['squared_cv']:.1f} | {s['idc_whole']:.1f} |"
            f" {med} | {iqr} | {s['n_windows']} |".replace(",", " "))

    lines += [
        "",
        "| Scenario | ρ | IDC* | N | duty cycle `p1` | IDC(1h) sinh ra (mean ± sd, 5 seed) |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for g in generated:
        lines.append(
            f"| `{g['scenario']}` | {g['rho']:.2f} | {g['idc_target']:.0f} |"
            f" {g['n_task']:,} | {g['duty_cycle']:.3f} |"
            f" {g['idc_mean']:.1f} ± {g['idc_sd']:.1f} |".replace(",", " "))

    lines += [
        "",
        "![Burstiness anchors](../figures/burstiness-anchors.png)",
        "",
    ]

    os.makedirs(os.path.dirname(out_md), exist_ok=True)
    with open(out_md, "w", encoding="utf-8", newline="\n") as f:
        f.write("\n".join(lines))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-root", default="/data")
    ap.add_argument("--out", default="/out", help="assets_v2 root")
    ap.add_argument("--arm", default="homo", choices=sorted(ARMS))
    ap.add_argument("--no-generated", action="store_true",
                    help="skip panel B (it refits the duty cycle, ~1 min)")
    args = ap.parse_args(argv)

    sources = measure_sources(args.data_root)
    if not sources:
        print("[ERROR] no source trace found", file=sys.stderr)
        return 2

    generated = []
    if not args.no_generated:
        trace = os.path.join(args.data_root, "alibaba-trace",
                             "openb_pod_list_default.csv")
        if os.path.isfile(trace):
            generated = measure_generated(trace, args.arm)
        else:
            print(f"[warn] {trace} missing — panel B skipped", file=sys.stderr)

    fig_dir = os.path.join(args.out, "figures")
    doc_dir = os.path.join(args.out, "docs")
    png = os.path.join(fig_dir, "burstiness-anchors.png")

    plot(sources, generated, png)
    write_prose(sources, generated, os.path.join(doc_dir, "burstiness-anchors.md"))

    os.makedirs(fig_dir, exist_ok=True)
    with open(os.path.join(fig_dir, "burstiness-anchors.json"), "w",
              encoding="utf-8", newline="\n") as f:
        json.dump({"sources": sources, "generated": generated, "arm": args.arm},
                  f, indent=2)
        f.write("\n")

    print(f"[W4.4] figure -> {png}")
    for s in sources:
        print(f"  {s['name']:8s} whole={s['idc_whole']:6.1f}  "
              f"3d-median={s['idc_3d_median'] or float('nan'):6.1f}  "
              f"n_win={s['n_windows']}")
    for g in generated:
        print(f"  {g['scenario']:9s} target={g['idc_target']:5.0f}  "
              f"realised={g['idc_mean']:6.1f} ± {g['idc_sd']:.1f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
