"""Hình B15 (slide dự phòng): cmdp-pid so với đối chứng hướng năng lượng PABFD.

Đọc report-{arm}-LOW.json do `rl-agent/src/eval/energy_baselines.py report` sinh ra
với `--include pabfd --tag=-pabfd` (data/results/energy-baselines/) và vẽ hai ô LOW — hai ô có tín hiệu vượt nhiễu.
Chạy trên host (cần matplotlib):  python scripts/plot-energy-baselines.py
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "data/results/energy-baselines"
OUT = ROOT / "assets_v2/report/figures/slide/b15-doi-chung-nang-luong"

BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
GRAY, INK, INK2, GRID = "#9b9a96", "#0b0b0b", "#52514e", "#e6e5e1"

NAMES = {"firstfit": "first-fit", "bestfit": "best-fit", "k8s": "Kubernetes mặc định",
         "random": "ngẫu nhiên", "roundrobin": "quay vòng"}
NEW = {"pabfd": ("PABFD", "D")}

# Nhãn đặt tay ở hai ô có điểm sát nhau: (văn bản, dx, dy [pt], căn lề, là-điểm-mới)
LABELS = {
    ("homo", "pabfd"):      ("PABFD = best-fit (trùng từng chữ số)", 60, 22, "left", True),
    ("homo", "bestfit"):    (None, 0, 0, "left", False),          # nhãn đã gộp vào PABFD
    ("homo", "firstfit"):   ("first-fit", 60, -6, "left", False),
    ("hetero", "pabfd"):    ("PABFD", -30, 22, "right", True),
    ("hetero", "bestfit"):  ("best-fit", -25, -16, "right", False),
}

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                     "axes.edgecolor": INK2, "axes.labelcolor": INK2,
                     "xtick.color": INK2, "ytick.color": INK2})


def panel(ax, arm, title):
    r = json.loads((SRC / arm / f"report-{arm}-LOW-pabfd.json").read_text(encoding="utf-8"))
    a = r["aggregate"]
    E = lambda m: a[m]["energy_mean"]
    C = lambda m: a[m]["sla_mean"] / 1e6

    # cmdp-pid: một đường qua 4 ngân sách
    # nối theo thứ tự ngân sách d (chặt → lỏng), không theo điện năng
    budgets = sorted((m for m in a if m.startswith("cmdp-d")), key=lambda m: float(m[6:]))
    ax.plot([E(m) for m in budgets], [C(m) for m in budgets], "-o", color=BLUE, lw=2,
            ms=8, mec="white", mew=2, zorder=5, label="Đề xuất (CMDP + PID), 4 ngân sách d")
    for m in budgets:
        ax.errorbar(E(m), C(m), xerr=a[m]["energy_ci95"], yerr=a[m]["sla_ci95"] / 1e6,
                    color=BLUE, alpha=.25, lw=1, zorder=2)

    # PPO trọng số cố định
    ppo = sorted((m for m in a if m.startswith("ppo-w")), key=lambda m: E(m))
    ax.plot([E(m) for m in ppo], [C(m) for m in ppo], "o", color=AQUA, ms=8, mec="white",
            mew=2, zorder=4, label="PPO trọng số cố định (w = 0,3/0,5/0,7)")

    # 5 heuristic cũ — xám, nhãn trực tiếp
    for m, lab in NAMES.items():
        ax.plot(E(m), C(m), "o", color=GRAY, ms=7, mec="white", mew=1.5, zorder=3)
        if (arm, m) in LABELS:
            continue
        ax.annotate(lab, (E(m), C(m)), xytext=(6, -3), textcoords="offset points",
                    fontsize=8.5, color=INK2)
    for (arm_, m), (text, dx, dy, ha, new) in LABELS.items():
        if arm_ != arm or text is None:
            continue
        ax.annotate(text, (E(m), C(m)), xytext=(dx, dy), textcoords="offset points",
                    ha=ha, va="center", fontsize=8.5, color=ORANGE if new else INK2,
                    weight="bold" if new else "normal", zorder=7,
                    arrowprops=dict(arrowstyle="-", color=ORANGE if new else GRAY, lw=.8,
                                    shrinkA=0, shrinkB=5))

    # 2 đối chứng hướng năng lượng — cam, nhãn + CI
    for m, (lab, mk) in NEW.items():
        ax.errorbar(E(m), C(m), xerr=a[m]["energy_ci95"], yerr=a[m]["sla_ci95"] / 1e6,
                    color=ORANGE, alpha=.35, lw=1, zorder=2)
        ax.plot(E(m), C(m), mk, color=ORANGE, ms=9, mec="white", mew=2, zorder=6)
    ax.plot([], [], "D", color=ORANGE, ms=8, label="Mới: PABFD (hướng năng lượng)")
    ax.plot([], [], "o", color=GRAY, ms=7, label="5 thuật toán kinh nghiệm cũ")

    n = r["aggregate"]["pabfd"]["n"]
    ax.set_title(f"{title} — {n} hạt giống", loc="left", color=INK, fontsize=11)
    ax.set_xlabel("Điện năng (kWh) — thấp hơn tốt hơn")
    ax.set_ylabel("Chi phí trễ hạn C_SLA (triệu) — thấp hơn tốt hơn")
    ax.grid(color=GRID, lw=.8)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


fig, axes = plt.subplots(1, 2, figsize=(13, 5.2))
panel(axes[0], "homo", "Cụm đồng nhất, ngày thưa việc (LOW)")
panel(axes[1], "hetero", "Cụm dị thể CPU–GPU, ngày thưa việc (LOW)")

h, l = axes[0].get_legend_handles_labels()
fig.legend(h, l, loc="lower center", ncol=4, frameon=False, fontsize=9)
fig.text(0.01, 0.995, "Thanh mờ = khoảng tin cậy 95%. PABFD không trội phương pháp đề xuất "
         "trên cả hai trục ở bất kỳ ngân sách nào.", va="top", fontsize=9, color=INK2)
fig.tight_layout(rect=(0, 0.07, 1, 0.97))
OUT.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(OUT.with_suffix(".pdf"))
fig.savefig(OUT.with_suffix(".png"), dpi=200)
print("->", OUT.with_suffix(".pdf"), OUT.with_suffix(".png"))
