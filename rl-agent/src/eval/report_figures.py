"""Sinh bốn hình từ dữ liệu cho Chương 9 (H6-H9) — xuất PDF vector cho LaTeX.

Chương 9 hiện có chín bảng và không hình nào. Ba trong bốn hình dưới đây không cần chạy
thêm gì; chỉ H7 cần quỹ đạo lambda, thứ mà ``sweep_budget.py`` không lưu (phải chạy
``train_cmdp.py --trajectory-csv``).

Xuất PDF chứ không PNG: LaTeX nhúng vector thì chữ trong hình giữ nguyên độ nét khi in,
và tệp nhỏ hơn nhiều so với PNG 200 dpi.

Chạy::

    python src/eval/report_figures.py --results /data/results/wm1-v3 \
        --results-5sc /data/results/wm1-v2 --out /data/results/figures
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")                                    # container không có màn hình
import matplotlib.pyplot as plt                          # noqa: E402

# Console Windows mặc định cp1252 và ném UnicodeEncodeError ở chữ có dấu — thường là
# SAU KHI hình đã ghi xong, nên trông như công cụ hỏng trong khi kết quả vẫn đúng.
# Đổi lỗi thành thay ký tự thay vì để nó giết tiến trình.
try:                                                     # pragma: no cover
    sys.stdout.reconfigure(errors="replace")
    sys.stderr.reconfigure(errors="replace")
except (AttributeError, OSError):
    pass

SURFACE, INK, MUTED, GRID = "#fcfcfb", "#1b1b1a", "#6b6b66", "#e5e5e0"
ACCENT = ["#2f6f4f", "#a35b2a", "#3a5a8c", "#8c3a5a", "#6b6b2a"]

# Tên hiển thị trùng với tên trong bảng của báo cáo; mã nội bộ (cmdp-pid, k8s...) không
# được lộ ra hình, nếu không người đọc phải tự đoán "cmdp-pid" là "phương pháp đề xuất".
DISPLAY = {
    "cmdp-pid": "đề xuất",
    "ppo-fixed": "trọng số cố định",
    "firstfit": "first-fit",
    "bestfit": "best-fit",
    "k8s": "Kubernetes",
    "random": "ngẫu nhiên",
    "roundrobin": "quay vòng",
}
# Mỗi họ một màu VÀ một hình dấu riêng: bảy họ mà chỉ năm màu thì hai cặp trùng màu, và
# bản in đen trắng chỉ còn phân biệt được bằng hình dấu.
FAMILY_STYLE = {
    "cmdp-pid": ("#2f6f4f", "o"),
    "ppo-fixed": ("#a35b2a", "o"),
    "firstfit": ("#3a5a8c", "s"),
    "bestfit": ("#5f9ec7", "D"),
    "k8s": ("#8c3a5a", "^"),
    "random": ("#77776f", "v"),
    "roundrobin": ("#b08a1e", "P"),
}


def _name(m: str) -> str:
    return DISPLAY.get(m, m)


def _vn_decimal(ax):
    """Dấu phẩy thập phân trên cả hai trục, như phần chữ của báo cáo."""
    from matplotlib.ticker import ScalarFormatter

    class _Comma(ScalarFormatter):
        def __call__(self, x, pos=None):
            return super().__call__(x, pos).replace(".", ",")

    for axis in (ax.xaxis, ax.yaxis):
        if isinstance(axis.get_major_formatter(), ScalarFormatter):
            axis.set_major_formatter(_Comma())


def _style(ax):
    ax.set_facecolor(SURFACE)
    ax.grid(True, color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelsize=9, length=0)


def _fig(w=6.4, h=4.2):
    f, ax = plt.subplots(figsize=(w, h), dpi=200)
    f.patch.set_facecolor(SURFACE)
    _style(ax)
    return f, ax


def _save(fig, path: Path, name: str):
    path.mkdir(parents=True, exist_ok=True)
    p = path / name
    # Đặt bộ định dạng SAU khi vẽ: đặt trước thì trục phân loại (tên phương pháp, tên
    # kịch bản) mất nhãn chữ, vì matplotlib chỉ gắn nhãn phân loại vào trục còn mặc định.
    for ax in fig.axes:
        _vn_decimal(ax)
    fig.savefig(p, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)
    print(f"  [fig] {p}")
    return p


def _points(root: Path, arm: str, sc: str) -> dict:
    out: dict = {}
    for sub in ("campaign", "sweep", "ppo-fixed"):
        f = root / arm / f"{sub}-{sc}" / "points.jsonl"
        if not f.is_file():
            continue
        for line in f.read_text().splitlines():
            if line.strip():
                r = json.loads(line)
                out.setdefault(r["method"], []).append(r)
    return out


def _family(m: str) -> str:
    if m.startswith("cmdp-d"):
        return "cmdp-pid"
    if m.startswith("ppo-w"):
        return "ppo-fixed"
    return m


# H6 - bien Pareto ----------------------------------------------------------
def h6(root: Path, arm: str, sc: str, out: Path):
    """Một ô: mỗi thuật toán kinh nghiệm là một điểm, hai họ học được là một đường."""
    pts = _points(root, arm, sc)
    if not pts:
        print(f"  [bỏ qua] H6 {arm}/{sc}: không có điểm")
        return
    fig, ax = _fig()
    fams: dict = {}
    for m, rows in pts.items():
        e = sum(r["energy_kwh"] for r in rows) / len(rows)
        c = sum(r["sla_cost"] for r in rows) / len(rows)
        fams.setdefault(_family(m), []).append((e, c, m))

    ci = 0
    for fam, xs in sorted(fams.items()):
        xs.sort()
        col, mk = FAMILY_STYLE.get(fam, (ACCENT[ci % len(ACCENT)], "s"))
        ci += 1
        if len(xs) > 1:                                  # một họ => nối thành đường
            ax.plot([p[0] for p in xs], [p[1] for p in xs], "-", marker=mk, color=col,
                    lw=1.6, ms=4.5, label=_name(fam), zorder=3)
        else:
            ax.plot(xs[0][0], xs[0][1], mk, color=col, ms=7, label=_name(fam), zorder=3)
            ax.annotate(_name(fam), (xs[0][0], xs[0][1]), textcoords="offset points",
                        xytext=(6, 4), fontsize=8, color=MUTED)
    ax.set_xlabel("Điện năng (kWh) — thấp hơn là tốt hơn", color=INK, fontsize=10)
    ax.set_ylabel("Chi phí vi phạm — thấp hơn là tốt hơn", color=INK, fontsize=10)
    ax.set_title(f"Biên Pareto — {arm}/{sc}", color=INK, fontsize=11, loc="left")
    ax.legend(frameon=False, fontsize=8, labelcolor=MUTED)
    _save(fig, out, f"h6-pareto-{arm}-{sc}.pdf")


# H7 - quy dao nhan tu ------------------------------------------------------
def h7(traj_dir: Path, out: Path, tag: str = "homo-LOW"):
    """Mỗi ngân sách một đường lambda; đây là bằng chứng trực quan vòng đã đóng."""
    files = sorted(traj_dir.glob("*.csv"))
    if not files:
        print(f"  [bỏ qua] H7: không có CSV quỹ đạo trong {traj_dir}")
        return
    fig, ax = _fig()
    drawn = 0
    for i, f in enumerate(files):
        with f.open() as fh:
            rows = list(csv.DictReader(fh))
        if not rows or "lambda" not in rows[0]:
            continue
        lam = [float(r["lambda"]) for r in rows]
        # "traj-d0.1604" -> "d = 0,1604" (dấu phẩy thập phân như phần chữ)
        label = f.stem.replace("traj-", "").replace("-", " ")
        if label.startswith("d") and label[1:].replace(".", "", 1).isdigit():
            label = "$d$ = " + label[1:].replace(".", ",")
        ax.plot(range(1, len(lam) + 1), lam, lw=1.6,
                color=ACCENT[i % len(ACCENT)], label=label)
        drawn += 1
    if not drawn:
        plt.close(fig)
        print("  [bỏ qua] H7: CSV không có cột lambda")
        return
    ax.set_xlabel("Lượt cập nhật đối ngẫu", color=INK, fontsize=10)
    ax.set_ylabel(r"Nhân tử $\lambda$", color=INK, fontsize=10)
    ax.set_title("Quỹ đạo nhân tử đối ngẫu theo ngân sách", color=INK,
                 fontsize=11, loc="left")
    ax.legend(frameon=False, fontsize=8, labelcolor=MUTED)
    _save(fig, out, f"h7-lambda-{tag}.pdf")


# H8 - nhan tu tai ngan sach chat theo muc tai -------------------------------
def h8(root: Path, arms, scenarios, out: Path):
    """lambda ở ngân sách chặt nhất, theo kịch bản — giá bóng của ràng buộc."""
    fig, ax = _fig()
    got = False
    for i, arm in enumerate(arms):
        xs, ys = [], []
        for sc in scenarios:
            f = root / arm / f"sweep-{sc}" / "points.jsonl"
            if not f.is_file():
                continue
            rows = [json.loads(l) for l in f.read_text().splitlines() if l.strip()]
            if not rows:
                continue
            dmin = min(r["extra"]["budget_d"] for r in rows)
            lam = [r["extra"]["lambda_final"] for r in rows
                   if abs(r["extra"]["budget_d"] - dmin) < 1e-9]
            if lam:
                xs.append(sc)
                ys.append(sum(lam) / len(lam))
        if xs:
            got = True
            ax.plot(xs, ys, "-o", lw=1.8, ms=6, color=ACCENT[i % len(ACCENT)], label=arm)
    if not got:
        plt.close(fig)
        print("  [bỏ qua] H8: không có dữ liệu sweep")
        return
    ax.set_xlabel("Kịch bản tải", color=INK, fontsize=10)
    ax.set_ylabel(r"$\lambda$ tại ngân sách chặt nhất", color=INK, fontsize=10)
    ax.set_title("Giá bóng của ràng buộc theo mức tải", color=INK,
                 fontsize=11, loc="left")
    ax.legend(frameon=False, fontsize=8, labelcolor=MUTED)
    _save(fig, out, "h8-lambda-vs-load.pdf")


# H9 - the tich chi phoi kem CI bootstrap ------------------------------------
def h9(root: Path, arm: str, sc: str, out: Path):
    boot = root / arm / f"campaign-{sc}" / "hv_bootstrap.json"
    summ = root / arm / f"campaign-{sc}" / "campaign_summary.json"
    if not summ.is_file():
        print(f"  [bỏ qua] H9 {arm}/{sc}: thiếu campaign_summary.json")
        return
    methods = json.loads(summ.read_text())["methods"]

    # hv_bootstrap.py ghi theo HỌ (`per_family`) với hai khoá rời `ci95_low`/`ci95_high`,
    # không phải một cặp lồng nhau. Vẫn chấp nhận vài dạng khác để không vỡ nếu schema đổi.
    lo_hi: dict = {}
    if boot.is_file():
        b = json.loads(boot.read_text())
        blob = b.get("per_family") or b.get("methods") or b.get("families") or {}
        for m, v in blob.items():
            if not isinstance(v, dict):
                continue
            if "ci95_low" in v and "ci95_high" in v:
                lo_hi[m] = (float(v["ci95_low"]), float(v["ci95_high"]))
                continue
            ci = v.get("ci95") or v.get("hv_ci95") or v.get("ci")
            if isinstance(ci, (list, tuple)) and len(ci) == 2:
                lo_hi[m] = (float(ci[0]), float(ci[1]))

    order = sorted(methods.items(), key=lambda kv: kv[1]["hypervolume"])
    names = [m for m, _ in order]
    vals = [v["hypervolume"] for _, v in order]
    err = None
    if lo_hi:
        err = [[max(0.0, v - lo_hi.get(m, (v, v))[0]) for m, v in zip(names, vals)],
               [max(0.0, lo_hi.get(m, (v, v))[1] - v) for m, v in zip(names, vals)]]

    fig, ax = _fig(6.4, 0.42 * len(names) + 1.6)
    cols = [ACCENT[0] if m == "cmdp-pid" else MUTED for m in names]
    ax.barh([_name(m) for m in names], vals, color=cols, height=0.62, xerr=err,
            error_kw={"ecolor": INK, "lw": 1})
    ax.set_xlabel("Thể tích chi phối — cao hơn là tốt hơn", color=INK, fontsize=10)
    ax.set_title(f"Thể tích chi phối — {arm}/{sc}"
                 + ("" if lo_hi else "  (chưa có khoảng tin cậy bootstrap)"),
                 color=INK, fontsize=11, loc="left")
    _save(fig, out, f"h9-hv-{arm}-{sc}.pdf")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results", default="/data/results/wm1-v3")
    ap.add_argument("--results-5sc", default="/data/results/wm1-v2",
                    help="gốc có đủ 5 kịch bản, dùng cho H8")
    ap.add_argument("--traj-dir", default="/data/results/traj")
    ap.add_argument("--out", default="/data/results/figures")
    ap.add_argument("--h6", default="homo:LOW,hetero:LOW,homo:HIGH,hetero:BURST",
                    help="các ô vẽ biên Pareto, dạng arm:scenario")
    args = ap.parse_args(argv)

    root, root5, out = Path(args.results), Path(args.results_5sc), Path(args.out)
    print(f"[fig] sinh hình -> {out}")

    for cell in args.h6.split(","):
        arm, sc = cell.split(":")
        src = root if (root / arm / f"campaign-{sc}").is_dir() else root5
        h6(src, arm, sc, out)

    h7(Path(args.traj_dir), out)
    h8(root5, ("homo", "hetero"), ("LOW", "HIGH", "BURST", "OVERLOAD", "REPLAY"), out)
    for cell in ("homo:LOW", "hetero:LOW"):
        arm, sc = cell.split(":")
        src = root if (root / arm / f"campaign-{sc}").is_dir() else root5
        h9(src, arm, sc, out)
    print("[fig] xong")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
