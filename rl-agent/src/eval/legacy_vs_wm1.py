"""W6.5 — LEGACY vs WM-1: kết luận nào giữ, kết luận nào đổi.

Chỉ so những đại lượng **thực sự so được** giữa hai bộ dữ liệu tải, và đó là một danh
sách ngắn hơn nhiều so với cảm giác ban đầu:

* **kWh và C_SLA tuyệt đối thì KHÔNG.** Hai bộ có số tác vụ, khối lượng công việc và cửa
  sổ thời gian khác nhau, nên một con số kWh của LEGACY và một của WM-1 không nằm trên
  cùng một thang. Đặt cạnh nhau là vô nghĩa.
* **Tỉ số thì CÓ.** Độ trải giữa các thuật toán kinh nghiệm — cao nhất chia thấp nhất —
  là đại lượng không đơn vị, và nó chính là "dư địa mà một bộ lập lịch đa mục tiêu có để
  làm việc". Đây là câu hỏi trung tâm của W6.5.
* **Thứ hạng thì CÓ.** Nếu LEGACY và WM-1 xếp năm thuật toán theo cùng một thứ tự thì mô
  hình tải mới không đổi kết luận nào về phương pháp; nếu thứ tự đảo thì có.

Chỉ dùng **thuật toán kinh nghiệm**, có chủ đích: chúng độc lập với hàm phần thưởng, nên
so được xuyên qua lần sửa §19. Chính sách học được của LEGACY huấn luyện trên phần thưởng
đi ngược đại lượng chấm điểm, nên đưa chúng vào đây sẽ so một chính sách hỏng với một
chính sách đúng và gọi chênh lệch đó là "ảnh hưởng của mô hình tải".

Chạy::

    python src/eval/legacy_vs_wm1.py --out /data/results/w65
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

HEURISTICS = ("firstfit", "bestfit", "k8s", "random", "roundrobin")

#: t(0.975, df) — khớp eval/aggregate.py. df = n-1, KHÔNG phải n (Lưu ý #28).
_T95 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 9: 2.262, 14: 2.145}


def _t95(n: int) -> float:
    return _T95.get(n - 1, 1.96)


def load_points(path: Path) -> dict[str, list[dict]]:
    """``points.jsonl`` → ``{method: [row, ...]}``, chỉ giữ thuật toán kinh nghiệm."""
    out: dict[str, list[dict]] = {}
    if not path.is_file():
        return out
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if row.get("method") in HEURISTICS:
            out.setdefault(row["method"], []).append(row)
    return out


def summarise(points: dict[str, list[dict]]) -> dict:
    """Trung bình theo method + hai đại lượng KHÔNG ĐƠN VỊ so được xuyên bộ dữ liệu."""
    per = {}
    for m, rows in points.items():
        e = [r["energy_kwh"] for r in rows]
        c = [r["sla_cost"] for r in rows]
        n = len(e)
        per[m] = {
            "n": n,
            "energy_mean": statistics.fmean(e),
            "energy_ci95": (_t95(n) * statistics.stdev(e) / n ** 0.5) if n > 1 else None,
            "sla_mean": statistics.fmean(c),
            "sla_ci95": (_t95(n) * statistics.stdev(c) / n ** 0.5) if n > 1 else None,
        }
    if not per:
        return {}
    es = [v["energy_mean"] for v in per.values()]
    cs = [v["sla_mean"] for v in per.values()]
    return {
        "per_method": per,
        # Độ trải = dư địa. Đây là con số W6.5 tồn tại để trả lời.
        "energy_spread": max(es) / max(min(es), 1e-12),
        "sla_spread": max(cs) / max(min(cs), 1e-12),
        "rank_energy": [m for m, _ in sorted(per.items(), key=lambda kv: kv[1]["energy_mean"])],
        "rank_sla": [m for m, _ in sorted(per.items(), key=lambda kv: kv[1]["sla_mean"])],
    }


def spearman(a: list[str], b: list[str]) -> float | None:
    """Tương quan hạng giữa hai thứ tự xếp hạng trên cùng tập phương pháp."""
    common = [m for m in a if m in b]
    n = len(common)
    if n < 3:
        return None
    ra = {m: a.index(m) for m in common}
    rb = {m: b.index(m) for m in common}
    d2 = sum((ra[m] - rb[m]) ** 2 for m in common)
    return 1 - 6 * d2 / (n * (n * n - 1))


def compare(legacy: dict, wm1: dict) -> dict:
    """Ghép hai bản tóm tắt và nói thẳng cái gì đổi."""
    if not legacy or not wm1:
        return {"error": "thiếu một trong hai bộ điểm"}
    rho_e = spearman(legacy["rank_energy"], wm1["rank_energy"])
    rho_c = spearman(legacy["rank_sla"], wm1["rank_sla"])
    return {
        "energy_spread": {"legacy": legacy["energy_spread"], "wm1": wm1["energy_spread"]},
        "sla_spread": {"legacy": legacy["sla_spread"], "wm1": wm1["sla_spread"]},
        "rank_energy": {"legacy": legacy["rank_energy"], "wm1": wm1["rank_energy"],
                        "spearman": rho_e},
        "rank_sla": {"legacy": legacy["rank_sla"], "wm1": wm1["rank_sla"],
                     "spearman": rho_c},
        # Thứ hạng giữ nguyên (ρ cao) nhưng dư địa đổi mạnh = mô hình tải KHÔNG đổi
        # "ai thắng", nhưng đổi hẳn "thắng bao nhiêu" - và đó mới là điều quan trọng.
        "verdict_ranking": ("giữ nguyên" if (rho_e or 0) >= 0.8 else "ĐỔI"),
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--legacy-results", default="/data/results")
    ap.add_argument("--wm1-results", default="/data/results/wm1-v3")
    ap.add_argument("--arm", default="homo")
    ap.add_argument("--pairs", default="LOW:LOW,HIGH:HIGH,BURST:BURST",
                    help="cặp <kịch bản LEGACY>:<kịch bản WM-1> so với nhau")
    ap.add_argument("--out", default="/data/results/w65")
    args = ap.parse_args(argv)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    report: dict = {"arm": args.arm, "pairs": {}}
    lines = ["# W6.5 — LEGACY vs WM-1 (chỉ thuật toán kinh nghiệm)", "",
             "Chỉ so đại lượng **không đơn vị**: kWh và C_SLA tuyệt đối của hai bộ dữ liệu",
             "tải không nằm trên cùng một thang (khác số tác vụ, khối lượng, cửa sổ thời gian).",
             ""]

    for pair in args.pairs.split(","):
        lg_sc, wm_sc = (pair.split(":") + [pair])[:2]
        lg = summarise(load_points(Path(args.legacy_results) / f"campaign-{lg_sc}" / "points.jsonl"))
        wm = summarise(load_points(Path(args.wm1_results) / args.arm / f"campaign-{wm_sc}" / "points.jsonl"))
        cmp_ = compare(lg, wm)
        report["pairs"][pair] = {"legacy": lg, "wm1": wm, "comparison": cmp_}

        lines += [f"## LEGACY `{lg_sc}`  vs  WM-1 `{args.arm}/{wm_sc}`", ""]
        if "error" in cmp_:
            lines += [f"⚠️ {cmp_['error']}", ""]
            continue
        lines += [
            "| Đại lượng (không đơn vị) | LEGACY | WM-1 |",
            "|---|---:|---:|",
            f"| Độ trải điện năng giữa 5 thuật toán | {cmp_['energy_spread']['legacy']:.3f}× | {cmp_['energy_spread']['wm1']:.3f}× |",
            f"| Độ trải chi phí vi phạm | {cmp_['sla_spread']['legacy']:.2f}× | {cmp_['sla_spread']['wm1']:.2f}× |",
            "",
            f"- Thứ hạng theo điện năng — LEGACY: `{' < '.join(cmp_['rank_energy']['legacy'])}`",
            f"- Thứ hạng theo điện năng — WM-1&nbsp;&nbsp;: `{' < '.join(cmp_['rank_energy']['wm1'])}`",
            f"- Tương quan hạng: **{cmp_['rank_energy']['spearman']}** ⇒ thứ hạng **{cmp_['verdict_ranking']}**",
            "",
        ]

    # encoding tường minh: mặc định của Windows là cp1252 và sẽ nổ ở chữ có dấu.
    (out / "legacy_vs_wm1.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    (out / "legacy_vs_wm1.md").write_text("\n".join(lines), encoding="utf-8")
    # Console Windows mặc định cp1252 và ném UnicodeEncodeError giữa chừng, SAU KHI tệp
    # đã ghi xong — trông như công cụ hỏng trong khi kết quả vẫn đúng và đầy đủ.
    enc = sys.stdout.encoding or "utf-8"
    for ln in lines:
        print(ln.encode(enc, "replace").decode(enc))
    print(f"\n[w65] -> {out / 'legacy_vs_wm1.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
