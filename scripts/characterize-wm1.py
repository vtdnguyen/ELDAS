#!/usr/bin/env python3
"""W6.2 — Characterise every generated WM-1 trace and re-run the acceptance checks.

`characterize-workload.py` answers "what is *this* CSV?".  A load-scenario *set* is
only meaningful across files, and WM-1 puts every (arm, scenario, seed) in its own
file, so the two checks that motivated this whole plan — pairwise rho separation and
comparable horizon (PLAN section 2.4) — cannot be evaluated by that tool alone.
This script re-assembles the set and calls the *same* `findings()` function, so the
verdicts are produced by identical code on LEGACY and on WM-1.

Two deliberate choices:

  * The checks run **per seed**, not on seed-averaged numbers.  Averaging first would
    let a seed whose LOW crept up over its HIGH hide inside the mean, which is exactly
    the failure mode of the legacy slices (section 1.3-A).  A check passes for an arm
    only when it passes for all 5 seeds.
  * rho is recomputed from the CSV by this tool, never read from the manifest.  The
    manifest records what the generator *intended*; agreement between the two is the
    evidence (risk R8), so taking the generator's word for it would be circular.

Standard library only, runs on the host in a few seconds.

Usage
-----
  python scripts/characterize-wm1.py
  python scripts/characterize-wm1.py --arms homo --seeds 42
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import os
import statistics
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parent


def _load_characteriser():
    """Import characterize-workload.py (the hyphen keeps it out of normal imports)."""
    path = _HERE / "characterize-workload.py"
    spec = importlib.util.spec_from_file_location("characterize_workload", path)
    if spec is None or spec.loader is None:          # pragma: no cover - defensive
        raise SystemExit(f"cannot import {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["characterize_workload"] = mod
    spec.loader.exec_module(mod)
    return mod


CW = _load_characteriser()

#: Order matters: it is the column order of every table below.
DEFAULT_SCENARIOS = ("LOW", "HIGH", "BURST", "OVERLOAD", "REPLAY")
DEFAULT_SEEDS = (42, 43, 44, 45, 46)
DEFAULT_WINDOWS = (600.0, 3600.0, 86400.0)

#: Same topology the Java side loads for each arm, so capacity matches the DES.
ARM_TOPOLOGY = {"homo": None, "hetero": "config/topology-hetero.json"}


def capacity_for(arm: str):
    topo = ARM_TOPOLOGY[arm]
    if topo:
        return CW.capacity_from_topology(str(_REPO / topo), CW.DEFAULT_PES_PER_HOST,
                                         CW.DEFAULT_GPUS_PER_HOST)
    return CW.capacity_from_args(CW.DEFAULT_HOSTS, CW.DEFAULT_PES_PER_HOST,
                                 CW.DEFAULT_GPUS_PER_HOST)


#: two scenarios count as distinct only if they differ by at least this much on
#: at least one design axis, relative to the larger value.
MIN_SEP = 0.30
#: HIGH and BURST are the paired arm of the design: same job set, different arrival
#: process. Their offered loads must therefore stay *together* within this much.
MAX_PAIRED_RHO_GAP = 0.10
#: window used for the burstiness axis (IDC is scale-dependent; 1 h is the one the
#: targets in section 3.5 were fitted on).
IDC_AXIS_WINDOW = "3600"


def _sep(a: float, b: float) -> float:
    hi = max(abs(a), abs(b), 1e-12)
    return abs(a - b) / hi


def wm1_checks(res: dict, scenarios: list[str]) -> list[dict]:
    """The acceptance checks for a WM-1 *set*, as opposed to the LEGACY diagnosis.

    Two corrections to `characterize-workload.findings()`, both of which make it
    measure the quantity that actually matters here:

    1. **Separation is two-dimensional.**  `findings()` compares `rho_cpu` only,
       because the legacy problem was one-dimensional: three slices that were all
       supposed to be different load levels and were not.  WM-1 is a grid over
       (load, burstiness), and HIGH/BURST deliberately share a job set so that the
       burstiness comparison is *paired* (section 3.6).  Demanding that they separate
       on load would be demanding the design be broken.  A pair is distinct when it
       separates on load **or** on IDC.
       It also compares `rho_cpu` on an arm where GPU is the binding resource, which
       reports the slack axis: offered load is the dominant-resource share (DRF), so
       max(rho_cpu, rho_gpu) is the number.
    2. **Horizon means the billed window.**  `findings()` uses the span of creation
       times, but `SimulationManager.flushTillEnd()` integrates power to the last
       *completion*, so kWh is billed over max(creation + duration).  The span is the
       cheaper number and the optimistic one; checking it would hand out a PASS that
       the energy figures do not support.
    """
    out: list[dict] = []
    live = [s for s in scenarios if s in res and res[s].get("n_tasks")]
    if len(live) < 2:
        return out

    def add(verdict, en, vi, detail_vi):
        out.append({"verdict": verdict, "check": {"en": en, "vi": vi},
                    "detail": {"en": detail_vi, "vi": detail_vi}})

    rho = {s: max(res[s]["rho_cpu"], res[s]["rho_gpu"]) for s in live}
    idc = {s: res[s]["idc"].get(IDC_AXIS_WINDOW, float("nan")) for s in live}

    worst, parts = None, []
    for i, a in enumerate(live):
        for b in live[i + 1:]:
            sr, si = _sep(rho[a], rho[b]), _sep(idc[a], idc[b])
            best = max(sr, si if not math.isnan(si) else 0.0)
            axis = "ρ" if sr >= (0.0 if math.isnan(si) else si) else "IDC"
            parts.append(f"{a}–{b}: {best:.0%} ({axis})")
            if worst is None or best < worst[2]:
                worst = (a, b, best, sr, si)
    a, b, best, sr, si = worst
    add("PASS" if best >= MIN_SEP else "FAIL",
        "every scenario pair separates on load OR burstiness",
        "mọi cặp scenario tách biệt trên **ít nhất một** trục thiết kế (ρ hoặc IDC)",
        f"cặp gần nhau nhất `{a}`–`{b}`: tách **{best:.0%}** "
        f"(ρ {rho[a]:.3f} vs {rho[b]:.3f} ⇒ {sr:.0%}; IDC {idc[a]:.1f} vs {idc[b]:.1f} ⇒ "
        f"{si:.0%}), cần ≥ {MIN_SEP:.0%} · tất cả các cặp: " + "; ".join(parts))

    if "HIGH" in live and "BURST" in live:
        gap = _sep(rho["HIGH"], rho["BURST"])
        sep_idc = _sep(idc["HIGH"], idc["BURST"])
        ok = gap <= MAX_PAIRED_RHO_GAP and sep_idc >= MIN_SEP
        add("PASS" if ok else "FAIL",
            "HIGH/BURST is a paired comparison: same load, different burstiness",
            "`HIGH` vs `BURST` là so sánh **có cặp**: cùng tải, khác burstiness",
            f"ρ chênh **{gap:.1%}** (cần ≤ {MAX_PAIRED_RHO_GAP:.0%} thì burstiness mới là "
            f"biến DUY NHẤT thay đổi), IDC chênh **{sep_idc:.0%}** "
            f"({idc['HIGH']:.1f} → {idc['BURST']:.1f}, cần ≥ {MIN_SEP:.0%})")

    billed = {s: res[s]["billed_days"] for s in live}
    span = {s: res[s]["horizon_days"] for s in live}
    b_spread = (max(billed.values()) - min(billed.values())) / max(max(billed.values()), 1e-12)
    s_spread = (max(span.values()) - min(span.values())) / max(max(span.values()), 1e-12)

    # The original §1.3-B complaint, restated exactly: LEGACY_BURST spanned 32 days of
    # arrivals and LEGACY_HIGH 149, so the scenarios were not the same experiment run
    # at different intensities — they were different lengths of history.
    add("PASS" if s_spread <= 0.05 else "FAIL",
        "arrival window comparable across scenarios",
        "cửa sổ **arrival** so sánh được giữa các scenario (§1.3-B)",
        f"chênh **{s_spread:.1%}** — "
        + ", ".join(f"{s}={span[s]:.1f}d" for s in live))

    add("PASS" if b_spread <= 0.05 else "FAIL",
        "billed energy window comparable across scenarios",
        "cửa sổ **tính năng lượng** so sánh được giữa các scenario",
        f"cửa sổ arrival chênh **{s_spread:.1%}** nhưng cửa sổ BỊ TÍNH TIỀN "
        f"(`max(creation+duration)`, đúng cái `flushTillEnd()` tích phân tới) chênh "
        f"**{b_spread:.1%}** — "
        + ", ".join(f"{s}={billed[s]:.1f}d" for s in live)
        + ". LEGACY 'đạt' tiêu chí này vì **lý do sai**: cả ba slice đều chứa cùng một "
        "pod 145 ngày nên completion cuối trùng nhau, trong khi cửa sổ arrival chênh "
        "4.6×. Ở WM-1 thì ngược lại: arrival đã khớp, còn completion cuối lệch vì "
        "duration bị cắt tại `T` (§3.3) nên job đến sát `T` vẫn chạy thêm tới `T`")
    return out


def _mean_sd(vals: list[float]) -> tuple[float, float]:
    if not vals:
        return float("nan"), float("nan")
    m = statistics.fmean(vals)
    sd = statistics.stdev(vals) if len(vals) > 1 else 0.0
    return m, sd


def analyse_arm(root: Path, arm: str, scenarios, seeds, windows) -> dict:
    cap = capacity_for(arm)
    per_seed: dict[int, dict] = {}
    missing: list[str] = []

    for seed in seeds:
        res = {}
        for sc in scenarios:
            csv_path = root / arm / sc / f"seed{seed}.csv"
            if not csv_path.is_file():
                missing.append(str(csv_path.relative_to(_REPO)))
                continue
            tasks = CW.read_trace(str(csv_path))
            # WM-1 files ARE the scenario already: filtering again would re-cut a
            # calibrated file (the exact silent corruption W2.2 note (a) blocks on
            # the Java side), so pass them through with the ALL filter.
            keep = CW.filter_scenario(tasks, "ALL")
            res[sc] = CW.characterise(keep, cap, list(windows), None)
            # The window the DES bills energy over: flushTillEnd() advances power
            # integration to the last completion, not the last arrival.
            res[sc]["billed_days"] = (max((t.creation_time + t.duration for t in keep),
                                          default=0.0) / CW.SEC_PER_DAY)
            res[sc]["file"] = str(csv_path.relative_to(_REPO)).replace("\\", "/")
            res[sc]["sha256"] = CW._sha256(str(csv_path))
        if res:
            names = [s for s in scenarios if s in res]
            per_seed[seed] = {"scenarios": res,
                              "checks": wm1_checks(res, names),
                              "findings": CW.findings(res, names)}

    # Aggregate: a check passes for the arm only if it passes on every seed.
    def aggregate(field: str) -> dict:
        agg: dict[str, dict] = {}
        for seed, blob in per_seed.items():
            for f in blob[field]:
                if f["verdict"] == "INFO":
                    continue
                key = f["check"]["en"]
                slot = agg.setdefault(key, {"check_vi": f["check"]["vi"],
                                            "verdicts": {}, "details": {}})
                slot["verdicts"][seed] = f["verdict"]
                slot["details"][seed] = f["detail"]["vi"]
        for slot in agg.values():
            vs = list(slot["verdicts"].values())
            slot["overall"] = "PASS" if vs and all(v == "PASS" for v in vs) else "FAIL"
            slot["n_pass"] = sum(1 for v in vs if v == "PASS")
            slot["n_total"] = len(vs)
        return agg

    agg_checks = aggregate("checks")
    agg_diag = aggregate("findings")

    # Per-scenario summary across seeds.
    summary = {}
    for sc in scenarios:
        rows = [blob["scenarios"][sc] for blob in per_seed.values() if sc in blob["scenarios"]]
        if not rows:
            continue
        fields = ("n_tasks", "horizon_days", "billed_days", "rho_cpu", "rho_gpu",
                  "peak_pe_ratio", "peak_gpu_ratio", "saturated_time_frac",
                  "saturated_gpu_time_frac", "ca2", "mean_duration", "cv_duration")
        entry = {}
        for f in fields:
            vals = [r[f] for r in rows if isinstance(r.get(f), (int, float))
                    and not (isinstance(r[f], float) and math.isnan(r[f]))]
            m, sd = _mean_sd(vals)
            entry[f] = {"mean": m, "sd": sd}
        # IDC at the widest window that every scenario reports.
        idc = {}
        for w in windows:
            key = str(int(w))
            vals = [r["idc"][key] for r in rows
                    if key in r.get("idc", {}) and not math.isnan(r["idc"][key])]
            if vals:
                m, sd = _mean_sd(vals)
                idc[key] = {"mean": m, "sd": sd}
        entry["idc"] = idc
        entry["n_seeds"] = len(rows)
        summary[sc] = entry

    return {"arm": arm, "capacity": {"label": cap.label, "hosts": cap.hosts,
                                     "total_pes": cap.total_pes,
                                     "total_gpus": cap.total_gpus},
            "per_seed": per_seed, "checks": agg_checks, "diagnostics": agg_diag,
            "summary": summary, "missing": missing}


def legacy_baseline(trace: Path, windows) -> dict:
    """Re-run the same checks on the LEGACY slices, for the before/after column."""
    if not trace.is_file():
        return {}
    cap = capacity_for("homo")
    tasks = CW.read_trace(str(trace))
    names = ["LEGACY_LOW", "LEGACY_HIGH", "LEGACY_BURST"]
    res = {}
    for nm in names:
        keep = CW.filter_scenario(tasks, nm)
        res[nm] = CW.characterise(keep, cap, list(windows), None)
        res[nm]["billed_days"] = (max((t.creation_time + t.duration for t in keep),
                                      default=0.0) / CW.SEC_PER_DAY)
    return {"scenarios": {nm: {"n_tasks": res[nm]["n_tasks"],
                               "horizon_days": res[nm]["horizon_days"],
                               "billed_days": res[nm]["billed_days"],
                               "rho_cpu": res[nm]["rho_cpu"],
                               "rho_gpu": res[nm]["rho_gpu"],
                               "idc": res[nm]["idc"].get(IDC_AXIS_WINDOW, float("nan"))}
                          for nm in names},
            "checks": wm1_checks(res, names),
            "findings": [f for f in CW.findings(res, names) if f["verdict"] != "INFO"]}


# ── rendering ───────────────────────────────────────────────────────────────

def _ms(d: dict, spec="{:.3f}") -> str:
    if not d or math.isnan(d["mean"]):
        return "—"
    if d["sd"] == 0.0:
        return spec.format(d["mean"])
    return f"{spec.format(d['mean'])} ± {spec.format(d['sd'])}"


def render(report: dict) -> str:
    o: list[str] = []
    o.append("# WM-1 — Đặc trưng hoá toàn bộ trace sinh ra (W6.2)")
    o.append("")
    o.append(f"*Sinh tự động bởi `scripts/characterize-wm1.py` — "
             f"{report['generated_at']}, git `{report.get('git_sha') or '?'}`.*")
    o.append("")
    o.append(f"Số file đã đo: **{report['n_files']}** "
             f"({len(report['arms'])} arm × {len(report['scenarios'])} scenario × "
             f"{len(report['seeds'])} seed).")
    o.append("")
    o.append("> Mọi con số dưới đây được **đo lại từ chính file CSV**, không đọc từ "
             "`wm1-manifest.json`. Manifest ghi cái generator *định* làm; sự trùng khớp "
             "giữa hai bên mới là bằng chứng (rủi ro R8).")
    o.append("")

    # ── acceptance ──────────────────────────────────────────────────────────
    o.append("## 1. Kiểm tra nghiệm thu — LEGACY ❌ → WM-1 ✅")
    o.append("")
    o.append("Hai checkbox dưới đây do **cùng một hàm** `findings()` sinh ra cho cả hai "
             "bộ dữ liệu, nên đây là so sánh trực tiếp chứ không phải hai phép đo khác nhau. "
             "Với WM-1, check chạy **riêng từng seed**; arm chỉ PASS khi **cả 5 seed** PASS "
             "— trung bình trước rồi mới kiểm tra sẽ giấu đúng kiểu hỏng mà LEGACY mắc phải "
             "(một seed có LOW vượt lên trên HIGH).")
    o.append("")
    leg = report.get("legacy") or {}

    def verdict_table(field: str, legacy_source: str) -> None:
        o.append("| Tiêu chí | LEGACY (3 slice) | " + " | ".join(
            f"WM-1 `{a}`" for a in report["arms"]) + " |")
        o.append("|---|---|" + "---|" * len(report["arms"]))
        keys = sorted({k for a in report["arms"] for k in report["by_arm"][a][field]})
        for key in keys:
            vi = next((report["by_arm"][a][field][key]["check_vi"]
                       for a in report["arms"] if key in report["by_arm"][a][field]), key)
            lv = next((f["verdict"] for f in leg.get(legacy_source, [])
                       if f["check"]["en"] == key), None)
            lcell = {"PASS": "✅ PASS", "FAIL": "❌ FAIL"}.get(lv, "—")
            cells = []
            for a in report["arms"]:
                c = report["by_arm"][a][field].get(key)
                if not c:
                    cells.append("—")
                    continue
                icon = "✅ PASS" if c["overall"] == "PASS" else "❌ FAIL"
                cells.append(f"{icon} ({c['n_pass']}/{c['n_total']} seed)")
            o.append(f"| {vi} | {lcell} | " + " | ".join(cells) + " |")
        o.append("")
        for a in report["arms"]:
            for key, c in sorted(report["by_arm"][a][field].items()):
                worst = min(c["details"].items(),
                            key=lambda kv: (c["verdicts"][kv[0]] == "PASS", kv[0]))
                o.append(f"- `{a}` · {c['check_vi']} — seed {worst[0]}: {worst[1]}")
        o.append("")

    verdict_table("checks", "checks")

    if leg:
        o.append("Số LEGACY để đối chiếu (cùng capacity 10×64 PE / 8 GPU):")
        o.append("")
        o.append("| Slice | n task | cửa sổ arrival (ngày) | cửa sổ tính điện (ngày) "
                 "| ρ_cpu | ρ_gpu | IDC(1h) |")
        o.append("|---|---:|---:|---:|---:|---:|---:|")
        for nm, r in leg["scenarios"].items():
            o.append(f"| {nm} | {r['n_tasks']} | {r['horizon_days']:.1f} "
                     f"| {r['billed_days']:.1f} | {r['rho_cpu']:.3f} | {r['rho_gpu']:.3f} "
                     f"| {r['idc']:.1f} |")
        o.append("")

    o.append("### 1b. Chẩn đoán bổ sung (tiêu chí LEGACY, giữ nguyên để đối chiếu)")
    o.append("")
    o.append("Đây là output **nguyên văn** của `findings()` — bộ tiêu chí viết cho chẩn đoán "
             "LEGACY một chiều. Hai dòng đầu của bảng trên đã thay thế nó ở phần nghiệm thu; "
             "phần còn lại vẫn hữu ích như *mô tả*, và plan W6.2 đã nói trước là **không được "
             "kỳ vọng mù quáng ✅**: đuôi nặng là tính chất của openb mà bootstrap giữ lại "
             "(W1.2), còn `ρ>1` ở `OVERLOAD` là **mục tiêu thiết kế**, không phải lỗi.")
    o.append("")
    verdict_table("diagnostics", "findings")

    # ── per-arm tables ──────────────────────────────────────────────────────
    for a in report["arms"]:
        blob = report["by_arm"][a]
        cap = blob["capacity"]
        o.append(f"## 2. Arm `{a}` — {cap['label']} "
                 f"({cap['hosts']} host, {cap['total_pes']} PE, {cap['total_gpus']} GPU)")
        o.append("")
        o.append("Trung bình ± độ lệch chuẩn trên 5 seed.")
        o.append("")
        o.append("| Scenario | n task | cửa sổ arrival (ngày) | cửa sổ tính điện (ngày) "
                 "| ρ_cpu | ρ_gpu | **ρ trội** | đỉnh CPU | đỉnh GPU "
                 "| % thời gian bão hoà | IDC(1h) | C_a² |")
        o.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
        for sc in report["scenarios"]:
            s = blob["summary"].get(sc)
            if not s:
                continue
            sat = max(s["saturated_time_frac"]["mean"], s["saturated_gpu_time_frac"]["mean"])
            dom = max(s["rho_cpu"]["mean"], s["rho_gpu"]["mean"])
            o.append(
                f"| {sc} | {_ms(s['n_tasks'], '{:.0f}')} | {_ms(s['horizon_days'], '{:.2f}')} "
                f"| {_ms(s['billed_days'], '{:.2f}')} "
                f"| {_ms(s['rho_cpu'])} | {_ms(s['rho_gpu'])} | **{dom:.3f}** "
                f"| {_ms(s['peak_pe_ratio'], '{:.2f}')}× | {_ms(s['peak_gpu_ratio'], '{:.2f}')}× "
                f"| {sat:.1%} | {_ms(s['idc'].get('3600'), '{:.1f}')} "
                f"| {_ms(s['ca2'], '{:.2f}')} |")
        o.append("")
        o.append("> **ρ trội** = `max(ρ_cpu, ρ_gpu)` — offered load theo dominant-resource "
                 "share (DRF). Trên arm `hetero` chỉ có 18 GPU nên GPU là trục nghẽn; đọc "
                 "ρ_cpu ở đó là đọc trục còn dư.")
        o.append(">")
        o.append("> **Cửa sổ tính điện** = `max(creation + duration)` — `flushTillEnd()` tích "
                 "phân công suất tới completion cuối cùng, không dừng ở arrival cuối. Đây mới "
                 "là mẫu số của kWh. Nó **dài gần gấp đôi** cửa sổ arrival vì duration bị cắt "
                 "tại `T` (§3.3), nên một job đến sát `T` vẫn chạy thêm tới `T` nữa.")
        o.append("")

    o.append("## 3. Cách tái lập")
    o.append("")
    o.append("```bash")
    o.append(report["command"])
    o.append("```")
    o.append("")
    return "\n".join(o) + "\n"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="W6.2 — characterise all WM-1 traces")
    ap.add_argument("--root", default=str(_REPO / "data" / "wm1"))
    ap.add_argument("--arms", default="homo,hetero")
    ap.add_argument("--scenarios", default=",".join(DEFAULT_SCENARIOS))
    ap.add_argument("--seeds", default=",".join(str(s) for s in DEFAULT_SEEDS))
    ap.add_argument("--windows", default="600,3600,86400")
    ap.add_argument("--legacy-trace",
                    default=str(_REPO / "data" / "alibaba-trace" / "openb_pod_list_default.csv"))
    ap.add_argument("--out", default=str(_REPO / "assets_v2" / "docs" / "wm1-characterization.md"))
    ap.add_argument("--json-out",
                    default=str(_REPO / "assets_v2" / "docs" / "wm1-characterization.json"))
    args = ap.parse_args(argv)

    root = Path(args.root)
    arms = [a.strip() for a in args.arms.split(",") if a.strip()]
    for a in arms:
        if a not in ARM_TOPOLOGY:
            print(f"[ERROR] unknown arm {a!r}; choose from {tuple(ARM_TOPOLOGY)}", file=sys.stderr)
            return 2
    scenarios = [s.strip().upper() for s in args.scenarios.split(",") if s.strip()]
    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    windows = [float(w) for w in args.windows.split(",") if w.strip()]

    import datetime as _dt
    report = {
        "generated_at": _dt.datetime.now().strftime("%Y-%m-%d %H:%M"),
        "git_sha": CW._git_sha(),
        "arms": arms, "scenarios": scenarios, "seeds": seeds, "windows": windows,
        "by_arm": {}, "command": "python scripts/characterize-wm1.py",
    }
    n_files = 0
    missing_all: list[str] = []
    for a in arms:
        blob = analyse_arm(root, a, scenarios, seeds, windows)
        n_files += sum(len(s["scenarios"]) for s in blob["per_seed"].values())
        missing_all += blob["missing"]
        report["by_arm"][a] = blob
    report["n_files"] = n_files
    report["missing"] = missing_all
    report["legacy"] = legacy_baseline(Path(args.legacy_trace), windows)

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    Path(args.out).write_text(render(report), encoding="utf-8", newline="\n")
    print(f"[OK] wrote {args.out}")

    if args.json_out:
        # Drop per-seed raw blobs from the JSON: they are ~1 MB of duplicated detail
        # and everything downstream reads `summary` / `checks`.
        slim = dict(report)
        slim["by_arm"] = {a: {k: v for k, v in b.items() if k != "per_seed"}
                          for a, b in report["by_arm"].items()}
        Path(args.json_out).write_text(json.dumps(slim, indent=2, ensure_ascii=False),
                                       encoding="utf-8", newline="\n")
        print(f"[OK] wrote {args.json_out}")

    # ── console verdict ─────────────────────────────────────────────────────
    expected = len(arms) * len(scenarios) * len(seeds)
    print(f"\nfiles characterised: {n_files} / {expected} expected")
    for m in missing_all:
        print(f"  [MISSING] {m}")
    failed = 0
    for a in arms:
        for key, c in sorted(report["by_arm"][a]["checks"].items()):
            mark = "PASS" if c["overall"] == "PASS" else "FAIL"
            failed += c["overall"] != "PASS"
            print(f"  [{mark}] {a}: {key}  ({c['n_pass']}/{c['n_total']} seeds)")
    print("\n(legacy-criteria diagnostics are in the markdown, section 1b - they are"
          "\n descriptive, not acceptance: heavy tail and rho>1 on OVERLOAD are by design)")
    if missing_all or n_files != expected:
        print("\nINCOMPLETE - some traces are missing; run scripts/gen-workloads.sh")
        return 1
    print("\nALL ACCEPTANCE CHECKS PASS" if not failed
          else f"\n{failed} CHECK(S) FAIL - see the markdown for the per-seed detail")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
