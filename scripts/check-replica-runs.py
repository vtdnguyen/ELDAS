import csv, sys
sys.stdout.reconfigure(encoding="utf-8")


def seq(path, keep=None):
    rows = [r for r in csv.DictReader(open(path, encoding="utf-8"))
            if not keep or r["pod_phase"] in keep]
    rows.sort(key=lambda r: float(r["creation_time"]))
    return [(r["cpu_milli"], r["memory_mib"], r["num_gpu"], r["gpu_milli"]) for r in rows], rows


def same_next(s):
    return sum(1 for a, b in zip(s, s[1:]) if a == b) / (len(s) - 1)


def shuffled_baseline(s):
    # expected share of identical consecutive pairs if order were random
    from collections import Counter
    n = len(s); c = Counter(s)
    return sum(v * (v - 1) for v in c.values()) / (n * (n - 1))


for name, path, keep in [("openb gốc", "data/alibaba-trace/openb_pod_list_default.csv", {"Running", "Succeeded"}),
                         ("homo/REPLAY seed42", "data/wm1/homo/REPLAY/seed42.csv", None),
                         ("homo/LOW seed42", "data/wm1/homo/LOW/seed42.csv", None),
                         ("homo/HIGH seed42", "data/wm1/homo/HIGH/seed42.csv", None)]:
    s, rows = seq(path, keep)
    zero = sum(1 for r in rows if float(r["creation_time"]) == 0.0)
    print(f"{name:22s} n={len(s):5d}  cặp liền kề giống hệt cỡ: {same_next(s)*100:5.1f}%  "
          f"nếu thứ tự ngẫu nhiên: {shuffled_baseline(s)*100:5.1f}%  (creation_time=0: {zero})")
