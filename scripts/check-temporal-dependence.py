"""Temporal-dependence check for reviewer question 2 (offline, reads CSVs only).

Sequence = tasks sorted by arrival; x_i = log(CPU-work_i) with CPU-work = cores x duration.
(1) Ljung-Box Q over lags 1..10, p-value from chi2(10) AND from a permutation null
    (shuffling the order; robust to the heavy tail).
(2) Size-arrival coupling: Spearman rho between x_i and the local arrival count
    (#arrivals within +-1 h of task i); permutation p-value by shuffling sizes vs times.
"""
import csv, glob, math, sys
import numpy as np

sys.stdout.reconfigure(encoding="utf-8")
H = 10
NPERM = 2000
rng = np.random.default_rng(0)


def load(path, keep_phase=None):
    rows = []
    for r in csv.DictReader(open(path, encoding="utf-8")):
        if keep_phase and r["pod_phase"] not in keep_phase:
            continue
        c = float(r["creation_time"]); s = float(r["scheduled_time"] or c)
        d = float(r["deletion_time"]) - max(c, s)
        if d <= 0:
            continue
        pes = max(1, int(r["cpu_milli"]) // 1000)
        rows.append((c, math.log(pes * d), int(r["num_gpu"]) > 0 or int(r["gpu_milli"]) > 0))
    rows.sort(key=lambda t: t[0])
    t = np.array([a for a, _, _ in rows]); x = np.array([b for _, b, _ in rows])
    return t, x


def acf(x, h):
    x = x - x.mean(); den = (x * x).sum()
    return np.array([(x[:-k] * x[k:]).sum() / den for k in range(1, h + 1)])


def lb_q(x, h):
    n = len(x); r = acf(x, h)
    return n * (n + 2) * (r * r / (n - np.arange(1, h + 1))).sum(), r


def chi2_sf(q, k):
    # regularized upper incomplete gamma Q(k/2, q/2) via series / continued fraction
    a, xx = k / 2.0, q / 2.0
    if xx < a + 1:
        s = term = 1.0 / a; ap = a
        for _ in range(500):
            ap += 1; term *= xx / ap; s += term
            if abs(term) < abs(s) * 1e-15: break
        return 1.0 - s * math.exp(-xx + a * math.log(xx) - math.lgamma(a))
    b = xx + 1 - a; c = 1e300; d = 1 / b; h_ = d
    for i in range(1, 500):
        an = -i * (i - a); b += 2
        d = an * d + b; d = 1e-300 if abs(d) < 1e-300 else d
        c = b + an / c; c = 1e-300 if abs(c) < 1e-300 else c
        d = 1 / d; de = d * c; h_ *= de
        if abs(de - 1) < 1e-15: break
    return math.exp(-xx + a * math.log(xx) - math.lgamma(a)) * h_


def rank(v):
    o = np.argsort(v, kind="mergesort"); r = np.empty(len(v)); r[o] = np.arange(len(v)); return r


def spearman(a, b):
    ra, rb = rank(a), rank(b); ra -= ra.mean(); rb -= rb.mean()
    return (ra * rb).sum() / math.sqrt((ra * ra).sum() * (rb * rb).sum())


def local_count(t, w=3600.0):
    lo = np.searchsorted(t, t - w, "left"); hi = np.searchsorted(t, t + w, "right")
    return hi - lo - 1


def analyse(name, t, x):
    q, r = lb_q(x, H)
    p_chi = chi2_sf(q, H)
    qs = np.array([lb_q(rng.permutation(x), H)[0] for _ in range(NPERM)])
    p_perm = (1 + (qs >= q).sum()) / (NPERM + 1)
    lc = local_count(t)
    rho = spearman(x, lc)
    rs = np.array([spearman(rng.permutation(x), lc) for _ in range(NPERM)])
    p_rho = (1 + (np.abs(rs) >= abs(rho)).sum()) / (NPERM + 1)
    return dict(name=name, n=len(x), r1=r[0], r5=r[4], q=q, p_chi=p_chi, p_perm=p_perm,
                rho=rho, p_rho=p_rho)


res = []
t, x = load("data/alibaba-trace/openb_pod_list_default.csv", {"Running", "Succeeded"})
res.append(analyse("openb gốc (Running+Succeeded), toàn vết tải", t, x))
for f in sorted(glob.glob("data/wm1/homo/REPLAY/seed42.csv")):
    t, x = load(f); res.append(analyse("homo/REPLAY seed42 (cửa sổ thật)", t, x))
for sc in ("LOW", "HIGH", "BURST"):
    for sd in (42, 43, 44, 45, 46):
        t, x = load(f"data/wm1/homo/{sc}/seed{sd}.csv")
        res.append(analyse(f"homo/{sc} seed{sd} (WM-1)", t, x))

print(f"{'chuỗi':48s} {'n':>5s} {'r1':>7s} {'r5':>7s} {'Q(10)':>8s} {'p_chi2':>9s} {'p_perm':>7s} {'rho_sz~den':>10s} {'p':>7s}")
for d in res:
    print(f"{d['name']:48s} {d['n']:5d} {d['r1']:7.3f} {d['r5']:7.3f} {d['q']:8.1f} {d['p_chi']:9.2e} {d['p_perm']:7.4f} {d['rho']:10.3f} {d['p_rho']:7.4f}")
