# ELDAS — Thu thập bằng chứng (3 bước)

> Làm theo thứ tự. Bước 2 tự động hoá toàn bộ phần còn lại.

---

## Bước 1 — Khởi động hệ thống

```powershell
cd "d:\HCMUT-uni\252\DACn\ELDAS"
docker compose up --build -d
# Chờ đến khi cả hai container healthy (~30 s)
docker compose ps
```

**Cần chụp màn hình:**
- `docker compose ps` cho thấy `STATUS = healthy` → lưu làm `figures/screenshot-compose-ps.png`
- Terminal khi build xong → lưu làm `figures/screenshot-build-success.png`

---

## Bước 2 — Chạy tự động (smoke test + baselines + plots)

```powershell
# Cài deps một lần (trên host)
pip install pandas numpy matplotlib

# Chạy toàn bộ — khoảng 15-30 phút
python assets/report/scripts/collect_all_data.py
```

Script sẽ tự động:
1. Chạy smoke test (LOW, 3000 steps) → `data/results/smoke/`
2. Chạy baseline cho LOW / HIGH / BURST × k8s / random → `data/results/baseline-*/`
   - Mỗi policy xuất `metrics.csv` + `summary.json` vào thư mục riêng
3. Phân tích trace Alibaba → tính toán thống kê
4. Sinh tất cả figures → `assets/report/figures/*.pdf`
5. **In ra toàn bộ số liệu** cần điền vào `\TODOnum{...}`

**Nếu muốn bỏ qua Docker (đã có dữ liệu):**
```powershell
python assets/report/scripts/collect_all_data.py --skip-docker
```

**Chụp màn hình sau bước này:**
- Terminal smoke test với 8 dấu ✓ → `figures/screenshot-smoke-pass.png`
- Chạy lại 2 lần cùng seed, `diff` output trống → `figures/screenshot-reproducibility.png`:
  ```powershell
  # Lần 2 (lần 1 đã chạy ở bước 2)
  docker compose run --rm rl-agent python src/baseline_eval.py --scenario HIGH --seed 42 --output /data/results/repro-2
  # So sánh
  diff data/results/baseline-HIGH/random/summary.json data/results/repro-2/baseline-HIGH/random/summary.json
  # Kết quả phải trống → screenshot
  ```

---

## Bước 3 — Điền số liệu + build PDF

### 3a. Điền số vào báo cáo

Copy các số từ output của bước 2 vào `chapters/05-ket-qua.tex`:

| Vị trí trong .tex | Lấy từ |
|---|---|
| Bảng `tab:trace-stats` (§5.4) | Phần "TABLE 5.5" in ra cuối bước 2 |
| Bảng `tab:baseline-results` (§5.3) | Phần "TABLE 5.3" in ra cuối bước 2 |
| Text §5.3 (tỉ lệ %, hệ số ×) | Phần "Ratios" in ra cuối bước 2 |
| Bảng `tab:smoke-results` (§5.2) | Phần "TABLE 5.2" in ra cuối bước 2 |
| Bảng `tab:image-size` (§4.3) | `docker images` output từ bước 1 |

### 3b. Thay placeholder hình trong `05-ket-qua.tex`

Tìm các `\fbox{\parbox...}` và thay bằng `\includegraphics`:

| Hình | File cần có | Thay `\fbox{...}` bằng |
|---|---|---|
| 5.1 compose-ps | `figures/screenshot-compose-ps.png` | `\includegraphics[width=0.85\textwidth]{screenshot-compose-ps.png}` |
| 5.2 smoke-pass | `figures/screenshot-smoke-pass.png` | `\includegraphics[width=0.85\textwidth]{screenshot-smoke-pass.png}` |
| 5.3 energy bar | `figures/baseline-energy.pdf` | `\includegraphics[width=0.7\textwidth]{baseline-energy.pdf}` |
| 5.4 SLA bar | `figures/baseline-sla.pdf` | `\includegraphics[width=0.7\textwidth]{baseline-sla.pdf}` |
| 5.5 time-series | `figures/baseline-energy-time.pdf` | `\includegraphics[width=\textwidth]{baseline-energy-time.pdf}` |
| 5.6 repro diff | `figures/screenshot-reproducibility.png` | `\includegraphics[width=0.7\textwidth]{screenshot-reproducibility.png}` |
| 5.7 GPU hist | `figures/trace-numgpu-hist.pdf` | `\includegraphics[width=0.6\textwidth]{trace-numgpu-hist.pdf}` |
| 5.8 QoS pie | `figures/trace-qos-pie.pdf` | `\includegraphics[width=0.5\textwidth]{trace-qos-pie.pdf}` |

### 3c. Build PDF cuối cùng

```powershell
cd assets/report
pdflatex -output-directory=out main.tex
pdflatex -output-directory=out main.tex   # 2nd pass để refs đúng
```

Kiểm tra: không còn `??` trong PDF, không còn hình placeholder `[Hình N.N]`.

```powershell
# Kiểm tra còn placeholder nào không
grep -rn "TODOnum" chapters/    # phải không có output
```

---

## Cấu trúc output sau bước 2

```
data/results/
├── smoke/
│   ├── metrics.csv          # 9 cột × N dòng (mỗi task một dòng)
│   └── summary.json         # 9 trường tổng hợp
├── baseline-LOW/
│   ├── baseline_results.json        # tóm tắt cả 2 policy (Python-side)
│   ├── k8s/
│   │   ├── metrics.csv      # per-step từ Java MetricsExporter
│   │   └── summary.json
│   └── random/
│       ├── metrics.csv
│       └── summary.json
├── baseline-HIGH/  (tương tự)
└── baseline-BURST/ (tương tự)

assets/report/figures/
├── baseline-energy.pdf         # Fig 5.3
├── baseline-sla.pdf            # Fig 5.4
├── baseline-energy-time.pdf    # Fig 5.5
├── trace-numgpu-hist.pdf       # Fig 5.7
└── trace-qos-pie.pdf           # Fig 5.8
```
