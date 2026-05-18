# ELDAS — Run Guide

Tài liệu tham khảo nhanh: chỉ những lệnh cần dùng và mỗi lệnh làm gì. Không có giải thích lý thuyết — cho phần đó xem `CLAUDE.md` (root) hoặc báo cáo trong `assets/report/`.

---

## 1. Yêu cầu

| Thành phần       | Phiên bản tối thiểu | Ghi chú                                 |
|------------------|---------------------|-----------------------------------------|
| Docker Desktop   | 24.x (Compose v2)   | bắt buộc                                |
| RAM / Disk       | 4 GB / 2 GB         | image core ~600 MB; +600 MB monitoring  |
| Trace data       | có sẵn ở `data/alibaba-trace/` | thiếu → chạy `bash scripts/download-trace.sh` |

---

## 2. Cấu hình `.env`

```env
PY4J_PORT=25333
ENERGY_WEIGHT=0.8           # default Phase-2 sẽ sweep
SLA_WEIGHT=0.2
RANDOM_SEED=42              # dùng chung Java + Python
WANDB_API_KEY=              # trống → log offline
MONITORING_ENABLED=true     # Phase 1.5; false → JVM exporter no-op
METRICS_PORT=9091
GF_ADMIN_USER=admin         # optional override
GF_ADMIN_PASSWORD=admin
```

---

## 3. Docker — vòng đời container

```powershell
# Build cả hai service core
docker compose build

# Khởi động core stack (cloudsim-java + rl-agent)
docker compose up --build              # foreground
docker compose up -d cloudsim-java     # chỉ Java service, background

# Trạng thái + logs
docker compose ps
docker compose logs -f cloudsim-java

# Mở shell trong container
docker compose run --rm rl-agent bash
docker compose run --rm --no-deps --entrypoint bash cloudsim-java

# Dừng
docker compose down                    # giữ volume
docker compose down -v                 # XOÁ luôn data/results — cẩn thận
```

---

## 4. Tests

### 4.1. Java validation (offline, không cần Python)

```powershell
# Rebuild image trước nếu source Java đổi
docker compose build cloudsim-java

# Chạy ValidationRunner — 16 bug-regression assertion (B1–B10)
docker compose run --rm --no-deps --entrypoint java cloudsim-java `
    -cp simulation.jar sim.ValidationRunner

# Chạy MetricsRegistryTest — test riêng Prometheus exporter
docker compose run --rm --no-deps --entrypoint java cloudsim-java `
    -cp simulation.jar sim.MetricsRegistryTest                          # disabled mode
docker compose run --rm --no-deps -e MONITORING_ENABLED=true `
    --entrypoint java cloudsim-java -cp simulation.jar sim.MetricsRegistryTest  # enabled mode
```

Exit code 0 = tất cả assertion PASS. Output có `[PASS] / [FAIL]` từng dòng.

### 4.2. Python smoke test (full RL loop qua Py4J)

```powershell
# Khởi động Java service trước, đợi healthy
docker compose up -d cloudsim-java

# Smoke test — 8 pha (connection → reset → step → reward → done → export)
docker compose run --rm rl-agent python src/smoke_test.py
docker compose run --rm rl-agent python src/smoke_test.py --scenario LOW --max-steps 100
```

### 4.3. Baseline evaluation

```powershell
# Chạy K8s + Random trên 1 kịch bản
docker compose run --rm rl-agent python src/baseline_eval.py --scenario HIGH --seed 42

# Chạy cả 3 kịch bản (sinh data cho báo cáo)
foreach ($scen in 'LOW','HIGH','BURST') {
    docker compose run --rm rl-agent python src/baseline_eval.py `
        --scenario $scen --output /data/results/baseline-$scen
}
```

Output: `data/results/baseline-<SCEN>/{k8s,random}/{metrics.csv, summary.json}` và `data/results/baseline_results.json` (so sánh).

### 4.4. Unit tests Python (offline, host)

```powershell
pip install numpy gymnasium pytest py4j wandb
python -m pytest rl-agent/tests/ -v
```

---

## 5. Monitoring stack (Phase 1.5, opt-in)

### 5.1. Bật/tắt

```powershell
# Bật cả 3 container (cloudsim + prometheus + grafana)
docker compose --profile monitoring up -d --build

# Khởi động chỉ stack quan sát (cloudsim đã chạy sẵn)
docker compose --profile monitoring up -d prometheus grafana

# Tắt — chỉ container monitoring, giữ data
docker compose --profile monitoring stop prometheus grafana

# Tắt + xoá volume Prometheus/Grafana
docker compose --profile monitoring down -v
```

### 5.2. URLs

| Service              | Host URL                                          | Credentials      |
|----------------------|---------------------------------------------------|------------------|
| Java exporter        | http://localhost:9091/metrics                     | —                |
| Prometheus           | http://localhost:9090                             | —                |
| Grafana              | http://localhost:3000                             | `admin / admin`  |
| Overview dashboard   | http://localhost:3000/d/eldas-live                | đã provisioned   |
| Host detail (param.) | http://localhost:3000/d/eldas-host-detail         | đã provisioned   |
| Scheduler comparison | http://localhost:3000/d/eldas-scheduler-comparison| đã provisioned   |

Overview dùng **single-select** `$scenario` + `$scheduler` (chọn 1 run đang xem). Host-detail thêm biến `$host` (0–9) — chọn host nào sẽ drill xuống đó. Hai dashboard có link qua lại ở góc trên.

### 5.3. Verify end-to-end

```bash
# Smoke test 13 check — Java exporter → Prometheus → Grafana proxy
bash scripts/check-monitoring.sh

# Override endpoint khi chạy từ container khác
ELDAS_PROM_URL=http://prometheus:9090 \
ELDAS_GRAFANA_URL=http://grafana:3000 \
    bash scripts/check-monitoring.sh
```

### 5.4. Truy vấn nhanh

```powershell
# Liệt kê các metric ELDAS
curl -s http://localhost:9091/metrics | Select-String '^eldas_'

# PromQL trực tiếp
curl -s "http://localhost:9090/api/v1/query?query=eldas_host_cpu_util"

# Health datasource Grafana qua proxy
$cred = [Convert]::ToBase64String([Text.Encoding]::ASCII.GetBytes('admin:admin'))
curl -s -H "Authorization: Basic $cred" `
    http://localhost:3000/api/datasources/uid/eldas-prometheus/health
```

---

## 6. Scripts tiện ích

```bash
# Tải Alibaba GPU Cluster Trace v2023 nếu thiếu data
bash scripts/download-trace.sh

# Kiểm tra monitoring stack (mục §5.3)
bash scripts/check-monitoring.sh
```

```powershell
# Pipeline thu thập số liệu cho báo cáo (smoke + 3 baseline + plots)
pip install pandas numpy matplotlib   # cài 1 lần trên host
python assets/report/scripts/collect_all_data.py     # ~15–30 phút

# Vẽ figures riêng (chạy sau khi đã có metrics.csv)
python assets/report/scripts/plot_baseline.py
python assets/report/scripts/plot_chapter5.py
python assets/report/scripts/analyze-trace.py
```

---

## 7. Build báo cáo LaTeX

```powershell
# Từ thư mục assets/report/
.\build.ps1
# Output: assets/report/out/main.pdf
```

---

## 8. Output files Java tạo ra

| File                                         | Nguồn                          | Nội dung                       |
|----------------------------------------------|--------------------------------|--------------------------------|
| `/data/results/<run>/metrics.csv`            | `MetricsExporter.writeCsv`     | 1 dòng/snapshot                |
| `/data/results/<run>/summary.json`           | `MetricsExporter.writeJson`    | tổng hợp episode               |
| `/data/results/baseline_results.json`        | `baseline_eval.save_results`   | so sánh scheduler              |
| `:9091/metrics` (chỉ khi `MONITORING_ENABLED=true`) | `MetricsRegistry`        | Prometheus text format         |

Xem từ host: `docker compose run --rm rl-agent ls -la /data/results/`

---

## 9. Tài liệu phân tích & tracking (rải rác trong repo)

Những file này KHÔNG phải hướng dẫn chạy — chúng là ghi chép phân tích, log validation, evidence. Liệt kê ở đây để không bị quên.

| File                                                                                   | Mục đích                                                                                              |
|----------------------------------------------------------------------------------------|-------------------------------------------------------------------------------------------------------|
| [`assets/report/analyze-numbers.md`](../report/analyze-numbers.md)                     | Audit khách quan từng con số phase 1, đánh dấu anomaly (energy magnitude, SLA = 0, throughput, …)      |
| [`assets/report/java-validation-report.md`](../report/java-validation-report.md)       | 10 bug Java (B1–B10) — định vị file:dòng, mức độ, trạng thái fix. Đầu vào cho `ValidationRunner`.     |
| [`assets/report/TODO-evidence.md`](../report/TODO-evidence.md)                         | Quy trình 3 bước thu thập bằng chứng (screenshots + data) cho báo cáo Phase 1.                        |
| [`assets/report/collect-all.log`](../report/collect-all.log)                           | Log output của `collect_all_data.py` lần chạy gần nhất (data version 2026-05-11).                     |
| [`cloudsim-java/test/README.md`](../../cloudsim-java/test/README.md)                   | Mô tả từng test trong `ValidationRunner`. Lệnh chạy đã hợp nhất vào §4.1 ở đây.                       |

Để thêm thông tin tổng quan: xem [`CLAUDE.md`](../../CLAUDE.md) (tiến độ + quyết định kỹ thuật) và [`README.md`](../../README.md) (giới thiệu + kiến trúc) ở root.

---

## 10. Quick reference — các lệnh dùng thường xuyên

```powershell
docker compose build                                                        # build core
docker compose up                                                           # chạy core, foreground
docker compose --profile monitoring up -d                                   # chạy core + monitoring
docker compose run --rm rl-agent python src/smoke_test.py                   # smoke test
docker compose run --rm rl-agent python src/baseline_eval.py --scenario HIGH # baseline
docker compose run --rm --no-deps --entrypoint java cloudsim-java `
    -cp simulation.jar sim.ValidationRunner                                  # Java tests
bash scripts/check-monitoring.sh                                            # monitoring health
docker compose logs -f cloudsim-java                                        # logs Java
docker compose down                                                         # dừng (giữ data)
docker compose down -v                                                      # dừng + xoá volume
```
