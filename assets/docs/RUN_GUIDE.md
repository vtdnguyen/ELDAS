# ELDAS — Hướng dẫn chạy & kiểm tra

Tài liệu này hướng dẫn cách build, chạy, smoke test, và đánh giá baseline cho hệ thống ELDAS sau khi hoàn thành Phase 1 (T1–T4).

---

## 1. Yêu cầu hệ thống

| Phần mềm | Phiên bản tối thiểu | Ghi chú |
|---|---|---|
| Docker Desktop | 24.x | có Compose v2 |
| RAM | 4 GB | tối thiểu để chạy CloudSim + Python |
| Disk | 2 GB | image + trace data |
| Trace data | có sẵn tại `data/alibaba-trace/openb_pod_list_default.csv` | nếu thiếu, chạy `bash scripts/download-trace.sh` |

---

## 2. Cấu hình môi trường

File [.env](../.env) đã có sẵn các giá trị mặc định:

```env
PY4J_PORT=25333
ENERGY_WEIGHT=0.8
SLA_WEIGHT=0.2
RANDOM_SEED=42
WANDB_API_KEY=         # để trống → log offline ra console
```

Để bật WandB: lấy API key tại https://wandb.ai/authorize và điền vào `WANDB_API_KEY=...`.

---

## 3. Build & khởi động hệ thống

### Bước 1 — Build cả 2 service

```bash
docker compose build
```

**Output mong đợi**: 2 image `eldas-cloudsim-java` và `eldas-rl-agent` được build thành công, mỗi cái khoảng 1–3 phút lần đầu.

### Bước 2 — Khởi động Java gateway

```bash
docker compose up -d cloudsim-java
```

**Kiểm tra healthcheck**:

```bash
docker compose ps
```

Đợi đến khi cột `STATUS` của `cloudsim-java` chuyển từ `starting` sang `healthy` (~15–30 giây).

```
NAME        IMAGE                 STATUS                  PORTS
cloudsim    eldas-cloudsim-java   Up 30 seconds (healthy) 0.0.0.0:25333->25333/tcp
```

**Kiểm tra log Java**:

```bash
docker compose logs cloudsim-java
```

Bạn sẽ thấy:

```
CloudSim simulation container started.
[GatewayEntryPoint] Py4J GatewayServer listening on 0.0.0.0:25333
```

---

## 4. T4.5 — Smoke Test (kiểm tra full RL loop)

Đây là test **end-to-end** quan trọng nhất, kiểm tra xem Python ↔ Java có thực sự nói chuyện được không, và toàn bộ vòng `reset → step → reward → done` có hoạt động.

### Cách chạy

```bash
docker compose run --rm rl-agent python src/smoke_test.py
```

Hoặc tùy chỉnh:

```bash
docker compose run --rm rl-agent python src/smoke_test.py --scenario LOW --max-steps 100
```

### Output mong đợi

```
╔══════════════════════════════════════════════════════════════════════╗
║       ELDAS Smoke Test — T4.5 — Full RL Loop via Py4J              ║
╚══════════════════════════════════════════════════════════════════════╝
ℹ Scenario: LOW
ℹ Seed: 42

══════════════════════════════════════════════════════════════════════
  Phase 1: Py4J Connection
══════════════════════════════════════════════════════════════════════
ℹ Gateway endpoint: cloudsim-java:25333
✓ Got 10 hosts from Java
✓ Action space size = num hosts (10)
✓ Observation shape = (3H+4,) = (34,)

══════════════════════════════════════════════════════════════════════
  Phase 2: Reset
══════════════════════════════════════════════════════════════════════
✓ Observation has correct shape (34,)
✓ Observation dtype is float32
✓ Observation is within declared space [0, 1]
✓ Initial task index = 0

══════════════════════════════════════════════════════════════════════
  Phase 3: Action Mask
══════════════════════════════════════════════════════════════════════
✓ Mask shape matches num hosts (10)
✓ At least one host is feasible
ℹ Feasible hosts: 10/10

══════════════════════════════════════════════════════════════════════
  Phase 4: Single Step
══════════════════════════════════════════════════════════════════════
✓ Reward is vector of shape (2,)
✓ R_energy ≤ 0 (got -125.4321)
✓ R_sla ≤ 0 (got -0.0000)

══════════════════════════════════════════════════════════════════════
  Phase 5: Full Episode (max 200 steps)
══════════════════════════════════════════════════════════════════════
  step    0: action= 3, R_energy=  -125.4321, R_sla=  0.0000
  step   20: action= 7, R_energy=  -243.1100, R_sla= -2.5000
  ...
✓ Episode terminated naturally after 134 steps
ℹ Total energy: 0.012345 kWh

══════════════════════════════════════════════════════════════════════
  Phase 6: Metrics Export
══════════════════════════════════════════════════════════════════════
✓ Java MetricsExporter wrote files (check /data/results/smoke/...)

══════════════════════════════════════════════════════════════════════
  ✓ ALL SMOKE TESTS PASSED
══════════════════════════════════════════════════════════════════════
The full RL loop is operational.
```

### Output files

Sau smoke test, kiểm tra files được Java exporter ghi:

```bash
docker compose run --rm rl-agent ls -la /data/results/smoke/
```

Bạn sẽ thấy:
- **`metrics.csv`** — một dòng/snapshot, gồm `timestamp, cpu_energy_kwh, gpu_energy_kwh, total_energy_kwh, tasks_scheduled, sla_violations, ...`
- **`summary.json`** — tổng hợp cuối episode

### Khi smoke test fail

| Triệu chứng | Nguyên nhân | Cách fix |
|---|---|---|
| `Cannot connect to Py4J gateway` | Java chưa healthy | `docker compose ps` → đợi healthy hoặc xem log Java |
| `Trace file is empty` / `IOException` | Thiếu trace data | `bash scripts/download-trace.sh` |
| `Observation length mismatch` | Java/Python desync về số host | Kiểm tra `SimulationConfig.DEFAULT_DC` |
| `R_energy > 0` | Bug logic năng lượng Java | Kiểm tra `SimulationManager.computeReward()` |

---

## 5. Chạy Baseline Evaluation

Sau khi smoke test pass, chạy 2 baseline scheduler (K8s + Random) để có điểm so sánh cho RL.

```bash
docker compose run --rm rl-agent python src/baseline_eval.py --scenario HIGH --seed 42
```

### Output mong đợi

```
[baseline_eval] Running baseline: k8s (scenario=HIGH, seed=42)
[baseline_eval] k8s: 1234 steps, energy=0.5432 kWh, r_energy=-1953.21, r_sla=-12.50

[baseline_eval] Running baseline: random (scenario=HIGH, seed=42)
[baseline_eval] random: 1234 steps, energy=0.6789 kWh, r_energy=-2444.10, r_sla=-45.00

[tracker] === Scheduler Comparison ===
  k8s: energy_kwh=0.5432, total_energy_reward=-1953.21, total_sla_reward=-12.5
  random: energy_kwh=0.6789, total_energy_reward=-2444.10, total_sla_reward=-45.0

[baseline_eval] Results saved to /data/results/baseline_results.json
```

### Output files

```bash
docker compose run --rm rl-agent cat /data/results/baseline_results.json
```

```json
{
  "k8s": {
    "scheduler": "k8s",
    "scenario": "HIGH",
    "steps": 1234,
    "total_energy_kwh": 0.5432,
    "total_energy_reward": -1953.21,
    "total_sla_reward": -12.5
  },
  "random": { ... }
}
```

### Chạy với cả 3 kịch bản

```bash
for scen in LOW HIGH BURST; do
    docker compose run --rm rl-agent python src/baseline_eval.py \
        --scenario $scen --output /data/results/baseline-$scen
done
```

---

## 6. Chạy unit tests (offline, không cần Java)

```bash
# Trên máy host (không cần Docker)
pip install numpy gymnasium pytest py4j wandb
python -m pytest rl-agent/tests/ -v
```

**Output mong đợi**:
```
============================= 96 passed in 1.14s ==============================
```

---

## 7. Default `train.py` (entry point của container)

Khi chạy `docker compose up rl-agent` (không override command), container sẽ chạy `python src/train.py`, mặc định bằng smoke test:

```bash
docker compose up
```

Cả 2 container khởi động, smoke test chạy tự động → bạn thấy ngay vòng RL có hoạt động không.

---

## 8. Quick reference — các lệnh hay dùng

```bash
# Build mọi thứ
docker compose build

# Khởi chạy stack (cả 2 container)
docker compose up

# Chỉ start Java service ở background
docker compose up -d cloudsim-java

# Smoke test
docker compose run --rm rl-agent python src/smoke_test.py

# Baseline evaluation
docker compose run --rm rl-agent python src/baseline_eval.py --scenario HIGH

# Mở shell trong rl-agent
docker compose run --rm rl-agent bash

# Xem log Java
docker compose logs -f cloudsim-java

# Dừng + xóa
docker compose down

# Dừng + xóa CẢ volume (data results sẽ mất!)
docker compose down -v

# Chạy unit tests trên host
python -m pytest rl-agent/tests/ -v
```

---

## 9. Các file output Java sinh ra

| File | Sinh từ | Format | Nội dung |
|---|---|---|---|
| `/data/results/<run>/metrics.csv` | `MetricsExporter.writeCsv` | CSV | snapshot per task |
| `/data/results/<run>/summary.json` | `MetricsExporter.writeJson` | JSON | aggregated summary |
| `/data/results/baseline_results.json` | `baseline_eval.save_results` | JSON | so sánh các scheduler |

Truy cập từ host (Docker volume `results`):

```bash
docker compose run --rm rl-agent ls -la /data/results/
```

---

## 10. Bước tiếp theo (Phase 2)

Sau khi smoke test pass + baseline eval chạy xong:

1. **P2.1** — implement PPO + MaskablePPO training loop trong `train.py`
2. **P2.2** — weight sweep: chạy 9 training runs với `ENERGY_WEIGHT` từ 0.1 → 0.9
3. **P2.5** — vẽ Pareto front: energy vs SLA violation, so sánh MORL vs K8s vs Random

Xem thêm chi tiết trong [CLAUDE.md](../CLAUDE.md) → mục **Phase 2**.
