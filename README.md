# ELDAS — Energy-aware Learning-based Datacenter AI Scheduler

Nền tảng mô phỏng phân bổ tác vụ huấn luyện AI trong datacenter, tối ưu đồng thời **tiết kiệm năng lượng** và **đảm bảo SLA** bằng Học tăng cường Đa mục tiêu (Multi-Objective Reinforcement Learning).

## Bài toán

Trong datacenter AI, hàng nghìn tác vụ huấn luyện cần được phân bổ lên các máy chủ. Hai mục tiêu luôn xung đột:

| Mục tiêu | Chiến lược | Hệ quả |
|----------|-----------|--------|
| **Tiết kiệm năng lượng** | Tắt server nhàn rỗi, dồn tác vụ vào ít máy | Tăng nguy cơ trễ deadline |
| **Đảm bảo SLA** | Phân tán tác vụ, dự phòng tài nguyên | Nhiều server chạy dưới tải → lãng phí điện |

Không tồn tại một giải pháp duy nhất tối ưu cả hai — thay vào đó, hệ thống tìm **tập nghiệm Pareto**: mỗi điểm trên đường Pareto front đại diện cho một mức đánh đổi tối ưu giữa năng lượng và SLA.

## Kiến trúc hệ thống

```
┌────────────────────────────────────────────────────────────────────────┐
│                          Docker Compose                                │
│                                                                        │
│  ┌─────────────────────┐        ┌─────────────────────────┐            │
│  │   cloudsim-java     │        │      rl-agent           │            │
│  │                     │  Py4J  │                         │            │
│  │  CloudSim Plus      │◄──────►│  MO-Gymnasium           │            │
│  │  (Java 21)          │ :25333 │  (Python 3.11)          │            │
│  │                     │        │                         │            │
│  │  • Mô phỏng DC      │        │  • Agent MORL (PPO)     │            │
│  │  • Energy model +   │        │  • Vector reward        │            │
│  │    idle/suspend FSM │        │  • Action masking       │            │
│  │  • Alibaba Trace    │        │  • WandB tracking       │            │
│  │  • 5 baselines (FF/ │        │  • PPO-min checkpoint   │            │
│  │    BF/RR/K8s/Random)│        │                         │            │
│  └──────────┬──────────┘        └────────────┬────────────┘            │
│             │ :9091 (opt)                    │ :8000 (opt, Phase 2)    │
│             ▼                                ▼                         │
│        ┌─────────────────────────────────────────────┐                 │
│        │  Prometheus :9090  ◄────►  Grafana :3000    │  profile:       │
│        │  (opt-in monitoring stack)                  │  monitoring     │
│        └─────────────────────────────────────────────┘                 │
│                                                                        │
│        trace-data (ro) ──┐     ┌── model-store                         │
│                          └─ results ─┘                                 │
└────────────────────────────────────────────────────────────────────────┘
```

Hệ thống gồm **4 lớp core** + **1 lớp Observability tùy chọn**:

**Lớp 1 — Docker Compose:** Điều phối 2 container core trên cùng network `sim-net`, chia sẻ dữ liệu qua 3 volume (`trace-data` read-only, `results` ghi chung, `model-store` cho RL artifacts).

**Lớp 2 — CloudSim Plus (Java):** Engine mô phỏng discrete-event. Tạo datacenter ảo gồm các Host với mô hình năng lượng `P(U) = P_idle + U × (P_max − P_idle)` cùng **state machine** `ACTIVE → IDLE → SUSPENDED` (host nhàn rỗi quá `IDLE_THRESHOLD_SEC` sẽ tắt, wake-up tốn `WAKE_ENERGY_KWH` + `WAKE_LATENCY_SEC`) — tạo gradient năng lượng có ý nghĩa cho RL. Đọc workload thực từ Alibaba GPU Cluster Trace v2023. `SimulationManager` quản lý vòng đời (reset/step/done) thông qua 2 `SynchronousQueue` đồng bộ sim-thread ↔ gateway-thread.

**Lớp 3 — Py4J Gateway:** Cầu nối RPC in-memory giữa JVM và Python. `GatewayEntryPoint` expose `reset(scenario, seed)`, `step(hostIndex)`, `getActionMask()`, `selectBaselineAction(policy)`. Agent Python gọi trực tiếp object Java mà không cần serialize qua REST — cho phép vòng lặp RL tốc độ cao.

**Lớp 4 — MO-Gymnasium (Python):** `CloudSimEnv` wrap Java side thành môi trường Gymnasium chuẩn. Mỗi bước: nhận trạng thái datacenter `(6H+4)` chiều (CPU/MEM/GPU util + state one-hot ACTIVE/IDLE/SUSPENDED + 4 task features) → chọn host (đã masked) → nhận vector phần thưởng `[R_energy, R_SLA]` → cập nhật policy. `ScalarRewardWrapper` cộng linear scalarization cho MaskablePPO; reward normalization (Welford) là **bắt buộc** để tránh value-loss explode.

**Lớp 5 — Observability (opt-in, Phase 1.5):** Prometheus scrape `cloudsim-java:9091` (Java exporter) mỗi 2s, Grafana :3000 hiển thị dashboard live (heatmap host utilization, total energy, SLA violations, queue length). Tách hoàn toàn khỏi đường tới hạn — bật bằng `docker compose --profile monitoring up`. **Không thay thế** `metrics.csv` + WandB cho figures báo cáo khoa học (vì Prometheus theo wall-time, không phải sim-time).

## Mô hình toán học

### Hàm phần thưởng (Reward Vector)

Phần thưởng là **vector 2 chiều**, không phải scalar — điều kiện bắt buộc để vẽ Pareto front:

```
R_energy = −(E_{t+1} − E_t)          // phạt năng lượng tăng thêm
R_SLA    = −λ · max(0, T_finish − T_deadline)   // phạt vi phạm deadline
```

Hệ số `λ` phụ thuộc mức QoS của tác vụ:
- **Burstable:** λ = 1.0 (chấp nhận trễ nhẹ)
- **Latency-Sensitive:** λ = 3.0 (phạt nặng nếu trễ)

### Không gian trạng thái & hành động

- **State:** CPU/RAM/GPU utilization từng host, power hiện tại, thông tin tác vụ đang chờ
- **Action:** `a ∈ {1, 2, ..., N}` — chọn host thứ `j` để đặt tác vụ (có action masking cho host không đủ tài nguyên)

## Phương pháp tạo Pareto Front

Sử dụng **Linear Scalarization + Weight Sweep**: chạy PPO nhiều lần với trọng số `w` thay đổi từ 0.1 đến 0.9. Mỗi lần training cho một điểm tối ưu trên Pareto front.

```
R_scalar = w · R_energy + (1−w) · R_SLA
```

## Dữ liệu

**Alibaba GPU Cluster Trace v2023 (OpenB):** Dữ liệu thực từ cụm GPU của Alibaba, bao gồm thông tin CPU, RAM, số GPU yêu cầu, mức QoS, và thời điểm tạo của mỗi tác vụ.

Dữ liệu được phân lớp thành 3 kịch bản:

| Kịch bản | Đặc điểm | Agent học được gì |
|----------|---------|-------------------|
| **Low Load** | Khung giờ đêm, tác vụ thưa | Tắt host nhàn rỗi để tiết kiệm điện |
| **High Load** | Giờ cao điểm, utilization > 80% | Bin-packing để tránh vi phạm SLA |
| **Burst Load** | Đột biến GPU request | Xử lý phân mảnh tài nguyên |

## So sánh & Đánh giá

Kết quả agent MORL được đối chiếu với **5 baseline cổ điển** (xếp từ "spread nhất" sang "pack nhất"):

| Scheduler | Mô tả |
|-----------|-------|
| **Round-Robin** | Pointer xoay vòng modulo `NUM_HOSTS`, công bằng tuyệt đối |
| **Random** | Chọn host ngẫu nhiên trong số các host hợp lệ (control) |
| **K8s Default** | Mô phỏng Kubernetes: Filter + LeastRequestedPriority (spread theo điểm tài nguyên trống) |
| **First-Fit** | Duyệt theo index, lấy host đầu tiên đủ tài nguyên |
| **Best-Fit** | Score = max(cpu/ram/gpu after-ratio), chọn host CAO nhất vẫn fit → pack chặt nhất |

**Kết quả Phase 1.8 (HIGH, seed=42):** Pareto front đã tách thành 3 cụm rõ rệt — `{BestFit, FirstFit}` pack-tight (~21.8k kWh, SLA tệ); `{Random}` middle; `{K8s, RoundRobin}` spread (~27.3k kWh, SLA tốt nhất). **PPO-min** (100k steps, weights 0.5/0.5) đạt vị trí interior Pareto và **dominate Random** trên cả hai trục. Energy spread BestFit↔RoundRobin: **20–26%** tuỳ scenario.

Tiêu chí đánh giá:
- Tổng năng lượng tiêu thụ (kWh, có cộng `wake_energy_kwh`)
- Số lần wake-up host (`total_wakeups`)
- Tỷ lệ vi phạm SLA + R_sla cộng dồn
- Trực quan hóa Pareto Front (Phase 2 weight sweep)

## Tech Stack

| Thành phần | Công nghệ | Phiên bản |
|-----------|----------|-----------|
| Mô phỏng DC | CloudSim Plus | 8.5.7 |
| Java Runtime | Eclipse Temurin | JDK 21 |
| Build tool | Maven | 3.9.9 |
| RL Framework | Stable-Baselines3 + MO-Gymnasium | SB3 2.7.0, MO-Gym 1.3.2 |
| Java-Python bridge | Py4J | 0.10.9.7 |
| Experiment tracking | Weights & Biases | 0.19.1 |
| Container | Docker Compose v2 | — |
| Monitoring (opt-in) | Prometheus + Grafana | prom 2.55.x / grafana 11.x |

## Cách chạy

```bash
# Core stack — Java sim + Python agent
docker compose up --build

# Smoke test (T4.5) — verify full RL loop qua Py4J
docker compose run --rm rl-agent python src/smoke_test.py

# Baseline evaluation — 5 schedulers (roundrobin, random, k8s, firstfit, bestfit)
docker compose run --rm rl-agent python src/baseline_eval.py --scenario HIGH

# Minimum PPO training (100k steps, w=0.5/0.5, ~40 phút CPU)
docker compose run --rm rl-agent python src/train_min.py --scenario HIGH

# Bật monitoring stack (Prometheus :9090 + Grafana :3000)
docker compose --profile monitoring up --build
# → mở http://localhost:3000  (admin / xem GF_SECURITY_ADMIN_PASSWORD trong .env)
```

Chi tiết xem [assets/docs/RUN_GUIDE.md](assets/docs/RUN_GUIDE.md).

## Cấu trúc thư mục

```
ELDAS/
├── docker-compose.yml          # Điều phối 2 container
├── cloudsim-java/              # Engine mô phỏng datacenter (Java)
│   ├── Dockerfile
│   ├── pom.xml
│   └── src/main/java/sim/
├── rl-agent/                   # Agent học tăng cường (Python)
│   ├── Dockerfile
│   ├── requirements.txt
│   └── src/
├── monitoring/                 # (Phase 1.5) Prometheus + Grafana provisioning
│   ├── prometheus.yml
│   └── grafana/provisioning/
├── data/
│   ├── alibaba-trace/          # Dữ liệu workload thực
│   └── results/                # baseline-{LOW,HIGH,BURST}/{roundrobin,random,k8s,firstfit,bestfit,ppo-min}/{metrics.csv,summary.json}
├── scripts/                    # download-trace.sh, check-monitoring.sh
└── assets/
    ├── docs/RUN_GUIDE.md       # Hướng dẫn build/chạy/test chi tiết
    ├── planning/               # PDFs đề cương Phase 1 & 2
    └── report/                 # LaTeX báo cáo
```
