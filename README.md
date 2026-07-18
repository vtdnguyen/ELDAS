# ELDAS — Energy-aware Learning-based Datacenter AI Scheduler

Nền tảng mô phỏng phân bổ tác vụ huấn luyện AI trong datacenter, tối ưu đồng thời **tiết kiệm năng lượng** và **đảm bảo SLA**.

**Giai đoạn 2 (đề tài tốt nghiệp)** nâng bài toán từ gộp trọng số cố định sang **Constrained MDP** giải bằng **PID-Lagrangian**: năng lượng là *mục tiêu*, vi phạm SLA là *ràng buộc*, nhân tử đối ngẫu `λ` tự điều chỉnh — có bảo đảm Pareto-optimal (zero duality gap, Paternain 2019).

> Đọc kèm [`CLAUDE.md`](CLAUDE.md) (kế hoạch + checkbox + **Lưu ý quan trọng**) và [`Tracking_detail.md`](Tracking_detail.md) (nhật ký audit + số liệu có thể tái tạo).

## Bài toán

Trong datacenter AI, hàng nghìn tác vụ huấn luyện cần được phân bổ lên các máy chủ. Hai mục tiêu luôn xung đột:

| Mục tiêu | Chiến lược | Hệ quả |
|----------|-----------|--------|
| **Tiết kiệm năng lượng** | Tắt server nhàn rỗi, dồn tác vụ vào ít máy | Tăng nguy cơ trễ deadline |
| **Đảm bảo SLA** | Phân tán tác vụ, dự phòng tài nguyên | Nhiều server chạy dưới tải → lãng phí điện |

Không tồn tại một giải pháp duy nhất tối ưu cả hai — hệ thống tìm **tập nghiệm Pareto**: mỗi điểm trên front là một mức đánh đổi tối ưu giữa năng lượng và SLA.

## Kiến trúc hệ thống

```
┌────────────────────────────────────────────────────────────────────────┐
│                          Docker Compose                                │
│                                                                        │
│  ┌─────────────────────┐        ┌─────────────────────────┐            │
│  │   cloudsim-java     │ packed │      rl-agent           │            │
│  │                     │  Py4J  │                         │            │
│  │  CloudSim Plus      │◄──────►│  MO-Gymnasium           │            │
│  │  (Java 21)          │ :25333 │  (Python 3.11)          │            │
│  │                     │  +i    │                         │            │
│  │  • Mô phỏng DC      │        │  • MaskablePPO (primal) │            │
│  │  • Energy model +   │        │  • PID-Lagrangian (dual)│            │
│  │    idle/suspend FSM │        │  • Vector reward + cost │            │
│  │  • Alibaba Trace    │        │  • Action masking       │            │
│  │  • Hetero 3 SKU     │        │  • NSGA-II / HV / IGD+  │            │
│  │  • 5 baselines      │        │  • WandB tracking       │            │
│  └──────────┬──────────┘        └────────────┬────────────┘            │
│             │ :9091 (opt)                    │ :8000 (opt)             │
│             ▼                                ▼                         │
│        ┌─────────────────────────────────────────────┐                 │
│        │  Prometheus :9090  ◄────►  Grafana :3000    │  profile:       │
│        │  (opt-in monitoring stack)                  │  monitoring     │
│        └─────────────────────────────────────────────┘                 │
│                                                                        │
│        trace-data (ro) ──┐     ┌── model-store   config/ (ro)          │
│                          └─ results ─┘                                 │
└────────────────────────────────────────────────────────────────────────┘
```

**Lớp 1 — Docker Compose:** 2 container core trên `sim-net`, chia sẻ volume (`trace-data` ro, `results`, `model-store`).

**Lớp 2 — CloudSim Plus (Java):** Engine discrete-event. Mô hình năng lượng `P(U) = P_idle + U × (P_max − P_idle)` + **state machine** `ACTIVE → IDLE → SUSPENDED` (host trống quá `IDLE_THRESHOLD_SEC` sẽ tắt; đánh thức tốn `WAKE_ENERGY_KWH` + `WAKE_LATENCY_SEC`) — **nguồn gradient năng lượng** cho RL. Hỗ trợ **topology dị thể 3 SKU** qua `TOPOLOGY_CONFIG` + **affinity masking** (task cần GPU chỉ fit host có GPU). `SimulationManager` đồng bộ sim-thread ↔ gateway-thread bằng 2 `SynchronousQueue`.

**Lớp 3 — Py4J Gateway:** RPC in-memory JVM↔Python. **Transport gói (SYS.2)**: cả payload 1 step (obs + reward + cost + done + mask) đi trong **một `byte[]`** ⇒ 1 RPC/step thay vì ~127 (xem [Hiệu năng](#hiệu-năng)). Một JVM phục vụ được **N gateway độc lập** (`NUM_GATEWAYS`) cho các run song song.

**Lớp 4 — MO-Gymnasium (Python):** `CloudSimEnv` trả `(obs 6H+4, reward=[R_energy, R_sla], cost=C_SLA, done, info)`. `MaskablePPO` tối ưu `R_energy − λ·C_SLA` (primal); `PIDLagrangian` chỉnh `λ` trên **timescale chậm hơn** (dual). Reward normalization (Welford) là **bắt buộc**, energy và cost normalize **riêng biệt**.

**Lớp 5 — Observability (opt-in):** Prometheus scrape `cloudsim-java:9091`, Grafana :3000. Tách khỏi đường tới hạn; **không** dùng cho figure báo cáo (wall-time ≠ sim-time).

## Mô hình toán học

### Constrained MDP (thay cho gộp trọng số của Phase 1)

```
maximise  E[Σ R_energy]     subject to     E[C_SLA] ≤ d

R_energy = −(E_{t+1} − E_t)                      // gồm cả wake energy
C_SLA    = Σ κ · max(0, T_finish − T_deadline)   // ≥ 0, ràng buộc bởi ngân sách d
L(θ, λ)  = E[R_energy] − λ·(E[C_SLA] − d),  λ ≥ 0
```

> ⚠️ **`κ` ≠ `λ`.** `κ` = hệ số QoS của task (BE 0.5 / Burstable 1.0 / Guaranteed 2.0 / LS 3.0). `λ` = **nhân tử Lagrange**, do PID tự chỉnh. Nhầm hai ký hiệu này là lỗi logic nghiêm trọng nhất của GĐ2.

Luật cập nhật dual (Stooke 2020): `Δ = J − d; I ← max(0, I + K_I·Δ); λ = max(0, K_P·Δ + I + K_D·(Δ − Δ_prev))`.

### Không gian trạng thái & hành động

- **State:** `6H+4` chiều — CPU/MEM/GPU util mỗi host + state one-hot (ACTIVE/IDLE/SUSPENDED) + 4 task feature.
- **Action:** `a ∈ {0..H−1}` — chọn host, có **action masking** (host thiếu tài nguyên hoặc thiếu GPU bị loại).

## Phương pháp tạo Pareto Front

**Quét ngân sách SLA `d`** (không phải sweep trọng số như Phase 1): mỗi `d` → PID tự tìm `λ` → **1 policy hội tụ = 1 điểm** trên front. Front phải **đơn điệu** (`d` chặt hơn ⇒ SLA tốt hơn, energy cao hơn); điểm phi đơn điệu = **chưa hội tụ** ⇒ train lại, không vẽ.

## Dữ liệu

**Alibaba GPU Cluster Trace v2023 (OpenB):** CPU, RAM, số GPU, mức QoS, thời điểm tạo của mỗi tác vụ. Phân lớp thành 3 kịch bản:

| Kịch bản | Đặc điểm | Agent học được gì |
|----------|---------|-------------------|
| **Low Load** | Khung giờ đêm, tác vụ thưa | Tắt host nhàn rỗi để tiết kiệm điện |
| **High Load** | Giờ cao điểm, utilization > 80% | Bin-packing để tránh vi phạm SLA |
| **Burst Load** | Đột biến GPU request | Xử lý phân mảnh tài nguyên |

## So sánh & Đánh giá

Đối chiếu với **5 baseline cổ điển** (xếp từ "spread nhất" sang "pack nhất"):

| Scheduler | Mô tả |
|-----------|-------|
| **Round-Robin** | Pointer xoay vòng modulo `NUM_HOSTS` |
| **Random** | Chọn ngẫu nhiên trong các host hợp lệ (control) |
| **K8s Default** | Filter + LeastRequestedPriority (spread) |
| **First-Fit** | Host đầu tiên đủ tài nguyên theo index |
| **Best-Fit** | Pack chặt nhất |

Cộng thêm **fixed-weight PPO** (Phase 1.8), **CMDP-PID** (sweep `d`), và **NSGA-II** (pymoo) làm reference front tĩnh.

**Metric:** energy kWh (gồm wake) · `C_SLA` · **hypervolume + IGD+** (pymoo, Pareto-compliant, **reference point cố định** chung mọi method) · SLA violation rate · total wakeups.

> ⚠️ **NSGA-II là tham chiếu TĨNH, không phải đối thủ online:** nó biết trước toàn bộ trace, dùng **evaluator khác** (static surrogate ≠ CloudSim DES) và **instance khác** ⇒ **bị loại khỏi HV/IGD+ chung** theo mặc định. Không tuyên bố RL "thua NSGA-II" như so sánh công bằng.

**Kết quả Phase 1.8 (HIGH, seed=42):** front tách 3 cụm — `{BestFit, FirstFit}` pack-tight (~21.8k kWh, SLA tệ); `{Random}` middle; `{K8s, RoundRobin}` spread (~27.3k kWh, SLA tốt nhất). **PPO-min** đạt vị trí interior và **dominate Random** trên cả hai trục. Energy spread BestFit↔RoundRobin: **20–26%** tuỳ scenario.

**Trạng thái GĐ2:** GĐ 2.1 (CMDP + PID-Lagrangian) và GĐ 2.2 (hetero + HV/IGD+) đã xong code + test + validate. **Chưa có số CMDP chính thức** — cần chạy sweep hội tụ ≥5 seeds (xem **C6** trong `Tracking_detail.md`). Không báo cáo số chưa hội tụ.

## Hiệu năng

Nguyên tắc: **đo trước khi tối ưu**, và **chỉ tối ưu transport/song song — không bao giờ đổi hyperparameter để lấy tốc độ**.

| Tối ưu | Trước | Sau | Ghi chú |
|---|---:|---:|---|
| **SYS.2** packed transport, H=10 | 40.4 steps/s | **2236.1** | **55×** — Py4J proxy `double[]` **mỗi phần tử 1 RPC** (~127 RPC/step; JVM chỉ 0.25ms) |
| **SYS.2** packed transport, H=50 | 9.7 steps/s | **1279.6** | **133×** — chi phí cũ scale theo `H`, không phải theo CloudSim |
| **SYS.5** `torch_threads=1` | 572.0 steps/s | **623.8** | mạng `[64,64]` quá nhỏ để đa luồng có lợi; thêm: tất định + chống oversubscription |
| **SYS.1/C9** sweep `--parallel 4` | ~4h (25 run) | **~1h** | song song theo **RUN**, mỗi run 1 gateway |

*Số đo trên container 12-core, LOW, 300 step (`data/results/perf/*.json`); tỉ lệ dao động theo tải máy — chạy lại `profile_step.py` để có số của máy bạn.*

**Bit-identical:** packed transport trả **đẳng thức chính xác** với đường tham chiếu (energy/`C_SLA` trùng từng chữ số) — kiểm bằng `verify_packed_parity.py`. Sau SYS.2, **learner PPO chiếm 67%** ngân sách step, env chỉ còn 33% ⇒ `--n-envs` (SubprocVecEnv) chỉ được **1.35×**, **không phải** đòn bẩy cho sweep.

> ⚠️ **Trước mọi job dài:** `reset()` từng **leak 1 thread + ~1.4 MB mỗi episode** (thread daemon kẹt ⇒ vô hình với unit test, chỉ giết job hàng giờ). Đã sửa + khoá bằng test B19 — nhưng nếu đổi code Java, hãy chạy `soak_resets.py` và xác nhận số thread **phẳng**.

## Tech Stack

| Thành phần | Công nghệ | Phiên bản |
|-----------|----------|-----------|
| Mô phỏng DC | CloudSim Plus | 8.5.7 |
| Java Runtime | Eclipse Temurin | JDK 21 |
| Build tool | Maven | 3.9.9 |
| RL Framework | Stable-Baselines3 + **sb3-contrib** (MaskablePPO) | 2.7.0 |
| MORL env | MO-Gymnasium | 1.3.2 |
| Multi-objective | **pymoo** (NSGA-II + HV/IGD+) | 0.6.1.3 |
| Java-Python bridge | Py4J | 0.10.9.7 |
| Experiment tracking | Weights & Biases | 0.19.1 |
| Container | Docker Compose v2 | — |
| Monitoring (opt-in) | Prometheus + Grafana | 2.55.x / 11.x |

## Cách chạy

```bash
# ── Core ─────────────────────────────────────────────────────────────────
docker compose up --build                                        # stack cơ bản
docker compose run --rm rl-agent python src/smoke_test.py        # verify full loop
docker compose run --rm rl-agent python src/baseline_eval.py --scenario HIGH
docker compose run --rm rl-agent python src/train_min.py --scenario HIGH   # PPO fixed-weight

# Train CMDP tại một ngân sách d (lõi GĐ2)
docker compose run --rm rl-agent python src/train_cmdp.py \
    --scenario LOW --sla-budget 0.04 --k-p 0.05 --k-i 0.05 --total-timesteps 50000

# Kiểm chứng toán học (offline, không cần gateway)
docker compose run --rm --no-deps rl-agent python src/optim/validate_cmdp_kkt.py
docker compose run --rm --no-deps rl-agent python src/optim/validate_lambda_vs_omnisafe.py

# Dị thể CPU-GPU (opt-in; bỏ trống env ⇒ homogeneous)
TOPOLOGY_CONFIG=/config/topology-hetero.json docker compose up --build

# ── Đánh giá Pareto ──────────────────────────────────────────────────────
docker compose run --rm --no-deps rl-agent \
    python src/eval/nsga2_baseline.py --scenario LOW --max-tasks 80   # front tĩnh, offline
docker compose up -d cloudsim-java
docker compose run --rm rl-agent python src/eval/run_campaign.py \
    --scenario LOW --seeds 42,43,44,45,46 --run-baselines   # bảng mean±CI + HV/IGD+ + figure

# ── Hiệu năng & chẩn đoán (SYS) ──────────────────────────────────────────
# So 2 transport (legacy per-element vs packed 1-RPC) + ~RPC/step
docker compose run --rm rl-agent python src/perf/profile_step.py \
    --scenario LOW --steps 300 --json-out /data/results/perf/hosts10.json
# Tách env vs learner + quét torch threads
docker compose run --rm rl-agent python src/perf/profile_train.py \
    --scenario LOW --timesteps 6000 --threads 1,2,6
# Parity: packed PHẢI bằng CHÍNH XÁC đường tham chiếu (điều kiện để tin mọi số)
docker compose run --rm rl-agent python src/perf/verify_packed_parity.py --scenario LOW --steps 200
# Soak: bắt leak thread/memory (so /proc trước–sau; threads phải PHẲNG)
docker exec cloudsim sh -c "grep -E 'Threads|VmRSS' /proc/1/status"
docker compose run --rm rl-agent python src/perf/soak_resets.py --resets 60
docker exec cloudsim sh -c "grep -E 'Threads|VmRSS' /proc/1/status"

# ── OUTPUT CUỐI: chạy TOÀN BỘ phần còn nợ bằng MỘT lệnh ──────────────────
# Đã gồm mọi bước tăng tốc (packed transport + torch_threads=1 + NUM_GATEWAYS +
# sweep --parallel) và MONITORING tắt sẵn. Với mỗi scenario: NSGA-II + ppo-min +
# CMDP sweep + campaign → bảng mean±95%CI + HV/IGD+ + Pareto figure.
bash scripts/run-full-campaign.sh pilot        # ~5 phút — CHẠY TRƯỚC để verify + xem hướng hội tụ
bash scripts/run-full-campaign.sh full LOW     # validate hội tụ trên LOW trước (~1h)
bash scripts/run-full-campaign.sh full         # cả 3 scenario (~8–11h, xem note)
# Scenario truyền THẲNG làm tham số (ổn định hơn env var): "full LOW" · "full LOW,BURST"

# (Hoặc chạy tay 1 scenario) sweep ngân sách d — song song theo RUN:
NUM_GATEWAYS=4 PY4J_PORT_MAX=25336 docker compose up -d --force-recreate cloudsim-java
NUM_GATEWAYS=4 PY4J_PORT_MAX=25336 docker compose run --rm rl-agent \
    python src/eval/sweep_budget.py --scenario LOW \
    --budgets 0.02,0.04,0.06,0.08,0.10 --seeds 42,43,44,45,46 \
    --total-timesteps 362000 --k-p 0.05 --k-i 0.05 --parallel 4

# ── Test ─────────────────────────────────────────────────────────────────
# Java: chạy image TRỰC TIẾP — qua compose thì NUM_HOSTS=10 đè sys-prop của B12
docker run --rm -v eldas_trace-data:/data/trace:ro --entrypoint java \
    eldas-cloudsim-java -cp simulation.jar sim.ValidationRunner        # → PASS=60 FAIL=0
docker compose run --rm --no-deps -v "$PWD/rl-agent/tests:/app/tests:ro" \
    --entrypoint bash rl-agent -c 'pip install -q pytest; cd /app && python -m pytest tests/ -q'
#   → 236 passed

# Monitoring (Prometheus :9090 + Grafana :3000)
docker compose --profile monitoring up --build
```

> **Windows/Git Bash:** prefix `MSYS_NO_PATHCONV=1` cho mọi lệnh có path kiểu `/data/...`, nếu không path bị dịch sai và file ghi nhầm chỗ trong container.

**Trạng thái test:** Java **60/60** (homogeneous) · **57/0-fail/1-skip** (hetero — B12 skip cố ý) · Python **236 passed**.

## Cấu trúc thư mục

```
ELDAS/
├── docker-compose.yml          # 2 container core + monitoring profile
├── CLAUDE.md                   # kế hoạch GĐ2 + checkbox + ⚠️ Lưu ý quan trọng
├── Tracking_detail.md          # nhật ký audit: số liệu, quyết định, cách re-run
├── config/topology-hetero.json # 3 SKU dị thể (GPU-heavy / balanced / CPU-only)
├── cloudsim-java/src/main/java/sim/
│   ├── SimulationManager.java   # engine state + vòng lặp RL + energy/SLA
│   ├── GatewayEntryPoint.java   # façade Py4J (+ stepPacked, ping)
│   ├── StepCodec.java           # SYS.2 — pack 1 step → 1 byte[]
│   ├── TopologyConfig.java      # G2.1 — load 3 SKU từ JSON
│   ├── VmAllocationPolicy*.java # 5 baseline
│   └── ValidationRunner.java    # 60 test B1–B19 (standalone)
├── rl-agent/src/
│   ├── environment.py           # CloudSimEnv (CMDP: reward + cost)
│   ├── cmdp.py                  # CMDPRewardWrapper (R_eff = e_n − λ·c_n)
│   ├── train_cmdp.py            # MaskablePPO + LambdaUpdateCallback
│   ├── vec_env.py               # SYS.1 — SubprocVecEnv + λ broadcast
│   ├── optim/                   # pid_lagrangian + 2 validator (KKT, vs OmniSafe)
│   ├── eval/                    # nsga2, pareto_metrics, sweep_budget, run_campaign
│   └── perf/                    # profile_step/train, verify_packed_parity, soak, tuning
├── data/{alibaba-trace, results}    # trace gốc · baseline-*/, campaign-*/, perf/
├── scripts/                         # download-trace.sh, check-monitoring.sh, ...
├── assets/                          # docs Phase 1 + planning + LaTeX
└── assets_v2/                       # figures GĐ2 + provenance
```

Chi tiết build/chạy/test: [assets/docs/RUN_GUIDE.md](assets/docs/RUN_GUIDE.md) · Figures + cách đọc đúng: [assets_v2/README.md](assets_v2/README.md)
