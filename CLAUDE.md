# ELDAS — Ngữ cảnh dự án

## Dự án này là gì
Nền tảng mô phỏng lập lịch phân bổ tác vụ AI trong datacenter bằng học tăng cường
đa mục tiêu (MORL). Xây dựng với CloudSim Plus (Java) + MO-Gymnasium (Python) +
cầu nối Py4J, đóng gói bằng Docker Compose.

## Kiến trúc — 4 lớp
- L1: Docker Compose — 2 container (cloudsim-java, rl-agent) + 3 volume
- L2: CloudSim Plus (Java 21) — engine mô phỏng, mô hình năng lượng, parser trace Alibaba
- L3: Py4J Gateway (cổng 25333) — cầu nối RPC giữa Java và Python
- L4: MO-Gymnasium (Python 3.11) — agent MORL, vector phần thưởng, theo dõi thí nghiệm

## Tech stack & phiên bản chính xác
- cloudsim-java: eclipse-temurin:21-jdk-jammy, cloudsimplus:8.5.7, py4j:0.10.9.7
- rl-agent: python:3.11.9-slim-bookworm, mo-gymnasium:1.3.2, stable-baselines3:2.7.0
- Docker Compose v2, Maven 3.9.9

## Tiến độ hiện tại

### Phase 1 — Xây dựng môi trường mô phỏng
- [x] T1.2 Dockerfile cloudsim-java (multi-stage, Java 21)
- [x] T1.3 Dockerfile rl-agent (multi-stage, Python 3.11)
- [x] T1.4 Cả hai service build và chạy thành công
- [ ] T0.0 Download + validate Alibaba trace data — TASK TIẾP THEO
- [ ] T2.0 SimulationConfig.java (topology config tập trung)
- [ ] T2.1 DatacenterFactory.java (dùng SimulationConfig, GPU tracking riêng)
- [ ] T2.2 AlibabaTraceReader.java (parse CSV → Cloudlet + Vm)
- [ ] T2.3 ScenarioFilter.java (Low / High / Burst load)
- [ ] T2.4 MetricsExporter.java (energy + SLA metrics → CSV/JSON)
- [ ] T2.5 SimulationManager.java (pause/resume/reset lifecycle — CRITICAL)
- [ ] T3.1 GatewayEntryPoint + GatewayServer (expose SimulationManager qua Py4J)
- [ ] T3.2 VmAllocationPolicyK8sDefault (Filter + LeastRequestedPriority)
- [ ] T3.3 VmAllocationPolicyRandom (baseline #2)
- [ ] T4.1 state_builder.py (+ observation normalization)
- [ ] T4.2 reward.py (vector [R_energy, R_SLA])
- [ ] T4.3 environment.py (MO-Gymnasium env + action masking via MaskablePPO)
- [ ] T4.4 tracker.py + baseline_eval.py (WandB integration)
- [ ] T4.5 Smoke test: full RL loop qua Py4J (reset → step → reward → done)

### Phase 2 — Huấn luyện & Đánh giá
- [ ] P2.1 PPO + MLP training loop (linear scalarization, single weight)
- [ ] P2.2 Weight sweep (w=0.1→0.9) → thu thập Pareto points
- [ ] P2.3 Hyperparameter tuning (learning_rate, discount_factor) via WandB
- [ ] P2.4 Chạy 3 kịch bản (Low/High/Burst) × 3 schedulers (Random/K8s/MORL)
- [ ] P2.5 Vẽ Pareto front + bảng so sánh energy/makespan/SLA violation
- [ ] P2.6 (Stretch goal) Thử GNN head nếu MLP đã converge

## Các quyết định kỹ thuật quan trọng
- Dùng artifact ID `cloudsimplus` (không gạch ngang) — `cloudsim-plus` cũ đã deprecated
- GPU tracking: CloudSim Plus không có GPU concept → track GPU qua HashMap/metadata riêng trên Host, power model GPU tách biệt
- RL stepping: Dùng `SimulationManager` với `addOnClockTickListener()` + `simulation.pause()` + Py4J callback
- Environment reset: `SimulationManager.resetSimulation()` tạo lại CloudSim, reuse host specs + cached trace
- Action masking: Dùng MaskablePPO (SB3-contrib) thay PPO thuần, thêm `action_mask` vào observation
- Algorithm Phase 2: PPO + MLP trước, GNN chỉ thử nếu MLP đã converge (stretch goal)
- MORL strategy: Linear scalarization + weight sweep (w=0.1→0.9) để tạo Pareto front
- Reward vector (KHÔNG cộng gộp scalar):
  - R_energy = -(E_{t+1} - E_t)
  - R_SLA = -λ * max(0, completion - deadline)
- λ từ trường qos: Burstable=1.0, LatencySensitive=3.0
- ENERGY_WEIGHT/SLA_WEIGHT trong .env là default cho dev; Phase 2 sẽ sweep
- healthcheck trên cổng 25333 hiện bị tắt — bật lại sau T3.1
- gymnasium version KHÔNG pin — để pip tự resolve giữa sb3 và mo-gymnasium
- Random seed chung (`RANDOM_SEED`) cho reproducibility (Java + Python)

## Cấu trúc thư mục
```
ELDAS/
├── docker-compose.yml
├── .env                        # PY4J_PORT, ENERGY_WEIGHT, SLA_WEIGHT, RANDOM_SEED
├── CLAUDE.md
├── cloudsim-java/
│   ├── Dockerfile
│   ├── pom.xml
│   └── src/main/java/sim/
│       ├── Main.java                          # stub — Thread.join() only
│       ├── SimulationConfig.java              # TODO T2.0
│       ├── DatacenterFactory.java             # TODO T2.1
│       ├── AlibabaTraceReader.java            # TODO T2.2
│       ├── ScenarioFilter.java                # TODO T2.3
│       ├── MetricsExporter.java               # TODO T2.4
│       ├── SimulationManager.java             # TODO T2.5
│       ├── GatewayEntryPoint.java             # TODO T3.1
│       ├── VmAllocationPolicyK8sDefault.java  # TODO T3.2
│       └── VmAllocationPolicyRandom.java      # TODO T3.3
├── rl-agent/
│   ├── Dockerfile
│   ├── requirements.txt
│   └── src/
│       ├── train.py            # stub
│       ├── state_builder.py    # TODO T4.1
│       ├── reward.py           # TODO T4.2
│       ├── environment.py      # TODO T4.3
│       ├── tracker.py          # TODO T4.4
│       └── baseline_eval.py    # TODO T4.4
├── data/
│   └── alibaba-trace/
│       └── openb_pod_list_default.csv   # TODO T0.0 — download
├── scripts/
│   └── download-trace.sh       # TODO T0.0
└── docs/
    ├── full_system_architecture.html
    └── baseline-results.md
```

## Cách chạy
```
docker compose up --build
```

## Khi bắt đầu task mới
Luôn đọc CLAUDE.md trước để kiểm tra tiến độ hiện tại.
Bắt đầu từ item chưa check đầu tiên trong danh sách trên.
Sau khi hoàn thành task, cập nhật checkbox tương ứng trong file này.
