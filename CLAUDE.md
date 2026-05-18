# ELDAS — Ngữ cảnh dự án

## Dự án này là gì
Nền tảng mô phỏng lập lịch phân bổ tác vụ AI trong datacenter bằng học tăng cường
đa mục tiêu (MORL). Xây dựng với CloudSim Plus (Java) + MO-Gymnasium (Python) +
cầu nối Py4J, đóng gói bằng Docker Compose.

## Kiến trúc — 4 lớp core + lớp Observability (opt-in)
- L1: Docker Compose — 2 container core (`cloudsim-java`, `rl-agent`) + named volume `trace-data` (ro), `results`, `model-store`; chung `sim-net` (bridge).
- L2: CloudSim Plus (Java 21) — engine mô phỏng discrete-event, mô hình năng lượng tuyến tính `P(U) = P_idle + U·(P_max − P_idle)`, GPU tracking riêng (HashMap metadata trên Host), parser trace Alibaba, 2 baseline scheduler (K8s-default + Random) là policy thuần đọc trạng thái live từ `SimulationManager`.
- L3: Py4J Gateway (cổng 25333) — RPC in-memory giữa JVM và Python. `GatewayEntryPoint` expose `reset(scenario, seed)`, `step(hostIndex)`, `getActionMask()`, `selectBaselineAction(policy)`, `getTotalEnergyKwh()`, `exportMetrics(dir)`.
- L4: MO-Gymnasium (Python 3.11) — `CloudSimEnv` (Gymnasium API) trả vector reward `[R_energy, R_SLA]`; `ScalarRewardWrapper` để cộng linear scalarization cho MaskablePPO; `baseline_eval.py` chạy K8s/Random; `tracker.py` log WandB (offline fallback nếu không có key).
- **L5 (opt-in) Monitoring**: Prometheus (cổng 9090) + Grafana (cổng 3000). Bật bằng `MONITORING_ENABLED=true` trong `.env` hoặc `docker compose --profile monitoring up`. Nguồn metric chính là **Java exporter cổng 9091** (long-running, ổn định); Python exporter cổng 8000 chỉ phát metric training (Phase 2). Tách hoàn toàn khỏi đường tới hạn — nếu monitoring chết, vòng RL vẫn chạy.

## Tech stack & phiên bản chính xác
- cloudsim-java: eclipse-temurin:21-jdk-jammy, cloudsimplus:8.5.7, py4j:0.10.9.7, (Phase-1.5) simpleclient_httpserver 0.16.0 (Prometheus Java client)
- rl-agent: python:3.11.9-slim-bookworm, mo-gymnasium:1.3.2, stable-baselines3:2.7.0, sb3-contrib (MaskablePPO), wandb:0.19.1, (Phase-1.5) prometheus_client 0.21.x
- monitoring: prom/prometheus:v2.55.x, grafana/grafana:11.x
- Docker Compose v2, Maven 3.9.9

## Tiến độ hiện tại

### Phase 1 — Xây dựng môi trường mô phỏng
- [x] T1.2 Dockerfile cloudsim-java (multi-stage, Java 21)
- [x] T1.3 Dockerfile rl-agent (multi-stage, Python 3.11)
- [x] T1.4 Cả hai service build và chạy thành công
- [x] T0.0 Download + validate Alibaba trace data (scripts/download-trace.sh)
- [x] T2.0 SimulationConfig.java (topology config tập trung)
- [x] T2.1 DatacenterFactory.java (dùng SimulationConfig, GPU tracking riêng)
- [x] T2.2 AlibabaTraceReader.java (parse CSV → Cloudlet + Vm)
- [x] T2.3 ScenarioFilter.java (Low / High / Burst load)
- [x] T2.4 MetricsExporter.java (energy + SLA metrics → CSV/JSON)
- [x] T2.5 SimulationManager.java (pause/resume/reset lifecycle — CRITICAL)
- [x] T3.1 GatewayEntryPoint + GatewayServer (expose SimulationManager qua Py4J)
- [x] T3.2 VmAllocationPolicyK8sDefault (Filter + LeastRequestedPriority)
- [x] T3.3 VmAllocationPolicyRandom (baseline #2)
- [x] T4.1 state_builder.py (+ observation normalization)
- [x] T4.2 reward.py (vector [R_energy, R_SLA])
- [x] T4.3 environment.py (MO-Gymnasium env + action masking via MaskablePPO)
- [x] T4.4 tracker.py + baseline_eval.py (WandB integration)
- [x] T4.5 Smoke test: full RL loop qua Py4J (reset → step → reward → done)

### Phase 1.5 — Observability stack (opt-in, an toàn với core)
Mục tiêu: trực quan hóa **live** chiến lược đặt task của K8s vs Random (và sau này MORL) bằng heatmap host-utilization. KHÔNG thay thế `metrics.csv` + WandB cho figures báo cáo (xem decision §monitoring).

- [x] T5.1 Thêm Prometheus + Grafana vào `docker-compose.yml` dưới `profiles: ["monitoring"]` — mặc định KHÔNG khởi động, không thay đổi behavior `docker compose up` hiện tại.
- [x] T5.2 `monitoring/prometheus.yml` — 2 job: `cloudsim_java` (target `cloudsim-java:9091`, `scrape_interval: 2s`), `rl_agent` (target `rl-agent:8000`, `honor_labels: true`, đánh dấu `scrape_timeout: 1s` + cho phép target down vì rl-agent là ephemeral).
- [x] T5.3 Java metrics exporter: thêm `simpleclient_httpserver` vào `pom.xml`, tạo `MetricsRegistry.java` (singleton) + start `HTTPServer` cổng 9091 trong `Main.java` khi `MONITORING_ENABLED=true`. Hook vào `SimulationManager`:
  - `Gauge eldas_host_cpu_util{host_id, scenario, scheduler}` — set ở `addOnClockTickListener()`
  - `Gauge eldas_host_mem_util{host_id, ...}`, `eldas_host_gpu_util{host_id, ...}`
  - `Gauge eldas_host_power_watt{host_id, ...}`
  - `Gauge eldas_total_energy_kwh{scenario, scheduler, episode}` — reset về 0 mỗi `resetSimulation()`
  - `Counter eldas_sla_violations_total{scenario, scheduler, qos}` — KHÔNG reset (cộng dồn lifetime, dùng `rate()` trong PromQL)
  - `Counter eldas_tasks_scheduled_total{scenario, scheduler}`
  - `Gauge eldas_pending_tasks{scenario, scheduler}`
  - `Gauge eldas_sim_clock_seconds` — để debug sim-time vs wall-time
- [ ] T5.4 (Phase 2 only) Python metrics exporter trong `rl-agent`: `start_http_server(8000)` lazy khi `MONITORING_ENABLED=true`, expose `eldas_episode_reward_energy`, `eldas_episode_reward_sla`, `eldas_training_loss`, `eldas_action_distribution{host_id}`.
- [x] T5.5 Grafana provisioning: `monitoring/grafana/provisioning/datasources/prometheus.yml` (auto-add Prometheus URL `http://prometheus:9090`) + `monitoring/grafana/provisioning/dashboards/dashboards.yml` (provider) + `monitoring/grafana/provisioning/dashboards/eldas-live.json` (8 panel starter dashboard). Tài khoản admin: `admin/admin` (override qua `GF_ADMIN_USER/PASSWORD` trong `.env`).
- [x] T5.6 Dashboard ELDAS Live:
  - Panel 1 — **Host utilization (State timeline)**: 10 hàng host × 3 màu (CPU/MEM/GPU). Cho phép thấy chiến lược "spread" của K8s vs "pack" của Random/MORL.
  - Panel 2 — **Power per host (Time series, stacked)**: tổng kết power draw datacenter.
  - Panel 3 — **Total energy (Stat)**: kWh hiện tại, threshold màu xanh < 0.5 / cam < 1.0 / đỏ ≥ 1.0.
  - Panel 4 — **SLA violations (Stat + sparkline)**: `increase(eldas_sla_violations_total[1m])` theo scenario.
  - Panel 5 — **Pending queue (Time series)**: thấy backpressure.
  - Panel 6 — **Sim-clock vs wall-clock (Time series)**: 2 line, để verify simulation đang chạy đúng tốc độ.
- [x] T5.7 Smoke test cho monitoring: script `scripts/check-monitoring.sh` curl `http://localhost:9091/metrics` + `http://localhost:9090/api/v1/targets` + `http://localhost:3000/api/health` + Grafana datasource proxy. Pretty TTY output, color-coded PASS/FAIL, exit code 0/1 cho CI. Env override: `ELDAS_JAVA_URL`, `ELDAS_PROM_URL`, `ELDAS_GRAFANA_URL`.

### Phase 1.6 — Đào sâu Lý thuyết & Đánh giá Quy mô (báo cáo + kiểm chứng)
Mục tiêu: bổ sung phần **bản chất CloudSim Plus** và **vị trí quy mô datacenter** vào chương "Kiến trúc" của báo cáo. Output là LaTeX + 1–2 bảng/figure, KHÔNG là code mới.

- [ ] T6.1 Tóm tắt nội dung từ `assets/docs/CloudSim_Plus_investigate.pdf` thành tiểu mục báo cáo: (a) CloudSim Plus mô phỏng Host/VM/Cloudlet bằng **Plain Old Java Objects (POJO)**, KHÔNG dùng Thread/Process — cả simulation chạy single-thread trên một `CloudSim` clock; (b) cơ chế **Discrete Event Simulation** với `FutureQueue` (priority queue theo timestamp) + `DeferredQueue`; (c) lý do chuyển từ multi-threading sang POJO là để mở rộng quy mô hàng vạn host (mỗi Thread JVM tốn ~512KB stack). Nguồn tham chiếu trong báo cáo dùng PDF này.
- [ ] T6.2 Bảng **Limit / Pros / Cons** ngắn gọn: hard limit thực tế (~10⁴–10⁵ host trên một JVM 16GB), pros (reproducibility, no race conditions, easy debug), cons (không tận dụng được multi-core, mọi event đều serial → simulation wall-time tăng tuyến tính theo #event).
- [ ] T6.3 Đánh giá vị trí quy mô datacenter ELDAS: bảng so sánh **3 cột**: ELDAS (10 host, ~hàng nghìn task từ Alibaba trace) | Normal-scale DC (vài trăm host, ví dụ academic cluster HCMUT) | Hyperscale (Google/AWS region: 10⁵–10⁶ host). Kết luận: ELDAS thuộc **micro-cluster / lab-scale**, đủ để chứng minh thuật toán nhưng KHÔNG đại diện cho hyperscale → nêu rõ giới hạn này trong chương "Kết luận".

### Phase 1.7 — Sensitivity Study (thay đổi topology để mô tả đúng datacenter)
Mục tiêu: cho thấy hệ thống **phản ứng đúng** khi thay đổi #host, vCPU, GPU. Mỗi run sinh 1 dòng trong bảng so sánh + 1 điểm trên scatter chart.

- [x] T7.1 Tham số hóa `SimulationConfig.java`: `SimulationConfig.fromEnv()` đọc `NUM_HOSTS`, `VCPU_PER_HOST`, `GPU_PER_HOST`, `RAM_PER_HOST_GB`, `MIPS_PER_PE`, các knob power + state-machine (`IDLE_THRESHOLD_SEC`, `P_SUSPENDED_W`, `WAKE_ENERGY_KWH`, `WAKE_LATENCY_SEC`). Resolution: env → JVM sys-prop (`-Deldas.num_hosts=20`) → default. `SimulationManager` thêm constructor env-driven, re-read mỗi `buildSimulation()`. Verified bởi `ValidationRunner.B12` (sys-prop override → 7 hosts, 16 vCPU, 2 GPU).
- [ ] T7.2 Script `scripts/sweep-topology.sh` (hoặc Python) chạy ma trận:
  - #hosts ∈ {5, 10, 20, 50} × scheduler ∈ {K8s, Random}, fix scenario=HIGH
  - vCPU/host ∈ {8, 16, 32, 64} × scheduler ∈ {K8s, Random}, fix #hosts=10
  - GPU/host ∈ {0, 1, 2, 4} × scheduler ∈ {K8s, Random}, fix #hosts=10
  - Mỗi cell xuất 1 thư mục `data/results/sweep-{axis}-{value}-{scheduler}/`
- [ ] T7.3 Notebook/script tổng hợp `assets/report/figures/sweep-*.pdf`: line chart **energy vs #hosts**, **SLA violation rate vs vCPU**, **GPU utilization vs #GPU**. Mỗi figure có 2 line (K8s vs Random) để chứng minh chính sách scheduler ảnh hưởng độc lập với topology.
- [ ] T7.4 Tiểu mục báo cáo "Khảo sát độ nhạy theo cấu hình hạ tầng" + bảng tổng hợp.

### Phase 1.8 — Mở rộng baseline + Sửa energy model + Minimum MORL PPO
Mục tiêu: KHÔNG dừng ở K8s + Random, mà có **5 schedulers** cổ điển (FF/BF/RR/K8s/Random) **+ một MORL PPO version tối thiểu** đã train, **chạy trên energy model đã được sửa để có gradient có ý nghĩa**. Đây là điều kiện đầu ra của Phase 1.

**Vấn đề gốc rễ phải fix trước khi train**: setup hiện tại (10 host đồng nhất + power tuyến tính + luôn bật) khiến `E_total ≈ N·P_idle·T + (P_max−P_idle)·∫ΣU dt`, mà tích phân chỉ phụ thuộc **tổng demand** không phải cách phân bổ → K8s ≈ Random đến 2 chữ số là kết quả lý thuyết dự đoán được, không phải bug. Hệ quả: PPO mất gradient năng lượng. Fix bằng **idle threshold + host shutdown** (không cần heterogeneity ở Phase 1).

#### Block A — Diagnostic & energy model rewrite (làm TRƯỚC PPO)
- [x] T8.0 **Diagnostic baseline cũ**: `scripts/diagnose-baseline-flatness.py` so sánh `baseline-*-flat/` (pre-fix) vs `baseline-*/` (post-fix). 7 panel/scenario: cumulative energy curves overlay flat-vs-new, end-of-sim bar chart với % gap, host-state stack, per-host CPU util timeline (K8s + Random), host-state heat-strip per host. Output: `assets/report/figures/diagnostic-baseline-flatness{,-LOW,-BURST}.pdf`. Confirm: K8s LRP spread → avg_suspended_hosts ≈ 0.005; Random vô tình pack → avg_suspended_hosts ≈ 0.118 (24× more), wakeups 147 vs 9.
- [x] T8.1 **Host state machine** trong `SimulationManager`:
  - Enum `HostState { SUSPENDED, IDLE, ACTIVE }` + helper `stateOf(host, now)`. Transitions: `ACTIVE` (any U > 0) → `IDLE` (U=0, < `IDLE_THRESHOLD_SEC`) → `SUSPENDED` (U=0, ≥ threshold) → wake-up khi `allocateTask`.
  - Wake-up cost: `WAKE_ENERGY_KWH` cộng vào `cumulativeWakeEnergyWs` (one-shot) + `WAKE_LATENCY_SEC` cộng vào `endTime` của task (ảnh hưởng release + SLA estimate).
  - `advanceEnergy` mới: break interval ở 3 sự kiện — `toTime`, next completion, `nextSuspendTransition()` (smallest `zeroLoadSince + threshold > lastEnergyTimestamp`). Power per host theo state: ACTIVE `P_idle+span·U+GPU_active`, IDLE `P_idle+GPU_idle`, SUSPENDED `P_suspended` (override toàn bộ).
  - Defaults: `IDLE_THRESHOLD_SEC=30`, `P_SUSPENDED_W=10`, `WAKE_ENERGY_KWH=0.0005`, `WAKE_LATENCY_SEC=5`. Verified bởi `ValidationRunner.B11` (≥1 SUSPENDED snapshot + ≥1 wakeup + state codes sum to host count).
- [x] T8.2 **Cập nhật `MetricsExporter` + Prometheus gauges**:
  - `Snapshot` mở rộng: `activeHosts/idleHosts/suspendedHosts/totalWakeups/wakeEnergyKwh + hostCpuUtil[]/hostState[]`. CSV header tự động đính `h{i}_cpu_util,h{i}_state` per host.
  - `Summary` thêm `total_wakeups`, `wake_energy_kwh`, `avg_active_hosts/avg_idle_hosts/avg_suspended_hosts`.
  - Prometheus: `eldas_host_state{host_id,...}` (gauge 0/1/2), `eldas_host_state_count{state,...}` (stacked count), `eldas_total_wakeups_total` (counter).
- [x] T8.3 **Re-run K8s + Random với energy model mới**: archive cũ → `baseline-*-flat/`. Kết quả 3 scenarios:
  | scenario | K8s kWh | Random kWh | **gap** | K8s wakeups | Random wakeups | K8s SLA penalty | Random SLA penalty |
  |---|---|---|---|---|---|---|---|
  | LOW   | 24,630.71 | 22,745.99 |  **8.29%** |   2 |  79 | -25.6M | -58.3M |
  | HIGH  | 27,349.52 | 25,671.89 |  **6.55%** |   9 | 147 | -50.3M | -85.8M |
  | BURST |  3,330.58 |  2,942.81 | **13.18%** |  ~  |  ~  |  -0.2M |  -3.0M |
  Cả 3 scenario gap > 5%. Pre-fix gap đều ≈ 0.06%. **Pareto trade-off rõ rệt**: K8s LRP spread → ít wakeups, tốt SLA, **tốn năng lượng**; Random vô tình pack → nhiều wakeups, tệ SLA, **tiết kiệm năng lượng**. Sẵn sàng cho PPO-min học cân bằng 2 trục.

#### Block B — Thêm 3 baseline cổ điển
- [x] T8.4 `VmAllocationPolicyFirstFit.java`: duyệt host theo index, chọn host **đầu tiên** đủ tài nguyên. Trả về 0 nếu không có host fit (caller redirect/drop).
- [x] T8.5 `VmAllocationPolicyBestFit.java`: scoring `score = max(cpu_after/cpu_total, ram_after/ram_total, gpu_after/gpu_total)`, lấy host score CAO nhất nhưng vẫn fit; tie-break theo lowest host_id (strict `>`). GPU dimension chỉ tính khi `totalGpu > 0`.
- [x] T8.6 `VmAllocationPolicyRoundRobin.java`: state-ful pointer `nextHostIndex`, advance modulo `NUM_HOSTS` sau mỗi allocation thành công; khi pointer trỏ vào host không fit thì probe forward không advance. Reset bằng cách re-construct trong `GatewayEntryPoint.reset()`.
- [x] T8.7 `GatewayEntryPoint.selectBaselineAction(policy)` hỗ trợ 5 tên: `k8s`, `random`, `firstfit`, `bestfit`, `roundrobin`. `baseline_eval.BASELINE_POLICIES = ["roundrobin", "random", "k8s", "firstfit", "bestfit"]` (thứ tự từ "spread nhất" sang "pack nhất" cho dễ đọc log).

**Kết quả empirical 5 baselines** (Phase 1.8 energy model, seed=42):

| scenario | bestfit kWh | firstfit kWh | random kWh | k8s kWh | roundrobin kWh | Δ% bestfit↔RR |
|---|---|---|---|---|---|---|
| LOW   | 18,824.42 | **18,822.12** | 22,745.99 | 24,630.71 | 24,630.81 | **26.7%** |
| HIGH  | **21,816.67** | 21,910.22 | 25,671.89 | 27,349.52 | 27,368.77 | **22.6%** |
| BURST | 2,486.20 | **2,469.61** |  2,942.81 |  3,330.58 |  3,129.23 | **23.4%** |

SLA reward (cao hơn = tốt hơn, đơn vị `R_sla` cộng dồn):

| scenario | bestfit | firstfit | random | k8s | roundrobin |
|---|---|---|---|---|---|
| LOW   | -150.3M | -151.3M |  -58.3M | **-25.6M** |  -34.5M |
| HIGH  | -188.0M | -189.7M |  -85.8M | **-50.3M** |  -61.5M |
| BURST |  -13.1M |  -12.2M |   -3.0M |   **-0.2M** |   -1.5M |

**Quan sát quan trọng**:
- Pareto front đã rõ rệt thành 3 cụm: {bestfit, firstfit} pack-tight → low energy + worst SLA; {random} middle; {k8s, roundrobin} spread → high energy + best SLA.
- Empirical KHÔNG đúng hoàn toàn với prediction `BestFit < K8s ≤ FirstFit` — vì với 10 host đồng nhất, FirstFit cũng tự pack vào host index thấp (sau đó suspend phần còn lại), nên gần như tương đương BestFit. Sự khác biệt 2 chính sách này chỉ rõ khi host heterogeneous (Phase 2E).
- K8s và RoundRobin sát nhau về cả energy và SLA (chênh < 0.1% energy ở LOW/HIGH) — K8s LRP với 10 host đồng nhất ≈ round-robin theo điểm số.

#### Block C — RL stepping audit + Minimum PPO
- [x] T8.8 **Rà soát logic Java cho RL stepping** (ValidationRunner B13/B14/B15):
  - B13: `step()` advance `currentTaskIdx` đúng 1 task/call (10 lần step → idx tăng từ 0 → 10).
  - B14: `getActionMask()` phản ánh feasibility theo TASK HIỆN TẠI. Saturate host 0 → mask[0] flip false ở task tiếp theo cần resource host 0 không còn.
  - B15: SUSPENDED host vẫn mask=true (có thể wake). Empirical: sau 50 step pump host 0, ít nhất 1 host SUSPENDED, mask của nó vẫn true.
  - Tất cả 28/28 ValidationRunner test PASS.
- [x] T8.9 **Minimum MORL PPO training**: `rl-agent/src/train_min.py` — MaskablePPO 100k steps, weights `(0.5, 0.5)`, scenario=HIGH, `device=cpu`, `n_steps=512`, `batch_size=64`. **Observation mở rộng thành `6H + 4`** (CPU/MEM/GPU util + state one-hot 3-way + 4 task features), Java `SimulationManager.buildObservation` ghi state-hot dạng host-major. **Reward normalization (Welford) BẮT BUỘC** — không normalize thì value_loss explode (1e10) và approx_kl ≈ 1e-9 (verified empirically với 5k smoke). Với normalize: value_loss 2–5, approx_kl 0.01, explained_variance 0–0.57. Wall-time ~40 phút CPU. Output: `/data/models/ppo-min.zip` + `data/results/baseline-HIGH/ppo-min/{metrics.csv,summary.json}`.
- [x] T8.10 `scripts/phase1-comparison.py` + `scripts/verify-ppo-min.py` (skeptical audit). **Bảng 6 hàng × 5 cột final (HIGH, seed=42)**:

  | Rank | Scheduler | Energy kWh | R_sla | Viol rate | Avg CPU util | #Wakeups |
  |---|---|---|---|---|---|---|
  | 1 | bestfit | 21,816.67 | -188,040,725 | 55.99% | 0.677 | 246 |
  | 2 | firstfit | 21,910.22 | -189,707,682 | 55.59% | 0.677 | 255 |
  | 3 | **ppo-min** | **25,423.82** | **-65,989,191** | **57.42%** | **0.671** | **30** |
  | 4 | random | 25,671.89 | -85,836,183 | 55.85% | 0.672 | 147 |
  | 5 | k8s | 27,349.52 | -50,297,165 | 57.19% | 0.670 | 9 |
  | 6 | roundrobin | 27,368.77 | -61,475,286 | 57.63% | 0.672 | 9 |

  **PPO-min Pareto-dominates Random** (energy lower + R_sla higher đồng thời). Không dominate được {BestFit, FirstFit} (energy thua) hay {K8s, RoundRobin} (SLA thua) — là điểm interior balanced, đúng kỳ vọng cho weight (0.5, 0.5). Verify pass tất cả 4 check ở `verify-ppo-min.py`: (1) Pareto dom Random, (2) all 10 hosts used, (3) 30 wakeups ở giữa K8s-9 và Random-147, (4) chiến thắng so 1/5 baseline đúng kỳ vọng cho minimum PPO.
  Energy spread tổng (BestFit↔RoundRobin) HIGH = **20.29%**, LOW = 23.58%, BURST = 25.85%.

**Phase 1 đóng cửa.** Pipeline kết thúc: state machine + 5 baseline cổ điển + PPO-min đã train, đủ chứng minh `(a)` energy gradient có thực, `(b)` MaskablePPO trên CloudSim Plus qua Py4J hoạt động đúng, `(c)` có thể đặt agent vào Pareto-front với 100k step minimum. Phase 2 sẽ tune + sweep để chuyển từ điểm interior sang Pareto-optimal front.

### Phase 2 — Hoàn thiện MORL, đối chuẩn SOTA, tối ưu hệ thống & UI
Output của Phase 2 chia làm 4 nhóm: (A) PPO tối ưu, (B) đối chuẩn SOTA, (C) tối ưu hiệu năng simulator, (D) UI quản lý.

#### 2A — Hoàn thiện PPO
- [ ] P2A.1 Hyperparameter tuning (`learning_rate`, `gamma`, `n_steps`, `clip_range`) qua WandB sweep
- [ ] P2A.2 Weight sweep linear scalarization (w=0.1→0.9, bước 0.1) → 9 model → 9 Pareto points
- [ ] P2A.3 Chạy 3 scenario × {6 baseline + 9 PPO} = 45 cấu hình → matrix kết quả
- [ ] P2A.4 Pareto front plot + bảng tổng hợp Energy/Makespan/SLA (figures từ `metrics.csv` + WandB, KHÔNG từ Grafana)
- [ ] P2A.5 (Stretch) GNN head nếu MLP converge tốt

#### 2B — State-of-the-Art MORL Schedulers (đối chuẩn nghiên cứu)
- [ ] P2B.1 Khảo sát literature 2022–2025, chọn **2 thuật toán SOTA** (ứng viên: PCN — Pareto Conditioned Network, Envelope Q-learning, CAPQL — Conditioned Average Policy via Q-learning, hoặc MO-DQN). Quyết định dựa trên: có public implementation, tương thích MO-Gymnasium, scope thesis.
- [ ] P2B.2 Tích hợp 2 thuật toán đã chọn vào `rl-agent/src/sota/`
- [ ] P2B.3 Train + đánh giá song song với PPO ở P2A.3 → cập nhật bảng so sánh
- [ ] P2B.4 Phân tích định tính: PPO scalarization vs MORL không-scalar trong thực tế (hypervolume metric, sparsity của Pareto front)

#### 2C — Tối ưu Hệ thống
- [ ] P2C.1 Profile JVM (JFR/async-profiler) trong `cloudsim-java` → xác định bottleneck event loop khi #host=50 hoặc trace dài
- [ ] P2C.2 Tối ưu Py4J: batch RPC (ví dụ gửi nhiều `step` trong một call) nếu RTT là bottleneck
- [ ] P2C.3 Cache trace parsing giữa các episode (đã có nhưng cần verify hit rate)
- [ ] P2C.4 Đo lại wall-time training trước/sau, đưa vào báo cáo

#### 2E — Heterogeneity + Nonlinear power (mở rộng energy realism)
Mục tiêu: sau khi idle-threshold (P1.8) đã tạo gradient năng lượng dạng allocation, thêm **task-host affinity gradient** + đường cong power thực tế hơn.

- [ ] P2E.1 Heterogeneous topology: chia 10 host thành 3 SKU — 3× GPU-heavy (A100-like: P_idle=200W, P_max=500W, 4 GPU), 3× balanced (V100-like: P_idle=120W, P_max=350W, 2 GPU), 4× CPU-only (P_idle=60W, P_max=200W, 0 GPU). Cập nhật `SimulationConfig` để load topology profile từ JSON (`config/topology-hetero.json`) — homogeneous vẫn là default cho reproducibility.
- [ ] P2E.2 Trace mapping: task có `gpu_milli > 0` chỉ fit vào SKU có GPU; cập nhật `getActionMask()` reflect affinity.
- [ ] P2E.3 Nonlinear power model: `P(U) = P_idle + α·U + β·U²` (concave, β < 0). Hằng số từ benchmark SPEC Power 2008 — note rõ trong báo cáo. DVFS optional (stretch).
- [ ] P2E.4 Re-run weight sweep + SOTA comparison (P2A.2 + P2B.3) trên topology heterogeneous → cập nhật Pareto front; expected: Pareto front giãn rộng vì có thêm trục affinity.
- [ ] P2E.5 So sánh báo cáo: figure "Homogeneous vs Heterogeneous Pareto front" để chứng minh giới hạn của Phase 1 + giá trị của Phase 2.

#### 2D — UI Quản lý (thay thế terminal)
Mục tiêu: chạy test, tùy chỉnh #host/vCPU/GPU, xem kết quả qua web UI; số liệu live vẫn đi qua Prometheus/Grafana đã có sẵn.

- [ ] P2D.1 Quyết định stack: React + FastAPI (chạy trong `rl-agent` hoặc container mới `eldas-ui`). Ưu tiên FastAPI vì đã có Python ecosystem.
- [ ] P2D.2 Backend API: `POST /experiments` (config: scenario, scheduler, #hosts, vCPU, GPU, weights) → spawn run; `GET /experiments` (list); `GET /experiments/{id}/status`; `GET /experiments/{id}/metrics` (proxy đến Prometheus query). Auth: tạm thời basic auth, không cần SSO.
- [ ] P2D.3 Frontend: 3 trang — (1) **Topology Designer** (form chọn #hosts/vCPU/GPU/scenario), (2) **Experiment Runner** (chọn scheduler, weights, start/stop, log stream), (3) **Results Dashboard** (embed Grafana panels qua iframe + bảng so sánh các experiment cũ).
- [ ] P2D.4 Tích hợp Grafana embed: cấu hình `GF_AUTH_ANONYMOUS_ENABLED=true` trong dev, dùng `allow_embedding=true` để iframe hoạt động.
- [ ] P2D.5 Thêm service `eldas-ui` vào `docker-compose.yml` (không thuộc profile monitoring, là core mới ở Phase 2).

## Các quyết định kỹ thuật quan trọng
- Dùng artifact ID `cloudsimplus` (không gạch ngang) — `cloudsim-plus` cũ đã deprecated
- GPU tracking: CloudSim Plus không có GPU concept → track GPU qua HashMap/metadata riêng trên Host, power model GPU tách biệt
- RL stepping: Dùng `SimulationManager` với `addOnClockTickListener()` + `simulation.pause()` + Py4J callback (2 `SynchronousQueue` đồng bộ sim-thread ↔ Py4J-thread)
- Environment reset: `SimulationManager.resetSimulation()` tạo lại CloudSim, reuse host specs + cached trace
- Action masking: Dùng MaskablePPO (SB3-contrib) thay PPO thuần, thêm `action_mask` vào observation
- Algorithm Phase 1.8: PPO + MLP **minimum** (1 weight 0.5/0.5) để đủ điều kiện so sánh với 5 baseline (FF/BF/RR/K8s/Random) trước khi đóng Phase 1
- Algorithm Phase 2: PPO tối ưu (hyperparameter tune + weight sweep) + 2 SOTA MORL (PCN / Envelope / CAPQL — chốt sau khi khảo sát literature ở P2B.1); GNN là stretch goal
- MORL strategy: Linear scalarization + weight sweep (w=0.1→0.9) để tạo Pareto front; so sánh với SOTA non-scalar bằng hypervolume metric
- Baseline schedulers: 5 thuật toán cổ điển — FirstFit (nguyên thủy), BestFit (pack, ép tối ưu năng lượng), RoundRobin (fairness), K8s-default (filter+score), Random (control). Tất cả đều là policy thuần, đọc trạng thái live từ `SimulationManager`, không train.
- Topology sensitivity: `SimulationConfig` phải tham số hóa qua env var để Phase 1.7 sweep được #host/vCPU/GPU mà không rebuild image
- Reward vector (KHÔNG cộng gộp scalar):
  - R_energy = -(E_{t+1} - E_t)
  - R_SLA = -λ * max(0, completion - deadline)
- **Energy model có idle threshold + host shutdown (Phase 1.8)** — KHÔNG để Phase 2. Lý do:
  - Setup hiện tại (10 host đồng nhất + power tuyến tính + always-on) khiến `E_total ≈ N·P_idle·T + (P_max−P_idle)·∫ΣU dt`. Tích phân chỉ phụ thuộc tổng demand, không phải cách phân bổ → K8s ≈ Random đến 2 chữ số, PPO mất gradient năng lượng.
  - Idle threshold (host tắt khi U=0 quá `T_IDLE`) tạo gradient ngay với host đồng nhất + power tuyến tính: BestFit pack chặt → nhiều host SUSPENDED → tiết kiệm `N_off · P_idle · T`. Đây là can thiệp rẻ nhất (~50 dòng code) và là điều kiện tiên quyết để PPO-min có ý nghĩa.
  - Heterogeneity (3 SKU GPU/balanced/CPU) là **affinity gradient** (trục thứ 2), đẩy sang Phase 2E sau khi đã verify gradient cơ bản hoạt động ở Phase 1.8.
  - DVFS / nonlinear power là refinement, KHÔNG fix gradient gốc — đặt ở Phase 2E.
- λ từ trường qos: BE=0.5, Burstable=1.0, Guaranteed=2.0, LS=3.0
- ENERGY_WEIGHT/SLA_WEIGHT trong .env là default cho dev; Phase 2 sẽ sweep
- gymnasium version KHÔNG pin — để pip tự resolve giữa sb3 và mo-gymnasium
- Random seed chung (`RANDOM_SEED`) cho reproducibility (Java + Python)

### Quyết định cho lớp Monitoring (Phase 1.5)
- **Nguồn metric chính = Java (Lớp 2)**, KHÔNG phải Python. Lý do:
  1. `cloudsim-java` là long-running (đã healthcheck), Prometheus scrape ổn định. `rl-agent` chạy ephemeral (`docker compose run --rm`) → nếu phát từ Python, Prometheus báo target-down liên tục.
  2. Java có **giá trị thực** (kWh, queue length); Python chỉ có observation đã normalize 0–1.
- **Grafana chỉ dùng cho demo live + debug**, KHÔNG thay thế `metrics.csv` + WandB cho figures báo cáo. Lý do: Prometheus scrape theo wall-time, không phải sim-time → trục thời gian không khớp với báo cáo khoa học.
- **Counter chỉ cho cumulative-lifetime** (SLA violations, tasks scheduled). Energy/utilization dùng **Gauge** có label `episode` để hỗ trợ multi-episode sweep.
- **Opt-in qua Compose profile** (`profiles: ["monitoring"]`): mặc định KHÔNG khởi động → không phá test suite hiện tại, không ảnh hưởng CI.
- **Heatmap panel**: dùng **"State timeline"** hoặc **"Status history"** (không phải panel "Heatmap" — panel đó cho histogram buckets).
- **Không thêm monitoring vào `depends_on`** của `cloudsim-java`/`rl-agent`. Monitoring chết KHÔNG được làm chết simulation.

## Cấu trúc thư mục
```
ELDAS/
├── docker-compose.yml             # core services + monitoring profile
├── .env                           # PY4J_PORT, ENERGY_WEIGHT, SLA_WEIGHT, RANDOM_SEED, MONITORING_ENABLED
├── CLAUDE.md
├── README.md
├── cloudsim-java/
│   ├── Dockerfile
│   ├── pom.xml
│   └── src/main/java/sim/
│       ├── Main.java
│       ├── SimulationConfig.java
│       ├── DatacenterFactory.java
│       ├── AlibabaTraceReader.java
│       ├── ScenarioFilter.java
│       ├── MetricsExporter.java                # CSV/JSON exporter (offline analysis)
│       ├── MetricsRegistry.java                # TODO T5.3 — Prometheus client (live)
│       ├── SimulationManager.java
│       ├── GatewayEntryPoint.java
│       ├── ValidationRunner.java
│       ├── VmAllocationPolicyK8sDefault.java
│       ├── VmAllocationPolicyRandom.java
│       ├── VmAllocationPolicyFirstFit.java     # TODO T8.1
│       ├── VmAllocationPolicyBestFit.java      # TODO T8.2
│       └── VmAllocationPolicyRoundRobin.java   # TODO T8.3
├── rl-agent/
│   ├── Dockerfile
│   ├── requirements.txt
│   └── src/
│       ├── train.py                            # entry point (defaults → smoke test)
│       ├── train_min.py                        # TODO T8.7 — minimum PPO 100k–300k steps
│       ├── smoke_test.py                       # T4.5
│       ├── state_builder.py
│       ├── reward.py
│       ├── environment.py
│       ├── tracker.py
│       ├── baseline_eval.py                    # cập nhật 5 baseline ở T8.4
│       └── sota/                               # TODO P2B — 2 SOTA MORL algorithms
├── monitoring/                                 # TODO Phase 1.5
│   ├── prometheus.yml
│   └── grafana/
│       └── provisioning/
│           ├── datasources/prometheus.yml
│           └── dashboards/eldas.json
├── data/
│   ├── alibaba-trace/openb_pod_list_default.csv
│   └── results/                                # baseline-{LOW,HIGH,BURST}/{k8s,random}/{metrics.csv,summary.json}
├── scripts/
│   ├── download-trace.sh
│   ├── check-monitoring.sh
│   └── sweep-topology.sh                       # TODO T7.2 — sensitivity study runner
├── eldas-ui/                                   # TODO P2D — FastAPI + React UI
│   ├── backend/
│   └── frontend/
└── assets/
    ├── docs/RUN_GUIDE.md
    ├── planning/
    └── report/                                 # LaTeX report sources
```

## Cách chạy
```bash
# Core stack (không monitoring) — đường hot path, mặc định
docker compose up --build

# Có monitoring (Prometheus :9090 + Grafana :3000)
docker compose --profile monitoring up --build
```

## Khi bắt đầu task mới
Luôn đọc CLAUDE.md trước để kiểm tra tiến độ hiện tại.
Bắt đầu từ item chưa check đầu tiên trong danh sách trên.
Sau khi hoàn thành task, cập nhật checkbox tương ứng trong file này.
