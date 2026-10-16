# ELDAS

Bộ lập lịch học tăng cường đặt tác vụ AI lên máy chủ trong trung tâm dữ liệu, cân bằng **điện năng** và **trễ hạn**.

- **Bài toán:** mỗi tác vụ đến phải được đặt ngay lên một máy. Dồn việc vào ít máy thì tiết kiệm điện (máy khác được ngủ) nhưng dễ trễ hạn; trải ra thì ngược lại.
- **Cách giải:** điện năng là *mục tiêu*, trễ hạn là *ràng buộc* `E[C_SLA] ≤ d`. Tác tử `MaskablePPO` học chọn máy; bộ điều khiển **PID** tự dò nhân tử `λ` (giá của trễ hạn) từ ngân sách `d`. Đổi `d` là chọn được điểm trên đường đánh đổi, không phải dò trọng số.
- **Dữ liệu:** bộ sinh **WM-1** dựng từ vết tải Alibaba GPU v2023, đặt riêng được **mức tải** và **mức dồn cục**; 5 kịch bản (`LOW`, `HIGH`, `BURST`, `OVERLOAD`, `REPLAY`) × 2 cấu hình cụm 10 máy.

```
 rl-agent (Python 3.11)                         cloudsim-java (Java 21)
 MaskablePPO + PID-Lagrangian   ◄── Py4J ──►    CloudSim Plus: cụm 10 máy,
 đánh giá: HV, IGD+, EAF           :25333       điện CPU+GPU, máy tự ngủ/thức
```

## Yêu cầu

- Docker + Docker Compose v2 (đã đủ Java, Python và thư viện bên trong image)
- ~16 GB RAM; chạy song song 4 lần huấn luyện cần ~12 luồng CPU
- Windows / Git Bash: thêm `MSYS_NO_PATHCONV=1` trước mọi lệnh có đường dẫn `/data/...`

## Cài đặt

```bash
bash scripts/download-trace.sh      # tải vết tải Alibaba vào data/alibaba-trace/
docker compose build                # dựng hai image
```

## Cách chạy

### 1. Khởi động và kiểm tra nhanh

```bash
docker compose up -d cloudsim-java                              # bật bộ mô phỏng
docker compose run --rm rl-agent python src/smoke_test.py       # chạy thử trọn một vòng
```

### 2. Chạy 5 thuật toán đối chứng

First-fit, best-fit, Kubernetes, quay vòng, ngẫu nhiên:

```bash
docker compose run --rm rl-agent python src/baseline_eval.py --scenario HIGH
```

### 3. Huấn luyện phương pháp đề xuất tại một ngân sách `d`

```bash
docker compose run --rm rl-agent python src/train_cmdp.py \
    --scenario LOW --sla-budget 0.16 \
    --k-p 0.10 --k-i 0.30 --cost-freeze-after 4505 --total-timesteps 270300
```

`--cost-freeze-after` là bắt buộc khi quét ngân sách (nên đặt bằng 5 × số tác vụ một lượt). `d` là số trên thang đã chuẩn hoá, không có đơn vị vật lý.

### 4. Dùng dữ liệu WM-1

```bash
bash scripts/gen-workloads.sh       # sinh data/wm1/{homo,hetero}/{kịch bản}/seed*.csv

# Cho bộ mô phỏng đọc tệp WM-1 thay vì vết tải gốc:
export MSYS_NO_PATHCONV=1
TRACE_PATTERN="/data/wm1/homo/{scenario}/seed{seed}.csv" \
    docker compose up -d --force-recreate cloudsim-java
```

Sau đó `--scenario` (LOW…REPLAY) và `--seed` ở bước 2–3 sẽ chọn đúng tệp.

### 5. Cụm không đồng nhất (3 loại máy CPU/GPU)

```bash
TOPOLOGY_CONFIG=/config/topology-hetero.json \
    docker compose up -d --force-recreate cloudsim-java
```

Bỏ trống biến này thì quay về cụm 10 máy giống nhau.

### 6. Chạy cả chiến dịch đánh giá (một lệnh, chạy qua đêm)

```bash
RESULTS_ROOT=/data/results/wm1-v4 PILOT_EPISODES=300 bash scripts/w6-overnight.sh
#   ARMS=homo ...          chỉ một cấu hình cụm (~11 h)
#   SCENARIOS="LOW" ...    chỉ một kịch bản (~2 h)
```

Lệnh này lần lượt dò lưới ngân sách, huấn luyện mọi phương pháp × 5 hạt giống, rồi xuất bảng (trung bình ± khoảng tin cậy 95%), thể tích chi phối kèm bootstrap và kiểm định EAF. Nếu bị ngắt, chạy lại đúng lệnh đó thì nó làm tiếp phần còn thiếu. Mỗi chiến dịch nên dùng một `RESULTS_ROOT` mới.

Tổng hợp lại bảng từ kết quả có sẵn (không cần bộ mô phỏng):

```bash
docker compose run --rm --no-deps rl-agent python src/eval/run_campaign.py \
    --scenario LOW --seeds 42,43,44,45,46 --results /data/results/wm1-v3/homo
```

### 7. Kiểm thử

```bash
# Java (bộ kiểm ValidationRunner)
docker run --rm -v "$PWD/data/alibaba-trace:/data/trace:ro" --entrypoint java \
    eldas-cloudsim-java -cp simulation.jar sim.ValidationRunner

# Python
docker compose run --rm --no-deps -v "$PWD/rl-agent/tests:/app/tests:ro" \
    --entrypoint bash rl-agent -c 'pip install -q pytest; cd /app && python -m pytest tests/ -q'

# Kiểm chứng toán học của vòng đối ngẫu (không cần bộ mô phỏng)
docker compose run --rm --no-deps rl-agent python src/optim/validate_cmdp_kkt.py
```

### 8. Theo dõi trực tiếp (tuỳ chọn)

```bash
docker compose --profile monitoring up -d     # Prometheus :9090 · Grafana :3000
```

Chỉ dùng để quan sát khi chạy; số liệu báo cáo lấy từ tệp kết quả, không lấy từ Grafana.

## Kết quả và nơi lưu

| Thư mục | Nội dung |
|---|---|
| `data/results/wm1-v3/` | chiến dịch chính: `LOW`, `HIGH`, `REPLAY` (ô `homo/LOW` có 15 hạt giống) |
| `data/results/wm1-v2/` | `BURST`, `OVERLOAD` |
| `…/campaign-{kịch bản}/` | `table.md`, `campaign_summary.json`, `hv_bootstrap.json`, `eaf_compare.json` |

Chỉ so thể tích chi phối **trong cùng một chiến dịch**; hai thư mục khác nhau có hai thang chuẩn hoá khác nhau.

## Cấu trúc thư mục

```
cloudsim-java/src/main/java/sim/   bộ mô phỏng: SimulationManager, 5 thuật toán đối chứng, ValidationRunner
rl-agent/src/
  train_cmdp.py  cmdp.py           huấn luyện phương pháp đề xuất
  optim/                           bộ điều khiển PID + kiểm chứng KKT
  workload/                        bộ sinh dữ liệu WM-1
  eval/                            quét ngân sách, chiến dịch, HV / IGD+ / EAF
config/topology-hetero.json        cụm 3 loại máy
scripts/                           tải dữ liệu, sinh WM-1, chạy chiến dịch
data/                              vết tải gốc, dữ liệu WM-1, kết quả (không đưa lên git)
```
