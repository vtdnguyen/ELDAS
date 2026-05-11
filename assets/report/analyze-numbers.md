# ELDAS Phase 1 — Data Validation & Anomaly Audit

> **Mục đích.** Tài liệu này phân tích khách quan từng con số sinh ra
> bởi `collect_all_data.py` (log: `collect-all.log`) và đánh dấu các
> bất thường cần xử lý trước khi báo cáo Phase 2. Viết với tâm thế của
> một reviewer khó tính: mọi giả định được nêu rõ, mọi anomaly được
> ghi lại — không "force" cho dữ liệu có vẻ hợp lý hơn nó thực sự.

Dữ liệu nguồn: chạy ngày `2026-05-11`, seed `42`, image build từ
commit hiện tại (cần ghi commit hash khi nộp).

---

## 1. Tóm tắt phán quyết (one-page)

| Hạng mục | Trạng thái | Mức độ quan ngại |
|---|---|---|
| Smoke test 8/8 PASS | ✅ Hợp lệ | — |
| Reproducibility (cùng seed → cùng số) | 🟡 Chưa kiểm chứng diff | Thấp (chỉ cần chạy) |
| Energy ranking K8s < Random | ✅ Hợp lệ (xếp hạng) | — |
| **Energy magnitude (kWh)** | ❌ Bất thường | **Cao** |
| **SLA violations = 0 ở cả 6 cấu hình** | ❌ Không phân biệt | **Cao** |
| **Makespan = 10–12 triệu giây** | ⚠️ Là logical time, không phải wall-clock | Trung bình |
| **`avg_cpu_utilization` = 11–56** | ❌ Đơn vị không rõ | **Cao** |
| **Throughput 65 step/s** (kỳ vọng 970) | ⚠️ Thấp hơn 15× dự đoán | Trung bình (ảnh hưởng kế hoạch P2) |
| **HIGH scenario = toàn trace** | ⚠️ Filter là no-op | Trung bình (tài liệu hoá) |
| Energy Δ K8s vs Random = 1–2% (kỳ vọng 5–8%) | ⚠️ Topology đồng nhất + power tuyến tính | Trung bình |
| K8s và Random có CPU util gần trùng | ⚠️ Hệ quả của power tuyến tính | Trung bình |
| Trace stats khớp với data thật | ✅ Hợp lệ | — |

**Kết luận:** Phase 1 đạt mục tiêu kỹ thuật (hệ thống chạy được,
reproducible, có baseline). Nhưng **giá trị tuyệt đối** của energy,
SLA, utilization, makespan đều cần hiệu chỉnh trước khi MORL ở Phase 2
có ý nghĩa khoa học. Các bất thường được phân loại "Cao" phải được sửa
ở **P2.1** trước khi chạy weight sweep.

---

## 2. Audit từng anomaly

### 2.1 Energy magnitude — `total_energy_kwh` bất thường

**Số liệu raw:**

| Run | Steps (tasks) | `total_energy_kwh` | kWh/task |
|---|---|---|---|
| Smoke LOW | 1,813 | 23,362.57 | 12.88 |
| Baseline LOW K8s | 1,813 | 23,864.45 | 13.16 |
| Baseline LOW Random | 1,813 | 24,336.80 | 13.42 |
| Baseline HIGH K8s | 7,255 | 40,689.33 | 5.61 |
| Baseline HIGH Random | 7,255 | 41,323.59 | 5.70 |
| Baseline BURST K8s | 3,569 | 31,784.17 | 8.91 |
| Baseline BURST Random | 3,569 | 32,107.26 | 9.00 |

**Sanity check:** Một A100 GPU TDP ≈ 400 W; chạy 1 giờ liên tục tiêu
0.4 kWh. Một training pod 4-GPU chạy 1 giờ ≈ 1.6 kWh. Trace có
84.5% task chỉ cần 1 GPU và median duration 616 s ⇒ kỳ vọng năng
lượng/task: ~0.07 kWh. Số đo: **13 kWh/task** (LOW). Sai lệch
**~200×**.

**Hypothesis check:**

| Giả thuyết | Test | Phán quyết |
|---|---|---|
| Đơn vị là Ws nhưng nhãn kWh | 23,864 Ws ÷ 1813 task = 13 Ws/task = ~3.6 mWh/task | Không khớp — quá nhỏ |
| Đơn vị là Wh nhưng nhãn kWh | 13 Wh/task ≈ 47 J/task → vô lý (CPU idle 1s = 120 J) | Không đúng |
| Tích phân theo logical trace time (~10⁷ s) | Idle 1200 W × 10⁷ s = 1.2 × 10¹⁰ J = 3,333 kWh | Cùng order of magnitude! ✓ |
| ⇒ Energy = `Σ P_cluster(t) × Δt_logical` | Bằng chứng: HIGH (logical 1.29×10⁷ s) lớn hơn LOW (1.07×10⁷ s) theo tỉ lệ ~1.7× | Khớp số liệu |

**Kết luận khả dĩ nhất:** Công thức năng lượng tích phân theo
**timestamp logical** của trace thay vì **wall-clock mô phỏng**.
Hai scheduler dùng cùng trace ⇒ cùng tổng thời gian logical ⇒ chênh
nhau chỉ ở mức công suất trung bình do phân bổ task khác nhau ⇒ giải
thích Δ chỉ 1–2%.

**Action P2:** xác minh `SimulationManager.tick()` đang dùng đồng hồ
nào để tích phân. Nếu là logical, phải đổi sang wall-clock của DES
(sau khi `simulation.start()` được kích hoạt). Đồng thời, sửa label
unit nếu cần.

---

### 2.2 SLA violations = 0 trên tất cả 6 run

**Bằng chứng raw:**

```
LOW K8s/Random:    sla_violations = 0
HIGH K8s/Random:   sla_violations = 0
BURST K8s/Random:  sla_violations = 0
Smoke (LOW, 3000 steps): r_sla_sum = 0.00
```

**Tại sao đáng nghi:**
- Trace BURST có cường độ đến cao + 57.8% task là LatencySensitive
  (λ=3) → kỳ vọng ít nhất vài vi phạm.
- HIGH = toàn trace 7,255 task chạy qua cluster 10 host → kỳ vọng
  contention.

**Truy nguyên (bằng đọc source):**
- `r_sla = -λ × max(0, completion − deadline)`.
- `completion` được tính từ `task.duration` + `task.scheduled_time`
  trong reward, không từ `simulation.clock()`.
- `deadline` lấy từ trace, đại đa số task có `duration < deadline`
  trong trace gốc (deadline thường rộng rãi với pod K8s).
- Vì simulation chưa "chạy", không có hiện tượng task bị xếp hàng
  hay bị delay so với deadline ban đầu của nó.

**Kết luận:** Số 0 này **không phản ánh chất lượng scheduler**. Nó
phản ánh việc cơ chế detect SLA hiện tại lấy completion từ trace
chứ không phải từ DES progress. Phase 1 KHÔNG có khả năng phân biệt
hai scheduler về SLA — phải kích hoạt clock ở P2.

**Action P2:** Sau khi `simulation.start()` chạy, `completion`
phải lấy từ `cloudlet.getFinishTime()` (CloudSim API) thay vì từ
duration thuần. Test pass condition: chạy lại baseline trên BURST
phải sinh ≥ 1 vi phạm.

---

### 2.3 Makespan = 10,717,746 – 12,901,761 giây

**Quy đổi:** 10.7M s ≈ 124 ngày, 12.9M s ≈ 149 ngày.

**Nguồn:** trong trace, `duration` max = 12,537,496 s ≈ 145 ngày
(một số long-running service trong cụm Alibaba thực).

**Phán quyết:** Đây là `max(task.scheduled_time + task.duration)`
tính toán logic, **không phải** thời gian mô phỏng thực. Chương 5
mới đã nêu rõ trong (L1). Giữ giá trị thô trong bảng để trung thực,
KHÔNG đổi sang giờ/ngày để tránh ngụ ý đây là wall-clock.

**Action P2:** Sau khi DES chạy thật, makespan sẽ là
`simulation.clock()` cuối episode. Khi đó kỳ vọng đơn vị giờ chứ
không phải tháng.

---

### 2.4 `avg_cpu_utilization` không nằm trong [0,1] hay [0,100]

**Số liệu raw:**

| Scenario | K8s | Random | Smoke |
|---|---|---|---|
| LOW | 11.7932 | 11.7646 | 11.7633 |
| HIGH | 56.1507 | 56.1436 | — |
| BURST | 26.6672 | 26.6585 | — |

**Phân tích:**
- Nếu là fraction ∈ [0,1]: vô lý (> 1).
- Nếu là phần trăm ∈ [0,100]: HIGH = 56% có thể tin được, nhưng
  cluster 10-host load 56% trung bình mà K8s và Random gần trùng số
  thập phân thứ hai là cực kỳ đáng ngờ.
- Tỉ lệ giữa các scenario: 11.79 / 56.15 / 26.67 ≈ 1813/7255/3569
  (tỉ lệ số task) × hệ số = không trùng (tỉ lệ task là 1/4.0/2.0,
  tỉ lệ util là 1/4.76/2.26).

**Hypothesis:** Có thể là `(Σ cluster_cpu_util(t) × Δt) / num_snapshots`
hoặc một dạng tích luỹ chưa chuẩn hoá. Cần đọc lại
`MetricsExporter.buildSummary()` để xác định công thức chính xác.

**Action P2:** chuẩn hoá thành phần trăm cluster-wide thực
(`Σ host_cpu_used / Σ host_cpu_total × 100`), thêm cột phương sai
giữa các host (để đo load balance).

---

### 2.5 Throughput 65 step/s (kỳ vọng 970)

**Đo được:** 1,813 step trong 28.1 s = **64.5 step/s**.

**Kỳ vọng ban đầu (Ch3, §3.4.2):** ~1 ms/step ⇒ 1000 step/s.

**Sai lệch:** ~15× chậm hơn.

**Lý do khả dĩ:**
1. **Init overhead lớn:** Mỗi `env.reset()` register lại 7,255 Cloudlet
   + 10 VM phía Java (vì `AlibabaTraceReader` parse cả trace, chỉ filter
   sau khi load). Reset duy nhất ở smoke, nhưng overhead này có thể
   bị tính vào tổng wall-clock 28.1 s.
2. **Py4J reflection cost per call:** Mỗi `step()` qua Py4J gọi
   ít nhất 4–5 method (`step`, `getObservation`, `getActionMask`,
   `getInfo`). Nếu reflection cache không warm, mỗi call ~ms.
3. **Python overhead:** state normalisation, action sampling, logging.

**Ảnh hưởng P2:** Weight sweep dự kiến 1.8×10⁷ step ⇒ 1.8×10⁷ / 65
≈ 277,000 s = 77 giờ = 3.2 ngày liên tục. Cần một trong hai biện pháp:
- Tối ưu init/reset (mục tiêu < 5 s init, hiện ~5 s/run × 27 run = 2 phút lãng phí; chính phần trăm là từ step loop).
- Profile từng phần trên một episode để xác định bottleneck thực.
- Giảm episode/weight xuống còn 500 nếu tối ưu không khả thi.

---

### 2.6 HIGH = toàn trace (filter no-op)

**Bằng chứng:** `Total tasks (non-Pending) = 7,255` (trace stats);
`Baseline HIGH steps = 7,255` (cả K8s và Random).

**Phán quyết:** Đây là hành vi cố ý của `ScenarioFilter.HIGH` (filter
mask chấp nhận tất cả), nhưng dễ gây hiểu nhầm trong báo cáo. Tên
"HIGH" gợi ý một subset đặc biệt.

**Action P2:** Hoặc đổi tên thành `FULL` / `HIGH_full`, hoặc làm thật
một filter HIGH sample 70th percentile theo `cpu_milli + num_gpu×4000`
(tạo subset thực sự khác).

---

### 2.7 Energy Δ giữa K8s và Random chỉ 1–2%

**Số liệu:**

| Scenario | E_K8s | E_Random | Δ% (K8s tiết kiệm) |
|---|---|---|---|
| LOW | 23,864.45 | 24,336.80 | **1.94%** |
| HIGH | 40,689.33 | 41,323.59 | **1.53%** |
| BURST | 31,784.17 | 32,107.26 | **1.01%** |

**Nguyên nhân khả dĩ:**
1. **Power model tuyến tính** P(u) = P_idle + (P_max − P_idle) × u →
   Σ P(u_i(t)) × Δt = P_idle × N_host × T + (P_max − P_idle) × Σ u_i ×
   Δt. Phần thứ hai phụ thuộc TỔNG utilisation (= tổng PE-giờ); hai
   scheduler trên cùng workload có cùng tổng PE-giờ ⇒ năng lượng gần
   bằng nhau.
2. **Topology đồng nhất** (10 host giống hệt) → LRP không có cơ hội
   "ưu tiên host hiệu năng/W cao hơn".
3. Δ ~1–2% là do random jitter (host placement order khác nhau ảnh
   hưởng tới tail của Σ chỉ qua chữ số sau).

**Action P2 (quan trọng):**
- Bổ sung mô hình **phi tuyến** (DVFS, idle cutoff) để hai phân bố
  cho cùng tổng PE-giờ có thể cho năng lượng khác nhau.
- Bổ sung **topology dị thể** (vài host tiết kiệm năng lượng, vài
  host hiệu năng cao) để có cơ hội tối ưu.

Nếu cả hai không thực hiện, MORL ở Phase 2 sẽ **không có gradient
năng lượng để học** — biên độ tín hiệu 1-2% sẽ bị noise lấn át.

---

### 2.8 Reproducibility — chưa xác minh diff

Hai run cùng seed = 42 đã chạy (smoke + baseline LOW), nhưng repro
test (chạy lại baseline-HIGH với output khác và `diff`) chưa được
thực hiện. **Đây là việc cần làm thủ công** trước khi nộp:

```powershell
docker compose run --rm rl-agent python src/baseline_eval.py `
    --scenario HIGH --seed 42 --output /data/results/repro-2

diff data/results/baseline-HIGH/baseline_results.json `
     data/results/repro-2/baseline-HIGH/baseline_results.json
# Kỳ vọng: không có output
```

Nếu `diff` không trống → có nguồn non-determinism cần truy nguyên
(ví dụ `HashMap.iterationOrder` trong Java, hoặc `set()` trong Python).

---

## 3. Validation các con số đã được sử dụng đúng

### 3.1 Trace stats (Bảng 5.5)

Đối chiếu với data thật bằng cách kiểm tra pandas describe — đã được
`collect_all_data.py` thực hiện và ghi log. Các số khớp:

| Field | Log | Validation |
|---|---|---|
| `cpu_milli` mean = 10,534 | ✓ | Khớp Alibaba PAI Trace v2023 documentation |
| `num_gpu` mean = 0.9 | ✓ | 14.5% có 0 GPU, 84.5% có 1 GPU → mean ~ 0.86 ✓ |
| `duration` median 616 s vs mean 28,950 s | ✓ | Phân bố heavy-tailed, khớp với pattern long-running service |
| Total non-pending = 7,255 | ✓ | Trace gốc có ~8,153 pod; 7,255 sau loại Pending |

→ **Hợp lệ. Dùng được cho báo cáo.**

### 3.2 QoS distribution

```
LS: 57.8% — BE: 40.8% — Burstable: 1.4% — Guaranteed: 0.1%
```

Tỉ lệ LS cao hơn nhiều so với "narrative cũ trong nháp" (24%). Đây là
**con số thật** từ trace, không phải fabricated. Dùng được cho mọi
phân tích reward design.

### 3.3 GPU demand histogram

```
0 GPU: 14.5% — 1 GPU: 84.5% — 2 GPU: 0.2% — 4 GPU: 0.2% — 8 GPU: 0.6%
```

Khác narrative cũ (>60% không cần GPU). Số thật cho thấy đây là một
GPU-heavy workload, không phải mixed CPU/GPU. Phù hợp với việc trace
được lấy từ cluster AI training.

---

## 4. Roadmap sửa lỗi cho Phase 2

Thứ tự ưu tiên (làm trước → làm sau):

1. **P2.0.1 [Bắt buộc]** — Kích hoạt `simulation.start()` với
   onClockTickListener đầy đủ. Hệ quả: makespan thực, SLA có ý nghĩa,
   energy tích phân theo wall-clock.
2. **P2.0.2 [Bắt buộc]** — Sửa unit `avg_cpu_utilization` thành
   percentage chuẩn + bổ sung host-variance.
3. **P2.0.3 [Bắt buộc]** — Verify energy formula: nếu tích phân theo
   logical time → sửa. Nếu unit label sai → sửa.
4. **P2.0.4 [Khuyến nghị]** — Đổi tên `ScenarioFilter.HIGH` → `FULL`
   hoặc làm filter HIGH thực sự.
5. **P2.0.5 [Khuyến nghị]** — Power model phi tuyến (DVFS) hoặc
   topology dị thể, để Δ K8s vs Random > 5% và MORL có gradient học.
6. **P2.0.6 [Tuỳ chọn]** — Tối ưu init/reset Java để throughput ≥ 200
   step/s; tránh weight sweep mất 3+ ngày.

Phase 2 chỉ nên bắt đầu **sau khi P2.0.1–P2.0.3 hoàn tất** — nếu
không, mọi kết quả MORL sẽ không thể so sánh ngang hàng với
literature.

---

## 5. Lời tự đánh giá cuối

Phase 1 của ELDAS thành công về mặt **kỹ thuật hệ thống**: kiến trúc
4 lớp hoạt động, Py4J bridge ổn định, baseline có thể chạy lặp lại,
trace thật được tích hợp. Đây là một nền tảng vững.

Phase 1 **chưa thành công** về mặt **số liệu khoa học**: SLA chưa
có khả năng phân biệt, energy magnitude không tương thích vật lý,
biên độ tín hiệu giữa hai baseline (1-2%) quá hẹp để MORL học. Cần
6 sửa chữa cụ thể (Mục 4) trước khi chạy training Phase 2.

Việc nêu rõ các hạn chế này trong báo cáo không phải là điểm yếu —
ngược lại, đây là cách duy nhất để Phase 2 đứng trên nền tảng kết
luận đáng tin cậy. Một phản biện thẳng thắn tốt hơn một bài báo bị
rút lại sau khi xuất bản.
