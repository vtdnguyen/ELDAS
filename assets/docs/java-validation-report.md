# ELDAS — Java-side Validation Report

> **Mục đích.** Đối chiếu từng nghi vấn trong `analyze-numbers.md` với
> source Java thực tế, định vị nguyên nhân gốc theo file:dòng, và ghi
> nhận thêm các bug *chưa* được nêu trong báo cáo phân tích số liệu.
>
> Phương pháp: đọc kỹ source + thử thực nghiệm bằng `ValidationRunner`
> (xem [`cloudsim-java/test/`](../../cloudsim-java/test/)).
>
> Kết luận tóm tắt: **10 lỗi** ở phía Java, trong đó **4 lỗi tối nghiêm
> trọng** vô hiệu hoá hoàn toàn ý nghĩa khoa học của các baseline ở
> Phase 1. Các con số trong báo cáo Phase 1 không nên dùng để đối chiếu
> ngang hàng với literature.

---

## 1. Bảng tổng hợp

| # | Bug | Mức | Trạng thái | File:dòng | Hiện tượng quan sát |
|---|-----|-----|-----------|-----------|---------------------|
| B1 | **K8s scheduler luôn chọn host index = 0** | Critical | ✅ **FIXED** | [VmAllocationPolicyK8sDefault.java](../../cloudsim-java/src/main/java/sim/VmAllocationPolicyK8sDefault.java) | đọc state qua `SimulationManager.freePes/freeRam` |
| B2 | **Tài nguyên CPU/RAM/GPU không bao giờ được giải phóng** | Critical | ✅ **FIXED** | [SimulationManager.java](../../cloudsim-java/src/main/java/sim/SimulationManager.java) (T2.5) | util giờ bounded ≤ capacity; releases qua `globalCompletions` PQ |
| B3 | **Energy tích phân theo logical trace-time** | Critical | ✅ **FIXED** | [SimulationManager.java#advanceEnergy](../../cloudsim-java/src/main/java/sim/SimulationManager.java) | piecewise integration với release events; bounded bởi `[idle×T, max×T]` |
| B4 | **SLA violation về mặt toán học không thể trigger** | Critical | ✅ **FIXED** | [SimulationManager.java#computeReward](../../cloudsim-java/src/main/java/sim/SimulationManager.java), [AlibabaTraceReader.java](../../cloudsim-java/src/main/java/sim/AlibabaTraceReader.java), [SimulationConfig.java](../../cloudsim-java/src/main/java/sim/SimulationConfig.java) | deadline = QoS slack × duration; completion ước lượng có congestion factor |
| B5 | `avg_cpu_utilization` công thức sai đơn vị | High | 🟡 partial | [SimulationManager.java#recordSnapshot](../../cloudsim-java/src/main/java/sim/SimulationManager.java) | clamp ≤ 1.0 đã fix; time-weighting average vẫn sai |
| B6 | clamp không nhất quán giữa energy & snapshot | Medium | ✅ **FIXED** | [SimulationManager.java](../../cloudsim-java/src/main/java/sim/SimulationManager.java) | cả hai dùng cùng `Math.min(1.0, …)` |
| B7 | `ScenarioFilter.HIGH` là no-op | Medium | ❌ chưa | [ScenarioFilter.java:80-82](../../cloudsim-java/src/main/java/sim/ScenarioFilter.java#L80-L82) | HIGH = toàn trace (7,255 task) |
| B8 | Custom `VmAllocationPolicy*` không bao giờ được sử dụng qua CloudSim | Medium | ❌ chưa | [DatacenterFactory.java:67-71](../../cloudsim-java/src/main/java/sim/DatacenterFactory.java#L67-L71) | DC luôn dùng `VmAllocationPolicySimple` |
| B9 | Default-constructor truyền `pesCount` (64) làm seed | Low | ❌ chưa | [SimulationManager.java:120-125](../../cloudsim-java/src/main/java/sim/SimulationManager.java#L120-L125) | seed không phải `RANDOM_SEED` env |
| B10 | `getActionMask` trả mask all-false khi `episodeDone` | Low | ❌ chưa | [SimulationManager.java#getActionMask](../../cloudsim-java/src/main/java/sim/SimulationManager.java) | tiềm tàng panic nếu Python sample từ mask rỗng |

**Cập nhật 2026-05-11 (lần 2):** Bốn bug Critical đã fix xong (B1, B2, B3, B4). Còn lại 4 bug Medium/Low (B7, B8, B9, B10) — **không gate Phase 2**, có thể defer thoải mái.

### Known limitations — chấp nhận để Phase 2 xử lý

Sau review lần 2, ba điểm sau được xác định **không phải code bug** mà là design choice. Cắt khỏi danh sách fix bắt buộc:

| Điểm | Bản chất | Vì sao defer |
|---|---|---|
| Reward thiếu gradient | RL design issue do topology đồng nhất + power tuyến tính (analyze-numbers.md §2.7) | Đã có trong roadmap P2.0.5 (power phi tuyến / topology dị thể) |
| Energy magnitude lớn (trace span ~120 ngày) | Experiment design issue | Sẽ xử lý ở P2 bằng cách subset trace per-episode |
| B5 time-weighting trong `buildSummary` | Metric tinh chỉnh | Sau partial-fix, `clusterCpuUtil ∈ [0,1]` đã ổn cho training; cải thiện không gate Phase 2 |

### File layout change (2026-05-11)

- `cloudsim-java/test/sim/ValidationRunner.java` đã được **dời vào** [`cloudsim-java/src/main/java/sim/ValidationRunner.java`](../../cloudsim-java/src/main/java/sim/ValidationRunner.java).
- Lý do: Dockerfile có `ENTRYPOINT ["java", …, "-jar", "simulation.jar"]`; image runtime không có Maven hay source. Đặt validator trong `src/main` để nó được shade vào fat jar, chạy bằng `--entrypoint java -cp simulation.jar sim.ValidationRunner`. Đơn giản hơn maintain test compilation pipeline song song.
- `cloudsim-java/test/` chỉ còn [`README.md`](../../cloudsim-java/test/README.md) làm pointer.

---

## 2. Phân tích chi tiết

### B1 — K8s "đui mắt" (✅ FIXED 2026-05-11 lần 2)

**Trạng thái trước fix:**
[`VmAllocationPolicyK8sDefault.selectHostForTask`](../../cloudsim-java/src/main/java/sim/VmAllocationPolicyK8sDefault.java)
gọi `host.getFreePesNumber()` / `getRam().getAvailableResource()` từ
CloudSim, nhưng vì không Vm nào thật sự được submit, hai accessor đó
luôn báo full ⇒ leastRequestedScore = 1.0 cho mọi host ⇒ tie-break = index 0.

**Fix:**
1. Thêm public accessors vào
   [`SimulationManager`](../../cloudsim-java/src/main/java/sim/SimulationManager.java):
   `freePes(Host)`, `freeRam(Host)`, `freeGpus(Host)`, `canHost(Host, TaskRecord)`,
   `hostSpec()`. Tất cả đọc từ `hostPeUsage`/`hostRamUsage`/`gpuRegistry`
   (sau B2 fix, các con số này phản ánh load tức thời, không cumulative).
2. Đổi constructor `VmAllocationPolicyK8sDefault(SimulationManager mgr)`
   và `VmAllocationPolicyRandom(SimulationManager mgr, long seed)`.
   `leastRequestedScore` giờ đọc qua `mgr.freePes`, `mgr.freeRam`.
   `canHost` được delete khỏi từng policy, dùng chung `mgr.canHost` —
   ít trùng lặp, ít rủi ro lệch logic.
3. [`GatewayEntryPoint.reset`](../../cloudsim-java/src/main/java/sim/GatewayEntryPoint.java)
   truyền `manager` thay vì `manager.getGpuRegistry()`.

**Hệ quả số liệu mong đợi:** Δ K8s vs Random sẽ vượt khá xa biên 1-2 %
artefact cũ, vì K8s thật sự ưu tiên host load thấp ⇒ phân bố đều hơn ⇒
ít host bị bão hoà ⇒ ít SLA violation (nhờ B4 fix) ⇒ năng lượng vẫn
tương đương trong topology đồng nhất nhưng SLA differential rõ rệt.

**Verification:** Test `testB1_K8sAlwaysPicksLowestIndex` (giờ chỉ assert
"index 0 trên cluster trống → đúng vì tie-break determinism" và "index ≠ 0
sau khi pump 1 task vào host 0" — đúng bằng chứng K8s không còn đui mắt).

---

### B2 — Resource leak (✅ FIXED 2026-05-11)

**Trạng thái trước fix:** [`SimulationManager.allocateTask`](../../cloudsim-java/src/main/java/sim/SimulationManager.java)
chỉ có `merge(..., Integer::sum)` và `gpuState.allocate(...)`; `gpuState.release(...)`
được viết sẵn nhưng không có call-site.

**Fix:**
- Thêm `PriorityQueue<TaskCompletion> globalCompletions` (priority: ascending `endTime`).
- Mỗi `allocateTask` đẩy thêm 1 entry `{endTime = creationTime + duration, host, pes, ramMib, gpus}`.
- `advanceEnergy(toTime)` giờ là piecewise integrator: liên tục pop completion
  có `endTime ≤ tStop`, integrate sub-interval với state hiện tại, áp dụng
  release, lặp.
- Hoàn tất qua `flushTillEnd()` ở cuối episode để xử lý phần "tail" của
  tasks vẫn đang chạy quá lúc task cuối arrives.

**Verification:** Test `testB2_ResourcesNeverReleased` trong
[`ValidationRunner`](../../cloudsim-java/src/main/java/sim/ValidationRunner.java)
kiểm tra `sumPeUsage ≤ cluster capacity (640)` và quan sát được ≥1 lần
decrease.

**B2 hardening (cập nhật sau run đầu):** Run validator lần đầu phát hiện
`max PE = 660 > 640` — vượt capacity 20 PE. Nguyên nhân: `step(hostIdx)`
chỉ clamp index out-of-range, không check feasibility. Khi test
deliberately bơm `step(0)` bất chấp mask, `allocateTask` chạy ngay cả
khi host 0 hết chỗ → over-allocate.

Đây là gap defensive thật, không chỉ "lỗi test". Đã thêm fallback vào
`steppingLoop`:
- Nếu host được chọn không feasible → redirect sang host feasible đầu tiên.
- Nếu KHÔNG host nào fit → drop task, reward = [0, 0], log warning.

Trong production (MaskablePPO + baselines respect mask), code-path này
là no-op. Chỉ trigger khi agent/test bỏ qua mask — đúng vai trò defensive.

**Cleanup phụ:** xoá `host.enableUtilizationStats()` ở
[`DatacenterFactory.createHost`](../../cloudsim-java/src/main/java/sim/DatacenterFactory.java)
— vì không Vm nào được submit, CloudSim log spam `INFO Automatically
enabling computation of utilization statistics…` mỗi lần build host
(100+ dòng / run). Util đã track manual rồi, accessor đó là dead noise.

---

### B3 — Energy accounting (✅ FIXED 2026-05-11)

> **Reframe so với analyze-numbers.md:** Báo cáo đó suy đoán
> "tích phân theo logical time → đổi sang wall-clock của DES". Sau khi
> đọc kỹ source, kết luận này không hoàn toàn đúng — `creationTime` từ
> trace **CHÍNH LÀ** wall-clock của simulation: nó là dòng thời gian thật
> các task arrives, đơn vị giây. Vấn đề thật nằm ở B2: utilisation bị
> kẹt ở 1.0 khắp cluster do không release, nên `integrand = P_max`
> ⇒ `E = P_max × cluster × T` ≈ hai chục nghìn kWh. Sau khi B2 fix,
> integrand phản ánh đúng load tức thời.

**Fix code:**
1. Đảo thứ tự trong stepping loop: `advanceEnergy(task.creationTime())`
   gọi *trước* `allocateTask` để energy `[t_{i-1}, t_i]` được tính với
   state TRƯỚC khi task mới được thêm (state đúng vật lý cho khoảng đó).
2. `advanceEnergy` mới là piecewise integrator: nó dừng tại mỗi sự kiện
   release trong `(lastEnergyTimestamp, toTime]`, integrate sub-interval
   với utilisation đúng tại thời điểm đó, rồi áp dụng release, rồi tiếp.
3. `flushTillEnd()` chạy sau task cuối, kéo `lastEnergyTimestamp` tới
   `max(endTime)` để không bỏ sót phần đuôi.

**Verification:** `testB3_EnergyIntegratesLogicalTime` kiểm tra
`idle×T ≤ kwh ≤ max×T` (T = makespan). Với LOW chạy hết 1813 task,
energy sẽ giảm đáng kể so với 23,864 kWh báo cáo cũ — tuỳ trace, có thể
quanh `idle_cluster × T = 10 × 360 W × T / 3.6e6`. Nếu vẫn ra hàng chục
nghìn kWh, đó là do trace span ~120 ngày — không phải bug, mà là thiết
kế experiment (cần subset time window cho 1 episode).

**Lưu ý design:** `recordSnapshot` được dời ra sau `flushTillEnd`, nên
snapshot cuối cùng có `timestamp = lastCompletion`, không phải
`lastTask.creationTime`. ⇒ `makespan_sec` trong JSON giờ là *thời gian
thật cluster còn bận*, không phải "thời gian task cuối đến". Đây là
hành vi đúng cho metric.

---

### B4 — SLA violation (✅ FIXED 2026-05-11 lần 2)

**Trạng thái trước fix:** `deadline = deletionTime` và
`estimatedCompletion = creationTime + duration` (với `duration = deletion − max(creation, scheduled)`)
⇒ slack ≤ 0 cho mọi task ⇒ số 0 violations là *guaranteed*, không phụ
thuộc scheduler.

**Hai vấn đề độc lập cần fix cùng lúc:**
1. Định nghĩa deadline phải khác `deletionTime` (vốn là *thời điểm task
   đã thật sự xong* trong trace gốc — luôn đáp ứng theo định nghĩa).
2. Phải có cơ chế làm `estimatedCompletion` *phụ thuộc vào lựa chọn host*
   — không thì mọi scheduler ra cùng số violation, không có signal.

**Fix:**

1. **Deadline mới — slack budget theo QoS.** Thêm
   [`SimulationConfig.qosToSlackFactor(qos)`](../../cloudsim-java/src/main/java/sim/SimulationConfig.java):
   `LS=1.1, Guaranteed=1.3, Burstable=1.7, BE=3.0`. Ở
   [`AlibabaTraceReader`](../../cloudsim-java/src/main/java/sim/AlibabaTraceReader.java):
   ```
   deadline = creation + duration × slackFactor(qos)
   ```
   ⇒ LS chỉ được phép trễ 10 %; BE thoải mái 200 %.

2. **Completion ước lượng — congestion factor.** Trong
   [`SimulationManager.computeReward`](../../cloudsim-java/src/main/java/sim/SimulationManager.java):
   ```
   bgUtil          = (usedPes − pesNeeded) / pesCount   ∈ [0, 1]
   congestionFactor = 1.0 + bgUtil                       ∈ [1, 2]
   estimatedCompletion = creation + duration × congestionFactor
   ```
   Đặt task lên host TRỐNG: factor = 1.0, không trễ. Đặt lên host 80%:
   factor = 1.8, task chạy "lâu hơn 80%".

3. **Điều kiện trigger:** đổi `slaSlack > 0` thành `slaSlack > 1.0` (giây)
   để tránh false-positive do số dư float.

**Tính chất:** Với LS trên host empty (bgUtil = 0):
`slack = duration × (1 − 1.1) = −0.1 × duration ⇒ không vi phạm`.
Với LS trên host bgUtil ≥ 0.1 (factor ≥ 1.1):
`slack = duration × (factor − 1.1) ≥ 0 ⇒ vi phạm nếu > 1 s`.
BE: slack chỉ dương khi factor > 3.0, tức bgUtil > 2.0 (không thể) ⇒
gần như miễn nhiễm. Đúng semantics của QoS.

**Verification:** Test `testB4_SlaMathCannotTrigger` có hai assertion
độc lập:
- **B4a (integration):** chạy 1 episode LOW round-robin → assert
  `slaViolationCount ≥ 1`. Phủ định mệnh đề "không bao giờ trigger được"
  đại số trước fix.
- **B4b (mechanism):** tính trực tiếp công thức cho 1 task LS giả lập:
  trên host 50 % loaded → slack > 0 (vi phạm); trên host idle → slack ≤ 0
  (không vi phạm). Validation cơ chế độc lập với trace.

**Lý do bỏ "pump-vs-spread" assertion:** Phiên bản đầu của B4b assert
`pump_violations > spread_violations`. Run thực tế cho 156 vs 162 —
đảo chiều. Lý do: defensive fallback (B2 hardening) khiến `step(0)`
thực tế là first-fit, phân bố task lên host gần idle (sau khi host
trước có completion) → ít violation hơn round-robin (tích lũy đều).
Test đã sai về thiết kế — depend on emergent behavior. Xem mục 3.5 cho
chi tiết số liệu.

**Lưu ý design (limitation tự nhận):** `endTime` trong `globalCompletions`
vẫn dùng `creation + duration` (không nhân congestionFactor), nên
resource release vẫn theo trace's natural duration. Đây là cố ý chấp
nhận inconsistency để giữ implementation đơn giản — congestion là *signal
heuristic cho reward*, không phải mô phỏng chính xác của contention. Nếu
muốn fully self-consistent, phải đợi DES thực sự (P2 sau).

---

### B5. `avg_cpu_utilization` — đơn vị không xác định

[`SimulationManager.recordSnapshot`](../../cloudsim-java/src/main/java/sim/SimulationManager.java#L436-L438):
```java
double avgCpuUtil = hosts.isEmpty() ? 0.0 : hostPeUsage.values().stream()
        .mapToInt(Integer::intValue).average().orElse(0.0)
        / dcSpec.hostSpec().pesCount();
```
- Tử số = mean PE đã consume trên 10 host. Do B2, đây là *cumulative*
  count, **không clamp**, không trừ task đã xong.
- Mẫu số = 64.
- Kết quả: con số không có ý nghĩa "utilization" theo cách hiểu chuẩn.

Sau đó `buildSummary` ([MetricsExporter.java:131](../../cloudsim-java/src/main/java/sim/MetricsExporter.java#L131))
average lại trên *toàn bộ snapshot* (mỗi snapshot = 1 task arrival, không
phải Δt đều nhau) ⇒ kết quả là gần như "tích phân cumulative-load /
số task", một đại lượng vô danh.

**Sanity check số liệu** (giả định avg PE/task = 8.3 với LOW):
- Snapshot i có `avgCpuUtil_i = (i × 8.3) / (10 × 64) = i / 771`.
- Mean over 1813 snapshot = (1813+1)/2 / 771 ≈ **1.18** → khớp với report ÷ 10 = 11.79 / 10. Có lẽ percent_or_not chỉ là factor 10, nhưng dù sao bản chất công thức đã sai.

**Action P2:** thay bằng đúng đẳng thức:
```
clusterCpuUtil(t) = Σ min(usedPes_i(t), pesCount) / (n_hosts × pesCount)
```
và lấy trung bình theo *thời gian* (`Σ util × Δt / total_T`), không phải theo số snapshot.

Đồng thời cần export host-variance để đo load balance (đề xuất section 2.4 báo cáo của bạn cũng đã ghi).

---

### B6. Clamp không nhất quán

| Ngữ cảnh | Clamp? | Reference |
|---|---|---|
| Energy accounting | ✅ `Math.min(1.0, ...)` | [SimulationManager.java:339](../../cloudsim-java/src/main/java/sim/SimulationManager.java#L339), [SimulationManager.java:372](../../cloudsim-java/src/main/java/sim/SimulationManager.java#L372) |
| Observation builder | ✅ | [SimulationManager.java:409-410](../../cloudsim-java/src/main/java/sim/SimulationManager.java#L409-L410) |
| Snapshot `avgCpuUtil` | ❌ | [SimulationManager.java:436-438](../../cloudsim-java/src/main/java/sim/SimulationManager.java#L436-L438) |

Hai phần trên dùng cùng `hostPeUsage` nhưng cho ra hai dải giá trị khác
nhau. Reward (clamp) đúng vật lý hơn; metric (no-clamp) sai. Phải đồng
bộ — đề xuất giữ logic clamp ở mức "single source of truth", ví dụ
private method `hostCpuUtil(host) -> double`.

---

### B7. `ScenarioFilter.HIGH` = no-op

```java
private static List<TaskRecord> filterHigh(List<TaskRecord> tasks) {
    return new ArrayList<>(tasks);  // ← chỉ copy
}
```
Bug nhãn dán nhiều hơn là bug logic, nhưng dễ gây hiểu nhầm trong báo
cáo ("HIGH" gợi ý subset). Khuyến nghị đổi tên thành `FULL` *hoặc* implement
như đề xuất 2.6 của analyze-numbers.md (lấy windows ở 70th-percentile
theo `cpu_milli + num_gpu×4000`).

---

### B8. Custom allocation policy không vào CloudSim path

[`DatacenterFactory.create(simulation, dcSpec, gpuRegistry)`](../../cloudsim-java/src/main/java/sim/DatacenterFactory.java#L67-L71)
mặc định dùng `VmAllocationPolicySimple`. [`SimulationManager.buildSimulation`](../../cloudsim-java/src/main/java/sim/SimulationManager.java#L202)
gọi đúng overload đó — KHÔNG truyền policy custom.

⇒ Hai class `VmAllocationPolicyK8sDefault` / `VmAllocationPolicyRandom`
hiện chỉ có ích qua hàm "phụ" `selectHostForTask` (Python gọi qua Py4J).
Mọi `defaultFindHostForVm` (cloudsim path) là dead code.

Đây không phải bug nếu intent rõ ràng. Đề xuất ghi chú "intentional" trong
class javadoc hoặc khi DES được kích hoạt, switch sang truyền policy thực
sự vào datacenter — tùy P2.

---

### B9. Default constructor truyền `pesCount()` làm seed

[SimulationManager.java:107-112](../../cloudsim-java/src/main/java/sim/SimulationManager.java#L107-L112):
```java
public SimulationManager() {
    this(SimulationConfig.DEFAULT_DC,
         SimulationConfig.TRACE_FILE,
         Scenario.HIGH,
         SimulationConfig.DEFAULT_DC.hostSpec().pesCount()); // just reuse seed from env
}
```
Comment ghi "reuse seed from env" nhưng giá trị thực truyền là `pesCount = 64`.
Lỗi copy-paste. Đường code đi qua `GatewayEntryPoint.reset(String, long)`
nên seed đúng được set từ Python — bug này chỉ ảnh hưởng khi ai đó dùng
default constructor (vd. test JVM main). Vẫn nên sửa.

**Fix:**
```java
public SimulationManager() {
    this(SimulationConfig.DEFAULT_DC, SimulationConfig.TRACE_FILE, Scenario.HIGH,
         Long.parseLong(System.getenv().getOrDefault("RANDOM_SEED", "42")));
}
```

---

### B10. `getActionMask` khi episodeDone

[SimulationManager.java:179-181](../../cloudsim-java/src/main/java/sim/SimulationManager.java#L179-L181):
```java
if (episodeDone || currentTaskIdx >= tasks.size()) {
    return new boolean[hosts.size()];  // all false
}
```
MaskablePPO sample từ mask all-false sẽ throw. Trên Python side đã có
check `done` trước nhưng đề phòng race condition (Python lấy mask trước
khi process xong `done`), nên trả mask all-true hoặc raise rõ ràng tốt
hơn.

---

## 3. Roadmap fix — trạng thái 2026-05-11

| Bước | Bug | Trạng thái | Ghi chú |
|---|---|---|---|
| 1 | B2 — resource release | ✅ done | piecewise integrator + completion PQ |
| 2 | B3 — energy accounting | ✅ done | đã đảo thứ tự advance ↔ allocate, thêm flushTillEnd |
| 3 | B6 — clamp consistency | ✅ done | side-effect của B2/B3 fix |
| 4 | B5 — util formula | 🟡 partial | clamp ổn, time-weighting defer (xem Known limitations) |
| 5 | B1 — K8s state | ✅ done | policy đọc state qua `mgr.freePes/freeRam/canHost` |
| 6 | B4 — SLA detection | ✅ done | deadline = duration × slack(qos); completion có congestion factor |
| 7 | B7 — HIGH = no-op | ❌ defer | rename → FULL, hoặc làm filter thực — không gate P2 |
| 8 | B8/B9/B10 — cleanup | ❌ defer | không gate Phase 2 |

**Cả 4 Critical bug đã đóng. Phase 2 weight sweep có thể bắt đầu sau khi
chạy `ValidationRunner` confirm 10/10 PASS và một baseline run nhanh trên
LOW để sanity-check số mới.**

---

## 3.5. Validation runs — observed numbers (post-fix, 2026-05-11)

### Run 1 — trước B2 hardening (1 FAIL: B2a)

Test B2a fail vì `step(0)` không check feasibility ⇒ over-allocation
khi test cố tình bỏ qua mask. Đã fix bằng defensive fallback trong
steppingLoop (xem B2 hardening).

### Run 2 — sau B2 hardening (1 FAIL: B4b)

| Test | Số liệu thực |
|---|---|
| B1 | empty cluster → K8s pick 0; sau khi pump 1 task vào host 0 → K8s pick 1 |
| B2 | max PE-usage = **484 ≤ 640** ✓ |
| B3 | T_end = 12.9M s (~149 ngày), energy = **27,094 kWh**; bounds: idle=12,903, max=100,356 |
| B4 | pump-to-0 = **156**, round-robin = **162** — **đảo chiều so với kỳ vọng** |
| B5 | max clusterCpuUtil = **0.76** (max load momentary) |
| B6 | host0 obs util = 1.0, snapshot cluster avg = **0.49** — đều ∈ [0,1] |

**Phân tích B4 đảo chiều:**

Test giả định "pump tất cả lên host 0 → contention cao → nhiều violation
hơn". Nhưng defensive fallback (B2 hardening) khiến `step(0)` thực tế
trở thành **first-fit**: lấp đầy host 0 → host 1 → … (vì khi host 0 hết
chỗ thì redirect sang host kế).

First-fit ⇒ phần lớn task land lên host gần idle (host trống tiếp theo,
sau khi host trước đã có tasks complete) ⇒ bgUtil thấp ⇒ ít violation.
Round-robin ⇒ tích lũy load đều trên 10 host ⇒ bgUtil trung bình cao
hơn ⇒ nhiều violation hơn. Số liệu khớp.

⇒ Test B4b ban đầu sai về thiết kế — nó phụ thuộc emergent behavior
trên một workload không phân biệt được scheduler. **Đã sửa**: thay B4b
bằng direct formula check (LS slack > 0 trên host 50 % loaded, ≤ 0 trên
host idle). Validation cơ chế độc lập với trace data — đúng deterministic.

### Run 3 — sau B4 test fix (kỳ vọng 15/15 PASS)

Đang chạy. Console output ghi:
- B4 round-robin violations ~162 (cứ giữ làm baseline số)
- B4 formula: slack trên 50 %-loaded = 40 s, trên idle = -10 s

### Quan sát cross-cutting

1. **Task drops:** 4-6 task trên 1813 (≤ 0.3 %) bị drop vì yêu cầu
   `cpu=88, ram=320 GB, gpu=8` vượt capacity host (64 PE, 256 GB, 8 GPU).
   Đây là task từ cluster heterogeneous gốc — không scheduable trên
   topology hiện tại. Không phải bug; ghi vào limitations.
2. **Energy 27 kWh (run 2 vs run 1's 18.6 kWh):** chênh ~46 % do B2
   hardening làm utilization thật cao hơn (max 0.76 vs 0.10 trước đó),
   năng lượng active cao hơn. Cả hai đều trong bounds vật lý.
3. **Cluster max util 0.76 (vs 0.10 cũ):** trước hardening, test "pump
   to host 0" làm host 0 over-allocate (counted via raw count), nhưng
   các snapshot khác clamp về 1.0 cho host 0, mean ~0.1. Sau hardening,
   tasks phân bố hơn, mean cluster lên 0.49 — phản ánh tải thực tế khi
   trace gặp các burst arrival.
4. **MORL Phase 2:** Workload LOW thưa ⇒ scheduler-differential nhỏ.
   Khi switch HIGH/BURST, differential rộng hơn. Đừng tweak
   `slackFactor`/`congestionFactor` để "ép" số đẹp ở Phase 1.

---

## 4. Cách reproduce / chạy validator

Validator là một `main()` class duy nhất nằm trong
[`cloudsim-java/src/main/java/sim/ValidationRunner.java`](../../cloudsim-java/src/main/java/sim/ValidationRunner.java),
được shade vào fat jar nên không cần build pipeline riêng:

```powershell
docker compose build cloudsim-java
docker compose run --rm --no-deps --entrypoint java cloudsim-java `
    -cp simulation.jar sim.ValidationRunner
```

`--entrypoint java` ghi đè `ENTRYPOINT ["java", …, "-jar", "simulation.jar"]`
trong Dockerfile (nếu không override, args được nối vào sau `-jar` và Main.java
chỉ start Py4J server rồi block). `--no-deps` không kéo rl-agent lên.
Trace volume vẫn auto-mount.

Output `[PASS]`/`[FAIL]` per assertion. Dùng làm CI smoke check trước
mỗi lần thay đổi `SimulationManager` hoặc trước khi nộp Phase 2.
