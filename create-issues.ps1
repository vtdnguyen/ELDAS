# Thiết lập encoding UTF-8 cho PowerShell
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
$OutputEncoding = [System.Text.Encoding]::UTF8
$PSDefaultParameterValues['Out-File:Encoding'] = 'utf8'

$REPO = gh repo view --json nameWithOwner -q ".nameWithOwner"
Write-Host "Creating issues for repo: $REPO" -ForegroundColor Cyan

# ── Tạo Milestone ──────────────────────────────────────────
Write-Host "Creating milestone..." -ForegroundColor Yellow
$MILESTONE_TITLE = "Phase 1 - Foundation"

# Thử tạo milestone mới (nếu đã tồn tại từ những lần chạy trước, nó sẽ báo lỗi nhẹ nhưng không sao)
gh api repos/$REPO/milestones `
  --method POST `
  -F title=$MILESTONE_TITLE `
  -F description="Xây dựng nền tảng mô phỏng CloudSim + MORL environment." `
  -F due_on="2026-05-01T00:00:00Z" `
  --silent 2>$null

Write-Host "Milestone Title: $MILESTONE_TITLE" -ForegroundColor Green

# ── Hàm tạo issue (Sử dụng file tạm để chống lỗi Font) ─────
function New-GithubIssue {
    param([string]$Title, [string]$Labels, [string]$Body)
    Write-Host "  Creating: $Title"
    $tmpFile = [System.IO.Path]::GetTempFileName()
    [System.IO.File]::WriteAllText($tmpFile, $Body, [System.Text.Encoding]::UTF8)
    
    # Sử dụng thẳng tên Milestone thay vì ID để tránh lỗi not found
    gh issue create --title $Title --label $Labels --milestone $MILESTONE_TITLE --body-file $tmpFile
    Remove-Item $tmpFile
}

Write-Host "`n=== WEEK 1: Docker Foundation ===" -ForegroundColor Magenta

New-GithubIssue -Title "[T1.1] Scaffold docker-compose.yml với 2 service" -Labels "infrastructure,docker" -Body @"
## Mô tả
Tạo file docker-compose.yml định nghĩa 2 service: cloudsim-java và rl-agent.

## Acceptance criteria
- [ ] docker-compose.yml có đủ 2 service, 3 volume, 1 network
- [ ] .env có đủ biến: PY4J_PORT, ENERGY_WEIGHT, SLA_WEIGHT, WANDB_API_KEY
- [ ] docker compose config chạy không lỗi
- [ ] .env.example được commit (không commit .env thật)
"@

New-GithubIssue -Title "[T1.2] Dockerfile cho cloudsim-java (multi-stage)" -Labels "infrastructure,docker,java" -Body @"
## Mô tả
Tạo Dockerfile multi-stage cho service cloudsim-java.

## Acceptance criteria
- [ ] Stage 1: eclipse-temurin:21-jdk-jammy + Maven 3.9.9
- [ ] Stage 2: eclipse-temurin:21-jre-jammy chỉ chứa JAR
- [ ] pom.xml dùng cloudsimplus:8.5.7 và py4j:0.10.9.7
- [ ] docker build thành công không lỗi
"@

New-GithubIssue -Title "[T1.3] Dockerfile cho rl-agent (multi-stage)" -Labels "infrastructure,docker,python" -Body @"
## Mô tả
Tạo Dockerfile multi-stage cho service rl-agent.

## Acceptance criteria
- [ ] Stage 1: python:3.11.9-slim-bookworm + gcc để compile deps
- [ ] Stage 2: runtime slim không có build tools
- [ ] Import test pass: mo_gymnasium==1.3.2, stable_baselines3==2.7.0, py4j==0.10.9.7
"@

New-GithubIssue -Title "[T1.4] Validate cả 2 service build và chạy" -Labels "infrastructure,docker" -Body @"
## Mô tả
Xác nhận toàn bộ stack khởi động thành công, 2 container chạy song song.

## Acceptance criteria
- [ ] docker compose build không lỗi
- [ ] docker compose up in log từ cả 2 container
- [ ] Java container: in 'CloudSim simulation container started.'
- [ ] Python import test: mo_gymnasium, py4j, wandb, stable_baselines3 đều OK
"@

Write-Host "`n=== WEEK 2: Data & CloudSim ===" -ForegroundColor Magenta

New-GithubIssue -Title "[T2.1] DatacenterFactory - Host với PowerModel" -Labels "java,cloudsim" -Body @"
## Mô tả
Viết DatacenterFactory.java tạo Datacenter ảo với các Host có mô hình tiêu thụ điện.

## Acceptance criteria
- [ ] Tạo 5 Host, mỗi Host: 96 CPU PE (1000 MIPS/PE), 256GB RAM, 1TB storage
- [ ] Mỗi Host gắn PowerModelHostSimple(maxPower=350, idlePower=150)
- [ ] Chạy được trong Docker container
"@

New-GithubIssue -Title "[T2.2] AlibabaTraceReader - parse CSV thành Cloudlet" -Labels "java,cloudsim,data" -Body @"
## Mô tả
Viết AlibabaTraceReader.java đọc file Alibaba GPU Cluster Trace v2023.

## Acceptance criteria
- [ ] Đọc được openb_pod_list_default.csv từ /data/trace/
- [ ] Map cpu_milli -> số PE của Vm
- [ ] Map memory_mib -> RAM của Vm (MB)
- [ ] Map num_gpu -> extended resource attribute
"@

New-GithubIssue -Title "[T2.3] ScenarioFilter - Low / High / Burst load" -Labels "java,cloudsim,data" -Body @"
## Mô tả
Viết ScenarioFilter.java phân loại Cloudlet thành 3 kịch bản tải từ Alibaba Trace.

## Acceptance criteria
- [ ] Low load: creation_time trong khung 0h-6h, inter-arrival time > 60s
- [ ] High load: mật độ task cao, tổng cpu_milli > 80% capacity
- [ ] Burst load: xuất hiện task num_gpu >= 4 trong window 5 phút
"@

New-GithubIssue -Title "[T2.4] MetricsExporter - xuất energy và SLA ra CSV" -Labels "java,cloudsim" -Body @"
## Mô tả
Viết MetricsExporter.java ghi kết quả mỗi simulation episode ra file CSV.

## Acceptance criteria
- [ ] Ghi file results/metrics_{timestamp}.csv
- [ ] Columns: episode, scenario, host_id, energy_joules, makespan_s, sla_violations
"@

Write-Host "`n=== WEEK 3: Py4J Bridge & Baseline ===" -ForegroundColor Magenta

New-GithubIssue -Title "[T3.1] GatewayEntryPoint + GatewayServer - Py4J bridge" -Labels "java,python,py4j" -Body @"
## Mô tả
Xây dựng cầu nối RPC giữa Java CloudSim và Python RL agent qua Py4J Gateway.

## Acceptance criteria
- [ ] GatewayServer start trên port 25333 khi Main.java chạy
- [ ] Python kết nối được từ container rl-agent
- [ ] Python gọi được: bindCloudletToHost(cloudletId, hostId)
"@

New-GithubIssue -Title "[T3.2] VmAllocationPolicyK8sDefault - baseline scheduler" -Labels "java,cloudsim" -Body @"
## Mô tả
Implement Kubernetes default scheduler làm baseline so sánh với MORL agent.

## Acceptance criteria
- [ ] Filter phase: loại Host không đủ CPU/RAM/GPU
- [ ] Score phase: LeastRequestedPriority
- [ ] Chạy được qua 3 scenario (Low/High/Burst)
"@

Write-Host "`n=== WEEK 4: Python RL Environment ===" -ForegroundColor Magenta

New-GithubIssue -Title "[T4.1] state_builder.py - chuẩn hoá observation vector" -Labels "python,rl,gymnasium" -Body @"
## Mô tả
Viết state_builder.py chuyển trạng thái CloudSim thành numpy vector chuẩn hoá.

## Acceptance criteria
- [ ] Extract util, active_power, idle_power của từng Host
- [ ] Tất cả giá trị normalize về [0, 1]
"@

New-GithubIssue -Title "[T4.2] reward.py - hàm phần thưởng kép R_energy và R_SLA" -Labels "python,rl" -Body @"
## Mô tả
Viết reward.py tính vector phần thưởng 2 chiều cho MORL agent.

## Acceptance criteria
- [ ] R_energy = -(E_t+1 - E_t)
- [ ] R_SLA = -λ * max(0, predicted_completion - deadline)
"@

New-GithubIssue -Title "[T4.3] environment.py - custom MOEnv kết nối Py4J" -Labels "python,rl,gymnasium" -Body @"
## Mô tả
Viết environment.py định nghĩa MORL environment theo chuẩn gymnasium.

## Acceptance criteria
- [ ] Kế thừa gymnasium.Env, reward space là Box(2,)
- [ ] observation_space: Box shape (N_hosts * 5 + 4,)
- [ ] step(action): gửi bindCloudletToHost -> Java, tính reward
"@

New-GithubIssue -Title "[T4.4] tracker.py + baseline_eval.py" -Labels "python,tracking,evaluation" -Body @"
## Mô tả
Tích hợp experiment tracking và viết script đánh giá baseline.

## Acceptance criteria
- [ ] Init Wandb run với config từ .env
- [ ] Log mỗi episode: reward_energy, reward_sla, energy_joules
- [ ] Chạy K8sDefault qua 3 scenario và lưu bảng kết quả
"@

New-GithubIssue -Title "[T4.5] Báo cáo tổng kết Giai đoạn 1" -Labels "documentation" -Body @"
## Mô tả
Viết báo cáo chứng minh môi trường mô phỏng hoàn chỉnh.

## Acceptance criteria
- [ ] Mô tả kiến trúc 4 phân hệ
- [ ] Bảng baseline K8sDefault qua 3 scenario
- [ ] Kết luận: môi trường sẵn sàng cho MORL training
"@

Write-Host "`n✅ All issues created successfully!" -ForegroundColor Green
Write-Host "👉 View your project: https://github.com/$REPO/issues" -ForegroundColor Cyan