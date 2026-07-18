#!/usr/bin/env bash
# =============================================================================
# ELDAS — chạy TOÀN BỘ phần còn nợ của GĐ2 → output cuối (bảng + figure).
#
# Một lệnh, đã bao gồm mọi bước tăng tốc:
#   • SYS.2 packed transport (tự bật qua capability-probe khi build image mới)
#   • SYS.5 torch_threads=1 (mặc định trong train_cmdp / train_min)
#   • SYS.1/C9 NUM_GATEWAYS + sweep --parallel  (song song theo RUN)
#   • MONITORING_ENABLED=false ép cho toàn job (bắt buộc khi NUM_GATEWAYS>1, C11)
#
# Với mỗi scenario, chạy đúng thứ tự các món đang nợ (C6):
#   1) NSGA-II reference front (offline)
#   2) fixed-weight PPO "ppo-min" (baseline Phase-1, seed 42)
#   3) CMDP-PID budget sweep  (5 budget × N seed, song song --parallel)
#   4) heuristics + campaign → bảng mean±95%CI + HV/IGD+ + Pareto figure
#      (tự gộp sweep + ppo-min + NSGA-II)
#
# CÁCH DÙNG:
#   bash scripts/run-full-campaign.sh pilot     # ~10–15 phút, LOW, 1 budget — CHẠY CÁI NÀY TRƯỚC
#   bash scripts/run-full-campaign.sh full      # toàn bộ 3 scenario (xem estimate ở cuối/README)
#   MODE=full SCENARIOS="LOW BURST" bash scripts/run-full-campaign.sh   # tuỳ biến
#
# Knob (env override): SCENARIOS SEEDS BUDGETS EPISODES PARALLEL PPO_EPISODES
# =============================================================================
set -uo pipefail
export MSYS_NO_PATHCONV=1          # Git Bash: đừng dịch path /data/... (D2)

# ── Che do + scenario + budget ───────────────────────────────────────────────
# CACH DUNG:  bash scripts/run-full-campaign.sh <MODE> [SCENARIOS] [BUDGETS]
#   MODE       = pilot | full
#   SCENARIOS  = (tuy chon) danh sach scenario, ngan bang dau phay HOAC khoang
#                trang-trong-ngoac-kep. Truyen THANG lam tham so la cach ON DINH
#                nhat, KHONG phu thuoc bien moi truong (env PowerShell->bash hay loi).
#   BUDGETS    = (tuy chon) danh sach ngan sach d, ngan bang dau phay. Moi scenario
#                co "san SLA" rieng nen luoi budget TOT NHAT khac nhau (xem Tracking
#                §14): HIGH ~ 0.02..0.06 | LOW ~ 0.04..0.08 | BURST ~ 0.005..0.04.
# Vi du:
#   bash scripts/run-full-campaign.sh full LOW                              # chi LOW, luoi mac dinh
#   bash scripts/run-full-campaign.sh full HIGH 0.02,0.03,0.04,0.05,0.06    # HIGH, luoi tuned
#   bash scripts/run-full-campaign.sh full BURST 0.005,0.01,0.02,0.03,0.04  # BURST, luoi tight
#   bash scripts/run-full-campaign.sh full "HIGH BURST"                     # 2 scenario, luoi mac dinh
MODE="${1:-${MODE:-pilot}}"
SCENARIOS_ARG="${2:-}"           # tham so vi tri #2 (uu tien cao nhat)
BUDGETS_ARG="${3:-}"             # tham so vi tri #3 (uu tien cao nhat)

if [[ "$MODE" == "pilot" ]]; then
  DEF_SCENARIOS="LOW"; DEF_SEEDS="42,43"; DEF_BUDGETS="0.04"
  DEF_EPISODES=40; DEF_PPO=40; DEF_PAR=2
elif [[ "$MODE" == "full" ]]; then
  DEF_SCENARIOS="LOW HIGH BURST"; DEF_SEEDS="42,43,44,45,46"
  DEF_BUDGETS="0.02,0.04,0.06,0.08,0.10"; DEF_EPISODES=200; DEF_PPO=200; DEF_PAR=4
else
  echo "MODE phai la 'pilot' hoac 'full'. Nhan: '$MODE'" >&2; exit 2
fi

# Uu tien: tham so dong lenh > env > mac dinh theo MODE.
SCENARIOS="${SCENARIOS_ARG:-${SCENARIOS:-$DEF_SCENARIOS}}"
SCENARIOS="${SCENARIOS//,/ }"    # "LOW,BURST" -> "LOW BURST"
BUDGETS="${BUDGETS_ARG:-${BUDGETS:-$DEF_BUDGETS}}"

SEEDS="${SEEDS:-$DEF_SEEDS}"
EPISODES="${EPISODES:-$DEF_EPISODES}"
PPO_EPISODES="${PPO_EPISODES:-$DEF_PPO}"
PARALLEL="${PARALLEL:-$DEF_PAR}"

# Fail-fast: chan scenario go sai (khong de job chay hang gio roi moi loi).
for SC in $SCENARIOS; do
  case "$SC" in
    LOW|HIGH|BURST) ;;
    *) echo "Scenario khong hop le: '$SC'. Chi LOW | HIGH | BURST" >&2; exit 2 ;;
  esac
done

# Episode length mỗi scenario (đã ĐO trên trace thật) → đổi #episode ra #timestep
# để số dual-update (≈ episode) ĐỒNG NHẤT giữa các scenario (điều kiện so sánh công bằng).
tasks_of() { case "$1" in
  LOW) echo 1813 ;; HIGH) echo 7255 ;; BURST) echo 3569 ;; *) echo 1813 ;; esac; }

PY4J_PORT="${PY4J_PORT:-25333}"
PORT_MAX=$(( PY4J_PORT + PARALLEL - 1 ))
export NUM_GATEWAYS="$PARALLEL" PY4J_PORT_MAX="$PORT_MAX" MONITORING_ENABLED=false

R() { docker compose run --rm "$@"; }                 # gateway-facing
Ro() { docker compose run --rm --no-deps "$@"; }       # offline (không cần gateway)
hr() { printf '\n\033[1m── %s ──\033[0m\n' "$*"; }

t0=$(date +%s)
hr "SETUP  (MODE=$MODE · scenarios=[$SCENARIOS] · seeds=$SEEDS · budgets=$BUDGETS · episodes=$EPISODES · parallel=$PARALLEL)"

# 0) Build — MẶC ĐỊNH TỰ BỎ QUA nếu cả 2 image đã tồn tại (tránh vô tình tải lại
#    ~1GB ML stack). Chỉ build khi image THIẾU, hoặc khi ép FORCE_BUILD=1 (dùng
#    sau khi đổi code Java/Python/requirements). SKIP_BUILD=1 = ép bỏ qua kể cả
#    khi thiếu image (hiếm khi cần).
have_imgs=1
docker image inspect eldas-cloudsim-java >/dev/null 2>&1 || have_imgs=0
docker image inspect eldas-rl-agent      >/dev/null 2>&1 || have_imgs=0

if [[ "${SKIP_BUILD:-0}" == "1" ]]; then
  echo "SKIP_BUILD=1 → bỏ qua build (dùng image sẵn có)."
elif [[ "${FORCE_BUILD:-0}" == "1" ]]; then
  echo "FORCE_BUILD=1 → build lại image..."
  docker compose build cloudsim-java rl-agent || { echo "build failed"; exit 1; }
elif [[ "$have_imgs" == "1" ]]; then
  echo "Image eldas-cloudsim-java + eldas-rl-agent đã có ⇒ bỏ qua build."
  echo "  (đổi code rồi? chạy lại với  FORCE_BUILD=1  để build mới.)"
else
  echo "Chưa có image ⇒ build lần đầu (có thể lâu tùy mạng)..."
  docker compose build cloudsim-java rl-agent || { echo "build failed"; exit 1; }
fi

# 1) N gateway độc lập trong 1 JVM, monitoring TẮT
docker compose up -d --force-recreate cloudsim-java || exit 1
echo "chờ gateway sẵn sàng..."; sleep 14
GW=$(docker compose logs cloudsim-java 2>&1 | grep -c "GatewayServer listening")
echo "gateways đang chạy: $GW (cần $PARALLEL)"
[[ "$GW" -ge "$PARALLEL" ]] || { echo "THIEU gateway - kiem tra NUM_GATEWAYS/PY4J_PORT_MAX"; exit 1; }

# 2) Sanity nhanh: Java validation (fail fast trước khi tốn hàng giờ).
#    Dùng direct bind mount (không phải named volume eldas_trace-data — đã bỏ).
hr "Java ValidationRunner (sanity)"
docker run --rm -v "$PWD/data/alibaba-trace:/data/trace:ro" --entrypoint java \
  eldas-cloudsim-java -cp simulation.jar sim.ValidationRunner 2>&1 | grep -E "^PASS =|^FAIL ="

# ── Vòng theo scenario ──────────────────────────────────────────────────────
for SC in $SCENARIOS; do
  TASKS=$(tasks_of "$SC")
  TS=$(( EPISODES * TASKS ))
  PPO_TS=$(( PPO_EPISODES * TASKS ))
  hr "SCENARIO $SC  (timesteps/run = $EPISODES ep × $TASKS task = $TS)"

  # 1) NSGA-II reference front (offline, không cần gateway)
  echo "[1/4] NSGA-II reference front..."
  Ro --entrypoint python rl-agent \
    src/eval/nsga2_baseline.py --scenario "$SC" --max-tasks 80 --pop-size 80 --n-gen 60 \
    || echo "  [warn] NSGA-II loi, bo qua - campaign van chay khong co no"

  # 2) fixed-weight PPO (ppo-min, seed 42) — baseline Phase-1
  echo "[2/4] ppo-min (fixed-weight PPO, $PPO_EPISODES ep)..."
  R --entrypoint python rl-agent \
    src/train_min.py --scenario "$SC" --seed 42 --total-timesteps "$PPO_TS" \
    --model-out "/data/models/ppo-min-${SC}.zip" \
    --eval-out "/data/results/baseline-${SC}/ppo-min" \
    || echo "  ⚠ ppo-min lỗi (bỏ qua)"

  # 3) CMDP-PID budget sweep — song song theo RUN (đòn bẩy C9)
  echo "[3/4] CMDP sweep: budgets=[$BUDGETS] × seeds=[$SEEDS], --parallel $PARALLEL..."
  R --entrypoint python rl-agent \
    src/eval/sweep_budget.py --scenario "$SC" \
    --budgets "$BUDGETS" --seeds "$SEEDS" \
    --total-timesteps "$TS" --k-p 0.05 --k-i 0.05 --parallel "$PARALLEL" \
    || echo "  [warn] sweep loi/non-monotone - xem verdict o tren, co the can train lau hon. C1"

  # 4) Campaign: heuristics + gộp sweep + ppo-min + NSGA-II → bảng + figure
  echo "[4/4] campaign aggregate..."
  R --entrypoint python rl-agent \
    src/eval/run_campaign.py --scenario "$SC" --seeds "$SEEDS" --run-baselines \
    || echo "  ⚠ campaign lỗi"

  echo "→ $SC xong: /data/results/campaign-${SC}/{table.md, pareto-${SC}.png, campaign_summary.json}"
done

# ── Kết thúc ────────────────────────────────────────────────────────────────
docker compose up -d --force-recreate cloudsim-java >/dev/null 2>&1   # tra ve 1 gateway mac dinh
t_end=$(date +%s)
mins=$(( (t_end - t0) / 60 ))
hr "HOAN TAT sau ~${mins} phut"
echo "Output: data/results/campaign-<SC>/table.md + pareto-<SC>.png cho: $SCENARIOS"
echo
echo "GHI CHU (day la nhac chung, KHONG phai ket qua):"
echo " - Verdict MONOTONE/NON-MONOTONE THAT cua tung scenario nam o khoi 'G2.5 sweep'"
echo "   phia tren (moi scenario in rieng), hoac trong sweep-<SC>/sweep_summary.json."
echo " - NON-MONOTONE thuong do NHIEU SEED > khoang cach budget (front nong, nhat la LOW),"
echo "   KHONG phai loi hoi tu: lambda*(d) van don dieu. Xem Tracking_detail.md muc 14.3."
echo " - Hinh Pareto + HV/IGD+ dung bao NON-DOMINATED, khong phu thuoc check don dieu nay."
