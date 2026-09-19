#!/usr/bin/env bash
# =============================================================================
# ELDAS — ĐIỀU PHỐI JOB CHẠY ĐÊM (một lệnh, treo máy, có log + audit)
#
# Chạy toàn bộ pipeline GĐ2 với THAM SỐ lấy từ scripts/overnight.conf:
#   0) archive data/results cũ  → data/results-archive/<ts>-<label>/  (audit)
#   1) build (nếu thiếu image) + N gateway độc lập (MONITORING tắt)
#   2) Java ValidationRunner (sanity, fail-fast)
#   Mỗi scenario (LOW/HIGH/BURST tuỳ config):
#     a) NSGA-II reference front            (offline)
#     b) CMDP-PID budget sweep (lưới TUNED) (song song --parallel)
#     c) fixed-weight PPO FRONT (weights)   (song song --parallel)  ← đối thủ công bằng
#     d) campaign: 5 heuristic + gộp tất cả → bảng mean±95%CI + HV/IGD+ + Pareto figure
#   3) trả gateway về mặc định 1
#
# TẤT CẢ log → data/results/logs/overnight-<ts>.log  (tee: vừa hiện, vừa lưu).
#
# CÁCH DÙNG:
#   bash scripts/overnight-run.sh                    # nạp scripts/overnight.conf
#   CONFIG=scripts/my.conf bash scripts/overnight-run.sh
#   SCENARIOS=HIGH EPISODES=300 bash scripts/overnight-run.sh   # override tạm
#
#   # CHẠY THỬ 15 PHÚT TRƯỚC KHI TREO CẢ ĐÊM (bắt buộc — kiểm tra wiring):
#   SCENARIOS=LOW SEEDS=42,43 EPISODES=25 PPO_EPISODES=25 \
#     PPO_WEIGHTS=0.5 BUDGETS_LOW=0.05,0.06 PARALLEL=2 RUN_LABEL=pilot \
#     bash scripts/overnight-run.sh
# =============================================================================
set -uo pipefail
export MSYS_NO_PATHCONV=1              # Git Bash: đừng dịch path /data/... (D2)

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

# ── Nạp cấu hình (env override thắng, nhờ idiom :"${VAR:=...}") ───────────────
CONFIG="${CONFIG:-scripts/overnight.conf}"
[[ -f "$CONFIG" ]] || { echo "Khong thay config: $CONFIG" >&2; exit 2; }
# shellcheck disable=SC1090
source "$CONFIG"

# ── Chuẩn bị log (tee toàn bộ phiên) ─────────────────────────────────────────
TS="$(date +%Y%m%d-%H%M%S)"
LOG_DIR="data/results/logs"; mkdir -p "$LOG_DIR"
LOG="$LOG_DIR/${RUN_LABEL}-${TS}.log"
exec > >(tee -a "$LOG") 2>&1          # mọi stdout/stderr từ đây → màn hình + file

hr() { printf '\n\033[1m── %s ──\033[0m\n' "$*"; }
tasks_of() { case "$1" in LOW) echo 1813;; HIGH) echo 7255;; BURST) echo 3569;; *) echo 1813;; esac; }
budgets_of() { local v="BUDGETS_$1"; echo "${!v:-0.02,0.04,0.06,0.08,0.10}"; }

SCENARIOS="${SCENARIOS//,/ }"         # "LOW,BURST" → "LOW BURST"
for SC in $SCENARIOS; do case "$SC" in LOW|HIGH|BURST);; *)
  echo "Scenario khong hop le: '$SC'" >&2; exit 2;; esac; done

# ── Gateway / port / monitoring ──────────────────────────────────────────────
PY4J_PORT="${PY4J_PORT:-25333}"
PORT_MAX=$(( PY4J_PORT + PARALLEL - 1 ))
# METRICS_PORT: exporter tắt, nhưng compose vẫn publish cổng ⇒ phải NGOÀI dải Windows
# cấm (9091–9190). Config đặt default 19091; export để `docker compose` (tiến trình con)
# thấy — env của shell THẮNG giá trị 9091 trong .env.
export NUM_GATEWAYS="$PARALLEL" PY4J_PORT_MAX="$PORT_MAX" MONITORING_ENABLED=false \
       METRICS_PORT="${METRICS_PORT:-19091}"

R()  { docker compose run --rm "$@"; }             # cần gateway
Ro() { docker compose run --rm --no-deps "$@"; }   # offline

t0=$(date +%s)
hr "OVERNIGHT START  ($TS)"
cat <<EOF
config      : $CONFIG
log         : $LOG
scenarios   : [$SCENARIOS]
seeds       : $SEEDS
episodes    : CMDP=$EPISODES · PPO=$PPO_EPISODES
budgets     : LOW=[$(budgets_of LOW)] HIGH=[$(budgets_of HIGH)] BURST=[$(budgets_of BURST)]
ppo weights : $PPO_WEIGHTS
parallel    : $PARALLEL  (NUM_GATEWAYS=$NUM_GATEWAYS, ports $PY4J_PORT–$PORT_MAX)
metrics port: $METRICS_PORT  (monitoring OFF; chi can bind — tranh dai Windows cam)
PID gains   : K_P=$K_P K_I=$K_I
stages      : archive=$DO_ARCHIVE nsga2=$DO_NSGA2 cmdp=$DO_CMDP_SWEEP ppo=$DO_PPO_FIXED heur/campaign=$DO_CAMPAIGN
EOF

# 0) ARCHIVE kết quả cũ trước khi ghi đè
if [[ "$DO_ARCHIVE" == "1" ]]; then
  hr "ARCHIVE ket qua cu"
  NOTE="auto-snapshot truoc job $RUN_LABEL-$TS" bash scripts/archive-results.sh "before-${RUN_LABEL}"
fi

# 1) Build (tự bỏ qua nếu đủ image; FORCE_BUILD=1 để ép sau khi đổi code)
have=1
docker image inspect eldas-cloudsim-java >/dev/null 2>&1 || have=0
docker image inspect eldas-rl-agent      >/dev/null 2>&1 || have=0
if [[ "${FORCE_BUILD:-0}" == "1" || "$have" == "0" ]]; then
  hr "BUILD image"
  docker compose build cloudsim-java rl-agent || { echo "build FAILED"; exit 1; }
else
  echo "[build] image da co ⇒ bo qua (FORCE_BUILD=1 de build lai sau khi doi code)."
fi

# 1b) N gateway độc lập trong 1 JVM
hr "START $PARALLEL gateway (monitoring OFF)"
docker compose up -d --force-recreate cloudsim-java || exit 1
echo "cho gateway san sang..."; sleep 14
GW=$(docker compose logs cloudsim-java 2>&1 | grep -c "GatewayServer listening")
echo "gateway dang chay: $GW (can $PARALLEL)"
[[ "$GW" -ge "$PARALLEL" ]] || { echo "THIEU gateway — kiem tra NUM_GATEWAYS/PY4J_PORT_MAX"; exit 1; }

# 2) Java sanity (fail-fast)
hr "Java ValidationRunner (sanity)"
docker run --rm -v "$PWD/data/alibaba-trace:/data/trace:ro" --entrypoint java \
  eldas-cloudsim-java -cp simulation.jar sim.ValidationRunner 2>&1 | grep -E "^PASS =|^FAIL =" \
  || echo "  [warn] khong doc duoc ket qua validation (bo qua, van chay)"

# ── Vòng theo scenario ───────────────────────────────────────────────────────
FAILED=""
for SC in $SCENARIOS; do
  TASKS=$(tasks_of "$SC")
  TS_CMDP=$(( EPISODES * TASKS ))
  TS_PPO=$(( PPO_EPISODES * TASKS ))
  BUDGETS="$(budgets_of "$SC")"
  hr "SCENARIO $SC  (CMDP ts/run=$EPISODES×$TASKS=$TS_CMDP · PPO ts/run=$TS_PPO · budgets=[$BUDGETS])"

  if [[ "$DO_NSGA2" == "1" ]]; then
    echo "[a] NSGA-II reference front..."
    Ro --entrypoint python rl-agent \
      src/eval/nsga2_baseline.py --scenario "$SC" --max-tasks 80 --pop-size 80 --n-gen 60 \
      || { echo "  [warn] NSGA-II loi (bo qua)"; FAILED="$FAILED $SC/nsga2"; }
  fi

  if [[ "$DO_CMDP_SWEEP" == "1" ]]; then
    echo "[b] CMDP-PID sweep: budgets=[$BUDGETS] × seeds=[$SEEDS] --parallel $PARALLEL..."
    R --entrypoint python rl-agent \
      src/eval/sweep_budget.py --scenario "$SC" \
      --budgets "$BUDGETS" --seeds "$SEEDS" \
      --total-timesteps "$TS_CMDP" --k-p "$K_P" --k-i "$K_I" --parallel "$PARALLEL" \
      || { echo "  [warn] CMDP sweep loi"; FAILED="$FAILED $SC/cmdp"; }
  fi

  if [[ "$DO_PPO_FIXED" == "1" ]]; then
    echo "[c] fixed-weight PPO front: weights=[$PPO_WEIGHTS] × seeds=[$SEEDS] --parallel $PARALLEL..."
    R --entrypoint python rl-agent \
      src/eval/ppo_fixed_sweep.py --scenario "$SC" \
      --weights "$PPO_WEIGHTS" --seeds "$SEEDS" \
      --total-timesteps "$TS_PPO" --parallel "$PARALLEL" \
      || { echo "  [warn] ppo-fixed loi"; FAILED="$FAILED $SC/ppo"; }
  fi

  if [[ "$DO_CAMPAIGN" == "1" ]]; then
    echo "[d] campaign aggregate (heuristic + gop tat ca)..."
    CAMP_FLAG=""; [[ "$DO_HEURISTICS" == "1" ]] && CAMP_FLAG="--run-baselines"
    R --entrypoint python rl-agent \
      src/eval/run_campaign.py --scenario "$SC" --seeds "$SEEDS" $CAMP_FLAG \
      || { echo "  [warn] campaign loi"; FAILED="$FAILED $SC/campaign"; }
  fi

  echo "→ $SC xong: data/results/campaign-${SC}/{table.md, pareto-${SC}.png, campaign_summary.json}"
done

# 3) Trả gateway về mặc định 1
docker compose up -d --force-recreate cloudsim-java >/dev/null 2>&1 || true

t_end=$(date +%s); mins=$(( (t_end - t0) / 60 ))
hr "OVERNIGHT DONE  (~${mins} phut)"
echo "log         : $LOG"
echo "output       : data/results/campaign-<SC>/{table.md, pareto-<SC>.png}"
echo "audit        : data/results-archive/  (snapshot truoc job + MANIFEST)"
if [[ -n "$FAILED" ]]; then
  echo "⚠ BUOC LOI (xem log):$FAILED"
else
  echo "✓ tat ca buoc chay tron."
fi
echo
echo "GHI CHU:"
echo " - ppo-w* gio gop thanh 1 family 'ppo-fixed' trong bang campaign ⇒ so FRONT-voi-FRONT vs CMDP-PID."
echo " - Front NON-MONOTONE ⇒ xem sweep-<SC>/sweep_summary.json; thuong do nhieu-seed > khoang budget,"
echo "   KHONG phai loi hoi tu (lambda*(d) van don dieu). Tang EPISODES/seed hoac chinh luoi (§14.3)."
