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
  # KHONG co budget mac dinh cho 'full': luoi 0.02-0.10 cu co truoc san deadline
  # (W1.5) va truoc phi drop (W3), ca hai doi THANG cua C_SLA. Phai do san kha thi
  # bang pilot (W6.1) roi truyen luoi tuong minh.
  DEF_BUDGETS=""; DEF_EPISODES=200; DEF_PPO=200; DEF_PAR=4
else
  echo "MODE phai la 'pilot' hoac 'full'. Nhan: '$MODE'" >&2; exit 2
fi

# Uu tien: tham so dong lenh > env > mac dinh theo MODE.
SCENARIOS="${SCENARIOS_ARG:-${SCENARIOS:-$DEF_SCENARIOS}}"
SCENARIOS="${SCENARIOS//,/ }"    # "LOW,BURST" -> "LOW BURST"
BUDGETS="${BUDGETS_ARG:-${BUDGETS:-$DEF_BUDGETS}}"
if [[ -z "$BUDGETS" ]]; then
  echo "Thieu BUDGETS." >&2
  echo "  Luoi cu 0.02..0.10 da VO NGHIA: san deadline tuyet doi (W1.5) va phi cho" >&2
  echo "  task bi drop (W3) deu doi THANG cua C_SLA, nen luoi do gio nam o cho tuy y" >&2
  echo "  tren truc moi - rat co the nam tron trong vung slack, noi moi budget hoi tu" >&2
  echo "  ve CUNG mot policy va 'Pareto front' la mot diem ve nam lan." >&2
  echo "  Chay pilot (W6.1) de do san SLA kha thi, roi:" >&2
  echo "    bash scripts/run-full-campaign.sh full HIGH 0.02,0.03,0.04,0.05,0.06" >&2
  exit 2
fi

SEEDS="${SEEDS:-$DEF_SEEDS}"
EPISODES="${EPISODES:-$DEF_EPISODES}"
PPO_EPISODES="${PPO_EPISODES:-$DEF_PPO}"
PARALLEL="${PARALLEL:-$DEF_PAR}"
# Gain cua vong doi ngau. Mac dinh giu nguyen duong cu; override khi can.
K_P="${K_P:-0.05}"
K_I="${K_I:-0.05}"

# ── WM-1 arm (W1-W5). ARM khong dat = duong LEGACY nhu truoc, khong doi gi. ──
#   ARM=homo|hetero  =>  doc trace sinh san o data/wm1/<ARM>/<SC>/seed<NN>.csv
#                        va ghi ket qua vao /data/results/wm1/<ARM>/  (PLAN §4.1)
#   Arm nam TRONG duong ghi ket qua => homo/hetero khong the ghi de nhau (risk R6,
#   loi C16 da tung xay ra). Guard trong eval/paths.py chan luon truong hop tro vao
#   /data/results (LEGACY) khi TRACE_PATTERN dang bat (risk R5).
ARM="${ARM:-}"
if [[ -n "$ARM" ]]; then
  case "$ARM" in
    homo|hetero) ;;
    *) echo "ARM phai la 'homo' hoac 'hetero'. Nhan: '$ARM'" >&2; exit 2 ;;
  esac
  MANIFEST="$PWD/data/wm1/$ARM/wm1-manifest.json"
  [[ -r "$MANIFEST" ]] || { echo "Thieu $MANIFEST - chay 'bash scripts/gen-workloads.sh' truoc" >&2; exit 2; }
  export TRACE_PATTERN="/data/wm1/${ARM}/{scenario}/seed{seed}.csv"
  # RESULTS_ROOT lets a re-run land somewhere new instead of on top of the old one.
  # It is not cosmetic: points.jsonl is APPENDED to and run_campaign.py aggregates
  # whatever it finds there, so re-running into a directory that still holds policies
  # trained on the OLD reward (§19) silently averages the two together and the table
  # looks entirely normal. New reward => new root.
  RESULTS="${RESULTS_ROOT:-/data/results/wm1}/${ARM}"
  # The arm decides the topology BOTH ways. Only setting it for hetero leaves an
  # inherited TOPOLOGY_CONFIG in place when a caller runs hetero and then homo in
  # the same shell (exactly what the W6 driver does), so the homo campaign would
  # run on the hetero cluster and write its results under .../homo/ - wrong numbers
  # filed under the right name, which is the hardest kind to notice later.
  if [[ "$ARM" == "hetero" ]]; then
    export TOPOLOGY_CONFIG="${TOPOLOGY_CONFIG:-/config/topology-hetero.json}"
  else
    export TOPOLOGY_CONFIG=""
  fi
  VALID_SC="LOW|HIGH|BURST|OVERLOAD|REPLAY"
else
  RESULTS="/data/results"
  VALID_SC="LOW|HIGH|BURST"
fi

# Fail-fast: chan scenario go sai (khong de job chay hang gio roi moi loi).
for SC in $SCENARIOS; do
  if [[ ! "$SC" =~ ^($VALID_SC)$ ]]; then
    echo "Scenario khong hop le: '$SC'. Cho phep: ${VALID_SC//|/ | }" >&2; exit 2
  fi
done

# Episode length moi scenario -> doi #episode ra #timestep de so dual-update
# (~ so episode) DONG NHAT giua cac scenario (dieu kien so sanh cong bang).
#
# Voi WM-1: DOC TU MANIFEST, khong hardcode. So task da doi HAI lan (W1.2 roi W3.1
# loc job khong host don nao chay noi); mot bang hardcode se lang le cho sai ngan
# sach timestep, va sai o day khong crash - no chi lam moi run ngan/dai hon y muon.
tasks_of() {
  local sc="$1" n=""
  if [[ -n "$ARM" ]]; then
    # Doc TRONG CONTAINER, khong dung python cua host. Script nay export
    # MSYS_NO_PATHCONV=1 (bat buoc cho tham so docker), nen Git Bash KHONG con
    # doi /d/... sang D:/... va python.exe cua Windows se khong tim thay file:
    # da gap that - tasks_of tra ve rong => TS=0 => moi run "thanh cong" tuc thi
    # voi 0 timestep. Trong container duong dan la Linux that su.
    n=$(docker compose run --rm --no-deps --entrypoint python rl-agent -c \
      "import json;m=json.load(open('/data/wm1/${ARM}/wm1-manifest.json'));\
print({t['scenario']:t['n_task'] for t in m['traces']}.get('${sc}',''))" 2>/dev/null \
      | tr -d '\r' | tail -1)
  else
    case "$sc" in
      LOW) n=1813 ;; HIGH) n=7255 ;; BURST) n=3569 ;; *) n=1813 ;;
    esac
  fi
  # KHONG BAO GIO tra ve rong: mot TASKS rong bien thanh TS=0 va ca job chay het
  # trong vai giay, sinh ra file ket qua trong nhin nhu that.
  if [[ ! "$n" =~ ^[0-9]+$ ]] || [[ "$n" -le 0 ]]; then
    echo "Khong doc duoc so task cho scenario '$sc' (arm='${ARM:-legacy}'): '$n'" >&2
    return 1
  fi
  echo "$n"
}

PY4J_PORT="${PY4J_PORT:-25333}"
PORT_MAX=$(( PY4J_PORT + PARALLEL - 1 ))
export NUM_GATEWAYS="$PARALLEL" PY4J_PORT_MAX="$PORT_MAX" MONITORING_ENABLED=false

R() { docker compose run --rm "$@"; }                 # gateway-facing
Ro() { docker compose run --rm --no-deps "$@"; }       # offline (không cần gateway)
hr() { printf '\n\033[1m── %s ──\033[0m\n' "$*"; }

t0=$(date +%s)
hr "SETUP  (MODE=$MODE · arm=${ARM:-legacy} · scenarios=[$SCENARIOS] · seeds=$SEEDS · budgets=$BUDGETS · episodes=$EPISODES · parallel=$PARALLEL)"
echo "results -> $RESULTS${TRACE_PATTERN:+   trace -> $TRACE_PATTERN}"

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
  # R13: rl-agent COPY src luc build va KHONG mount ./rl-agent/src, nen "image da co"
  # KHONG dong nghia "image co code hien tai". Bo qua build sau khi sua .py = ca job qua
  # dem chay code CU, im lang, ra ket qua trong nhu that. So mtime source vs image.
  IMG_TS=$(docker image inspect -f '{{.Created}}' eldas-rl-agent 2>/dev/null)
  IMG_EPOCH=$(docker image inspect -f '{{.Created}}' eldas-rl-agent 2>/dev/null \
    | python3 -c "import sys,datetime;print(int(datetime.datetime.fromisoformat(sys.stdin.read().strip().replace('Z','+00:00')).timestamp()))" 2>/dev/null || echo 0)
  NEWER=$(find rl-agent/src rl-agent/requirements.txt cloudsim-java/src -type f \
            -newermt "@${IMG_EPOCH}" 2>/dev/null | head -1)
  if [[ -n "$NEWER" ]]; then
    echo "Source MOI HON image ($IMG_TS) — vd: $NEWER"
    echo "  => build lai, neu khong ca job se chay code cu (R13)."
    docker compose build cloudsim-java rl-agent || { echo "build failed"; exit 1; }
  else
    echo "Image da co va khong cu hon source ⇒ bỏ qua build."
  fi
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
VR_OUT=$(docker run --rm -v "$PWD/data/alibaba-trace:/data/trace:ro" \
  -v "$PWD/data/wm1:/data/wm1:ro" -v "$PWD/config:/config:ro" --entrypoint java \
  eldas-cloudsim-java -cp simulation.jar sim.ValidationRunner 2>&1)
echo "$VR_OUT" | grep -E "^PASS =|^FAIL =|^SKIP ="
# FAIL o day = dung han. Mot job qua dem tren code da hong chi sinh ra so lieu sai,
# va sai kieu "chay het, co file, so nhin hop ly" moi la kieu ton kem nhat.
JFAIL=$(echo "$VR_OUT" | sed -n "s/^FAIL = \([0-9]*\).*/\1/p")
if [[ "${JFAIL:-1}" != "0" ]]; then
  echo "Java validation FAIL=${JFAIL:-?} - DUNG truoc khi ton hang gio." >&2
  echo "$VR_OUT" | grep -E "^\[FAIL\]" >&2
  exit 1
fi

# ── Vòng theo scenario ──────────────────────────────────────────────────────
for SC in $SCENARIOS; do
  TASKS=$(tasks_of "$SC") || exit 1
  TS=$(( EPISODES * TASKS ))
  PPO_TS=$(( PPO_EPISODES * TASKS ))
  # Luu y #23: thang cua tin hieu rang buoc phai DONG BANG. Neu khong, J = mean(c)/std(c)
  # do HINH DANG phan bo chi phi chu khong do MUC, nen ngan sach d khong lai duoc gi:
  # do duoc o W6.3 la ca hai dot campaign deu cho 5 budget -> 5 diem gan trung, va lambda
  # chi cong don mot sai lech khong doi. Dong bang sau 5 episode: du de uoc luong on dinh
  # ma chua kip bi policy lam lech.
  COST_FREEZE="${COST_FREEZE:-$(( 5 * TASKS ))}"
  hr "SCENARIO $SC  (timesteps/run = $EPISODES ep × $TASKS task = $TS)"

  # 1) NSGA-II reference front (offline, không cần gateway)
  echo "[1/4] NSGA-II reference front..."
  Ro --entrypoint python rl-agent \
    src/eval/nsga2_baseline.py --scenario "$SC" --max-tasks 80 --pop-size 80 --n-gen 60 \
    --output "$RESULTS" \
    || echo "  [warn] NSGA-II loi, bo qua - campaign van chay khong co no"

  # 2) fixed-weight PPO (ppo-min, seed 42) — baseline Phase-1
  echo "[2/4] ppo-min (fixed-weight PPO, $PPO_EPISODES ep)..."
  R --entrypoint python rl-agent \
    src/train_min.py --scenario "$SC" --seed 42 --total-timesteps "$PPO_TS" \
    --model-out "/data/models/ppo-min-${SC}.zip" \
    --eval-out "${RESULTS}/baseline-${SC}/ppo-min" \
    || echo "  ⚠ ppo-min lỗi (bỏ qua)"

  # 2b) fixed-weight PPO FRONT (ppo-w*) — chi chay khi PPO_FIXED_WEIGHTS duoc dat.
  #     Day la doi thu that su cua CMDP-PID trong RQ4: "chinh tay trong so" vs "chinh
  #     ngan sach co nguyen tac". Mot diem ppo-min don le KHONG tao thanh front, nen
  #     ho "ppo-fixed" ma hv_bootstrap/eaf_compare so sanh se RONG neu bo qua buoc nay
  #     - va ca hai cong cu chi im lang bo qua cap so sanh do.
  if [[ -n "${PPO_FIXED_WEIGHTS:-}" ]]; then
    echo "[2b/4] ppo-fixed front: weights=[$PPO_FIXED_WEIGHTS] × seeds=[$SEEDS]..."
    R --entrypoint python rl-agent \
      src/eval/ppo_fixed_sweep.py --scenario "$SC" \
      --weights "$PPO_FIXED_WEIGHTS" --seeds "$SEEDS" \
      --total-timesteps "$TS" --parallel "$PARALLEL" \
      --output "$RESULTS" --skip-existing \
      || echo "  [warn] ppo-fixed sweep loi - campaign van chay, nhung cap cmdp-pid vs ppo-fixed se trong"
  fi

  # 3) CMDP-PID budget sweep — song song theo RUN (đòn bẩy C9)
  echo "[3/4] CMDP sweep: budgets=[$BUDGETS] × seeds=[$SEEDS], --parallel $PARALLEL..."
  R --entrypoint python rl-agent \
    src/eval/sweep_budget.py --scenario "$SC" \
    --budgets "$BUDGETS" --seeds "$SEEDS" \
    --total-timesteps "$TS" --k-p "$K_P" --k-i "$K_I" \
    --cost-freeze-after "$COST_FREEZE" --parallel "$PARALLEL" \
    --output "$RESULTS" \
    || echo "  [warn] sweep loi/non-monotone - xem verdict o tren, co the can train lau hon. C1"

  # 4) Campaign: heuristics + gộp sweep + ppo-min + NSGA-II → bảng + figure
  echo "[4/4] campaign aggregate..."
  R --entrypoint python rl-agent \
    src/eval/run_campaign.py --scenario "$SC" --seeds "$SEEDS" --run-baselines \
    --results "$RESULTS" \
    || echo "  ⚠ campaign lỗi"

  echo "→ $SC xong: ${RESULTS}/campaign-${SC}/{table.md, pareto-${SC}.png, campaign_summary.json}"
done

# ── Kết thúc ────────────────────────────────────────────────────────────────
docker compose up -d --force-recreate cloudsim-java >/dev/null 2>&1   # tra ve 1 gateway mac dinh
t_end=$(date +%s)
mins=$(( (t_end - t0) / 60 ))
hr "HOAN TAT sau ~${mins} phut"
echo "Output: ${RESULTS}/campaign-<SC>/table.md + pareto-<SC>.png cho: $SCENARIOS"
echo
echo "GHI CHU (day la nhac chung, KHONG phai ket qua):"
echo " - Verdict MONOTONE/NON-MONOTONE THAT cua tung scenario nam o khoi 'G2.5 sweep'"
echo "   phia tren (moi scenario in rieng), hoac trong sweep-<SC>/sweep_summary.json."
echo " - NON-MONOTONE thuong do NHIEU SEED > khoang cach budget (front nong, nhat la LOW),"
echo "   KHONG phai loi hoi tu: lambda*(d) van don dieu. Xem Tracking_detail.md muc 14.3."
echo " - Hinh Pareto + HV/IGD+ dung bao NON-DOMINATED, khong phu thuoc check don dieu nay."
