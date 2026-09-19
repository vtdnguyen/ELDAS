#!/usr/bin/env bash
# =============================================================================
# ELDAS — chạy nốt TOÀN BỘ phần máy tính của đồ án. Một lệnh, chạy được qua đêm.
#
#   bash scripts/finish-data.sh
#
# Script này làm mọi việc còn lại NGOẠI TRỪ viết báo cáo bằng tay:
#
#   [prep]  docker sẵn sàng · build lại nếu source mới hơn image · Java validator
#           · kiểm tương đương codec  (hai cái sau FAIL thì DỪNG HẲN, vì qua đó thì
#           số liệu là SAI chứ không phải THIẾU)
#   [gen]   sinh tệp dữ liệu tải cho hạt giống mới — ở WM-1 hạt giống quyết định CẢ
#           dữ liệu tải, nên thêm hạt giống là phải sinh thêm tệp trước
#   [seeds] tăng homo/LOW lên số hạt giống yêu cầu — thí nghiệm DUY NHẤT còn
#           có thể đổi kết luận trung tâm (§23.4). Nhiễu giữa hạt (±77 kWh) đang lớn
#           hơn khoảng cách giữa các điểm ngân sách (≈24 kWh); 15 hạt thu hẹp khoảng
#           tin cậy ~1,7 lần
#   [traj]  4 lần chạy homo/LOW kèm --trajectory-csv  → dữ liệu cho hình H7
#           (sweep_budget.py KHÔNG lưu quỹ đạo lambda, nên bắt buộc chạy riêng)
#   [agg]   dựng lại bảng + bootstrap + kiểm định hoán vị cho ô đã thêm hạt giống
#   [w65]   W6.5 — LEGACY vs WM-1, chỉ trên đại lượng KHÔNG ĐƠN VỊ
#   [figs]  sinh H6/H7/H8/H9 dạng PDF rồi chép vào assets_v2/report/figures/
#   [delta] in ra NHỮNG SỐ ĐÃ ĐỔI, để biết chính xác chỗ nào trong Chương 9 phải sửa
#
# RESUMABLE. Mỗi việc xong ghi một marker dưới $RESULTS_ROOT/.finish-state/.
# Chạy lại thì chỉ làm phần còn thiếu. FORCE=1 bỏ qua marker.
#
# ⚠️ THÊM HẠT GIỐNG LÀ GHI THÊM VÀO points.jsonl CỦA CÙNG MỘT CHIẾN DỊCH.
#    Điều đó ĐÚNG ở đây — cùng lưới ngân sách, cùng mã, cùng tham số, chỉ thêm lần
#    lặp. Nhưng hộp chuẩn hoá ideal/nadir sẽ tính lại, nên **thể tích chi phối của ô
#    homo/LOW sẽ đổi số**. Giai đoạn [delta] in ra trước/sau đúng vì lý do này
#    (Lưu ý #26). Muốn giữ nguyên số cũ thì đặt STAGES=traj,w65,figs.
#
# Knob:  SEEDS_TOTAL  ARM  SCENARIO  RESULTS_ROOT  PARALLEL  EPISODES  STAGES  FORCE
# =============================================================================
set -uo pipefail
export MSYS_NO_PATHCONV=1          # Git Bash: đừng dịch /data/... (D2)

RESULTS_ROOT="${RESULTS_ROOT:-/data/results/wm1-v3}"
RESULTS_5SC="${RESULTS_5SC:-/data/results/wm1-v2}"     # gốc có đủ 5 kịch bản (H8)
ARM="${ARM:-homo}"
SCENARIO="${SCENARIO:-LOW}"
SEEDS_BASE="${SEEDS_BASE:-42,43,44,45,46}"             # đã chạy rồi
SEEDS_NEW="${SEEDS_NEW:-47,48,49,50,51,52,53,54,55,56}"
EPISODES="${EPISODES:-300}"
PARALLEL="${PARALLEL:-4}"
K_P="${K_P:-0.10}"
K_I="${K_I:-0.30}"
PPO_FIXED_WEIGHTS="${PPO_FIXED_WEIGHTS:-0.3,0.5,0.7}"
STAGES="${1:-${STAGES:-all}}"
FORCE="${FORCE:-0}"

SEEDS_ALL="${SEEDS_BASE},${SEEDS_NEW}"

PY4J_PORT="${PY4J_PORT:-25333}"
export NUM_GATEWAYS="$PARALLEL" PY4J_PORT_MAX=$(( PY4J_PORT + PARALLEL - 1 ))
export MONITORING_ENABLED=false                        # NUM_GATEWAYS>1 ⇒ gauge trộn (C11)
export TRACE_PATTERN="/data/wm1/${ARM}/{scenario}/seed{seed}.csv"
if [[ "$ARM" == "hetero" ]]; then
  export TOPOLOGY_CONFIG="/config/topology-hetero.json"
else
  export TOPOLOGY_CONFIG=""
fi

HOST_ROOT="${RESULTS_ROOT#/}"
STATE_DIR="$HOST_ROOT/.finish-state"
LOG_DIR="$HOST_ROOT/logs"
FIG_OUT="$RESULTS_ROOT/figures"
TRAJ_DIR="$RESULTS_ROOT/traj"
REPORT_FIGS="assets_v2/report/figures"
mkdir -p "$STATE_DIR" "$LOG_DIR" "$REPORT_FIGS"

RUN_ID="$(date +%Y%m%d-%H%M%S)"
LOG="$LOG_DIR/finish-$RUN_ID.log"
exec > >(tee -a "$LOG") 2>&1

t0=$(date +%s)
hr()  { printf '\n\033[1m== %s ==\033[0m\n' "$*"; }
sub() { printf '\n\033[1m-- %s --\033[0m\n' "$*"; }
say() { printf '   %s\n' "$*"; }
elapsed() { local s=$(( $(date +%s) - t0 )); printf '%dh%02dm' $((s/3600)) $(((s%3600)/60)); }

FAILURES=()
fail() { FAILURES+=("$*"); printf '\033[1;31m   [FAIL] %s\033[0m\n' "$*"; }

is_done()   { [[ "$FORCE" != "1" && -f "$STATE_DIR/$1.done" ]]; }
mark_done() { date -Iseconds > "$STATE_DIR/$1.done"; }

# Marker chỉ là một LỜI KHAI. Khi có bằng chứng rẻ để đối chiếu thì phải đối chiếu:
# lần chạy đầu để lại `seeds.done` và `agg.done` trong khi KHÔNG một điểm nào được ghi
# thêm, nên mọi lần chạy sau đều "xong" trong không giây. Sửa cách đánh dấu chỉ chặn
# marker sai MỚI; nó không dọn được marker sai CŨ, và cũng không bắt được trường hợp
# ai đó xoá dữ liệu mà marker vẫn còn. Kiểm bằng chứng thì bắt được cả ba.
seeds_present() {   # in ra số hạt giống thực có trong points.jsonl của sweep
  Ro --entrypoint python rl-agent -c "
import json,pathlib
p=pathlib.Path('$RESULTS_ROOT/$ARM/sweep-$SCENARIO/points.jsonl')
print(len({json.loads(l)['seed'] for l in p.read_text().splitlines() if l.strip()})
      if p.is_file() else 0)" 2>/dev/null | tr -d '\r' | tail -1
}

# Marker + bằng chứng: chỉ coi là xong khi points.jsonl có đủ số hạt giống yêu cầu.
seeds_done() {
  is_done "seeds" || return 1
  local have want
  have=$(seeds_present)
  want=$(echo "$SEEDS_ALL" | tr ',' '\n' | grep -c .)
  if [[ "${have:-0}" -ge "$want" ]]; then
    return 0
  fi
  say "marker 'seeds' có nhưng dữ liệu chỉ $have/$want hạt giống ⇒ BỎ marker, chạy lại"
  rm -f "$STATE_DIR/seeds.done" "$STATE_DIR/agg.done"
  return 1
}

# Đánh dấu xong CHỈ KHI giai đoạn không ghi nhận lỗi nào. Bản đầu gọi mark_done vô
# điều kiện ngay sau fail, nên một lần chạy hỏng vẫn để lại marker và lần chạy sau
# **bỏ qua** đúng phần đã hỏng — hỏng một lần thành hỏng vĩnh viễn, im lặng.
_marks=0
stage_begin() { _marks=${#FAILURES[@]}; }
stage_end()   {   # stage_end <tên>
  if [[ ${#FAILURES[@]} -eq $_marks ]]; then
    mark_done "$1"
  else
    say "giai đoạn '$1' có lỗi ⇒ KHÔNG đánh dấu xong (chạy lại sẽ làm lại phần này)"
  fi
}

R()  { docker compose run --rm -T "$@" < /dev/null; }             # cần gateway
Ro() { docker compose run --rm -T --no-deps "$@" < /dev/null; }   # offline

# Docker Desktop trên Windows thỉnh thoảng rụng engine pipe. Không canh thì cả đêm
# hỏng trong chín mươi giây; chờ nó quay lại thay vì bỏ cuộc.
docker_ready() {
  local tries="${1:-60}" i=0
  while ! docker info >/dev/null 2>&1; do
    i=$(( i + 1 ))
    if (( i > tries )); then
      fail "Docker không phản hồi sau $(( tries * 10 ))s — dừng. Chạy lại lệnh cũ để tiếp tục."
      return 1
    fi
    [[ $i -eq 1 ]] && say "Docker chưa sẵn sàng — đang chờ..."
    sleep 10
  done
  (( i > 0 )) && say "Docker đã trở lại sau $(( i * 10 ))s"
  return 0
}

gateway_up() {
  say "gateway: arm=$ARM topology='${TOPOLOGY_CONFIG:-đồng nhất}' số cổng=$PARALLEL"
  docker compose up -d --force-recreate cloudsim-java >/dev/null 2>&1 || return 1
  sleep 14
  local gw
  gw=$(docker compose logs cloudsim-java 2>&1 | grep -c "GatewayServer listening")
  say "gateway đang lắng nghe: $gw (cần $PARALLEL)"
  [[ "$gw" -ge "$PARALLEL" ]]
}

# Số tác vụ đọc TỪ MANIFEST, trong container. Hardcode thì sai lặng lẽ khi trace đổi,
# và một TASKS rỗng biến thành 0 timestep — job "thành công" trong vài giây.
tasks_of() {
  local n
  n=$(Ro --entrypoint python rl-agent -c \
    "import json;m=json.load(open('/data/wm1/${ARM}/wm1-manifest.json'));\
print({t['scenario']:t['n_task'] for t in m['traces']}.get('${SCENARIO}',''))" \
    2>/dev/null | tr -d '\r' | tail -1)
  [[ "$n" =~ ^[0-9]+$ && "$n" -gt 0 ]] || { echo "khong doc duoc so task: '$n'" >&2; return 1; }
  echo "$n"
}

budgets_of() {
  Ro --entrypoint python rl-agent src/eval/pilot_floor.py --print-grid \
    "$RESULTS_ROOT/$ARM/pilot-$SCENARIO/pilot_floor.json" 2>/dev/null \
    | tr -d '\r' | tail -1
}

# Chụp lại các số sẽ đổi, để giai đoạn [delta] so được trước/sau.
snapshot() {   # snapshot <nhãn>
  local f="$STATE_DIR/snapshot-$1.json"
  Ro --entrypoint python rl-agent -c "
import json,pathlib
p=pathlib.Path('$RESULTS_ROOT/$ARM/campaign-$SCENARIO/campaign_summary.json')
out={}
if p.is_file():
    b=json.loads(p.read_text())
    out={'ideal':b.get('ideal'),'nadir':b.get('nadir'),
         'hv':{m:v.get('hypervolume') for m,v in b.get('methods',{}).items()},
         'igd':{m:v.get('igd_plus') for m,v in b.get('methods',{}).items()},
         'n':{m:v.get('n_points') for m,v in b.get('methods',{}).items()}}
print(json.dumps(out))" 2>/dev/null | tr -d '\r' | tail -1 > "$f"
  [[ -s "$f" ]] || echo '{}' > "$f"
}

# ════════════════════════════════════════════════════════════════════════════
stage_prep() {
  hr "PREP  (run $RUN_ID · log $LOG)"
  docker_ready || return 1

  sub "image"
  local rebuild=0
  docker image inspect eldas-cloudsim-java >/dev/null 2>&1 || rebuild=1
  docker image inspect eldas-rl-agent      >/dev/null 2>&1 || rebuild=1
  if [[ "$rebuild" == "0" ]]; then
    # rl-agent COPY src lúc build và KHÔNG mount — "image đã có" KHÔNG đồng nghĩa
    # "image có mã hiện tại". Bỏ qua build sau khi sửa .py = cả đêm chạy mã cũ (R13).
    local img_epoch newer
    img_epoch=$(docker image inspect -f '{{.Created}}' eldas-rl-agent 2>/dev/null \
      | python3 -c "import sys,datetime;print(int(datetime.datetime.fromisoformat(sys.stdin.read().strip().replace('Z','+00:00')).timestamp()))" 2>/dev/null || echo 0)
    newer=$(find rl-agent/src rl-agent/requirements.txt cloudsim-java/src -type f \
              -newermt "@${img_epoch}" 2>/dev/null | head -1)
    [[ -n "$newer" ]] && { say "source mới hơn image (vd $newer) -> build lại"; rebuild=1; }
  fi
  if [[ "$rebuild" == "1" ]]; then
    docker compose build cloudsim-java rl-agent || { fail "docker build"; return 1; }
  else
    say "image đã cập nhật"
  fi

  sub "Java ValidationRunner"
  local vr jfail
  vr=$(docker run --rm -v "$PWD/data/alibaba-trace:/data/trace:ro" \
        -v "$PWD/data/wm1:/data/wm1:ro" -v "$PWD/config:/config:ro" --entrypoint java \
        eldas-cloudsim-java -cp simulation.jar sim.ValidationRunner 2>&1)
  echo "$vr" | grep -E "^PASS =|^FAIL =|^SKIP ="
  jfail=$(echo "$vr" | sed -n "s/^FAIL = \([0-9]*\).*/\1/p")
  if [[ "${jfail:-1}" != "0" ]]; then
    echo "$vr" | grep -E "^\[FAIL\]"
    fail "Java validation FAIL=${jfail:-?} — DỪNG HẲN (mã hỏng sinh ra số SAI, không phải số thiếu)"
    return 1
  fi

  sub "kiểm tương đương codec (SYS.2)"
  if is_done "parity"; then
    say "đã xong"
  else
    gateway_up || { fail "gateway cho parity"; return 1; }
    # OVERLOAD chứ không LOW: thay đổi codec cần kiểm là droppedTasks, mà ở kịch bản
    # không có task nào bị loại thì phép kiểm pass một cách rỗng (R4).
    if R --entrypoint python rl-agent src/perf/verify_packed_parity.py \
         --scenario OVERLOAD --steps 200; then
      mark_done "parity"
    else
      fail "parity — DỪNG HẲN (mọi số phía sau đọc quan sát qua codec này)"
      return 1
    fi
  fi
  return 0
}

# ════════════════════════════════════════════════════════════════════════════
# GEN — sinh tệp dữ liệu tải cho hạt giống mới.
#
# Ở WM-1, **hạt giống quyết định cả dữ liệu tải**, không chỉ khởi tạo mạng. Thêm hạt
# giống nghĩa là phải sinh thêm tệp; không có tệp thì Java ném
# "TRACE_PATTERN resolved to .../seed47.csv which is not readable" và cả giai đoạn
# huấn luyện chết sau vài phút. Lần chạy đầu tiên vấp đúng chỗ này.
#
# Sinh **TOÀN BỘ kịch bản**, không riêng kịch bản đang chạy: `generate_arm` ghi đè
# `wm1-manifest.json` bằng đúng những gì nó vừa sinh, nên sinh riêng LOW sẽ xoá mọi
# kịch bản khác khỏi manifest — và `tasks_of` khi đó trả rỗng, biến thành 0 timestep,
# tức một job "thành công" trong vài giây với đầy đủ tệp kết quả trông như thật.
# ════════════════════════════════════════════════════════════════════════════
stage_gen() {
  hr "GEN — dữ liệu tải cho hạt giống mới  (đã trôi $(elapsed))"
  stage_begin
  if is_done "gen"; then say "đã xong"; return 0; fi
  docker_ready || return 1

  local missing=()
  for s in ${SEEDS_ALL//,/ }; do
    [[ -r "data/wm1/$ARM/$SCENARIO/seed${s}.csv" ]] || missing+=("$s")
  done
  if [[ ${#missing[@]} -eq 0 ]]; then
    say "đủ tệp cho mọi hạt giống — không cần sinh"
    stage_end "gen"; return 0
  fi
  say "thiếu tệp cho hạt giống: ${missing[*]}"

  # Sinh lại là TẤT ĐỊNH (mọi thứ đều theo hạt giống), nên tệp của hạt giống cũ phải
  # ra y hệt. Nếu không thì mọi kết quả đã báo cáo đứng trên dữ liệu khác ⇒ dừng hẳn.
  local before="$STATE_DIR/manifest-sha-before.json"
  Ro --entrypoint python rl-agent -c "
import json,pathlib
p=pathlib.Path('/data/wm1/$ARM/wm1-manifest.json')
m=json.loads(p.read_text()) if p.is_file() else {}
print(json.dumps({t['path']: t['sha256'] for t in m.get('traces',[])}))" \
    2>/dev/null | tr -d '\r' | tail -1 > "$before"
  [[ -s "$before" ]] || echo '{}' > "$before"

  sub "sinh $ARM × mọi kịch bản × hạt giống $SEEDS_ALL"
  ARMS="$ARM" SEEDS="$SEEDS_ALL" bash scripts/gen-workloads.sh \
    || { fail "gen-workloads"; stage_end "gen"; return 1; }

  sub "kiểm tệp cũ KHÔNG đổi"
  local verdict
  verdict=$(Ro --entrypoint python rl-agent -c "
import json,pathlib
old=json.loads(pathlib.Path('/data/results/${STATE_DIR#*results/}/manifest-sha-before.json').read_text())
new={t['path']: t['sha256'] for t in json.loads(
     pathlib.Path('/data/wm1/$ARM/wm1-manifest.json').read_text())['traces']}
bad=[k for k,v in old.items() if k in new and new[k]!=v]
print('KHACNHAU:'+','.join(bad) if bad else f'OK {len(old)} tep cu trung khop')" \
    2>/dev/null | tr -d '\r' | tail -1)
  say "$verdict"
  if [[ "$verdict" == KHACNHAU:* ]]; then
    fail "sinh lại làm ĐỔI tệp dữ liệu cũ ($verdict) — DỪNG HẲN, mọi kết quả đã báo cáo sẽ không còn đúng nguồn"
    stage_end "gen"; return 1
  fi
  stage_end "gen"
  return 0
}

# ════════════════════════════════════════════════════════════════════════════
stage_seeds() {
  hr "SEEDS — tăng $ARM/$SCENARIO lên $(( $(echo "$SEEDS_ALL" | tr ',' '\n' | wc -l) )) hạt giống  (đã trôi $(elapsed))"
  stage_begin
  docker_ready || return 1
  if seeds_done; then say "đã xong (đã kiểm: points.jsonl đủ hạt giống)"; return 0; fi

  # Chặn trước: thiếu một tệp thôi là cả giai đoạn chết giữa chừng (job dài chết ở
  # phút thứ ba = mất cả đêm). Rẻ hơn nhiều so với phát hiện sau ba tiếng.
  local nomiss=1
  for s in ${SEEDS_ALL//,/ }; do
    if [[ ! -r "data/wm1/$ARM/$SCENARIO/seed${s}.csv" ]]; then
      say "THIẾU data/wm1/$ARM/$SCENARIO/seed${s}.csv"
      nomiss=0
    fi
  done
  if [[ "$nomiss" != "1" ]]; then
    fail "thiếu tệp dữ liệu tải — chạy giai đoạn 'gen' trước (STAGES=gen)"
    stage_end "seeds"; return 1
  fi

  local tasks ts budgets freeze
  tasks=$(tasks_of) || { fail "không đọc được số tác vụ"; return 1; }
  ts=$(( EPISODES * tasks ))
  freeze=$(( 5 * tasks ))
  budgets=$(budgets_of)
  if [[ -z "$budgets" || "$budgets" != *,* ]]; then
    fail "không suy được lưới ngân sách từ pilot ('${budgets:-rỗng}') — chạy w6-overnight.sh giai đoạn pilot trước"
    return 1
  fi
  say "tác vụ/vòng=$tasks · timestep/run=$ts · đóng băng thang sau $freeze · lưới=$budgets"

  snapshot before
  gateway_up || { fail "gateway"; return 1; }

  sub "quét ngân sách cho hạt giống MỚI: $SEEDS_NEW"
  # Lưu ý #23: BẮT BUỘC --cost-freeze-after, nếu không J đo HÌNH DẠNG phân bố chi phí
  # chứ không đo MỨC, và vòng đối ngẫu chạy hở trong khi log trông hoàn toàn bình thường.
  R --entrypoint python rl-agent src/eval/sweep_budget.py \
      --scenario "$SCENARIO" --budgets "$budgets" --seeds "$SEEDS_NEW" \
      --total-timesteps "$ts" --k-p "$K_P" --k-i "$K_I" \
      --cost-freeze-after "$freeze" --parallel "$PARALLEL" \
      --output "$RESULTS_ROOT/$ARM" \
    || fail "sweep hạt giống mới"

  sub "PPO trọng số cố định cho hạt giống MỚI"
  R --entrypoint python rl-agent src/eval/ppo_fixed_sweep.py \
      --scenario "$SCENARIO" --weights "$PPO_FIXED_WEIGHTS" --seeds "$SEEDS_NEW" \
      --total-timesteps "$ts" --parallel "$PARALLEL" \
      --output "$RESULTS_ROOT/$ARM" --skip-existing \
    || fail "ppo-fixed hạt giống mới"

  stage_end "seeds"
  say "xong lúc $(elapsed)"
  return 0
}

# ════════════════════════════════════════════════════════════════════════════
stage_traj() {
  hr "TRAJ — quỹ đạo lambda cho hình H7  (đã trôi $(elapsed))"
  stage_begin
  if is_done "traj"; then say "đã xong"; return 0; fi
  docker_ready || return 1

  local tasks ts freeze
  tasks=$(tasks_of) || return 1
  ts=$(( EPISODES * tasks ))
  freeze=$(( 5 * tasks ))
  budgets=$(budgets_of)
  [[ -n "$budgets" ]] || { fail "thiếu lưới ngân sách cho traj"; stage_end "traj"; return 1; }

  gateway_up || { fail "gateway"; return 1; }
  mkdir -p "${TRAJ_DIR#/}"

  # Một hạt giống là đủ: H7 minh hoạt HÌNH DẠNG quỹ đạo lambda theo ngân sách, không
  # phải phân phối của nó. Chạy song song, mỗi run một cổng gateway riêng.
  local i=0 pids=() names=()
  for d in ${budgets//,/ }; do
    local port=$(( PY4J_PORT + (i % PARALLEL) ))
    say "d=$d (cổng $port)"
    docker compose run --rm -T -e GATEWAY_PORT="$port" --entrypoint python rl-agent \
      src/train_cmdp.py --scenario "$SCENARIO" --seed 42 --sla-budget "$d" \
      --total-timesteps "$ts" --k-p "$K_P" --k-i "$K_I" \
      --cost-freeze-after "$freeze" \
      --model-out "/data/models/traj-${SCENARIO}-d${d}.zip" \
      --trajectory-csv "$TRAJ_DIR/traj-d${d}.csv" \
      --eval-out "$RESULTS_ROOT/$ARM/traj-eval" \
      > "$LOG_DIR/traj-d${d}.log" 2>&1 < /dev/null &
    pids+=($!); names+=("$d")
    i=$(( i + 1 ))
    if (( i % PARALLEL == 0 )); then
      local k=0
      for pid in "${pids[@]}"; do
        wait "$pid" || fail "traj d=${names[$k]} (xem $LOG_DIR/traj-d${names[$k]}.log)"
        k=$(( k + 1 ))
      done
      pids=(); names=()
    fi
  done
  local k=0
  for pid in "${pids[@]:-}"; do
    [[ -z "$pid" ]] && continue
    wait "$pid" || fail "traj d=${names[$k]}"
    k=$(( k + 1 ))
  done

  local n want
  n=$(ls "${TRAJ_DIR#/}"/*.csv 2>/dev/null | wc -l)
  want=$(echo "$budgets" | tr ',' '
' | wc -l)
  say "đã sinh $n/$want tệp quỹ đạo"
  # Thiếu một ngân sách thì hình H7 vẫn vẽ được nhưng khuyết một đường — đánh dấu xong
  # lúc đó là biến một hình thiếu thành một hình thiếu VĨNH VIỄN.
  [[ "$n" -lt "$want" ]] && fail "traj: chỉ có $n/$want tệp quỹ đạo"
  stage_end "traj"
  return 0
}

# ════════════════════════════════════════════════════════════════════════════
stage_agg() {
  hr "AGG — dựng lại bảng + thống kê cho $ARM/$SCENARIO  (đã trôi $(elapsed))"
  stage_begin
  docker_ready || return 1
  if is_done "agg" && seeds_done; then say "đã xong"; return 0; fi
  gateway_up || { fail "gateway"; return 1; }

  # --run-baselines: 5 thuật toán kinh nghiệm phải chạy cho CẢ hạt giống mới, nếu không
  # bảng sẽ trộn n=15 của phương pháp học với n=5 của đường cơ sở.
  R --entrypoint python rl-agent src/eval/run_campaign.py \
      --scenario "$SCENARIO" --seeds "$SEEDS_ALL" --run-baselines \
      --results "$RESULTS_ROOT/$ARM" \
    || fail "run_campaign"

  Ro --entrypoint python rl-agent src/eval/hv_bootstrap.py \
      --scenario "$SCENARIO" --results "$RESULTS_ROOT/$ARM" || fail "hv_bootstrap"
  # eaf_compare trả mã khác 0 khi phát hiện mâu thuẫn với bootstrap — đó là tín hiệu
  # có chủ đích, không phải hỏng, nên chỉ ghi nhận.
  Ro --entrypoint python rl-agent src/eval/eaf_compare.py \
      --scenario "$SCENARIO" --results "$RESULTS_ROOT/$ARM" \
    || say "eaf_compare báo mâu thuẫn/khác biệt — đọc kỹ phần in ra ở trên"

  snapshot after
  stage_end "agg"
  return 0
}

# ════════════════════════════════════════════════════════════════════════════
stage_w65() {
  hr "W6.5 — LEGACY vs WM-1  (đã trôi $(elapsed))"
  stage_begin
  if is_done "w65"; then say "đã xong"; return 0; fi
  docker_ready || return 1
  Ro --entrypoint python rl-agent src/eval/legacy_vs_wm1.py \
      --legacy-results /data/results --wm1-results "$RESULTS_ROOT" --arm "$ARM" \
      --out "$RESULTS_ROOT/w65" \
    && mark_done "w65" || fail "legacy_vs_wm1"
  return 0
}

# ════════════════════════════════════════════════════════════════════════════
stage_figs() {
  hr "FIGS — sinh H6/H7/H8/H9  (đã trôi $(elapsed))"
  stage_begin
  docker_ready || return 1
  Ro --entrypoint python rl-agent src/eval/report_figures.py \
      --results "$RESULTS_ROOT" --results-5sc "$RESULTS_5SC" \
      --traj-dir "$TRAJ_DIR" --out "$FIG_OUT" \
    || { fail "report_figures"; return 1; }

  sub "chép PDF sang $REPORT_FIGS"
  local n=0
  for f in "${FIG_OUT#/}"/*.pdf; do
    [[ -e "$f" ]] || continue
    cp "$f" "$REPORT_FIGS/" && n=$(( n + 1 ))
  done
  say "đã chép $n hình"
  [[ "$n" -eq 0 ]] && fail "figs: không sinh được hình nào"
  stage_end "figs"
  return 0
}

# ════════════════════════════════════════════════════════════════════════════
stage_delta() {
  hr "DELTA — những số ĐÃ ĐỔI, cần sửa tay trong Chương 9"
  local b="$STATE_DIR/snapshot-before.json" a="$STATE_DIR/snapshot-after.json"
  if [[ ! -s "$b" || ! -s "$a" ]]; then
    say "chưa có đủ hai bản chụp (giai đoạn seeds/agg chưa chạy) — bỏ qua"
    return 0
  fi
  Ro --entrypoint python rl-agent -c "
import json,pathlib
b=json.loads(pathlib.Path('/data/results/${STATE_DIR#*results/}/snapshot-before.json').read_text() or '{}')
a=json.loads(pathlib.Path('/data/results/${STATE_DIR#*results/}/snapshot-after.json').read_text() or '{}')
if not a:
    print('  (khong co ban chup sau)'); raise SystemExit
print(f\"  {'phuong phap':<14}{'HV truoc':>10}{'HV sau':>10}{'doi':>9}   {'n truoc':>8}{'n sau':>7}\")
for m in sorted(a.get('hv',{})):
    hb=(b.get('hv') or {}).get(m); ha=a['hv'][m]
    nb=(b.get('n') or {}).get(m);  na=(a.get('n') or {}).get(m)
    d='' if hb is None else f'{ha-hb:+.4f}'
    print(f\"  {m:<14}{(hb if hb is not None else float('nan')):>10.4f}{ha:>10.4f}{d:>9}   {str(nb):>8}{str(na):>7}\")
print()
print('  ideal truoc:', b.get('ideal'), '-> sau:', a.get('ideal'))
print('  nadir truoc:', b.get('nadir'), '-> sau:', a.get('nadir'))
" 2>/dev/null || say "không đọc được bản chụp"

  say ""
  say "⇒ Bảng 9.4 (thể tích chi phối) và bảng chi tiết $ARM/$SCENARIO ở mục 9.3 phải cập nhật."
  say "   Nguồn số ĐÚNG để chép: $RESULTS_ROOT/$ARM/campaign-$SCENARIO/table.md"
  say "   (Lưu ý #28: đừng tự tính lại khoảng tin cậy — công cụ dùng t với bậc tự do n−1.)"
  return 0
}

# ════════════════════════════════════════════════════════════════════════════
main() {
  hr "FINISH-DATA  ·  arm=$ARM  kịch bản=$SCENARIO  hạt giống=$SEEDS_ALL"
  say "gốc kết quả=$RESULTS_ROOT · vòng lặp/run=$EPISODES · song song=$PARALLEL"
  say "trạng thái=$STATE_DIR (xoá một tệp .done để làm lại riêng phần đó; FORCE=1 làm lại hết)"

  case "$STAGES" in
    all)   stage_prep && stage_gen && stage_seeds; stage_traj; stage_agg; stage_w65; stage_figs; stage_delta ;;
    prep)  stage_prep ;;
    gen)   stage_prep && stage_gen ;;
    seeds) stage_prep && stage_gen && stage_seeds ;;
    traj)  stage_prep && stage_traj ;;
    agg)   stage_prep && stage_agg && stage_delta ;;
    w65)   stage_w65 ;;
    figs)  stage_figs ;;
    delta) stage_delta ;;
    *) echo "STAGES phải là: all prep gen seeds traj agg w65 figs delta (nhận '$STAGES')" >&2; exit 2 ;;
  esac

  # Trả gateway về cấu hình nghỉ. Để TRACE_PATTERN treo là cách một lần chạy tay sau đó
  # âm thầm đọc WM-1 trong khi kết quả rơi cạnh số Phase-1.
  unset TRACE_PATTERN
  export TOPOLOGY_CONFIG="" NUM_GATEWAYS=1 PY4J_PORT_MAX="$PY4J_PORT"
  docker compose up -d --force-recreate cloudsim-java >/dev/null 2>&1

  hr "KẾT THÚC sau $(elapsed)"
  if [[ ${#FAILURES[@]} -eq 0 ]]; then
    say "không có lỗi nào"
  else
    printf '\033[1;31m   %d vấn đề:\033[0m\n' "${#FAILURES[@]}"
    for f in "${FAILURES[@]}"; do printf '     - %s\n' "$f"; done
    say ""
    say "Chạy lại đúng lệnh cũ: phần đã xong được bỏ qua, chỉ làm lại phần trên."
  fi
  echo
  say "hình cho báo cáo : $REPORT_FIGS/h*.pdf"
  say "bảng cập nhật    : ${RESULTS_ROOT#/}/$ARM/campaign-$SCENARIO/table.md"
  say "W6.5             : ${RESULTS_ROOT#/}/w65/legacy_vs_wm1.md"
  say "nhật ký          : $LOG"
  [[ ${#FAILURES[@]} -eq 0 ]]
}

main
