#!/usr/bin/env bash
# =============================================================================
# ELDAS — W6 in one unattended command.
#
#   bash scripts/w6-overnight.sh
#
# Runs, in order, and without needing anyone at the keyboard:
#
#   [prep]     rebuild stale images (R13), Java ValidationRunner, packed-transport
#              parity, W6.2 characterisation of all 50 WM-1 traces
#   [pilot]    W6.1 — per arm x scenario, measure the feasible SLA floor and DERIVE
#              the budget grid.  The old grid 0.02..0.10 predates the absolute
#              deadline floor (W1.5) and the drop charge (W3), both of which moved
#              the scale of C_SLA; measured LOW is already ~0.11, i.e. the entire
#              old grid sits BELOW the floor, where every budget is infeasible.
#              Deriving beats guessing, and it is why this stage exists at all (R2).
#   [campaign] W6.3 — per arm x scenario: NSGA-II front, ppo-min, CMDP sweep over
#              the derived grid x 5 seeds, heuristics, table + Pareto figure
#   [stats]    W6.4 — hv_bootstrap + eaf_compare per arm x scenario
#
# RESUMABLE.  Every unit of work drops a marker under data/results/wm1/.w6-state/.
# Re-running skips whatever already finished, so a crash, a Docker Desktop restart,
# or Ctrl-C costs only the step that was in flight.  FORCE=1 ignores the markers.
#
# A failing scenario does NOT stop the night: it is recorded and the run moves on,
# and the summary at the end lists exactly what was skipped and why.  The two things
# that DO stop everything are a failing Java validator and a failing transport parity
# check, because past that point the numbers would be wrong rather than missing.
#
# Knobs (env):  ARMS SCENARIOS SEEDS EPISODES PILOT_EPISODES PARALLEL FORCE STAGES
# Examples:
#   ARMS=homo bash scripts/w6-overnight.sh              # one arm tonight, other tomorrow
#   STAGES=pilot bash scripts/w6-overnight.sh           # just re-derive the grids
#   FORCE=1 STAGES=stats bash scripts/w6-overnight.sh   # redo W6.4 on existing points
# =============================================================================
set -uo pipefail
export MSYS_NO_PATHCONV=1          # Git Bash: do not translate /data/... (D2)

ARMS="${ARMS:-homo hetero}"
SCENARIOS="${SCENARIOS:-LOW HIGH BURST OVERLOAD REPLAY}"
SEEDS="${SEEDS:-42,43,44,45,46}"
# 300, not 200: the verified post-fix sweep (§20.4) used 300 and its budget effect was
# already only just above the seed noise. Shortening the runs is the one economy that
# directly attacks the thing being measured.
EPISODES="${EPISODES:-300}"
PILOT_EPISODES="${PILOT_EPISODES:-40}"
PARALLEL="${PARALLEL:-4}"
# W6.3 asks for a fixed-weight PPO *front*, not the single ppo-min point: the RQ4
# claim is that tuning a budget beats tuning a weight, and one point cannot be a
# front. Three weights bracket the trade-off (energy-heavy / balanced / SLA-heavy).
# Set to "" to skip it and cut roughly a third off the night.
PPO_FIXED_WEIGHTS="${PPO_FIXED_WEIGHTS:-0.3,0.5,0.7}"
STAGES="${1:-${STAGES:-all}}"
FORCE="${FORCE:-0}"

PY4J_PORT="${PY4J_PORT:-25333}"
PORT_MAX=$(( PY4J_PORT + PARALLEL - 1 ))
export NUM_GATEWAYS="$PARALLEL" PY4J_PORT_MAX="$PORT_MAX" MONITORING_ENABLED=false

# Dual-loop gains. The default pair 0.05/0.05 predates the reward fix (§20); 0.10/0.30
# is what the verified homo/LOW sweep used and is the one to reproduce.
export K_P="${K_P:-0.10}" K_I="${K_I:-0.30}"

# Where results land, as the CONTAINER sees it. Overriding this is how a re-run avoids
# writing on top of an older one — and after the §19 reward fix that is mandatory, not
# tidiness: points.jsonl is appended to, so a re-run into the old root would average
# policies trained on the corrected reward together with policies trained on the broken
# one, and produce a table that looks completely normal.
RESULTS_ROOT="${RESULTS_ROOT:-/data/results/wm1}"
export RESULTS_ROOT
# The bind mount is ./data/results -> /data/results, so the host path is the container
# path without its leading slash. Markers live under the root too: a new root therefore
# starts with no "already done" markers, which is the behaviour you want when the whole
# point of the re-run is that the old answers are wrong.
HOST_ROOT="${RESULTS_ROOT#/}"
STATE_DIR="$HOST_ROOT/.w6-state"
LOG_DIR="$HOST_ROOT/logs"
mkdir -p "$STATE_DIR" "$LOG_DIR"
RUN_ID="$(date +%Y%m%d-%H%M%S)"
LOG="$LOG_DIR/w6-$RUN_ID.log"

# Everything below is teed, so the log is the whole night including failures.
exec > >(tee -a "$LOG") 2>&1

t0=$(date +%s)
hr()  { printf '\n\033[1m== %s ==\033[0m\n' "$*"; }
sub() { printf '\n\033[1m-- %s --\033[0m\n' "$*"; }
say() { printf '   %s\n' "$*"; }
elapsed() { local s=$(( $(date +%s) - t0 )); printf '%dh%02dm' $((s/3600)) $(((s%3600)/60)); }

FAILURES=()
fail() { FAILURES+=("$*"); printf '\033[1;31m   [FAIL] %s\033[0m\n' "$*"; }

done_marker() { echo "$STATE_DIR/$1.done"; }
is_done() { [[ "$FORCE" != "1" && -f "$(done_marker "$1")" ]]; }
mark_done() { date -Iseconds > "$(done_marker "$1")"; }

# The budget grid is DERIVED from the pilot's J_floor/J_free every time it is needed,
# not read back from the field the pilot happened to write. The grid policy (how far
# below the floor to reach, how far to stay under J_free) is a judgement call that has
# already been revised once; recomputing means revising it does not silently leave the
# campaign running on a grid chosen by an older rule.
#
# The read happens IN the container on purpose. This script exports MSYS_NO_PATHCONV=1
# (required for docker arguments), so Git Bash no longer rewrites /data/... for a
# Windows python.exe and a host-side read comes back EMPTY rather than failing loudly
# - the bug that once turned a run into "0 timesteps, finished instantly, full set of
# plausible result files".
grid_for() {  # grid_for <arm> <scenario>
  docker compose run --rm --no-deps -e ELDAS_BUDGET_FRACTIONS \
    --entrypoint python rl-agent \
    src/eval/pilot_floor.py --print-grid \
    "$RESULTS_ROOT/$1/pilot-$2/pilot_floor.json" 2>/dev/null | tr -d '\r' | tail -1
}

# Docker Desktop on Windows drops its engine pipe occasionally — it happened twice
# during a single afternoon of development on this machine. Unattended, that turns
# into: every remaining docker command fails instantly, the loop burns through all
# remaining scenarios recording failures, and the night is over in ninety seconds.
# Wait for it to come back instead, and give up only if it really is gone.
docker_ready() {
  local tries="${1:-60}"          # 60 x 10 s = 10 minutes
  local i=0
  while ! docker info >/dev/null 2>&1; do
    i=$(( i + 1 ))
    if (( i > tries )); then
      fail "Docker engine unreachable for $(( tries * 10 ))s — stopping. Re-run the same command to resume."
      return 1
    fi
    [[ $i -eq 1 ]] && say "Docker engine unreachable — waiting for it to come back..."
    sleep 10
  done
  (( i > 0 )) && say "Docker engine back after $(( i * 10 ))s"
  return 0
}

gateway_up() {   # gateway_up <arm>
  local arm="$1"
  export TRACE_PATTERN="/data/wm1/${arm}/{scenario}/seed{seed}.csv"
  if [[ "$arm" == "hetero" ]]; then
    export TOPOLOGY_CONFIG="/config/topology-hetero.json"
  else
    export TOPOLOGY_CONFIG=""
  fi
  say "gateway: arm=$arm topology='${TOPOLOGY_CONFIG:-homogeneous}' gateways=$PARALLEL"
  docker compose up -d --force-recreate cloudsim-java >/dev/null 2>&1 || return 1
  sleep 14
  local gw
  gw=$(docker compose logs cloudsim-java 2>&1 | grep -c "GatewayServer listening")
  say "gateways listening: $gw (need $PARALLEL)"
  [[ "$gw" -ge "$PARALLEL" ]]
}

# ════════════════════════════════════════════════════════════════════════════
# STAGE prep
# ════════════════════════════════════════════════════════════════════════════
stage_prep() {
  hr "PREP  (run $RUN_ID · log $LOG)"
  docker_ready || return 1

  sub "images"
  local rebuild=0
  docker image inspect eldas-cloudsim-java >/dev/null 2>&1 || rebuild=1
  docker image inspect eldas-rl-agent      >/dev/null 2>&1 || rebuild=1
  if [[ "$rebuild" == "0" ]]; then
    # rl-agent COPYs src at build time and does NOT mount it, so "the image exists"
    # is not "the image has the current code". Skipping the build after editing a .py
    # means the whole night runs last week's code, silently, producing results that
    # look entirely normal (R13).
    local img_epoch newer
    img_epoch=$(docker image inspect -f '{{.Created}}' eldas-rl-agent 2>/dev/null \
      | python3 -c "import sys,datetime;print(int(datetime.datetime.fromisoformat(sys.stdin.read().strip().replace('Z','+00:00')).timestamp()))" 2>/dev/null || echo 0)
    newer=$(find rl-agent/src rl-agent/requirements.txt cloudsim-java/src -type f \
              -newermt "@${img_epoch}" 2>/dev/null | head -1)
    [[ -n "$newer" ]] && { say "source newer than image (e.g. $newer) -> rebuilding"; rebuild=1; }
  fi
  if [[ "$rebuild" == "1" ]]; then
    docker compose build cloudsim-java rl-agent || { fail "docker build"; return 1; }
  else
    say "images current"
  fi

  sub "W6.2 — characterise all 50 WM-1 traces"
  if is_done "characterize"; then
    say "already done (marker present)"
  else
    # Host-side, stdlib only, no docker paths involved.
    if python scripts/characterize-wm1.py; then
      mark_done "characterize"
    else
      # The billed-window check is a KNOWN fail and is documented as a limitation,
      # so this must not stop the night; the report is written either way.
      say "characterisation reported failing checks — see assets_v2/docs/wm1-characterization.md"
      mark_done "characterize"
    fi
  fi

  sub "Java ValidationRunner"
  local vr
  vr=$(docker run --rm -v "$PWD/data/alibaba-trace:/data/trace:ro" \
        -v "$PWD/data/wm1:/data/wm1:ro" -v "$PWD/config:/config:ro" --entrypoint java \
        eldas-cloudsim-java -cp simulation.jar sim.ValidationRunner 2>&1)
  echo "$vr" | grep -E "^PASS =|^FAIL =|^SKIP ="
  local jfail
  jfail=$(echo "$vr" | sed -n "s/^FAIL = \([0-9]*\).*/\1/p")
  if [[ "${jfail:-1}" != "0" ]]; then
    echo "$vr" | grep -E "^\[FAIL\]"
    fail "Java validation FAIL=${jfail:-?} — HARD STOP (bad code produces wrong numbers, not missing ones)"
    return 1
  fi

  sub "SYS.2 packed-transport parity"
  if is_done "parity"; then
    say "already done"
  else
    gateway_up homo || { fail "gateway for parity check"; return 1; }
    # OVERLOAD, not LOW: the codec change that matters carries droppedTasks, and on a
    # scenario with zero drops the parity check passes vacuously (R4).
    if docker compose run --rm --entrypoint python rl-agent \
         src/perf/verify_packed_parity.py --scenario OVERLOAD --steps 200; then
      mark_done "parity"
    else
      fail "packed-transport parity — HARD STOP (every number downstream reads obs through this codec)"
      return 1
    fi
  fi
  return 0
}

# ════════════════════════════════════════════════════════════════════════════
# STAGE pilot  (W6.1)
# ════════════════════════════════════════════════════════════════════════════
stage_pilot() {
  hr "PILOT — W6.1 feasible SLA floor  (elapsed $(elapsed))"
  for arm in $ARMS; do
    local todo=()
    for sc in $SCENARIOS; do
      is_done "pilot-$arm-$sc" && { say "pilot $arm/$sc: already done"; continue; }
      todo+=("$sc")
    done
    [[ ${#todo[@]} -eq 0 ]] && continue

    docker_ready || return 1
    gateway_up "$arm" || { fail "gateway for pilot arm=$arm"; continue; }
    local results="${RESULTS_ROOT}/${arm}"

    # Run up to PARALLEL scenarios at once, each pinned to its own gateway port so
    # two pilots can never share one simulation.
    local i=0 pids=() names=()
    for sc in "${todo[@]}"; do
      local port=$(( PY4J_PORT + (i % PARALLEL) ))
      sub "pilot $arm/$sc  (port $port, $PILOT_EPISODES episodes x 2 endpoints)"
      docker compose run --rm -e GATEWAY_PORT="$port" -e ELDAS_BUDGET_FRACTIONS \
        --entrypoint python rl-agent \
        src/eval/pilot_floor.py --scenario "$sc" --episodes "$PILOT_EPISODES" \
        --output "$results" > "$LOG_DIR/pilot-$arm-$sc.log" 2>&1 &
      pids+=($!); names+=("$sc")
      i=$(( i + 1 ))
      if (( i % PARALLEL == 0 )); then
        local k=0
        for pid in "${pids[@]}"; do
          wait "$pid"; local rc=$?
          # rc=1 means some acceptance check failed; the floor is still measured and
          # written, so record it and keep going. rc>1 is a real crash.
          [[ $rc -gt 1 ]] && fail "pilot $arm/${names[$k]} crashed (rc=$rc) — see $LOG_DIR/pilot-$arm-${names[$k]}.log"
          [[ $rc -le 1 ]] && mark_done "pilot-$arm-${names[$k]}"
          k=$(( k + 1 ))
        done
        pids=(); names=()
      fi
    done
    local k=0
    for pid in "${pids[@]:-}"; do
      [[ -z "$pid" ]] && continue
      wait "$pid"; local rc=$?
      [[ $rc -gt 1 ]] && fail "pilot $arm/${names[$k]} crashed (rc=$rc) — see $LOG_DIR/pilot-$arm-${names[$k]}.log"
      [[ $rc -le 1 ]] && mark_done "pilot-$arm-${names[$k]}"
      k=$(( k + 1 ))
    done

    sub "derived budget grids — $arm"
    for sc in $SCENARIOS; do
      local g
      g=$(grid_for "$arm" "$sc")
      say "$(printf '%-9s' "$sc") -> ${g:-<missing>}"
    done
  done
  return 0
}

# ════════════════════════════════════════════════════════════════════════════
# STAGE campaign  (W6.3)
# ════════════════════════════════════════════════════════════════════════════
stage_campaign() {
  hr "CAMPAIGN — W6.3  (elapsed $(elapsed))"
  for arm in $ARMS; do
    for sc in $SCENARIOS; do
      if is_done "campaign-$arm-$sc"; then say "campaign $arm/$sc: already done"; continue; fi
      docker_ready || return 1

      local grid
      grid=$(grid_for "$arm" "$sc")
      if [[ -z "$grid" || "$grid" != *,* ]]; then
        # Empty means the pilot never produced a usable J_free for this scenario — it
        # crashed, or the run was too short to record a dual update. Running the sweep
        # anyway would need a budget invented here, and an invented budget lands
        # wherever it lands: most likely in the slack region, where all five runs
        # converge to the same policy and the "Pareto front" is one point drawn five
        # times (§14.3). Skip loudly and let the summary say so.
        fail "campaign $arm/$sc skipped: pilot produced no usable budget grid ('${grid:-empty}')"
        continue
      fi

      sub "campaign $arm/$sc  (budgets $grid · seeds $SEEDS · $EPISODES ep · parallel $PARALLEL)"
      # Clear the topology inherited from a previous arm's stage before handing over;
      # run-full-campaign.sh sets it from ARM, but belt-and-braces because getting
      # this wrong files hetero numbers under homo/ and nothing looks broken.
      export TOPOLOGY_CONFIG=""
      # run-full-campaign.sh owns the gateway, the validator, NSGA-II, ppo-min, the
      # sweep and the aggregation for one scenario. Reusing it keeps exactly one
      # definition of "a campaign" instead of a second, drifting copy here.
      if ARM="$arm" SEEDS="$SEEDS" EPISODES="$EPISODES" PPO_EPISODES="$EPISODES" \
         PARALLEL="$PARALLEL" PPO_FIXED_WEIGHTS="$PPO_FIXED_WEIGHTS" \
         RESULTS_ROOT="$RESULTS_ROOT" K_P="$K_P" K_I="$K_I" \
         bash scripts/run-full-campaign.sh full "$sc" "$grid"; then
        mark_done "campaign-$arm-$sc"
        say "done at $(elapsed)"
      else
        fail "campaign $arm/$sc (rc=$?)"
      fi
    done
  done
  return 0
}

# ════════════════════════════════════════════════════════════════════════════
# STAGE stats  (W6.4)
# ════════════════════════════════════════════════════════════════════════════
stage_stats() {
  hr "STATS — W6.4 hypervolume bootstrap + EAF  (elapsed $(elapsed))"
  for arm in $ARMS; do
    local results="${RESULTS_ROOT}/${arm}"
    for sc in $SCENARIOS; do
      is_done "stats-$arm-$sc" && { say "stats $arm/$sc: already done"; continue; }
      is_done "campaign-$arm-$sc" || { say "stats $arm/$sc: no campaign, skipping"; continue; }

      sub "stats $arm/$sc"
      local ok=1
      docker compose run --rm --no-deps --entrypoint python rl-agent \
        src/eval/hv_bootstrap.py --scenario "$sc" --results "$results" \
        || { fail "hv_bootstrap $arm/$sc"; ok=0; }
      docker compose run --rm --no-deps --entrypoint python rl-agent \
        src/eval/eaf_compare.py --scenario "$sc" --results "$results" \
        || { fail "eaf_compare $arm/$sc"; ok=0; }
      [[ "$ok" == "1" ]] && mark_done "stats-$arm-$sc"
    done
  done
  return 0
}

# ════════════════════════════════════════════════════════════════════════════
main() {
  hr "W6 OVERNIGHT  ·  arms=[$ARMS]  scenarios=[$SCENARIOS]  seeds=$SEEDS"
  say "episodes/run=$EPISODES  pilot episodes=$PILOT_EPISODES  parallel=$PARALLEL"
  say "state=$STATE_DIR (delete a .done file to redo just that piece; FORCE=1 redoes all)"

  case "$STAGES" in
    all)      stage_prep && stage_pilot && stage_campaign && stage_stats ;;
    prep)     stage_prep ;;
    pilot)    stage_pilot ;;
    campaign) stage_campaign ;;
    stats)    stage_stats ;;
    *) echo "STAGES must be one of: all prep pilot campaign stats (got '$STAGES')" >&2; exit 2 ;;
  esac

  # Return the gateway to its resting configuration: one gateway, LEGACY trace,
  # homogeneous. Leaving TRACE_PATTERN set is how a later ad-hoc run silently
  # reads WM-1 while its output lands next to Phase-1 numbers.
  unset TRACE_PATTERN
  export TOPOLOGY_CONFIG="" NUM_GATEWAYS=1 PY4J_PORT_MAX="$PY4J_PORT"
  docker compose up -d --force-recreate cloudsim-java >/dev/null 2>&1

  hr "FINISHED after $(elapsed)"
  if [[ ${#FAILURES[@]} -eq 0 ]]; then
    say "no failures recorded"
  else
    printf '\033[1;31m   %d problem(s):\033[0m\n' "${#FAILURES[@]}"
    for f in "${FAILURES[@]}"; do printf '     - %s\n' "$f"; done
    say ""
    say "Re-run the same command: finished work is skipped, only the above is retried."
  fi
  echo
  say "outputs : $HOST_ROOT/<arm>/campaign-<SC>/{table.md,pareto-<SC>.png}"
  say "pilots  : $HOST_ROOT/<arm>/pilot-<SC>/pilot_floor.json"
  say "workload: assets_v2/docs/wm1-characterization.md"
  say "log     : $LOG"
  [[ ${#FAILURES[@]} -eq 0 ]]
}

main
