#!/usr/bin/env bash
# T5.7 — End-to-end smoke check for the ELDAS monitoring stack.
#
# Verifies that the four moving parts agree with each other:
#   1. Java exporter   (cloudsim-java :9091/metrics) — registry is live
#   2. Prometheus      (:9090/api/v1/targets)        — scrape target UP
#   3. Prometheus      (:9090/api/v1/query)          — ELDAS series exist
#   4. Grafana         (:3000/api/health)            — service healthy
#   5. Grafana         (datasource proxy)            — can talk to Prom
#
# Default endpoints assume you ran `docker compose --profile monitoring up`
# on the local host. Override via env vars when running from elsewhere:
#
#   ELDAS_JAVA_URL=http://cloudsim-java:9091   (default localhost)
#   ELDAS_PROM_URL=http://prometheus:9090      (default localhost)
#   ELDAS_GRAFANA_URL=http://grafana:3000      (default localhost)
#   GF_ADMIN_USER, GF_ADMIN_PASSWORD           (default admin/admin)
#
# Exit code: 0 = all checks passed, 1 = at least one failed.

set -u

# ── Config ────────────────────────────────────────────────────────────────

JAVA_URL="${ELDAS_JAVA_URL:-http://localhost:9091}"
PROM_URL="${ELDAS_PROM_URL:-http://localhost:9090}"
GRAFANA_URL="${ELDAS_GRAFANA_URL:-http://localhost:3000}"
GF_USER="${GF_ADMIN_USER:-admin}"
GF_PASS="${GF_ADMIN_PASSWORD:-admin}"

# ── Pretty output ─────────────────────────────────────────────────────────

if [[ -t 1 ]]; then
  C_PASS='\033[32m'; C_FAIL='\033[31m'; C_INFO='\033[36m'; C_RESET='\033[0m'
else
  C_PASS=''; C_FAIL=''; C_INFO=''; C_RESET=''
fi

pass_count=0
fail_count=0

pass() { printf "${C_PASS}[PASS]${C_RESET} %s\n" "$1"; pass_count=$((pass_count + 1)); }
fail() { printf "${C_FAIL}[FAIL]${C_RESET} %s\n" "$1"; fail_count=$((fail_count + 1)); }
info() { printf "${C_INFO}[--]${C_RESET}   %s\n" "$1"; }
section() { printf "\n${C_INFO}=== %s ===${C_RESET}\n" "$1"; }

# Returns 0 if `curl` can reach $1 and HTTP status is 2xx.
# Body (if any) is captured into the variable named by $2 (passed by name).
fetch() {
  local url="$1"; local -n out_var="$2"
  out_var=$(curl --silent --show-error --max-time 5 \
                --write-out '\nHTTP_STATUS:%{http_code}' "$url" 2>&1) || return 1
  local status="${out_var##*HTTP_STATUS:}"
  out_var="${out_var%$'\n'HTTP_STATUS:*}"
  [[ "$status" =~ ^2 ]]
}

fetch_auth() {
  local url="$1"; local -n out_var="$2"
  out_var=$(curl --silent --show-error --max-time 5 -u "${GF_USER}:${GF_PASS}" \
                --write-out '\nHTTP_STATUS:%{http_code}' "$url" 2>&1) || return 1
  local status="${out_var##*HTTP_STATUS:}"
  out_var="${out_var%$'\n'HTTP_STATUS:*}"
  [[ "$status" =~ ^2 ]]
}

# Tiny JSON field extractor — avoids requiring jq. Looks for `"key": value`
# patterns where value is a JSON literal (string/number/bool). Good enough
# for the small responses we inspect here.
json_field() {
  local key="$1"; local body="$2"
  # shellcheck disable=SC2001
  echo "$body" | sed -n "s/.*\"${key}\":[ ]*\([\"a-zA-Z0-9._-][^,}]*\).*/\1/p" \
                    | head -n1 | tr -d '"'
}

# Health of ONE Prometheus scrape target, by job name.
#
# The /api/v1/targets body is a single line containing every target, so a naive
# grep -E '"job":"X".*"health":"up"' matches ACROSS target boundaries: job X is
# reported UP whenever ANY later target is up. That produced a false PASS for
# the (actually down) rl_agent exporter. Splitting on '{' puts each target
# object on its own chunk so the match is scoped correctly.
target_health() {
  local job="$1"; local body="$2"
  echo "$body" \
    | tr '{' '\n' \
    | grep -A20 "\"job\":\"${job}\"" \
    | grep -oE '"health":"[a-z]+"' \
    | head -n1 | cut -d'"' -f4
}

# ── 1. Java exporter ──────────────────────────────────────────────────────

section "1. Java exporter — ${JAVA_URL}/metrics"

if fetch "${JAVA_URL}/metrics" body; then
  pass "Java /metrics is reachable (HTTP 200)"

  registered_count=$(echo "$body" | grep -cE '^# TYPE eldas_' || true)
  if (( registered_count >= 8 )); then
    pass "All 8 ELDAS metric families are registered (found ${registered_count})"
  else
    fail "Only ${registered_count} ELDAS metric families registered; expected ≥ 8"
    info "Hint: MONITORING_ENABLED must be 'true' in cloudsim-java's env."
  fi

  # Are any series populated? (Header lines start with #; a series line does not.)
  series_count=$(echo "$body" | grep -cE '^eldas_' || true)
  if (( series_count > 0 )); then
    pass "Java exporter has ${series_count} live series"
  else
    info "No populated series yet — run baseline_eval or smoke_test to drive data."
  fi
else
  fail "Java /metrics is unreachable at ${JAVA_URL}/metrics"
  info "Hint: 'docker compose up cloudsim-java' and check MONITORING_ENABLED=true."
fi

# ── 2. Prometheus targets ─────────────────────────────────────────────────

section "2. Prometheus targets — ${PROM_URL}/api/v1/targets"

if fetch "${PROM_URL}/api/v1/targets" body; then
  pass "Prometheus targets API reachable"

  java_health=$(target_health cloudsim_java "$body")
  if [[ "$java_health" == "up" ]]; then
    pass "Scrape target cloudsim_java is UP"
  else
    fail "Scrape target cloudsim_java is ${java_health:-missing} — check prometheus.yml + container network"
    last_err=$(echo "$body" | grep -oE '"lastError":"[^"]*"' | head -n1)
    [[ -n "$last_err" ]] && info "Prometheus says: ${last_err}"
  fi

  if echo "$body" | grep -qE '"job":"rl_agent"'; then
    rl_health=$(target_health rl_agent "$body")
    if [[ "$rl_health" == "up" ]]; then
      pass "Scrape target rl_agent is UP (Phase 2 / training run active)"
    else
      info "Scrape target rl_agent is ${rl_health:-missing} — expected when no training run is active."
      info "If a run IS active, it must be started with 'docker compose run --use-aliases'"
      info "(without it the container has no 'rl-agent' DNS alias and can never be scraped)."
    fi
  fi
else
  fail "Prometheus targets API unreachable at ${PROM_URL}"
fi

# ── 3. Prometheus query ───────────────────────────────────────────────────

section "3. Prometheus query — count(eldas_host_cpu_util)"

query_url="${PROM_URL}/api/v1/query?query=count(eldas_host_cpu_util)"
if fetch "$query_url" body; then
  pass "Prometheus query API reachable"
  # Response shape: {"status":"success","data":{"resultType":"vector","result":[{"metric":{},"value":[<t>, "<n>"]}]}}
  count_val=$(echo "$body" \
              | grep -oE '"value":\[[0-9.]+,"[0-9]+"' \
              | head -n1 \
              | grep -oE '"[0-9]+"$' \
              | tr -d '"')
  if [[ -n "${count_val:-}" && "$count_val" -gt 0 ]]; then
    pass "Prometheus has eldas_host_cpu_util data (${count_val} series, expected ≥ 1)"
    # Expected host count comes from NUM_HOSTS (same default as SimulationConfig),
    # never a hardcoded 10 — a 50-host topology sweep would otherwise fail a
    # check that has nothing to do with monitoring health.
    expected_hosts="${NUM_HOSTS:-10}"
    if [[ "$count_val" -ge "$expected_hosts" ]]; then
      pass "All ${expected_hosts} host series are present (count=${count_val})"
    else
      info "Only ${count_val} host series vs NUM_HOSTS=${expected_hosts} — partial run, or stale series from an earlier topology."
    fi
  else
    fail "No data for eldas_host_cpu_util yet — Java side may not have scheduled any task."
    info "Hint: run 'baseline_eval.py' or hit the gateway via Py4J first."
  fi
else
  fail "Prometheus query API unreachable"
fi

# ── 4. Grafana health ─────────────────────────────────────────────────────

section "4. Grafana — ${GRAFANA_URL}/api/health"

if fetch "${GRAFANA_URL}/api/health" body; then
  db_state=$(json_field database "$body")
  if [[ "$db_state" == "ok" ]]; then
    pass "Grafana service is up (database=${db_state})"
  else
    fail "Grafana health check returned database=${db_state:-?}"
  fi
else
  fail "Grafana unreachable at ${GRAFANA_URL}/api/health"
fi

# ── 5. Grafana → Prometheus datasource proxy ──────────────────────────────

section "5. Grafana datasource proxy → Prometheus"

if fetch_auth "${GRAFANA_URL}/api/datasources/uid/eldas-prometheus/health" body; then
  status=$(json_field status "$body")
  if [[ "$status" == "OK" ]]; then
    pass "Grafana can reach Prometheus through the provisioned datasource"
  else
    fail "Datasource health returned status=${status:-?}; body: ${body:0:200}"
  fi
else
  fail "Datasource health endpoint unreachable (auth ${GF_USER}/${GF_PASS} ?)"
  info "Body: ${body:0:200}"
fi

if fetch_auth "${GRAFANA_URL}/api/search?type=dash-db&query=ELDAS" body; then
  if echo "$body" | grep -q '"uid":"eldas-live"'; then
    pass "Dashboard 'ELDAS Live' is provisioned"
  else
    fail "Dashboard 'ELDAS Live' not found in Grafana — check provisioning/dashboards/"
  fi
  if echo "$body" | grep -q '"uid":"eldas-scheduler-comparison"'; then
    pass "Dashboard 'ELDAS Scheduler Comparison' is provisioned"
  else
    fail "Dashboard 'ELDAS Scheduler Comparison' not found — re-check provisioning/dashboards/"
  fi
fi

# ── 6. Optional: drive a real Py4J episode and verify counters move ──────
# Skipped unless --drive is passed. Useful for full end-to-end validation
# in CI or after first install. Requires docker compose + a built rl-agent.

if [[ "${1:-}" == "--drive" ]]; then
  section "6. Drive a Py4J episode (50 K8s steps) and verify metrics move"

  before_body=""
  if fetch "${PROM_URL}/api/v1/query?query=sum(eldas_tasks_scheduled_total)" before_body; then
    before_val=$(echo "$before_body" \
                  | grep -oE '"value":\[[0-9.]+,"[0-9.e+-]+"' \
                  | head -n1 \
                  | grep -oE '"[0-9.e+-]+"$' \
                  | tr -d '"')
    info "tasks_scheduled_total BEFORE = ${before_val:-0}"
  fi

  # The rl-agent image already has py4j installed. Use --no-deps so we don't
  # spin up the monitoring containers a second time.
  episode_out=$(docker compose run --rm --no-deps rl-agent python -c "
from py4j.java_gateway import JavaGateway, GatewayParameters
gw = JavaGateway(gateway_parameters=GatewayParameters(address='cloudsim-java', port=25333))
ep = gw.entry_point
ep.reset('LOW', 42)
for _ in range(50):
    ep.step(int(ep.selectBaselineAction('k8s')))
print('OK', round(ep.getTotalEnergyKwh(), 4))
" 2>&1 | tail -n1)

  if [[ "$episode_out" == OK* ]]; then
    pass "Py4J episode completed: ${episode_out}"
  else
    fail "Py4J episode failed: ${episode_out}"
  fi

  # Let Prometheus pick up the new values (2 scrape cycles).
  sleep 5

  after_body=""
  if fetch "${PROM_URL}/api/v1/query?query=sum(eldas_tasks_scheduled_total)" after_body; then
    after_val=$(echo "$after_body" \
                  | grep -oE '"value":\[[0-9.]+,"[0-9.e+-]+"' \
                  | head -n1 \
                  | grep -oE '"[0-9.e+-]+"$' \
                  | tr -d '"')
    info "tasks_scheduled_total AFTER  = ${after_val:-0}"

    # Counter must have strictly increased.
    if awk "BEGIN { exit !((${after_val:-0}) > (${before_val:-0})) }"; then
      pass "Counter eldas_tasks_scheduled_total increased (data flowing end-to-end)"
    else
      fail "Counter did not increase — metrics NOT propagating to Prometheus"
    fi
  fi

  # Verify the scheduler tag rotated to "k8s" on host gauges.
  if fetch "${PROM_URL}/api/v1/query?query=count(eldas_host_cpu_util%7Bscheduler%3D%22k8s%22%7D)" k8s_body; then
    k8s_count=$(echo "$k8s_body" \
                  | grep -oE '"value":\[[0-9.]+,"[0-9]+"' \
                  | head -n1 \
                  | grep -oE '"[0-9]+"$' \
                  | tr -d '"')
    if [[ -n "${k8s_count:-}" && "$k8s_count" -ge 10 ]]; then
      pass "Scheduler tag rotated to k8s on ${k8s_count} host series"
    else
      fail "Expected ≥10 series with scheduler=\"k8s\"; got ${k8s_count:-0}"
    fi
  fi
fi

# ── Summary ───────────────────────────────────────────────────────────────

section "Summary"
echo "  PASS: ${pass_count}"
echo "  FAIL: ${fail_count}"

if (( fail_count == 0 )); then
  printf "\n${C_PASS}✓ Monitoring stack is healthy end-to-end.${C_RESET}\n"
  exit 0
else
  printf "\n${C_FAIL}✗ Monitoring stack has issues — see [FAIL] lines above.${C_RESET}\n"
  exit 1
fi
