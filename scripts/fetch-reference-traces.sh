#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────
# W4.1 — Fetch the independent burstiness anchors (Philly, Helios).
#
# These two production GPU-cluster traces anchor the WM-1 BURST parameter
# (PLAN-Workload-Model.md 3.5) so it is measured rather than chosen by hand.
#
# READ-ONLY reference: verified on the real files, wall_time is 0 in every row
# of both traces and cpu_num is 0 in every row of Philly, so neither can serve
# as a job-size or deadline source. Only the arrival process is used.
#
# Source: https://github.com/DIR-LAB/Gen-Parallel-Workloads
# Paper : Soundar Raj, MacDougall, Zhang, Dai - JSSPP 2024
# ─────────────────────────────────────────────────────────────
set -euo pipefail

DIR="$(cd "$(dirname "$0")/../data/reference-traces" 2>/dev/null && pwd || true)"
if [[ -z "${DIR}" ]]; then
    DIR="$(cd "$(dirname "$0")/.." && pwd)/data/reference-traces"
    mkdir -p "${DIR}"
fi

BASE="https://raw.githubusercontent.com/DIR-LAB/Gen-Parallel-Workloads/main"

# name | remote path | expected sha256 | expected data rows
ENTRIES=(
  "philly_data_training.csv|Philly/training_data/philly_data_training.csv|98d2ca0f525309a4|15000"
  "helios_data_training.csv|Helios/training_data/helios_data_training.csv|042b2648eb850b8f|15000"
)

info() { echo "[INFO]  $*"; }
ok()   { echo "[OK]    $*"; }
fail() { echo "[ERROR] $*" >&2; exit 1; }

command -v curl >/dev/null 2>&1 || fail "curl not found"

for entry in "${ENTRIES[@]}"; do
    IFS='|' read -r name path want_sha want_rows <<< "${entry}"
    dest="${DIR}/${name}"

    if [[ -s "${dest}" ]]; then
        info "${name} already present - skipping download (delete to refetch)"
    else
        info "downloading ${name} ..."
        curl -fSL --progress-bar -o "${dest}" "${BASE}/${path}" \
            || fail "download failed: ${BASE}/${path}"
    fi

    # sha256 prefix check - catches a truncated download or an upstream change
    got_sha="$(sha256sum "${dest}" | cut -c1-16)"
    [[ "${got_sha}" == "${want_sha}" ]] \
        || fail "${name}: sha256 prefix ${got_sha}, expected ${want_sha}. Upstream changed - re-measure the anchors in PLAN 3.5 before trusting them."

    rows="$(( $(wc -l < "${dest}") - 1 ))"
    [[ "${rows}" -eq "${want_rows}" ]] || fail "${name}: ${rows} data rows, expected ${want_rows}"

    header="$(head -n 1 "${dest}")"
    [[ "${header}" == *"interval"* && "${header}" == *"run_time"* && "${header}" == *"gpu_num"* ]] \
        || fail "${name}: unexpected header: ${header}"

    ok "${name} - ${rows} rows, sha256 ${got_sha}, header OK"
done

echo ""
ok "Reference traces ready in ${DIR}"
echo "Next: python scripts/wm1-design-reference.py --anchors    # must reproduce PLAN 3.5"
