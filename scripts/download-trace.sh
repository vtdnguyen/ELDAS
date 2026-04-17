#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────
# T0.0 — Download & validate Alibaba GPU cluster trace (OpenB)
# Source: https://github.com/alibaba/clusterdata/tree/master/cluster-trace-gpu-v2023
# ─────────────────────────────────────────────────────────────
set -euo pipefail

# ── Config ──────────────────────────────────────────────────
TRACE_DIR="$(cd "$(dirname "$0")/../data/alibaba-trace" && pwd)"
FILE_NAME="openb_pod_list_default.csv"
FILE_PATH="${TRACE_DIR}/${FILE_NAME}"
DOWNLOAD_URL="https://raw.githubusercontent.com/alibaba/clusterdata/master/cluster-trace-gpu-v2023/csv/${FILE_NAME}"

# Cột bắt buộc trong CSV (theo README của Alibaba clusterdata)
EXPECTED_COLUMNS="name,cpu_milli,memory_mib,num_gpu,gpu_milli,gpu_spec,qos,pod_phase,creation_time,deletion_time,scheduled_time"
# Số dòng dữ liệu tối thiểu (README ghi ~8152 tasks)
MIN_ROWS=8000

# ── Helpers ─────────────────────────────────────────────────
info()  { echo "[INFO]  $*"; }
error() { echo "[ERROR] $*" >&2; }
ok()    { echo "[OK]    $*"; }

# ── Download ────────────────────────────────────────────────
download_trace() {
    mkdir -p "${TRACE_DIR}"

    if [[ -f "${FILE_PATH}" ]]; then
        info "File already exists: ${FILE_PATH}"
        info "Skipping download. Delete the file to re-download."
        return 0
    fi

    info "Downloading ${FILE_NAME} ..."
    info "URL: ${DOWNLOAD_URL}"

    if command -v curl &>/dev/null; then
        curl -fSL --progress-bar -o "${FILE_PATH}" "${DOWNLOAD_URL}"
    elif command -v wget &>/dev/null; then
        wget --show-progress -O "${FILE_PATH}" "${DOWNLOAD_URL}"
    else
        error "Neither curl nor wget found. Please install one of them."
        exit 1
    fi

    ok "Download complete: ${FILE_PATH}"
}

# ── Validate ────────────────────────────────────────────────
validate_trace() {
    info "Validating ${FILE_NAME} ..."

    # 1. File tồn tại và không rỗng
    if [[ ! -s "${FILE_PATH}" ]]; then
        error "File is missing or empty: ${FILE_PATH}"
        exit 1
    fi
    ok "File exists and is non-empty ($(du -h "${FILE_PATH}" | cut -f1))"

    # 2. Kiểm tra header — tất cả cột bắt buộc phải có mặt
    local header
    header=$(head -n 1 "${FILE_PATH}")

    local missing=()
    IFS=',' read -ra cols <<< "${EXPECTED_COLUMNS}"
    for col in "${cols[@]}"; do
        if ! echo "${header}" | grep -q "${col}"; then
            missing+=("${col}")
        fi
    done

    if [[ ${#missing[@]} -gt 0 ]]; then
        error "Missing columns: ${missing[*]}"
        error "Actual header: ${header}"
        exit 1
    fi
    ok "All ${#cols[@]} expected columns present"

    # 3. Kiểm tra số dòng dữ liệu (trừ header)
    local row_count
    row_count=$(tail -n +2 "${FILE_PATH}" | wc -l | tr -d ' ')

    if [[ "${row_count}" -lt "${MIN_ROWS}" ]]; then
        error "Expected at least ${MIN_ROWS} data rows, got ${row_count}"
        exit 1
    fi
    ok "Row count: ${row_count} (>= ${MIN_ROWS} required)"

    # 4. Kiểm tra giá trị qos — phải chứa ít nhất Burstable và LatencySensitive
    #    (quan trọng vì reward.py dùng qos để tính lambda)
    local qos_col_idx
    qos_col_idx=$(echo "${header}" | tr ',' '\n' | grep -n '^qos$' | cut -d: -f1)

    if [[ -n "${qos_col_idx}" ]]; then
        local qos_values
        qos_values=$(tail -n +2 "${FILE_PATH}" | cut -d',' -f"${qos_col_idx}" | sort -u | tr '\n' ' ')
        ok "QoS values found: ${qos_values}"

        if ! echo "${qos_values}" | grep -q "Burstable"; then
            error "QoS value 'Burstable' not found in data"
            exit 1
        fi
    fi

    # 5. Spot-check: vài dòng đầu phải parse được (không bị lỗi encoding)
    local sample_lines
    sample_lines=$(head -n 5 "${FILE_PATH}" | wc -l | tr -d ' ')
    if [[ "${sample_lines}" -lt 2 ]]; then
        error "File appears corrupted — fewer than 2 readable lines"
        exit 1
    fi
    ok "Spot-check passed (first ${sample_lines} lines readable)"

    echo ""
    ok "Validation PASSED — ${FILE_NAME} is ready to use"
}

# ── Main ────────────────────────────────────────────────────
main() {
    echo "=================================================="
    echo " Alibaba OpenB Trace — Download & Validate"
    echo "=================================================="
    echo ""
    download_trace
    echo ""
    validate_trace
}

main "$@"
