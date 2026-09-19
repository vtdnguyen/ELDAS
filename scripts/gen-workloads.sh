#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────
# W1.6 — Generate the whole WM-1 matrix (arm x scenario x seed).
#
# Runs inside the rl-agent image with src mounted, so a freshly added module is
# picked up without a rebuild (the Dockerfile uses COPY src ./src — PLAN risk R13).
# PYTHONPATH=/app/src is required because -m needs the package on the path; the
# existing entrypoints get away without it only because they run src/foo.py as a
# script, which puts src/ on sys.path implicitly.
#
# Output: data/wm1/<arm>/<SCENARIO>/seed<NN>.csv + <arm>/wm1-manifest.json
# Nothing is written outside data/wm1, so LEGACY results are untouched (risk R5).
#
# Env overrides: ARMS, SCENARIOS, SEEDS, IMAGE
# ─────────────────────────────────────────────────────────────
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
ARMS="${ARMS:-homo,hetero}"
SCENARIOS="${SCENARIOS:-}"
SEEDS="${SEEDS:-}"
IMAGE="${IMAGE:-eldas-rl-agent}"

mkdir -p "${ROOT}/data/wm1"

ARGS=(--source /data/trace/openb_pod_list_default.csv
      generate --out /data/wm1 --arms "${ARMS}")
if [[ -n "${SCENARIOS}" ]]; then ARGS+=(--scenarios "${SCENARIOS}"); fi
if [[ -n "${SEEDS}" ]]; then ARGS+=(--seeds "${SEEDS}"); fi

echo "[gen-workloads] arms=${ARMS} scenarios=${SCENARIOS:-all} seeds=${SEEDS:-default}"

MSYS_NO_PATHCONV=1 docker run --rm \
    -v "${ROOT}/data/alibaba-trace:/data/trace:ro" \
    -v "${ROOT}/data/wm1:/data/wm1" \
    -v "${ROOT}/rl-agent/src:/app/src:ro" \
    -e PYTHONPATH=/app/src \
    --entrypoint python "${IMAGE}" -m workload.cli "${ARGS[@]}"

echo ""
echo "[gen-workloads] manifests written:"
find "${ROOT}/data/wm1" -name wm1-manifest.json -print
