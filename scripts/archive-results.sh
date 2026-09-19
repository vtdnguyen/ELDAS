#!/usr/bin/env bash
# =============================================================================
# ELDAS — snapshot data/results/ vào kho lưu trữ có dấu thời gian (AUDIT).
#
# Mục đích: mỗi lần chạy job mới ĐÈ LÊN data/results/ (cùng đường dẫn). Trước khi
# ghi đè, đóng băng kết quả cũ lại kèm MANIFEST (git SHA, ngày, tham số) để:
#   • so sánh giữa các lần chạy,
#   • trích số vào báo cáo LV mà biết chính xác nó sinh ra từ commit/tham số nào,
#   • quay lui khi lần chạy mới tệ hơn.
#
# CÁCH DÙNG:
#   bash scripts/archive-results.sh                       # nhãn = timestamp
#   bash scripts/archive-results.sh baseline-5seed        # nhãn tuỳ ý
#   NOTE="truoc khi tuned budget" bash scripts/archive-results.sh v1-smoke
#
# Kết quả → data/results-archive/<YYYYMMDD-HHMMSS>-<label>/
# =============================================================================
set -uo pipefail
export MSYS_NO_PATHCONV=1

# Chạy được từ bất kỳ đâu: về thư mục gốc repo (script nằm ở scripts/).
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

SRC="data/results"
LABEL="${1:-run}"
LABEL="$(echo "$LABEL" | tr -c 'A-Za-z0-9._-' '-')"   # nhãn an toàn cho tên thư mục
TS="$(date +%Y%m%d-%H%M%S)"
DEST="data/results-archive/${TS}-${LABEL}"

if [[ ! -d "$SRC" ]]; then
  echo "[archive] Khong thay $SRC — chua co ket qua de luu. Bo qua."
  exit 0
fi

mkdir -p "$DEST"
# cp -r toan bo results (nho: summaries + png + jsonl; model o data/models KHONG copy).
cp -r "$SRC/." "$DEST/" 2>/dev/null || {
  echo "[archive] cp loi — thu rsync..." >&2
  rsync -a "$SRC/" "$DEST/" || { echo "[archive] FAIL"; exit 1; }
}

# ── MANIFEST — dau vet audit ────────────────────────────────────────────────
GIT_SHA="$(git rev-parse --short HEAD 2>/dev/null || echo 'n/a')"
GIT_BRANCH="$(git rev-parse --abbrev-ref HEAD 2>/dev/null || echo 'n/a')"
GIT_DIRTY="$(git status --porcelain 2>/dev/null | wc -l | tr -d ' ')"
{
  echo "ELDAS results snapshot"
  echo "======================"
  echo "timestamp   : $TS"
  echo "label       : $LABEL"
  echo "note        : ${NOTE:-}"
  echo "git branch  : $GIT_BRANCH"
  echo "git commit  : $GIT_SHA"
  echo "git dirty   : ${GIT_DIRTY} file(s) uncommitted"
  echo "source      : $SRC"
  echo
  echo "--- .env (tham so mo phong) ---"
  grep -vE '^\s*#|^\s*$' .env 2>/dev/null | grep -vi 'API_KEY' || true
  echo
  echo "--- noi dung snapshot ---"
  ( cd "$DEST" && find . -maxdepth 2 -type d | sort )
} > "$DEST/MANIFEST.txt"

echo "[archive] ✓ da luu $SRC → $DEST"
echo "[archive]   git $GIT_BRANCH@$GIT_SHA · ${GIT_DIRTY} file chua commit · MANIFEST.txt kem theo"
