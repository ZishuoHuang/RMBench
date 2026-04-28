#!/usr/bin/env bash
set -euo pipefail

ROOT="/mnt/RoboTwin/data"
CAMERA="world_camera1"
JOBS=2
LOG_FILE="/mnt/RoboTwin/logs/async_pipeline/backfill_depth_compress.log"
CONDA_ENV="RoboTwin"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --root)
      ROOT="$2"; shift 2 ;;
    --camera)
      CAMERA="$2"; shift 2 ;;
    --jobs)
      JOBS="$2"; shift 2 ;;
    --log)
      LOG_FILE="$2"; shift 2 ;;
    --conda-env)
      CONDA_ENV="$2"; shift 2 ;;
    *)
      echo "Unknown arg: $1" >&2
      exit 1 ;;
  esac
done

mkdir -p "$(dirname "$LOG_FILE")"
: > "$LOG_FILE"

if ! command -v conda >/dev/null 2>&1; then
  echo "[ERROR] conda not found in PATH" | tee -a "$LOG_FILE"
  exit 127
fi

mapfile -t TARGETS < <(
  find "$ROOT" -type f -path "*/observation/${CAMERA}/depth.npy" | while read -r f; do
    d="$(dirname "$f")"
    [[ -f "$d/depth_dt.b2nd" ]] || echo "$d"
  done | sort -u
)

TOTAL="${#TARGETS[@]}"
echo "[$(date '+%F %T')] backfill start: camera=${CAMERA}, jobs=${JOBS}, targets=${TOTAL}" | tee -a "$LOG_FILE"

if [[ "$TOTAL" -eq 0 ]]; then
  echo "[$(date '+%F %T')] no missing depth_dt.b2nd targets" | tee -a "$LOG_FILE"
  exit 0
fi

active=0
started=0
ok=0
fail=0

run_one() {
  local seg_dir="$1"
  local prefix="[$(date '+%F %T')]"

  if command -v ionice >/dev/null 2>&1; then
    ionice -c2 -n7 nice -n 10 conda run -n "$CONDA_ENV" python /mnt/RoboTwin/script/point_compress.py --mode compress --seg_dir "$seg_dir" >> "$LOG_FILE" 2>&1
  else
    nice -n 10 conda run -n "$CONDA_ENV" python /mnt/RoboTwin/script/point_compress.py --mode compress --seg_dir "$seg_dir" >> "$LOG_FILE" 2>&1
  fi
}

for seg_dir in "${TARGETS[@]}"; do
  started=$((started + 1))
  echo "[$(date '+%F %T')] start ${started}/${TOTAL}: ${seg_dir}" | tee -a "$LOG_FILE"

  (
    if run_one "$seg_dir"; then
      echo "[$(date '+%F %T')] done: ${seg_dir}" >> "$LOG_FILE"
      exit 0
    else
      echo "[$(date '+%F %T')] fail: ${seg_dir}" >> "$LOG_FILE"
      exit 1
    fi
  ) &

  active=$((active + 1))
  if [[ "$active" -ge "$JOBS" ]]; then
    if wait -n; then
      ok=$((ok + 1))
    else
      fail=$((fail + 1))
    fi
    active=$((active - 1))
  fi
done

while [[ "$active" -gt 0 ]]; do
  if wait -n; then
    ok=$((ok + 1))
  else
    fail=$((fail + 1))
  fi
  active=$((active - 1))
done

echo "[$(date '+%F %T')] backfill done: ok=${ok}, fail=${fail}, total=${TOTAL}" | tee -a "$LOG_FILE"
