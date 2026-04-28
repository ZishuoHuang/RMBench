#!/bin/bash
set -u
set -o pipefail

TASK_NAME=${1:-adjust_bottle}
TASK_CONFIG=${2:-aloha-agilex_worldcamera1_randomized_500}
DATA_ROOT=${3:-/mnt/RoboTwin/data}
GPU_LIST=${4:-0,1,2,3}
WORKERS_PER_GPU=${5:-1}
CAMERA_NAME=${6:-world_camera1}
SCENEFLOW_ROOT=${7:-}

if [[ "$WORKERS_PER_GPU" -lt 1 ]]; then
  echo "[ERROR] WORKERS_PER_GPU must be >= 1"
  exit 1
fi

IFS=',' read -r -a GPUS <<< "$GPU_LIST"
if [[ ${#GPUS[@]} -eq 0 ]]; then
  echo "[ERROR] GPU_LIST is empty"
  exit 1
fi

TARGET_DIR="$DATA_ROOT/$TASK_NAME/$TASK_CONFIG"
DATA_DIR="$TARGET_DIR/data"

if [[ ! -d "$DATA_DIR" ]]; then
  echo "[ERROR] Data dir not found: $DATA_DIR"
  exit 1
fi

if [[ -z "$SCENEFLOW_ROOT" ]]; then
  SCENEFLOW_ROOT="$TARGET_DIR/sceneflow_all_points_${CAMERA_NAME}"
fi
mkdir -p "$SCENEFLOW_ROOT"

LOG_DIR="$SCENEFLOW_ROOT/_logs"
mkdir -p "$LOG_DIR"

mapfile -t EPISODES < <(find "$DATA_DIR" -maxdepth 1 -type f -name 'episode*.hdf5' \
  | sed -E 's#.*episode([0-9]+)\.hdf5#\1#' \
  | sort -n)

if [[ ${#EPISODES[@]} -eq 0 ]]; then
  echo "[ERROR] No episode hdf5 found in $DATA_DIR"
  exit 1
fi

for gpu in "${GPUS[@]}"; do
  : > "$LOG_DIR/gpu${gpu}_episodes.txt"
done

idx=0
for ep in "${EPISODES[@]}"; do
  out_dir="$SCENEFLOW_ROOT/episode${ep}"
  if [[ -f "$out_dir/sceneflow_meta.json" ]]; then
    continue
  fi
  gpu_idx=$(( idx % ${#GPUS[@]} ))
  gpu="${GPUS[$gpu_idx]}"
  echo "$ep" >> "$LOG_DIR/gpu${gpu}_episodes.txt"
  idx=$((idx + 1))
done

if [[ "$idx" -eq 0 ]]; then
  echo "[INFO] Nothing to do. All existing episodes already have sceneflow_meta.json"
  exit 0
fi

echo "[INFO] pending episodes: $idx"

pids=()
for gpu in "${GPUS[@]}"; do
  ep_file="$LOG_DIR/gpu${gpu}_episodes.txt"
  gpu_log="$LOG_DIR/gpu${gpu}.log"
  (
    if [[ -s "$ep_file" ]]; then
      xargs -r -a "$ep_file" -P "$WORKERS_PER_GPU" -I{} bash -lc '
        cd /mnt/RoboTwin && \
        CUDA_VISIBLE_DEVICES="$1" PYTHONWARNINGS=ignore::UserWarning \
        python script/replay_sceneflow_all_points.py "$2" "$3" \
          --episode "{}" --camera "$4" --skip-existing --sceneflow-root "$5"
      ' _ "$gpu" "$TASK_NAME" "$TASK_CONFIG" "$CAMERA_NAME" "$SCENEFLOW_ROOT"
    fi
  ) > "$gpu_log" 2>&1 &
  pids+=("$!")
done

fail=0
for pid in "${pids[@]}"; do
  if ! wait "$pid"; then
    fail=$((fail + 1))
  fi
done

echo "[SUMMARY] fail_process_groups=$fail logs=$LOG_DIR"
if [[ "$fail" -gt 0 ]]; then
  exit 2
fi
