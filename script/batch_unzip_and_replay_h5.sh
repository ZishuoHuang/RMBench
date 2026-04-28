#!/bin/bash
set -u
set -o pipefail

DATASET_ROOT=${1:-/mnt/RoboTwin2.0/dataset}
ZIP_NAME=${2:-aloha-agilex_randomized_500.zip}
TASK_CONFIG=${3:-aloha-agilex_worldcamera1_randomized_500}
REPO_ROOT=${4:-/mnt/RoboTwin}
GPU_COUNT=${5:-4}
WORKERS_PER_GPU=${6:-10}
SHARD_MODE=${7:-mod}
LOG_ROOT=${8:-$REPO_ROOT/logs/replay_h5}
PYTHON_BIN=${9:-python}
SKIP_UNZIP=${SKIP_UNZIP:-0}

# Note: DATASET_ROOT is kept for backwards compatibility but not used in replay-only mode
# if [[ ! -d "$DATASET_ROOT" ]]; then
#   echo "[ERROR] Dataset root not found: $DATASET_ROOT"
#   exit 1
# fi

if [[ ! -d "$REPO_ROOT" ]]; then
  echo "[ERROR] Repo root not found: $REPO_ROOT"
  exit 1
fi

if [[ ! -f "$REPO_ROOT/task_config/$TASK_CONFIG.yml" ]]; then
  echo "[ERROR] Missing task config: $REPO_ROOT/task_config/$TASK_CONFIG.yml"
  exit 1
fi

if [[ "$GPU_COUNT" -lt 1 ]]; then
  echo "[ERROR] GPU_COUNT must be >= 1"
  exit 1
fi

if [[ "$WORKERS_PER_GPU" -lt 1 ]]; then
  echo "[ERROR] WORKERS_PER_GPU must be >= 1"
  exit 1
fi

if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
  echo "[ERROR] Python executable not found: $PYTHON_BIN"
  exit 1
fi

if ! "$PYTHON_BIN" -c "import sapien" >/dev/null 2>&1; then
  echo "[ERROR] '$PYTHON_BIN' cannot import sapien. Please activate the correct env first, or pass PYTHON_BIN as the 9th argument."
  echo "[ERROR] Example: bash script/batch_unzip_and_replay_h5.sh ... /path/to/env/bin/python"
  exit 1
fi

if ! "$PYTHON_BIN" -c "import sys; sys.path.append('/mnt/RoboTwin'); import script.collect_data" >/dev/null 2>&1; then
  echo "[ERROR] '$PYTHON_BIN' cannot import script.collect_data from /mnt/RoboTwin"
  exit 1
fi

mapfile -t ALL_TASK_DIRS < <(find "$DATASET_ROOT" -mindepth 1 -maxdepth 1 -type d | sort)
if [[ ${#ALL_TASK_DIRS[@]} -eq 0 ]]; then
  echo "[ERROR] No task folders found in: $DATASET_ROOT"
  exit 1
fi

mkdir -p "$LOG_ROOT"
RUN_ID=$(date +%Y%m%d_%H%M%S)
RUN_LOG_DIR="$LOG_ROOT/$TASK_CONFIG/$RUN_ID"
mkdir -p "$RUN_LOG_DIR"

AVAILABLE_TASK_DIRS=()
MISSING_ZIP_TASKS=()

echo "[INFO] Reusing existing prepared data under $REPO_ROOT/data/$TASK_CONFIG (unzip skipped)"
mapfile -t AVAILABLE_TASK_DIRS < <(
  find "$REPO_ROOT/data" -mindepth 2 -maxdepth 2 -type d -name "$TASK_CONFIG" | sort
)
if [[ ${#AVAILABLE_TASK_DIRS[@]} -eq 0 ]]; then
  echo "[ERROR] No prepared task dirs found for task_config=$TASK_CONFIG under $REPO_ROOT/data"
  exit 1
fi

prepare_task_data() {
  local task_dir="$1"
  local task_name="$2"
  local task_log="$3"
  local zip_path tmp_dir first_entry root_prefix seed_src traj_src target_dir

  zip_path="$task_dir/$ZIP_NAME"
  if [[ ! -f "$zip_path" ]]; then
    echo "[WARN] Missing zip, skip: $zip_path" >> "$task_log"
    return 10
  fi

  target_dir="$REPO_ROOT/data/$task_name/$TASK_CONFIG"
  mkdir -p "$target_dir"

  tmp_dir="$REPO_ROOT/temp_extract/_seed_unpack_${task_name}_$$_$(date +%s%N)"
  rm -rf "$tmp_dir"
  mkdir -p "$tmp_dir"

  first_entry=$(unzip -Z1 "$zip_path" | head -n 1)
  if [[ -z "$first_entry" ]]; then
    echo "[ERROR] Empty or unreadable zip: $zip_path" >> "$task_log"
    rm -rf "$tmp_dir"
    return 11
  fi

  if [[ "$first_entry" == */* ]]; then
    root_prefix=${first_entry%%/*}
    unzip -oq "$zip_path" "$root_prefix/seed.txt" "$root_prefix/_traj_data/*" -d "$tmp_dir"
  else
    unzip -oq "$zip_path" "seed.txt" "_traj_data/*" -d "$tmp_dir"
  fi

  seed_src=$(find "$tmp_dir" -maxdepth 4 -type f -name "seed.txt" | head -n 1)
  traj_src=$(find "$tmp_dir" -maxdepth 5 -type d -name "_traj_data" | head -n 1)

  if [[ -z "$seed_src" || -z "$traj_src" ]]; then
    echo "[ERROR] seed.txt or _traj_data not found after unzip for $task_name" >> "$task_log"
    rm -rf "$tmp_dir"
    return 12
  fi

  cp "$seed_src" "$target_dir/seed.txt"
  rm -rf "$target_dir/_traj_data"
  cp -a "$traj_src" "$target_dir/_traj_data"
  rm -rf "$tmp_dir"

  return 0
}

validate_prepared_task_data() {
  local task_name="$1"
  local task_log="$2"
  local target_dir seed_path traj_dir

  target_dir="$REPO_ROOT/data/$task_name/$TASK_CONFIG"
  seed_path="$target_dir/seed.txt"
  traj_dir="$target_dir/_traj_data"

  if [[ ! -f "$seed_path" ]]; then
    echo "[ERROR] missing seed.txt: $seed_path" >> "$task_log"
    return 20
  fi

  if [[ ! -d "$traj_dir" ]]; then
    echo "[ERROR] missing _traj_data dir: $traj_dir" >> "$task_log"
    return 21
  fi

  if ! find "$traj_dir" -maxdepth 1 -type f -name 'episode*.pkl' | grep -q .; then
    echo "[ERROR] no episode*.pkl found in: $traj_dir" >> "$task_log"
    return 22
  fi

  return 0
}

run_task_in_chunks() {
  local task_name="$1"
  local gpu_id="$2"
  local task_log="$3"
  local target_dir seed_path total_eps chunk_workers chunk_size worker start end worker_log

  target_dir="$REPO_ROOT/data/$task_name/$TASK_CONFIG"
  seed_path="$target_dir/seed.txt"

  if [[ ! -f "$seed_path" ]]; then
    echo "[ERROR] seed file missing after preparation: $seed_path" >> "$task_log"
    return 20
  fi

  total_eps=$(wc -w < "$seed_path")
  if [[ "$total_eps" -le 0 ]]; then
    echo "[WARN] empty seed list for task=$task_name, skip" >> "$task_log"
    return 21
  fi

  chunk_workers="$WORKERS_PER_GPU"
  if [[ "$chunk_workers" -gt "$total_eps" ]]; then
    chunk_workers="$total_eps"
  fi

  chunk_size=$(( (total_eps + chunk_workers - 1) / chunk_workers ))
  echo "[INFO] task=$task_name gpu=$gpu_id total_eps=$total_eps chunk_workers=$chunk_workers chunk_size=$chunk_size" >> "$task_log"

  worker_pids=()
  for ((worker = 0; worker < chunk_workers; worker++)); do
    start=$((worker * chunk_size))
    end=$((start + chunk_size))
    if [[ "$end" -gt "$total_eps" ]]; then
      end="$total_eps"
    fi
    if [[ "$start" -ge "$end" ]]; then
      continue
    fi

    worker_log="$RUN_LOG_DIR/${task_name}.gpu${gpu_id}.w${worker}.log"
    (
      cd "$REPO_ROOT" && \
      CUDA_VISIBLE_DEVICES="$gpu_id" \
      PYTHONWARNINGS=ignore::UserWarning \
      "$PYTHON_BIN" script/collect_data.py "$task_name" "$TASK_CONFIG" --episode-start "$start" --episode-end "$end"
    ) >> "$worker_log" 2>&1 &
    worker_pids+=("$!")
  done

  worker_fail=0
  for pid in "${worker_pids[@]}"; do
    if ! wait "$pid"; then
      worker_fail=$((worker_fail + 1))
    fi
  done

  if [[ "$worker_fail" -gt 0 ]]; then
    echo "[ERROR] task=$task_name gpu=$gpu_id worker_fail=$worker_fail" >> "$task_log"
    return 22
  fi

  echo "[OK] replay done: task=$task_name gpu=$gpu_id" >> "$task_log"
  return 0
}

gpu_worker_loop() {
  local gpu_id="$1"
  local task_list_file="$2"
  local task_dir task_name task_log

  while IFS= read -r task_dir; do
    [[ -z "$task_dir" ]] && continue
    task_name=$(basename $(dirname "$task_dir"))
    task_log="$RUN_LOG_DIR/${task_name}.log"
    echo "============================================================" >> "$task_log"
    echo "[INFO] task=$task_name gpu=$gpu_id replay_start=$(date -Is)" >> "$task_log"
    run_task_in_chunks "$task_name" "$gpu_id" "$task_log" || return 1
  done < "$task_list_file"

  return 0
}

# Phase 1: prepare or validate all tasks first
PREPARED_TASK_DIRS=()
PREPARE_FAIL_COUNT=0
if [[ "$SKIP_UNZIP" -eq 1 ]]; then
  echo "[INFO] Phase 1/2: validating existing prepared data ..."
  for task_dir in "${AVAILABLE_TASK_DIRS[@]}"; do
    task_name=$(basename $(dirname "$task_dir"))  # Extract task name from path like /path/task/config
    task_log="$RUN_LOG_DIR/${task_name}.log"
    echo "============================================================" >> "$task_log"
    echo "[INFO] task=$task_name validate_start=$(date -Is)" >> "$task_log"

    validate_prepared_task_data "$task_name" "$task_log"
    prep_rc=$?
    if [[ "$prep_rc" -eq 0 ]]; then
      PREPARED_TASK_DIRS+=("$task_dir")
      echo "[OK] task=$task_name validate done" >> "$task_log"
    else
      PREPARE_FAIL_COUNT=$((PREPARE_FAIL_COUNT + 1))
      echo "[ERROR] task=$task_name validate failed rc=$prep_rc" >> "$task_log"
    fi
  done
else
  echo "[INFO] Phase 1/2: skipping unzip, using existing prepared data ..."
  # 解压部分已被注释掉，直接使用 /mnt/RoboTwin/data/<task>/<config> 下的现有数据
  for task_dir in "${AVAILABLE_TASK_DIRS[@]}"; do
    task_name=$(basename $(dirname "$task_dir"))
    task_log="$RUN_LOG_DIR/${task_name}.log"
    echo "============================================================" >> "$task_log"
    echo "[INFO] task=$task_name (no unzip, using existing data)" >> "$task_log"

    # 注释掉: prepare_task_data "$task_dir" "$task_name" "$task_log"
    # 直接认为数据已经准备好
    PREPARED_TASK_DIRS+=("$task_dir")
    echo "[OK] task=$task_name ready to replay" >> "$task_log"
  done
fi

TASK_COUNT=${#PREPARED_TASK_DIRS[@]}
if [[ "$TASK_COUNT" -eq 0 ]]; then
  echo "[ERROR] Phase 1 completed, but no task is prepared successfully."
  exit 3
fi

if [[ "$GPU_COUNT" -gt "$TASK_COUNT" ]]; then
  GPU_COUNT="$TASK_COUNT"
fi

echo "[INFO] Phase 1/2 done: prepared_tasks=$TASK_COUNT, prepare_fail_count=$PREPARE_FAIL_COUNT"
echo "[INFO] Phase 2/2: start replay on prepared tasks ..."

shard_size=$(( (TASK_COUNT + GPU_COUNT - 1) / GPU_COUNT ))

for ((gpu_id = 0; gpu_id < GPU_COUNT; gpu_id++)); do
  task_list_file="$RUN_LOG_DIR/gpu${gpu_id}_tasks.txt"
  : > "$task_list_file"
done

if [[ "$SHARD_MODE" == "mod" ]]; then
  idx=1
  for task_dir in "${PREPARED_TASK_DIRS[@]}"; do
    gpu_id=$(( (idx - 1) % GPU_COUNT ))
    printf '%s\n' "$task_dir" >> "$RUN_LOG_DIR/gpu${gpu_id}_tasks.txt"
    idx=$((idx + 1))
  done
else
  start_idx=1
  for ((gpu_id = 0; gpu_id < GPU_COUNT; gpu_id++)); do
    end_idx=$((start_idx + shard_size - 1))
    if [[ "$end_idx" -gt "$TASK_COUNT" ]]; then
      end_idx="$TASK_COUNT"
    fi
    for ((idx = start_idx; idx <= end_idx; idx++)); do
      printf '%s\n' "${PREPARED_TASK_DIRS[$((idx - 1))]}" >> "$RUN_LOG_DIR/gpu${gpu_id}_tasks.txt"
    done
    start_idx=$((end_idx + 1))
    if [[ "$start_idx" -gt "$TASK_COUNT" ]]; then
      break
    fi
  done
fi

pids=()
for ((gpu_id = 0; gpu_id < GPU_COUNT; gpu_id++)); do
  task_list_file="$RUN_LOG_DIR/gpu${gpu_id}_tasks.txt"
  (
    gpu_worker_loop "$gpu_id" "$task_list_file"
  ) &
  pids+=("$!")
done

fail_count=0
for pid in "${pids[@]}"; do
  if ! wait "$pid"; then
    fail_count=$((fail_count + 1))
  fi
done

summary_file="$RUN_LOG_DIR/summary.txt"
{
  echo "run_id=$RUN_ID"
  echo "task_config=$TASK_CONFIG"
  echo "dataset_root=$DATASET_ROOT"
  echo "zip_name=$ZIP_NAME"
  echo "gpu_count=$GPU_COUNT"
  echo "workers_per_gpu=$WORKERS_PER_GPU"
  echo "shard_mode=$SHARD_MODE"
  echo "task_count=$TASK_COUNT"
  echo "task_count_all=${#ALL_TASK_DIRS[@]}"
  echo "task_count_available_zip=${#AVAILABLE_TASK_DIRS[@]}"
  echo "task_count_prepared=$TASK_COUNT"
  echo "prepare_fail_count=$PREPARE_FAIL_COUNT"
  echo "task_count_missing_zip=${#MISSING_ZIP_TASKS[@]}"
  echo "skip_unzip=$SKIP_UNZIP"
  echo "log_dir=$RUN_LOG_DIR"
  echo "fail_count=$fail_count"
} | tee "$summary_file"

if [[ "$fail_count" -gt 0 ]]; then
  exit 2
fi
