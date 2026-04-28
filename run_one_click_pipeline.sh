#!/usr/bin/env bash
set -euo pipefail

# Async one-click pipeline launcher.
# Defaults follow the requested deployment:
# - GPUs: 0,1,2,3
# - Mainline workers: 1
# - Compress workers: 1 start, scale up to 1
# - Chunk size: 50 episodes (traj -> hdf5/offline begins at each 50)

CONDA_BIN="${CONDA_BIN:-}"
CONDA_ENV="${CONDA_ENV:-RoboTwin}"
PYTHON_BIN="${PYTHON_BIN:-}"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TASK_CONFIG="aloha-agilex_world832_replay1"
CAMERA="world_camera1"
TASKS="all"
EPISODE_NUM=""
CHUNK_SIZE="50"
GPUS="0,1,2,3"
MAIN_WORKERS="1"
COMPRESS_WORKERS_START="1"
COMPRESS_WORKERS_MAX="1"
DISABLE_OBJECTFLOW="false"
OBJECTFLOW_MESH_ONLY="false"
OBJECTFLOW_MAX_POINTS_PER_MESH="5000"
OBJECTFLOW_SEED="0"
OBJECTFLOW_OVERWRITE="false"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --repo-root)
      REPO_ROOT="$2"; shift 2 ;;
    --task-config|--cfg)
      TASK_CONFIG="$2"; shift 2 ;;
    --camera|--cam)
      CAMERA="$2"; shift 2 ;;
    --tasks)
      TASKS="$2"; shift 2 ;;
    --episode-num)
      EPISODE_NUM="$2"; shift 2 ;;
    --chunk-size)
      CHUNK_SIZE="$2"; shift 2 ;;
    --gpus)
      GPUS="$2"; shift 2 ;;
    --main-workers)
      MAIN_WORKERS="$2"; shift 2 ;;
    --compress-workers-start)
      COMPRESS_WORKERS_START="$2"; shift 2 ;;
    --compress-workers-max)
      COMPRESS_WORKERS_MAX="$2"; shift 2 ;;
    --disable-objectflow)
      DISABLE_OBJECTFLOW="true"; shift 1 ;;
    --objectflow-mesh-only)
      OBJECTFLOW_MESH_ONLY="true"; shift 1 ;;
    --objectflow-max-points-per-mesh)
      OBJECTFLOW_MAX_POINTS_PER_MESH="$2"; shift 2 ;;
    --objectflow-seed)
      OBJECTFLOW_SEED="$2"; shift 2 ;;
    --objectflow-overwrite)
      OBJECTFLOW_OVERWRITE="true"; shift 1 ;;
    --help|-h)
      cat <<'USAGE'
Usage: bash run_one_click_pipeline.sh [options]

Options:
  --repo-root <path>                 Repo root (default: script directory)
  --task-config|--cfg <name>         Task config name (default: aloha-agilex_world832_replay1)
  --camera|--cam <name>              Camera name (default: world_camera1)
  --tasks <all|t1,t2,...>            Tasks to run (default: all)
  --episode-num <int>                Override episode_num from task config
  --chunk-size <int>                 Chunk size for traj->hdf5/offline pipeline (default: 50)
  --gpus <csv>                       GPU ids (default: 0,1,2,3)
  --main-workers <int>               Mainline workers (default: 1)
  --compress-workers-start <int>     Async compress workers at start (default: 1)
  --compress-workers-max <int>       Async compress workers max scale (default: 1)
  --disable-objectflow               Disable objectflow export stage (default: enabled)
  --objectflow-mesh-only             Save only mesh + sampled points metadata, skip per-frame flow export
  --objectflow-max-points-per-mesh   Sample points per object mesh (default: 5000)
  --objectflow-seed <int>            Random seed for object mesh sampling (default: 0)
  --objectflow-overwrite             Rebuild objectflow even if meta already exists
USAGE
      exit 0 ;;
    *)
      echo "Unknown argument: $1"
      echo "Use --help for usage."
      exit 1 ;;
  esac
done

cd "$REPO_ROOT"

# Resolve conda binary dynamically unless explicitly provided.
if [[ -z "$CONDA_BIN" ]]; then
  if command -v conda >/dev/null 2>&1; then
    CONDA_BIN="$(command -v conda)"
  elif [[ -x "$HOME/miniconda3/bin/conda" ]]; then
    CONDA_BIN="$HOME/miniconda3/bin/conda"
  elif [[ -x "$HOME/anaconda3/bin/conda" ]]; then
    CONDA_BIN="$HOME/anaconda3/bin/conda"
  fi
fi

if [[ -z "$CONDA_BIN" || ! -x "$CONDA_BIN" ]]; then
  echo "[ERROR] Could not find an executable conda binary."
  echo "[ERROR] Set CONDA_BIN=/path/to/conda or add conda to PATH."
  exit 127
fi

if [[ -z "$PYTHON_BIN" ]]; then
  PYTHON_BIN="$($CONDA_BIN run -n "$CONDA_ENV" which python 2>/dev/null || true)"
fi

if [[ -z "$PYTHON_BIN" ]]; then
  # Common direct env path fallback when `conda run ... which python` is unavailable.
  if [[ -x "$HOME/miniconda3/envs/$CONDA_ENV/bin/python" ]]; then
    PYTHON_BIN="$HOME/miniconda3/envs/$CONDA_ENV/bin/python"
  elif [[ -x "$HOME/anaconda3/envs/$CONDA_ENV/bin/python" ]]; then
    PYTHON_BIN="$HOME/anaconda3/envs/$CONDA_ENV/bin/python"
  fi
fi

if [[ -z "$PYTHON_BIN" ]]; then
  # Last-resort resolution from current shell, useful when env is already activated.
  CANDIDATE_PY="$(command -v python 2>/dev/null || true)"
  if [[ -n "$CANDIDATE_PY" ]]; then
    PY_BASENAME="$(basename "$CANDIDATE_PY")"
    if [[ "$PY_BASENAME" == "python" || "$PY_BASENAME" == python3* ]]; then
      PYTHON_BIN="$CANDIDATE_PY"
    fi
  fi
fi

if [[ -z "$PYTHON_BIN" || ! -x "$PYTHON_BIN" ]]; then
  echo "[ERROR] Could not resolve python executable in conda env '$CONDA_ENV'."
  echo "[ERROR] Set PYTHON_BIN=/path/to/python manually."
  exit 127
fi

CMD=(
  "$PYTHON_BIN" script/async_pipeline_launcher.py
  --repo-root "$REPO_ROOT"
  --task-config "$TASK_CONFIG"
  --camera "$CAMERA"
  --tasks "$TASKS"
  --chunk-size "$CHUNK_SIZE"
  --gpus "$GPUS"
  --main-workers "$MAIN_WORKERS"
  --compress-workers-start "$COMPRESS_WORKERS_START"
  --compress-workers-max "$COMPRESS_WORKERS_MAX"
  --objectflow-max-points-per-mesh "$OBJECTFLOW_MAX_POINTS_PER_MESH"
  --objectflow-seed "$OBJECTFLOW_SEED"
  --python-bin "$PYTHON_BIN"
)

if [[ "$DISABLE_OBJECTFLOW" == "true" ]]; then
  CMD+=(--disable-objectflow)
fi

if [[ "$OBJECTFLOW_MESH_ONLY" == "true" ]]; then
  CMD+=(--objectflow-mesh-only)
fi

if [[ "$OBJECTFLOW_OVERWRITE" == "true" ]]; then
  CMD+=(--objectflow-overwrite)
fi

if [[ -n "$EPISODE_NUM" ]]; then
  CMD+=(--episode-num "$EPISODE_NUM")
fi

echo "[INFO] Repo root: $REPO_ROOT"
echo "[INFO] Task config: $TASK_CONFIG"
echo "[INFO] Camera: $CAMERA"
echo "[INFO] Tasks: $TASKS"
echo "[INFO] GPUs: $GPUS"
echo "[INFO] Main workers: $MAIN_WORKERS"
echo "[INFO] Compress workers: $COMPRESS_WORKERS_START -> $COMPRESS_WORKERS_MAX"
echo "[INFO] Python bin: $PYTHON_BIN"
echo "[INFO] Chunk size: $CHUNK_SIZE"
if [[ "$DISABLE_OBJECTFLOW" == "true" ]]; then
  echo "[INFO] Objectflow export: disabled"
else
  echo "[INFO] Objectflow export: enabled"
fi
if [[ "$OBJECTFLOW_MESH_ONLY" == "true" ]]; then
  echo "[INFO] Objectflow mode: mesh-only"
fi
echo "[INFO] Objectflow max points/mesh: $OBJECTFLOW_MAX_POINTS_PER_MESH"
echo "[INFO] Objectflow seed: $OBJECTFLOW_SEED"
if [[ "$OBJECTFLOW_OVERWRITE" == "true" ]]; then
  echo "[INFO] Objectflow overwrite: true"
fi
if [[ -n "$EPISODE_NUM" ]]; then
  echo "[INFO] Episode num override: $EPISODE_NUM"
fi

"${CMD[@]}"

echo "[DONE] Async pipeline finished."
