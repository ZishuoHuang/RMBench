#!/bin/bash
# 这是一个一键生成 SceneFlow 并自动进行 H.265 无损压缩的流水线脚本

# 检查参数输入
if [ "$#" -lt 4 ]; then
    echo "用法: ./generate_and_compress.sh <任务名> <配置名> <Episode编号> <相机名>"
    echo "示例: ./generate_and_compress.sh pick_diverse_bottles aloha-agilex_clean_50 5 world_camera1"
    exit 1
fi

TASK_NAME=$1
CONFIG_NAME=$2
EPISODE=$3
CAMERA=$4

# Any4D 压缩脚本的绝对路径（不需要移动它）
COMPRESS_SCRIPT="/home/CNS2026497693/.maniskill/ManiSkill/scripts/flow_compress.py"

# 定义统一的输出目录（确保生成和压缩对齐）
# 默认配置下 save_path 是 ./data/<任务名>/<配置名>
SCENEFLOW_DIR="./data/${TASK_NAME}/${CONFIG_NAME}/sceneflow_ep${EPISODE}_${CAMERA}"

echo "================================================================="
echo "[1/2] 正在运行 RoboTwin 物理级仿真，生成高精度 SceneFlow..."
echo "输出目录将被强制指定为: $SCENEFLOW_DIR"
echo "================================================================="

# 启动 Python 脚本生产数据，直接通过 --sceneflow-dir 强制指定目录
python script/replay_point_tracking_observer.py \
    "$TASK_NAME" \
    "$CONFIG_NAME" \
    --episode "$EPISODE" \
    --camera "$CAMERA" \
    --sampling-mode pixel_all \
    --save-sceneflow \
    --sceneflow-keyframes 5 \
    --include-background \
    --sceneflow-dir "$SCENEFLOW_DIR"

# 检查生成是否成功
if [ $? -ne 0 ]; then
    echo "❌ 错误: SceneFlow 生成失败，中止压缩步骤。"
    exit 1
fi

echo ""
echo "================================================================="
echo "[2/2] 生成完毕！开始调用 Any4D 编码器压制 H.265 无损小体积视频..."
echo "压缩目录: $SCENEFLOW_DIR"
echo "================================================================="

# 调用 Any4D 脚本，使用绝对路径，自动清理原 npy 文件
python "$COMPRESS_SCRIPT" compress \
    --out_dir "$SCENEFLOW_DIR" \
    --delete_npy \
    --codec libx265 \
    --crf 0 \
    --bits 10

if [ $? -eq 0 ]; then
    echo ""
    echo "✅ 整个流水线执行完毕！数据已落盘且压制完成。"
else
    echo "❌ 警告: 压缩过程中出现了错误。"
fi
