#!/bin/bash

# 压缩刚刚生成的 SceneFlow 文件夹
TARGET_DIR="data/pick_diverse_bottles/aloha-agilex_clean_50/sceneflow_ep5_world_camera1"
COMPRESS_SCRIPT="/home/CNS2026497693/.maniskill/ManiSkill/scripts/flow_compress.py"

if [ ! -d "$TARGET_DIR" ]; then
    echo "Error: Directory $TARGET_DIR does not exist!"
    exit 1
fi

echo "======================================="
echo "开始对目录进行 Any4D SceneFlow 视频级压缩:"
echo "$TARGET_DIR"
echo "======================================="

# 调用 Any4D 提供的流压缩脚本
# --delete_npy 会在视频压缩成功后自动删除巨大的 .npy 原文件，为你节约空间
# --bits 10 --crf 0 保证了近乎无损的高精度压缩
python $COMPRESS_SCRIPT compress --out_dir "$TARGET_DIR" --delete_npy --codec libx265 --crf 0 --bits 10

echo "======================================="
echo "压缩完成！"
