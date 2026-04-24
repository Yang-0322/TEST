#!/bin/bash

# --- 配置区 ---
DEFAULT_GPU_ID=2      # 默认使用的显卡ID
HOLD_GB=20            # 训练完后想占住多少GB
TRAIN_SCRIPT="train_climb_occlusion.py"
CONFIG_FILE="./config/climb-vit-marketocclusion.yml"

# --- 逻辑处理 ---
# 如果执行脚本时传入了第一个参数，则使用该参数作为 GPU_ID，否则使用默认值
# 例如执行: bash run.sh 1
GPU_ID=${1:-$DEFAULT_GPU_ID}

echo "****************************************"
echo "使用显卡 ID: $GPU_ID"
echo "显存占位量: ${HOLD_GB}GB"
echo "****************************************"

# 1. 执行训练任务
echo "--- [Step 1] 正在启动训练任务 ---"
# 使用环境变量方式指定 GPU，这样可以确保训练命令内部能识别到
CUDA_VISIBLE_DEVICES=$GPU_ID python $TRAIN_SCRIPT --config_file $CONFIG_FILE

# 2. 无论训练成功还是失败，只要进程结束，就执行抢占脚本
echo "--- 继续执行训练任务 ---"
python train_qiuxiu.py