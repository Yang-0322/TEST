#!/bin/bash

# 训练 Qiuxiu 数据集 - 路径交换版本

CUDA_VISIBLE_DEVICES=0 python train_climb_patch_swap.py \
    --config_file ./config/climb-vit-qiuxiu-patchswap-DTAR.yml

echo "--- 继续执行训练任务 ---"
python train_qiuxiu.py