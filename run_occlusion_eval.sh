#!/bin/bash

# 遮挡鲁棒性评估脚本
# 用法: bash run_occlusion_eval.sh

# 设置CUDA设备
export CUDA_VISIBLE_DEVICES=0

# 配置文件路径
CONFIG="configs/speedplus_v2_diffpose.yml"

# 最佳模型路径（请根据实际情况修改）
MODEL_PATH="exp/speedplus_v2_diffpose_uvxyz_gt/best_model.pth"

# 实验输出目录
EXP_DIR="exp"
DOC_NAME="occlusion_eval_$(date +%Y%m%d_%H%M%S)"

# 遮挡比例列表
OCCLUSION_RATIOS="0.0 0.2 0.4 0.6 0.8"

echo "================================================"
echo "遮挡鲁棒性评估"
echo "================================================"
echo "配置文件: $CONFIG"
echo "模型路径: $MODEL_PATH"
echo "遮挡比例: $OCCLUSION_RATIOS"
echo "输出目录: $EXP_DIR/$DOC_NAME"
echo "================================================"
echo ""

# 运行评估
python evaluate_occlusion_robustness.py \
    --config $CONFIG \
    --model_path $MODEL_PATH \
    --occlusion_ratios $OCCLUSION_RATIOS \
    --doc $DOC_NAME \
    --exp $EXP_DIR \
    --skip_type uniform \
    --eta 0.0

echo ""
echo "================================================"
echo "评估完成！结果保存在: $EXP_DIR/$DOC_NAME/"
echo "================================================"
