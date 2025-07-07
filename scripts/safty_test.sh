#!/bin/bash
set -e

# ========== 环境变量与路径设置 ==========
# 如需API推理版本，请参考 scripts/run_all_taggers_api.sh
# 如需自定义输入输出路径、模型路径等，请修改下方变量
export CUDA_VISIBLE_DEVICES=1
source /root/github_projet/data-tagger/.env

INPUT_FILE="/root/github_projet/data-tagger/data/alpaca_zh_demo.json"
OUTPUT_DIR="/root/github_projet/data-tagger/data/tagged"
SAFETY_MODEL_PATH="/mnt/public/sunjinfeng/base_llms/hub/AI-ModelScope/Llama-Guard-3-8B"

mkdir -p "$OUTPUT_DIR"

# 公共参数
COMMON_PARAMS="\
    --prompt_field instruction \
    --output_field output \
    --enable_thinking False \
    --batch_size 5 \
    --checkpoint_every 10 \
    --device 0 \
    --tensor_parallel_size 1 \
    --gpu_memory_utilization 0.3"

echo "========== 开始安全性评估任务（SAFETY，本地推理） =========="

python -m datatagger.tagger.unified_tagger_vllm \
    $COMMON_PARAMS \
    --vllm_model_path "$SAFETY_MODEL_PATH" \
    --tag_mission SAFETY \
    --input_file "$INPUT_FILE" \
    --output_file "$OUTPUT_DIR/safety_tagged.json"

echo "========== 安全性评估任务完成！ =========="
echo "输出文件: $OUTPUT_DIR/safety_tagged.json"