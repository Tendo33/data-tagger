#!/bin/bash
set -e

# ========== 环境变量与路径设置 ==========
export CUDA_VISIBLE_DEVICES=1
source /root/github_projet/data-tagger/.env

INPUT_FILE="/root/github_projet/data-tagger/data/alpaca_zh_demo.json"
OUTPUT_DIR="/root/github_projet/data-tagger/data/tagged"

# 模型路径（本地推理）
LLM_MODEL_PATH="/mnt/public/sunjinfeng/base_llms/hub/AI-ModelScope/Qwen3-8B"
REWARD_MODEL_PATH="/mnt/public/sunjinfeng/base_llms/hub/AI-ModelScope/Skywork-Reward-Llama-3.1-8B-v0.2"
EMBEDDING_MODEL_PATH="/mnt/public/sunjinfeng/base_llms/hub/AI-ModelScope/Qwen3-Embedding-4B"
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

echo "========== 开始运行所有tagger任务（VLLM本地推理） =========="

# 1. 质量评估（QUALITY）
echo "[1/7] 质量评估..."
python -m datatagger.tagger.unified_tagger_vllm \
    $COMMON_PARAMS \
    --vllm_model_path "$LLM_MODEL_PATH" \
    --tag_mission QUALITY \
    --input_file "$INPUT_FILE" \
    --output_file "$OUTPUT_DIR/quality_tagged.json"

# 2. 难度评估（DIFFICULTY）
echo "[2/7] 难度评估..."
python -m datatagger.tagger.unified_tagger_vllm \
    $COMMON_PARAMS \
    --vllm_model_path "$LLM_MODEL_PATH" \
    --tag_mission DIFFICULTY \
    --input_file "$OUTPUT_DIR/quality_tagged.json" \
    --output_file "$OUTPUT_DIR/difficulty_tagged.json"

# 3. 分类任务（CLASSIFICATION）
echo "[3/7] 分类任务..."
python -m datatagger.tagger.unified_tagger_vllm \
    $COMMON_PARAMS \
    --vllm_model_path "$LLM_MODEL_PATH" \
    --tag_mission CLASSIFICATION \
    --input_file "$OUTPUT_DIR/difficulty_tagged.json" \
    --output_file "$OUTPUT_DIR/classification_tagged.json"

# 4. 安全性评估（SAFETY）
echo "[4/7] 安全性评估..."
python -m datatagger.tagger.unified_tagger_vllm \
    $COMMON_PARAMS \
    --vllm_model_path "$SAFETY_MODEL_PATH" \
    --tag_mission SAFETY \
    --input_file "$OUTPUT_DIR/classification_tagged.json" \
    --output_file "$OUTPUT_DIR/safety_tagged.json"

# 5. 奖励评分（REWARD）
echo "[5/7] 奖励评分..."
python -m datatagger.tagger.unified_tagger_vllm \
    $COMMON_PARAMS \
    --vllm_model_path "$REWARD_MODEL_PATH" \
    --tag_mission REWARD \
    --input_file "$OUTPUT_DIR/safety_tagged.json" \
    --output_file "$OUTPUT_DIR/reward_tagged.json"

# 6. 语言识别（LANGUAGE）
echo "[6/7] 语言识别..."
python -m datatagger.tagger.unified_tagger_vllm \
    $COMMON_PARAMS \
    --vllm_model_path "$LLM_MODEL_PATH" \
    --tag_mission LANGUAGE \
    --input_file "$OUTPUT_DIR/reward_tagged.json" \
    --output_file "$OUTPUT_DIR/language_tagged.json"

# 7. 嵌入向量（EMBEDDING）
echo "[7/7] 嵌入向量..."
python -m datatagger.tagger.unified_tagger_vllm \
    $COMMON_PARAMS \
    --vllm_model_path "$EMBEDDING_MODEL_PATH" \
    --tag_mission EMBEDDING \
    --dimension 2560 \
    --faiss_store_embeddings True \
    --input_file "$OUTPUT_DIR/language_tagged.json" \
    --output_file "$OUTPUT_DIR/final_tagged.json"

# 8. 格式化数据
echo "[8/8] 格式化数据..."
python -m datatagger.formatter.data_formatter \
    --input_file "$OUTPUT_DIR/final_tagged.json" \
    --output_file "$OUTPUT_DIR/final_tagged_formatted.json"

echo "========== 所有tagger任务完成！ =========="
echo "最终输出文件: $OUTPUT_DIR/final_tagged.json"