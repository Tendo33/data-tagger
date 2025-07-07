#!/bin/bash
set -e

# ========== 环境变量与路径设置 ==========
source /root/github_projet/data-tagger/.env

INPUT_FILE="/root/github_projet/data-tagger/data/alpaca_zh_demo.json"
OUTPUT_DIR="/root/github_projet/data-tagger/data/tagged"

# API模型名称
API_MODEL_NAME="Qwen2.5-72B-Instruct-AWQ"
EMBEDDING_API_MODEL_NAME="Qwen3-Embedding-0.6B"
API_URL="$OPENAI_BASE_URL"
API_KEY="$OPENAI_API_KEY"

mkdir -p "$OUTPUT_DIR"

# 公共参数
COMMON_PARAMS="\
    --prompt_field instruction \
    --output_field output \
    --enable_thinking False \
    --batch_size 5 \
    --checkpoint_every 10 \
    --api_url $API_URL \
    --api_key $API_KEY"

echo "========== 开始运行所有tagger任务（API推理） =========="

# 1. 质量评估（QUALITY）
echo "[1/7] 质量评估..."
python -m datatagger.tagger.unified_tagger_api \
    $COMMON_PARAMS \
    --api_model_name "$API_MODEL_NAME" \
    --tag_mission QUALITY \
    --input_file "$INPUT_FILE" \
    --output_file "$OUTPUT_DIR/quality_tagged.json"

# 2. 难度评估（DIFFICULTY）
echo "[2/7] 难度评估..."
python -m datatagger.tagger.unified_tagger_api \
    $COMMON_PARAMS \
    --api_model_name "$API_MODEL_NAME" \
    --tag_mission DIFFICULTY \
    --input_file "$OUTPUT_DIR/quality_tagged.json" \
    --output_file "$OUTPUT_DIR/difficulty_tagged.json"

# 3. 分类任务（CLASSIFICATION）
echo "[3/7] 分类任务..."
python -m datatagger.tagger.unified_tagger_api \
    $COMMON_PARAMS \
    --api_model_name "$API_MODEL_NAME" \
    --tag_mission CLASSIFICATION \
    --input_file "$OUTPUT_DIR/difficulty_tagged.json" \
    --output_file "$OUTPUT_DIR/classification_tagged.json"

# 4. 安全性评估（SAFETY）
# API模式暂不支持SAFETY任务，如需本地推理请使用VLLM脚本
# echo "[4/7] 安全性评估..."
# python -m datatagger.tagger.unified_tagger_api \
#     $COMMON_PARAMS \
#     --api_model_name "$API_MODEL_NAME" \
#     --tag_mission SAFETY \
#     --input_file "$OUTPUT_DIR/classification_tagged.json" \
#     --output_file "$OUTPUT_DIR/safety_tagged.json"

# 5. 奖励评分（REWARD）
# API模式暂不支持REWARD任务，如需本地推理请使用VLLM脚本
# echo "[5/7] 奖励评分..."
# python -m datatagger.tagger.unified_tagger_api \
#     $COMMON_PARAMS \
#     --api_model_name "$API_MODEL_NAME" \
#     --tag_mission REWARD \
#     --input_file "$OUTPUT_DIR/safety_tagged.json" \
#     --output_file "$OUTPUT_DIR/reward_tagged.json"

# 6. 语言识别（LANGUAGE）
echo "[6/7] 语言识别..."
python -m datatagger.tagger.unified_tagger_api \
    $COMMON_PARAMS \
    --api_model_name "$API_MODEL_NAME" \
    --tag_mission LANGUAGE \
    --input_file "$OUTPUT_DIR/classification_tagged.json" \
    --output_file "$OUTPUT_DIR/language_tagged.json"

# 7. 嵌入向量（EMBEDDING）
echo "[7/7] 嵌入向量..."
python -m datatagger.tagger.unified_tagger_api \
    $COMMON_PARAMS \
    --api_model_name "$EMBEDDING_API_MODEL_NAME" \
    --tag_mission EMBEDDING \
    --faiss_store_embeddings True \
    --dimension 1024 \
    --input_file "$OUTPUT_DIR/language_tagged.json" \
    --output_file "$OUTPUT_DIR/final_tagged.json"

echo "========== 所有tagger任务完成！ =========="
echo "最终输出文件: $OUTPUT_DIR/final_tagged.json" 