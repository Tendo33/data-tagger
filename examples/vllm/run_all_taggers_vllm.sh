#!/bin/bash
set -e

# ========== Environment variables and path settings ==========
export CUDA_VISIBLE_DEVICES=1
source /root/github_projet/data-tagger/.env

INPUT_FILE="/root/github_projet/data-tagger/data/alpaca_zh_demo.json"
OUTPUT_DIR="/root/github_projet/data-tagger/data/tagged"

# Model paths (local inference)
LLM_MODEL_PATH="/mnt/public/sunjinfeng/base_llms/hub/AI-ModelScope/Qwen3-8B"
REWARD_MODEL_PATH="/mnt/public/sunjinfeng/base_llms/hub/AI-ModelScope/Skywork-Reward-V2-Llama-3.1-8B"
EMBEDDING_MODEL_PATH="/mnt/public/sunjinfeng/base_llms/hub/AI-ModelScope/Qwen3-Embedding-4B"
SAFETY_MODEL_PATH="/mnt/public/sunjinfeng/base_llms/hub/AI-ModelScope/Llama-Guard-3-8B"

mkdir -p "$OUTPUT_DIR"

# Common parameters
COMMON_PARAMS="\
    --prompt_field instruction \
    --output_field output \
    --enable_thinking False \
    --batch_size 5 \
    --checkpoint_every 10 \
    --device 0 \
    --tensor_parallel_size 1 \
    --gpu_memory_utilization 0.23"

echo "========== Starting all tagger tasks (VLLM local inference) =========="

# 1. Quality Evaluation (QUALITY)
echo "[1/7] Quality Evaluation..."
python -m datatagger.tagger.unified_tagger_vllm \
    $COMMON_PARAMS \
    --vllm_model_path "$LLM_MODEL_PATH" \
    --tag_mission QUALITY \
    --input_file "$INPUT_FILE" \
    --output_file "$OUTPUT_DIR/quality_tagged.json"

# 2. Difficulty Evaluation (DIFFICULTY)
echo "[2/7] Difficulty Evaluation..."
python -m datatagger.tagger.unified_tagger_vllm \
    $COMMON_PARAMS \
    --vllm_model_path "$LLM_MODEL_PATH" \
    --tag_mission DIFFICULTY \
    --input_file "$OUTPUT_DIR/quality_tagged.json" \
    --output_file "$OUTPUT_DIR/difficulty_tagged.json"

# 3. Classification Task (CLASSIFICATION)
echo "[3/7] Classification Task..."
python -m datatagger.tagger.unified_tagger_vllm \
    $COMMON_PARAMS \
    --vllm_model_path "$LLM_MODEL_PATH" \
    --tag_mission CLASSIFICATION \
    --input_file "$OUTPUT_DIR/difficulty_tagged.json" \
    --output_file "$OUTPUT_DIR/classification_tagged.json"

# 4. Safety Evaluation (SAFETY)
echo "[4/7] Safety Evaluation..."
python -m datatagger.tagger.unified_tagger_vllm \
    $COMMON_PARAMS \
    --vllm_model_path "$SAFETY_MODEL_PATH" \
    --tag_mission SAFETY \
    --input_file "$OUTPUT_DIR/classification_tagged.json" \
    --output_file "$OUTPUT_DIR/safety_tagged.json"

# 5. Reward Scoring (REWARD)
echo "[5/7] Reward Scoring..."
python -m datatagger.tagger.unified_tagger_vllm \
    $COMMON_PARAMS \
    --vllm_model_path "$REWARD_MODEL_PATH" \
    --tag_mission REWARD \
    --input_file "$OUTPUT_DIR/safety_tagged.json" \
    --output_file "$OUTPUT_DIR/reward_tagged.json"

# 6. Language Detection (LANGUAGE)
echo "[6/7] Language Detection..."
python -m datatagger.tagger.unified_tagger_vllm \
    $COMMON_PARAMS \
    --vllm_model_path "$LLM_MODEL_PATH" \
    --tag_mission LANGUAGE \
    --input_file "$OUTPUT_DIR/reward_tagged.json" \
    --output_file "$OUTPUT_DIR/language_tagged.json"

# 7. Embedding Vector (EMBEDDING)
echo "[7/7] Embedding Vector..."
python -m datatagger.tagger.unified_tagger_vllm \
    $COMMON_PARAMS \
    --vllm_model_path "$EMBEDDING_MODEL_PATH" \
    --tag_mission EMBEDDING \
    --dimension 2560 \
    --faiss_store_embeddings True \
    --input_file "$OUTPUT_DIR/language_tagged.json" \
    --output_file "$OUTPUT_DIR/final_tagged.json"

# 8. Format Data
echo "[8/8] Format Data..."
python -m datatagger.formatter.data_formatter \
    --input_file "$OUTPUT_DIR/final_tagged.json" \
    --output_file "$OUTPUT_DIR/final_tagged_formatted.json"

echo "========== All tagger tasks completed! =========="
echo "Final output file: $OUTPUT_DIR/final_tagged.json"