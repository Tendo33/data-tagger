#!/bin/bash
set -e

# ========== Environment variables and path settings ==========
source /root/github_projet/data-tagger/.env

INPUT_FILE="/root/github_projet/data-tagger/data/alpaca_zh_demo.json"
OUTPUT_DIR="/root/github_projet/data-tagger/data/tagged"

# API model names
API_MODEL_NAME="$CHAT_MODEL_NAME"
EMBEDDING_MODEL_NAME="$EMBEDDING_MODEL_NAME"
API_URL="$OPENAI_BASE_URL"
API_KEY="$OPENAI_API_KEY"

mkdir -p "$OUTPUT_DIR"

# Common parameters
COMMON_PARAMS="\
    --prompt_field instruction \
    --output_field output \
    --enable_thinking False \
    --batch_size 5 \
    --checkpoint_every 10 \
    --api_url $API_URL \
    --api_key $API_KEY"

echo "========== Starting all tagger tasks (API inference) =========="

# 1. Quality Evaluation (QUALITY)
echo "[1/7] Quality Evaluation..."
python -m datatagger.tagger.unified_tagger_api \
    $COMMON_PARAMS \
    --api_model_name "$API_MODEL_NAME" \
    --tag_mission QUALITY \
    --input_file "$INPUT_FILE" \
    --output_file "$OUTPUT_DIR/quality_tagged.json"

# 2. Difficulty Evaluation (DIFFICULTY)
echo "[2/7] Difficulty Evaluation..."
python -m datatagger.tagger.unified_tagger_api \
    $COMMON_PARAMS \
    --api_model_name "$API_MODEL_NAME" \
    --tag_mission DIFFICULTY \
    --input_file "$OUTPUT_DIR/quality_tagged.json" \
    --output_file "$OUTPUT_DIR/difficulty_tagged.json"

# 3. Classification Task (CLASSIFICATION)
echo "[3/7] Classification Task..."
python -m datatagger.tagger.unified_tagger_api \
    $COMMON_PARAMS \
    --api_model_name "$API_MODEL_NAME" \
    --tag_mission CLASSIFICATION \
    --input_file "$OUTPUT_DIR/difficulty_tagged.json" \
    --output_file "$OUTPUT_DIR/classification_tagged.json"

# 4. Language Detection (LANGUAGE)
echo "[4/7] Language Detection..."
python -m datatagger.tagger.unified_tagger_api \
    $COMMON_PARAMS \
    --api_model_name "$API_MODEL_NAME" \
    --tag_mission LANGUAGE \
    --input_file "$OUTPUT_DIR/classification_tagged.json" \
    --output_file "$OUTPUT_DIR/language_tagged.json"

# 5. Embedding Vector (EMBEDDING)
echo "[5/7] Embedding Vector..."
python -m datatagger.tagger.unified_tagger_api \
    $COMMON_PARAMS \
    --api_model_name "$EMBEDDING_MODEL_NAME" \
    --tag_mission EMBEDDING \
    --faiss_store_embeddings True \
    --dimension 1024 \
    --input_file "$OUTPUT_DIR/language_tagged.json" \
    --output_file "$OUTPUT_DIR/final_tagged.json"

echo "========== All tagger tasks completed! =========="
echo "Final output file: $OUTPUT_DIR/final_tagged.json" 