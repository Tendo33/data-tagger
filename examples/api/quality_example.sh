#!/bin/bash
# Example: API 质量评估任务
source /root/github_projet/data-tagger/.env

API_MODEL_NAME="$CHAT_MODEL_NAME"
API_URL="$OPENAI_BASE_URL"
API_KEY="$OPENAI_API_KEY"

python -m datatagger.tagger.unified_tagger_api \
  --mission QUALITY \
  --input_file data/alpaca_zh_demo.json \
  --output_file data/tagged/alpaca_zh_demo_quality.jsonl \
  --api_url $API_URL \
  --api_key $API_KEY \
  --batch_size 8 \
  --api_model_name $API_MODEL_NAME
