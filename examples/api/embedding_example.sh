#!/bin/bash
# Example: API 嵌入任务
source /root/github_projet/data-tagger/.env

API_MODEL_NAME="$CHAT_MODEL_NAME"
EMBEDDING_MODEL_NAME="$EMBEDDING_MODEL_NAME"
API_URL="$OPENAI_BASE_URL"
API_KEY="$OPENAI_API_KEY"
python -m datatagger.tagger.unified_tagger_api \
  --mission EMBEDDING \
  --input_file data/alpaca_zh_demo.json \
  --output_file data/tagged/alpaca_zh_demo_embedding.jsonl \
  --api_url $API_URL \
  --api_key $API_KEY \
  --batch_size 8 \
  --api_model_name $API_MODEL_NAME \
  --embedding_api_model_name $EMBEDDING_MODEL_NAME