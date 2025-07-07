#!/bin/bash
# Example: API 安全任务
python -m datatagger.tagger.unified_tagger_api \
  --mission safety \
  --input_file data/safety_input.jsonl \
  --output_file output/safety_output.jsonl \
  --api_url http://localhost:8000 \
  --api_key your_api_key \
  --batch_size 8 