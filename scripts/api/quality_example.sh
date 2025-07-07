#!/bin/bash
# Example: API 质量评估任务
python -m datatagger.tagger.unified_tagger_api \
  --mission quality \
  --input_file data/quality_input.jsonl \
  --output_file output/quality_output.jsonl \
  --api_url http://localhost:8000 \
  --api_key your_api_key \
  --batch_size 8
