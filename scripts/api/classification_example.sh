#!/bin/bash
# Example: API 分类任务
python -m datatagger.tagger.unified_tagger_api \
  --mission classification \
  --input_file data/classification_input.jsonl \
  --output_file output/classification_output.jsonl \
  --api_url http://localhost:8000 \
  --api_key your_api_key \
  --batch_size 8 