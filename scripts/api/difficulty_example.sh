#!/bin/bash
# Example: API 难度评估任务
python -m datatagger.tagger.unified_tagger_api \
  --mission difficulty \
  --input_file data/difficulty_input.jsonl \
  --output_file output/difficulty_output.jsonl \
  --api_url http://localhost:8000 \
  --api_key your_api_key \
  --batch_size 8 