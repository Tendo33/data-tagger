#!/bin/bash
# Example: API 语言检测任务
python -m datatagger.tagger.unified_tagger_api \
  --mission language \
  --input_file data/language_input.jsonl \
  --output_file output/language_output.jsonl \
  --api_url http://localhost:8000 \
  --api_key your_api_key \
  --batch_size 8 