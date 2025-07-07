#!/bin/bash
# Example: API 语言检测任务
python -m datatagger.tagger.unified_tagger_api \
  --mission LANGUAGE \
  --input_file data/alpaca_zh_demo.json \
  --output_file data/tagged/alpaca_zh_demo_language.jsonl \
  --batch_size 8 