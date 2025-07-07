#!/bin/bash
# Example: API 嵌入任务
python -m datatagger.tagger.unified_tagger_api \
  --mission embedding \
  --input_file data/embedding_input.jsonl \
  --output_file output/embedding_output.jsonl \
  --api_url http://localhost:8000 \
  --api_key your_api_key \
  --batch_size 8 