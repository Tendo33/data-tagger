#!/bin/bash
# Example: API 奖励任务
python -m datatagger.tagger.unified_tagger_api \
  --mission reward \
  --input_file data/reward_input.jsonl \
  --output_file output/reward_output.jsonl \
  --api_url http://localhost:8000 \
  --api_key your_api_key \
  --batch_size 8 