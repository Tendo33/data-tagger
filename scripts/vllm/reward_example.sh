#!/bin/bash
# Example: VLLM 奖励任务
python -m datatagger.tagger.unified_tagger_vllm \
  --mission reward \
  --input_file data/reward_input.jsonl \
  --output_file output/reward_output.jsonl \
  --vllm_model_path models/vllm_reward_model \
  --batch_size 8 