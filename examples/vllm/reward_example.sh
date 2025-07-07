#!/bin/bash
# Example: VLLM 奖励任务
python -m datatagger.tagger.unified_tagger_vllm \
  --mission reward \
  --input_file data/reward_input.jsonl \
  --output_file output/reward_output.jsonl \
  --vllm_model_path models/vllm_reward_model \
  --batch_size 8 \
  --device 0 \
  --tensor_parallel_size 1 \
  --gpu_memory_utilization 0.3 \
  --prompt_field instruction \
  --output_field output \
  --enable_thinking False \
  --checkpoint_every 10 \
  --save_as jsonl \
  --log_level INFO