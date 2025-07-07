#!/bin/bash
# Example: VLLM 安全任务
python -m datatagger.tagger.unified_tagger_vllm \
  --mission safety \
  --input_file data/safety_input.jsonl \
  --output_file output/safety_output.jsonl \
  --vllm_model_path models/vllm_safety_model \
  --batch_size 8 