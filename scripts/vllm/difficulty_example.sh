#!/bin/bash
# Example: VLLM 难度评估任务
python -m datatagger.tagger.unified_tagger_vllm \
  --mission difficulty \
  --input_file data/difficulty_input.jsonl \
  --output_file output/difficulty_output.jsonl \
  --vllm_model_path models/vllm_difficulty_model \
  --batch_size 8 