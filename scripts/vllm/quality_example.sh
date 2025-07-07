#!/bin/bash
# Example: VLLM 质量评估任务
python -m datatagger.tagger.unified_tagger_vllm \
  --mission quality \
  --input_file data/quality_input.jsonl \
  --output_file output/quality_output.jsonl \
  --vllm_model_path models/vllm_quality_model \
  --batch_size 8 