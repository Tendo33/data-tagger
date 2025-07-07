#!/bin/bash
# Example: VLLM 分类任务
python -m datatagger.tagger.unified_tagger_vllm \
  --mission classification \
  --input_file data/classification_input.jsonl \
  --output_file output/classification_output.jsonl \
  --vllm_model_path models/vllm_model \
  --batch_size 8 