#!/bin/bash
# Example: VLLM 语言检测任务
python -m datatagger.tagger.unified_tagger_vllm \
  --mission language \
  --input_file data/language_input.jsonl \
  --output_file output/language_output.jsonl \
  --vllm_model_path models/vllm_language_model \
  --batch_size 8 