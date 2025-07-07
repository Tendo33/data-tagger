#!/bin/bash
# Example: VLLM 嵌入任务
python -m datatagger.tagger.unified_tagger_vllm \
  --mission embedding \
  --input_file data/embedding_input.jsonl \
  --output_file output/embedding_output.jsonl \
  --vllm_model_path models/vllm_embedding_model \
  --batch_size 8 