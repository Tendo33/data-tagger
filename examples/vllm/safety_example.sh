#!/bin/bash
# Example: VLLM 安全任务
python -m datatagger.tagger.unified_tagger_vllm \
  --mission SAFETY \
  --input_file data/alpaca_zh_demo.json \
  --output_file data/tagged/alpaca_zh_demo_safety.jsonl \
  --vllm_model_path /mnt/public/sunjinfeng/base_llms/hub/AI-ModelScope/Qwen3-8B \
  --batch_size 8 \
  --device 0 \
  --tensor_parallel_size 1 \
  --gpu_memory_utilization 0.3 \
  --prompt_field instruction \
  --output_field output \
  --enable_thinking False \
  --checkpoint_every 10 \
  --log_level INFO