#!/bin/bash

# 用法: ./run_formatter.sh 输入文件 输出文件 [输出格式: json/jsonl] [日志等级]
# 示例: ./run_formatter.sh examples/example.jsonl output.jsonl jsonl INFO

INPUT_FILE=${1:-data/alpaca_zh_demo.json}
OUTPUT_FILE=${2:-data/alpaca_zh_demo_formatted.json}
SAVE_AS=${3:-jsonl}
LOG_LEVEL=${4:-INFO}
mkdir -p data/formatted

python3 datatagger/formatter/data_formatter.py \
    --input-file "$INPUT_FILE" \
    --output-file "$OUTPUT_FILE" \
    --save-as "$SAVE_AS" \
    --log-level "$LOG_LEVEL"
