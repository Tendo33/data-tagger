#!/bin/bash

# 用法: ./run_formatter.sh 输入文件 输出文件 [输出格式: json/jsonl] [日志等级]
# 示例: ./run_formatter.sh examples/example.jsonl output.jsonl jsonl INFO

INPUT_FILE=data/tagged/final_tagged.json
OUTPUT_FILE=data/tagged/final_tagged_formatted.json
LOG_LEVEL=INFO

python3 datatagger/formatter/data_formatter.py \
    --input_file "$INPUT_FILE" \
    --output_file "$OUTPUT_FILE" \
    --log_level "$LOG_LEVEL"
