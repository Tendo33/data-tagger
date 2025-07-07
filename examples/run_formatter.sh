#!/bin/bash


INPUT_FILE=data/tagged/final_tagged.json
OUTPUT_FILE=data/tagged/final_tagged_formatted.json
LOG_LEVEL=INFO

python3 datatagger/formatter/data_formatter.py \
    --input_file "$INPUT_FILE" \
    --output_file "$OUTPUT_FILE" \
    --log_level "$LOG_LEVEL" \
    --prompt_field instruction \
    --output_field output 
