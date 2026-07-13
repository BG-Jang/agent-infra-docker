#!/bin/bash
# llama.cpp GGUF 변환 스크립트
# HuggingFace 모델 → GGUF(NVFP4/FP16) 변환
set -euo pipefail

MODEL_NAME="${1:-unsloth/Qwen3.6-27B-NVFP4}"
OUTPUT_DIR="${2:-/workspace/data}"
OUTPUT="${OUTPUT_DIR}/$(echo $MODEL_NAME | sed 's|/|_|g').gguf"

mkdir -p "$OUTPUT_DIR"
cd /workspace/llama.cpp

echo "=== GGUF 변환 시작 ==="
echo "  모델: $MODEL_NAME"
echo "  출력: $OUTPUT"
echo ""

python3 convert.py \
    --model "$MODEL_NAME" \
    --output "$OUTPUT" \
    --outfile "$OUTPUT"

echo ""
echo "✅ 변환 완료: $OUTPUT"
ls -lh "$OUTPUT"