#!/bin/bash
# DGX Spark GPU Inference Container — 실행 스크립트
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

cd "$PROJECT_ROOT"

echo "🚀 DGX Spark Inference Container 시작..."
echo "   GPU: NVIDIA GB10 (ARM64 CUDA 12.5)"
echo "   엔진: llama.cpp (nvFP4/GGUF)"
echo "   API: OpenAI 호환 (:8000)"
echo ""

# 모델 디렉토리 생성
mkdir -p models

# 컨테이너 빌드 & 시작
docker compose up --build -d

echo ""
echo "⏳ 컨테이너 시작 대기 중..."
sleep 10

# 헬스 체크
echo ""
echo "📡 헬스 체크:"
curl -s http://localhost:8000/health 2>/dev/null || echo "  아직 로딩 중..."

echo ""
echo "=== 구동 완료 ==="
echo "  API: http://localhost:8000/v1/chat/completions"
echo "  Swagger: http://localhost:8000/docs"
echo ""
echo "  컨테이너 확인: docker ps"
echo "  로그 확인: docker logs -f dgx-inference"