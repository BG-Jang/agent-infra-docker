#!/bin/bash
# DGX Spark Inference Container — 실행 스크립트 (vLLM + NVFP4)
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

cd "$PROJECT_ROOT"

echo "🚀 DGX Spark vLLM Inference Container 시작..."
echo "   GPU: NVIDIA GB10 (ARM64 sm_121a)"
echo "   엔진: vLLM 0.8.5 + cute-DSL + flashinfer_b12x"
echo "   모델: unsloth/Qwen3.6-27B-NVFP4 (NVFP4 양자화)"
echo "   API: OpenAI 호환 (:8000)"
echo "   MTP: Multi-Token Prediction (num_speculative_tokens=2)"
echo ""

# 컨테이너 빌드 & 시작
docker compose up --build -d

echo ""
echo "⏳ 컨테이너 시작 대기 중..."
echo "   (최초 모델 다운로드 & 컴파일에 최대 10분 소요)"
echo ""

# 헬스 체크 (10분 대기)
for i in $(seq 1 40); do
    sleep 15
    if curl -s http://localhost:8000/health >/dev/null 2>&1; then
        echo ""
        echo "✅ vLLM API 시작 완료!"
        break
    fi
    echo "   대기 중... ($i/40)"
    if [ $i -eq 40 ]; then
        echo "   ⚠️  vLLM이 10분 이내에 시작되지 않았습니다."
        echo "   docker compose logs -f 로 확인하세요."
    fi
done

echo ""
echo "=== 구동 완료 ==="
echo "  API: http://localhost:8000/v1/chat/completions"
echo "  Health: http://localhost:8000/health"
echo ""
echo "  컨테이너 확인: docker ps"
echo "  로그 확인: docker logs -f dgx-inference"
echo ""
echo "  테스트 명령:"
echo '  curl http://localhost:8000/v1/chat/completions \\'
echo '    -H "Content-Type: application/json" \\'
echo '    -d "{\"model\":\"unsloth/Qwen3.6-27B-NVFP4\",\"messages\":[{\"role\":\"user\",\"content\":\"hello\"}]}"'