#!/bin/bash
# 헬스체크 스크립트
# vLLM API가 구동 중인지 확인
set -euo pipefail

if curl -s http://localhost:8000/health >/dev/null 2>&1; then
    echo '{"status":"ok"}'
    exit 0
else
    # 프로세스가 살아있는지 확인
    if pgrep -f "vllm.entrypoints.openai.api_server" >/dev/null 2>&1; then
        echo '{"status":"loading"}'
        exit 0
    else
        echo '{"status":"unhealthy"}'
        exit 1
    fi
fi