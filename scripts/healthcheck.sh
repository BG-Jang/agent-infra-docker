#!/bin/bash
# 헬스체크 스크립트
# llama-server가 구동 중인지 확인
set -euo pipefail

if curl -s http://localhost:8000/health >/dev/null 2>&1; then
    echo '{"status":"ok"}'
    exit 0
else
    # 프로세스가 살아있는지 확인
    if pgrep -f llama-server >/dev/null 2>&1; then
        echo '{"status":"loading","pid":'$(pgrep -f llama-server)'}'
        exit 0
    else
        echo '{"status":"unhealthy"}'
        exit 1
    fi
fi