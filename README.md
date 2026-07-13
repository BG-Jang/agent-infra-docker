# DGX Spark Inference Container — llama.cpp + GGUF

> **DGX Spark의 128GB unified memory를 활용한 LLM 추론 전용 컨테이너**
> Hermes Agent는 호스트(OS)에서 실행하고, 추론만 컨테이너로 분리

## 아키텍처

```
┌─────────────────────────────────────────────────────┐
│              DGX Spark (128GB Unified)                │
│                                                     │
│  ┌──────────────────┐    ┌──────────────────┐       │
│  │   Hermes Agent   │    │  Inference       │       │
│  │  (4 프로필)      │◀──▶│  Container       │       │
│  │  :8420~8423      │    │  dgx-inference   │       │
│  │  (호스트 실행)    │    │  :8000           │       │
│  └──────────────────┘    └──────────────────┘       │
│                                                     │
│  Memory:                                           │
│    ├─ 컨테이너: 110GB (GPU 메모리)                  │
│    └─ 호스트(Hermes): 18GB (여유)                   │
│                                                     │
│  GPU: NVIDIA GB10 (ARM64 CUDA 12.5)                 │
│  CPU: 20 cores ARM64                                │
│  RAM: 128GB unified memory                          │
└─────────────────────────────────────────────────────┘
```

## 메모리 구성

| 구성 | 메모리 | 용도 |
|---|---|---|
| **컨테이너** | 110GB | llama.cpp + GPU 메모리 |
| **호스트(Hermes)** | 18GB | Python, Node.js, Telegram 연동 |

## 지원 모델 (128GB 기준)

| 모델 | 양자화 | 필요 메모리 | 가능 |
|---|---|---|---|
| **Qwen3.6-27B** | BF16 | ~54GB | ✅ 완벽 |
| **Qwen3.6-27B** | INT8 | ~28GB | ✅ 여유 |
| **Qwen3.6-27B** | INT4 | ~16GB | ✅ 여유 |
| **Qwen3.6-35B-A3B** | BF16 | ~70GB | ✅ 완벽 |
| **Qwen3.6-35B-A3B** | INT8 | ~36GB | ✅ 여유 |
| **Qwen3.6-35B-A3B** | INT4 | ~20GB | ✅ 여유 |
| **Qwen3.6-72B** | INT8 | ~72GB | ✅ 완벽 |
| **Qwen3.6-72B** | INT4 | ~40GB | ✅ 여유 |
| **Llama-4-Scout** | BF16 | ~24GB | ✅ 완벽 |
| **Llama-4-Maverick** | BF16 | ~120GB | ⚠️ 임계 |

## 구성 파일

| 파일 | 설명 |
|---|---|
| `Dockerfile` | llama.cpp ARM64 + CUDA 12.5 빌드 |
| `docker-compose.yaml` | 컨테이너 설정 (110GB 메모리) |
| `scripts/start.sh` | 실행 스크립트 |
| `scripts/convert.sh` | HuggingFace → GGUF 변환 |
| `scripts/healthcheck.sh` | 헬스 체크 |
| `models/` | GGUF 모델 파일 저장 |

## 실행

```bash
cd /opt/data/agent-infra-docker
bash scripts/start.sh
```

## 모델 선택

### GGUF 변환 (처음 한 번)

```bash
# 27B 모델
bash scripts/convert.sh unsloth/Qwen3.6-27B-NVFP4

# 35B-A3B 모델
bash scripts/convert.sh unsloth/Qwen3.6-35B-A3B-NVFP4

# 72B 모델 (128GB면 완벽)
bash scripts/convert.sh unsloth/Qwen3.6-72B-NVFP4
```

### 컨테이너 실행

```bash
# 기본 설정 (35B-A3B 최적화)
docker compose up -d

# 27B 모델 (더 빠름, 128K 컨텍스트)
CONTEXT_SIZE=131072 BATCH_SIZE=32768 docker compose up -d

# 72B 모델 (더 큰 컨텍스트 필요)
CONTEXT_SIZE=32768 BATCH_SIZE=8192 docker compose up -d
```

## 문제 해결

### Out of Memory (OOM)

```yaml
# docker-compose.yaml에서 mem_limit 줄이기
mem_limit: 80g
```

### GPU 메모리 부족

```bash
# 컨텍스트 크기 줄이기
CONTEXT_SIZE=8192 docker compose up -d
```

### CUDA 오류

```bash
# NVIDIA Container Toolkit 재설치
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
```

## Hermes 연결

Hermes의 `~/.env`에서 LLM 엔드포인트만 변경:

```bash
# 컨테이너가 docker network에 연결된 경우
export TDAI_LLM_BASE_URL=http://dgx-inference:8000/v1

# 로컬 루프백
export TDAI_LLM_BASE_URL=http://localhost:8000/v1
```