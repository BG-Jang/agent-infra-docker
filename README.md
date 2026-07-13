# DGX Spark vLLM Inference Container (Unsloth NVFP4)

> **DGX Spark의 128GB unified memory를 활용한 LLM 추론 전용 컨테이너**
> Hermes Agent는 호스트(OS)에서 실행하고, 추론만 컨테이너로 분리
> 엔진: vLLM 0.25.0+ + cute-DSL + flashinfer_b12x

## 아키텍처

```
┌─────────────────────────────────────────────────────┐
│              DGX Spark (128GB Unified)                │
│                                                     │
│  ┌──────────────────┐    ┌──────────────────┐       │
│  │   Hermes Agent   │    │  vLLM            │       │
│  │  (4 프로필)      │◀──▶│  Container       │       │
│  │  :8420~8423      │    │  dgx-inference   │       │
│  │  (호스트 실행)    │    │  :8000           │       │
│  └──────────────────┘    └──────────────────┘       │
│                                                     │
│  Memory:                                           │
│    ├─ 컨테이너: 110GB (GPU 메모리)                  │
│    └─ 호스트(Hermes): 18GB (여유)                   │
│                                                     │
│  GPU: NVIDIA GB10 (ARM64 sm_121a)                   │
│  CPU: 20 cores ARM64                                │
│  RAM: 128GB unified memory                          │
└─────────────────────────────────────────────────────┘
```

## 메모리 구성

| 구성 | 메모리 | 용도 |
|---|---|---|
| **컨테이너** | 110GB | vLLM + GPU 메모리 |
| **호스트(Hermes)** | 18GB | Python, Node.js, Telegram 연동 |

## 공식 권장 사항 준수 사항

| 항목 | 공식 | 우리 설정 | 상태 |
|---|---|---|---|
| **엔진** | vLLM 0.25.0+ | vLLM 0.25.0+ | ✅ |
| **Backend** | cute-DSL / CUTLASS | `CUTE_DSL_ARCH=sm_121a` | ✅ |
| **GPU 아키텍처** | `sm_121a` (GB10) | Dockerfile ENV | ✅ |
| **MTP 추론** | `--speculative-config '{"method": "mtp", "num_speculative_tokens": 2}'` | docker-compose에서 명시 | ✅ |
| **Marlin 금지** | "Marlin은 2배 느림" | marlin 백엔드 사용하지 않음 | ✅ |
| **flashinfer** | `flashinfer-python>=0.6.13` | pip 설치 | ✅ |

## 공식 실행 명령

```bash
# 1. venv 생성
uv venv unsloth-nvfp4-env --python 3.13
source unsloth-nvfp4-env/bin/activate

# 2. 의존성 설치
uv pip install "vllm>=0.25.0" \
    "flashinfer-python>=0.6.13" \
    "nvidia-cutlass-dsl>=4.5.2" \
    --torch-backend=auto

# 3. 모델 서빙 (MTP 추론)
vllm serve unsloth/Qwen3.6-27B-NVFP4 \
    --speculative-config '{"method": "mtp", "num_speculative_tokens": 2}'

# ⚠️ DGX Spark 필수 환경변수
export CUTE_DSL_ARCH=sm_121a
# ← 이것 없으면 2배 느립니다!
```

## 컨테이너 실행

```bash
cd /opt/data/agent-infra-docker
bash scripts/start.sh
```

## 모델 선택

> unsloth/Qwen3.6-27B-NVFP4는 멀티모달 모델입니다 (텍스트/이미지/영상 입력 지원).
> 기본(native) 컨텍스트는 262,144 토큰이며, YaRN 적용 시 최대 1,010,000 토큰까지 확장 가능합니다.
>
> ⚠️ 본 컨테이너는 **텍스트 전용**으로 서빙합니다. ARM64/sm_121a 환경에서 비전 인코더
> 프로파일링 시 segfault가 발생해, `docker-compose.yaml`에서
> `--limit-mm-per-prompt '{"image": 0, "video": 0}'`로 이미지/영상 입력을 비활성화했습니다.
> 멀티모달이 필요하면 해당 값을 올리고 vision 경로를 별도 검증하세요.

### supported 모델

| 모델 | 양자화 | 필요 VRAM | 작업 메모리(추정) | 가능 |
|---|---|---|---|---|
| **Qwen3.6-27B** | NVFP4 | ~12GB | 14GB | ✅ 완벽 |
| **Qwen3.6-27B** | BF16 | ~54GB | 60GB | ✅ 완벽 |
| **Qwen3.6-35B-A3B** | NVFP4 | ~14GB | 14GB | ✅ 완벽 |
| **Qwen3.6-35B-A3B** | BF16 | ~70GB | 72GB | ✅ 완벽 |

> 위 "작업 메모리(추정)" 값은 모델별 실사용 추정치이며, 컨테이너 전체 상한(`mem_limit`)인 110GB와는 별개입니다.

## 성능 벤치마크 (공식)

### 1. 추론 속도 (throughput)

| 모델 | 백엔드 | decode tok/s | throughput tok/s |
|---|---|---|---|
| nvidia 27B | marlin | 115.6 | 2,403 |
| **unsloth 27B** | **cute-DSL** | 125.9 | **6,863** |
| nvidia 35B-A3B | marlin | 240.8 | 8,721 |
| **unsloth 35B-A3B** | **cute-DSL + trtllm** | 295.2 | **15,636** |

> **Unsloth cute-DSL 백엔드가 NVIDIA Marlin 대비 2.85배 빠름!**

### 2. 정확도 (Accuracy)

| 양자화 | MMLU-Pro | GPQA | AIME 2025 |
|---|---|---|---|
| **Unsloth NVFP4** | 86.25 | 86.34 | 93.12 |
| NVIDIA NVFP4 | 85.96 | 86.87 | 93.12 |
| FP8 | 86.11 | 86.87 | 93.75 |
| BF16 | 85.96 | 88.13 | 93.33 |

### 3. MMLU-Pro 원문

> MMLU-Pro: Multi-task Measurement Language Understanding Pro (확장 버전)
> Unsloth NVFP4가 가장 높은 점수를 기록했습니다.

## 문제 해결

### Out of Memory (OOM)

```yaml
# docker-compose.yaml에서 mem_limit 줄이기
mem_limit: 80g
```

### CUDA 컴파일 오류

```bash
# Dockerfile에서 CUTE_DSL_ARCH 확인
# DGX Spark GB10 = sm_121a (필수!)
# 다른 NVIDIA GPU라면 적절한 아키텍처로 변경
```

### 컨테이너 시작 실패

```bash
# 모델 다운로드/컴파일에 최대 10분 소요
docker logs -f dgx-inference

# 10분 후에 다시 확인
curl http://localhost:8000/health
```

### Marlin 백엔드 사용 금지

공식 문서에서 명시적으로 경고합니다:
> "Also do NOT use the Marlin backend since it's 2x slower"

vLLM은 기본적으로 cute-DSL/CUTLASS를 사용합니다.

## Hermes 연결

Hermes의 `~/.env`에서 LLM 엔드포인트만 변경:

```bash
# 컨테이너가 docker network에 연결된 경우
export TDAI_LLM_BASE_URL=http://dgx-inference:8000/v1

# 로컬 루프백
export TDAI_LLM_BASE_URL=http://localhost:8000/v1
```

## 구성 파일

| 파일 | 설명 |
|---|---|
| `Dockerfile` | vLLM + cute-DSL + flashinfer ARM64 빌드 |
| `docker-compose.yaml` | 컨테이너 설정 (110GB 메모리) |
| `scripts/start.sh` | 실행 스크립트 |
| `scripts/healthcheck.sh` | 헬스 체크 |
| `models/` | 모델 파일 저장 |

## 참고 링크

- 공식 모델 카드: https://huggingface.co/unsloth/Qwen3.6-27B-NVFP4
- Unsloth NVFP4 문서: https://huggingface.co/unsloth
- vLLM 공식: https://docs.vllm.ai/
- cute-DSL 공식: https://github.com/NVIDIA/cute-dsl
- flashinfer 공식: https://flashinfer.ai/