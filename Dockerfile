# DGX Spark — vLLM Inference Container (UNSLOTH NVFP4)
# GPU: NVIDIA GB10 (ARM64, sm_121a) — 128GB unified memory
# 모델: unsloth/Qwen3.6-27B-NVFP4 (safetensors/NVFP4)
# 엔진: vLLM 0.25.0+ + cute-DSL + flashinfer_b12x

FROM --platform=linux/arm64 nvidia/cuda:12.5.0-cudnn8-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# ── 의존성 ─────────────────────────────────────────────────────────────
# 공식 카드는 python 3.13 venv를 사용하나, ubuntu22.04 apt에서 3.13이 기본
# 제공되지 않아 python3.11로 유지 (venv 생성 라인과 버전 일치시킬 것)
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.11 python3.11-venv python3.11-dev \
    libgl1-mesa-glx libglib2.0-0 \
    curl wget git ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# ── Python venv ────────────────────────────────────────────────────────
RUN python3.11 -m venv /opt/venv && \
    /opt/venv/bin/pip install --upgrade pip setuptools

ENV PATH="/opt/venv/bin:$PATH"

# ── vLLM + FlashInfer + cute-DSL (DGX Spark용 ARM64) ──────────────────
# CUTE_DSL_ARCH: DGX Spark GB10 = sm_121a (필수!)
ENV CUTE_DSL_ARCH=sm_121a

# uv로 설치 (--torch-backend=auto가 aarch64용 torch를 자동으로 해석함)
RUN pip install --no-cache-dir uv

RUN uv pip install --python /opt/venv/bin/python \
    "vllm>=0.25.0" \
    "flashinfer-python>=0.6.13" \
    "nvidia-cutlass-dsl>=4.5.2" \
    --torch-backend=auto

# ── 작업 디렉토리 ──────────────────────────────────────────────────────
WORKDIR /workspace

# ── 헬스 체크 스크립트 ─────────────────────────────────────────────────
COPY scripts/healthcheck.sh /workspace/healthcheck.sh
RUN chmod +x /workspace/healthcheck.sh

# ── 헬스 체크 ───────────────────────────────────────────────────────────
HEALTHCHECK --interval=30s --timeout=10s --start-period=10m --retries=3 \
    CMD /workspace/healthcheck.sh

# ── 포트 노출 (OpenAI 호환 API) ────────────────────────────────────────
EXPOSE 8000

# ── 실행 ───────────────────────────────────────────────────────────────
# 모델명, 디타입 등 파라미터는 docker-compose.yaml에서 override
CMD ["python3", "-m", "vllm.entrypoints.openai.api_server", \
     "--host", "0.0.0.0", \
     "--port", "8000"]