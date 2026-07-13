# DGX Spark — Inference Container (llama.cpp + GGUF)
# GPU: NVIDIA GB10 (16GB unified memory, ARM64)
# RAM: 128GB unified memory → 컨테이너 mem_limit=100g
# 모델: Qwen3.6-27B-NVFP4 또는 35B-A3B-NVFP4

FROM nvidia/cuda:12.5.0-base-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# ── 의존성 ─────────────────────────────────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential git cmake python3 python3-pip wget curl \
    && rm -rf /var/lib/apt/lists/*

# ── llama.cpp 빌드 ──────────────────────────────────────────────────────
WORKDIR /workspace

# llama.cpp 소스 클론 및 빌드 (CUDA 12.5 + ARM64 최적화)
RUN git clone --depth 1 --branch v1.0.0 https://github.com/ggml-org/llama.cpp.git /workspace/llama.cpp && \
    cd /workspace/llama.cpp && \
    cmake -DGGML_CUDA=ON -DGGML_FP16=ON -DGGML_AVX512=OFF -DGGML_AVX2=OFF \
          -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=/usr/local \
          -DCMAKE_CUDA_ARCHITECTURES=110 \
          .. && \
    make -j$(nproc) && \
    make install-strip

# ── Python 도구 (GGUF 변환용) ──────────────────────────────────────────
RUN pip3 install --upgrade pip && \
    pip3 install --no-cache-dir \
        huggingface-hub \
        transformers \
        torch --index-url https://download.pytorch.org/whl/cu121 \
        safetensors \
        sentencepiece \
        protobuf \
        pyyaml

# ── 헬스 체크 스크립트 ─────────────────────────────────────────────────
COPY scripts/healthcheck.sh /workspace/healthcheck.sh
RUN chmod +x /workspace/healthcheck.sh

WORKDIR /workspace/data

# ── 헬스 체크 ───────────────────────────────────────────────────────────
HEALTHCHECK --interval=30s --timeout=10s --start-period=5m --retries=3 \
    CMD /workspace/healthcheck.sh

# ── 포트 노출 ──────────────────────────────────────────────────────────
EXPOSE 8000

# ── 실행 ───────────────────────────────────────────────────────────────
CMD ["llama-server", \
     "--host", "0.0.0.0", \
     "--port", "8000"]