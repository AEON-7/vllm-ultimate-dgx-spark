# Ultimate vLLM image for DGX Spark (GB10 / sm_121a)
# - Base: ghcr.io/aeon-7/aeon-gemma-4-26b-a4b-dflash:latest
#         (already has PyTorch 2.11.0+cu130, transformers, modelopt, flashinfer)
# - Headline: PR #44389 (Triton NVFP4 KV cache, 3x KV capacity, commit e8c77b85)
# - AEON sm_121a patches (idempotent; no-op when upstream merged equivalent)
# - TurboQuant AEON-7 fork (CUDA-graph-safe QJL powers)
# - DFlash speculative decoding (already in vLLM main via --speculative-config)

FROM ghcr.io/aeon-7/aeon-gemma-4-26b-a4b-dflash:latest AS base

# Use bash so we get `set -o pipefail` etc.
SHELL ["/bin/bash", "-o", "pipefail", "-c"]

# Spark sm_121a target arch
ENV TORCH_CUDA_ARCH_LIST="12.1a"
# Don't build SM100-only kernels that fail on SM121
ENV ENABLE_NVFP4_SM100=0
ENV CCACHE_DISABLE=1
ENV CMAKE_BUILD_PARALLEL_LEVEL=8
ENV MAX_JOBS=12
ENV NVCC_THREADS=2
# Skip setuptools-scm's git-based version detection (we COPY without .git)
ENV SETUPTOOLS_SCM_PRETEND_VERSION="0.22.1+pr44389.aeon"
ENV SETUPTOOLS_SCM_PRETEND_VERSION_FOR_VLLM="0.22.1+pr44389.aeon"

WORKDIR /build

# Install git — needed by CMake's FetchContent for cutlass clone during compile
# Install full CUDA toolkit dev headers — vLLM source build needs cusparse, cublas, etc.
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        git ca-certificates \
        cuda-nvrtc-dev-13-0 \
        libcusparse-dev-13-0 \
        libcublas-dev-13-0 \
        libcusolver-dev-13-0 \
        libcufft-dev-13-0 \
        libcurand-dev-13-0 \
        libnvjitlink-dev-13-0 && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# Sanity: create libnvrtc.so symlink if dev package didn't (some CUDA versions miss it)
RUN CUDA_LIB=/usr/local/cuda-13.0/targets/sbsa-linux/lib && \
    if [ ! -f $CUDA_LIB/libnvrtc.so ] && [ -f $CUDA_LIB/libnvrtc.so.13.0.88 ]; then \
        ln -sf $CUDA_LIB/libnvrtc.so.13.0.88 $CUDA_LIB/libnvrtc.so; \
        echo "Created libnvrtc.so symlink"; \
    fi && \
    ls -la $CUDA_LIB/libnvrtc.so* | head -5

# Bring in pre-cloned vLLM PR #44389 + patches via COPY
COPY patches/ /aeon-patches/
COPY vllm-src/ /build/vllm-src/

# Remove existing vllm install
RUN pip uninstall -y vllm 2>&1 | tail -3

# Build vLLM from source with sm_121a
WORKDIR /build/vllm-src
# Bash + pipefail set at top so exit code propagates through `| tee | tail`
RUN pip install --no-deps -v . 2>&1 | tee /tmp/vllm-install.log | tail -120 && \
    echo "[vllm pip install] exit=$?"

# Apply AEON sm_121a patches (idempotent — no-op when upstream merged equivalent)
RUN cd /aeon-patches && \
    echo "=== AEON patch 1: cuda_optional_import ===" && \
    python3 patch_cuda_optional_import.py && \
    echo "=== AEON patch 2: kv_cache_utils ===" && \
    python3 patch_kv_cache_utils.py && \
    echo "=== AEON patch 3: cudagraph_align ===" && \
    python3 patch_cudagraph_align.py && \
    echo "[AEON patches] all applied"

# Install scipy first (TurboQuant runtime dep), then TurboQuant AEON-7 fork
RUN pip install --no-cache-dir "scipy>=1.11" 2>&1 | tail -3 && \
    pip install --no-cache-dir --no-deps \
      "turboquant @ git+https://github.com/AEON-7/turboquant.git@fix/cuda-graph-safe-qjl-powers" \
      2>&1 | tail -3 || \
    echo "[WARN] turboquant install attempted; check logs above if needed"

# Upgrade transformers to HEAD — Gemma-4 (gemma4_unified) needs 5.10+,
# released 5.5.4 doesn't recognize the architecture
RUN pip install --no-cache-dir --upgrade \
      "transformers @ git+https://github.com/huggingface/transformers.git@main" \
      2>&1 | tail -3 && \
    python3 -c "import transformers; print('transformers:', transformers.__version__)"

# Install humming-stub — vLLM's quantization registry eagerly imports from
# the `humming` library at vLLM init under `if current_platform.is_cuda()`.
# The real humming library is NVIDIA-internal; absent it, vLLM fails to load
# even for non-humming models like ours. The stub provides empty symbols
# so the imports succeed; actual humming usage will raise.
COPY humming-stub/ /tmp/humming-stub/
RUN pip install --no-cache-dir /tmp/humming-stub && rm -rf /tmp/humming-stub && \
    python3 -c "from humming.dtypes import DataType; print('humming-stub: importable')"

# Verify the install — COPY verify script so heredoc shell weirdness can't bite
COPY verify.py /tmp/verify.py
RUN python3 /tmp/verify.py && rm /tmp/verify.py

# Smoke import: confirm core vLLM imports succeed
RUN python3 -c "\
from vllm import LLM, SamplingParams; \
print('vllm.LLM import: ok'); \
from vllm.config import VllmConfig; \
print('VllmConfig import: ok')"

# Cleanup build artifacts to shrink image
RUN rm -rf /build /aeon-patches /root/.cache/pip

ENTRYPOINT ["/bin/bash"]
