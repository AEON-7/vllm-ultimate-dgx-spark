# Ultimate vLLM image for DGX Spark (GB10 / sm_121a) — v0.24.0 base
# - Source: vLLM v0.24.0 (ee0da84ab) 3-way merged with our AEON branch (aeon-v0.24.0):
#     preserves #44389 (Triton NVFP4 KV) + #40898/#41703 (DFlash SWA + ctx-slot mask + Gemma4 fixes),
#     takes v0.24.0 improvements (Triton non-causal, DFlash-on-FlashInfer #43081, UMA pressure
#     release #45179, KernelConfig --moe/--linear-backend incl. flashinfer_b12x, async-sched default).
# - Baked into SOURCE this release (no runtime patch step anymore):
#     * dflash-blocktable-unpad ([: cad.num_reqs] slice, port of #43982 for our multi-KV-group DFlash)
#     * cudagraph_align_spec_decode_all_modes (upstream twin: open PR #46324)
#     * UMA negative-cudagraph-estimate clamp (port of open PR #46932, GB10 issue #44740)
# - patch_cuda_optional_import DROPPED — v0.24.0 stable-ABI migration arch-gates the offending
#     kernels (verified: only _C_stable_libtorch is imported unguarded and it loads clean on sm_121);
#     replaced by a build-time import smoke test below.
# - FlashInfer 0.6.8.post1 -> 0.6.12 (REQUIRED: v0.24.0 lazy-imports flashinfer.fused_moe b12x).
# - transformers pinned 5.12.1 (first stable release covering the whole fleet; replaces git-HEAD).
# - GCC 12 for C++20 (upstream #44923); TurboQuant AEON-7 fork kept (retire after stock K8V4 test);
#   humming-stub unchanged (v0.24.0 lazy facade verified compatible).

FROM ghcr.io/aeon-7/aeon-gemma-4-26b-a4b-dflash:latest AS base

SHELL ["/bin/bash", "-o", "pipefail", "-c"]

ENV TORCH_CUDA_ARCH_LIST="12.1a"
ENV ENABLE_NVFP4_SM100=0
ENV CCACHE_DISABLE=1
ENV CMAKE_BUILD_PARALLEL_LEVEL=8
ENV MAX_JOBS=12
ENV NVCC_THREADS=2
ENV SETUPTOOLS_SCM_PRETEND_VERSION="0.24.0+aeon.sm121a.dflash"
ENV SETUPTOOLS_SCM_PRETEND_VERSION_FOR_VLLM="0.24.0+aeon.sm121a.dflash"

WORKDIR /build

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        git ca-certificates \
        gcc-12 g++-12 \
        cuda-nvrtc-dev-13-0 \
        libcusparse-dev-13-0 \
        libcublas-dev-13-0 \
        libcusolver-dev-13-0 \
        libcufft-dev-13-0 \
        libcurand-dev-13-0 \
        libnvjitlink-dev-13-0 && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# v0.24.0 needs GCC >= 12 for C++20 (#44923); base ships 11.4
ENV CC=gcc-12
ENV CXX=g++-12
ENV CUDAHOSTCXX=g++-12

RUN CUDA_LIB=/usr/local/cuda-13.0/targets/sbsa-linux/lib && \
    if [ ! -f $CUDA_LIB/libnvrtc.so ] && [ -f $CUDA_LIB/libnvrtc.so.13.0.88 ]; then \
        ln -sf $CUDA_LIB/libnvrtc.so.13.0.88 $CUDA_LIB/libnvrtc.so; \
        echo "Created libnvrtc.so symlink"; \
    fi && \
    ls -la $CUDA_LIB/libnvrtc.so* | head -5

COPY vllm-src/ /build/vllm-src/

RUN pip uninstall -y vllm 2>&1 | tail -3

WORKDIR /build/vllm-src
RUN pip install --no-deps -v . 2>&1 | tee /tmp/vllm-install.log | tail -120 && \
    echo "[vllm pip install] exit=$?"

# FlashInfer 0.6.12 to match the v0.24.0 pin (0.6.8.post1 lacks fused_moe b12x symbols)
RUN pip install --no-cache-dir \
      "flashinfer-python==0.6.12" "flashinfer-cubin==0.6.12" 2>&1 | tail -3 && \
    (pip install --no-cache-dir flashinfer-jit-cache==0.6.12 \
       --extra-index-url https://flashinfer.ai/whl/cu130 2>&1 | tail -3 || \
     echo "[WARN] flashinfer-jit-cache cu130 wheel unavailable; JIT will compile on first use") && \
    (flashinfer download-cubin 2>&1 | tail -3 || echo "[WARN] cubin download failed; runtime fallback") && \
    python3 -c "import flashinfer; print('flashinfer:', flashinfer.__version__)"

RUN pip install --no-cache-dir "scipy>=1.11" 2>&1 | tail -3 && \
    pip install --no-cache-dir --no-deps \
      "turboquant @ git+https://github.com/AEON-7/turboquant.git@fix/cuda-graph-safe-qjl-powers" \
      2>&1 | tail -3 || \
    echo "[WARN] turboquant install attempted; check logs above if needed"

# Pinned stable transformers (>=5.5.3 hard floor in v0.24.0; 5.12.1 fleet-smoke-tested 2026-07-01)
RUN pip install --no-cache-dir --upgrade "transformers==5.12.1" 2>&1 | tail -3 && \
    python3 -c "import transformers; print('transformers:', transformers.__version__)"

COPY humming-stub/ /tmp/humming-stub/
RUN pip install --no-cache-dir /tmp/humming-stub && rm -rf /tmp/humming-stub && \
    python3 -c "from humming.dtypes import DataType; print('humming-stub: importable')"

COPY verify.py /tmp/verify.py
RUN python3 /tmp/verify.py && rm /tmp/verify.py

# Smoke: confirm load-bearing AEON symbols + baked-in fixes survived into the INSTALLED package,
# and that the stable-ABI extensions dlopen clean without the retired RTLD_LAZY patch
# (catches reintroduced ungated sm_100-only symbols at build time).
# - WORKDIR must leave /build/vllm-src: python -c puts cwd first on sys.path, and the
#   source tree would shadow the installed package (no compiled .so -> false failure).
# - libcuda.so.1 (driver) is absent at build time; dlopen against the toolkit stub.
WORKDIR /
RUN mkdir -p /tmp/cuda-stub && \
    ln -s /usr/local/cuda-13.0/targets/sbsa-linux/lib/stubs/libcuda.so /tmp/cuda-stub/libcuda.so.1 && \
    LD_LIBRARY_PATH=/tmp/cuda-stub:$LD_LIBRARY_PATH python3 -c "\
import vllm._C_stable_libtorch; \
import vllm._moe_C_stable_libtorch; \
from vllm import LLM, SamplingParams; \
from vllm.config import VllmConfig; \
import inspect, vllm.model_executor.models.qwen3_dflash as q; \
assert 'sliding_attention_layer_names' in inspect.getsource(q), 'SWA lost'; \
import vllm.v1.spec_decode.utils as u; \
assert 'is_valid_ctx' in inspect.getsource(u), 'ctx-slot mask lost'; \
import vllm.v1.attention.backends.triton_attn as t; \
assert 'nvfp4' in inspect.getsource(t).lower(), 'NVFP4-KV lost'; \
import vllm.v1.spec_decode.dflash as d; \
assert 'dflash-blocktable-unpad' in inspect.getsource(d), 'blocktable slice lost'; \
import vllm.config.compilation as cc; \
assert 'cudagraph_align_spec_decode_all_modes' in inspect.getsource(cc), 'cudagraph align lost'; \
import vllm.v1.worker.gpu_model_runner as gmr; \
assert 'uma-negative-cudagraph-estimate-clamp' in inspect.getsource(gmr), 'UMA clamp lost'; \
print('vllm v0.24.0+aeon import OK; DFlash SWA + ctx-mask + NVFP4-KV + 3 baked fixes present')" && \
    rm -rf /tmp/cuda-stub

RUN rm -rf /build /root/.cache/pip

LABEL ai.aeon.vllm_base="vLLM 0.24.0 (from-source, sm_121a 3-way merge)" \
      ai.aeon.model="fleet: Gemma-4-26B-A4B, Qwen3.6-27B, Qwen3.6-35B-A3B" \
      ai.aeon.hardware="NVIDIA DGX Spark GB10 SM121" \
      ai.aeon.features="gemma4,qwen3.6,dflash,dflash-highconc-fix,prefix-cache-fix,nvfp4,nvfp4-kv,fp8-kv,flashinfer-0.6.12,flashinfer-b12x,kernel-config,uma-clamp,dynamic-sd,async-sched,turboquant,tool-calling" \
      org.opencontainers.image.description="AEON vLLM Ultimate — vLLM 0.24.0 built from source for DGX Spark / Blackwell (sm_121a/GB10). One image serves the whole AEON fleet (Gemma-4-26B-A4B, Qwen3.6-27B, Qwen3.6-35B-A3B) with DFlash speculative decoding, NVFP4 weights, Triton NVFP4/FP8 KV cache (PR #44389), DFlash SWA + prefix-cache + high-concurrency fixes (PR #40898/#41703/#43982-port), UMA cudagraph clamp (#46932-port), FlashInfer 0.6.12, TurboQuant K8V4." \
      org.opencontainers.image.documentation="https://github.com/AEON-7/vllm-ultimate-dgx-spark" \
      org.opencontainers.image.source="https://github.com/AEON-7/vllm-ultimate-dgx-spark" \
      org.opencontainers.image.licenses="Apache-2.0"

ENTRYPOINT ["/bin/bash"]
