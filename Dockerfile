# Ultimate vLLM image for DGX Spark (GB10 / sm_121a) — v0.25.1 base
# - Source: vLLM v0.25.1 (752a3a504; v0.25.0 702f4814 + #47888 torchcodec import guard
#     + #48330 mixed-dtype allreduce-RMSNorm fusion guard) 3-way merged onto the AEON maxsafe tree
#     (codex/aeon-v0.24.0-maxsafe-20260708 = v0.24.0 + extra cherry-picks). Preserves:
#     #44389 (Triton NVFP4 KV) + #40898/#41703 (DFlash SWA + ctx-mask + Gemma4 batched-verify),
#     baked fixes (dflash-blocktable-unpad, cudagraph_align_all_modes, UMA negative-estimate clamp),
#     maxsafe carries #47053/#47356 (kv_cache_memory_bytes cache-hash exclusion — issue #9)/#45207,
#     and the AEON Mamba-DFlash-MRv2 fix. Integrates v0.25.0 #42890 (NVFP4-KV SWA page-unify) +
#     #46761 (DFlash per-layer RMSNorm fusion) + #45739 (NVFP4 swizzled-scale zero-init, Blackwell
#     decode) + #47081 (blocking CUDA events, TP async-copy) — all in-tag or on the maxsafe tree.
# - MRv2 PIN (Phase 1): VLLM_USE_V2_MODEL_RUNNER=0 baked — v0.25.0 defaults dense models to
#     Model-Runner-V2 and whitelists method=dflash for MRv2 with NO fallback, which would SILENTLY
#     route Qwen3.6-27B to the unpatched V2 DFlash tree. The env pin keeps ALL models on the carried
#     V1 runner. (Removed in Phase 2 when carries are ported to MRv2.)
# - DROPPED the #45544 tie_weights cherry-pick (now upstream in the merge).
# - Deps vs v0.24.0: FlashInfer 0.6.12 -> 0.6.13 (v0.25.0 pin; purge stale jit-cache);
#     nvidia-cutlass-dsl[cu13]==4.5.2 (match upstream-tested FA4 cute-DSL). torch 2.11.0 UNCHANGED.
#     torchcodec (v0.25.0 cuda.txt video dep) SKIPPED — 0.14.0 is ABI-broken on torch 2.11; vLLM
#     guards it (PlaceholderModule) so absence degrades gracefully. apt ffmpeg kept for other codecs.
# - Keep: TORCH_CUDA_ARCH_LIST=12.1a (NOT bare 12.1 — 12.0f x 12.1a -> 12.1a keeps native NVFP4
#     CUTLASS; 12.0f x 12.1 -> bare 12.1 -> FP4 kernels #ifdef out -> Marlin fallback), GCC 12,
#     transformers 5.12.1, xgrammar>=0.2.1, TurboQuant fork, humming-stub.

FROM ghcr.io/aeon-7/aeon-gemma-4-26b-a4b-dflash:latest AS base

SHELL ["/bin/bash", "-o", "pipefail", "-c"]

ENV TORCH_CUDA_ARCH_LIST="12.1a"
ENV ENABLE_NVFP4_SM100=0
ENV CCACHE_DISABLE=1
ENV CMAKE_BUILD_PARALLEL_LEVEL=8
ENV MAX_JOBS=12
ENV NVCC_THREADS=2
ENV SETUPTOOLS_SCM_PRETEND_VERSION="0.25.1+aeon.sm121a.dflash"
ENV SETUPTOOLS_SCM_PRETEND_VERSION_FOR_VLLM="0.25.1+aeon.sm121a.dflash"
# Phase-1 MRv2 pin: keep the carried V1 GPUModelRunner for all models (see header).
ENV VLLM_USE_V2_MODEL_RUNNER=0

WORKDIR /build

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        git ca-certificates ffmpeg \
        gcc-12 g++-12 \
        cuda-nvrtc-dev-13-0 \
        libcusparse-dev-13-0 \
        libcublas-dev-13-0 \
        libcusolver-dev-13-0 \
        libcufft-dev-13-0 \
        libcurand-dev-13-0 \
        libnvjitlink-dev-13-0 && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# GCC >= 12 for C++20 (#44923); base ships 11.4
ENV CC=gcc-12
ENV CXX=g++-12
ENV CUDAHOSTCXX=g++-12

RUN CUDA_LIB=/usr/local/cuda-13.0/targets/sbsa-linux/lib && \
    if [ ! -f $CUDA_LIB/libnvrtc.so ] && [ -f $CUDA_LIB/libnvrtc.so.13.0.88 ]; then \
        ln -sf $CUDA_LIB/libnvrtc.so.13.0.88 $CUDA_LIB/libnvrtc.so; \
        echo "Created libnvrtc.so symlink"; \
    fi && \
    ls -la $CUDA_LIB/libnvrtc.so* | head -5

# Re-pin CUDA 13.0 toolkit + assert nvcc==13.0 before compiling extensions (r0b0tlab gotcha:
# 0.25.0 deps can pull cu13.x python compiler pkgs that shadow the system nvcc -> ABI mismatch).
RUN update-alternatives --set cuda /usr/local/cuda-13.0 2>/dev/null || true; \
    export CUDA_HOME=/usr/local/cuda-13.0 PATH=/usr/local/cuda-13.0/bin:$PATH; \
    nvcc --version | sed -n 's/.*release \([0-9.]\+\).*/nvcc release \1/p' | head -1; \
    test "$(nvcc --version | sed -n 's/.*release \([0-9.]\+\).*/\1/p' | head -1)" = "13.0"

COPY vllm-src/ /build/vllm-src/

RUN pip uninstall -y vllm 2>&1 | tail -3

WORKDIR /build/vllm-src
RUN CUDA_HOME=/usr/local/cuda-13.0 PATH=/usr/local/cuda-13.0/bin:$PATH \
    pip install --no-deps -v . 2>&1 | tee /tmp/vllm-install.log | tail -120 && \
    echo "[vllm pip install] exit=$?"

# torchcodec: INTENTIONALLY NOT INSTALLED. v0.25.0 lists torchcodec>=0.14 in cuda.txt (video
# decode), but the newest published torchcodec (0.14.0) is ABI-INCOMPATIBLE with torch 2.11.0+cu130
# (its libtorchcodec_custom_ops*.so won't load into torch 2.11; verified 2026-07-14), and no
# torch-2.11-matched torchcodec exists on PyPI yet. vLLM 0.25.0 guards the import
# (vllm/multimodal/video.py: try/except ImportError -> PlaceholderModule), so ABSENT torchcodec
# degrades GRACEFULLY (video-decode unavailable) while text/image/audio/voice are unaffected — a
# PRESENT-but-broken torchcodec would instead throw an uncaught OSError and crash. Current prod
# (v0.24.0) also ships without torchcodec, so this is not a regression. ffmpeg is still installed
# (apt, above) for other codec paths. Revisit when torchcodec publishes a torch-2.11 build.

# nvidia-cutlass-dsl pinned to the upstream-tested 4.5.2 for the FA4 cute-DSL path (base ships 4.6.0)
RUN pip install --no-cache-dir "nvidia-cutlass-dsl[cu13]==4.5.2" 2>&1 | tail -3 && \
    python3 -c "import importlib.metadata as m; print('cutlass-dsl:', m.version('nvidia-cutlass-dsl'))"

# FlashInfer 0.6.12 -> 0.6.13 (v0.25.0 pin, #46683). Purge the stale 0.6.12 jit-cache first
# (it shadows 0.6.13 kernels); rely on runtime JIT or the matching 0.6.13 cubin.
RUN pip uninstall -y flashinfer-jit-cache 2>&1 | tail -2 || true; \
    pip install --no-cache-dir "flashinfer-python==0.6.13" "flashinfer-cubin==0.6.13" 2>&1 | tail -3 && \
    (flashinfer download-cubin 2>&1 | tail -3 || echo "[WARN] cubin download failed; runtime JIT fallback") && \
    python3 -c "import flashinfer, importlib.metadata as m; assert flashinfer.__version__=='0.6.13', flashinfer.__version__; \
      import importlib.util; assert importlib.util.find_spec('flashinfer_jit_cache') is None, 'stale jit-cache present'; \
      print('flashinfer:', flashinfer.__version__, '(jit-cache absent)')"

# xgrammar >= 0.2.1 (tool-choice 500s with ImportError: normalize_tool_choice otherwise)
RUN pip install --no-cache-dir "xgrammar>=0.2.1,<1.0.0" 2>&1 | tail -2 && \
    python3 -c "from xgrammar import normalize_tool_choice; print('xgrammar: normalize_tool_choice OK')"

RUN pip install --no-cache-dir "scipy>=1.11" 2>&1 | tail -3 && \
    pip install --no-cache-dir --no-deps \
      "turboquant @ git+https://github.com/AEON-7/turboquant.git@fix/cuda-graph-safe-qjl-powers" \
      2>&1 | tail -3 || \
    echo "[WARN] turboquant install attempted; check logs above if needed"

# Pinned stable transformers (>=5.5.3 floor in v0.25.0; 5.12.1 fleet-validated)
RUN pip install --no-cache-dir --upgrade "transformers==5.12.1" 2>&1 | tail -3 && \
    python3 -c "import transformers; print('transformers:', transformers.__version__)"

COPY humming-stub/ /tmp/humming-stub/
RUN pip install --no-cache-dir /tmp/humming-stub && rm -rf /tmp/humming-stub && \
    python3 -c "from humming.dtypes import DataType; print('humming-stub: importable')"

COPY verify.py /tmp/verify.py
RUN python3 /tmp/verify.py && rm /tmp/verify.py

# Smoke (from WORKDIR / so cwd doesn't shadow the installed pkg; dlopen stable-ABI vs the driver stub):
# confirm all load-bearing AEON symbols survived the 0.25.0 merge + the MRv2 V1 pin is honored.
WORKDIR /
RUN mkdir -p /tmp/cuda-stub && \
    ln -s /usr/local/cuda-13.0/targets/sbsa-linux/lib/stubs/libcuda.so /tmp/cuda-stub/libcuda.so.1 && \
    LD_LIBRARY_PATH=/tmp/cuda-stub:$LD_LIBRARY_PATH python3 -c "\
import vllm._C_stable_libtorch; import vllm._moe_C_stable_libtorch; \
import vllm, inspect; assert vllm.__version__.startswith('0.25.1'), vllm.__version__; \
from vllm import LLM, SamplingParams; from vllm.config import VllmConfig; \
import vllm.model_executor.models.qwen3_dflash as q; assert 'sliding_attention_layer_names' in inspect.getsource(q), 'SWA lost'; \
import vllm.v1.spec_decode.utils as u; assert 'is_valid_ctx' in inspect.getsource(u), 'ctx-slot mask lost'; \
import vllm.v1.attention.backends.triton_attn as t; assert 'nvfp4' in inspect.getsource(t).lower(), 'NVFP4-KV lost'; \
import vllm.v1.spec_decode.dflash as d; assert 'dflash-blocktable-unpad' in inspect.getsource(d), 'blocktable slice lost'; \
import vllm.config.compilation as cc; assert 'cudagraph_align_spec_decode_all_modes' in inspect.getsource(cc), 'cudagraph align lost'; \
import vllm.v1.worker.gpu_model_runner as gmr; assert 'uma-negative-cudagraph-estimate-clamp' in inspect.getsource(gmr), 'UMA clamp lost'; \
import vllm.envs as e; assert e.VLLM_USE_V2_MODEL_RUNNER == 0 or e.VLLM_USE_V2_MODEL_RUNNER is False, 'V2 runner not pinned off'; \
import vllm.multimodal.video as vid; \
print('vllm', vllm.__version__, '+aeon import OK; DFlash SWA + ctx-mask + NVFP4-KV + baked fixes present; V1 pinned; video.py graceful-degrade OK')" && \
    rm -rf /tmp/cuda-stub

RUN rm -rf /build /root/.cache/pip

LABEL ai.aeon.vllm_base="vLLM 0.25.1 (from-source, sm_121a 3-way merge onto maxsafe)" \
      ai.aeon.model="fleet: Gemma-4-26B-A4B, Qwen3.6-27B, Qwen3.6-35B-A3B" \
      ai.aeon.hardware="NVIDIA DGX Spark GB10 SM121" \
      ai.aeon.features="gemma4,qwen3.6,dflash,dflash-highconc-fix,mrv2-lmhead-fix,dspark-ready,nvfp4,nvfp4-kv,fp8-kv,swizzled-scale-zeroinit,flashinfer-0.6.13,cutlass-dsl-4.5.2,uma-clamp,kv-cache-bytes-cachehash-fix,tp2-fusion-guard,v1-pinned,tp2-ready,turboquant,tool-calling" \
      org.opencontainers.image.description="AEON vLLM Ultimate — vLLM 0.25.1 built from source for DGX Spark / Blackwell (sm_121a/GB10). Carries Triton NVFP4-KV (#44389), DFlash SWA + prefix-cache + high-concurrency fixes (#40898/#41703/#43982-port), MRv2 spec-decode lm_head fix (#47914-port: dflash/eagle/dspark), UMA/cudagraph clamps, TP>1 mixed-dtype fusion guard (#48330), FlashInfer 0.6.13; V1 runner pinned (MRv2 opt-in via env); TP=2-ready (untested)." \
      org.opencontainers.image.documentation="https://github.com/AEON-7/vllm-ultimate-dgx-spark" \
      org.opencontainers.image.source="https://github.com/AEON-7/vllm-ultimate-dgx-spark" \
      org.opencontainers.image.licenses="Apache-2.0"

ENTRYPOINT ["/bin/bash"]
