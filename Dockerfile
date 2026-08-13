# Ultimate vLLM image for DGX Spark (GB10 / sm_121a) — v0.26.0 base
# - Source: vLLM v0.26.0 (568afb3a1) 3-way merged onto the AEON tree. #44455 re-based the
#     KV layout 5-D -> 4-D packed, so the #44389 NVFP4-KV carry was RE-DERIVED (slicing-based
#     views; upstream's as_strided math silently corrupts under NHD = the GB10 default).
#     Our MRv2 lm_head fix is now upstream (#47914 merged) and was dropped.
# - MRv2 PIN (Phase 1): VLLM_USE_V2_MODEL_RUNNER=0 baked — v0.25.0 defaults dense models to
#     Model-Runner-V2 and whitelists method=dflash for MRv2 with NO fallback, which would SILENTLY
#     route Qwen3.6-27B to the unpatched V2 DFlash tree. The env pin keeps ALL models on the carried
#     V1 runner. (Removed in Phase 2 when carries are ported to MRv2.)
# - DROPPED the #45544 tie_weights cherry-pick (now upstream in the merge).
# - Deps vs v0.24.0: FlashInfer 0.6.12 -> 0.6.14 (v0.25.0 pin; purge stale jit-cache);
#     nvidia-cutlass-dsl 4.6.0 (upstream-tested; our 4.5.2 downgrade is retired). torch 2.11.0 UNCHANGED.
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
ENV SETUPTOOLS_SCM_PRETEND_VERSION="0.26.0+aeon.sm121a.dflash"
ENV SETUPTOOLS_SCM_PRETEND_VERSION_FOR_VLLM="0.26.0+aeon.sm121a.dflash"
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

# nvidia-cutlass-dsl: base image ships 4.4.2; v0.26.0 moved the upstream-tested
# pin to 4.6.0 (#47442) for the FA4 cute-DSL path, so upgrade to match upstream
# (we previously pinned 4.5.2 for v0.25.x).
# quack-kernels MUST move with it: the base ships 0.4.1, which calls
# `cutlass.cute.core.ThrMma` — REMOVED in cutlass-dsl 4.6.0, so the engine dies at
# model boot with AttributeError (the import-only smoke gate does NOT catch this;
# only a real model load does). Upstream's `quack-kernels>=0.4.0` is unbounded and
# resolves to 0.6.x in a fresh env; pin that explicitly.
# apache-tvm-ffi moves WITH cutlass-dsl: v0.26.0 bumped 0.1.9 -> 0.1.10 in the same
# commit as cutlass 4.5.2 -> 4.6.0. Taking cutlass without tvm-ffi makes
# _warmup_ll_bf16_router_gemm() die with
#   TypeError: make_kwargs_wrapper() got an unexpected keyword argument 'map_dataclass_to_tuple'
# and that warmup is device-gated only (capability>=90, so GB10 qualifies) and called
# with NO try/except from gpu_worker.compile_or_warm_up_model -> it kills EVERY model boot.
RUN pip install --no-cache-dir "nvidia-cutlass-dsl[cu13]==4.6.0" "quack-kernels>=0.6,<0.7" "apache-tvm-ffi==0.1.10" 2>&1 | tail -3 && \
    python3 -c "import importlib.metadata as m; \
      c=m.version('nvidia-cutlass-dsl'); q=m.version('quack-kernels'); \
      print('cutlass-dsl:', c, '| quack-kernels:', q); \
      assert c.startswith('4.6'), c; assert q.startswith('0.6'), q; \
      t=m.version('apache-tvm-ffi'); print('apache-tvm-ffi:', t); assert t=='0.1.10', t; \
      import inspect as _i; from tvm_ffi.utils.kwargs_wrapper import make_kwargs_wrapper as _mkw; \
      assert 'map_dataclass_to_tuple' in _i.signature(_mkw).parameters, 'tvm-ffi too old for cutlass 4.6 cute.compile'; \
      from quack import layout_utils; print('quack/cutlass/tvm-ffi ABI compatible')"

# FlashInfer 0.6.13 -> 0.6.14, matching upstream v0.26.0's pin. NOTE: PyPI carries
# flashinfer-cubin only to 0.6.13, but the FlashInfer index publishes 0.6.14 as a
# platform-independent (py3-none-any) cubin wheel — hence --extra-index-url. The
# pair MUST match exactly: flashinfer >=0.6.14 refuses to import against a
# mismatched cubin ("does not match flashinfer version"). Purge the stale 0.6.8
# jit-cache first (it shadows newer kernels).
RUN pip uninstall -y flashinfer-jit-cache 2>&1 | tail -2 || true; \
    pip install --no-cache-dir --extra-index-url https://flashinfer.ai/whl/ \
        "flashinfer-python==0.6.14" "flashinfer-cubin==0.6.14" 2>&1 | tail -3 && \
    (flashinfer download-cubin 2>&1 | tail -3 || echo "[WARN] cubin download failed; runtime JIT fallback") && \
    python3 -c "import flashinfer, importlib.metadata as m; assert flashinfer.__version__=='0.6.14', flashinfer.__version__; \
      import importlib.metadata as mm; assert mm.version('flashinfer-cubin')=='0.6.14', mm.version('flashinfer-cubin'); \
      print('flashinfer:', flashinfer.__version__, '+ matched cubin 0.6.14')"

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
import vllm, inspect; assert vllm.__version__.startswith('0.26.0'), vllm.__version__; \
from vllm import LLM, SamplingParams; from vllm.config import VllmConfig; \
import vllm.model_executor.models.qwen3_dflash as q; assert 'sliding_attention_layer_names' in inspect.getsource(q), 'SWA lost'; \
import vllm.v1.spec_decode.utils as u; assert 'is_valid_ctx' in inspect.getsource(u), 'ctx-slot mask lost'; \
import vllm.v1.attention.backends.triton_attn as t; assert 'nvfp4' in inspect.getsource(t).lower(), 'NVFP4-KV lost'; \
import vllm.v1.spec_decode.dflash as d; assert 'dflash-blocktable-unpad' in inspect.getsource(d), 'blocktable slice lost'; \
import vllm.config.compilation as cc; assert 'cudagraph_align_spec_decode_all_modes' in inspect.getsource(cc), 'cudagraph align lost'; \
import vllm.v1.worker.gpu_model_runner as gmr; assert 'uma-negative-cudagraph-estimate-clamp' in inspect.getsource(gmr), 'UMA clamp lost'; \
import vllm.envs as e; assert e.VLLM_USE_V2_MODEL_RUNNER == 0 or e.VLLM_USE_V2_MODEL_RUNNER is False, 'V2 runner not pinned off'; \
import vllm.v1.worker.gpu.spec_decode.eagle.utils as eu; assert hasattr(eu, 'get_target_lm_head'), 'upstream #47914 lm_head helper missing'; \
import vllm.config.vllm as cv; assert '_dflash_needs_multi_kv_group' in inspect.getsource(cv), 'v0.26.0 force-V2 trigger missing'; \
assert inspect.getsource(cv.VllmConfig.use_v2_model_runner.fget).index('VLLM_USE_V2_MODEL_RUNNER') < inspect.getsource(cv.VllmConfig.use_v2_model_runner.fget).index('_dflash_needs_multi_kv_group'), 'env pin no longer precedes the force-V2 trigger'; \
import vllm.utils.torch_utils as tu; assert hasattr(tu, 'nvfp4_kv_cache_split_views'), 'nvfp4 split helper missing (ImportError at boot)'; \
assert 'as_strided' not in tu._nvfp4_split_data_scale.__code__.co_names, 'nvfp4 splitter reverted to upstream as_strided math (NHD-unsafe)'; \
assert hasattr(q, 'dflash_has_any_non_causal'), 'dflash_has_any_non_causal missing (V1 proposer ImportError)'; \
import vllm.multimodal.video as vid; \
print('vllm', vllm.__version__, '+aeon import OK; carries present (SWA/ctx-mask/NVFP4-KV/blocktable/cudagraph/UMA); V1 pin precedes force-V2; nvfp4 views slicing-based; video graceful-degrade OK')" && \
    rm -rf /tmp/cuda-stub

# AEON build gate: NVFP4 KV data/scale views must be DISJOINT under BOTH cache
# layouts. #44455 packed K/V into the content dim; upstream's as_strided offset
# math only tiles correctly under HND, and GB10 defaults to NHD -> silent KV
# corruption (measured: upstream covers 7760/9216 bytes with 5 overlaps).
# This gate fails the build if the slicing-based splitter ever regresses.
COPY nvfp4_kv_gate.py /tmp/nvfp4_kv_gate.py
RUN python3 /tmp/nvfp4_kv_gate.py && rm /tmp/nvfp4_kv_gate.py

RUN rm -rf /build /root/.cache/pip

LABEL ai.aeon.vllm_base="vLLM 0.26.0 (from-source, sm_121a 3-way merge onto maxsafe)" \
      ai.aeon.model="fleet: Gemma-4-26B-A4B, Gemma-4-31B-DECKARD (NVFP4_AWQ), Qwen3.6-27B, Qwen3.6-35B-A3B" \
      ai.aeon.hardware="NVIDIA DGX Spark GB10 SM121" \
      ai.aeon.features="gemma4,qwen3.6,dflash,dflash-swa,dflash-highconc-fix,nvfp4,nvfp4-awq,nvfp4-kv,nvfp4-kv-nhd-safe,fp8-kv,spec-kv-dtype,prefix-match-unit,hybrid-apc,mrv2-lmhead-upstream,flashinfer-0.6.14,cutlass-dsl-4.6.0,quack-0.6.1,tvm-ffi-0.1.10,uma-clamp,tp2-cudagraph-fix,v1-pinned,tp2-ready,turboquant,tool-calling" \
      org.opencontainers.image.description="AEON vLLM Ultimate — vLLM 0.26.0 built from source for DGX Spark / Blackwell (sm_121a/GB10). Carries Triton NVFP4-KV (#44389, re-derived for the 4-D packed layout with NHD-safe slicing), DFlash SWA + ctx-mask + high-concurrency fixes (#40898/#41703/#43982-port), NVFP4_AWQ checkpoint support (Gemma-4-31B DECKARD), UMA/cudagraph clamps, TP>=2 cudagraph fix (#46253-port), FlashInfer 0.6.14 + cutlass-dsl 4.6.0. 1151 tok/s @128 concurrent on Gemma-4-26B-A4B. V1 runner pinned; TP=2-ready (untested)." \
      org.opencontainers.image.documentation="https://github.com/AEON-7/vllm-ultimate-dgx-spark" \
      org.opencontainers.image.source="https://github.com/AEON-7/vllm-ultimate-dgx-spark" \
      org.opencontainers.image.licenses="Apache-2.0"

ENTRYPOINT ["/bin/bash"]
