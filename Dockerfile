# Ultimate vLLM image for DGX Spark (GB10 / sm_121a) — v0.27.1 base
# - Source: vLLM v0.27.1 (= v0.27.0 + #50424 quantized DSpark Markov heads) 3-way merged onto
#     the AEON tree (aeon-v0.26.0). Only 7 conflicts this cycle. Post-merge cherry-picks
#     (correctness-first, all merged upstream, none in the v0.27.1 tag): #50276 packed-KV
#     zeroer stride (critical under our NHD packed NVFP4-KV), #51812 GDN gate alignment with
#     spec tokens (Qwen3.6+DFlash), #51843 hybrid-KV fine-grained prefix-cache fix, DSpark set
#     #51602/#50693/#52288/#50910/#49969. SKIPPED #47808/#52436 (depend on post-0.27.1 MRv2
#     refactors; arrive with v0.27.2). SKIPPED #49718 XQA (accuracy regression measured on Spark).
# - V1 PIN REMOVED: VLLM_USE_V2_MODEL_RUNNER is NO LONGER BAKED. DSpark is MRv2-only and
#     v0.27.1 hard-raises rather than falling back; mixed sliding/full DFlash drafters (all
#     z-lab fleet drafters: 4-sliding+1-full) now auto-force MRv2 where upstream #47914/#48113
#     handle SWA natively via multi-KV-groups. Fleet spec decode therefore runs on MRv2 by
#     routing, with our V1 carries kept as a fallback tree (re-pin per-service with
#     VLLM_USE_V2_MODEL_RUNNER=0 only for thinking_token_budget users — V2 silently ignores it).
# - Carries kept: #44389 Triton NVFP4-KV (NHD-safe slicing views + upstream-name alias;
#     upstream's as_strided math still corrupts under NHD = GB10 default — build gate below),
#     #41703 ctx-slot masking (V1), #40898 SWA-on-V1 (dormant fallback), cudagraph align
#     widening (V1-scoped; MRv2 handled upstream by #45953), #46932 UMA negative-estimate
#     clamp (issue #44740 still open), #48053 thread_local capture_error_mode extended to ALL
#     5 torch.cuda.graph sites (issue #46253; TP=2 across two Sparks — graphs still eager-first
#     cross-node). Dropped (now upstream): #43982 port, #50065, #49659.
# - Deps vs v0.26.0 image: torch 2.11.0 -> 2.13.0+cu130 (breaking env change #48155; brings
#     Triton 3.7.1 + torchvision 0.28.0; torchaudio stays 2.11.0 like upstream), FlashInfer
#     0.6.14 -> 0.6.16.post3 as an exact python+cubin+jit-cache TRIO (jit-cache aarch64 wheel
#     now exists on flashinfer.ai/whl/cu130), apache-tvm-ffi 0.1.10 -> 0.1.11, quack-kernels
#     ==0.6.1 EXACT (0.6.2+ untested by upstream CI), cutlass-dsl 4.6.0 UNCHANGED (quack 0.6.1
#     hard-pins it; do NOT bump to 4.7), tilelang 0.1.12 (new hard dep), transformers 5.12.1 ->
#     5.14.1 (upstream-tested pin), nvidia-nccl-cu13==2.30.7 (TP=2 floor is 2.30.4),
#     torchcodec 0.16.0 RE-ADDED (stable-ABI, torch>=2.11-compatible; was ABI-broken at 0.14).
# - Keep: TORCH_CUDA_ARCH_LIST=12.1a (NOT bare 12.1 — keeps native NVFP4 CUTLASS; bare 12.1
#     #ifdefs FP4 kernels out -> Marlin fallback), GCC 12, xgrammar>=0.2.1, TurboQuant fork,
#     humming-stub. New: TRITON_PTXAS_PATH pinned to the CUDA toolkit ptxas (#32704 GB10 fix).

FROM ghcr.io/aeon-7/aeon-gemma-4-26b-a4b-dflash:latest AS base

SHELL ["/bin/bash", "-o", "pipefail", "-c"]

ENV TORCH_CUDA_ARCH_LIST="12.1a"
ENV ENABLE_NVFP4_SM100=0
ENV CCACHE_DISABLE=1
ENV PIP_DEFAULT_TIMEOUT=180
ENV PIP_RETRIES=5
ENV CMAKE_BUILD_PARALLEL_LEVEL=8
ENV MAX_JOBS=12
ENV NVCC_THREADS=2
ENV SETUPTOOLS_SCM_PRETEND_VERSION="0.27.1+aeon.sm121a.dspark"
ENV SETUPTOOLS_SCM_PRETEND_VERSION_FOR_VLLM="0.27.1+aeon.sm121a.dspark"
# GB10: Triton must use the CUDA-toolkit ptxas, not its bundled one (#32704).
ENV TRITON_PTXAS_PATH=/usr/local/cuda/bin/ptxas
# NOTE: VLLM_USE_V2_MODEL_RUNNER deliberately NOT set (see header).

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
    if [ ! -f $CUDA_LIB/libnvrtc.so ] && ls $CUDA_LIB/libnvrtc.so.13.0.* >/dev/null 2>&1; then \
        ln -sf $(ls $CUDA_LIB/libnvrtc.so.13.0.* | head -1) $CUDA_LIB/libnvrtc.so; \
        echo "Created libnvrtc.so symlink"; \
    fi && \
    ls -la $CUDA_LIB/libnvrtc.so* | head -5

# Re-pin CUDA 13.0 toolkit + assert nvcc==13.0 before compiling extensions.
RUN update-alternatives --set cuda /usr/local/cuda-13.0 2>/dev/null || true; \
    export CUDA_HOME=/usr/local/cuda-13.0 PATH=/usr/local/cuda-13.0/bin:$PATH; \
    nvcc --version | sed -n 's/.*release \([0-9.]\+\).*/nvcc release \1/p' | head -1; \
    test "$(nvcc --version | sed -n 's/.*release \([0-9.]\+\).*/\1/p' | head -1)" = "13.0"

# torch 2.13.0+cu130 BEFORE the vLLM build (v0.27.x pin; "breaking environment change"
# per upstream #48155). Pulls Triton 3.7.1 + matched torchvision. torchaudio stays at the
# base's 2.11.0 — upstream v0.27.1 pins torchaudio==2.11.0 alongside torch 2.13 (last
# published torchaudio); do NOT try to bump it.
RUN pip install --no-cache-dir "torch==2.13.0" "torchvision==0.28.0" \
        --index-url https://download.pytorch.org/whl/cu130 2>&1 | tail -4 && \
    python3 -c "import torch, triton, torchaudio; \
      assert torch.__version__.startswith('2.13.0'), torch.__version__; \
      assert torch.version.cuda.startswith('13.0'), torch.version.cuda; \
      assert triton.__version__.startswith('3.7'), triton.__version__; \
      print('torch', torch.__version__, '| triton', triton.__version__, '| torchaudio', torchaudio.__version__)"

# NCCL: match upstream v0.27.1 (2.30.7); the TP=2 dual-Spark floor is 2.30.4 (issue #52504).
RUN pip install --no-cache-dir "nvidia-nccl-cu13==2.30.7" 2>&1 | tail -2 && \
    python3 -c "import importlib.metadata as m; v=m.version('nvidia-nccl-cu13'); assert v=='2.30.7', v; print('nccl', v)"

COPY vllm-src/ /build/vllm-src/

RUN pip uninstall -y vllm 2>&1 | tail -3

WORKDIR /build/vllm-src
# Build WITHOUT pip build isolation: the isolated env re-downloads a multi-GB torch
# from PyPI (read-timeout-prone — killed the first 0.27.1 build) and can ABI-drift from
# our installed 2.13.0+cu130. use_existing_torch.py is vLLM's supported flow for building
# against a preinstalled torch; build deps are provided in-image instead (base already
# ships ninja/packaging/setuptools<81/setuptools-scm/jinja2; rust ext never compiles on
# this box — setuptools-rust is only a declared hook).
RUN pip install --no-cache-dir "cmake>=3.26.1" "setuptools-rust>=1.9.0" wheel 2>&1 | tail -2 && \
    python3 use_existing_torch.py 2>&1 | tail -5
RUN CUDA_HOME=/usr/local/cuda-13.0 PATH=/usr/local/cuda-13.0/bin:$PATH \
    pip install --no-deps --no-build-isolation -v . 2>&1 | tee /tmp/vllm-install.log | tail -120 && \
    echo "[vllm pip install] exit=$?"

# Matched trio + tilelang (v0.27.1 pins): cutlass-dsl 4.6.0 (quack 0.6.1 hard-pins it),
# quack-kernels ==0.6.1 EXACT, apache-tvm-ffi 0.1.11, tilelang 0.1.12 (new cuda.txt hard dep:
# DSv4 mHC TileLang warmup imports it).
RUN pip install --no-cache-dir "nvidia-cutlass-dsl[cu13]==4.6.0" "quack-kernels==0.6.1" \
        "apache-tvm-ffi==0.1.11" "tilelang==0.1.12" 2>&1 | tail -3 && \
    python3 -c "import importlib.metadata as m; \
      c=m.version('nvidia-cutlass-dsl'); q=m.version('quack-kernels'); t=m.version('apache-tvm-ffi'); \
      print('cutlass-dsl:', c, '| quack:', q, '| tvm-ffi:', t, '| tilelang:', m.version('tilelang')); \
      assert c.startswith('4.6'), c; assert q=='0.6.1', q; assert t=='0.1.11', t; \
      import inspect as _i; from tvm_ffi.utils.kwargs_wrapper import make_kwargs_wrapper as _mkw; \
      assert 'map_dataclass_to_tuple' in _i.signature(_mkw).parameters, 'tvm-ffi too old for cutlass 4.6 cute.compile'; \
      from quack import layout_utils; print('quack/cutlass/tvm-ffi ABI compatible')"

# FlashInfer 0.6.16.post3 — python+cubin+jit-cache MUST match exactly. cubin lives on
# flashinfer.ai/whl/, the aarch64 cu130 jit-cache wheel on flashinfer.ai/whl/cu130/ (new:
# prebuilt JIT cache now works on GB10 — no more purge-and-rebuild). Purge stale caches first.
RUN pip uninstall -y flashinfer-jit-cache flashinfer-cubin 2>&1 | tail -2 || true; \
    pip install --no-cache-dir \
        --extra-index-url https://flashinfer.ai/whl/ \
        --extra-index-url https://flashinfer.ai/whl/cu130/ \
        "flashinfer-python==0.6.16.post3" "flashinfer-cubin==0.6.16.post3" \
        "flashinfer-jit-cache==0.6.16.post3" 2>&1 | tail -4 && \
    python3 -c "import flashinfer, importlib.metadata as m; \
      v=flashinfer.__version__; assert v.startswith('0.6.16.post3'), v; \
      c=m.version('flashinfer-cubin'); assert c.startswith('0.6.16.post3'), c; \
      j=m.version('flashinfer-jit-cache'); assert j.startswith('0.6.16.post3'), j; \
      print('flashinfer trio matched:', v, '| cubin', c, '| jit-cache', j)"
# NOTE: no `flashinfer download-cubin` here — the cubin + jit-cache WHEELS ship the
# kernels at 0.6.16 (the downloader re-fetches 23k cubins for 20+ min and can flake;
# absent kernels JIT at runtime anyway). cu130-index wheels carry a +cu130 local
# version — asserts use startswith, not exact match.

# torchcodec 0.16.0 (2026-08-13): stable-ABI, torch>=2.11-compatible, aarch64 wheels.
# Re-added after the 0.14-vs-torch-2.11 ABI break; v0.27.1 lists torchcodec>=0.14 in cuda.txt.
# Import-probe hard-gates it: a PRESENT-but-broken torchcodec throws uncaught OSError at
# serve time, which is strictly worse than absence.
RUN pip install --no-cache-dir "torchcodec==0.16.0" 2>&1 | tail -2 && \
    python3 -c "import torchcodec; from torchcodec.decoders import VideoDecoder; print('torchcodec', torchcodec.__version__, 'loads')" || \
    (pip uninstall -y torchcodec 2>&1 | tail -1 && echo "[WARN] torchcodec 0.16.0 broken on this stack — removed; video decode degrades gracefully")

# xgrammar >= 0.2.1 (tool-choice 500s with ImportError: normalize_tool_choice otherwise)
RUN pip install --no-cache-dir "xgrammar>=0.2.1,<1.0.0" 2>&1 | tail -2 && \
    python3 -c "from xgrammar import normalize_tool_choice; print('xgrammar: normalize_tool_choice OK')"

RUN pip install --no-cache-dir "scipy>=1.11" 2>&1 | tail -3 && \
    pip install --no-cache-dir --no-deps \
      "turboquant @ git+https://github.com/AEON-7/turboquant.git@fix/cuda-graph-safe-qjl-powers" \
      2>&1 | tail -3 || \
    echo "[WARN] turboquant install attempted; check logs above if needed"

# transformers 5.14.1 — the v0.27.1 upstream-tested pin (was 5.12.1). Fleet A/B before push.
RUN pip install --no-cache-dir --upgrade "transformers==5.14.1" 2>&1 | tail -3 && \
    python3 -c "import transformers; print('transformers:', transformers.__version__)"

COPY humming-stub/ /tmp/humming-stub/
RUN pip install --no-cache-dir /tmp/humming-stub && rm -rf /tmp/humming-stub && \
    python3 -c "from humming.dtypes import DataType; print('humming-stub: importable')"

COPY verify.py /tmp/verify.py
RUN python3 /tmp/verify.py && rm /tmp/verify.py

# Smoke (from WORKDIR / so cwd doesn't shadow the installed pkg; dlopen stable-ABI vs the
# driver stub): confirm all load-bearing AEON symbols survived the 0.27.1 merge, the DSpark
# Markov-head machinery is present+quantizable, and the runner routing is as designed.
WORKDIR /
RUN mkdir -p /tmp/cuda-stub && \
    ln -s /usr/local/cuda-13.0/targets/sbsa-linux/lib/stubs/libcuda.so /tmp/cuda-stub/libcuda.so.1 && \
    LD_LIBRARY_PATH=/tmp/cuda-stub:$LD_LIBRARY_PATH python3 -c "\
import vllm._C_stable_libtorch; import vllm._moe_C_stable_libtorch; \
import vllm, inspect; assert vllm.__version__.startswith('0.27.1'), vllm.__version__; \
from vllm import LLM, SamplingParams; from vllm.config import VllmConfig; \
import vllm.model_executor.models.qwen3_dflash as q; assert 'sliding_attention_layer_names' in inspect.getsource(q), 'V1 SWA carry lost'; \
assert hasattr(q, 'dflash_has_any_non_causal'), 'dflash_has_any_non_causal missing'; \
import vllm.v1.spec_decode.utils as u; assert 'is_valid_ctx' in inspect.getsource(u), 'ctx-slot mask lost'; \
import vllm.v1.attention.backends.triton_attn as t; assert 'nvfp4' in inspect.getsource(t).lower(), 'NVFP4-KV lost'; \
import vllm.v1.spec_decode.dflash as d; assert 'dflash-blocktable-unpad' in inspect.getsource(d), 'blocktable slice lost'; \
import vllm.config.compilation as cc; assert 'AEON widened gate' in inspect.getsource(cc), 'cudagraph align widening lost'; \
import vllm.v1.worker.gpu_model_runner as gmr; assert 'uma-negative-cudagraph-estimate-clamp' in inspect.getsource(gmr), 'UMA clamp lost'; \
import vllm.envs as e; assert e.VLLM_USE_V2_MODEL_RUNNER is None, 'V2 runner pin unexpectedly baked'; \
import vllm.config.vllm as cv; src=inspect.getsource(cv.VllmConfig.use_v2_model_runner.fget); \
assert src.index('VLLM_USE_V2_MODEL_RUNNER') < src.index('dspark'), 'env precedence changed'; \
assert '_dflash_needs_multi_kv_group' in inspect.getsource(cv), 'mixed-SWA force-V2 trigger missing'; \
import vllm.v1.worker.gpu.spec_decode.dspark.speculator as dsp; \
import vllm.model_executor.models.qwen3_dspark as qds; \
assert 'quant_config' in inspect.signature(qds.DSparkMarkovHead.__init__).parameters, '#50424 quantized Markov heads missing'; \
import vllm.v1.worker.gpu.spec_decode.eagle.utils as eu; assert hasattr(eu, 'get_target_lm_head'), '#47914 lm_head helper missing'; \
import vllm.utils.torch_utils as tu; assert hasattr(tu, 'nvfp4_kv_cache_split_views'), 'nvfp4 split helper missing'; \
assert hasattr(tu, 'nvfp4_split_data_scale'), 'upstream-name nvfp4 alias missing (FlashInfer callers)'; \
assert 'as_strided' not in tu._nvfp4_split_data_scale.__code__.co_names, 'nvfp4 splitter reverted to as_strided (NHD-unsafe)'; \
import vllm.config.speculative as sp; assert 'dspark_draft_topk' in inspect.getsource(sp), '#49969 top-k Markov projection missing'; \
import vllm.compilation.cuda_graph as cg; assert 'thread_local' in inspect.getsource(cg), '#48053 thread_local lost (V1 graphs)'; \
import vllm.v1.worker.gpu.cudagraph_utils as cgu; assert 'thread_local' in inspect.getsource(cgu), '#48053 thread_local lost (MRv2 graphs)'; \
import vllm.multimodal.video as vid; \
print('vllm', vllm.__version__, '+aeon import OK; carries present; DSpark Markov heads (quantized) present; MRv2 routing intact')" && \
    rm -rf /tmp/cuda-stub

# AEON build gate: NVFP4 KV data/scale views must be DISJOINT and COMPLETE under BOTH cache
# layouts (NHD is the GB10 default; upstream's as_strided formulation covers 7760/9216 bytes
# with 5 overlaps). Fails the build if the slicing-based splitter ever regresses.
COPY nvfp4_kv_gate.py /tmp/nvfp4_kv_gate.py
RUN python3 /tmp/nvfp4_kv_gate.py && rm /tmp/nvfp4_kv_gate.py

RUN rm -rf /build /root/.cache/pip

LABEL ai.aeon.vllm_base="vLLM 0.27.1 (from-source, sm_121a 3-way merge + 8 cherry-picks)" \
      ai.aeon.model="fleet: Gemma-4-26B-A4B, Gemma-4-31B-DECKARD (NVFP4_AWQ), Qwen3.6-27B, Qwen3.6-35B-A3B" \
      ai.aeon.hardware="NVIDIA DGX Spark GB10 SM121" \
      ai.aeon.features="gemma4,qwen3.6,dflash,dflash-swa-mrv2,dspark,dspark-markov-heads,dspark-markov-quantized,dspark-topk,nvfp4,nvfp4-awq,nvfp4-kv,nvfp4-kv-nhd-safe,fp8-kv,spec-kv-dtype,hybrid-apc-fix,packed-kv-zeroer-fix,gdn-gate-align,flashinfer-0.6.16.post3,torch-2.13.0,triton-3.7.1,cutlass-dsl-4.6.0,quack-0.6.1,tvm-ffi-0.1.11,tilelang-0.1.12,nccl-2.30.7,torchcodec-0.16.0,uma-clamp,thread-local-capture,tp2-ready,turboquant,tool-calling,mrv2-default-routing" \
      org.opencontainers.image.description="AEON vLLM Ultimate — vLLM 0.27.1 built from source for DGX Spark / Blackwell (sm_121a/GB10). DSpark Markov heads (quantized, #50424) + top-k projection (#49969), DFlash SWA on MRv2 (upstream #47914/#48113) with V1 carries as fallback, Triton NVFP4-KV (NHD-safe slicing), NVFP4_AWQ, packed-KV zeroer fix (#50276), GDN spec-token gate fix (#51812), hybrid-APC fix (#51843), DSpark bugfix set, thread_local capture at all graph sites (TP=2 2x-Spark ready, eager-first cross-node), torch 2.13.0 + Triton 3.7.1 + FlashInfer 0.6.16.post3 trio + NCCL 2.30.7." \
      org.opencontainers.image.documentation="https://github.com/AEON-7/vllm-ultimate-dgx-spark" \
      org.opencontainers.image.source="https://github.com/AEON-7/vllm-ultimate-dgx-spark" \
      org.opencontainers.image.licenses="Apache-2.0"

ENTRYPOINT ["/bin/bash"]
