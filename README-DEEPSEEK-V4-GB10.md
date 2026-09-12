# DeepSeek-V4-Flash on 2× DGX Spark (GB10 / sm_121a) — TP2 enablement layer

`Dockerfile.deepseek-v4-gb10-layer` makes the stock `aeon-vllm-ultimate:2026-07-01-v0.24.0`
image serve **DeepSeek-V4-Flash** across two DGX Sparks (TP=2, expert-parallel, RoCE), keeping
the base's stock **nvidia-cutlass-dsl 4.6.0**. The base already ships the native
`vllm/models/deepseek_v4` package; on GB10 it dies on a cascade of gaps, each fixed here
(validated end-to-end on 2× DGX Spark; **66.4 tok/s aggregate** over 10 concurrent multi-turn
conversations, full benchmark with zero engine errors):

| # | Symptom on stock 0.24 (GB10) | Root cause | Fix in this layer |
|---|---|---|---|
| 1 | `AttributeError: cutlass.cute.core has no attribute 'ThrMma'` on **any** DeepSeek-V4 load | cutlass-dsl 4.6.0 moved `ThrMma`/`TiledMma` from `cute.core` to the `cutlass.cute` top level; the base's own cutedsl ops still use the old path | `thrmma_shim.py`: PEP-562 lazy re-export appended to `cute/core.py` |
| 2 | DeepGEMM asserts `Unsupported architecture` (`hyperconnection.hpp`, `attention.hpp`, einsum `layout.hpp`) | vendored DeepGEMM has sm90/sm100 kernels only | install DeepGEMM `nv_dev` @ `a6b593d` ([deepseek-ai/DeepGEMM#324](https://github.com/deepseek-ai/DeepGEMM/pull/324): native sm120 kernels); site-packages takes precedence |
| 3 | `trtllm_batch_decode_sparse_mla_dsv4() got an unexpected keyword 'swa_topk_lens'` | the base's `flashinfer_sparse.py` targets the 0.6.14 sparse-MLA API; image ships 0.6.12 | upgrade flashinfer trio to 0.6.14 (python+cubin+jit-cache cu130/aarch64) |
| 4 | `make_kwargs_wrapper() got an unexpected keyword argument 'map_dataclass_to_tuple'` at first cutedsl AOT compile | cutlass 4.6.0's tvm-ffi provider needs apache-tvm-ffi ≥0.1.10 (base ships 0.1.9); but 0.1.12 aborts tilelang (`TypeAttr __ffi_repr__ already registered`) | the working pair: **apache-tvm-ffi 0.1.11 + tilelang 0.1.11** |
| 5 | `fmax() takes 2 positional arguments but 3 ... given` in cutedsl kernels | `vllm_flash_attn/cute/utils.py` picks the nvvm.fmax binding by `CUDA_VERSION`, and cutlass-dsl 4.6.0 reports CUDA 12.9 while shipping the new bindings | overlay: feature-detect the binding signature instead of the version proxy |
| 6 | nv_dev `fp8_einsum` asserts `t.dim() == N`; `cooperative_topk launch failed: invalid argument`; triton `KeyError: 'float8_e8m0fnu'` | o_proj passes SM100 packed-ue8m0 layout; cooperative topk needs cluster launch GB10 lacks; triton JIT has no E8M0 dtype | three sm12x-gated overlays: o_proj scale/shape adapter → nv_dev API; route sm12x to the existing `persistent_topk`; exact e8m0→fp32 upcast |
| 7 | `deep_gemm_warmup` asserts `sfb_dtype == kFloat or kInt` | the linear warmup pass feeds raw e8m0 scales to `fp8_gemm_nt` | serve flag `VLLM_DEEP_GEMM_WARMUP=skip` |

## Build

```bash
docker build -t aeon-vllm-ultimate:deepseek-v4-gb10 -f Dockerfile.deepseek-v4-gb10-layer .
```

## Serve (2 nodes, TP2 + EP over RoCE)

Head (rank 0) — swap in your fabric IPs/ifaces; worker (rank 1) is identical plus `--headless`
and `--node-rank 1`:

```bash
docker run -d --name vllm-ds4 --runtime nvidia --gpus all --ipc host --network host \
  --shm-size 16g --cap-add=SYS_PTRACE --cap-add=IPC_LOCK --ulimit memlock=-1:-1 \
  --device=/dev/infiniband \
  -v /path/to/models:/models -v /path/to/flashinfer-cache:/root/.cache/flashinfer \
  -v /path/to/triton-cache:/root/.triton \
  -e VLLM_HOST_IP=<this-node-fabric-ip> \
  -e NCCL_IB_HCA=<roce-hca> -e NCCL_IB_GID_INDEX=3 -e NCCL_IB_DISABLE=0 \
  -e NCCL_SOCKET_IFNAME=<fabric-if> -e GLOO_SOCKET_IFNAME=<fabric-if> \
  -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True -e TORCH_CUDA_ARCH_LIST=12.1a \
  -e VLLM_ALLOW_LONG_MAX_MODEL_LEN=1 -e FLASHINFER_DISABLE_VERSION_CHECK=1 \
  -e VLLM_DEEP_GEMM_WARMUP=skip \
  -e VLLM_RPC_TIMEOUT=3600000 -e TRITON_CACHE_DIR=/root/.triton \
  -e VLLM_USE_BREAKABLE_CUDAGRAPH=0 -e VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=3600 \
  --entrypoint vllm aeon-vllm-ultimate:deepseek-v4-gb10 \
  serve /models/deepseek-ai/DeepSeek-V4-Flash \
  --served-model-name DeepSeek-V4-Flash --trust-remote-code --tokenizer-mode deepseek_v4 \
  --tensor-parallel-size 2 --enable-expert-parallel \
  --moe-backend marlin --linear-backend triton \
  --distributed-executor-backend mp --nnodes 2 --node-rank 0 \
  --master-addr <head-fabric-ip> --master-port 29519 \
  --kv-cache-dtype fp8 --block-size 256 --enable-prefix-caching \
  --max-model-len 65536 --max-num-seqs 8 --max-num-batched-tokens 4096 \
  --gpu-memory-utilization 0.80 --no-enable-flashinfer-autotune \
  --compilation-config '{"cudagraph_mode":"PIECEWISE","custom_ops":["all"]}' \
  --reasoning-parser deepseek_v4 --enable-auto-tool-choice --tool-call-parser deepseek_v4 \
  --load-format safetensors --host 0.0.0.0 --port 8000
```

Notes:
- **First cold start JIT-compiles flashinfer 0.6.14 kernels (~20 min).** Launch both ranks
  together only after the cache is warm, or rank 1's Gloo barrier can time out. Persist
  `/root/.cache/flashinfer` and `/root/.triton`.
- `--moe-backend marlin` covers the MXFP4/fp8 expert checkpoints; `--linear-backend triton`
  covers the E8M0 fp8 linears (with the overlay upcast).
- `NCCL 2.28.9` in the base works cross-node over RoCE as-is (no LD_PRELOAD needed).
- **Known issue — FULL decode-graph replay desyncs NCCL between the two ranks.** Any
  configuration that replays FULL decode graphs (`FULL_AND_PIECEWISE` at default capture
  sizes, with breakable-cudagraph on or off, or even restricted `cudagraph_capture_sizes:
  [1,2]`) dies under batched long-context decode with rank 1 `c10::DistBackendError`
  ("Some NCCL operations have failed or timed out"). Collectives must stay uncaptured on
  this 2-node TP2 fabric: **`cudagraph_mode: PIECEWISE` is fully stable** (2×10 concurrent
  9K-context generations + a complete 12-category multi-turn benchmark, zero engine errors)
  and is what the numbers below were measured with.

## Validation (2× DGX Spark GB10, DeepSeek-V4-Flash-DSpark fp8, PIECEWISE + seqs 8)

- **Aggregate 66.4 tok/s** over 10 concurrent multi-turn conversations; single-stream
  16.6 tok/s; **48/48 benchmark turns, zero engine errors**.
- Engine init ~58 s (warm cache); GPU KV cache 782,838 tokens at util 0.80.
- Output quality graded at parity with an independent jasl-fork GB10 build of the same
  checkpoint (no garbling; temp-0 arithmetic/factual probes consistent).
- Stability: 4× and 10× concurrent 9K-context/800-token generations pass under eager and
  PIECEWISE; the batch>2 long-context deadlock present with the 4.5.x-era dependency set
  is fixed by the tvm-ffi 0.1.11 AOT-recompiled cutedsl kernels.
