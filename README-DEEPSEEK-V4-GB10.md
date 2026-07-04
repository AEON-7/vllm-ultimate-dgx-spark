# DeepSeek-V4-Flash on 2× DGX Spark (GB10 / sm_121a) — TP2 enablement layer

`Dockerfile.deepseek-v4-gb10-layer` makes the stock `aeon-vllm-ultimate:2026-07-01-v0.24.0`
image serve **DeepSeek-V4-Flash** across two DGX Sparks (TP=2, expert-parallel, RoCE).
The base already ships the native `vllm/models/deepseek_v4` package; on GB10 it dies on a
cascade of five gaps, each fixed here (validated end-to-end on 2× DGX Spark):

| # | Symptom on stock 0.24 (GB10) | Root cause | Fix in this layer |
|---|---|---|---|
| 1 | `AttributeError: cutlass.cute.core has no attribute 'ThrMma'` on **any** DeepSeek-V4 load | image ships `nvidia-cutlass-dsl 4.6.0`, which removed `ThrMma`; the base's own `deepseek_v4/nvidia/ops/fused_indexer_q_cutedsl.py` still uses it | pin `nvidia-cutlass-dsl==4.5.2` |
| 2 | DeepGEMM asserts `Unsupported architecture` (`hyperconnection.hpp`, `attention.hpp`, einsum `layout.hpp`) | vendored DeepGEMM has sm90/sm100 kernels only | install DeepGEMM `nv_dev` @ `a6b593d` ([deepseek-ai/DeepGEMM#324](https://github.com/deepseek-ai/DeepGEMM/pull/324): native sm120 kernels); site-packages takes precedence over the vendored copy |
| 3 | `trtllm_batch_decode_sparse_mla_dsv4() got an unexpected keyword 'swa_topk_lens'` | the base's `flashinfer_sparse.py` targets the 0.6.14 sparse-MLA API; image ships flashinfer 0.6.12 | upgrade flashinfer trio to 0.6.14 (python+cubin+jit-cache cu130/aarch64) |
| 4 | nv_dev `fp8_einsum` asserts `t.dim() == N`; `cooperative_topk launch failed: invalid argument`; triton `KeyError: 'float8_e8m0fnu'` | o_proj passes SM100 packed-ue8m0 layout; cooperative topk needs cluster launch GB10 lacks; triton JIT has no E8M0 dtype | three thin `.py` overlays, all gated to sm12x (no change on sm90/100): o_proj scale/shape adapter → nv_dev API; route sm12x to the existing `persistent_topk`; exact e8m0→fp32 scale upcast |
| 5 | `deep_gemm_warmup` asserts `sfb_dtype == kFloat or kInt`; batched long-context decode hangs (`sample_tokens` RPC timeout) under every cudagraph mode | linear warmup feeds raw e8m0 scales to `fp8_gemm_nt`; graph-replay + sparse-MLA decode wedge on GB10 (cf. [vllm#40969](https://github.com/vllm-project/vllm/issues/40969); see Known issue below) | serve flags: `VLLM_DEEP_GEMM_WARMUP=skip`, `--enforce-eager`, `--max-num-seqs 2` |

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
  --max-model-len 65536 --max-num-seqs 2 --max-num-batched-tokens 4096 \
  --gpu-memory-utilization 0.80 --no-enable-flashinfer-autotune \
  --enforce-eager \
  --reasoning-parser deepseek_v4 --enable-auto-tool-choice --tool-call-parser deepseek_v4 \
  --load-format safetensors --host 0.0.0.0 --port 8000
```

Notes:
- **First cold start JIT-compiles flashinfer 0.6.14 kernels (~20 min).** Launch both ranks
  together only after the cache is warm, or rank 1's Gloo barrier can time out
  (`Application timeout caused pair closure`). Persist `/root/.cache/flashinfer`.
- `--moe-backend marlin` covers the MXFP4/fp8 expert checkpoints; `--linear-backend triton`
  covers the E8M0 fp8 linears (with the overlay upcast).
- `NCCL 2.28.9` in the base works cross-node over RoCE as-is (no LD_PRELOAD needed).
- **Known issue — batched long-context decode wedges the engine; run `--enforce-eager`
  and `--max-num-seqs 2`.** With any cudagraph mode (FULL, PIECEWISE, or the base's
  auto-enabled breakable-cudagraph) the workers go silent during batched decode once
  accumulated contexts grow past a few thousand tokens; `sample_tokens` RPC times out and
  the engine dies. `VLLM_USE_BREAKABLE_CUDAGRAPH=0` extends survival ~6× and eager mode +
  `max_num_seqs 2` is fully stable (complete 12-category multi-turn benchmark, 48/48 turns,
  zero engine errors). Suspect: sparse-MLA decode path (nv_dev paged-MQA / persistent_topk /
  flashinfer trtllm sparse) at batch>2 with long contexts — needs kernel-level investigation.
  `VLLM_RPC_TIMEOUT=3600000` + `VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=3600` are belt-and-braces
  for first-contact Triton JIT spikes; persist `/root/.triton` so those compile only once.

## Validation (2× DGX Spark GB10, DeepSeek-V4-Flash-DSpark fp8)

- Engine init 58 s (warm cache); GPU KV cache **782,838 tokens** (gpu_mem_util 0.80).
- Numerically sane at temp 0 (exact-arithmetic + factual probes match a known-good
  independent GB10 build within phrasing variance); coherent long-form generation; a full
  12-category multi-turn benchmark graded at quality parity with that independent build.
- Final config (eager, max_num_seqs 2): **complete benchmark run with zero engine errors —
  48/48 turns**, single-stream decode **17.8 tok/s** (TTFT ~2.8 s), **28.6 tok/s aggregate**
  across 10 concurrent multi-turn categories.
- 2×10 concurrent 1500-token generations: 20/20 HTTP 200, engine alive.
