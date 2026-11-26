# AEON vLLM Ultimate 0.33.0-omni-dflash2 (2026-11-23)

Stock image: `ghcr.io/aeon-7/aeon-vllm-ultimate:2026-11-23-v0.33.0-omni-dflash2`
Digest: `sha256:71c2b82b3b0c97ddca9ef35b279fb745ee0abe29b762a24c8c0771cf930f11ab`

1-Spark disk-offload image: `ghcr.io/aeon-7/aeon-vllm-ultimate:2026-11-25-v0.33.0-omni-dflash2-ngram-disk`
Digest: `sha256:4749bd067524ce41a4a6d59625f6b6c8531d08a14739bd0d416e935b9fc3a1ad`

## What changed

This build is `2026-11-13-v0.33.0-omni` (vLLM 0.33.0, built for DGX Spark GB10 / sm_121a, CUDA 13) plus one fix: DFlash2 speculative decoding now works on Qwen3.8-Flash-Next (the `qwen4_exp` model family).

- `vllm/models/qwen4_exp/nvidia/model.py`: the model now exposes the EAGLE3-style hidden-state hook DFlash2 reads. Both the text and the vision-language classes implement `SupportsEagle3`. After each target layer, the hook captures that layer's finished output, averaged over the hyper-connection streams, and gathers it across ranks when HC sequence parallelism is on. The default target layers are 6, 16, 26, 36 and 46, matching the drafter's `target_layer_ids` [5, 15, 25, 35, 45] plus one.
- `vllm/model_executor/models/config.py`: `dflash` is added to the speculative methods allowed for Qwen4Exp.

The full diff is in `patches/aeon-dflash2-qwen4exp.patch`, and the overlay build is in `flashnext-dflash2/Dockerfile`. Nothing else changed from 0.33.0-omni. When DFlash2 is active, the startup log prints `AEON_DFLASH2_CAPTURE`.

The 2026-11-23 tag does not include file-backed n-gram offload. Use `2026-11-25-v0.33.0-omni-dflash2-ngram-disk` for 1-Spark vLLM (the n-gram table is mounted at `/ple`; the offload code is in the image). Do not run the 1-Spark commands below on the 2026-11-23 stock image alone; it will not fit. 2-Spark and 4-Spark recipes below can stay on the 2026-11-23 tag.

## Recommended sampling

```
temperature 0.6
top_p 0.95
top_k 20
presence_penalty 0.5
repetition_penalty 1.0
```

Set them as a server default with `--override-generation-config`. Leave prefix caching on. Do not pass `--no-enable-prefix-caching`. `repetition_penalty` stays at 1.0. Strata and curl clients should send the same fields.

## Which engine?

If you value agent swarms, use vLLM with `ghcr.io/aeon-7/aeon-vllm-ultimate:2026-11-25-v0.33.0-omni-dflash2-ngram-disk`. If you prioritize single-stream performance, use Strata 0.2.18 (engine 0.1.64).

## Flash-Next recipe: 1 Spark, Strata (DFlash2 recommended)

The pack at the root of the XS repo, Strata 0.2.18 / engine 0.1.64, DFlash2 `--spec 6 --draft-conf 0.5`, `--parallel 4`, 131K context, vision on, KV int8, `--kv-resident 32768`. Same-settings A/B, 25 November 2026 (c1 / c4 tok/s):

| mode | Strata MTP | Strata DFlash2 | vLLM DFlash2, 4 sequences |
|---|---|---|---|
| Greedy, thinking off | 52.2 / 67.9 | 60.6 / 73.0 | 38.9 / 97.5 |
| Published sampling, thinking off | 47.1 / 64.3 | 55.2 / 66.4 | 36.2 / 79.3 |
| Published sampling, thinking on | 39.0 / 63.1 | 45.3 / 67.0 | 30.3 / 70.7 |

A `--parallel 32` cap on one Spark only fits after dropping context to 32K, `--kv-resident` 8192, conversation cache 1024 MiB, and vision off. Even then it is only safe to about 8 concurrent generations (12 sessions dropped below 5 GiB free; 16 hit 3.74 GiB and were stopped). The engine still decodes 8-wide.

## Flash-Next recipe: 1 Spark, vLLM disk offload (4 sequences, low latency)

Use the `ngram-disk` tag. Mount a host directory at `/ple` for the NVMe n-gram table (created on first boot, reused later). Do not bind-mount Python patch files.

```sh
IMG=ghcr.io/aeon-7/aeon-vllm-ultimate:2026-11-25-v0.33.0-omni-dflash2-ngram-disk
MODEL=/models/aeon-xs/vllm
DRAFT=/models/dflash2
TABLE=/models/aeon-xs-ple

docker run -d --name aeon-xs --ipc host --shm-size 16g --gpus all \
  -v $MODEL:/model:ro -v $DRAFT:/draft:ro -v $TABLE:/ple \
  --entrypoint vllm $IMG serve /model --trust-remote-code --served-model-name aeon-xs \
  --host 0.0.0.0 --port 8000 \
  --engram-config '{"cpu_offload":true}' \
  --limit-mm-per-prompt '{"image":2,"video":0}' \
  --reasoning-parser qwen3 --tool-call-parser qwen3_coder --enable-auto-tool-choice \
  --enable-prefix-caching \
  --max-model-len 131072 --max-num-seqs 4 --gpu-memory-utilization 0.74 \
  --max-num-batched-tokens 16384 \
  --speculative-config '{"method":"dflash","model":"/draft","num_speculative_tokens":3}' \
  --override-generation-config '{"temperature":0.6,"top_p":0.95,"top_k":20,"presence_penalty":0.5,"repetition_penalty":1.0}'
```

Leave utilization at 0.74. Use 3 draft tokens (11 also starts; 7 does not).

## Flash-Next recipe: 1 Spark, vLLM disk offload (32 sequences, swarm)

Same command with `--max-num-seqs 32`. KV at this utilization is about 8.7 GiB / 232K tokens (1.77x at a full 131K request). 32 short sessions fit; 32 full 131K contexts do not. Warm published-sampling ladder (256 new tokens): thinking off peaks at 86.4 tok/s aggregate (c=24) and is 83.5 at c=32; thinking on reaches 105.6 at c=32 with 0 loops. TTFT p95 stays under 1 s through 12 sessions, then grows (about 27 s at 16, 55 to 73 s at 32). Thinking-off loops appeared at c>=12 (1/12, 1/16, 2/24, 3/32). The 32-sequence cap also costs some low-concurrency speed vs the 4-sequence recipe (sampled off: 31.2 / 64.2 at c1 / c4, against 36.2 / 79.3).

## Flash-Next recipe: 2 Sparks, throughput (131K, 32 sequences, DFlash2 k=3)

Run on both nodes, with `RANK=0` on the head node and `RANK=1` on the other. The default is DFlash2 with 3 draft tokens and 32 sequences. This is the same image behind the published 2-Spark speed tables (23 November 2026). The `-e NCCL_SOCKET_IFNAME==$IF` line is intentional: NCCL treats a leading `=` as an exact interface match.

```sh
IMG=ghcr.io/aeon-7/aeon-vllm-ultimate:2026-11-23-v0.33.0-omni-dflash2
MODEL=/models/aeon-xs/vllm
DRAFT=/models/dflash2                    # incoai/Qwen3.8-Flash-Next-DFlash2
HEAD=<head-node-ip>; MYIP=<this-node-ip>; RANK=0; IF=<roce-if>; HCA=<roce-hca>
HEADLESS=$([ $RANK = 0 ] || echo --headless)

docker run -d --name flashnext --network host --ipc host --shm-size 16g --gpus all \
  --device /dev/infiniband --ulimit memlock=-1 \
  -e NCCL_SOCKET_IFNAME==$IF -e GLOO_SOCKET_IFNAME=$IF -e TP_SOCKET_IFNAME=$IF \
  -e NCCL_IB_HCA==$HCA -e NCCL_IB_GID_INDEX=3 -e NCCL_IB_DISABLE=0 \
  -e NCCL_IB_ROCE_VERSION_NUM=2 -e NCCL_IB_ADDR_FAMILY=AF_INET -e VLLM_HOST_IP=$MYIP \
  -v $MODEL:/model:ro -v $DRAFT:/draft:ro \
  --entrypoint vllm $IMG serve /model --trust-remote-code --served-model-name flashnext \
  --tensor-parallel-size 2 --nnodes 2 --node-rank $RANK \
  --master-addr $HEAD --master-port 29531 --distributed-executor-backend mp \
  --limit-mm-per-prompt '{"image":2,"video":0}' --host 0.0.0.0 --port 8000 \
  --reasoning-parser qwen3 --tool-call-parser qwen3_coder --enable-auto-tool-choice \
  --enable-prefix-caching \
  --max-model-len 131072 --max-num-seqs 32 --gpu-memory-utilization 0.80 \
  --max-num-batched-tokens 16384 \
  --speculative-config '{"method":"dflash","model":"/draft","num_speculative_tokens":3}' \
  --override-generation-config '{"temperature":0.6,"top_p":0.95,"top_k":20,"presence_penalty":0.5,"repetition_penalty":1.0}' \
  $HEADLESS
```

For the MTP alternate, omit the `/draft` mount, use `--max-num-seqs 16`, and replace the speculative config with `'{"method":"mtp","num_speculative_tokens":3}'`. For a text-only server, use `--language-model-only` in place of `--limit-mm-per-prompt`.

Use 3 speculative tokens. The model's attention kernel needs draft count + 5 to divide its 1,664-token block, so 7 and 8 fail at startup. 11 starts and is slower.

Published sampling, 25 November 2026, same-harness short-prompt ladder (256 new tokens). DFlash2 k=3 at 32 sequences, thinking off: 59.7 / 141.5 / 183.0 / 247.9 / 242.9 tok/s at 1 / 4 / 8 / 16 / 32 sessions (accept about 0.50 to 0.57). Thinking on: 56.8 / 131.0 / 175.2 / 235.1 / 312.5 (accept about 0.45 to 0.47). MTP at 16 sequences, thinking off: 35.0 / 97.1 / 166.3 / 230.0. Thinking on: 33.4 / 88.2 / 168.0 / 231.9. An earlier four-workload MTP pass at 16 sequences was 58.4 / 329.9 thinking off and 47.3 / 279.0 thinking on. KV cache on the DFlash2 serve was 1,688,812 tokens (12.88x at 131K), so 32 full 131K contexts still do not fit.

## Flash-Next recipe: 2 Sparks, quality / long-think (256K, 4 sequences)

Same image and weights.

```sh
# same docker flags as the throughput recipe, except:
  --enable-prefix-caching \
  --max-model-len 262144 --max-num-seqs 4 --gpu-memory-utilization 0.80 \
  --max-num-batched-tokens 16384 \
  --speculative-config '{"method":"mtp","num_speculative_tokens":3}' \
  --override-generation-config '{"temperature":0.6,"top_p":0.95,"top_k":20,"presence_penalty":0.5,"repetition_penalty":1.0}' \
  $HEADLESS
```

Keep a memory buffer on each node (back off under about 4 GiB free, prefer 5+).

Published sampling, 25 November 2026, MTP, thinking off: 51.3 tok/s for one user, 144.8 tok/s aggregate across 4. Thinking on: 43.4 / 100.2. Thinking-on no-loop check at 2048 new tokens: 5/5 clean. A 24-case long-think quality suite on this serve scored 14/24.

## 4 Sparks

Two copies of the 2-Spark throughput recipe behind a balancer reached 748 tok/s at 32 streams (23 November 2026, one DFlash2 replica and one MTP replica). A single tensor-parallel-size 4 job was not measured for XS and is not a published recipe yet.

## Measured (AEON Flash-Next NVFP4-XS, 2 x DGX Spark, greedy, thinking off, 512 output tokens)

The 23 November 2026 showdown means below are 11-workload decode averages (the short-QA latency probe is excluded). Earlier wording that called 65.1 a "12-category mean" was imprecise.

| setup | single user, 11-workload mean | 16 streams aggregate | 100K prompt TTFT / decode |
|---|---|---|---|
| DFlash2, 3 drafts | 65.1 tok/s | 357.8 tok/s | 45.4 s / 51.1 tok/s |
| MTP, 3 drafts | 58.2 tok/s | 358.6 tok/s | 41.3 s / 58.6 tok/s |
| no drafter | 34.3 tok/s | 253.6 tok/s | 38.9 s / 35.2 tok/s |

With DFlash2, each verify step yields 3.06 tokens on average out of a possible 4 (per-position acceptance 83% / 67% / 56%). Two TP=2 replicas on four Sparks reached 748 tok/s at 32 streams.

Release smoke test on this exact image: two TP=2 replicas run at once from the same checkpoint, 12 categories at 1 stream plus 16 streams on 4 categories, 0 errors.

| recipe | single user | 16 streams, 4-category mean | DFlash2 mean acceptance length |
|---|---|---|---|
| DFlash2, 3 drafts | 64.2 tok/s (patched dev run: 65.1) | 364.8 tok/s (dev run: 357.8) | 2.90-3.13 of 4 |
| MTP, 3 drafts | 59.5 tok/s (dev run: 58.2) | 361.1 tok/s (dev run: 358.6) | - |

vLLM TP=2 greedy decoding is not bit-reproducible from run to run on this stack. Two unpatched runs of the same image agree on 2-3 of 12 categories, and the new image agrees with the dev runs at that same level.

Model card with every recipe, the 1-Spark ladder, and the sampling defaults: https://huggingface.co/AEON-7/Aeon3.8-Flash-Next-ULTIMATE-UNCENSORED-NVFP4-XS
