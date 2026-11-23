# AEON vLLM Ultimate 0.33.0-omni-dflash2 (2026-11-23)

Image: `ghcr.io/aeon-7/aeon-vllm-ultimate:2026-11-23-v0.33.0-omni-dflash2`
Digest: `sha256:71c2b82b3b0c97ddca9ef35b279fb745ee0abe29b762a24c8c0771cf930f11ab`

## What changed

This build is `2026-11-13-v0.33.0-omni` (vLLM 0.33.0, built for DGX Spark GB10 / sm_121a, CUDA 13) plus one fix: DFlash2 speculative decoding now works on Qwen3.8-Flash-Next (the `qwen4_exp` model family).

- `vllm/models/qwen4_exp/nvidia/model.py`: the model now exposes the EAGLE3-style hidden-state hook DFlash2 reads. Both the text and the vision-language classes implement `SupportsEagle3`. After each target layer, the hook captures that layer's finished output, averaged over the hyper-connection streams, and gathers it across ranks when HC sequence parallelism is on. The default target layers are 6, 16, 26, 36 and 46, matching the drafter's `target_layer_ids` [5, 15, 25, 35, 45] plus one.
- `vllm/model_executor/models/config.py`: `dflash` is added to the speculative methods allowed for Qwen4Exp.

The full diff is in `patches/aeon-dflash2-qwen4exp.patch`, and the overlay build is in `flashnext-dflash2/Dockerfile`. Nothing else changed from 0.33.0-omni. When DFlash2 is active, the startup log prints `AEON_DFLASH2_CAPTURE`.

## Flash-Next recipe (2 Sparks, TP=2)

Run on both nodes, with `RANK=0` on the head node and `RANK=1` on the other:

```sh
IMG=ghcr.io/aeon-7/aeon-vllm-ultimate:2026-11-23-v0.33.0-omni-dflash2
MODEL=/models/your-flash-next-nvfp4      # safetensors checkpoint
DRAFT=/models/dflash2                    # incoai/Qwen3.8-Flash-Next-DFlash2
HEAD=10.100.200.1; MYIP=10.100.200.1; RANK=0; IF=enp1s0f0np0
HEADLESS=$([ $RANK = 0 ] || echo --headless)

docker run -d --name flashnext --network host --ipc host --shm-size 16g --gpus all \
  --device /dev/infiniband --ulimit memlock=-1 \
  -e NCCL_SOCKET_IFNAME==$IF -e GLOO_SOCKET_IFNAME=$IF -e TP_SOCKET_IFNAME=$IF \
  -e NCCL_IB_HCA==rocep1s0f0,roceP2p1s0f0 -e NCCL_IB_GID_INDEX=3 -e NCCL_IB_DISABLE=0 \
  -e NCCL_IB_ROCE_VERSION_NUM=2 -e NCCL_IB_ADDR_FAMILY=AF_INET -e VLLM_HOST_IP=$MYIP \
  -v $MODEL:/model:ro -v $DRAFT:/draft:ro \
  --entrypoint vllm $IMG serve /model --trust-remote-code --served-model-name flashnext \
  --tensor-parallel-size 2 --nnodes 2 --node-rank $RANK \
  --master-addr $HEAD --master-port 29531 --distributed-executor-backend mp \
  --language-model-only --host 0.0.0.0 --port 8000 \
  --reasoning-parser qwen3 --tool-call-parser qwen3_coder --enable-auto-tool-choice \
  --max-model-len 131072 --max-num-seqs 16 --gpu-memory-utilization 0.80 \
  --max-num-batched-tokens 16384 \
  --speculative-config '{"method":"dflash","model":"/draft","num_speculative_tokens":3}' $HEADLESS
```

For the built-in MTP head instead, drop the `/draft` mount and use `--speculative-config '{"method":"mtp","num_speculative_tokens":3}'`.

Use 3 (or 11) speculative tokens. The model's attention kernel needs draft count + 5 to divide its 1,664-token block, so 7 fails at startup.

## Measured (AEON Flash-Next NVFP4-XS, 2 x DGX Spark, greedy, 512 output tokens)

| setup | single user, 12-category mean | 16 streams aggregate | 100K prompt TTFT / decode |
|---|---|---|---|
| DFlash2, 3 drafts | 65.1 tok/s | 357.8 tok/s | 45.4 s / 51.1 tok/s |
| MTP, 3 drafts | 58.2 tok/s | 358.6 tok/s | 41.3 s / 58.6 tok/s |
| no drafter | 34.3 tok/s | 253.6 tok/s | 38.9 s / 35.2 tok/s |

With DFlash2, each verify step yields 3.06 tokens on average out of a possible 4 (per-position acceptance 83% / 67% / 56%). Two TP=2 replicas on four Sparks reached 748 tok/s at 32 streams.

Release smoke test on this exact image: two TP=2 replicas run at once from the same checkpoint, 12 categories at 1 stream plus 16 streams on 4 categories, 0 errors.

| recipe | single user, 12-category mean | 16 streams, 4-category mean | DFlash2 mean acceptance length |
|---|---|---|---|
| DFlash2, 3 drafts | 64.2 tok/s (patched dev run: 65.1) | 364.8 tok/s (dev run: 357.8) | 2.90-3.13 of 4 |
| MTP, 3 drafts | 59.5 tok/s (dev run: 58.2) | 361.1 tok/s (dev run: 358.6) | - |

vLLM TP=2 greedy decoding is not bit-reproducible from run to run on this stack. Two unpatched runs of the same image agree on 2-3 of 12 categories, and the new image agrees with the dev runs at that same level.

Model card with every recipe and table: https://huggingface.co/AEON-7/Aeon3.8-Flash-Next-ULTIMATE-UNCENSORED-NVFP4-XS
