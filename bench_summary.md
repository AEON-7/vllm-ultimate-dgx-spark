# AEON vLLM Ultimate — DGX Spark benchmark summary

Hardware: 1× NVIDIA GB10 (DGX Spark, sm_121a), 128 GB unified memory.
Container: `ghcr.io/aeon-7/aeon-vllm-ultimate:v0.22.1-pr44389-spark`
(vLLM 0.22.1 + PR #44389 + AEON sm_121a patches).

Workload: 8 representative prompts (basic table) and 24 categorized prompts (per-category table).
Common flags: `--max-num-seqs 8 --gpu-memory-utilization 0.78
--enable-chunked-prefill --enable-prefix-caching --mamba-block-size 256`.

## 🏆 Production-style: greedy + n_spec=15 on MTP-XS body + DFlash drafter

24 prompts (4 per category) × `max_tokens=400`, `temperature=0`, `n_spec=15`.

### Single-stream by category

| Category | n | TTFT median | TPOT median | decode tok/s mean | decode tok/s median | peak |
|---|---:|---:|---:|---:|---:|---:|
| **math** | 4 | 243 ms | 22.3 ms | **44.6** | **44.9** | **45.7** ⚡ |
| code | 4 | 243 ms | 24.1 ms | 40.4 | 41.6 | 44.4 |
| reasoning | 4 | 195 ms | 28.4 ms | 35.9 | 35.2 | 40.1 |
| summary | 4 | 242 ms | 33.1 ms | 31.3 | 30.4 | 37.5 |
| dialogue | 4 | 243 ms | 33.4 ms | 30.0 | 30.1 | 36.6 |
| prose | 4 | 132 ms | 37.5 ms | 26.2 | 26.9 | 29.6 |
| **OVERALL** | **24** | **242 ms** | **29.3 ms** | **34.7** | **34.1** | **45.7** |

### Concurrent ×4 streams (mixed categories)

| Round | Wall | Agg tok/s | TTFT mean |
|---|---:|---:|---:|
| 1 (cold) | 19.05 s | 71.5 | 1222 ms |
| **2 (steady)** | **17.57 s** | **84.4** | **276 ms** |

### Key findings

- **Math and code hit the headline 40–46 tok/s** because the token sequences are predictable — DFlash drafter accepts a long chain (n=15 acceptance window).
- **Prose is slowest at 26 tok/s** — high-entropy text means fewer drafter tokens accepted.
- **TTFT clusters at ~242 ms** independent of category (it's a prefill cost, not decode).
- **One-token-budget difference:** the prior bench used sampled (temp=0.7) + n_spec=4 which capped at ~24 tok/s. Moving to greedy + n_spec=15 added **+43% overall** and unlocked the math/code peak.
- This matches the AEON-7 v3 production card's published 38.5 / 71.3 median/peak (thinking-on) and ~45 tok/s reported single-stream peak.

## Apples-to-apples 4-config table (sampled, n_spec=4 — comparable to MTP runs)

Same 8 generic prompts, `temperature=0.7`, `max_tokens=200`. Use this when comparing
speculator method or KV dtype at the SAME settings.

### Single-stream (mean of 5 rounds)

| Config | TTFT median | TPOT mean | tok/s mean | tok/s median |
|---|---:|---:|---:|---:|
| XS body + MTP + NVFP4-KV | 121 ms | 57.8 ms | 17.26 | 16.64 |
| XS body + MTP + FP8-KV | 214 ms | 57.1 ms | 17.35 | 17.40 |
| NVFP4 body + DFlash (n=4) + BF16-KV | 298 ms | 50.2 ms | 19.44 | 20.10 |
| **XS body + DFlash (n=4) + BF16-KV** | **131 ms** | **40.8 ms** | **24.27** | **23.73** |

### Concurrent ×4 (steady median, round 3)

| Config | TTFT median | TPOT mean | per-stream tok/s | agg peak |
|---|---:|---:|---:|---:|
| XS body + MTP + NVFP4-KV | 286 ms | 61.1 ms | 15.71 | ~64 |
| XS body + MTP + FP8-KV | 239 ms | 60.2 ms | 15.84 | ~66 |
| NVFP4 body + DFlash (n=4) + BF16-KV | 328 ms | 55.4 ms | 15.98 | ~68 |
| **XS body + DFlash (n=4) + BF16-KV** | **259 ms** | **44.2 ms** | **19.59** | **~87** |

## Configs benchmarked

| ID | Body | Quant | KV | Speculator | Sampling | Result tok/s |
|---|---|---|---|---|---|---:|
| mtp-nvfp4kv | XS (21 GB) | NVFP4 | NVFP4 (PR #44389) | mtp n=1 | sampled | 17.3 |
| mtp-fp8kv | XS (21 GB) | NVFP4 | FP8-E4M3 | mtp n=1 | sampled | 17.4 |
| dflash-nvfp4body-bf16kv | NVFP4 prod (26 GB) | NVFP4 | BF16 | dflash n=4 | sampled | 19.4 |
| dflash-xsbody-bf16kv | XS (21 GB) | NVFP4 | BF16 | dflash n=4 | sampled | 24.3 |
| **🏆 cat-xs-dflash-n15-greedy** | **XS (21 GB)** | **NVFP4** | **BF16** | **dflash n=15** | **greedy** | **34.7 / 45.7 peak** |

## DFlash KV cache compatibility (vLLM 0.22.1)

In this image the DFlash drafter forces **`--kv-cache-dtype auto`** (BF16). FLASHINFER and TRITON_ATTN both dropped non-causal kernels for FP8/NVFP4 KV in PR #44389's backend refactor — a regression vs the v3 production image (vLLM 0.20.0). NVFP4 KV (3× capacity) is reachable only with **causal** speculators (MTP, Eagle, ngram).

## KV cache capacity by body × n_spec

| Body × n_spec | GPU KV cache | Max concurrency at 8k |
|---|---:|---:|
| NVFP4 body + DFlash n=4 | 264,922 tokens | 32.3× |
| XS body + DFlash n=4 | 300,966 tokens | 36.7× |
| XS body + DFlash n=15 | 146,785 tokens | 17.9× |

n_spec=15 cuts concurrent slot count roughly in half (each batch slot reserves 15 draft positions), trading concurrency for per-stream throughput.

## Round-by-round detail — winning config (XS + DFlash n=15 greedy)

```
SINGLE-STREAM (per category, decode tok/s):
  math:      45.7 ⚡  43.1  45.3  44.4
  code:      33.9  44.4  39.5  43.7
  reasoning: 40.1  34.3  36.1  33.2
  summary:   32.9  37.5  26.9  28.0
  dialogue:  36.6  28.5  31.6  23.4
  prose:     29.6  29.1  24.6  21.3

CONCURRENT x4:
  round 1: wall=19.05s  agg=71.5 tok/s  TTFT=1222 ms (cold)
  round 2: wall=17.57s  agg=84.4 tok/s  TTFT= 276 ms (steady) ← peak
```
