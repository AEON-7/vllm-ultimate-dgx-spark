# AEON vLLM Ultimate — `0.27.1-omni`

**One image. Text, vision, audio-in, speech-out, and every speculative-decoding
family that exists for this hardware — on a single DGX Spark.**

`vLLM 0.27.1+aeon.sm121a.dspark` · built from source for GB10 / sm_121a · aarch64 · CUDA 13

---

## What's new

### 1. The full omni stack is in the image

`vllm-omni 0.27.0rc1` is layered on top of the patched engine, which brings the
**Thinker → Talker → Code2Wav** pipeline — speech *output*, not just speech input.
The overlay is purely additive: it pins no `vllm`, `torch`, or `transformers`
version, so none of the AEON carries are disturbed. The build gate proves it
rather than assuming it:

```
nvfp4_kv_cache_split_views   present
_nvfp4_split_data_scale      no as_strided regression
cuda_graph                   thread_local capture intact
```

Ten speech-output model families are registered by the overlay, including
Qwen3-Omni, Qwen2.5-Omni, MiniCPM-o 4.5, Step-Audio 2, MiMo-Audio and the
Qwen3-TTS family (with its voice store: `/v1/audio/voices`, zero-shot cloning,
natural-language voice design, and WebSocket streaming).

> **Note:** stock vLLM cannot do speech output at all — its own loaders strip the
> voice half (`qwen3_omni_moe_thinker.py` loads with
> `skip_prefixes=["talker.", "code2wav."]`). The omni overlay is what puts it back.

### 2. Audio input, properly

`av` (PyAV), `soundfile`, `librosa` and `soxr` are baked in, so real-world audio
actually decodes and resamples. Without PyAV, a 24 kHz clip fails with a
misleading `400 "Invalid or unsupported audio file"` — a 16 kHz test tone passes
because it never needs resampling, which is exactly how this hides. The image
gate now forces a genuine 24 kHz → 16 kHz resample.

### 3. DFlash 2 — first image anywhere to ship it on 0.27.1

Upstream [vLLM #52816](https://github.com/vllm-project/vllm/pull/52816) merged
2026-08-21, *after* 0.27.1. It is cherry-picked here as a pure-Python layer
(10 files, no kernels), which makes the official
[`incoai/Qwen3.8-27B-DFlash2`](https://huggingface.co/incoai/Qwen3.8-27B-DFlash2)
drafter loadable (mirror: `z-lab/Qwen3.8-27B-DFlash2`).

The backport needs **two** fixes, and shipping only the first silently degrades
DFlash2 into DFlash1 with no error:

| fix | why |
|---|---|
| `layer_type` accepted + forwarded on `DFlash2Qwen3DecoderLayer` | the PR targets post-0.27.1 `main`, where the base layer no longer takes it; 0.27.1 passes it per-layer → `TypeError` on load |
| top-level `is_causal` precedence in `_dflash_layer_causal` | the checkpoint declares `is_causal: false`; without this the legacy rule runs the drafter **causal**, the candidate selector never fires, and you get DFlash1 behaviour under a DFlash2 label |

Verify it is genuinely active:

```bash
docker run --rm -v <drafter>:/d:ro --entrypoint python3 <image> -c "
import json, types
from vllm.model_executor.models.qwen3_dflash import dflash_has_any_non_causal
cfg = types.SimpleNamespace(**json.load(open('/d/config.json')))
print('DFlash2 active:', dflash_has_any_non_causal(cfg))"
```

### 4. Every speculative family, one image

| family | status | notes |
|---|---|---|
| **DFlash 2** | ✅ | Qwen3.8 only — no Gemma drafter exists |
| **DFlash** | ✅ | Qwen + Gemma |
| **DSpark** | ✅ | quantized Markov heads; needs `--mamba-cache-mode align` |
| **MTP** | ✅ | self-speculation, no external drafter |
| Eagle / Eagle3 | ✅ | inherited from upstream |

### 5. TP=2 across two Sparks, with RDMA proven by counters

Cross-node CUDA graphs work — capture **and** replay — contradicting upstream
[#46253](https://github.com/vllm-project/vllm/issues/46253), which reports that
multi-node GB10 must run `--enforce-eager`. The load-bearing carry is
[#48053](https://github.com/vllm-project/vllm/pull/48053)'s `thread_local`
capture-error-mode, extended to every `torch.cuda.graph` site.

**Correction to earlier releases:** this was previously credited to NCCL 2.30.7
with a stated floor of 2.30.4. That is wrong. The shipping image runs
**NCCL 2.29.7** and cross-node capture + replay held for 5+ hours. There is no
2.30.x floor.

RDMA is verified by counters, not by log-grepping — a 400-token generation moved:

```
port_xmit_data   +3,210 MB      (InfiniBand/RoCE)
netdev tx_bytes  +0 MB          (TCP)
```

### 6. 1M context on Qwen3.8-27B via official YaRN

Native window is 262,144; Qwen's own card prescribes YaRN factor 4.0. Two
non-obvious details decide whether it works:

```bash
# rope_parameters, NOT rope_scaling -- and NESTED under text_config.
# A flat override is a silent no-op that reports 1M and serves garbage.
--hf-overrides '{"text_config":{"rope_parameters":{
    "mrope_interleaved":true,"mrope_section":[11,11,10],
    "rope_type":"yarn","rope_theta":10000000,
    "partial_rotary_factor":0.25,"factor":4.0,
    "original_max_position_embeddings":262144}}}' \
--max-model-len 1000000
```

**Deliberately do NOT set `VLLM_ALLOW_LONG_MAX_MODEL_LEN`.** At factor 4.0 the
derived length is 1,048,576, so 1,000,000 validates on its own — which turns a
silently-dropped override into a loud refusal to boot instead of a server quietly
serving a broken 1M window.

Verified by needle-in-a-haystack: **3/3 at 271,904 tokens** and **3/3 at 543,761
tokens** — both past the native window.

> **Speculative decoding and >262K context are mutually exclusive.** The drafter
> builds its rope cache from *its own* config (262,144 rows) and `--hf-overrides`
> is target-only, so at position 262,144 a Triton kernel asserts
> `index out of bounds: ... < 262144` and takes down the engine. vLLM's guard
> (`_maybe_override_draft_max_position_embeddings`) is gated to `("eagle","eagle3")`
> and does not cover `dflash`. Dropping the drafter more than doubles the KV pool
> (2,167,255 → 5,244,929 tokens).

### 7. `mistral_common` fix

Every prior image — including the original 50.6 GB build — shipped
`mistral_common 1.11.1` while vLLM 0.27.1 requires `>=1.11.6`. Any import of
`vllm.tokenizers.mistral` raised `NameError: SpecialTokens`, breaking Mistral and
Voxtral models. The Qwen/Gemma fleet never touches that path, which is why it went
unnoticed. Now pinned and gated.

---

## Gotchas this release documents

Silent failures — they produce no error, only worse results:

1. **FlashInfer downgrades your CUDA graphs.** With speculative decoding, if the
   *draft* path lands on FlashInfer (`UNIFORM_SINGLE_TOKEN_DECODE`), vLLM logs
   `setting cudagraph_mode=PIECEWISE` and you lose every FULL graph. A top-level
   `--attention-backend TRITON_ATTN` does **not** fix it — the backend must also be
   set **inside** `--speculative-config`. Healthy signature: **6 FULL + 13 PIECEWISE**.
2. **Nested rope override** — see §6.
3. **Drafter rope cap at 262,144** — see §6.
4. **The two-part DFlash2 backport** — see §3.
5. **MAL, not acceptance %, is the correct optimand** for choosing
   `num_speculative_tokens`. Acceptance % falls mechanically as n rises even while
   throughput rises; picking by acceptance selects the slower arm. MAL needs
   `vllm:spec_decode_num_drafts_total` — note the `vllm:` prefix.

---

## Hardware limits (GB10, 121 GB unified LPDDR5X)

| knob | safe | at the limit | hard ceiling |
|---|---|---|---|
| `--gpu-memory-utilization` | **0.70–0.75** | 0.80 (sole workload) | **0.82** |
| `--max-num-seqs` | 4–8 | 64 | KV-bound, see below |
| `--max-num-batched-tokens` | 8192 | 16384 | mandatory with any drafter |

Above 0.82 the box does not OOM — it goes catatonic, because the unified pool is
shared with the OS and page cache. Under TP=2, subtract ~0.03.

**The drafter's KV footprint is what caps real concurrency**, and it is the single
most useful number for sizing:

| arm | KV pool | engine max concurrency |
|---|---|---|
| baseline | 1,521,371 | 92.9× |
| MTP n=3 | 832,022 | 50.8× |
| DFlash2 n=3 | 653,462 | 39.9× |
| DFlash2 n=7 | 464,675 | 28.4× |
| DFlash n=12 | 328,590 | 20.1× |

*(measured at 16K context, `--max-num-seqs 64`, util 0.75, fp8 KV, Qwen3.8-27B NVFP4-GDNFP8)*

---

## Performance

Benchmarks on the `0.27.1-omni` image are in flight; numbers land here rather than
being estimated. `<PENDING: omni-image sweep, 6 categories x c ∈ {1,8,16,32,64}>`

Established on the same matrix with the pre-omni image, for reference:
DFlash2 beat every alternative at every concurrency level on Qwen3.8-27B —
**3.39× at c=1** (n=7) and **1.41× at c=64** (n=3) — and Gemma-4-26B-A4B with
DFlash n=10 reached **1,296 tok/s** aggregate at c=64.

**Choose `n` against your concurrency target:** high n for interactive
single-stream, low n for batch. And note that **DFlash n=12 is actively harmful
above ~c=16** (0.53× and a 22.6 s TTFT at c=64) — if you are running the older
n=12 recipe at concurrency, that is a regression, not a speedup.
