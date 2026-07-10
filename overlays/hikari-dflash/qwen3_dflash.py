# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import os
from collections.abc import Iterable, Mapping

import torch
import torch.nn.functional as F
from torch import nn
from transformers import Qwen3Config

from vllm import _custom_ops as ops
from vllm.compilation.decorators import support_torch_compile
from vllm.config import CacheConfig, VllmConfig, get_current_vllm_config
from vllm.distributed import get_tensor_model_parallel_world_size
from vllm.logger import init_logger
from vllm.model_executor.layers.attention import Attention
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.linear import (
    QKVParallelLinear,
    ReplicatedLinear,
    RowParallelLinear,
)
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.quantization.base_config import QuantizationConfig
from vllm.model_executor.layers.rotary_embedding import get_rope
from vllm.model_executor.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
)
from vllm.model_executor.model_loader.weight_utils import (
    default_weight_loader,
    maybe_remap_kv_scale_name,
)
from vllm.multimodal.inputs import NestedTensors
from vllm.transformers_utils.config import set_default_rope_theta
from vllm.v1.attention.backend import AttentionType
from vllm.v1.kv_cache_interface import (
    FullAttentionSpec,
    KVCacheSpec,
    SlidingWindowSpec,
)

from .qwen2 import Qwen2MLP as Qwen3MLP
from .qwen3 import Qwen3ForCausalLM
from .utils import (
    AutoWeightsLoader,
    get_draft_quant_config,
    maybe_prefix,
    process_eagle_weight,
)

logger = init_logger(__name__)


_DFLASH_VALID_LAYER_TYPES = frozenset({"full_attention", "sliding_attention"})


def _get_dflash_layer_types(config: Qwen3Config) -> tuple[str, ...]:
    layer_types = getattr(config, "layer_types", None)
    if layer_types is None:
        return ("full_attention",) * config.num_hidden_layers
    if len(layer_types) != config.num_hidden_layers:
        raise ValueError(
            f"DFlash layer_types length {len(layer_types)} does not match "
            f"num_hidden_layers {config.num_hidden_layers}."
        )
    invalid = set(layer_types) - _DFLASH_VALID_LAYER_TYPES
    if invalid:
        raise ValueError(f"Invalid DFlash layer_type(s): {sorted(invalid)}.")
    if "sliding_attention" in layer_types and not getattr(
        config, "sliding_window", None
    ):
        raise ValueError(
            "DFlash sliding_attention layers require `sliding_window` in config."
        )
    return tuple(layer_types)


class DFlashAttention(Attention):
    """Attention with DFlash-specific KV allocation semantics.

    The compute path keeps the layer's configured sliding window. The KV cache
    spec is widened to full attention because DFlash writes every context KV
    before drafting and cannot evict old context blocks from draft layers.
    """

    def __init__(self, *args, **kwargs) -> None:
        # DFlash draft attention runs over text/query tokens with prewritten K/V.
        # Do not inherit a multimodal-prefix mask requirement from the target.
        kwargs.setdefault("use_mm_prefix", False)
        super().__init__(*args, **kwargs)

    def get_kv_cache_spec(self, vllm_config: VllmConfig) -> KVCacheSpec | None:
        spec = super().get_kv_cache_spec(vllm_config)
        if isinstance(spec, SlidingWindowSpec):
            return FullAttentionSpec(
                block_size=spec.block_size,
                num_kv_heads=spec.num_kv_heads,
                head_size=spec.head_size,
                head_size_v=getattr(spec, "head_size_v", spec.head_size),
                dtype=spec.dtype,
                kv_quant_mode=spec.kv_quant_mode,
                page_size_padded=spec.page_size_padded,
            )
        return spec


class _VanillaMarkov(nn.Module):
    """Inlined DSpark VanillaMarkov low-rank transition-bias head.

    Mirrors the training repo's markov_head.py VanillaMarkov param names EXACTLY
    (markov_w1 = nn.Embedding(vocab, rank); markov_w2 = nn.Linear(rank, vocab,
    bias=False)) so checkpoint tensors markov_head.markov_w1.weight /
    markov_head.markov_w2.weight line up. Inlined (NOT imported from the repo)
    because vLLM workers do not have the training repo root on sys.path.

    B(x_{k-1}, :) = W2(W1[x_{k-1}]); the corrected logit for draft position k is
    base_k + B(x_{k-1}, :), sampled left-to-right (semi-autoregressive).
    """

    def __init__(self, vocab_size: int, markov_rank: int) -> None:
        super().__init__()
        self.markov_w1 = nn.Embedding(vocab_size, markov_rank)
        self.markov_w2 = nn.Linear(markov_rank, vocab_size, bias=False)

    def compute_step_bias(
        self, prev_token_ids: torch.Tensor, hidden_states: torch.Tensor | None = None
    ) -> torch.Tensor:
        # (B,) prev token ids -> (B, V) additive logit bias
        # hidden_states ignored for vanilla (memoryless)
        del hidden_states
        return self.markov_w2(self.markov_w1(prev_token_ids.long()))

    def compute_step_vec(
        self, prev_token_ids: torch.Tensor, hidden_states: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Pre-W2 low-rank step vector v s.t. compute_step_bias == markov_w2(v).

        Returns (B, rank) so a top-N caller can gather only the N needed rows of
        markov_w2.weight ([V, rank]) instead of reading the full V x rank matrix.
        compute_step_bias(prev) == self.markov_w2(compute_step_vec(prev)) EXACTLY
        (same w1 lookup, same gate for the gated subclass); this method is the
        factored inner half, never a separate approximation.
        """
        del hidden_states
        return self.markov_w1(prev_token_ids.long())


class _GatedMarkovHead(_VanillaMarkov):
    """Inlined DSpark GatedMarkovHead (official DeepSpec).

    Adds a sigmoid gate conditioned on [hidden_state; prev_embedding] to
    modulate the markov bias. Uses backbone hidden state for adaptive gating.
    Param names match the training repo: markov_w1, markov_w2, gate_proj.
    """

    def __init__(self, vocab_size: int, markov_rank: int, hidden_size: int) -> None:
        super().__init__(vocab_size=vocab_size, markov_rank=markov_rank)
        self.gate_proj = nn.Linear(hidden_size + markov_rank, markov_rank)

    def compute_step_bias(
        self, prev_token_ids: torch.Tensor, hidden_states: torch.Tensor | None = None
    ) -> torch.Tensor:
        prev_emb = self.markov_w1(prev_token_ids.long())
        if hidden_states is None:
            return self.markov_w2(prev_emb)
        gate = torch.sigmoid(
            self.gate_proj(torch.cat([hidden_states, prev_emb], dim=-1))
        ).to(dtype=prev_emb.dtype)
        return self.markov_w2(gate * prev_emb)

    def compute_step_vec(
        self, prev_token_ids: torch.Tensor, hidden_states: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Pre-W2 low-rank step vector for the gated head (see base docstring).

        compute_step_bias(prev, h) == self.markov_w2(compute_step_vec(prev, h))
        EXACTLY: the gate (or the hidden_states-is-None passthrough) is applied
        here, identically to compute_step_bias, so the only thing the top-N
        caller changes is which rows of markov_w2.weight get multiplied in.
        """
        prev_emb = self.markov_w1(prev_token_ids.long())
        if hidden_states is None:
            return prev_emb
        gate = torch.sigmoid(
            self.gate_proj(torch.cat([hidden_states, prev_emb], dim=-1))
        ).to(dtype=prev_emb.dtype)
        return gate * prev_emb


def _markov_topn_from_env() -> int:
    """Read DSPARK_MARKOV_TOPN (opt-in). 0 (default) / <=0 / unparsable -> off.

    When >0 the markov transition-bias is computed only for the top-N base-logit
    candidates per position, replacing the full V x rank markov_w2 GEMV
    (~127 MB HBM read for V=248320) with an N x rank gather (~1 MB for N=2048).
    Default 0 keeps the byte-identical full path; this is a pure serve knob.
    """
    try:
        return int(os.environ.get("DSPARK_MARKOV_TOPN", "0") or "0")
    except (TypeError, ValueError):
        return 0


def _markov_topn_sparse_bias(
    step_vec: torch.Tensor,  # [B, rank] pre-W2 vector (compute_step_vec output)
    w2_weight: torch.Tensor,  # [V, rank] markov_w2.weight
    topn_idx: torch.Tensor,  # [B, N] long, per-row top-N base-logit indices
) -> torch.Tensor:
    """Sparse markov bias at only the top-N indices: bias[b, n] = <W2[idx], v>.

    Gathers N rows of markov_w2.weight per row and dots them with the low-rank
    step vector, so bias[b, n] == compute_step_bias(prev)[b, topn_idx[b, n]] up
    to fp accumulation order (same sum over `rank`). Returns [B, N] in the head
    dtype (caller casts to the base-logit dtype), mirroring compute_step_bias's
    own dtype before its .to(base.dtype) cast. HBM read is B*N*rank vs B*V*rank.
    """
    # w2_weight[topn_idx] -> [B, N, rank]; bmm with v[B, rank, 1] -> [B, N, 1].
    w2_rows = w2_weight[topn_idx]
    return torch.bmm(w2_rows, step_vec.unsqueeze(-1)).squeeze(-1)


def _markov_semiar_sample_block(
    base_logits: torch.Tensor,  # [B, num_spec, V] base draft logits (model dtype)
    first_prev_token_ids: torch.Tensor,  # [B] verified token before pos 0
    compute_step_bias,  # callable: prev_ids[B] long -> bias[B, V]
    temperature: torch.Tensor,  # [B, num_spec] per-request-per-pos sampling temp
    all_random: bool,  # sampling_metadata.all_random
    sample_fn,  # callable: probs[B, V] float32 -> tokens[B] long (random draw)
    sampling_eps: float = 1e-5,
    topn: int = 0,  # DSPARK_MARKOV_TOPN; >0 & <V -> sparse top-N bias path
    compute_step_vec=None,  # callable: prev_ids[B] long -> step_vec[B, rank]
    w2_weight: torch.Tensor | None = None,  # markov_w2.weight [V, rank]
) -> tuple[torch.Tensor, torch.Tensor]:
    """Pure-torch LEFT-TO-RIGHT semi-AR markov-biased SAMPLING block.

    Self-contained (only torch + injected callables) so it is unit-testable on
    CPU without vLLM/CUDA. Mirrors compute_probs_and_sample_next_token's math
    EXACTLY -- per-row temperature division, fp32 softmax, and the mixed-batch
    greedy-row torch.where override -- but folds in the markov bias and feeds the
    realized predecessor back at each step.

    At position k the realized previous token biases the logit; the returned
    probs[:, k] is the EXACT softmax distribution position k sampled from
    (q_k = softmax((base_k + bias(prev_{k-1})) / T_row)). This identity is what
    makes standard rejection sampling lossless: accepted dist == target dist iff
    reported q == the draft's actual sampling dist.

    Returns (tokens [B, num_spec] long, probs [B, num_spec, V] float32).
    """
    B, S, V = base_logits.shape
    if S == 0:
        return (
            base_logits.new_empty((B, 0), dtype=torch.long),
            base_logits.new_empty((B, 0, V), dtype=torch.float32),
        )
    out = base_logits.new_empty((B, S), dtype=torch.long)
    probs_out = base_logits.new_empty((B, S, V), dtype=torch.float32)
    prev = first_prev_token_ids.long()
    # Opt-in top-N sparse markov bias. topn>=V would cover the whole vocab, so it
    # falls back to the exact full path (identical result, no reason to gather V
    # rows); topn<=0 is off. Only 0<topn<V takes the gather/scatter branch.
    # PERF GUARD: the gather reads B*topn rows of markov_w2.weight (no cross-row
    # reuse) vs the full path streaming the [V, rank] weight ONCE for the whole
    # batch. Break-even is B*topn == V; above it the "sparse" path reads MORE HBM
    # than full (and gather is less bandwidth-efficient than a streamed GEMM), so
    # at high concurrency we must fall back to full. Only take top-N when
    # B*topn < V.
    use_topn = (
        topn > 0
        and topn < V
        and B * topn < V
        and compute_step_vec is not None
        and w2_weight is not None
    )
    for k in range(S):
        if use_topn:
            base_k = base_logits[:, k, :]
            # top-N base-logit candidates; only these receive a bias, every other
            # token keeps its exact base logit (full-V softmax preserved below).
            _, idx = torch.topk(base_k, topn, dim=-1)  # [B, N]
            step_vec = compute_step_vec(prev)
            sparse = _markov_topn_sparse_bias(step_vec, w2_weight, idx)
            logits_k = base_k.clone()
            logits_k.scatter_add_(-1, idx, sparse.to(logits_k.dtype))
        else:
            logits_k = base_logits[:, k, :] + compute_step_bias(prev).to(
                base_logits.dtype
            )
        t_k = temperature[:, k]
        is_greedy = None
        if not all_random:
            # Mixed batch: greedy rows divide by 1.0 (then argmax-override below),
            # exactly as compute_probs_and_sample_next_token does.
            is_greedy = t_k < sampling_eps
            t_k = torch.where(is_greedy, torch.ones_like(t_k), t_k)
        probs_k = (logits_k / t_k.unsqueeze(-1)).softmax(dim=-1, dtype=torch.float32)
        # Record the reported q BEFORE drawing, so an in-place sampler cannot
        # corrupt it (indexed assignment copies into probs_out's own storage).
        probs_out[:, k, :] = probs_k
        tok_k = sample_fn(probs_k).to(torch.long)
        if is_greedy is not None:
            greedy_tok = probs_k.argmax(dim=-1)
            tok_k = torch.where(is_greedy, greedy_tok, tok_k)
        out[:, k] = tok_k
        prev = out[:, k]
    return out, probs_out

class DFlashQwen3Attention(nn.Module):
    """Attention for DFlash speculative decoding.

    Context KVs are pre-inserted into the KV cache before the forward pass.
    This layer handles only query tokens via standard attention.
    Adapted from Qwen3Attention."""

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        rope_parameters: dict,
        max_position: int = 4096 * 32,
        head_dim: int | None = None,
        rms_norm_eps: float = 1e-06,
        attention_bias: bool = False,
        cache_config: CacheConfig | None = None,
        quant_config: QuantizationConfig | None = None,
        sliding_window: int | None = None,
        prefix: str = "",
        attn_type: str = AttentionType.DECODER,
    ) -> None:
        super().__init__()
        self.layer_name = prefix
        self.hidden_size = hidden_size
        tp_size = get_tensor_model_parallel_world_size()
        self.total_num_heads = num_heads
        assert self.total_num_heads % tp_size == 0
        self.num_heads = self.total_num_heads // tp_size
        self.total_num_kv_heads = num_kv_heads
        if self.total_num_kv_heads >= tp_size:
            assert self.total_num_kv_heads % tp_size == 0
        else:
            assert tp_size % self.total_num_kv_heads == 0
        self.num_kv_heads = max(1, self.total_num_kv_heads // tp_size)
        self.head_dim = head_dim or hidden_size // self.total_num_heads
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        self.scaling = self.head_dim**-0.5

        self.qkv_proj = QKVParallelLinear(
            hidden_size,
            self.head_dim,
            self.total_num_heads,
            self.total_num_kv_heads,
            bias=attention_bias,
            quant_config=quant_config,
            prefix=f"{prefix}.qkv_proj",
        )
        self.o_proj = RowParallelLinear(
            self.total_num_heads * self.head_dim,
            hidden_size,
            bias=attention_bias,  # DFlash has o_proj bias when using attention bias
            quant_config=quant_config,
            prefix=f"{prefix}.o_proj",
        )

        self.rotary_emb = get_rope(
            self.head_dim,
            max_position=max_position,
            rope_parameters=rope_parameters,
        )
        self.attn = DFlashAttention(
            self.num_heads,
            self.head_dim,
            self.scaling,
            num_kv_heads=self.num_kv_heads,
            cache_config=cache_config,
            quant_config=quant_config,
            per_layer_sliding_window=sliding_window,
            prefix=f"{prefix}.attn",
            attn_type=attn_type,
        )
        self.q_norm = RMSNorm(self.head_dim, eps=rms_norm_eps)
        self.k_norm = RMSNorm(self.head_dim, eps=rms_norm_eps)

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        """DFlash attention assumes that the KV cache is already populated
        with the context K/V from the target model's hidden states. This forward op
        computes attention for the query tokens only.
        See also: precompute_and_store_context_kv"""
        qkv = F.linear(hidden_states, self.qkv_proj.weight, self.qkv_proj.bias)
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)

        # Per-head RMSNorm
        q_shape, k_shape = q.shape, k.shape
        q = self.q_norm(
            q.view(*q_shape[:-1], q_shape[-1] // self.head_dim, self.head_dim)
        ).view(q_shape)
        k = self.k_norm(
            k.view(*k_shape[:-1], k_shape[-1] // self.head_dim, self.head_dim)
        ).view(k_shape)

        q, k = self.rotary_emb(positions, q, k)

        attn_output = self.attn(q, k, v)
        output, _ = self.o_proj(attn_output)
        return output


class DFlashQwen3DecoderLayer(nn.Module):
    def __init__(
        self,
        vllm_config: VllmConfig,
        *,
        config: Qwen3Config,
        cache_config: CacheConfig | None = None,
        quant_config: QuantizationConfig | None = None,
        layer_type: str = "full_attention",
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.hidden_size = config.hidden_size
        self.layer_type = layer_type
        set_default_rope_theta(config, default_theta=1000000)
        attn_type = AttentionType.DECODER
        sliding_window = (
            config.sliding_window if layer_type == "sliding_attention" else None
        )

        self.self_attn = DFlashQwen3Attention(
            hidden_size=self.hidden_size,
            num_heads=config.num_attention_heads,
            max_position=config.max_position_embeddings,
            num_kv_heads=config.num_key_value_heads,
            rms_norm_eps=config.rms_norm_eps,
            attention_bias=getattr(config, "attention_bias", False),
            head_dim=getattr(config, "head_dim", None),
            cache_config=cache_config,
            quant_config=quant_config,
            sliding_window=sliding_window,
            rope_parameters=config.rope_parameters,
            prefix=f"{prefix}.self_attn",
            attn_type=attn_type,
        )
        self.mlp = Qwen3MLP(
            hidden_size=self.hidden_size,
            intermediate_size=config.intermediate_size,
            hidden_act=config.hidden_act,
            quant_config=quant_config,
            prefix=f"{prefix}.mlp",
        )
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        residual: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if residual is not None:
            hidden_states, residual = self.input_layernorm(hidden_states, residual)
        else:
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)

        hidden_states = self.self_attn(
            positions=positions,
            hidden_states=hidden_states,
        )

        hidden_states, residual = self.post_attention_layernorm(hidden_states, residual)
        hidden_states = self.mlp(hidden_states)
        return hidden_states, residual


@support_torch_compile
class DFlashQwen3Model(nn.Module):
    def __init__(
        self,
        *,
        vllm_config: VllmConfig,
        start_layer_id: int = 0,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.config = vllm_config.speculative_config.draft_model_config.hf_config
        self.vocab_size = self.config.vocab_size
        self.quant_config = get_draft_quant_config(vllm_config)

        drafter_config = getattr(self.config, "eagle_config", {})
        drafter_config.update(getattr(self.config, "dflash_config", {}))

        if drafter_config is not None and "use_aux_hidden_state" in drafter_config:
            self.use_aux_hidden_state = drafter_config["use_aux_hidden_state"]
        else:
            self.use_aux_hidden_state = True

        current_vllm_config = get_current_vllm_config()

        self.embed_tokens = VocabParallelEmbedding(
            self.config.vocab_size,
            self.config.hidden_size,
            prefix=maybe_prefix(prefix, "embed_tokens"),
        )
        target_config = vllm_config.model_config.hf_text_config
        self.embed_normalizer: float | None = None
        if str(getattr(target_config, "model_type", "")).startswith("gemma4"):
            # Gemma4 scales token embeddings by sqrt(hidden_size). DFlash
            # shares the target embeddings, so the draft path must match.
            self.embed_normalizer = target_config.hidden_size**0.5

        self.layer_types = _get_dflash_layer_types(self.config)
        self.layers = nn.ModuleList(
            [
                DFlashQwen3DecoderLayer(
                    current_vllm_config,
                    config=self.config,
                    cache_config=current_vllm_config.cache_config,
                    quant_config=self.quant_config,
                    layer_type=self.layer_types[layer_idx],
                    prefix=maybe_prefix(prefix, f"layers.{layer_idx + start_layer_id}"),
                )
                for layer_idx in range(self.config.num_hidden_layers)
            ]
        )
        self.sliding_attention_layer_names = {
            layer.self_attn.attn.layer_name
            for layer in self.layers
            if layer.layer_type == "sliding_attention"
        }
        if self.use_aux_hidden_state:
            num_features_to_use = self.config.num_hidden_layers
            if "target_layer_ids" in drafter_config:
                num_features_to_use = len(drafter_config["target_layer_ids"])
            elif "layer_ids" in drafter_config:
                num_features_to_use = len(drafter_config["layer_ids"])
            if hasattr(self.config, "target_hidden_size"):
                fc_input_size = self.config.target_hidden_size * num_features_to_use
            else:
                fc_input_size = self.config.hidden_size * num_features_to_use
            self.fc = ReplicatedLinear(
                input_size=fc_input_size,
                output_size=self.config.hidden_size,
                bias=False,
                params_dtype=vllm_config.model_config.dtype,
                quant_config=self.quant_config,
                prefix=maybe_prefix(prefix, "fc"),
                return_bias=False,
            )
        self.hidden_norm = RMSNorm(
            self.config.hidden_size,
            eps=self.config.rms_norm_eps,
        )
        self.norm = RMSNorm(
            self.config.hidden_size,
            eps=self.config.rms_norm_eps,
        )

        markov_rank = int(getattr(self.config, "markov_rank", 0) or 0)
        markov_head_type = str(
            getattr(self.config, "markov_head_type", "vanilla") or "vanilla"
        ).lower()
        if markov_rank > 0:
            if markov_head_type in ("gated", "gated_markov", "gatedmarkov"):
                self.markov_head = _GatedMarkovHead(
                    self.config.vocab_size, markov_rank, self.config.hidden_size
                )
            else:
                self.markov_head = _VanillaMarkov(self.config.vocab_size, markov_rank)
        else:
            self.markov_head = None

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        embeds = self.embed_tokens(input_ids)
        return embeds * self.embed_normalizer if self.embed_normalizer else embeds

    def _build_fused_kv_buffers(self) -> None:
        """Build fused weight buffers for precompute_and_store_context_kv.

        Must be called after weights are loaded. Stacks the KV-projection
        weights, K-norm weights, and RoPE parameters from every attention
        layer so that precompute_and_store_context_kv can run one fused
        GEMM for all layers at once. Also aliases the weight of the hidden_norm.
        """
        layers_attn = [layer.self_attn for layer in self.layers]
        attn0 = layers_attn[0]
        has_bias = attn0.qkv_proj.bias is not None

        self._hidden_norm_weight = self.hidden_norm.weight.data

        # KV projection weights: [num_layers * 2 * kv_size, hidden_size]
        kv_weights = [a.qkv_proj.weight[a.q_size :] for a in layers_attn]
        self._fused_kv_weight = torch.cat(kv_weights, dim=0)
        if has_bias:
            kv_biases = [a.qkv_proj.bias[a.q_size :] for a in layers_attn]
            self._fused_kv_bias: torch.Tensor | None = torch.cat(kv_biases, dim=0)
        else:
            self._fused_kv_bias = None

        # K-norm weights: list of [head_dim] tensors, one per layer.
        self._k_norm_weights = [a.k_norm.weight.data for a in layers_attn]

        # RoPE parameters
        self._rope_head_size = attn0.rotary_emb.head_size
        self._rope_cos_sin_cache = attn0.rotary_emb.cos_sin_cache
        self._rope_is_neox = attn0.rotary_emb.is_neox_style
        # Validation that RoPE params are the same across all layers
        for attn in layers_attn[1:]:
            assert (
                attn.rotary_emb.head_size == self._rope_head_size
                and attn.rotary_emb.is_neox_style == self._rope_is_neox
            ), "All layers must have the same RoPE parameters for DFlash precomputation"

        # Layer metadata
        self._num_attn_layers = len(layers_attn)
        self._kv_size = attn0.kv_size
        self._head_dim = attn0.head_dim
        self._num_kv_heads = attn0.num_kv_heads
        self._rms_norm_eps = attn0.q_norm.variance_epsilon
        # Validation that all layers have the same attention config
        for attn in layers_attn[1:]:
            assert (
                attn.kv_size == self._kv_size
                and attn.head_dim == self._head_dim
                and attn.num_kv_heads == self._num_kv_heads
                and attn.q_norm.variance_epsilon == self._rms_norm_eps
            ), "All layers must have the same attn config for DFlash precomputation"

        # References to inner Attention layers for direct cache writes
        self._attn_layers = [layer.self_attn.attn for layer in self.layers]

    def precompute_and_store_context_kv(
        self,
        context_states: torch.Tensor,
        context_positions: torch.Tensor,
        context_slot_mapping: torch.Tensor | Mapping[str, torch.Tensor] | None = None,
    ) -> None:
        """Precompute K/V for context states write them into each layer's KV cache.

        Input context states are projected to K/V, normed, and have RoPE applied.
        Since the context shape is different than the query shape, we can't rely on the
        regular forward pass to apply torch.compile and CUDA graphs to this section.
        As such, this function is optimized to minimize the number of torch ops present:
        we use fused vLLM kernels for RMSNorm and RoPE, fuse the GEMM into one
        large projection, and avoid cloning buffers (with .contiguous()) where possible.

        When context_slot_mapping is None (e.g. during dummy_run) only
        the computation runs, and no K/V is written to cache.
        """
        if not hasattr(self, "_num_attn_layers"):
            logger.warning_once(
                "DFlash buffer initialization was skipped. If dummy weights are not "
                "in use, this may indicate an error in weight loading."
            )
            self._build_fused_kv_buffers()

        num_ctx = context_states.shape[0]
        L = self._num_attn_layers
        kv = self._kv_size
        hd = self._head_dim
        nkv = self._num_kv_heads

        # --- Fused KV projection (one GEMM for all layers) ---
        normed_context_states = torch.empty_like(context_states)
        ops.rms_norm(
            normed_context_states,
            context_states,
            self._hidden_norm_weight,
            self._rms_norm_eps,
        )
        all_kv_flat = F.linear(
            normed_context_states, self._fused_kv_weight, self._fused_kv_bias
        )
        # Single contiguous copy that separates K/V and transposes to
        # layer-major layout.  Result: [2, L, num_ctx, nkv, hd] contiguous.
        # Indexing dim-0 gives contiguous [L, num_ctx, nkv, hd] for K and V.
        all_kv = (
            all_kv_flat.view(num_ctx, L, 2, nkv, hd).permute(2, 1, 0, 3, 4).contiguous()
        )
        all_k = all_kv[0]  # [L, num_ctx, nkv, hd], contiguous
        all_v = all_kv[1]  # [L, num_ctx, nkv, hd], contiguous

        # --- Per-layer RMSNorm K (3D: [num_ctx, nkv, hd] per layer) ---
        all_k_normed = torch.empty_like(all_k)
        for i in range(L):
            ops.rms_norm(
                all_k_normed[i],
                all_k[i],
                self._k_norm_weights[i],
                self._rms_norm_eps,
            )

        # --- Fused RoPE across all layers ---
        # View as [L * num_ctx, kv] so RoPE sees one big batch (no copy).
        # In-place RoPE: pass K as the "query" arg with key=None.
        all_k_flat = all_k_normed.view(L * num_ctx, kv)
        positions_repeated = context_positions.repeat(L)
        cos_sin_cache = self._rope_cos_sin_cache
        if cos_sin_cache.dtype != all_k_flat.dtype:
            cos_sin_cache = cos_sin_cache.to(dtype=all_k_flat.dtype)
        ops.rotary_embedding(
            positions_repeated,
            all_k_flat,
            None,
            self._rope_head_size,
            cos_sin_cache,
            self._rope_is_neox,
        )

        if context_slot_mapping is None:
            return

        # --- Per-layer cache insert ---
        all_k_final = all_k_flat.view(L, num_ctx, nkv, hd)
        for i in range(L):
            attn = self._attn_layers[i]
            layer_slot_mapping = (
                context_slot_mapping[attn.layer_name]
                if isinstance(context_slot_mapping, Mapping)
                else context_slot_mapping
            )
            kv_cache = attn.kv_cache
            attn.impl.do_kv_cache_update(
                attn,
                all_k_final[i],
                all_v[i],
                kv_cache,
                layer_slot_mapping,
            )

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        input_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if input_embeds is None:
            input_embeds = self.embed_input_ids(input_ids)

        hidden_states = input_embeds

        residual = None
        for layer in self.layers:
            hidden_states, residual = layer(
                positions=positions,
                hidden_states=hidden_states,
                residual=residual,
            )
        hidden_states, _ = self.norm(hidden_states, residual)
        return hidden_states

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        stacked_params_mapping = [
            (".qkv_proj", ".q_proj", "q"),
            (".qkv_proj", ".k_proj", "k"),
            (".qkv_proj", ".v_proj", "v"),
            (".gate_up_proj", ".gate_proj", 0),
            (".gate_up_proj", ".up_proj", 1),
        ]
        params_dict = dict(self.named_parameters())
        loaded_params: set[str] = set()
        for name, loaded_weight in weights:
            if "midlayer." in name:
                name = name.replace("midlayer.", "layers.0.")
            if "scale" in name:
                name = maybe_remap_kv_scale_name(name, params_dict)
                if name is None:
                    continue
            for param_name, weight_name, shard_id in stacked_params_mapping:
                if weight_name not in name:
                    continue
                name = name.replace(weight_name, param_name)
                param = params_dict[name]
                weight_loader = param.weight_loader
                weight_loader(param, loaded_weight, shard_id)
                break
            else:
                param = params_dict[name]
                weight_loader = getattr(param, "weight_loader", default_weight_loader)
                weight_loader(param, loaded_weight)
            loaded_params.add(name)
        return loaded_params


class DFlashQwen3ForCausalLM(Qwen3ForCausalLM):
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        nn.Module.__init__(self)
        self.config = vllm_config.speculative_config.draft_model_config.hf_config
        if getattr(self.config, "draft_vocab_size", None) is None:
            self.config.draft_vocab_size = getattr(self.config, "vocab_size", None)
        target_layer_num = vllm_config.model_config.get_num_layers(
            vllm_config.parallel_config
        )
        self.model = DFlashQwen3Model(
            vllm_config=vllm_config,
            prefix=maybe_prefix(prefix, "model"),
            start_layer_id=target_layer_num,
        )

        logit_scale = getattr(self.config, "logit_scale", 1.0)
        self.lm_head = ParallelLMHead(
            self.config.draft_vocab_size,
            self.config.hidden_size,
            prefix=maybe_prefix(prefix, "lm_head"),
        )
        self.logits_processor = LogitsProcessor(
            self.config.draft_vocab_size,
            scale=logit_scale,
            soft_cap=getattr(self.config, "final_logit_softcapping", None),
        )
        target_vocab_size = vllm_config.model_config.get_vocab_size()
        if self.config.draft_vocab_size != target_vocab_size:
            self.draft_id_to_target_id = nn.Parameter(
                torch.zeros(self.config.draft_vocab_size, dtype=torch.long),
                requires_grad=False,
            )
        else:
            self.draft_id_to_target_id = None

    def embed_input_ids(
        self,
        input_ids: torch.Tensor,
        multimodal_embeddings: NestedTensors | None = None,
        is_multimodal: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return self.model.embed_input_ids(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        inputs_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return self.model(input_ids, positions, inputs_embeds)

    def compute_logits(
        self,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor | None:
        logits = self.logits_processor(self.lm_head, hidden_states)
        if self.draft_id_to_target_id is None:
            return logits

        base = torch.arange(self.config.draft_vocab_size, device=logits.device)
        targets = base + self.draft_id_to_target_id
        logits_new = logits.new_full(
            (logits.shape[0], self.config.vocab_size),
            float("-inf"),
        )
        logits_new[:, targets] = logits
        return logits_new

    @torch.no_grad()
    def sample_draft_block_semiar(
        self,
        sample_hidden_states: torch.Tensor,
        first_prev_token_ids: torch.Tensor,
        num_spec: int,
    ) -> torch.Tensor:
        """LEFT-TO-RIGHT (semi-autoregressive) markov-biased greedy draft block.

        Produces the parallel-drafted block one position at a time: position k's
        argmax is taken over base_k + markov_bias(prev_token), where prev_token
        is the token argmax'd at position k-1 (first_prev = the verified bonus
        token before draft pos 0). LOSSLESS under greedy because the target
        verify re-checks every drafted token against the target argmax,
        independent of how the draft was produced.

        Returns [B, num_spec] long tensor, matching the .view(-1, num_spec)
        contract of the parallel-drafting early-exit path.
        """
        mk = self.model.markov_head
        assert mk is not None, (
            "sample_draft_block_semiar called without a markov_head"
        )
        # The markov head is trained in the (draft == target) vocab space. The
        # d2t remap path is intentionally NOT supported here: with d2t != None
        # the base logits live in draft-vocab space while the markov bias lives
        # in target-vocab space (shape mismatch), and the L->R feedback loop
        # would index markov_w1 with draft-vocab ids. Fail loudly rather than
        # silently dropping the bias. (Follow-up could remap out at the end via
        # out = self.draft_id_to_target_id[out] AND project the bias/feedback
        # into draft space, but AEON has draft_vocab == vocab so d2t is None.)
        assert self.draft_id_to_target_id is None, (
            "VanillaMarkov semi-AR draft requires draft_vocab_size == target "
            "vocab_size (draft_id_to_target_id is None); the d2t remap path is "
            "not implemented."
        )
        # Base draft logits ONCE, draft-vocab space, mirroring compute_logits
        # but WITHOUT the d2t scatter (guarded None above). Stay on-device:
        # no .item()/D2H inside the loop.
        base = self.logits_processor(self.lm_head, sample_hidden_states)
        B = sample_hidden_states.shape[0] // num_spec
        V = base.shape[-1]
        assert V == mk.markov_w2.weight.shape[0], (
            f"draft logits vocab {V} != markov bias vocab "
            f"{mk.markov_w2.weight.shape[0]}"
        )
        assert first_prev_token_ids.shape[0] == B, (
            f"first_prev_token_ids batch {first_prev_token_ids.shape[0]} != "
            f"derived batch {B} (rows={sample_hidden_states.shape[0]}, "
            f"num_spec={num_spec})"
        )
        base = base.view(B, num_spec, V)
        # Fused semi-AR: use mk.compute_step_bias but with pre-allocated buffers
        # and in-place add+argmax to reduce kernel launches per iteration.
        # Original: 8 × (embed+linear+add+argmax) = 32 launches
        # Optimized: 8 × (compute_step_bias+add+argmax) = 24, no intermediate allocs
        # GatedMarkovHead: passes hidden_states per position for gate conditioning.
        out = base.new_empty((B, num_spec), dtype=torch.long)
        prev = first_prev_token_ids.long()
        # Pre-allocate bias buffer to avoid per-step allocation
        bias_buf = base.new_empty((B, V))
        # Reshape sample_hidden_states to [B, num_spec, H] for gated head access
        hs_per_pos = sample_hidden_states.view(B, num_spec, -1)
        # Opt-in top-N sparse markov bias (DSPARK_MARKOV_TOPN). Every non-top-N
        # token keeps its base logit (its bias is defined to be 0), so the top-N
        # argmax equals the TRUE full-base+bias argmax ONLY WHEN the full-bias
        # winner lies inside base-top-N; if a token outside base-top-N would have
        # won under the full bias, top-N misses it -> an accept-affecting
        # TRUNCATION (an approximation of the full-bias argmax), NOT an identity.
        # This is safe: losslessness is guaranteed by verify (the target re-checks
        # every drafted token); top-N can at most shift which token is proposed
        # (accept rate), never correctness. topn>=V falls back to the full path.
        # PERF GUARD (same as the block path): only worthwhile when B*topn < V,
        # else the per-row gather reads more HBM than the streamed full weight.
        topn = _markov_topn_from_env()
        use_topn = 0 < topn < V and B * topn < V
        w2w = mk.markov_w2.weight if use_topn else None
        for k in range(num_spec):
            hs_k = hs_per_pos[:, k, :] if hasattr(mk, 'gate_proj') else None
            if use_topn:
                base_k = base[:, k, :]
                _, idx = torch.topk(base_k, topn, dim=-1)  # [B, N]
                step_vec = mk.compute_step_vec(prev, hidden_states=hs_k)
                sparse = _markov_topn_sparse_bias(step_vec, w2w, idx).to(base.dtype)
                bias_buf.copy_(base_k)
                bias_buf.scatter_add_(-1, idx, sparse)
            else:
                bias = mk.compute_step_bias(prev, hidden_states=hs_k).to(base.dtype)
                # Fused add + argmax via in-place add then argmax
                torch.add(base[:, k, :], bias, out=bias_buf)
            tok = bias_buf.argmax(dim=-1)
            out[:, k] = tok
            prev = tok
        return out  # [B, num_spec]

    @torch.no_grad()
    def sample_draft_block_semiar_sample(
        self,
        sample_hidden_states: torch.Tensor,
        first_prev_token_ids: torch.Tensor,
        num_spec: int,
        temperature: torch.Tensor,
        all_random: bool,
        use_fp64_gumbel: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """LEFT-TO-RIGHT (semi-AR) markov-biased SAMPLING draft block (temp>0).

        LOSSLESS counterpart of sample_draft_block_semiar (the greedy variant):
        each position k is SAMPLED -- not argmax'd -- from
            q_k = softmax((base_k + markov_bias(prev_{k-1})) / T_row)
        and that EXACT q_k is returned as the corrected proposal probs. Standard
        rejection sampling is lossless iff the reported q equals the distribution
        the draft actually sampled from; here they are the same tensor, computed
        from the same logits, same per-row temperature, and the realized
        left-to-right predecessor -- so the contract holds by construction.

        Math mirrors compute_probs_and_sample_next_token EXACTLY (per-row temp
        division, fp32 softmax, exponential-noise / Gumbel-max draw, and the
        greedy-row argmax override for mixed batches), applied position-by-
        position with the markov bias folded in and prev fed back.

        temperature is sampling_metadata.temperature, length B*num_spec, laid out
        request-major / position-minor (same layout as sample_hidden_states and
        the parallel path's draft_probs.view(-1, num_spec, V)).

        Returns (sampled_tokens [B, num_spec] long, corrected_probs
        [B, num_spec, V] float32). corrected_probs matches the parallel path's
        _last_draft_probs layout (request-major, position, vocab; float32;
        contiguous).
        """
        from vllm.v1.sample.ops.topk_topp_sampler import (
            empty_exponential_noise_like,
            sample_with_exponential_noise,
        )
        from vllm.v1.sample.sampler import _SAMPLING_EPS

        mk = self.model.markov_head
        assert mk is not None, (
            "sample_draft_block_semiar_sample called without a markov_head"
        )
        # Same d2t guard as the greedy path: the markov head lives in the
        # draft==target vocab space; the remap path is not implemented.
        assert self.draft_id_to_target_id is None, (
            "VanillaMarkov semi-AR SAMPLING draft requires draft_vocab_size == "
            "target vocab_size (draft_id_to_target_id is None); the d2t remap "
            "path is not implemented."
        )
        base = self.logits_processor(self.lm_head, sample_hidden_states)
        B = sample_hidden_states.shape[0] // num_spec
        V = base.shape[-1]
        assert V == mk.markov_w2.weight.shape[0], (
            f"draft logits vocab {V} != markov bias vocab "
            f"{mk.markov_w2.weight.shape[0]}"
        )
        assert first_prev_token_ids.shape[0] == B, (
            f"first_prev_token_ids batch {first_prev_token_ids.shape[0]} != "
            f"derived batch {B} (rows={sample_hidden_states.shape[0]}, "
            f"num_spec={num_spec})"
        )
        assert temperature is not None, (
            "sample_draft_block_semiar_sample requires a temperature tensor "
            "(only reached when sampling_metadata is not all_greedy)."
        )
        base = base.view(B, num_spec, V)
        # temperature may be per-request [B] (vLLM sampling_metadata.temperature),
        # per-position [B*num_spec], or a scalar -> broadcast to [B, num_spec].
        _t = temperature.reshape(-1)
        if _t.numel() == B * num_spec:
            temp = _t.view(B, num_spec)
        elif _t.numel() == B:
            temp = _t.view(B, 1).expand(B, num_spec)
        else:
            temp = _t.reshape(1, 1).expand(B, num_spec)

        def _gumbel_sample(probs: torch.Tensor) -> torch.Tensor:
            # Mirror compute_probs_and_sample_next_token's draw exactly:
            # exponential (Gumbel-max) noise, sampled from probs.clone() so the
            # returned probs (used as rejection-sampling q) stay intact.
            noise = empty_exponential_noise_like(probs, use_fp64_gumbel)
            noise.exponential_()
            return sample_with_exponential_noise(probs.clone(), noise)

        # Opt-in top-N sparse markov bias (DSPARK_MARKOV_TOPN). The sampling path
        # keeps the full-V softmax; only the bias term is sparsified (top-N base
        # candidates get the bias, all other tokens keep their exact base logit).
        # The resulting q_k is a TRUNCATION of the full-bias distribution (tokens
        # whose bias would have mattered but sit outside base-top-N are omitted) --
        # an approximation that shifts accept, NOT correctness. Losslessness still
        # holds because rejection sampling only needs the reported q_k to equal the
        # distribution actually sampled from, and here they are the SAME tensor.
        # NOTE: like the full path here, the gated head is fed prev only
        # (hidden=None) -> compute_step_vec matches compute_step_bias(prev)
        # bit-for-bit at the inner (pre-W2) vector.
        # PERF: the block applies the same B*topn<V guard internally.
        topn = _markov_topn_from_env()
        return _markov_semiar_sample_block(
            base,
            first_prev_token_ids,
            mk.compute_step_bias,
            temp,
            all_random,
            _gumbel_sample,
            sampling_eps=_SAMPLING_EPS,
            topn=topn,
            compute_step_vec=mk.compute_step_vec,
            w2_weight=mk.markov_w2.weight,
        )

    def precompute_and_store_context_kv(
        self,
        context_states: torch.Tensor,
        context_positions: torch.Tensor,
        context_slot_mapping: torch.Tensor | Mapping[str, torch.Tensor] | None = None,
    ) -> None:
        """Precompute projected + RoPE'd K/V and write to cache."""
        self.model.precompute_and_store_context_kv(
            context_states, context_positions, context_slot_mapping
        )

    @property
    def sliding_attention_layer_names(self) -> set[str]:
        return self.model.sliding_attention_layer_names

    def combine_hidden_states(
        self,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        if not self.model.use_aux_hidden_state:
            return hidden_states
        needs_squeeze = hidden_states.dim() == 1
        if needs_squeeze:
            hidden_states = hidden_states.unsqueeze(0)
        result = self.model.fc(hidden_states)
        if needs_squeeze:
            result = result.squeeze(0)
        return result

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]):
        model_weights = {}
        includes_draft_id_mapping = False
        includes_embed_tokens = False
        for name, loaded_weight in weights:
            assert "mask_hidden" not in name, (
                "DFlash should use mask_token_id to embed the padding hidden state"
            )
            if "t2d" in name:
                continue
            if "d2t" in name:
                name = name.replace("d2t", "draft_id_to_target_id")
                includes_draft_id_mapping = True
            elif "lm_head" not in name:
                name = "model." + name
            if "embed_tokens" in name:
                includes_embed_tokens = True
            model_weights[name] = loaded_weight
            process_eagle_weight(self, name)

        skip_substrs = []
        if not includes_draft_id_mapping:
            skip_substrs.append("draft_id_to_target_id")
        if not includes_embed_tokens:
            skip_substrs.append("embed_tokens")
        if not self.model.use_aux_hidden_state:
            skip_substrs.append("fc.")
        loader = AutoWeightsLoader(
            self,
            skip_prefixes=None,
            skip_substrs=skip_substrs,
        )
        loader.load_weights(model_weights.items())
        self.model._build_fused_kv_buffers()
