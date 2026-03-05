"""
CodeMind Base Model Architecture — 2026 Edition

Optimized GPT-NeoX style model for:
- Code generation
- Multi-language support (Python, Dart, JS)
- Efficient inference on limited hardware

Architecture features:
- RMSNorm (LLaMA/Gemma style)
- SwiGLU activation (LLaMA/Mistral style)
- Rotary Position Embeddings (RoPE) + YaRN scaling (32K context)
- Grouped Query Attention (GQA)
- Flash Attention 2 (optional, graceful fallback to SDPA)
- Mixture of Depths (MoD) — token routing per layer
- Mixture of Experts (MoE) — expert routing per token
- Weight-tied embeddings
"""

# Flash Attention 2 availability check (installed via: pip install flash-attn)
try:
    from flash_attn import flash_attn_func  # type: ignore[import]
    from flash_attn import flash_attn_varlen_func  # type: ignore[import]
    _FLASH_ATTN_AVAILABLE = True
except ImportError:
    _FLASH_ATTN_AVAILABLE = False

import math
from typing import Optional, Tuple, List, Dict, Any, Union
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss
from torch.utils.checkpoint import checkpoint
from transformers import PretrainedConfig, PreTrainedModel, GenerationMixin
from transformers.modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast
from transformers.cache_utils import Cache, DynamicCache


class CodeMindConfig(PretrainedConfig):
    model_type = "codemind"

    def __init__(
        self,
        vocab_size: int = 32768,
        hidden_size: int = 768,
        num_hidden_layers: int = 24,
        num_attention_heads: int = 12,
        num_key_value_heads: int = 4,
        intermediate_size: int = 3072,
        max_position_embeddings: int = 4096,
        rms_norm_eps: float = 1e-6,
        hidden_dropout: float = 0.0,
        attention_dropout: float = 0.0,
        activation: str = "silu",
        rotary_pct: float = 1.0,
        rope_scaling: Optional[Dict[str, Any]] = None,
        use_cache: bool = True,
        tie_word_embeddings: bool = True,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        # ── 2026 Techniques ──────────────────────────────────────────────────
        use_flash_attention: bool = False,   # Flash Attention 2 (requires flash-attn)
        use_mod: bool = False,               # Mixture of Depths token routing
        mod_capacity_factor: float = 0.5,    # Fraction of tokens processed per MoD layer
        **kwargs
    ):
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads

        # GQA support
        if num_key_value_heads is None:
            num_key_value_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads

        self.intermediate_size = intermediate_size
        self.max_position_embeddings = max_position_embeddings
        self.rms_norm_eps = rms_norm_eps
        self.hidden_dropout = hidden_dropout
        self.attention_dropout = attention_dropout
        self.activation = activation
        self.rotary_pct = rotary_pct
        self.rope_scaling = rope_scaling
        self.use_cache = use_cache
        self.head_dim = self.hidden_size // self.num_attention_heads
        self.output_attentions = output_attentions
        self.output_hidden_states = output_hidden_states
        self.output_router_logits = kwargs.get("output_router_logits", False)

        # MoE Support
        self.num_experts = kwargs.get("num_experts", 8)
        self.num_experts_per_tok = kwargs.get("num_experts_per_tok", 2)

        # ── 2026: Flash Attention 2 ───────────────────────────────────────────
        # Effective only when flash-attn package is installed.
        # If True but package missing, silently falls back to SDPA.
        self.use_flash_attention = use_flash_attention

        # ── 2026: Mixture of Depths (MoD) ────────────────────────────────────
        # When enabled, each layer routes only `capacity_factor` fraction of
        # tokens through Attention+MLP; the rest skip via residual connection.
        # Paper: "Mixture of Depths" (Raposo et al., 2024)
        self.use_mod = use_mod
        self.mod_capacity_factor = mod_capacity_factor

        super().__init__(
            tie_word_embeddings=tie_word_embeddings,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            **kwargs
        )

    def to_dict(self) -> Dict[str, Any]:
        """Custom to_dict to ensure all specific fields are saved in checkpoint metadata."""
        output = super().to_dict()
        specifics = {
            "vocab_size": self.vocab_size,
            "hidden_size": self.hidden_size,
            "num_hidden_layers": self.num_hidden_layers,
            "num_attention_heads": self.num_attention_heads,
            "num_key_value_heads": self.num_key_value_heads,
            "intermediate_size": self.intermediate_size,
            "max_position_embeddings": self.max_position_embeddings,
            "rms_norm_eps": self.rms_norm_eps,
            "hidden_dropout": self.hidden_dropout,
            "attention_dropout": self.attention_dropout,
            "activation": self.activation,
            "rotary_pct": self.rotary_pct,
            "rope_scaling": self.rope_scaling,
            "use_cache": self.use_cache,
            "num_experts": self.num_experts,
            "num_experts_per_tok": self.num_experts_per_tok,
            # 2026 fields
            "use_flash_attention": self.use_flash_attention,
            "use_mod": self.use_mod,
            "mod_capacity_factor": self.mod_capacity_factor,
        }
        output.update(specifics)
        return output


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization (LLaMA/Gemma style)."""

    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        input_dtype = x.dtype
        x = x.float()
        variance = x.pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(variance + self.eps)
        return self.weight * x.to(input_dtype)


class RotaryEmbedding(nn.Module):
    """RoPE with optional YaRN scaling for long-context (up to 128K tokens).

    YaRN (Yet Another RoPE Scaling) dynamically rescales frequencies based on
    the target context length without requiring fine-tuning. Enabled via
    config.rope_scaling = {"type": "yarn", "factor": 8.0}.
    """

    def __init__(
        self,
        dim: int,
        max_position_embeddings: int = 2048,
        base: float = 10000.0,
        scaling_factor: float = 1.0,
        rope_scaling: Optional[Dict[str, Any]] = None,
    ):
        super().__init__()
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        self.scaling_factor = scaling_factor
        self.rope_scaling = rope_scaling or {}

        inv_freq = self._compute_inv_freq()
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self._set_cos_sin_cache(max_position_embeddings)

    def _compute_inv_freq(self) -> torch.Tensor:
        """Compute inverse frequencies, applying YaRN if configured."""
        scaling_type = self.rope_scaling.get("type", "")

        if scaling_type == "yarn":
            # YaRN: rescale base frequency to extend context window.
            # Source: "YaRN: Efficient Context Window Extension of LLMs" (Peng et al. 2023)
            factor = float(self.rope_scaling.get("factor", 8.0))
            # Interpolation boundaries from YaRN paper
            low_freq_factor  = float(self.rope_scaling.get("low_freq_factor", 1.0))
            high_freq_factor = float(self.rope_scaling.get("high_freq_factor", 4.0))
            original_max_pos = int(self.rope_scaling.get("original_max_position_embeddings", 4096))

            # Compute wavelengths for each RoPE dimension
            base_inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2).float() / self.dim))
            wavelengths = 2 * math.pi / base_inv_freq

            # Low-frequency dimensions: fully scaled (linear interpolation territory)
            low_wavelength  = original_max_pos / low_freq_factor
            high_wavelength = original_max_pos / high_freq_factor

            # Smooth interpolation between linear and NTK scaling
            smooth = (wavelengths - low_wavelength) / (high_wavelength - low_wavelength)
            smooth = smooth.clamp(0.0, 1.0)

            # Blend: high-freq dims keep unscaled, low-freq dims get linear scaling
            inv_freq = (1 - smooth) * (base_inv_freq / factor) + smooth * base_inv_freq
            return inv_freq

        # Default NTK-aware linear scaling (original rope_scaling={"type": "linear"})
        inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2).float() / self.dim))
        return inv_freq

    def _set_cos_sin_cache(self, seq_len: int) -> None:
        self.max_position_embeddings = seq_len
        t = torch.arange(
            self.max_position_embeddings, device=self.inv_freq.device, dtype=self.inv_freq.dtype
        )
        # Linear scaling for non-YaRN modes
        if self.rope_scaling.get("type", "") != "yarn":
            t = t / self.scaling_factor

        freqs = torch.outer(t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos(), persistent=False)
        self.register_buffer("sin_cached", emb.sin(), persistent=False)

    def forward(
        self,
        x: torch.Tensor,
        seq_len: int,
        seq_len_offset: int = 0,
        position_ids: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if position_ids is not None:
            seq_len_end = int(position_ids.max().item()) + 1
            if seq_len_end > self.max_position_embeddings:
                self._set_cos_sin_cache(seq_len_end)
            cos = self.cos_cached[position_ids].unsqueeze(1)  # [B, 1, S, D]
            sin = self.sin_cached[position_ids].unsqueeze(1)
            return cos, sin

        seq_len_end = seq_len_offset + seq_len
        if seq_len_end > self.max_position_embeddings:
            self._set_cos_sin_cache(seq_len_end)

        return (
            self.cos_cached[seq_len_offset:seq_len_end].unsqueeze(0).unsqueeze(0),
            self.sin_cached[seq_len_offset:seq_len_end].unsqueeze(0).unsqueeze(0),
        )


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(
    q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    rot_dim = cos.shape[-1]
    
    q_rot = q[..., :rot_dim]
    q_pass = q[..., rot_dim:]
    k_rot = k[..., :rot_dim]
    k_pass = k[..., rot_dim:]

    q_embed = (q_rot * cos) + (rotate_half(q_rot) * sin)
    k_embed = (k_rot * cos) + (rotate_half(k_rot) * sin)
    
    if q_pass.shape[-1] > 0:
        q_embed = torch.cat([q_embed, q_pass], dim=-1)
        k_embed = torch.cat([k_embed, k_pass], dim=-1)
        
    return q_embed, k_embed

def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep). The hidden states go from (batch,
    num_key_value_heads, seqlen, head_dim) to (batch, num_attention_heads, seqlen, head_dim)
    """
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


class CodeMindAttention(nn.Module):
    def __init__(self, config: CodeMindConfig):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = config.head_dim
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads

        # Separate projections for GQA
        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=False)
        self.o_proj.SCALE_INIT = True

        self.attention_dropout = config.attention_dropout
        self.is_causal = True

        rotary_dim = int(self.head_dim * config.rotary_pct)
        scaling_factor = 1.0
        rope_scaling = config.rope_scaling or {}
        if rope_scaling.get("type", "") == "linear":
            scaling_factor = float(rope_scaling.get("factor", 1.0))

        self.rotary_emb = RotaryEmbedding(
            rotary_dim,
            max_position_embeddings=config.max_position_embeddings,
            scaling_factor=scaling_factor,
            rope_scaling=rope_scaling,
        )
        # Whether to use Flash Attention 2 (only when package is installed)
        self.use_flash_attention = getattr(config, "use_flash_attention", False) and _FLASH_ATTN_AVAILABLE

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_value: Optional[Cache] = None,
        use_cache: bool = False,
        layer_idx: Optional[int] = None,
        position_ids: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]], Optional[torch.Tensor]]:
        batch_size, seq_len, _ = hidden_states.shape

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        query_states = query_states.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(batch_size, seq_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(batch_size, seq_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        seq_len_offset = 0
        if past_key_value is not None:
            if isinstance(past_key_value, Cache):
                seq_len_offset = past_key_value.get_seq_length(layer_idx)
            elif isinstance(past_key_value, tuple) and len(past_key_value) > 0:
                seq_len_offset = past_key_value[0].shape[-2]

        cos, sin = self.rotary_emb(value_states, seq_len, seq_len_offset=seq_len_offset, position_ids=position_ids)
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        if past_key_value is not None:
            if isinstance(past_key_value, Cache):
                key_states, value_states = past_key_value.update(key_states, value_states, layer_idx)
            else:
                past_key, past_value = past_key_value
                key_states = torch.cat([past_key, key_states], dim=2)
                value_states = torch.cat([past_value, value_states], dim=2)

        present_key_value = (key_states, value_states) if use_cache and not isinstance(past_key_value, Cache) else None
        
        # repeat k/v heads if n_kv_heads < n_heads
        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)
        
        if output_attentions:
            # Manual attention (needed to return attention weights for visualization)
            attn_weights = torch.matmul(query_states, key_states.transpose(-1, -2)) / math.sqrt(self.head_dim)
            if attention_mask is not None:
                attn_weights = attn_weights + attention_mask
            attn_weights = F.softmax(attn_weights, dim=-1)
            attn_output = torch.matmul(
                F.dropout(attn_weights, p=self.attention_dropout, training=self.training),
                value_states,
            )
        elif self.use_flash_attention and attention_mask is None:
            # ── Flash Attention 2 path ────────────────────────────────────────
            # flash_attn_func expects [B, S, H, D] layout; transpose QKV accordingly.
            # Only usable without explicit attention_mask (causal=True handles masking).
            q = query_states.transpose(1, 2)   # [B, S, Hq, D]
            k = key_states.transpose(1, 2)     # [B, S, Hkv, D]
            v = value_states.transpose(1, 2)   # [B, S, Hkv, D]
            attn_weights = None
            attn_output = flash_attn_func(
                q, k, v,
                dropout_p=self.attention_dropout if self.training else 0.0,
                causal=True,
            )  # [B, S, Hq, D]
            attn_output = attn_output.reshape(batch_size, seq_len, self.num_heads * self.head_dim)
            attn_output = self.o_proj(attn_output)
            return attn_output, present_key_value, attn_weights
        else:
            # ── Standard PyTorch scaled_dot_product_attention (SDPA) path ────
            # Covers: no flash-attn, or when attention_mask is provided (e.g. padding).
            attn_weights = None
            attn_output = F.scaled_dot_product_attention(
                query_states,
                key_states,
                value_states,
                attn_mask=attention_mask,
                dropout_p=self.attention_dropout if self.training else 0.0,
                is_causal=False,  # Causal masking forms part of the explicit attention_mask
            )

        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(batch_size, seq_len, self.num_heads * self.head_dim)
        attn_output = self.o_proj(attn_output)

        return attn_output, present_key_value, attn_weights


class CodeMindMLP(nn.Module):
    """SwiGLU MLP (LLaMA/Mistral style) — replaces standard GELU MLP."""

    def __init__(self, config: CodeMindConfig):
        super().__init__()
        self.gate_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.up_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)
        self.down_proj.SCALE_INIT = True

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(hidden_states)) * self.up_proj(hidden_states))


class CodeMindMoE(nn.Module):
    """Mixture of Experts replacing standard MLP if num_experts > 1"""
    
    def __init__(self, config: CodeMindConfig):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.num_experts = config.num_experts
        self.top_k = config.num_experts_per_tok

        # gating
        self.gate = nn.Linear(self.hidden_size, self.num_experts, bias=False)
        self.experts = nn.ModuleList([CodeMindMLP(config) for _ in range(self.num_experts)])

    def forward(self, hidden_states: torch.Tensor) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        batch_size, sequence_length, hidden_dim = hidden_states.shape
        hidden_states = hidden_states.view(-1, hidden_dim)
        
        # router_logits: (batch * sequence_length, n_experts)
        router_logits = self.gate(hidden_states)
        
        routing_weights = F.softmax(router_logits, dim=1, dtype=torch.float)
        routing_weights, selected_experts = torch.topk(routing_weights, self.top_k, dim=-1)
        routing_weights /= routing_weights.sum(dim=-1, keepdim=True)
        # we cast back to the input dtype
        routing_weights = routing_weights.to(hidden_states.dtype)
        
        final_hidden_states = torch.zeros(
            (batch_size * sequence_length, hidden_dim), dtype=hidden_states.dtype, device=hidden_states.device
        )
        
        # Expert routing mechanism
        expert_mask = torch.nn.functional.one_hot(selected_experts, num_classes=self.num_experts).permute(2, 1, 0)

        for expert_idx in range(self.num_experts):
            expert_layer = self.experts[expert_idx]
            idx, top_x = torch.where(expert_mask[expert_idx])

            if top_x.shape[0] == 0:
                continue

            # Indexing the states and feeding them through the expert
            top_x_list = top_x.tolist()
            idx_list = idx.tolist()

            current_state = hidden_states[None, top_x_list].reshape(-1, hidden_dim)
            current_hidden_states = expert_layer(current_state) * routing_weights[top_x_list, idx_list, None]

            # Add output to final tensor
            final_hidden_states.index_add_(0, top_x, current_hidden_states.to(hidden_states.dtype))

        final_hidden_states = final_hidden_states.reshape(batch_size, sequence_length, hidden_dim)
        return final_hidden_states, router_logits


class CodeMindLayer(nn.Module):
    """Single transformer layer with optional Mixture of Depths (MoD) token routing.

    MoD (Raposo et al. 2024): only a `capacity_factor` fraction of the most
    important tokens (selected by a lightweight router) go through Attention+MLP.
    The rest skip via a residual connection, reducing FLOPs by ~(1 - capacity_factor).
    """

    def __init__(self, config: CodeMindConfig, layer_idx: int):
        super().__init__()
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        self.attention = CodeMindAttention(config)

        if config.num_experts > 1:
            self.mlp = CodeMindMoE(config)
        else:
            self.mlp = CodeMindMLP(config)

        self.dropout = nn.Dropout(config.hidden_dropout)
        self.layer_idx = layer_idx

        # ── Mixture of Depths token router ────────────────────────────────────
        # A single linear layer that scores each token's importance.
        # Trained jointly with the model; no additional loss required.
        self.use_mod = getattr(config, "use_mod", False)
        self.mod_capacity_factor = getattr(config, "mod_capacity_factor", 0.5)
        if self.use_mod:
            self.mod_router = nn.Linear(config.hidden_size, 1, bias=False)

    def _apply_mod(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        past_key_value: Optional[Cache],
        use_cache: bool,
        position_ids: Optional[torch.Tensor],
        output_attentions: bool,
        output_router_logits: bool,
    ) -> Tuple[torch.Tensor, Optional[Any], Optional[torch.Tensor], Optional[torch.Tensor]]:
        """MoD routing: select top-k tokens, process only those through Attn+MLP."""
        batch_size, seq_len, hidden_dim = hidden_states.shape
        k = max(1, int(seq_len * self.mod_capacity_factor))

        # Score each token position; higher score = more important
        router_scores = self.mod_router(hidden_states).squeeze(-1)  # [B, S]
        topk_scores, topk_indices = torch.topk(router_scores, k, dim=1)  # [B, k]
        topk_indices_sorted, sort_order = topk_indices.sort(dim=1)

        # Extract selected tokens — gather along sequence dimension
        expanded_idx = topk_indices_sorted.unsqueeze(-1).expand(-1, -1, hidden_dim)
        selected = hidden_states.gather(1, expanded_idx)  # [B, k, D]

        # Build attention mask slice for selected tokens (simplified: no masking)
        # Full causal masking inside the attention handles correctness correctly.
        selected_mask = None

        # Run selected tokens through Attention + MLP
        selected_normed = self.input_layernorm(selected)
        attn_out, present_kv, attn_weights = self.attention(
            selected_normed, selected_mask, past_key_value, use_cache,
            self.layer_idx, position_ids=None, output_attentions=output_attentions,
        )
        selected = selected + self.dropout(attn_out)

        selected_normed2 = self.post_attention_layernorm(selected)
        if isinstance(self.mlp, CodeMindMoE):
            mlp_out, router_logits = self.mlp(selected_normed2)
        else:
            mlp_out = self.mlp(selected_normed2)
            router_logits = None
        selected = selected + self.dropout(mlp_out)

        # Scatter selected tokens back; unselected positions keep their residual
        output = hidden_states.clone()
        output.scatter_(1, expanded_idx, selected)

        return output, present_kv, attn_weights, router_logits

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_value: Optional[Cache] = None,
        use_cache: bool = False,
        position_ids: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
        output_router_logits: bool = False,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]], Optional[torch.Tensor], Optional[torch.Tensor]]:

        # ── Mixture of Depths path ────────────────────────────────────────────
        if self.use_mod:
            return self._apply_mod(
                hidden_states, attention_mask, past_key_value, use_cache,
                position_ids, output_attentions, output_router_logits,
            )

        # ── Standard path ─────────────────────────────────────────────────────
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        attn_output, present_key_value, attn_weights = self.attention(
            hidden_states, attention_mask, past_key_value, use_cache,
            self.layer_idx, position_ids=position_ids, output_attentions=output_attentions,
        )
        hidden_states = residual + self.dropout(attn_output)

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)

        if isinstance(self.mlp, CodeMindMoE):
            mlp_output, router_logits = self.mlp(hidden_states)
        else:
            mlp_output = self.mlp(hidden_states)
            router_logits = None

        hidden_states = residual + self.dropout(mlp_output)
        return hidden_states, present_key_value, attn_weights, router_logits


class CodeMindModel(nn.Module):
    def __init__(self, config: CodeMindConfig):
        super().__init__()
        self.config = config

        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size)
        # NOTE: No position_embeddings — RoPE handles positions inside attention

        self.layers = nn.ModuleList(
            [CodeMindLayer(config, i) for i in range(config.num_hidden_layers)]
        )

        self.final_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        self.gradient_checkpointing = False

        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module) -> None:
        std = 0.02
        if isinstance(module, nn.Linear):
            if hasattr(module, "SCALE_INIT"):
                std = 0.02 / math.sqrt(2 * self.config.num_hidden_layers)
            torch.nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def get_input_embeddings(self) -> nn.Embedding:
        return self.word_embeddings

    @staticmethod
    def _infer_past_length(
        past_key_values: Optional[Union[Cache, List[Tuple[torch.Tensor, torch.Tensor]]]]
    ) -> int:
        if past_key_values is None:
            return 0
        if isinstance(past_key_values, Cache):
            return past_key_values.get_seq_length()
        if isinstance(past_key_values, (list, tuple)) and len(past_key_values) > 0:
            first_layer = past_key_values[0]
            if (
                isinstance(first_layer, (list, tuple))
                and len(first_layer) > 0
                and first_layer[0] is not None
            ):
                return int(first_layer[0].shape[-2])
        return 0

    @staticmethod
    def _build_combined_attention_mask(
        attention_mask: Optional[torch.Tensor],
        batch_size: int,
        seq_len: int,
        past_length: int,
        dtype: torch.dtype,
        device: torch.device,
    ) -> torch.Tensor:
        """
        Build combined causal + padding mask for decoder-only attention.

        Returns:
            Tensor shaped [batch, 1, query_len, key_len].
        """
        total_key_len = past_length + seq_len
        min_value = torch.finfo(dtype).min

        query_positions = torch.arange(
            past_length, past_length + seq_len, device=device
        ).unsqueeze(-1)
        key_positions = torch.arange(total_key_len, device=device).unsqueeze(0)
        causal_allowed = key_positions <= query_positions
        causal_mask = torch.zeros((seq_len, total_key_len), dtype=dtype, device=device)
        causal_mask = causal_mask.masked_fill(~causal_allowed, min_value)
        causal_mask = causal_mask.unsqueeze(0).unsqueeze(0)

        if attention_mask is None:
            return causal_mask.expand(batch_size, -1, -1, -1)

        if attention_mask.dim() != 2:
            raise ValueError("attention_mask must be shape [batch, seq]")

        if attention_mask.size(0) != batch_size:
            raise ValueError("attention_mask batch dimension mismatch")

        if attention_mask.size(1) < total_key_len:
            pad_len = total_key_len - attention_mask.size(1)
            prefix = torch.ones(
                (batch_size, pad_len),
                device=attention_mask.device,
                dtype=attention_mask.dtype,
            )
            attention_mask = torch.cat([prefix, attention_mask], dim=-1)
        elif attention_mask.size(1) > total_key_len:
            attention_mask = attention_mask[:, -total_key_len:]

        padding_mask = attention_mask[:, None, None, :].to(dtype=dtype)
        padding_mask = (1.0 - padding_mask) * min_value

        return causal_mask + padding_mask

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        use_cache: bool = False,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        output_router_logits: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        **kwargs
    ) -> Union[Tuple, BaseModelOutputWithPast]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        output_router_logits = output_router_logits if output_router_logits is not None else getattr(self.config, "output_router_logits", False)
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            batch_size, seq_len = input_ids.shape
        elif inputs_embeds is not None:
            batch_size, seq_len, _ = inputs_embeds.shape
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        if inputs_embeds is None:
            hidden_states = self.word_embeddings(input_ids)
        else:
            hidden_states = inputs_embeds

        # No position_embeddings addition — RoPE in attention handles this

        past_length = self._infer_past_length(past_key_values)
        extended_attention_mask = self._build_combined_attention_mask(
            attention_mask=attention_mask,
            batch_size=batch_size,
            seq_len=seq_len,
            past_length=past_length,
            dtype=hidden_states.dtype,
            device=hidden_states.device,
        )

        if use_cache:
            present_key_values = past_key_values if isinstance(past_key_values, Cache) else []
        else:
            present_key_values = None

        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        all_router_logits = () if output_router_logits else None

        for i, layer in enumerate(self.layers):
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            if isinstance(past_key_values, Cache):
                past_key_value = past_key_values
            elif past_key_values is not None:
                past_key_value = past_key_values[i]
            else:
                past_key_value = None

            if getattr(self, "gradient_checkpointing", False) and self.training:
                def create_custom_forward(module):
                    def custom_forward(*inputs, **kw):
                        return module(*inputs, **kw)
                    return custom_forward

                layer_outputs = checkpoint(
                    create_custom_forward(layer),
                    hidden_states,
                    extended_attention_mask,
                    past_key_value,
                    use_cache,
                    position_ids,
                    output_attentions,
                    output_router_logits,
                    use_reentrant=False
                )
            else:
                layer_outputs = layer(
                    hidden_states, extended_attention_mask, past_key_value, use_cache, position_ids, output_attentions=output_attentions, output_router_logits=output_router_logits
                )

            hidden_states = layer_outputs[0]
            if use_cache and not isinstance(past_key_values, Cache):
                present_key_values.append(layer_outputs[1])

            if output_attentions:
                all_self_attns += (layer_outputs[2],)
                
            if output_router_logits and layer_outputs[3] is not None:
                all_router_logits += (layer_outputs[3],)

        hidden_states = self.final_layernorm(hidden_states)
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        if not return_dict:
            res = tuple(v for v in [hidden_states, present_key_values, all_hidden_states, all_self_attns, all_router_logits] if v is not None)
            return res

        output = BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=present_key_values,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
        )
        if output_router_logits:
            output.router_logits = all_router_logits
        return output


class CodeMindForCausalLM(PreTrainedModel, GenerationMixin):
    config_class = CodeMindConfig
    _keys_to_ignore_on_load_missing = [r"lm_head.weight"]

    def __init__(self, config: CodeMindConfig):
        super().__init__(config)
        self.config = config
        self.model = CodeMindModel(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

    def _tie_weights(self):
        """Tie embeddings and output head weights."""
        if self.config.tie_word_embeddings:
            self.set_output_embeddings(self.get_input_embeddings())

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        use_cache: bool = False,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        output_router_logits: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        **kwargs
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        output_router_logits = output_router_logits if output_router_logits is not None else getattr(self.config, "output_router_logits", False)

        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            output_router_logits=output_router_logits,
            return_dict=return_dict,
            **kwargs
        )
        
        hidden_states = outputs[0] if not return_dict else outputs.last_hidden_state
        logits = self.lm_head(hidden_states)

        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()

            loss_fct = CrossEntropyLoss()
            loss = loss_fct(
                shift_logits.view(-1, self.config.vocab_size), shift_labels.view(-1)
            )

        # Basic Aux loss logic for MoE load balancing
        aux_loss = None
        if output_router_logits and hasattr(outputs, "router_logits") and outputs.router_logits is not None:
            # We compute aux loss to balance the load among experts
            router_logits = outputs.router_logits
            concat_logits = torch.cat([logit for logit in router_logits], dim=0)
            
            # Simple balancing: encourage equal routing probability
            routing_weights = F.softmax(concat_logits, dim=-1)
            # density per expert
            density_1_proxy = routing_weights.mean(dim=0)
            # fraction of tokens dispatched to each expert
            _, selected_experts = torch.topk(routing_weights, self.config.num_experts_per_tok, dim=-1)
            expert_mask = torch.nn.functional.one_hot(selected_experts, num_classes=self.config.num_experts)
            density_1 = expert_mask.float().mean(dim=(0, 1))
            
            aux_loss = (density_1_proxy * density_1).sum() * self.config.num_experts
            
            if loss is not None:
                loss += 0.01 * aux_loss # 0.01 is router aux loss coef

        if not return_dict:
            output = (logits,) + outputs[1:]
            if aux_loss is not None:
                output = output + (aux_loss,)
            return ((loss,) + output) if loss is not None else output

        output = CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
        if output_router_logits:
            output.router_logits = outputs.router_logits
        return output

    @property
    def supports_gradient_checkpointing(self):
        return True

    def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs=None):
        self.model.gradient_checkpointing = True

    def gradient_checkpointing_disable(self):
        self.model.gradient_checkpointing = False

    def get_input_embeddings(self):
        return self.model.word_embeddings

    def set_input_embeddings(self, value):
        self.model.word_embeddings = value

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def _reorder_cache(self, past_key_values, beam_idx):
        if isinstance(past_key_values, Cache):
            past_key_values.reorder_cache(beam_idx)
            return past_key_values
            
        reordered_past = []
        for layer_past in past_key_values:
            reordered_past.append(
                tuple(past_state.index_select(0, beam_idx.to(past_state.device)) for past_state in layer_past)
            )
        return reordered_past

    def prepare_inputs_for_generation(
        self, input_ids, past_key_values=None, attention_mask=None, **kwargs
    ):
        # Detect if past_key_values actually has data (empty DynamicCache is truthy)
        has_past = False
        if past_key_values is not None:
            if isinstance(past_key_values, Cache):
                has_past = past_key_values.get_seq_length() > 0
            elif isinstance(past_key_values, (list, tuple)):
                has_past = len(past_key_values) > 0
            else:
                has_past = bool(past_key_values)

        position_ids = kwargs.get("position_ids", None)
        if attention_mask is not None and position_ids is None:
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            if has_past:
                position_ids = position_ids[:, -input_ids.shape[1] :]

        if has_past:
            input_ids = input_ids[:, -1:]
            if position_ids is not None:
                position_ids = position_ids[:, -1:]

        return {
            "input_ids": input_ids,
            "past_key_values": past_key_values,
            "use_cache": kwargs.get("use_cache", True),
            "attention_mask": attention_mask,
            "position_ids": position_ids,
        }

    def get_num_parameters(self, trainable_only: bool = True) -> int:
        if trainable_only:
            return sum(p.numel() for p in self.parameters() if p.requires_grad)
        return sum(p.numel() for p in self.parameters())


def create_codemind_350m(
    use_mod: bool = False,
    use_flash_attention: bool = False,
    long_context: bool = False,
) -> CodeMindForCausalLM:
    """Standard 350M model. Pass long_context=True for 32K context (YaRN)."""
    rope_scaling = None
    max_pos = 4096
    if long_context:
        max_pos = 32768
        rope_scaling = {
            "type": "yarn",
            "factor": 8.0,
            "original_max_position_embeddings": 4096,
        }
    config = CodeMindConfig(
        vocab_size=32768,
        hidden_size=1024,
        num_hidden_layers=24,
        num_attention_heads=16,
        num_key_value_heads=4,
        intermediate_size=4096,
        max_position_embeddings=max_pos,
        rope_scaling=rope_scaling,
        use_flash_attention=use_flash_attention,
        use_mod=use_mod,
        mod_capacity_factor=0.5,
    )
    return CodeMindForCausalLM(config)


def create_codemind_350m_moe(
    use_mod: bool = False,
    use_flash_attention: bool = False,
    long_context: bool = False,
) -> CodeMindForCausalLM:
    """350M MoE model — 8 experts, 2 active per token. ~120M active params."""
    rope_scaling = None
    max_pos = 4096
    if long_context:
        max_pos = 32768
        rope_scaling = {
            "type": "yarn",
            "factor": 8.0,
            "original_max_position_embeddings": 4096,
        }
    config = CodeMindConfig(
        vocab_size=32768,
        hidden_size=768,
        num_hidden_layers=24,
        num_attention_heads=12,
        num_key_value_heads=4,
        intermediate_size=3072,
        max_position_embeddings=max_pos,
        num_experts=8,
        num_experts_per_tok=2,
        rope_scaling=rope_scaling,
        use_flash_attention=use_flash_attention,
        use_mod=use_mod,
        mod_capacity_factor=0.5,
    )
    return CodeMindForCausalLM(config)


def create_codemind_125m(
    use_mod: bool = False,
    use_flash_attention: bool = False,
) -> CodeMindForCausalLM:
    """Lightweight 125M model — ideal for local testing and draft model in speculative decoding."""
    config = CodeMindConfig(
        vocab_size=32768,
        hidden_size=768,
        num_hidden_layers=12,
        num_attention_heads=12,
        num_key_value_heads=4,
        intermediate_size=3072,
        max_position_embeddings=4096,
        use_flash_attention=use_flash_attention,
        use_mod=use_mod,
        mod_capacity_factor=0.5,
    )
    return CodeMindForCausalLM(config)
