"""
CodeMind Base Model Architecture

Optimized GPT-NeoX style model for:
- Code generation
- Multi-language support (Python, Dart, JS)
- Efficient inference on limited hardware

Architecture features:
- RMSNorm (LLaMA/Gemma style)
- SwiGLU activation (LLaMA/Mistral style)
- Rotary Position Embeddings (RoPE)
- Weight-tied embeddings
"""

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
        hidden_size: int = 1024,
        num_hidden_layers: int = 24,
        num_attention_heads: int = 16,
        intermediate_size: int = 4096,
        max_position_embeddings: int = 2048,
        rms_norm_eps: float = 1e-6,
        hidden_dropout: float = 0.0,
        attention_dropout: float = 0.0,
        activation: str = "silu",
        rotary_pct: float = 1.0,
        use_cache: bool = True,
        tie_word_embeddings: bool = True,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        **kwargs
    ):
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.intermediate_size = intermediate_size
        self.max_position_embeddings = max_position_embeddings
        self.rms_norm_eps = rms_norm_eps
        self.hidden_dropout = hidden_dropout
        self.attention_dropout = attention_dropout
        self.activation = activation
        self.rotary_pct = rotary_pct
        self.use_cache = use_cache
        self.head_dim = self.hidden_size // self.num_attention_heads
        self.output_attentions = output_attentions
        self.output_hidden_states = output_hidden_states
        
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
            "intermediate_size": self.intermediate_size,
            "max_position_embeddings": self.max_position_embeddings,
            "rms_norm_eps": self.rms_norm_eps,
            "hidden_dropout": self.hidden_dropout,
            "attention_dropout": self.attention_dropout,
            "activation": self.activation,
            "rotary_pct": self.rotary_pct,
            "use_cache": self.use_cache,
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
    def __init__(
        self, dim: int, max_position_embeddings: int = 2048, base: float = 10000.0
    ):
        super().__init__()
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base

        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

        self._set_cos_sin_cache(max_position_embeddings)

    def _set_cos_sin_cache(self, seq_len: int):
        self.max_position_embeddings = seq_len
        t = torch.arange(
            self.max_position_embeddings, device=self.inv_freq.device, dtype=self.inv_freq.dtype
        )
        freqs = torch.outer(t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos(), persistent=False)
        self.register_buffer("sin_cached", emb.sin(), persistent=False)

    def forward(
        self, x: torch.Tensor, seq_len: int, seq_len_offset: int = 0, position_ids: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if position_ids is not None:
            seq_len_end = int(position_ids.max().item()) + 1
            if seq_len_end > self.max_position_embeddings:
                self._set_cos_sin_cache(seq_len_end)
            cos = self.cos_cached[position_ids].unsqueeze(1) # [batch, 1, seq_len, dim]
            sin = self.sin_cached[position_ids].unsqueeze(1)
            return cos, sin

        seq_len_end = seq_len_offset + seq_len
        if seq_len_end > self.max_position_embeddings:
            self._set_cos_sin_cache(seq_len_end)
            
        return (
            self.cos_cached[seq_len_offset : seq_len_end].unsqueeze(0).unsqueeze(0),
            self.sin_cached[seq_len_offset : seq_len_end].unsqueeze(0).unsqueeze(0),
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


class CodeMindAttention(nn.Module):
    def __init__(self, config: CodeMindConfig):
        super().__init__()
        self.config = config
        self.num_heads = config.num_attention_heads
        self.head_dim = config.head_dim
        self.hidden_size = config.hidden_size

        self.query_key_value = nn.Linear(
            config.hidden_size, 3 * config.hidden_size, bias=False
        )
        self.dense = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        self.dense.SCALE_INIT = True

        self.attention_dropout = nn.Dropout(config.attention_dropout)

        rotary_dim = int(config.head_dim * config.rotary_pct)
        self.rotary_emb = RotaryEmbedding(
            rotary_dim, max_position_embeddings=config.max_position_embeddings
        )

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

        qkv = self.query_key_value(hidden_states)
        qkv = qkv.view(batch_size, seq_len, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        query, key, value = qkv[0], qkv[1], qkv[2]

        seq_len_offset = 0
        if past_key_value is not None:
            if isinstance(past_key_value, Cache):
                seq_len_offset = past_key_value.get_seq_length(layer_idx)
            elif isinstance(past_key_value, tuple) and len(past_key_value) > 0:
                seq_len_offset = past_key_value[0].shape[-2]

        cos, sin = self.rotary_emb(value, seq_len, seq_len_offset=seq_len_offset, position_ids=position_ids)
        query, key = apply_rotary_pos_emb(query, key, cos, sin)

        if past_key_value is not None:
            if isinstance(past_key_value, Cache):
                key, value = past_key_value.update(key, value, layer_idx)
            else:
                past_key, past_value = past_key_value
                key = torch.cat([past_key, key], dim=2)
                value = torch.cat([past_value, value], dim=2)

        present_key_value = (key, value) if use_cache and not isinstance(past_key_value, Cache) else None
        
        if output_attentions:
            attn_weights = torch.matmul(query, key.transpose(-1, -2)) / math.sqrt(self.head_dim)
            if attention_mask is not None:
                attn_weights = attn_weights + attention_mask
            attn_weights = F.softmax(attn_weights, dim=-1)
            attn_output = torch.matmul(self.attention_dropout(attn_weights), value)
        else:
            attn_weights = None
            attn_output = F.scaled_dot_product_attention(
                query,
                key,
                value,
                attn_mask=attention_mask,
                dropout_p=self.config.attention_dropout if self.training else 0.0,
                is_causal=False
            )

        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(batch_size, seq_len, self.hidden_size)

        attn_output = self.dense(attn_output)

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


class CodeMindLayer(nn.Module):
    def __init__(self, config: CodeMindConfig, layer_idx: int):
        super().__init__()
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        self.attention = CodeMindAttention(config)
        self.mlp = CodeMindMLP(config)
        
        self.dropout = nn.Dropout(config.hidden_dropout)

        self.layer_idx = layer_idx

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_value: Optional[Cache] = None,
        use_cache: bool = False,
        position_ids: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]], Optional[torch.Tensor]]:
        residual = hidden_states

        hidden_states = self.input_layernorm(hidden_states)
        attn_output, present_key_value, attn_weights = self.attention(
            hidden_states, attention_mask, past_key_value, use_cache, self.layer_idx, position_ids=position_ids, output_attentions=output_attentions
        )
        hidden_states = residual + self.dropout(attn_output)

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        mlp_output = self.mlp(hidden_states)
        hidden_states = residual + self.dropout(mlp_output)

        return hidden_states, present_key_value, attn_weights


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
        return_dict: Optional[bool] = None,
        **kwargs
    ) -> Union[Tuple, BaseModelOutputWithPast]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
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
                    use_reentrant=False
                )
            else:
                layer_outputs = layer(
                    hidden_states, extended_attention_mask, past_key_value, use_cache, position_ids, output_attentions=output_attentions
                )

            hidden_states = layer_outputs[0]
            if use_cache and not isinstance(past_key_values, Cache):
                present_key_values.append(layer_outputs[1])

            if output_attentions:
                all_self_attns += (layer_outputs[2],)

        hidden_states = self.final_layernorm(hidden_states)
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        if not return_dict:
            return tuple(v for v in [hidden_states, present_key_values, all_hidden_states, all_self_attns] if v is not None)

        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=present_key_values,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
        )


class CodeMindForCausalLM(PreTrainedModel, GenerationMixin):
    config_class = CodeMindConfig
    _keys_to_ignore_on_load_missing = [r"lm_head.weight"]

    def __init__(self, config: CodeMindConfig):
        super().__init__(config)
        self.config = config
        self.model = CodeMindModel(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Tie weights
        self.lm_head.weight = self.model.word_embeddings.weight

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
        return_dict: Optional[bool] = None,
        **kwargs
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
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

        if not return_dict:
            output = (logits,) + outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

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


def create_codemind_350m() -> CodeMindForCausalLM:
    config = CodeMindConfig(
        vocab_size=32768,
        hidden_size=1024,
        num_hidden_layers=24,
        num_attention_heads=16,
        intermediate_size=4096,
        max_position_embeddings=2048,
    )
    return CodeMindForCausalLM(config)


def create_codemind_125m() -> CodeMindForCausalLM:
    config = CodeMindConfig(
        vocab_size=32768,
        hidden_size=768,
        num_hidden_layers=12,
        num_attention_heads=12,
        intermediate_size=3072,
        max_position_embeddings=2048,
    )
    return CodeMindForCausalLM(config)
