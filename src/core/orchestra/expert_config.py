"""
Expert Configuration — defines expert model specifications for the Latent Expert Orchestra.

Each ExpertConfig describes the architecture and training parameters for a single
specialized expert model. All experts share a common hidden_size and tokenizer
to enable latent-level inter-expert communication via the MLP Bridge.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional


class ExpertDomain(str, Enum):
    """Specialized domains in the Latent Expert Orchestra.

    MVP starts with TURKISH and CODE. Others are planned for future phases.
    """

    TURKISH = "turkish"          # Turkish + English bilingual
    CODE = "code"                # Python code generation
    MATH = "math"                # Mathematics and formal reasoning
    LOGIC = "logic"              # Logical reasoning and analysis
    GENERAL = "general"          # General knowledge fallback


# ── Shared Architecture Contract ──────────────────────────────────────────────
# All experts MUST share these values for latent communication to work.
SHARED_HIDDEN_SIZE = 768
SHARED_VOCAB_SIZE = 32768
SHARED_NUM_ATTENTION_HEADS = 12
SHARED_NUM_KV_HEADS = 4
SHARED_HEAD_DIM = SHARED_HIDDEN_SIZE // SHARED_NUM_ATTENTION_HEADS  # 64


@dataclass
class ExpertConfig:
    """Configuration for a single expert model in the orchestra.

    All experts share hidden_size=768 and the same tokenizer vocabulary.
    They differ in depth (num_hidden_layers) and intermediate_size to
    control parameter count and specialization capacity.
    """

    domain: ExpertDomain
    num_hidden_layers: int = 12
    intermediate_size: int = 3072
    max_position_embeddings: int = 4096
    description: str = ""
    # LoRA adapter config (applied during bridge fine-tuning, Faz 3+)
    lora_r: int = 8
    lora_alpha: int = 16
    lora_target_modules: List[str] = field(
        default_factory=lambda: ["q_proj", "v_proj", "k_proj", "o_proj"],
    )
    # Frozen flag: True = expert weights are frozen during bridge/router training
    frozen: bool = True

    # ── Derived / shared (read-only at construction) ──────────────────────────
    hidden_size: int = SHARED_HIDDEN_SIZE
    vocab_size: int = SHARED_VOCAB_SIZE
    num_attention_heads: int = SHARED_NUM_ATTENTION_HEADS
    num_key_value_heads: int = SHARED_NUM_KV_HEADS

    def estimated_params_m(self) -> float:
        """Rough parameter estimate in millions (decoder-only transformer)."""
        d = self.hidden_size
        ff = self.intermediate_size
        n = self.num_hidden_layers
        v = self.vocab_size
        # Attention: 4 * d^2 per layer (q, k, v, o projections — simplified)
        attn = 4 * d * d * n
        # SwiGLU MLP: 3 * d * ff per layer (gate, up, down)
        mlp = 3 * d * ff * n
        # Embeddings (tied): v * d
        emb = v * d
        # Norms: negligible
        total = attn + mlp + emb
        return total / 1e6

    def to_codemind_kwargs(self) -> Dict[str, Any]:
        """Convert to kwargs compatible with CodeMindConfig constructor."""
        return {
            "vocab_size": self.vocab_size,
            "hidden_size": self.hidden_size,
            "num_hidden_layers": self.num_hidden_layers,
            "num_attention_heads": self.num_attention_heads,
            "num_key_value_heads": self.num_key_value_heads,
            "intermediate_size": self.intermediate_size,
            "max_position_embeddings": self.max_position_embeddings,
            # Single-expert mode: no MoE inside the expert itself
            "num_experts": 1,
            "num_experts_per_tok": 1,
            "use_mod": False,
        }


# ── Pre-defined Expert Configurations ─────────────────────────────────────────

EXPERT_CONFIGS: Dict[ExpertDomain, ExpertConfig] = {
    ExpertDomain.TURKISH: ExpertConfig(
        domain=ExpertDomain.TURKISH,
        num_hidden_layers=12,
        intermediate_size=3072,
        description="Turkish-English bilingual language expert (~76M params)",
    ),
    ExpertDomain.CODE: ExpertConfig(
        domain=ExpertDomain.CODE,
        num_hidden_layers=16,
        intermediate_size=3072,
        description="Python code generation expert (~100M params)",
    ),
    ExpertDomain.MATH: ExpertConfig(
        domain=ExpertDomain.MATH,
        num_hidden_layers=12,
        intermediate_size=2048,
        description="Mathematics and formal reasoning expert (~56M params)",
    ),
    ExpertDomain.LOGIC: ExpertConfig(
        domain=ExpertDomain.LOGIC,
        num_hidden_layers=12,
        intermediate_size=2048,
        description="Logical reasoning and analysis expert (~56M params)",
    ),
    ExpertDomain.GENERAL: ExpertConfig(
        domain=ExpertDomain.GENERAL,
        num_hidden_layers=12,
        intermediate_size=3072,
        description="General knowledge fallback expert (~76M params)",
    ),
}
