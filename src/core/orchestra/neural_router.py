"""
Neural Router — lightweight transformer-based expert selector.

Analyzes the user prompt and decides:
  1. Which expert(s) to activate
  2. Expert weights (soft gating)
  3. Execution mode: SINGLE or CASCADE

Architecture (~2.5M params):
  - 2-layer Transformer Encoder (d_model=256, nhead=4)
  - Mean-pool over sequence → expert logits via Linear head
  - Softmax gating for expert weights
  - Mode classifier head for execution mode selection

The router is trained in two phases:
  1. Supervised: prompt → expert label pairs (Faz 4)
  2. End-to-end: fine-tuned with orchestration loss (Faz 5)
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.core.orchestra.expert_config import (
    ExpertDomain,
    SHARED_HIDDEN_SIZE,
    SHARED_VOCAB_SIZE,
)


class ExecutionMode(str, Enum):
    """How the orchestra should execute the expert pipeline."""
    SINGLE = "single"       # One expert handles the entire query
    CASCADE = "cascade"     # Sequential: Expert A → Bridge → Expert B


@dataclass
class RoutingDecision:
    """Output of the neural router.

    Attributes:
        expert_weights: Dict mapping ExpertDomain → weight (0-1, sum=1).
        primary_expert: The expert with the highest weight.
        secondary_expert: Second-highest expert (for cascade mode).
        execution_mode: SINGLE or CASCADE.
        confidence: Router's confidence in the primary expert choice.
        raw_logits: Raw router output before softmax (for loss computation).
    """
    expert_weights: Dict[ExpertDomain, float]
    primary_expert: ExpertDomain
    secondary_expert: Optional[ExpertDomain]
    execution_mode: ExecutionMode
    confidence: float
    raw_logits: Optional[torch.Tensor] = None


class NeuralRouter(nn.Module):
    """Lightweight transformer-based expert router.

    Takes tokenized prompt, encodes it with a small transformer encoder,
    then classifies which expert(s) should handle the query.

    Args:
        vocab_size: Vocabulary size (shared with all experts).
        d_model: Router's internal hidden dimension (not shared_hidden_size —
            the router is intentionally much smaller).
        nhead: Number of attention heads in the router encoder.
        num_encoder_layers: Depth of the router encoder.
        num_experts: Number of expert domains to route to.
        expert_domains: List of ExpertDomain values in routing order.
        max_seq_len: Maximum input sequence length for the router.
        cascade_threshold: If top-2 expert weight gap < this, use CASCADE mode.
        dropout: Dropout rate.
    """

    def __init__(
        self,
        vocab_size: int = SHARED_VOCAB_SIZE,
        d_model: int = 256,
        nhead: int = 4,
        num_encoder_layers: int = 2,
        num_experts: int = 2,
        expert_domains: Optional[List[ExpertDomain]] = None,
        max_seq_len: int = 512,
        cascade_threshold: float = 0.3,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.num_experts = num_experts
        self.max_seq_len = max_seq_len
        self.cascade_threshold = cascade_threshold

        # Default domains for MVP: Turkish + Code
        self.expert_domains = expert_domains or [
            ExpertDomain.TURKISH,
            ExpertDomain.CODE,
        ]
        assert len(self.expert_domains) == num_experts, (
            f"expert_domains length ({len(self.expert_domains)}) must match "
            f"num_experts ({num_experts})"
        )

        # Build domain-to-index mapping
        self._domain_to_idx: Dict[ExpertDomain, int] = {
            domain: i for i, domain in enumerate(self.expert_domains)
        }

        # ── Embedding ─────────────────────────────────────────────────────────
        # Small embedding projecting shared vocab into router's d_model space
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.position_embedding = nn.Embedding(max_seq_len, d_model)
        self.embed_dropout = nn.Dropout(dropout)
        self.embed_scale = math.sqrt(d_model)

        # ── Transformer Encoder ───────────────────────────────────────────────
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,  # Pre-norm for stability
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_encoder_layers,
        )
        self.encoder_norm = nn.LayerNorm(d_model)

        # ── Classification Heads ──────────────────────────────────────────────
        # Expert selection head: d_model → num_experts
        self.expert_head = nn.Linear(d_model, num_experts)

        # Execution mode head: d_model → 2 (SINGLE, CASCADE)
        self.mode_head = nn.Linear(d_model, 2)

        self._init_weights()

    def _init_weights(self) -> None:
        """Initialize router weights."""
        nn.init.normal_(self.token_embedding.weight, std=0.02)
        nn.init.normal_(self.position_embedding.weight, std=0.02)
        nn.init.xavier_uniform_(self.expert_head.weight, gain=0.1)
        nn.init.zeros_(self.expert_head.bias)
        nn.init.xavier_uniform_(self.mode_head.weight, gain=0.1)
        nn.init.zeros_(self.mode_head.bias)

    def _encode(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Encode input tokens → pooled representation.

        Args:
            input_ids: [batch, seq_len] token IDs.
            attention_mask: [batch, seq_len] — 1 for real, 0 for padding.

        Returns:
            [batch, d_model] mean-pooled encoder output.
        """
        batch_size, seq_len = input_ids.shape
        seq_len = min(seq_len, self.max_seq_len)
        input_ids = input_ids[:, :seq_len]
        if attention_mask is not None:
            attention_mask = attention_mask[:, :seq_len]

        # Token + position embeddings
        positions = torch.arange(seq_len, device=input_ids.device).unsqueeze(0)
        h = self.token_embedding(input_ids) * self.embed_scale
        h = h + self.position_embedding(positions)
        h = self.embed_dropout(h)

        # Create src_key_padding_mask for transformer: True = ignore
        src_key_padding_mask = None
        if attention_mask is not None:
            src_key_padding_mask = attention_mask == 0  # True where padded

        # Encode
        h = self.encoder(h, src_key_padding_mask=src_key_padding_mask)
        h = self.encoder_norm(h)

        # Mean pooling (respecting mask)
        if attention_mask is not None:
            mask_expanded = attention_mask.unsqueeze(-1).to(h.dtype)
            h = (h * mask_expanded).sum(dim=1) / mask_expanded.sum(dim=1).clamp(min=1e-9)
        else:
            h = h.mean(dim=1)

        return h  # [batch, d_model]

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute expert logits and mode logits.

        Used during training for loss computation.

        Args:
            input_ids: [batch, seq_len]
            attention_mask: [batch, seq_len]

        Returns:
            expert_logits: [batch, num_experts] — raw logits for expert selection
            mode_logits: [batch, 2] — raw logits for mode selection
        """
        pooled = self._encode(input_ids, attention_mask)
        expert_logits = self.expert_head(pooled)
        mode_logits = self.mode_head(pooled)
        return expert_logits, mode_logits

    @torch.inference_mode()
    def route(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> RoutingDecision:
        """Make a routing decision for inference.

        Args:
            input_ids: [1, seq_len] — single prompt (batch=1).
            attention_mask: [1, seq_len]

        Returns:
            RoutingDecision with expert weights and execution mode.
        """
        expert_logits, mode_logits = self.forward(input_ids, attention_mask)

        # Expert weights via softmax
        expert_weights_tensor = F.softmax(expert_logits[0], dim=-1)  # [num_experts]
        weights_dict: Dict[ExpertDomain, float] = {}
        for i, domain in enumerate(self.expert_domains):
            weights_dict[domain] = expert_weights_tensor[i].item()

        # Sort by weight descending
        sorted_domains = sorted(weights_dict.keys(), key=lambda d: weights_dict[d], reverse=True)
        primary = sorted_domains[0]
        secondary = sorted_domains[1] if len(sorted_domains) > 1 else None
        confidence = weights_dict[primary]

        # Execution mode
        mode_probs = F.softmax(mode_logits[0], dim=-1)
        if mode_probs[1] > mode_probs[0]:
            execution_mode = ExecutionMode.CASCADE
        else:
            execution_mode = ExecutionMode.SINGLE

        # Override: if the gap between top-2 is small, prefer CASCADE
        if secondary is not None:
            gap = weights_dict[primary] - weights_dict[secondary]
            if gap < self.cascade_threshold:
                execution_mode = ExecutionMode.CASCADE

        return RoutingDecision(
            expert_weights=weights_dict,
            primary_expert=primary,
            secondary_expert=secondary,
            execution_mode=execution_mode,
            confidence=confidence,
            raw_logits=expert_logits,
        )

    @property
    def num_parameters(self) -> int:
        """Total parameters in the router."""
        return sum(p.numel() for p in self.parameters())

    def domain_to_idx(self, domain: ExpertDomain) -> int:
        """Convert ExpertDomain to integer index for this router."""
        return self._domain_to_idx[domain]


class KeywordRouterFallback:
    """Rule-based fallback router for when the neural router is not loaded.

    Uses simple keyword matching to determine the primary expert domain.
    This is used as a baseline comparison and emergency fallback.
    """

    _KEYWORDS: Dict[ExpertDomain, List[str]] = {
        ExpertDomain.TURKISH: [
            "merhaba", "nasılsın", "teşekkür", "türkçe", "sohbet", "anlat",
            "açıkla", "ne demek", "nedir", "nasıl", "neden", "niye",
            "lütfen", "tamam", "evet", "hayır", "günaydın", "iyi geceler",
        ],
        ExpertDomain.CODE: [
            "python", "code", "function", "class", "def ", "import",
            "pip", "error", "bug", "debug", "algorithm", "print(",
            "for ", "while ", "if ", "return", "list", "dict",
            "fonksiyon", "kod yaz", "program", "yazılım",
        ],
        ExpertDomain.MATH: [
            "matematik", "hesapla", "integral", "türev", "matris",
            "denklem", "toplam", "çarpım", "formül", "istatistik",
            "calculate", "equation", "sum", "multiply", "derivative",
        ],
        ExpertDomain.LOGIC: [
            "mantık", "kanıtla", "ispat", "çıkarım", "çelişki",
            "doğru mu", "yanlış mı", "logic", "prove", "theorem",
            "reasoning", "deduce", "infer",
        ],
    }

    def route(self, query: str) -> ExpertDomain:
        """Keyword-based routing fallback."""
        query_lower = query.lower()
        scores: Dict[ExpertDomain, int] = {d: 0 for d in ExpertDomain}
        for domain, keywords in self._KEYWORDS.items():
            for kw in keywords:
                if kw in query_lower:
                    scores[domain] += 1

        best = max(scores, key=lambda d: scores[d])
        if scores[best] == 0:
            return ExpertDomain.TURKISH  # Default fallback
        return best
