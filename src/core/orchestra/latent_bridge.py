"""
Latent Bridge — MLP-based inter-expert communication network.

This module implements the core innovation of the Latent Expert Orchestra:
instead of text-to-text relay between experts, hidden states are projected
through a learned non-linear bridge that aligns the latent representations
of different experts into a shared communication space.

Architecture (MVP — 2-layer MLP bridge):

    h_source ∈ ℝ^{B × S × D}
        ↓
    LayerNorm
        ↓
    Linear(D → D_bottleneck)
        ↓
    GELU
        ↓
    Linear(D_bottleneck → D)
        ↓
    LayerNorm
        ↓
    h_target ∈ ℝ^{B × S × D}

The bridge is a lightweight adapter (~1.5M params for D=768, D_bottleneck=512)
that learns to map the output hidden states of Expert A into a form that
Expert B can consume as conditioning context.

Key design decisions:
  - Shared hidden_size across all experts eliminates the need for
    dimensional projection, keeping the bridge simple.
  - Bottleneck (D → D_bottleneck → D) acts as information compression,
    preventing the bridge from simply memorizing identity mapping.
  - LayerNorm on input/output stabilizes training with frozen experts.
  - Residual connection is optional — disabled by default to force the
    bridge to learn meaningful transformations.
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn

from src.core.orchestra.expert_config import (
    ExpertDomain,
    SHARED_HIDDEN_SIZE,
)


class LatentBridge(nn.Module):
    """MLP bridge for latent-level inter-expert communication.

    Transforms hidden states from one expert's representation space into
    a form consumable by another expert.

    Args:
        hidden_size: Model hidden dimension (must match all experts). Default 768.
        bottleneck_size: Internal bottleneck dimension. Controls information
            bandwidth. Smaller = more compression, less overfitting risk.
        dropout: Dropout probability applied after activation.
        use_residual: If True, adds a residual connection (h_out = bridge(h) + h).
            Disabled by default to force the bridge to learn transformations.
    """

    def __init__(
        self,
        hidden_size: int = SHARED_HIDDEN_SIZE,
        bottleneck_size: int = 512,
        dropout: float = 0.1,
        use_residual: bool = False,
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.bottleneck_size = bottleneck_size
        self.use_residual = use_residual

        # Input normalization — stabilizes frozen expert outputs
        self.input_norm = nn.LayerNorm(hidden_size)

        # Down-project → activate → up-project
        self.down_proj = nn.Linear(hidden_size, bottleneck_size, bias=True)
        self.activation = nn.GELU()
        self.up_proj = nn.Linear(bottleneck_size, hidden_size, bias=True)

        # Output normalization — stabilizes bridge output for target expert
        self.output_norm = nn.LayerNorm(hidden_size)

        self.dropout = nn.Dropout(dropout)

        # Initialize with small weights so bridge starts near identity
        self._init_weights()

    def _init_weights(self) -> None:
        """Initialize bridge weights for stable training startup."""
        nn.init.xavier_uniform_(self.down_proj.weight, gain=0.1)
        nn.init.zeros_(self.down_proj.bias)
        nn.init.xavier_uniform_(self.up_proj.weight, gain=0.1)
        nn.init.zeros_(self.up_proj.bias)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Project hidden states through the latent bridge.

        Args:
            hidden_states: Source expert output [batch, seq_len, hidden_size].
            attention_mask: Optional [batch, seq_len] mask. Masked positions
                are zeroed out after projection to prevent information leakage.

        Returns:
            Projected hidden states [batch, seq_len, hidden_size] ready for
            consumption by the target expert.
        """
        residual = hidden_states

        # Normalize → project down → activate → project up → normalize
        h = self.input_norm(hidden_states)
        h = self.down_proj(h)
        h = self.activation(h)
        h = self.dropout(h)
        h = self.up_proj(h)
        h = self.output_norm(h)

        if self.use_residual:
            h = h + residual

        # Zero out masked positions if mask is provided
        if attention_mask is not None:
            # attention_mask: [B, S] — 1 for real, 0 for padding
            h = h * attention_mask.unsqueeze(-1).to(h.dtype)

        return h

    @property
    def num_parameters(self) -> int:
        """Total trainable parameters in the bridge."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class BridgePool(nn.Module):
    """Manages bridges for all expert-pair communications.

    Instead of N² bridges (one per expert pair), we use a shared bridge
    with domain-specific bias vectors for efficiency. This keeps the
    parameter count tractable as expert count grows.

    For the MVP with 2 experts, a single shared bridge is sufficient.
    For 3+ experts, domain-conditioned adaptation activates.

    Architecture:
        shared_bridge: LatentBridge (main transformation)
        domain_biases: nn.Embedding(num_domains, hidden_size) — adds
            source-domain-specific bias before the bridge to help it
            distinguish which expert's representation it's processing.
    """

    def __init__(
        self,
        num_domains: int = 2,
        hidden_size: int = SHARED_HIDDEN_SIZE,
        bottleneck_size: int = 512,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.num_domains = num_domains
        self.hidden_size = hidden_size

        # Shared bridge — handles the core transformation
        self.bridge = LatentBridge(
            hidden_size=hidden_size,
            bottleneck_size=bottleneck_size,
            dropout=dropout,
            use_residual=False,
        )

        # Domain-specific bias: tells the bridge which expert's space
        # the input comes from. Cheap conditioning mechanism.
        self.source_domain_embed = nn.Embedding(num_domains, hidden_size)
        self.target_domain_embed = nn.Embedding(num_domains, hidden_size)

        # Scale down domain embeddings to avoid dominating the hidden states
        nn.init.normal_(self.source_domain_embed.weight, std=0.01)
        nn.init.normal_(self.target_domain_embed.weight, std=0.01)

    def forward(
        self,
        hidden_states: torch.Tensor,
        source_domain_id: int,
        target_domain_id: int,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Project hidden states from source expert to target expert space.

        Args:
            hidden_states: [batch, seq_len, hidden_size] from source expert.
            source_domain_id: Integer ID of the source expert domain.
            target_domain_id: Integer ID of the target expert domain.
            attention_mask: Optional [batch, seq_len] padding mask.

        Returns:
            [batch, seq_len, hidden_size] projected for target expert.
        """
        batch_size = hidden_states.shape[0]

        # Add source domain conditioning
        source_bias = self.source_domain_embed(
            torch.tensor(source_domain_id, device=hidden_states.device)
        )  # [hidden_size]
        target_bias = self.target_domain_embed(
            torch.tensor(target_domain_id, device=hidden_states.device)
        )  # [hidden_size]

        # Condition the hidden states with source/target domain information
        conditioned = hidden_states + source_bias.unsqueeze(0).unsqueeze(0) + target_bias.unsqueeze(0).unsqueeze(0)

        # Apply the shared bridge
        return self.bridge(conditioned, attention_mask)

    @property
    def num_parameters(self) -> int:
        """Total trainable parameters in the bridge pool."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
