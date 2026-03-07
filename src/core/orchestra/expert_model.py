"""
Expert Model Wrapper — loads and manages individual CodeMind expert instances.

Each ExpertModel wraps a CodeMindForCausalLM instance with:
  - LoRA adapter support (for bridge fine-tuning)
  - Hidden state extraction (for latent bridge communication)
  - Conditioned generation (can accept bridge output as prefix)
  - Freeze/unfreeze controls
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
from peft import LoraConfig, get_peft_model, PeftModel

from src.core.model.codemind import CodeMindConfig, CodeMindForCausalLM
from src.core.orchestra.expert_config import (
    ExpertConfig,
    ExpertDomain,
    EXPERT_CONFIGS,
    SHARED_HIDDEN_SIZE,
)

logger = logging.getLogger(__name__)


class ExpertModel(nn.Module):
    """Wrapper around a single CodeMind expert model.

    Provides latent-communication capabilities on top of the base model:
      - extract_hidden_states(): get intermediate representations
      - generate_conditioned(): generate text with bridge-injected context
      - freeze() / apply_lora(): control training behavior

    Args:
        config: ExpertConfig specifying domain and architecture.
        device: Target device (cuda / cpu).
    """

    def __init__(
        self,
        config: ExpertConfig,
        device: Optional[str] = None,
    ) -> None:
        super().__init__()
        self.expert_config = config
        self.domain = config.domain
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._is_frozen = False
        self._has_lora = False

        # Build fresh CodeMind model from expert config
        cm_config = CodeMindConfig(**config.to_codemind_kwargs())
        self.model = CodeMindForCausalLM(cm_config)
        self.model.to(self.device)

        logger.info(
            "ExpertModel[%s] created: %s, ~%.1fM params",
            config.domain.value,
            config.description,
            config.estimated_params_m(),
        )

    @classmethod
    def from_checkpoint(
        cls,
        domain: ExpertDomain,
        checkpoint_path: Union[str, Path],
        device: Optional[str] = None,
    ) -> "ExpertModel":
        """Load an expert from a saved checkpoint.

        Args:
            domain: Expert domain identifier.
            checkpoint_path: Path to .pt checkpoint file or HF-style directory.
            device: Target device.

        Returns:
            ExpertModel with loaded weights.
        """
        config = EXPERT_CONFIGS.get(domain)
        if config is None:
            raise ValueError(f"Unknown expert domain: {domain}")

        expert = cls(config=config, device=device)
        checkpoint_path = Path(checkpoint_path)

        if checkpoint_path.is_file() and checkpoint_path.suffix == ".pt":
            # Custom CodeMind .pt format
            checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
            state_dict = checkpoint.get("model_state_dict", checkpoint)
            expert.model.load_state_dict(state_dict, strict=False)
            logger.info("Expert[%s] loaded from %s", domain.value, checkpoint_path)
        elif checkpoint_path.is_dir():
            # HuggingFace-style directory with config.json + model.safetensors
            from safetensors.torch import load_file
            safetensors_path = checkpoint_path / "model.safetensors"
            if safetensors_path.exists():
                state_dict = load_file(str(safetensors_path))
                expert.model.load_state_dict(state_dict, strict=False)
            else:
                pt_path = checkpoint_path / "pytorch_model.bin"
                if pt_path.exists():
                    state_dict = torch.load(pt_path, map_location="cpu", weights_only=False)
                    expert.model.load_state_dict(state_dict, strict=False)
                else:
                    raise FileNotFoundError(
                        f"No model weights found in {checkpoint_path}"
                    )
            logger.info("Expert[%s] loaded from %s", domain.value, checkpoint_path)
        else:
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        expert.model.to(expert.device)
        return expert

    # ── Freeze / LoRA ──────────────────────────────────────────────────────────

    def freeze(self) -> None:
        """Freeze all expert parameters (for bridge/router training)."""
        for param in self.model.parameters():
            param.requires_grad = False
        self._is_frozen = True
        logger.info("Expert[%s] frozen.", self.domain.value)

    def unfreeze(self) -> None:
        """Unfreeze all expert parameters."""
        for param in self.model.parameters():
            param.requires_grad = True
        self._is_frozen = False
        logger.info("Expert[%s] unfrozen.", self.domain.value)

    def apply_lora(
        self,
        r: Optional[int] = None,
        alpha: Optional[int] = None,
        target_modules: Optional[List[str]] = None,
    ) -> None:
        """Apply LoRA adapters to the expert (for bridge fine-tuning).

        LoRA allows minimal parameter updates on frozen experts, enabling
        the expert to adapt to the bridge's communication protocol without
        risking catastrophic forgetting.

        Args:
            r: LoRA rank. Default from expert config.
            alpha: LoRA alpha scaling. Default from expert config.
            target_modules: Modules to apply LoRA to. Default from config.
        """
        if self._has_lora:
            logger.warning("LoRA already applied to Expert[%s].", self.domain.value)
            return

        cfg = self.expert_config
        lora_config = LoraConfig(
            r=r or cfg.lora_r,
            lora_alpha=alpha or cfg.lora_alpha,
            target_modules=target_modules or cfg.lora_target_modules,
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM",
        )
        self.model = get_peft_model(self.model, lora_config)
        self._has_lora = True

        trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.model.parameters())
        logger.info(
            "Expert[%s] LoRA applied: trainable=%d (%.2f%% of %d)",
            self.domain.value, trainable, 100 * trainable / total, total,
        )

    # ── Hidden State Extraction ────────────────────────────────────────────────

    @torch.inference_mode()
    def extract_hidden_states(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        layer_index: int = -1,
    ) -> torch.Tensor:
        """Extract hidden states from a specific layer.

        This is the core function for latent bridge communication.
        By default, extracts from the final layer (before the LM head).

        Args:
            input_ids: [batch, seq_len] token IDs.
            attention_mask: [batch, seq_len] padding mask.
            layer_index: Which layer's output to extract. -1 = final layer.

        Returns:
            hidden_states: [batch, seq_len, hidden_size]
        """
        input_ids = input_ids.to(self.device)
        if attention_mask is not None:
            attention_mask = attention_mask.to(self.device)

        # Access the base model (unwrap PEFT if needed)
        base_model = self.model
        if isinstance(base_model, PeftModel):
            base_model = base_model.base_model.model  # PeftModel → base → CodeMindForCausalLM

        # Get the transformer backbone
        backbone = base_model.model  # CodeMindForCausalLM.model → CodeMindModel

        # Get embeddings
        hidden_states = backbone.word_embeddings(input_ids)

        # Build attention mask
        batch_size, seq_len = input_ids.shape
        past_length = 0
        extended_mask = backbone._build_combined_attention_mask(
            attention_mask=attention_mask,
            batch_size=batch_size,
            seq_len=seq_len,
            past_length=past_length,
            dtype=hidden_states.dtype,
            device=hidden_states.device,
        )

        # Run through layers up to layer_index
        num_layers = len(backbone.layers)
        target_layer = layer_index if layer_index >= 0 else num_layers + layer_index

        for i, layer in enumerate(backbone.layers):
            layer_out = layer(
                hidden_states,
                attention_mask=extended_mask,
                past_key_value=None,
                use_cache=False,
                position_ids=None,
                output_attentions=False,
            )
            hidden_states = layer_out[0]

            if i == target_layer:
                break

        # Apply final layer norm
        hidden_states = backbone.final_layernorm(hidden_states)
        return hidden_states

    # ── Conditioned Generation ────────────────────────────────────────────────

    @torch.inference_mode()
    def generate(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        max_new_tokens: int = 256,
        temperature: float = 0.7,
        top_k: int = 50,
        top_p: float = 0.95,
        do_sample: bool = True,
    ) -> torch.Tensor:
        """Generate tokens from this expert.

        Args:
            input_ids: [batch, seq_len] prompt token IDs.
            attention_mask: [batch, seq_len] padding mask.
            max_new_tokens: Maximum tokens to generate.
            temperature: Sampling temperature.
            top_k: Top-K sampling parameter.
            top_p: Top-P (nucleus) sampling parameter.
            do_sample: Whether to sample or greedily decode.

        Returns:
            generated_ids: [batch, seq_len + new_tokens] full sequence.
        """
        input_ids = input_ids.to(self.device)
        if attention_mask is not None:
            attention_mask = attention_mask.to(self.device)

        return self.model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            do_sample=do_sample,
            pad_token_id=0,
        )

    @torch.inference_mode()
    def generate_conditioned(
        self,
        input_ids: torch.Tensor,
        bridge_hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        max_new_tokens: int = 256,
        temperature: float = 0.7,
        top_k: int = 50,
        top_p: float = 0.95,
    ) -> torch.Tensor:
        """Generate tokens conditioned on bridge output from another expert.

        The bridge hidden states are prepended to the input embeddings as a
        "soft prefix", allowing the expert to condition its generation on
        the latent context from another expert.

        This is conceptually similar to prefix-tuning, but the prefix comes
        from a dynamic bridge rather than learnable vectors.

        Args:
            input_ids: [batch, seq_len] the expert's own prompt tokens.
            bridge_hidden_states: [batch, bridge_seq_len, hidden_size]
                projected hidden states from another expert via the bridge.
            attention_mask: [batch, seq_len] padding mask for input_ids.
            max_new_tokens: Max tokens to generate.
            temperature: Sampling temperature.
            top_k: Top-K sampling.
            top_p: Top-P sampling.

        Returns:
            generated_ids: [batch, total_seq_len + new_tokens]
        """
        input_ids = input_ids.to(self.device)
        bridge_hidden_states = bridge_hidden_states.to(self.device)
        if attention_mask is not None:
            attention_mask = attention_mask.to(self.device)

        # Get input embeddings from the expert
        base_model = self.model
        if isinstance(base_model, PeftModel):
            base_model = base_model.base_model.model

        input_embeds = base_model.model.word_embeddings(input_ids)

        # Prepend bridge hidden states as soft prefix
        # [batch, bridge_len + seq_len, hidden_size]
        combined_embeds = torch.cat([bridge_hidden_states, input_embeds], dim=1)

        # Build combined attention mask
        bridge_seq_len = bridge_hidden_states.shape[1]
        bridge_mask = torch.ones(
            input_ids.shape[0], bridge_seq_len,
            device=self.device, dtype=torch.long,
        )
        if attention_mask is not None:
            combined_mask = torch.cat([bridge_mask, attention_mask], dim=1)
        else:
            combined_mask = torch.cat(
                [bridge_mask, torch.ones_like(input_ids)], dim=1,
            )

        # Generate using inputs_embeds instead of input_ids
        return self.model.generate(
            inputs_embeds=combined_embeds,
            attention_mask=combined_mask,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            do_sample=True,
            pad_token_id=0,
        )

    # ── Utilities ──────────────────────────────────────────────────────────────

    @property
    def num_parameters(self) -> int:
        """Total parameters."""
        return sum(p.numel() for p in self.model.parameters())

    @property
    def num_trainable_parameters(self) -> int:
        """Trainable parameters (considers frozen/LoRA state)."""
        return sum(p.numel() for p in self.model.parameters() if p.requires_grad)

    def save_checkpoint(self, path: Union[str, Path]) -> None:
        """Save expert checkpoint.

        If LoRA is applied, saves only the adapter weights.
        Otherwise saves the full model state dict.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        if self._has_lora and isinstance(self.model, PeftModel):
            # Save only LoRA adapter
            adapter_path = path.parent / f"{path.stem}_lora"
            self.model.save_pretrained(str(adapter_path))
            logger.info("Expert[%s] LoRA adapter saved to %s", self.domain.value, adapter_path)
        else:
            # Save full model
            state_dict = self.model.state_dict()
            torch.save({"model_state_dict": state_dict}, str(path))
            logger.info("Expert[%s] checkpoint saved to %s", self.domain.value, path)
