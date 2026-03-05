"""
Speculative Decoding for CodeMind.

Algorithm (Leviathan et al., 2023):
  1. Draft model generates K candidate tokens auto-regressively (fast).
  2. Target model evaluates ALL K tokens in one parallel forward pass (costly but parallelizable).
  3. Tokens are accepted/rejected via rejection sampling; rejected tokens are re-sampled from target.

Result: ~2-4x wall-clock speedup at no quality cost. Ideal when:
  - Draft model  : CodeMind-125M (tiny, sits permanently in VRAM)
  - Target model : CodeMind-350M or 350M-MoE (larger, slower)
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from typing import Optional, List

from src.utils.logger import get_logger

logger = get_logger("speculative_decoding")


class SpeculativeDecoder:
    """Runs speculative decoding with a draft + target model pair.

    Usage:
        decoder = SpeculativeDecoder(
            draft_model=model_manager_125m.model,
            target_model=model_manager_350m.model,
            tokenizer=tokenizer,
            draft_steps=5,        # K: how many tokens to draft each round
        )
        output_ids = decoder.generate(input_ids, max_new_tokens=200)
    """

    def __init__(
        self,
        draft_model: torch.nn.Module,
        target_model: torch.nn.Module,
        tokenizer,
        draft_steps: int = 5,
        temperature: float = 1.0,
        device: Optional[str] = None,
    ) -> None:
        self.draft_model  = draft_model
        self.target_model = target_model
        self.tokenizer    = tokenizer
        # K: tokens to speculatively generate per round
        self.draft_steps  = draft_steps
        self.temperature  = temperature
        self.device       = device or ("cuda" if torch.cuda.is_available() else "cpu")

        # Move both models to device
        self.draft_model.to(self.device).eval()
        self.target_model.to(self.device).eval()

    @torch.inference_mode()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 100,
        eos_token_id: Optional[int] = None,
    ) -> torch.Tensor:
        """Generate tokens using speculative decoding.

        Returns tensor of shape [1, input_len + generated_len].
        """
        eos = eos_token_id or getattr(self.tokenizer, "eos_token_id", None)
        generated = input_ids.clone().to(self.device)  # [1, S]
        total_new = 0

        while total_new < max_new_tokens:
            remaining = max_new_tokens - total_new
            k = min(self.draft_steps, remaining)

            # ── Step 1: Draft model generates K candidate tokens ──────────────
            draft_tokens: List[int] = []
            draft_probs:  List[torch.Tensor] = []
            draft_input   = generated.clone()

            for _ in range(k):
                with torch.no_grad():
                    draft_out = self.draft_model(draft_input)
                    # Support both plain and CausalLM output formats
                    logits = draft_out.logits if hasattr(draft_out, "logits") else draft_out[0]
                    next_token_logits = logits[:, -1, :] / max(self.temperature, 1e-8)
                    probs = F.softmax(next_token_logits, dim=-1)
                    next_token = torch.multinomial(probs, num_samples=1)  # [1, 1]
                    draft_tokens.append(next_token.item())
                    draft_probs.append(probs[0])  # [V]
                    draft_input = torch.cat([draft_input, next_token], dim=-1)

            # ── Step 2: Target model evaluates draft + prompt in one pass ─────
            target_input = torch.cat(
                [generated, torch.tensor(draft_tokens, device=self.device).unsqueeze(0)],
                dim=-1,
            )  # [1, S + K]

            with torch.no_grad():
                target_out = self.target_model(target_input)
                target_logits = target_out.logits if hasattr(target_out, "logits") else target_out[0]

            # Target probabilities at each of the K draft token positions
            seq_start = generated.shape[1] - 1  # index of last context token
            target_probs_list = []
            for i in range(k):
                lgt = target_logits[:, seq_start + i, :] / max(self.temperature, 1e-8)
                target_probs_list.append(F.softmax(lgt, dim=-1)[0])  # [V]

            # ── Step 3: Rejection sampling ────────────────────────────────────
            accepted_tokens: List[int] = []
            for i, token_id in enumerate(draft_tokens):
                p_target = target_probs_list[i][token_id].clamp(min=1e-9)
                p_draft  = draft_probs[i][token_id].clamp(min=1e-9)
                acceptance_ratio = (p_target / p_draft).clamp(max=1.0)

                if torch.rand(1, device=self.device).item() < acceptance_ratio.item():
                    # Accept draft token
                    accepted_tokens.append(token_id)
                    if eos and token_id == eos:
                        break
                else:
                    # Reject: sample corrected token from residual distribution
                    corrected = F.relu(target_probs_list[i] - draft_probs[i])
                    corrected_sum = corrected.sum()
                    if corrected_sum > 1e-9:
                        corrected = corrected / corrected_sum
                        replacement = torch.multinomial(corrected, 1).item()
                    else:
                        replacement = torch.multinomial(target_probs_list[i], 1).item()
                    accepted_tokens.append(replacement)
                    break  # stop after first rejection

            if not accepted_tokens:
                # Safety: sample one token directly from target
                lgt = target_logits[:, generated.shape[1] - 1, :] / max(self.temperature, 1e-8)
                p = F.softmax(lgt, dim=-1)
                fallback = torch.multinomial(p, 1)
                accepted_tokens = [fallback.item()]

            # Append accepted tokens to the sequence
            accepted_tensor = torch.tensor(accepted_tokens, device=self.device).unsqueeze(0)
            generated = torch.cat([generated, accepted_tensor], dim=-1)
            total_new += len(accepted_tokens)

            logger.debug(
                "Speculative step: drafted=%d accepted=%d  (%.0f%%)",
                k,
                len(accepted_tokens),
                100 * len(accepted_tokens) / k,
            )

            # Early stop on EOS
            if eos and accepted_tokens[-1] == eos:
                break

        return generated
