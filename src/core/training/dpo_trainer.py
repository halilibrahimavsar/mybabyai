"""
Direct Preference Optimization (DPO) Trainer for CodeMind.

DPO (Rafailov et al., 2023) is a simpler, more stable alternative to RLHF:
- No reward model training needed
- No RL policy gradient — just a single classification-style loss
- Loss: maximize log(chosen/rejected) relative to a frozen reference model

Integration with NightShift:
  High-quality System 2 (MCTS) responses can be automatically collected as
  "chosen" examples, while rejected System 1 drafts serve as "rejected".
  This creates a self-improving feedback loop without human labelers.

Input data format:
    [{"prompt": "...", "chosen": "...", "rejected": "..."}, ...]
"""

from __future__ import annotations

import copy
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW

from src.utils.logger import get_logger

logger = get_logger("dpo_trainer")


# ─────────────────────────────────────────────────────────────────────────────
# Dataset
# ─────────────────────────────────────────────────────────────────────────────

class DPODataset(Dataset):
    """Tokenizes preference triplets: (prompt, chosen, rejected)."""

    def __init__(
        self,
        data: List[Dict[str, str]],
        tokenizer,
        max_length: int = 512,
    ) -> None:
        self.tokenizer  = tokenizer
        self.max_length = max_length
        self.samples    = self._process(data)

    def _encode(self, text: str) -> List[int]:
        out = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            padding=False,
            return_tensors=None,
        )
        ids = out.get("input_ids", [])
        return list(ids[0] if ids and isinstance(ids[0], list) else ids)

    def _process(self, data: List[Dict[str, str]]) -> List[Dict[str, List[int]]]:
        samples = []
        for item in data:
            prompt  = item.get("prompt", "")
            chosen  = item.get("chosen", "")
            rejected = item.get("rejected", "")
            if not (prompt and chosen and rejected):
                continue
            samples.append({
                "chosen_ids":   self._encode(prompt + chosen),
                "rejected_ids": self._encode(prompt + rejected),
            })
        logger.info("DPO dataset ready: %d samples", len(samples))
        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, List[int]]:
        return self.samples[idx]


def _dpo_collate(batch: List[Dict[str, List[int]]]) -> Dict[str, torch.Tensor]:
    """Pad chosen and rejected sequences to same length within batch."""
    chosen_ids   = [torch.tensor(b["chosen_ids"],   dtype=torch.long) for b in batch]
    rejected_ids = [torch.tensor(b["rejected_ids"], dtype=torch.long) for b in batch]

    def pad(seqs: List[torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        max_len = max(s.shape[0] for s in seqs)
        ids  = torch.zeros(len(seqs), max_len, dtype=torch.long)
        mask = torch.zeros(len(seqs), max_len, dtype=torch.long)
        for i, s in enumerate(seqs):
            ids[i, : s.shape[0]]  = s
            mask[i, : s.shape[0]] = 1
        return ids, mask

    c_ids, c_mask = pad(chosen_ids)
    r_ids, r_mask = pad(rejected_ids)
    return {
        "chosen_input_ids":       c_ids,
        "chosen_attention_mask":  c_mask,
        "rejected_input_ids":     r_ids,
        "rejected_attention_mask": r_mask,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Loss
# ─────────────────────────────────────────────────────────────────────────────

def _log_probs_from_logits(
    logits: torch.Tensor,
    labels: torch.Tensor,
) -> torch.Tensor:
    """Compute average token log-probability for each sequence in batch.

    logits : [B, S, V]
    labels : [B, S]
    returns: [B]  (mean log-prob per valid token)
    """
    log_probs  = F.log_softmax(logits[:, :-1, :], dim=-1)  # [B, S-1, V]
    target_ids = labels[:, 1:].unsqueeze(-1)                # [B, S-1, 1]
    token_log_probs = log_probs.gather(2, target_ids).squeeze(-1)  # [B, S-1]
    # Create mask: ignore padding positions (label==0 is pad)
    mask = (labels[:, 1:] != 0).float()
    seq_log_prob = (token_log_probs * mask).sum(dim=-1) / mask.sum(dim=-1).clamp(min=1)
    return seq_log_prob  # [B]


def dpo_loss(
    policy_chosen_logprob:   torch.Tensor,  # [B]
    policy_rejected_logprob: torch.Tensor,  # [B]
    ref_chosen_logprob:      torch.Tensor,  # [B]
    ref_rejected_logprob:    torch.Tensor,  # [B]
    beta: float = 0.1,
) -> torch.Tensor:
    """DPO loss (Bradley-Terry preference model formulation).

    loss = -mean[ log σ( β * (log π_θ(chosen|x) - log π_θ(rejected|x)
                              - log π_ref(chosen|x) + log π_ref(rejected|x)) ) ]
    """
    # Relative log-ratio: policy minus reference
    chosen_ratio   = policy_chosen_logprob   - ref_chosen_logprob    # [B]
    rejected_ratio = policy_rejected_logprob - ref_rejected_logprob  # [B]
    logits = beta * (chosen_ratio - rejected_ratio)                   # [B]
    loss   = -F.logsigmoid(logits).mean()
    return loss


# ─────────────────────────────────────────────────────────────────────────────
# Trainer
# ─────────────────────────────────────────────────────────────────────────────

class DPOTrainer:
    """Train a CodeMind model using Direct Preference Optimization.

    Example:
        trainer = DPOTrainer(
            model=model_manager.model,
            tokenizer=model_manager.tokenizer,
            beta=0.1,
        )
        data = [
            {"prompt": "Sort a list in Python",
             "chosen":  "Use sorted() or list.sort()...",
             "rejected": "I don't know about that topic."},
        ]
        trainer.train(data, epochs=2)
    """

    def __init__(
        self,
        model: torch.nn.Module,
        tokenizer,
        beta: float = 0.1,
        learning_rate: float = 5e-7,
        max_length: int = 512,
        batch_size: int = 2,
        gradient_accumulation_steps: int = 4,
        device: Optional[str] = None,
        output_dir: str = "models/dpo",
    ) -> None:
        self.model     = model
        self.tokenizer = tokenizer
        self.beta      = beta
        self.lr        = learning_rate
        self.max_length = max_length
        self.batch_size = batch_size
        self.grad_acc   = gradient_accumulation_steps
        self.device     = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Frozen reference model (policy at step 0)
        self.ref_model = copy.deepcopy(model).eval()
        for p in self.ref_model.parameters():
            p.requires_grad = False
        self.ref_model.to(self.device)

        self.model.to(self.device)

    def _forward(
        self,
        model: torch.nn.Module,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        out    = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = out.logits if hasattr(out, "logits") else out[0]
        return logits

    def train(
        self,
        data: List[Dict[str, str]],
        epochs: int = 1,
        save_steps: int = 100,
    ) -> List[float]:
        """Run DPO training loop. Returns list of loss values per step."""
        dataset    = DPODataset(data, self.tokenizer, self.max_length)
        dataloader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=True,
            collate_fn=_dpo_collate,
        )
        optimizer = AdamW(
            [p for p in self.model.parameters() if p.requires_grad],
            lr=self.lr,
        )
        losses: List[float] = []

        logger.info("DPO training: %d samples, %d epochs, beta=%.3f", len(dataset), epochs, self.beta)

        step = 0
        for epoch in range(epochs):
            self.model.train()
            optimizer.zero_grad()

            for batch_idx, batch in enumerate(dataloader):
                c_ids  = batch["chosen_input_ids"].to(self.device)
                c_mask = batch["chosen_attention_mask"].to(self.device)
                r_ids  = batch["rejected_input_ids"].to(self.device)
                r_mask = batch["rejected_attention_mask"].to(self.device)

                # Policy log-probs
                c_logits = self._forward(self.model, c_ids, c_mask)
                r_logits = self._forward(self.model, r_ids, r_mask)
                policy_chosen_lp   = _log_probs_from_logits(c_logits, c_ids)
                policy_rejected_lp = _log_probs_from_logits(r_logits, r_ids)

                # Reference log-probs (no grad)
                with torch.no_grad():
                    ref_c_logits = self._forward(self.ref_model, c_ids, c_mask)
                    ref_r_logits = self._forward(self.ref_model, r_ids, r_mask)
                    ref_chosen_lp   = _log_probs_from_logits(ref_c_logits, c_ids)
                    ref_rejected_lp = _log_probs_from_logits(ref_r_logits, r_ids)

                loss = dpo_loss(
                    policy_chosen_lp,
                    policy_rejected_lp,
                    ref_chosen_lp,
                    ref_rejected_lp,
                    beta=self.beta,
                )
                # Gradient accumulation
                (loss / self.grad_acc).backward()

                if (batch_idx + 1) % self.grad_acc == 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                    optimizer.step()
                    optimizer.zero_grad()
                    step += 1

                    losses.append(loss.item())
                    logger.info("DPO step %d | loss=%.4f", step, loss.item())

                    if step % save_steps == 0:
                        ckpt = self.output_dir / f"dpo_step_{step}.pt"
                        torch.save(self.model.state_dict(), ckpt)
                        logger.info("Checkpoint saved: %s", ckpt)

        # Final save
        torch.save(self.model.state_dict(), self.output_dir / "dpo_final.pt")
        logger.info("DPO training complete. Final model saved to %s", self.output_dir)
        return losses
