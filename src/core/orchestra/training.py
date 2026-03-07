"""
Orchestra Training — training pipelines for the Latent Expert Orchestra components.

This module contains trainers for:
  1. BridgeTrainer: Trains the MLP bridge with alignment + downstream loss (Faz 3)
  2. RouterTrainer: Trains the neural router with supervised classification (Faz 4)
  3. OrchestraTrainer: End-to-end fine-tuning of bridge + router together (Faz 5)

Each trainer follows the phased training strategy from the architecture analysis,
ensuring components are trained in isolation before being combined.
"""

from __future__ import annotations

import json
import logging
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from src.core.orchestra.expert_config import ExpertDomain, SHARED_HIDDEN_SIZE
from src.core.orchestra.expert_model import ExpertModel
from src.core.orchestra.latent_bridge import BridgePool
from src.core.orchestra.neural_router import NeuralRouter, ExecutionMode

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════
#  Datasets
# ═══════════════════════════════════════════════════════════════════════════════


class PairedAlignmentDataset(Dataset):
    """Dataset for bridge alignment training (Faz 3).

    Each sample is a pair of texts that express the same concept in two
    different domains. The bridge learns to project Expert A's hidden
    states such that they are close to Expert B's hidden states for
    the corresponding paired text.

    Format: JSON lines file with {"source": "...", "target": "..."}
    Example:
        {"source": "Fibonacci fonksiyonu yaz", "target": "def fibonacci(n): ..."}
    """

    def __init__(
        self,
        data_path: str,
        tokenizer: Any,
        max_length: int = 256,
    ) -> None:
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.pairs: List[Dict[str, str]] = []

        with open(data_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    self.pairs.append(json.loads(line))

        logger.info("Loaded %d alignment pairs from %s", len(self.pairs), data_path)

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        pair = self.pairs[idx]
        source = self.tokenizer(
            pair["source"],
            return_tensors="pt",
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
        )
        target = self.tokenizer(
            pair["target"],
            return_tensors="pt",
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
        )
        return {
            "source_input_ids": source["input_ids"].squeeze(0),
            "source_attention_mask": source["attention_mask"].squeeze(0),
            "target_input_ids": target["input_ids"].squeeze(0),
            "target_attention_mask": target["attention_mask"].squeeze(0),
        }


class RouterDataset(Dataset):
    """Dataset for router supervised training (Faz 4).

    Format: JSON lines file with {"prompt": "...", "expert": "turkish", "mode": "single"}
    """

    def __init__(
        self,
        data_path: str,
        tokenizer: Any,
        domain_to_idx: Dict[ExpertDomain, int],
        max_length: int = 256,
    ) -> None:
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.domain_to_idx = domain_to_idx
        self.samples: List[Dict[str, Any]] = []

        with open(data_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    self.samples.append(json.loads(line))

        logger.info("Loaded %d routing samples from %s", len(self.samples), data_path)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        sample = self.samples[idx]
        inputs = self.tokenizer(
            sample["prompt"],
            return_tensors="pt",
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
        )
        expert_domain = ExpertDomain(sample["expert"])
        expert_label = self.domain_to_idx[expert_domain]
        mode_label = 0 if sample.get("mode", "single") == "single" else 1

        return {
            "input_ids": inputs["input_ids"].squeeze(0),
            "attention_mask": inputs["attention_mask"].squeeze(0),
            "expert_label": torch.tensor(expert_label, dtype=torch.long),
            "mode_label": torch.tensor(mode_label, dtype=torch.long),
        }


# ═══════════════════════════════════════════════════════════════════════════════
#  Bridge Trainer (Faz 3)
# ═══════════════════════════════════════════════════════════════════════════════


class BridgeTrainer:
    """Trains the latent bridge to align expert hidden states.

    Loss function:
        L = λ₁ · L_align + λ₂ · L_downstream

    Where:
        L_align = 1 - cosine_similarity(projected_source, target_hidden)
        L_downstream = CrossEntropy(target_expert_output_given_bridge, ground_truth)

    Both experts are frozen (optionally with LoRA adapters).
    Only bridge parameters are updated.
    """

    def __init__(
        self,
        source_expert: ExpertModel,
        target_expert: ExpertModel,
        bridge: BridgePool,
        tokenizer: Any,
        source_domain_id: int = 0,
        target_domain_id: int = 1,
        lr: float = 1e-4,
        alignment_weight: float = 1.0,
        downstream_weight: float = 0.5,
        device: Optional[str] = None,
    ) -> None:
        self.source_expert = source_expert
        self.target_expert = target_expert
        self.bridge = bridge
        self.tokenizer = tokenizer
        self.source_domain_id = source_domain_id
        self.target_domain_id = target_domain_id
        self.alignment_weight = alignment_weight
        self.downstream_weight = downstream_weight
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        # Ensure experts are frozen
        self.source_expert.freeze()
        self.target_expert.freeze()

        # Optimizer only updates bridge parameters
        self.optimizer = torch.optim.AdamW(
            self.bridge.parameters(), lr=lr, weight_decay=0.01,
        )

        self.cosine_sim = nn.CosineSimilarity(dim=-1)

    def train(
        self,
        dataset: PairedAlignmentDataset,
        epochs: int = 10,
        batch_size: int = 16,
        save_path: Optional[str] = None,
        log_every: int = 10,
    ) -> Dict[str, List[float]]:
        """Train the bridge.

        Args:
            dataset: Paired alignment dataset.
            epochs: Number of training epochs.
            batch_size: Training batch size.
            save_path: Path to save bridge checkpoint.
            log_every: Log metrics every N steps.

        Returns:
            Training metrics dictionary.
        """
        dataloader = DataLoader(
            dataset, batch_size=batch_size, shuffle=True, drop_last=True,
        )

        self.bridge.train()
        metrics: Dict[str, List[float]] = {
            "total_loss": [],
            "alignment_loss": [],
            "cosine_similarity": [],
        }

        global_step = 0
        for epoch in range(epochs):
            epoch_loss = 0.0
            epoch_align = 0.0
            epoch_sim = 0.0
            num_batches = 0

            for batch in dataloader:
                source_ids = batch["source_input_ids"].to(self.device)
                source_mask = batch["source_attention_mask"].to(self.device)
                target_ids = batch["target_input_ids"].to(self.device)
                target_mask = batch["target_attention_mask"].to(self.device)

                # Extract hidden states from both experts
                with torch.no_grad():
                    source_hidden = self.source_expert.extract_hidden_states(
                        source_ids, source_mask,
                    )
                    target_hidden = self.target_expert.extract_hidden_states(
                        target_ids, target_mask,
                    )

                # Project source through bridge
                projected = self.bridge(
                    source_hidden,
                    source_domain_id=self.source_domain_id,
                    target_domain_id=self.target_domain_id,
                    attention_mask=source_mask,
                )

                # Alignment loss: cosine similarity between projected and target
                # Mean-pool both for sequence-level alignment
                proj_pooled = self._masked_mean(projected, source_mask)
                target_pooled = self._masked_mean(target_hidden, target_mask)

                cos_sim = self.cosine_sim(proj_pooled, target_pooled)
                alignment_loss = (1 - cos_sim).mean()

                # Total loss (downstream loss can be added in Faz 5)
                total_loss = self.alignment_weight * alignment_loss

                self.optimizer.zero_grad()
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.bridge.parameters(), 1.0)
                self.optimizer.step()

                epoch_loss += total_loss.item()
                epoch_align += alignment_loss.item()
                epoch_sim += cos_sim.mean().item()
                num_batches += 1
                global_step += 1

                if global_step % log_every == 0:
                    logger.info(
                        "Step %d | Loss: %.4f | Align: %.4f | CosSim: %.4f",
                        global_step,
                        total_loss.item(),
                        alignment_loss.item(),
                        cos_sim.mean().item(),
                    )

            avg_loss = epoch_loss / max(num_batches, 1)
            avg_align = epoch_align / max(num_batches, 1)
            avg_sim = epoch_sim / max(num_batches, 1)

            metrics["total_loss"].append(avg_loss)
            metrics["alignment_loss"].append(avg_align)
            metrics["cosine_similarity"].append(avg_sim)

            logger.info(
                "Epoch %d/%d | Avg Loss: %.4f | Avg CosSim: %.4f",
                epoch + 1, epochs, avg_loss, avg_sim,
            )

        # Save checkpoint
        if save_path:
            save_dir = Path(save_path)
            save_dir.parent.mkdir(parents=True, exist_ok=True)
            torch.save(
                {"model_state_dict": self.bridge.state_dict()},
                str(save_dir),
            )
            logger.info("Bridge checkpoint saved to %s", save_path)

        return metrics

    @staticmethod
    def _masked_mean(
        hidden_states: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """Mean-pool hidden states with mask."""
        mask_expanded = mask.unsqueeze(-1).to(hidden_states.dtype)
        summed = (hidden_states * mask_expanded).sum(dim=1)
        count = mask_expanded.sum(dim=1).clamp(min=1e-9)
        return summed / count


# ═══════════════════════════════════════════════════════════════════════════════
#  Router Trainer (Faz 4)
# ═══════════════════════════════════════════════════════════════════════════════


class RouterTrainer:
    """Trains the neural router with supervised classification.

    Loss function:
        L = L_expert + α · L_mode + β · L_balance

    Where:
        L_expert = CrossEntropy(expert_logits, expert_label)
        L_mode = CrossEntropy(mode_logits, mode_label)
        L_balance = Variance(expert_usage_counts) — prevents router collapse
    """

    def __init__(
        self,
        router: NeuralRouter,
        lr: float = 3e-4,
        balance_weight: float = 0.1,
        mode_weight: float = 0.5,
        device: Optional[str] = None,
    ) -> None:
        self.router = router
        self.balance_weight = balance_weight
        self.mode_weight = mode_weight
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        self.optimizer = torch.optim.AdamW(
            self.router.parameters(), lr=lr, weight_decay=0.01,
        )

        self.router.to(self.device)

    def train(
        self,
        dataset: RouterDataset,
        epochs: int = 20,
        batch_size: int = 32,
        save_path: Optional[str] = None,
        log_every: int = 10,
    ) -> Dict[str, List[float]]:
        """Train the router.

        Returns:
            Training metrics.
        """
        dataloader = DataLoader(
            dataset, batch_size=batch_size, shuffle=True, drop_last=True,
        )

        self.router.train()
        metrics: Dict[str, List[float]] = {
            "total_loss": [],
            "expert_accuracy": [],
        }

        global_step = 0
        for epoch in range(epochs):
            epoch_loss = 0.0
            correct = 0
            total = 0

            for batch in dataloader:
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                expert_labels = batch["expert_label"].to(self.device)
                mode_labels = batch["mode_label"].to(self.device)

                expert_logits, mode_logits = self.router(input_ids, attention_mask)

                # Expert classification loss
                expert_loss = F.cross_entropy(expert_logits, expert_labels)

                # Mode classification loss
                mode_loss = F.cross_entropy(mode_logits, mode_labels)

                # Load balancing loss: penalize uneven expert usage
                expert_probs = F.softmax(expert_logits, dim=-1)
                avg_probs = expert_probs.mean(dim=0)
                balance_loss = avg_probs.var()

                total_loss = (
                    expert_loss
                    + self.mode_weight * mode_loss
                    + self.balance_weight * balance_loss
                )

                self.optimizer.zero_grad()
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.router.parameters(), 1.0)
                self.optimizer.step()

                # Track accuracy
                preds = expert_logits.argmax(dim=-1)
                correct += (preds == expert_labels).sum().item()
                total += expert_labels.size(0)
                epoch_loss += total_loss.item()
                global_step += 1

                if global_step % log_every == 0:
                    acc = correct / max(total, 1)
                    logger.info(
                        "Step %d | Loss: %.4f | Acc: %.2f%%",
                        global_step, total_loss.item(), acc * 100,
                    )

            num_batches = len(dataloader)
            avg_loss = epoch_loss / max(num_batches, 1)
            accuracy = correct / max(total, 1)

            metrics["total_loss"].append(avg_loss)
            metrics["expert_accuracy"].append(accuracy)

            logger.info(
                "Epoch %d/%d | Avg Loss: %.4f | Accuracy: %.2f%%",
                epoch + 1, epochs, avg_loss, accuracy * 100,
            )

        if save_path:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            torch.save(
                {"model_state_dict": self.router.state_dict()},
                str(save_path),
            )
            logger.info("Router checkpoint saved to %s", save_path)

        return metrics
