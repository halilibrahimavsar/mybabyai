"""
Knowledge Distillation for CodeMind

Train a small student model to mimic a larger teacher model.
This enables the 125M/350M parameter CodeMind to learn from
models like Phi-3, Mistral, or GPT-4.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Generator, List, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from src.core.checkpointing import attach_checkpoint_metadata, build_checkpoint_metadata


@dataclass
class DistillationConfig:
    """Configuration for knowledge distillation."""

    temperature: float = 4.0
    alpha: float = 0.7
    beta: float = 0.3

    learning_rate: float = 1e-4
    batch_size: int = 8
    num_epochs: int = 3
    max_length: int = 512

    teacher_model: str = "microsoft/Phi-3-mini-4k-instruct"
    output_dir: str = "codemind/distilled"

    save_every: int = 500
    eval_every: int = 100

    use_hard_labels: bool = True
    use_soft_labels: bool = True
    sequence_level_distillation: bool = True


class DistillationDataset(Dataset):
    """Dataset for knowledge distillation."""

    def __init__(
        self,
        data_path: str,
        tokenizer: Any,
        teacher_tokenizer: Any,
        max_length: int = 512,
    ):
        self.tokenizer = tokenizer
        self.teacher_tokenizer = teacher_tokenizer
        self.max_length = max_length

        self.data: List[Dict[str, Any]] = []
        self._load_data(data_path)

    def _load_data(self, path: str) -> None:
        path = Path(path)

        if path.is_file():
            with open(path, "r", encoding="utf-8") as f:
                items = json.load(f)
            self.data.extend(items)
        else:
            for file_path in path.glob("**/*.json"):
                try:
                    with open(file_path, "r", encoding="utf-8") as f:
                        items = json.load(f)
                    self.data.extend(items)
                except Exception:
                    continue

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        item = self.data[idx]

        prompt = item.get("user", item.get("prompt", item.get("input", "")))
        response = item.get("assistant", item.get("output", item.get("response", "")))

        full_text = f"<|python|>{prompt}\n<|answer|>{response}<|eos|>"

        student_ids = self.tokenizer.encode(full_text, add_special_tokens=False)

        if len(student_ids) > self.max_length:
            student_ids = student_ids[: self.max_length]

        return {
            "input_ids": torch.tensor(student_ids[:-1], dtype=torch.long),
            "labels": torch.tensor(student_ids[1:], dtype=torch.long),
            "prompt": prompt,
            "response": response,
        }


def distillation_loss(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    labels: torch.Tensor,
    temperature: float = 4.0,
    alpha: float = 0.7,
) -> torch.Tensor:
    """
    Compute knowledge distillation loss.

    Args:
        student_logits: Logits from student model
        teacher_logits: Logits from teacher model
        labels: Ground truth labels
        temperature: Temperature for softening distributions
        alpha: Weight for KL divergence loss (soft labels)

    Returns:
        Combined loss value
    """
    teacher_probs = F.softmax(teacher_logits / temperature, dim=-1)
    teacher_log_probs = F.log_softmax(teacher_logits / temperature, dim=-1)
    student_log_probs = F.log_softmax(student_logits / temperature, dim=-1)

    kl_loss = F.kl_div(
        student_log_probs,
        teacher_probs,
        reduction="batchmean",
    ) * (temperature**2)

    ce_loss = F.cross_entropy(
        student_logits.view(-1, student_logits.size(-1)),
        labels.view(-1),
        ignore_index=-100,
    )

    total_loss = alpha * kl_loss + (1 - alpha) * ce_loss

    return total_loss, kl_loss, ce_loss


class KnowledgeDistiller:
    """
    Knowledge Distillation trainer.

    Trains a small student model to mimic a larger teacher model.
    """

    def __init__(
        self,
        student_model: nn.Module,
        teacher_model: Optional[nn.Module],
        student_tokenizer: Any,
        teacher_tokenizer: Optional[Any],
        config: DistillationConfig,
        device: str = "auto",
    ):
        self.config = config
        self.device = self._get_device(device) if device == "auto" else device

        self.student = student_model.to(self.device)
        self.teacher = teacher_model
        self.student_tokenizer = student_tokenizer
        self.teacher_tokenizer = teacher_tokenizer
        self._soft_label_compatible = self._can_use_soft_labels()
        self._soft_label_warning_emitted = False

        if self.teacher is not None:
            self.teacher = self.teacher.to(self.device)
            self.teacher.eval()
            for param in self.teacher.parameters():
                param.requires_grad = False

        self.optimizer = AdamW(
            self.student.parameters(),
            lr=config.learning_rate,
            weight_decay=0.01,
        )

        self.global_step = 0
        self.best_loss = float("inf")

    def _get_device(self, _: str) -> str:
        if torch.cuda.is_available():
            return "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
        return "cpu"

    def _get_teacher_outputs(
        self,
        prompts: List[str],
        responses: List[str],
    ) -> List[str]:
        """Get teacher model outputs for given prompts."""
        if self.teacher is None:
            return responses

        teacher_outputs = []

        for prompt, response in zip(prompts, responses):
            full_prompt = f"{prompt}\n{response}"

            inputs = self.teacher_tokenizer(
                full_prompt,
                return_tensors="pt",
                truncation=True,
                max_length=self.config.max_length,
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            with torch.no_grad():
                outputs = self.teacher.generate(
                    **inputs,
                    max_new_tokens=len(response.split()) + 50,
                    temperature=0.7,
                    do_sample=True,
                )

            generated = self.teacher_tokenizer.decode(
                outputs[0],
                skip_special_tokens=True,
            )
            teacher_outputs.append(generated)

        return teacher_outputs

    def _can_use_soft_labels(self) -> bool:
        if self.teacher is None:
            return False
        teacher_vocab = getattr(getattr(self.teacher, "config", None), "vocab_size", None)
        student_vocab = getattr(getattr(self.student, "config", None), "vocab_size", None)
        if teacher_vocab is None or student_vocab is None:
            return False
        return int(teacher_vocab) == int(student_vocab)

    def _build_student_targets_batch(
        self,
        prompts: List[str],
        targets: List[str],
    ) -> Dict[str, torch.Tensor]:
        sequences: List[torch.Tensor] = []
        labels: List[torch.Tensor] = []
        attention_masks: List[torch.Tensor] = []

        for prompt, target in zip(prompts, targets):
            full_text = f"<|python|>{prompt}\n<|answer|>{target}<|eos|>"
            token_ids = self.student_tokenizer.encode(full_text, add_special_tokens=False)
            token_ids = token_ids[: self.config.max_length]
            if len(token_ids) < 2:
                token_ids = token_ids + [self.student_tokenizer.unk_token_id]

            input_ids = torch.tensor(token_ids[:-1], dtype=torch.long)
            label_ids = torch.tensor(token_ids[1:], dtype=torch.long)
            attn = torch.ones_like(input_ids, dtype=torch.long)

            sequences.append(input_ids)
            labels.append(label_ids)
            attention_masks.append(attn)

        max_len = max(item.size(0) for item in sequences)
        padded_inputs = []
        padded_labels = []
        padded_masks = []

        for input_ids, label_ids, attn in zip(sequences, labels, attention_masks):
            pad_len = max_len - input_ids.size(0)
            if pad_len > 0:
                padded_inputs.append(
                    torch.cat(
                        [
                            input_ids,
                            torch.full(
                                (pad_len,),
                                self.student_tokenizer.pad_token_id,
                                dtype=torch.long,
                            ),
                        ]
                    )
                )
                padded_labels.append(
                    torch.cat(
                        [label_ids, torch.full((pad_len,), -100, dtype=torch.long)]
                    )
                )
                padded_masks.append(
                    torch.cat([attn, torch.zeros((pad_len,), dtype=torch.long)])
                )
            else:
                padded_inputs.append(input_ids)
                padded_labels.append(label_ids)
                padded_masks.append(attn)

        return {
            "input_ids": torch.stack(padded_inputs),
            "labels": torch.stack(padded_labels),
            "attention_mask": torch.stack(padded_masks),
        }

    def train(
        self,
        train_data_path: str,
        eval_data_path: Optional[str] = None,
    ) -> Dict[str, List[float]]:
        """
        Train student model with knowledge distillation.

        Args:
            train_data_path: Path to training data
            eval_data_path: Path to evaluation data

        Returns:
            Training history
        """
        train_dataset = DistillationDataset(
            train_data_path,
            self.student_tokenizer,
            self.teacher_tokenizer,
            self.config.max_length,
        )

        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=0,
            collate_fn=self._collate_fn,
        )

        history = {
            "total_loss": [],
            "kl_loss": [],
            "ce_loss": [],
        }

        output_dir = Path(self.config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        self.student.train()

        for epoch in range(self.config.num_epochs):
            epoch_losses = {"total": 0, "kl": 0, "ce": 0}

            progress = tqdm(
                train_loader,
                desc=f"Epoch {epoch + 1}/{self.config.num_epochs}",
            )

            for batch in progress:
                batch = {
                    k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                    for k, v in batch.items()
                }

                self.optimizer.zero_grad()

                if self.teacher is not None and self.config.sequence_level_distillation:
                    teacher_targets = self._get_teacher_outputs(
                        batch["prompts"],
                        batch["responses"],
                    )
                    rebuilt_batch = self._build_student_targets_batch(
                        batch["prompts"],
                        teacher_targets,
                    )
                    batch["input_ids"] = rebuilt_batch["input_ids"].to(self.device)
                    batch["labels"] = rebuilt_batch["labels"].to(self.device)
                    batch["attention_mask"] = rebuilt_batch["attention_mask"].to(self.device)

                student_outputs = self.student(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                )

                student_logits = student_outputs["logits"]
                labels = batch["labels"]

                if self.teacher is not None and self.config.use_soft_labels and self._soft_label_compatible:
                    with torch.no_grad():
                        teacher_outputs = self.teacher(
                            input_ids=batch["input_ids"],
                            attention_mask=batch["attention_mask"],
                        )
                    teacher_logits = teacher_outputs["logits"]

                    total_loss, kl_loss, ce_loss = distillation_loss(
                        student_logits,
                        teacher_logits,
                        labels,
                        self.config.temperature,
                        self.config.alpha,
                    )
                else:
                    if (
                        self.teacher is not None
                        and self.config.use_soft_labels
                        and not self._soft_label_compatible
                        and not self._soft_label_warning_emitted
                    ):
                        print(
                            "Soft-label KL disabled: teacher/student vocabularies differ. "
                            "Using sequence-level CE supervision."
                        )
                        self._soft_label_warning_emitted = True
                    total_loss = F.cross_entropy(
                        student_logits.view(-1, student_logits.size(-1)),
                        labels.view(-1),
                        ignore_index=-100,
                    )
                    kl_loss = torch.tensor(0.0)
                    ce_loss = total_loss

                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.student.parameters(), 1.0)
                self.optimizer.step()

                epoch_losses["total"] += total_loss.item()
                epoch_losses["kl"] += (
                    kl_loss.item() if isinstance(kl_loss, torch.Tensor) else kl_loss
                )
                epoch_losses["ce"] += (
                    ce_loss.item() if isinstance(ce_loss, torch.Tensor) else ce_loss
                )

                progress.set_postfix(
                    {
                        "loss": f"{total_loss.item():.4f}",
                        "kl": f"{kl_loss.item() if isinstance(kl_loss, torch.Tensor) else 0:.4f}",
                    }
                )

                self.global_step += 1

                if self.global_step % self.config.save_every == 0:
                    self._save_checkpoint(
                        output_dir / f"checkpoint_{self.global_step}.pt"
                    )

            n_batches = len(train_loader)
            history["total_loss"].append(epoch_losses["total"] / n_batches)
            history["kl_loss"].append(epoch_losses["kl"] / n_batches)
            history["ce_loss"].append(epoch_losses["ce"] / n_batches)

            print(
                f"\nEpoch {epoch + 1} - "
                f"Total: {history['total_loss'][-1]:.4f}, "
                f"KL: {history['kl_loss'][-1]:.4f}, "
                f"CE: {history['ce_loss'][-1]:.4f}"
            )

        self._save_checkpoint(output_dir / "model_final.pt", is_final=True)

        return history

    def _collate_fn(self, batch: List[Dict]) -> Dict[str, Any]:
        """Collate function for DataLoader."""
        max_len = max(item["input_ids"].size(0) for item in batch)

        input_ids = []
        attention_mask = []
        labels = []
        prompts = []
        responses = []

        for item in batch:
            seq_len = item["input_ids"].size(0)
            pad_len = max_len - seq_len

            input_ids.append(
                torch.cat(
                    [
                        item["input_ids"],
                        torch.zeros(pad_len, dtype=torch.long),
                    ]
                )
            )

            attention_mask.append(
                torch.cat(
                    [
                        torch.ones(seq_len, dtype=torch.long),
                        torch.zeros(pad_len, dtype=torch.long),
                    ]
                )
            )

            labels.append(
                torch.cat(
                    [
                        item["labels"],
                        torch.tensor([-100] * pad_len, dtype=torch.long),
                    ]
                )
            )

            prompts.append(item["prompt"])
            responses.append(item["response"])

        return {
            "input_ids": torch.stack(input_ids),
            "attention_mask": torch.stack(attention_mask),
            "labels": torch.stack(labels),
            "prompts": prompts,
            "responses": responses,
        }

    def _save_checkpoint(
        self,
        path: Path,
        is_final: bool = False,
    ) -> None:
        """Save training checkpoint."""
        model_config = {
            "vocab_size": getattr(self.student.config, "vocab_size", 0),
            "hidden_size": getattr(self.student.config, "hidden_size", 0),
            "num_hidden_layers": getattr(self.student.config, "num_hidden_layers", 0),
            "num_attention_heads": getattr(self.student.config, "num_attention_heads", 0),
            "intermediate_size": getattr(self.student.config, "intermediate_size", 0),
            "max_position_embeddings": getattr(
                self.student.config, "max_position_embeddings", 0
            ),
        }
        checkpoint = {
            "model_state_dict": self.student.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "global_step": self.global_step,
            "config": model_config,
            "distillation_config": self.config.__dict__,
        }
        metadata = build_checkpoint_metadata(
            model_config=model_config,
            tokenizer=self.student_tokenizer,
            tokenizer_type=type(self.student_tokenizer).__name__,
        )
        checkpoint = attach_checkpoint_metadata(checkpoint, metadata)

        torch.save(checkpoint, path)

        if is_final:
            print(f"Final model saved to: {path}")
        else:
            print(f"Checkpoint saved: {path}")


def create_teacher_student_pipeline(
    teacher_name: str = "microsoft/Phi-3-mini-4k-instruct",
    student_vocab_size: int = 16384,
    student_hidden_size: int = 768,
    student_layers: int = 12,
) -> Tuple[Any, Any, Any, Any]:
    """
    Create teacher-student pipeline for distillation.

    Args:
        teacher_name: Name of teacher model
        student_vocab_size: Vocabulary size for student
        student_hidden_size: Hidden size for student
        student_layers: Number of layers for student

    Returns:
        Tuple of (teacher, student, teacher_tokenizer, student_tokenizer)
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig

    print(f"Loading teacher model: {teacher_name}")

    config = AutoConfig.from_pretrained(teacher_name, trust_remote_code=True)
    if hasattr(config, "rope_scaling") and config.rope_scaling is not None:
        config.rope_scaling = None

    teacher_tokenizer = AutoTokenizer.from_pretrained(
        teacher_name, trust_remote_code=True
    )
    teacher = AutoModelForCausalLM.from_pretrained(
        teacher_name,
        trust_remote_code=True,
        torch_dtype=torch.float16,
        device_map="auto",
        config=config,
    )

    print("Creating student model...")

    from src.core.model.codemind import CodeMindConfig, CodeMindForCausalLM

    student_config = CodeMindConfig(
        vocab_size=student_vocab_size,
        hidden_size=student_hidden_size,
        num_hidden_layers=student_layers,
        num_attention_heads=12,
        intermediate_size=student_hidden_size * 4,
    )

    student = CodeMindForCausalLM(student_config)

    from src.core.tokenizer.advanced_tokenizer import AdvancedCodeTokenizer

    student_tokenizer = AdvancedCodeTokenizer(vocab_size=student_vocab_size)

    return teacher, student, teacher_tokenizer, student_tokenizer


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Knowledge Distillation for CodeMind")
    parser.add_argument("--data", type=str, required=True, help="Training data path")
    parser.add_argument("--output", type=str, default="codemind/distilled")
    parser.add_argument(
        "--teacher", type=str, default="microsoft/Phi-3-mini-4k-instruct"
    )
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--temperature", type=float, default=4.0)

    args = parser.parse_args()

    teacher, student, teacher_tok, student_tok = create_teacher_student_pipeline(
        teacher_name=args.teacher,
    )

    config = DistillationConfig(
        teacher_model=args.teacher,
        output_dir=args.output,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        temperature=args.temperature,
    )

    distiller = KnowledgeDistiller(
        student_model=student,
        teacher_model=teacher,
        student_tokenizer=student_tok,
        teacher_tokenizer=teacher_tok,
        config=config,
    )

    history = distiller.train(args.data)

    print("\nTraining complete!")
    print(f"Final loss: {history['total_loss'][-1]:.4f}")
