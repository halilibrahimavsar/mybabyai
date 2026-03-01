"""
CodeMind Training Pipeline

Complete training pipeline for CodeMind model:
1. Tokenizer training
2. Pre-training
3. Meta-learning fine-tuning
4. Active learning integration
"""

import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Any
import json
import yaml
from datetime import datetime

import torch
from torch.optim import AdamW
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm

ROOT_DIR = Path(__file__).parent.parent


from src.core.tokenizer.code_tokenizer import CodeTokenizer
from src.core.model.codemind import (
    CodeMindConfig,
    CodeMindForCausalLM,
    create_codemind_350m,
    create_codemind_125m,
)
from src.core.checkpointing import attach_checkpoint_metadata, build_checkpoint_metadata
from src.core.data.schema import normalize_sample
from src.core.prompting import TOKENS
from src.core.training.meta_learning import MAMLTrainer, MAMLConfig, TaskSampler
from src.core.training.active_learning import ActiveLearner, ContinuousLearningPipeline


class CodeDataset(Dataset):
    def __init__(
        self,
        data_path: str,
        tokenizer: CodeTokenizer,
        max_length: int = 2048,
        lang: Optional[str] = None,
    ):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.lang = lang

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
        normalized = normalize_sample(
            item,
            default_language=self.lang or "python",
        )
        if normalized is None:
            normalized = {
                "user": "",
                "assistant": "",
                "language": self.lang or "python",
                "task_type": "code_gen",
                "metadata": {},
            }

        prompt = normalized["user"]
        response = normalized["assistant"]
        lang = normalized["language"]

        full_text = f"<|{lang}|>{prompt}\n{TOKENS.answer}{response}{TOKENS.eos}"

        tokens = self.tokenizer.encode(full_text, add_special_tokens=False, lang=lang)

        if len(tokens) > self.max_length:
            tokens = tokens[: self.max_length]

        input_ids = tokens[:-1]
        labels = tokens[1:]

        attention_mask = [1] * len(input_ids)

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }


def collate_fn(batch: List[Dict]) -> Dict[str, torch.Tensor]:
    max_len = max(item["input_ids"].size(0) for item in batch)

    input_ids = []
    attention_mask = []
    labels = []

    for item in batch:
        seq_len = item["input_ids"].size(0)
        pad_len = max_len - seq_len

        input_ids.append(
            torch.cat([item["input_ids"], torch.zeros(pad_len, dtype=torch.long)])
        )
        attention_mask.append(
            torch.cat([item["attention_mask"], torch.zeros(pad_len, dtype=torch.long)])
        )
        labels.append(
            torch.cat(
                [item["labels"], torch.tensor([-100] * pad_len, dtype=torch.long)]
            )
        )

    return {
        "input_ids": torch.stack(input_ids),
        "attention_mask": torch.stack(attention_mask),
        "labels": torch.stack(labels),
    }


from src.core.model_manager import ModelManager
from src.utils.config import Config

class CodeMindTrainer:
    def __init__(self, config_path: Optional[str] = None, device: str = "auto"):
        self.config_obj = Config(config_path) if config_path else Config()
        self.config = self.config_obj.data
        
        self.model_manager = ModelManager(config=self.config_obj)
        self.device = self.model_manager.device

        print(f"Using device: {self.device}")

        self.tokenizer: Optional[Any] = None
        self.model: Optional[Any] = None
        self.scaler = GradScaler() if self.device == "cuda" else None

    def setup_tokenizer(
        self,
        training_files: Optional[List[str]] = None,
        save_path: Optional[str] = None,
    ) -> Any:
        # If we have a ModelManager, it might already have a tokenizer or can load one
        if not self.tokenizer:
            self.model_manager.load_model() # This ensures tokenizer is loaded from config
            self.tokenizer = self.model_manager.tokenizer
            
        print(f"Tokenizer ready.")
        return self.tokenizer

    def setup_model(self, model_size: str = "350m") -> Any:
        self.model, self.tokenizer = self.model_manager.load_model()
        
        num_params = sum(p.numel() for p in self.model.parameters())
        print(f"Model loaded via ModelManager. Parameters: {num_params:,}")

        return self.model

    def pretrain(
        self,
        data_path: str,
        epochs: int = 3,
        batch_size: int = 8,
        learning_rate: float = 3e-4,
        save_path: Optional[str] = None,
        checkpoint_steps: int = 1000,
    ) -> List[float]:
        if self.model is None:
            self.setup_model()

        if self.tokenizer is None:
            raise ValueError("Tokenizer not initialized")

        dataset = CodeDataset(data_path, self.tokenizer)
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            collate_fn=collate_fn,
            num_workers=4,
            pin_memory=True,
        )

        optimizer = AdamW(self.model.parameters(), lr=learning_rate, weight_decay=0.01)

        total_steps = epochs * len(dataloader)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=total_steps
        )

        history = []
        global_step = 0

        self.model.train()

        for epoch in range(epochs):
            epoch_loss = 0.0

            progress_bar = tqdm(dataloader, desc=f"Epoch {epoch + 1}/{epochs}")

            for batch in progress_bar:
                batch = {k: v.to(self.device) for k, v in batch.items()}

                optimizer.zero_grad()

                if self.device == "cuda" and self.scaler:
                    with autocast():
                        outputs = self.model(
                            input_ids=batch["input_ids"],
                            attention_mask=batch["attention_mask"],
                            labels=batch["labels"],
                        )
                        loss = outputs["loss"]

                    self.scaler.scale(loss).backward()
                    self.scaler.step(optimizer)
                    self.scaler.update()
                else:
                    outputs = self.model(
                        input_ids=batch["input_ids"],
                        attention_mask=batch["attention_mask"],
                        labels=batch["labels"],
                    )
                    loss = outputs["loss"]
                    loss.backward()
                    optimizer.step()

                scheduler.step()

                epoch_loss += loss.item()
                global_step += 1

                progress_bar.set_postfix({"loss": f"{loss.item():.4f}"})

                if save_path and global_step % checkpoint_steps == 0:
                    self.save_checkpoint(save_path, global_step)

            avg_loss = epoch_loss / len(dataloader)
            history.append(avg_loss)
            print(f"Epoch {epoch + 1} completed. Average loss: {avg_loss:.4f}")

        if save_path:
            self.save_checkpoint(save_path, global_step, is_final=True)

        return history

    def meta_train(
        self, data_path: str, iterations: int = 1000, save_path: Optional[str] = None
    ) -> List[Dict[str, float]]:
        if self.model is None:
            self.setup_model()

        if self.tokenizer is None:
            raise ValueError("Tokenizer not initialized")

        maml_config = MAMLConfig(
            inner_lr=self.config.get("training", {})
            .get("meta_learning", {})
            .get("inner_lr", 1e-3),
            outer_lr=self.config.get("training", {})
            .get("meta_learning", {})
            .get("outer_lr", 1e-4),
            inner_steps=self.config.get("training", {})
            .get("meta_learning", {})
            .get("inner_steps", 5),
            meta_batch_size=self.config.get("training", {})
            .get("meta_learning", {})
            .get("meta_batch_size", 8),
        )

        maml_trainer = MAMLTrainer(self.model, self.tokenizer, maml_config, self.device)

        task_sampler = self._create_task_sampler(data_path)

        history = maml_trainer.train(
            task_sampler, num_iterations=iterations, save_path=save_path
        )

        return history

    def _create_task_sampler(self, data_path: str) -> TaskSampler:
        from src.core.training.meta_learning import create_task_sampler_from_data

        return create_task_sampler_from_data(data_path, support_size=5, query_size=15)

    def save_checkpoint(self, path: str, step: int, is_final: bool = False) -> None:
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        model_config = {
            "vocab_size": self.model.config.vocab_size,
            "hidden_size": self.model.config.hidden_size,
            "num_hidden_layers": self.model.config.num_hidden_layers,
            "num_attention_heads": self.model.config.num_attention_heads,
            "intermediate_size": self.model.config.intermediate_size,
            "max_position_embeddings": self.model.config.max_position_embeddings,
        }
        checkpoint = {
            "step": step,
            "model_state_dict": self.model.state_dict(),
            "config": model_config,
            "trainer_config": self.config,
        }
        metadata = build_checkpoint_metadata(
            model_config=model_config,
            tokenizer=self.tokenizer,
            tokenizer_type="code_tokenizer",
        )
        checkpoint = attach_checkpoint_metadata(checkpoint, metadata)

        filename = "model_final.pt" if is_final else f"checkpoint_{step}.pt"
        torch.save(checkpoint, path / filename)

        print(f"Checkpoint saved: {path / filename}")

    def load_checkpoint(self, path: str) -> None:
        checkpoint = torch.load(path, map_location=self.device)

        if self.model is None:
            self.setup_model()

        self.model.load_state_dict(checkpoint["model_state_dict"])
        print(f"Checkpoint loaded: {path}")


def train_codemind(
    data_path: str,
    output_path: str,
    model_size: str = "350m",
    epochs: int = 3,
    meta_iterations: int = 500,
    tokenizer_files: Optional[List[str]] = None,
) -> None:
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    trainer = CodeMindTrainer()

    print("=" * 50)
    print("Step 1: Training Tokenizer")
    print("=" * 50)
    trainer.setup_tokenizer(
        training_files=tokenizer_files, save_path=str(output_path / "tokenizer")
    )

    print("\n" + "=" * 50)
    print("Step 2: Pre-training Model")
    print("=" * 50)
    trainer.pretrain(
        data_path, epochs=epochs, save_path=str(output_path / "checkpoints")
    )

    print("\n" + "=" * 50)
    print("Step 3: Meta-learning Fine-tuning")
    print("=" * 50)
    trainer.meta_train(
        data_path,
        iterations=meta_iterations,
        save_path=str(output_path / "meta_checkpoints"),
    )

    print("\n" + "=" * 50)
    print("Training Complete!")
    print("=" * 50)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train CodeMind model")
    parser.add_argument("--data", type=str, required=True, help="Path to training data")
    parser.add_argument(
        "--output", type=str, default="checkpoints", help="Output directory"
    )
    parser.add_argument(
        "--model-size", type=str, default="350m", choices=["125m", "350m"]
    )
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--meta-iterations", type=int, default=500)
    parser.add_argument(
        "--tokenizer-files", nargs="+", help="Files for tokenizer training"
    )

    args = parser.parse_args()

    train_codemind(
        data_path=args.data,
        output_path=args.output,
        model_size=args.model_size,
        epochs=args.epochs,
        meta_iterations=args.meta_iterations,
        tokenizer_files=args.tokenizer_files,
    )
