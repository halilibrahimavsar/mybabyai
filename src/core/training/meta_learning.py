"""
Meta-Learning Module for CodeMind

Implements MAML (Model-Agnostic Meta-Learning) for few-shot learning
Code patterns can be learned with just 5-10 examples
"""

import os
import random
import copy
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Callable
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm


@dataclass
class MAMLConfig:
    inner_lr: float = 1e-3
    outer_lr: float = 1e-4
    inner_steps: int = 5
    meta_batch_size: int = 8
    support_size: int = 5
    query_size: int = 15
    first_order: bool = False
    gradient_clipping: float = 1.0


class CodeTask:
    def __init__(
        self,
        name: str,
        language: str,
        support_data: List[Dict[str, str]],
        query_data: List[Dict[str, str]],
    ):
        self.name = name
        self.language = language
        self.support_data = support_data
        self.query_data = query_data

    def __repr__(self) -> str:
        return f"CodeTask({self.name}, lang={self.language}, support={len(self.support_data)}, query={len(self.query_data)})"


class TaskSampler:
    def __init__(
        self,
        data_by_pattern: Dict[str, List[Dict[str, str]]],
        support_size: int = 5,
        query_size: int = 15,
    ):
        self.data_by_pattern = data_by_pattern
        self.support_size = support_size
        self.query_size = query_size
        self.patterns = list(data_by_pattern.keys())

    def sample_task(self, pattern: Optional[str] = None) -> CodeTask:
        if pattern is None:
            pattern = random.choice(self.patterns)

        data = self.data_by_pattern[pattern]

        if len(data) < self.support_size + self.query_size:
            raise ValueError(f"Not enough data for pattern {pattern}")

        sampled = random.sample(data, self.support_size + self.query_size)

        return CodeTask(
            name=pattern,
            language=sampled[0].get("language", "python"),
            support_data=sampled[: self.support_size],
            query_data=sampled[self.support_size :],
        )

    def sample_batch(self, batch_size: int) -> List[CodeTask]:
        return [self.sample_task() for _ in range(batch_size)]


class MAMLTrainer:
    def __init__(
        self, model: nn.Module, tokenizer: Any, config: MAMLConfig, device: str = "auto"
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.config = config

        if device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        self.model = self.model.to(self.device)

        self.meta_optimizer = AdamW(self.model.parameters(), lr=config.outer_lr)

        self.loss_fn = nn.CrossEntropyLoss(ignore_index=-100)

    def _prepare_batch(
        self, data: List[Dict[str, str]], max_length: int = 512
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        input_ids_list = []
        labels_list = []

        for item in data:
            prompt = item.get("user", item.get("prompt", ""))
            response = item.get("assistant", item.get("output", ""))

            full_text = f"<|code|>{prompt}\n<|answer|>{response}<|eos|>"

            tokens = self.tokenizer.encode(full_text, add_special_tokens=False)

            if len(tokens) > max_length:
                tokens = tokens[:max_length]

            input_ids = tokens[:-1]
            labels = tokens[1:]

            prompt_tokens = self.tokenizer.encode(
                f"<|code|>{prompt}\n<|answer|>", add_special_tokens=False
            )
            prompt_len = len(prompt_tokens) - 1

            labels = [-100] * prompt_len + labels[prompt_len:]

            input_ids_list.append(input_ids)
            labels_list.append(labels)

        max_len = max(len(ids) for ids in input_ids_list)

        padded_inputs = []
        padded_labels = []

        for ids, lbls in zip(input_ids_list, labels_list):
            pad_len = max_len - len(ids)
            padded_inputs.append(ids + [0] * pad_len)
            padded_labels.append(lbls + [-100] * pad_len)

        return (
            torch.tensor(padded_inputs, device=self.device),
            torch.tensor(padded_labels, device=self.device),
        )

    def _inner_loop(
        self, task: CodeTask, model_copy: nn.Module
    ) -> Tuple[nn.Module, torch.Tensor]:
        inner_optimizer = AdamW(model_copy.parameters(), lr=self.config.inner_lr)

        support_inputs, support_labels = self._prepare_batch(task.support_data)

        for _ in range(self.config.inner_steps):
            outputs = model_copy(support_inputs, labels=support_labels)
            loss = outputs["loss"]

            inner_optimizer.zero_grad()
            loss.backward()

            if self.config.gradient_clipping > 0:
                torch.nn.utils.clip_grad_norm_(
                    model_copy.parameters(), self.config.gradient_clipping
                )

            inner_optimizer.step()

        query_inputs, query_labels = self._prepare_batch(task.query_data)
        outputs = model_copy(query_inputs, labels=query_labels)
        query_loss = outputs["loss"]

        return model_copy, query_loss

    def meta_train_step(self, tasks: List[CodeTask]) -> Dict[str, float]:
        self.meta_optimizer.zero_grad()

        total_loss = 0.0
        task_losses = []

        for task in tasks:
            model_copy = copy.deepcopy(self.model)
            model_copy, query_loss = self._inner_loop(task, model_copy)

            if not self.config.first_order:
                query_loss.backward()

            task_losses.append(query_loss.item())
            total_loss += query_loss.item()

            del model_copy

        if self.config.first_order:
            avg_loss = total_loss / len(tasks)
            avg_loss_tensor = torch.tensor(
                avg_loss, requires_grad=True, device=self.device
            )
            avg_loss_tensor.backward()
        else:
            for param in self.model.parameters():
                if param.grad is not None:
                    param.grad = param.grad / len(tasks)

        if self.config.gradient_clipping > 0:
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), self.config.gradient_clipping
            )

        self.meta_optimizer.step()

        return {"meta_loss": total_loss / len(tasks), "task_losses": task_losses}

    def adapt(
        self, support_data: List[Dict[str, str]], num_steps: Optional[int] = None
    ) -> nn.Module:
        adapted_model = copy.deepcopy(self.model)
        optimizer = AdamW(adapted_model.parameters(), lr=self.config.inner_lr)

        steps = num_steps or self.config.inner_steps * 2

        support_inputs, support_labels = self._prepare_batch(support_data)

        adapted_model.train()
        for _ in range(steps):
            outputs = adapted_model(support_inputs, labels=support_labels)
            loss = outputs["loss"]

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        return adapted_model

    def train(
        self,
        task_sampler: TaskSampler,
        num_iterations: int = 1000,
        save_path: Optional[str] = None,
        eval_callback: Optional[Callable] = None,
    ) -> List[Dict[str, float]]:
        history = []

        for iteration in tqdm(range(num_iterations), desc="Meta-training"):
            tasks = task_sampler.sample_batch(self.config.meta_batch_size)

            metrics = self.meta_train_step(tasks)
            history.append(metrics)

            if (iteration + 1) % 10 == 0:
                print(
                    f"Iteration {iteration + 1}: Meta Loss = {metrics['meta_loss']:.4f}"
                )

            if save_path and (iteration + 1) % 100 == 0:
                self.save_checkpoint(save_path, iteration)

            if eval_callback and (iteration + 1) % 50 == 0:
                eval_callback(self, iteration)

        return history

    def save_checkpoint(self, path: str, iteration: int) -> None:
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        checkpoint = {
            "iteration": iteration,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.meta_optimizer.state_dict(),
            "config": self.config.__dict__,
        }

        torch.save(checkpoint, path / f"checkpoint_{iteration}.pt")

    def load_checkpoint(self, path: str) -> None:
        checkpoint = torch.load(path, map_location=self.device)

        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.meta_optimizer.load_state_dict(checkpoint["optimizer_state_dict"])


class FewShotEvaluator:
    def __init__(self, model: nn.Module, tokenizer: Any, device: str = "auto"):
        self.model = model
        self.tokenizer = tokenizer

        if device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        self.model = self.model.to(self.device)

    def evaluate_task(
        self, task: CodeTask, maml_trainer: Optional[MAMLTrainer] = None
    ) -> Dict[str, float]:
        if maml_trainer:
            adapted_model = maml_trainer.adapt(task.support_data)
        else:
            adapted_model = self.model

        adapted_model.eval()

        correct = 0
        total = 0

        with torch.no_grad():
            for item in task.query_data:
                prompt = item.get("user", item.get("prompt", ""))
                expected = item.get("assistant", item.get("output", ""))

                input_ids = self.tokenizer.encode(
                    f"<|code|>{prompt}\n<|answer|>", add_special_tokens=False
                )
                input_tensor = torch.tensor([input_ids], device=self.device)

                output_ids = adapted_model.generate(
                    input_tensor, max_new_tokens=256, temperature=0.1, do_sample=False
                )

                generated = self.tokenizer.decode(
                    output_ids[0].tolist(), skip_special_tokens=True
                )

                if expected.strip() in generated:
                    correct += 1
                total += 1

        accuracy = correct / total if total > 0 else 0.0

        return {"accuracy": accuracy, "correct": correct, "total": total}


def create_task_sampler_from_data(
    data_path: str, support_size: int = 5, query_size: int = 15
) -> TaskSampler:
    import json

    data_by_pattern: Dict[str, List[Dict[str, str]]] = {}

    path = Path(data_path)

    if path.is_file():
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        for item in data:
            pattern = item.get("pattern", "general")
            if pattern not in data_by_pattern:
                data_by_pattern[pattern] = []
            data_by_pattern[pattern].append(item)

    else:
        for file_path in path.glob("**/*.json"):
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            pattern = file_path.stem
            data_by_pattern[pattern] = data

    return TaskSampler(data_by_pattern, support_size, query_size)
