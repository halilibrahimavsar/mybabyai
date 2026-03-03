import os
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import pandas as pd

import sys


from src.utils.logger import get_logger


class DatasetLoader:
    SUPPORTED_FORMATS = [".json", ".jsonl", ".csv", ".txt", ".md"]

    def __init__(self):
        self.logger = get_logger("dataset_loader")

    def load_from_file(self, filepath: str) -> List[Dict[str, str]]:
        path = Path(filepath)

        if not path.exists():
            raise FileNotFoundError(f"Dosya bulunamadı: {filepath}")

        suffix = path.suffix.lower()

        if suffix == ".json":
            return self._load_json(filepath)
        elif suffix == ".jsonl":
            return self._load_jsonl(filepath)
        elif suffix == ".csv":
            return self._load_csv(filepath)
        elif suffix in [".txt", ".md"]:
            return self._load_text(filepath)
        else:
            raise ValueError(f"Desteklenmeyen dosya formatı: {suffix}")

    def _load_json(self, filepath: str) -> List[Dict[str, str]]:
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)

        if isinstance(data, list):
            return self._normalize_conversations(data)
        elif isinstance(data, dict):
            return self._normalize_conversations([data])

        return []

    def _load_jsonl(self, filepath: str) -> List[Dict[str, str]]:
        conversations = []
        with open(filepath, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        item = json.loads(line)
                        conversations.append(item)
                    except json.JSONDecodeError:
                        continue

        return self._normalize_conversations(conversations)

    def _load_csv(self, filepath: str) -> List[Dict[str, str]]:
        df = pd.read_csv(filepath)

        column_mapping = self._detect_csv_columns(df.columns)

        conversations = []
        for _, row in df.iterrows():
            conv = {
                "user": str(row.get(column_mapping.get("user", "user"), "")),
                "assistant": str(
                    row.get(column_mapping.get("assistant", "assistant"), "")
                ),
            }
            if conv["user"] or conv["assistant"]:
                conversations.append(conv)

        return conversations

    def _detect_csv_columns(self, columns: List[str]) -> Dict[str, str]:
        mapping = {}
        columns_lower = [c.lower() for c in columns]

        user_aliases = [
            "user", "question", "input", "prompt", "human", "sorular", "soru", 
            "instruction", "context", "text", "query", "request"
        ]
        assistant_aliases = [
            "assistant", "answer", "output", "response", "ai", "cevaplar", "cevap",
            "completion", "target", "solution", "explanation"
        ]

        for alias in user_aliases:
            if alias in columns_lower:
                idx = columns_lower.index(alias)
                mapping["user"] = columns[idx]
                break

        for alias in assistant_aliases:
            if alias in columns_lower:
                idx = columns_lower.index(alias)
                mapping["assistant"] = columns[idx]
                break

        return mapping

    def _load_text(self, filepath: str) -> List[Dict[str, str]]:
        with open(filepath, "r", encoding="utf-8") as f:
            content = f.read()

        conversations = []
        sections = content.split("\n---\n")

        for section in sections:
            section = section.strip()
            if not section:
                continue

            lines = section.split("\n")
            user_text = []
            assistant_text = []
            current_role = None

            for line in lines:
                line_lower = line.lower().strip()
                if line_lower.startswith(("user:", "kullanıcı:", "q:", "soru:")):
                    current_role = "user"
                    line = line.split(":", 1)[1].strip() if ":" in line else line
                    user_text.append(line)
                elif line_lower.startswith(("assistant:", "asistan:", "a:", "cevap:")):
                    current_role = "assistant"
                    line = line.split(":", 1)[1].strip() if ":" in line else line
                    assistant_text.append(line)
                elif current_role == "user":
                    user_text.append(line)
                elif current_role == "assistant":
                    assistant_text.append(line)

        if user_text or assistant_text:
            conversations.append(
                {"user": " ".join(user_text), "assistant": " ".join(assistant_text)}
            )
        else:
            # Fallback for raw text: Use a generic prompt
            conversations.append({
                "user": "Metni analiz et ve öğren.",
                "assistant": content.strip()
            })

        return conversations

    def _normalize_conversations(
        self, data: List[Dict[str, Any]]
    ) -> List[Dict[str, str]]:
        normalized = []

        for item in data:
            if "conversations" in item:
                convs = item["conversations"]
                user_msg = ""
                assistant_msg = ""

                for conv in convs:
                    role = conv.get("role", conv.get("from", "")).lower()
                    content = conv.get("content", conv.get("value", ""))

                    if role in ["user", "human"]:
                        user_msg = content
                    elif role in ["assistant", "gpt", "ai"]:
                        assistant_msg = content

                if user_msg or assistant_msg:
                    normalized.append({"user": user_msg, "assistant": assistant_msg})

            elif "messages" in item:
                messages = item["messages"]
                user_msg = ""
                assistant_msg = ""

                for msg in messages:
                    role = msg.get("role", "").lower()
                    content = msg.get("content", "")

                    if role == "user":
                        user_msg = content
                    elif role == "assistant":
                        assistant_msg = content

                if user_msg or assistant_msg:
                    normalized.append({"user": user_msg, "assistant": assistant_msg})

            elif "user" in item or "question" in item or "input" in item:
                user_msg = item.get("user", item.get("question", item.get("input", "")))
                assistant_msg = item.get(
                    "assistant", item.get("answer", item.get("output", ""))
                )

                if user_msg or assistant_msg:
                    normalized.append(
                        {"user": str(user_msg), "assistant": str(assistant_msg)}
                    )
            else:
                # If it's a dict but doesn't have standard keys, try to find any text fields
                potential_user = ""
                potential_assistant = ""
                for k, v in item.items():
                    if not isinstance(v, (str, bytes)):
                        continue
                    if any(x in k.lower() for x in ["question", "input", "prompt", "user", "instruction"]):
                        potential_user = v
                    elif any(x in k.lower() for x in ["answer", "output", "response", "assistant", "completion"]):
                        potential_assistant = v
                
                if potential_user or potential_assistant:
                    normalized.append({"user": str(potential_user), "assistant": str(potential_assistant)})

        return normalized

    def load_from_directory(
        self, directory: str, recursive: bool = True
    ) -> List[Dict[str, str]]:
        all_conversations = []
        dir_path = Path(directory)

        if not dir_path.exists():
            raise FileNotFoundError(f"Dizin bulunamadı: {directory}")

        pattern = "**/*" if recursive else "*"

        for filepath in dir_path.glob(pattern):
            if filepath.is_file() and filepath.suffix.lower() in self.SUPPORTED_FORMATS:
                try:
                    conversations = self.load_from_file(str(filepath))
                    all_conversations.extend(conversations)
                    self.logger.info(
                        f"Yüklendi: {filepath} ({len(conversations)} konuşma)"
                    )
                except Exception as e:
                    self.logger.error(f"Dosya yükleme hatası {filepath}: {e}")

        return all_conversations

    def create_few_shot_dataset(
        self, examples: List[Dict[str, str]], instruction: str = ""
    ) -> List[Dict[str, str]]:
        formatted = []

        for ex in examples:
            user_text = ex.get("user", "")
            assistant_text = ex.get("assistant", "")

            if instruction:
                user_text = f"{instruction}\n\n{user_text}"

            formatted.append({"user": user_text, "assistant": assistant_text})

        return formatted

    def split_dataset(
        self,
        conversations: List[Dict[str, str]],
        train_ratio: float = 0.8,
        val_ratio: float = 0.1,
        test_ratio: float = 0.1,
    ) -> Dict[str, List[Dict[str, str]]]:
        import random

        random.shuffle(conversations)

        total = len(conversations)
        train_size = int(total * train_ratio)
        val_size = int(total * val_ratio)

        return {
            "train": conversations[:train_size],
            "validation": conversations[train_size : train_size + val_size],
            "test": conversations[train_size + val_size :],
        }

    def save_dataset(
        self,
        conversations: List[Dict[str, str]],
        output_path: str,
        format: str = "jsonl",
    ) -> None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        if format == "json":
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(conversations, f, ensure_ascii=False, indent=2)

        elif format == "jsonl":
            with open(output_path, "w", encoding="utf-8") as f:
                for conv in conversations:
                    f.write(json.dumps(conv, ensure_ascii=False) + "\n")

        elif format == "csv":
            df = pd.DataFrame(conversations)
            df.to_csv(output_path, index=False)

        self.logger.info(
            f"Dataset kaydedildi: {output_path} ({len(conversations)} konuşma)"
        )

    def get_dataset_stats(self, conversations: List[Dict[str, str]]) -> Dict[str, Any]:
        if not conversations:
            return {"total": 0}

        user_lengths = [len(c["user"].split()) for c in conversations]
        assistant_lengths = [len(c["assistant"].split()) for c in conversations]

        return {
            "total": len(conversations),
            "avg_user_length": sum(user_lengths) / len(user_lengths),
            "avg_assistant_length": sum(assistant_lengths) / len(assistant_lengths),
            "max_user_length": max(user_lengths),
            "max_assistant_length": max(assistant_lengths),
            "min_user_length": min(user_lengths),
            "min_assistant_length": min(assistant_lengths),
        }
