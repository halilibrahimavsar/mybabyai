from typing import List, Dict, Callable, Optional
from collections import deque
from torch.utils.data import Dataset
from src.core.prompting import build_instruction_prompt

class TextDataset(Dataset):
    def __init__(
        self,
        texts: List[str],
        tokenizer,
        max_length: int = 256,
        formatting_func: Optional[Callable] = None,
        pack_sequences: bool = True,
    ):
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.formatting_func = formatting_func or self._default_format
        self.pack_sequences = pack_sequences
        self.samples: List[Dict[str, List[int]]] = self._build_samples()

    def _default_format(self, text: str) -> str:
        return build_instruction_prompt(user=text, assistant=None, include_eos=False)

    def _get_eos_token_id(self) -> Optional[int]:
        eos_id = getattr(self.tokenizer, "eos_token_id", None)
        if isinstance(eos_id, int):
            return eos_id
        if hasattr(self.tokenizer, "token_to_id"):
            try:
                token_id = self.tokenizer.token_to_id("<|eos|>")
                if isinstance(token_id, int):
                    return token_id
            except Exception:
                return None
        return None

    def _encode_text(self, text: str) -> List[int]:
        encoded = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            padding=False,
            return_tensors=None,
        )
        input_ids = encoded.get("input_ids", [])
        if input_ids and isinstance(input_ids[0], list):
            return list(input_ids[0])
        return list(input_ids)

    def _build_samples(self) -> List[Dict[str, List[int]]]:
        eos_id = self._get_eos_token_id()
        samples: List[Dict[str, List[int]]] = []

        formatted_texts = [self.formatting_func(t) for t in self.texts]
        chunk_size = 5000

        if not self.pack_sequences:
            for i in range(0, len(formatted_texts), chunk_size):
                batch_texts = formatted_texts[i:i+chunk_size]
                encoded = self.tokenizer(
                    batch_texts,
                    truncation=True,
                    max_length=self.max_length,
                    padding=False,
                    return_attention_mask=False,
                )
                for ids in encoded.get("input_ids", []):
                    if len(ids) >= 2:
                        samples.append({"input_ids": ids, "attention_mask": [1] * len(ids)})
            return samples

        token_buffer: deque[int] = deque()
        for i in range(0, len(formatted_texts), chunk_size):
            batch_texts = formatted_texts[i:i+chunk_size]
            encoded = self.tokenizer(
                batch_texts,
                truncation=False,
                padding=False,
                return_attention_mask=False,
            )
            for ids in encoded.get("input_ids", []):
                if not ids:
                    continue
                if eos_id is not None and ids[-1] != eos_id:
                    ids.append(eos_id)

                token_buffer.extend(ids)
                while len(token_buffer) >= self.max_length:
                    chunk = [token_buffer.popleft() for _ in range(self.max_length)]
                    samples.append(
                        {"input_ids": chunk, "attention_mask": [1] * self.max_length}
                    )

        if len(token_buffer) > 1:
            remainder = list(token_buffer)
            samples.append(
                {"input_ids": remainder, "attention_mask": [1] * len(remainder)}
            )

        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, List[int]]:
        return self.samples[idx]


class ConversationDataset(Dataset):
    def __init__(
        self,
        conversations: List[Dict[str, str]],
        tokenizer,
        max_length: int = 256,
        pack_sequences: bool = True,
    ):
        self.conversations = conversations
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.pack_sequences = pack_sequences
        self.samples: List[Dict[str, List[int]]] = self._build_samples()

    def _get_eos_token_id(self) -> Optional[int]:
        eos_id = getattr(self.tokenizer, "eos_token_id", None)
        if isinstance(eos_id, int):
            return eos_id
        if hasattr(self.tokenizer, "token_to_id"):
            try:
                token_id = self.tokenizer.token_to_id("<|eos|>")
                if isinstance(token_id, int):
                    return token_id
            except Exception:
                return None
        return None

    def _encode_text(self, text: str) -> List[int]:
        encoded = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            padding=False,
            return_tensors=None,
        )
        input_ids = encoded.get("input_ids", [])
        if input_ids and isinstance(input_ids[0], list):
            return list(input_ids[0])
        return list(input_ids)

    def _build_samples(self) -> List[Dict[str, List[int]]]:
        eos_id = self._get_eos_token_id()
        samples: List[Dict[str, List[int]]] = []

        formatted_texts = [
            build_instruction_prompt(
                user=conv["user"],
                assistant=conv["assistant"],
                language="general",
                include_eos=False,
            ) for conv in self.conversations
        ]
        
        chunk_size = 5000

        if not self.pack_sequences:
            for i in range(0, len(formatted_texts), chunk_size):
                batch_texts = formatted_texts[i:i+chunk_size]
                encoded = self.tokenizer(
                    batch_texts,
                    truncation=True,
                    max_length=self.max_length,
                    padding=False,
                    return_attention_mask=False,
                )
                for ids in encoded.get("input_ids", []):
                    if len(ids) >= 2:
                        samples.append({"input_ids": ids, "attention_mask": [1] * len(ids)})
            return samples

        token_buffer: deque[int] = deque()
        for i in range(0, len(formatted_texts), chunk_size):
            batch_texts = formatted_texts[i:i+chunk_size]
            encoded = self.tokenizer(
                batch_texts,
                truncation=False,
                padding=False,
                return_attention_mask=False,
            )
            for ids in encoded.get("input_ids", []):
                if not ids:
                    continue
                if eos_id is not None and ids[-1] != eos_id:
                    ids.append(eos_id)

                token_buffer.extend(ids)
                while len(token_buffer) >= self.max_length:
                    chunk = [token_buffer.popleft() for _ in range(self.max_length)]
                    samples.append(
                        {"input_ids": chunk, "attention_mask": [1] * self.max_length}
                    )

        if len(token_buffer) > 1:
            remainder = list(token_buffer)
            samples.append(
                {"input_ids": remainder, "attention_mask": [1] * len(remainder)}
            )

        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, List[int]]:
        return self.samples[idx]
