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
        language: str = "tr",
    ):
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.formatting_func = formatting_func  # Store original formatting_func
        self.pack_sequences = pack_sequences
        self.language = language
        self.samples: List[Dict[str, List[int]]] = self._build_samples()

    def _get_formatted_text(self, text: str) -> str:
        if self.formatting_func:
            return self.formatting_func(text)
        return build_instruction_prompt(user=text, assistant=None, language=self.language, include_eos=False)

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

        formatted_texts = [self._get_formatted_text(t) for t in self.texts]
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
        language: str = "tr",
    ):
        self.conversations = conversations
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.pack_sequences = pack_sequences
        self.language = language
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
                language=self.language,
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

from typing import Iterator, Iterable
from torch.utils.data import IterableDataset

class StreamingConversationDataset(IterableDataset):
    """
    HuggingFace'in IterableDataset objesinden gelen (streaming=True)
    veya devasa boyutlu listelerden iteratör ile dönen konuşmaları,
    RAM'i doldurmadan anlık olarak tokenize edip döndürür.
    """
    def __init__(
        self,
        conversations_generator: Iterable[Dict[str, str]],
        tokenizer,
        max_length: int = 256,
        pack_sequences: bool = True,
        language: str = "tr",
    ):
        self.conversations_generator = conversations_generator
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.pack_sequences = pack_sequences
        self.language = language
        
        # eos ayarlaması
        self.eos_id = None
        eos_token = getattr(tokenizer, "eos_token_id", None)
        if isinstance(eos_token, int):
            self.eos_id = eos_token
        elif hasattr(tokenizer, "token_to_id"):
            try:
                t_id = tokenizer.token_to_id("<|eos|>")
                if isinstance(t_id, int):
                    self.eos_id = t_id
            except Exception:
                pass

    def _generate_samples(self) -> Iterator[Dict[str, List[int]]]:
        token_buffer = deque()
        
        # If conversations_generator is a factory function, call it to get a fresh generator
        conv_list = self.conversations_generator() if callable(self.conversations_generator) else self.conversations_generator
        
        for conv in conv_list:
            try:
                # 1. Prompt formatlama
                formatted_text = build_instruction_prompt(
                    user=conv.get("user", ""),
                    assistant=conv.get("assistant", ""),
                    language=self.language,
                    include_eos=False,
                )
                
                # 2. Tokenize (padding/truncation yapmıyoruz, buffer'a atacağız)
                encoded = self.tokenizer(
                    formatted_text,
                    truncation=False,
                    padding=False,
                    return_attention_mask=False,
                )
                
                input_ids = encoded.get("input_ids", [])
                if not input_ids:
                    continue
                    
                # EOS ekle
                if self.eos_id is not None and input_ids[-1] != self.eos_id:
                    input_ids.append(self.eos_id)
                
                if not self.pack_sequences:
                    # Sadece max_length kadar kesip gönder (packing yok)
                    ids = input_ids[:self.max_length]
                    if len(ids) >= 2:
                        yield {"input_ids": ids, "attention_mask": [1] * len(ids)}
                    continue

                # 3. Packing açık ise Buffer'a doldur ve max_length bloklar halinde çıkar
                token_buffer.extend(input_ids)
                
                while len(token_buffer) >= self.max_length:
                    chunk = [token_buffer.popleft() for _ in range(self.max_length)]
                    yield {"input_ids": chunk, "attention_mask": [1] * len(chunk)}
                    
            except Exception:
                continue

        # Kalan son chunk'ı (ufak olsa da) gönder
        if self.pack_sequences and len(token_buffer) > 1:
            remainder = list(token_buffer)
            # Opsiyonel: remainder = remainder + [pad_token] * (max_length - len)
            yield {"input_ids": remainder, "attention_mask": [1] * len(remainder)}

    def __iter__(self):
        return self._generate_samples()
