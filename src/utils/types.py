"""
Type Definitions Module

Centralized type definitions for the entire codebase.
Provides type safety and better IDE support.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import (
    Any,
    Callable,
    Dict,
    Generator,
    Generic,
    List,
    Optional,
    Protocol,
    Tuple,
    TypeVar,
    Union,
    runtime_checkable,
)
import torch
from torch import nn


class DeviceType(str, Enum):
    """Available device types for computation."""

    CUDA = "cuda"
    CPU = "cpu"
    MPS = "mps"
    AUTO = "auto"


class ModelType(str, Enum):
    """Supported model types."""

    CODEMIND = "codemind"
    HUGGINGFACE = "huggingface"
    OPENAI = "openai"


class LanguageType(str, Enum):
    """Supported programming languages."""

    PYTHON = "python"
    DART = "dart"
    JAVASCRIPT = "javascript"
    TYPESCRIPT = "typescript"
    GENERAL = "general"


class TrainingMode(str, Enum):
    """Training modes."""

    PRETRAIN = "pretrain"
    FINETUNE = "finetune"
    META_LEARNING = "meta_learning"
    DISTILLATION = "distillation"


class MessageRole(str, Enum):
    """Message roles in conversation."""

    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"


@dataclass
class ModelConfig:
    """Configuration for model loading."""

    name: str
    model_type: ModelType
    device: DeviceType = DeviceType.AUTO
    dtype: str = "float16"
    load_in_4bit: bool = False
    max_memory: str = "8GB"
    trust_remote_code: bool = True

    def __post_init__(self):
        if isinstance(self.device, str):
            self.device = DeviceType(self.device)
        if isinstance(self.model_type, str):
            self.model_type = ModelType(self.model_type)


@dataclass
class LoRAConfig:
    """Configuration for LoRA fine-tuning."""

    r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    target_modules: List[str] = field(
        default_factory=lambda: ["q_proj", "v_proj", "k_proj", "o_proj"]
    )
    bias: str = "none"

    def to_peft_config(self):
        """Convert to PEFT LoraConfig."""
        from peft import LoraConfig as PeftLoraConfig

        return PeftLoraConfig(
            r=self.r,
            lora_alpha=self.lora_alpha,
            lora_dropout=self.lora_dropout,
            target_modules=self.target_modules,
            bias=self.bias,
            task_type="CAUSAL_LM",
        )


@dataclass
class GenerationConfig:
    """Configuration for text generation."""

    max_new_tokens: int = 512
    temperature: float = 0.7
    top_p: float = 0.95
    top_k: int = 50
    repetition_penalty: float = 1.1
    do_sample: bool = True

    def to_dict(self) -> Dict[str, Any]:
        return {
            "max_new_tokens": self.max_new_tokens,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "top_k": self.top_k,
            "repetition_penalty": self.repetition_penalty,
            "do_sample": self.do_sample,
        }


@dataclass
class TrainingConfig:
    """Configuration for model training."""

    output_dir: str = "models/fine_tuned"
    num_epochs: int = 3
    batch_size: int = 4
    gradient_accumulation_steps: int = 4
    learning_rate: float = 2e-4
    warmup_steps: int = 100
    logging_steps: int = 10
    save_steps: int = 500
    max_length: int = 2048

    def to_dict(self) -> Dict[str, Any]:
        return {
            "output_dir": self.output_dir,
            "num_epochs": self.num_epochs,
            "batch_size": self.batch_size,
            "gradient_accumulation_steps": self.gradient_accumulation_steps,
            "learning_rate": self.learning_rate,
            "warmup_steps": self.warmup_steps,
            "logging_steps": self.logging_steps,
            "save_steps": self.save_steps,
            "max_length": self.max_length,
        }


@dataclass
class Message:
    """Single message in a conversation."""

    role: MessageRole
    content: str
    language: Optional[LanguageType] = None

    def to_dict(self) -> Dict[str, str]:
        return {
            "role": self.role.value,
            "content": self.content,
            "language": self.language.value if self.language else None,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Message":
        return cls(
            role=MessageRole(data.get("role", "user")),
            content=data.get("content", ""),
            language=LanguageType(data["language"]) if data.get("language") else None,
        )


@dataclass
class Conversation:
    """A conversation with history."""

    id: Optional[int] = None
    title: str = ""
    messages: List[Message] = field(default_factory=list)

    def add_message(self, role: MessageRole, content: str) -> Message:
        message = Message(role=role, content=content)
        self.messages.append(message)
        return message

    def get_history(self) -> List[Dict[str, str]]:
        history = []
        for i in range(0, len(self.messages) - 1, 2):
            if i + 1 < len(self.messages):
                history.append(
                    {
                        "user": self.messages[i].content,
                        "assistant": self.messages[i + 1].content,
                    }
                )
        return history


@dataclass
class ModelInfo:
    """Information about a loaded model."""

    name: str
    model_type: ModelType
    device: str
    total_parameters: int
    trainable_parameters: int
    quantized: bool = False
    vocab_size: int = 0

    @property
    def trainable_percentage(self) -> float:
        return (
            100 * self.trainable_parameters / self.total_parameters
            if self.total_parameters > 0
            else 0
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "model_type": self.model_type.value,
            "device": self.device,
            "total_parameters": self.total_parameters,
            "trainable_parameters": self.trainable_parameters,
            "trainable_percentage": f"{self.trainable_percentage:.2f}%",
            "quantized": self.quantized,
            "vocab_size": self.vocab_size,
        }


@runtime_checkable
class TokenizerProtocol(Protocol):
    """Protocol for tokenizer interface."""

    def encode(self, text: str, **kwargs) -> List[int]: ...
    def decode(self, ids: List[int], **kwargs) -> str: ...
    @property
    def vocab_size(self) -> int: ...
    @property
    def eos_token_id(self) -> int: ...
    @property
    def pad_token_id(self) -> int: ...


@runtime_checkable
class ModelProtocol(Protocol):
    """Protocol for model interface."""

    def __call__(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]: ...

    def generate(self, input_ids: torch.Tensor, **kwargs) -> torch.Tensor: ...

    def parameters(self): ...

    def train(self, mode: bool = True): ...

    def eval(self): ...

    def to(self, device: str): ...


T = TypeVar("T")


class Result(Generic[T]):
    """
    Result type for better error handling.
    Similar to Rust's Result<T, E>.
    """

    def __init__(
        self,
        value: Optional[T] = None,
        error: Optional[Exception] = None,
    ):
        self._value = value
        self._error = error

    @classmethod
    def ok(cls, value: T) -> "Result[T]":
        return cls(value=value)

    @classmethod
    def err(cls, error: Exception) -> "Result[T]":
        return cls(error=error)

    @property
    def is_ok(self) -> bool:
        return self._error is None

    @property
    def is_err(self) -> bool:
        return self._error is not None

    def unwrap(self) -> T:
        if self._error:
            raise self._error
        return self._value

    def unwrap_or(self, default: T) -> T:
        if self._error:
            return default
        return self._value

    def map(self, f: Callable[[T], Any]) -> "Result":
        if self._error:
            return Result.err(self._error)
        try:
            return Result.ok(f(self._value))
        except Exception as e:
            return Result.err(e)


TokenIds = List[int]
TokenIdsTensor = torch.Tensor
Text = str
Probability = float

ModelOutput = Dict[str, torch.Tensor]
GenerationOutput = Tuple[str, Dict[str, Any]]

StreamGenerator = Generator[str, None, str]
BatchGenerator = Generator[List[str], None, List[str]]
