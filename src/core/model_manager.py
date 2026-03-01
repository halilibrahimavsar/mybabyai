import os
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    AutoConfig,
    BitsAndBytesConfig,
)
from peft import PeftModel, LoraConfig, get_peft_model

import sys


from src.utils.config import Config
from src.utils.logger import get_logger


class ModelManager:
    def __init__(self, config: Optional[Config] = None):
        self.config = config or Config()
        self.logger = get_logger("model_manager")

        self.model: Optional[Union[AutoModelForCausalLM, Any]] = None
        self.tokenizer: Optional[Union[AutoTokenizer, Any]] = None
        self.model_name: str = ""
        self.device: str = self._get_device()
        self.lora_config: Optional[LoraConfig] = None
        self.is_codemind: bool = False
        self.codemind_adapter: Optional[Any] = None
        self.is_fine_tuned: bool = False

        self._model_aliases = {
            "codemind": "CodeMind-125M",
            "codemind-125m": "CodeMind-125M",
            "codemind-125m (local)": "CodeMind-125M",
            "codemind local": "CodeMind-125M",
        }

    def _get_device(self) -> str:
        device_config = self.config.get("model.device", "auto")

        if device_config == "auto":
            if torch.cuda.is_available():
                return "cuda"
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                return "mps"
            else:
                return "cpu"
        return device_config

    def _get_quantization_config(self) -> Optional[BitsAndBytesConfig]:
        if not self.config.get("model.load_in_4bit", False):
            return None

        if self.device == "cpu":
            self.logger.warning(
                "4-bit quantization CUDA gerektirir, CPU modunda devre dışı"
            )
            return None

        return BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type=self.config.get(
                "training.qlora.bnb_4bit_quant_type", "nf4"
            ),
            bnb_4bit_compute_dtype=getattr(
                torch,
                self.config.get("training.qlora.bnb_4bit_compute_dtype", "float16"),
            ),
            bnb_4bit_use_double_quant=True,
        )

    def _load_codemind(self) -> Tuple[Any, Any]:
        from src.core.codemind_adapter import CodeMindAdapter

        self.logger.info("CodeMind model yükleniyor...")
        self.is_codemind = True

        self.codemind_adapter = CodeMindAdapter(self.config)
        self.model, self.tokenizer, _ = self.codemind_adapter.load_model()
        self.model_name = "CodeMind-125M"

        self.logger.info("CodeMind model başarıyla yüklendi")
        
        self._check_and_load_adapter()
        
        return self.model, self.tokenizer

    def _check_and_load_adapter(self) -> None:
        adapter_dir = self.config.get("training.output_dir", "models/fine_tuned")
        adapter_config = Path(adapter_dir) / "adapter_config.json"
        
        if adapter_config.exists():
            self.logger.info(f"Mevcut fine-tuning adapter tespit edildi: {adapter_dir}")
            try:
                self._load_lora_adapter(adapter_dir)
                self.is_fine_tuned = True
            except Exception as e:
                self.logger.error(f"Adapter yüklenirken hata oluştu: {e}")
                self.is_fine_tuned = False
        else:
            self.is_fine_tuned = False
            self.logger.info("Mevcut adapter bulunamadı. Model ham (base) modda çalışacak.")

    def resolve_model_name(self, model_name: Optional[str] = None) -> str:
        if model_name:
            # Check if there is an alias match
            alias = model_name.lower()
            if alias in self._model_aliases:
                return self._model_aliases[alias]
            return model_name
        
        # Check config
        config_name = self.config.get("model.name")
        if config_name:
            return config_name

        # Fallback to alias check or default
        return "CodeMind-125M"

    def load_model(
        self,
        model_name: Optional[str] = None,
        load_lora: bool = False,
        lora_path: Optional[str] = None,
    ) -> Tuple[Any, Any]:
        self.model_name = self.resolve_model_name(model_name)
        
        if "codemind" in self.model_name.lower():
            return self._load_codemind()
        
        # Generic HuggingFace Model Loading
        self.logger.info(f"HuggingFace model yükleniyor: {self.model_name}...")
        self.is_codemind = False
        
        quant_config = self._get_quantization_config()
        
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            quantization_config=quant_config,
            device_map="auto" if self.device != "cpu" else None,
            torch_dtype=torch.float16 if self.device != "cpu" else torch.float32,
        )
        
        if load_lora and lora_path:
            self._load_lora_adapter(lora_path)
            self.is_fine_tuned = True
        
        self.logger.info(f"Model başarıyla yüklendi: {self.model_name}")
        return self.model, self.tokenizer

    def _setup_lora(self) -> None:
        lora_config = self.config.get("model.lora", {})

        self.lora_config = LoraConfig(
            r=lora_config.get("r", 16),
            lora_alpha=lora_config.get("lora_alpha", 32),
            lora_dropout=lora_config.get("lora_dropout", 0.05),
            target_modules=lora_config.get("target_modules", ["q_proj", "v_proj"]),
            bias="none",
            task_type="CAUSAL_LM",
        )

        self.model = get_peft_model(self.model, self.lora_config)
        self.logger.info("LoRA adapterları uygulandı")

    def _load_lora_adapter(self, lora_path: str) -> None:
        if not os.path.exists(lora_path):
            self.logger.warning(f"LoRA yolu bulunamadı: {lora_path}")
            return

        self.model = PeftModel.from_pretrained(self.model, lora_path)
        self.logger.info(f"LoRA adapter yüklendi: {lora_path}")

    def save_model(self, output_path: str, save_tokenizer: bool = True) -> None:
        if self.model is None:
            raise ValueError("Kaydedilecek model yok")

        output_path = Path(output_path)
        output_path.mkdir(parents=True, exist_ok=True)

        self.model.save_pretrained(output_path)

        if save_tokenizer and self.tokenizer:
            self.tokenizer.save_pretrained(output_path)

        self.logger.info(f"Model kaydedildi: {output_path}")

    def get_model_info(self) -> Dict[str, Any]:
        if self.model is None:
            return {"status": "not_loaded"}

        info = {}
        if self.is_codemind and self.codemind_adapter:
            info = self.codemind_adapter.get_model_info()
        else:
            total_params = sum(p.numel() for p in self.model.parameters())
            trainable_params = sum(
                p.numel() for p in self.model.parameters() if p.requires_grad
            )
            info = {
                "model_name": self.model_name,
                "device": self.device,
                "total_parameters": total_params,
                "trainable_parameters": trainable_params,
                "trainable_percentage": f"{100 * trainable_params / total_params:.2f}%",
                "quantized": self.config.get("model.load_in_4bit", False),
            }

        info["status"] = "Eğitilmiş (Fine-tuned)" if getattr(self, "is_fine_tuned", False) else "Ham (Base)"
        return info

    def unload_model(self) -> None:
        if self.is_codemind and self.codemind_adapter:
            self.codemind_adapter.unload_model()
            self.codemind_adapter = None
        else:
            if self.model is not None:
                del self.model
                self.model = None

            if self.tokenizer is not None:
                del self.tokenizer
                self.tokenizer = None

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        self.logger.info("Model bellekten temizlendi")
