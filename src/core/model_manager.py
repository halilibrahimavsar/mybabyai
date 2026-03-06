import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
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

        # Always validate CUDA is actually available, regardless of config.
        # This prevents an AssertionError crash in environments like Colab
        # CPU runtimes where CUDA is not compiled into the Torch installation.
        cuda_available = torch.cuda.is_available()
        mps_available = hasattr(torch.backends, "mps") and torch.backends.mps.is_available()

        if device_config == "auto":
            if cuda_available:
                return "cuda"
            elif mps_available:
                return "mps"
            else:
                return "cpu"

        # Explicit config override — but still validate
        if device_config == "cuda" and not cuda_available:
            self.logger.warning(
                "Config 'cuda' olarak ayarlandı fakat CUDA mevcut değil. CPU'ya geçiliyor."
            )
            return "cpu"
        if device_config == "mps" and not mps_available:
            self.logger.warning(
                "Config 'mps' olarak ayarlandı fakat MPS mevcut değil. CPU'ya geçiliyor."
            )
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

    def _load_codemind(
        self, checkpoint_path: Optional[str] = None, allow_fresh_fallback: bool = False
    ) -> Tuple[Any, Any]:
        from src.core.codemind_adapter import CodeMindAdapter

        self.logger.info("CodeMind model yükleniyor...")
        self.is_codemind = True

        self.codemind_adapter = CodeMindAdapter(self.config)
        try:
            self.model, self.tokenizer, report = self.codemind_adapter.load_model(
                checkpoint_path
            )

            # If a path was provided, use the filename as model name
            if checkpoint_path:
                self.model_name = Path(checkpoint_path).name
            else:
                self.model_name = "CodeMind-125M"
        except FileNotFoundError as e:
            if allow_fresh_fallback:
                self.logger.warning(
                    f"Checkpoint bulunamadı ({e}). Sıfır (eğitilmemiş) model ile devam ediliyor..."
                )
                return self.load_fresh_model()
            raise e

        self.logger.info(f"CodeMind model başarıyla yüklendi: {self.model_name}")

        self._check_and_load_adapter()

        return self.model, self.tokenizer

    def _check_and_load_adapter(self) -> None:
        adapter_dir = self.config.get("training.output_dir", "models/fine_tuned")
        adapter_config = Path(adapter_dir) / "adapter_config.json"
        
        if adapter_config.exists():
            self.logger.info(f"Mevcut fine-tuning adapter tespit edildi: {adapter_dir}")
            try:
                import json
                with open(adapter_config, "r", encoding="utf-8") as f:
                    config_data = json.load(f)
                
                # Check base_model_name_or_path if possible to prevent catastrophic mismatches
                base_name = config_data.get("base_model_name_or_path", "").lower()
                if self.is_codemind and base_name and "codemind" not in base_name and "model" not in base_name and not base_name.endswith(".pt"):
                    self.logger.warning(f"Adapter ({base_name}) CodeMind modeli ile uyumlu görünmüyor. Yüklenmeyecek.")
                    self.is_fine_tuned = False
                    return
                elif not self.is_codemind and base_name and "codemind" in base_name:
                    self.logger.warning(f"Adapter (CodeMind) HF modeli ile uyumlu görünmüyor. Yüklenmeyecek.")
                    self.is_fine_tuned = False
                    return

                # Check if target_modules match the base model
                target_modules = config_data.get("target_modules", [])
                if isinstance(target_modules, list) and len(target_modules) > 0 and self.model is not None:
                    model_modules = [name for name, _ in self.model.named_modules()]
                    matches_found = False
                    for target in target_modules:
                        if any(target in name or name.endswith(target) for name in model_modules):
                            matches_found = True
                            break
                    if not matches_found:
                        self.logger.warning(f"Adapter hedef katmanları {target_modules} modelde bulunamadı. Uyumsuz adapter yoksayılıyor.")
                        self.is_fine_tuned = False
                        return

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
            # Check if it's a direct path to a .pt file
            if model_name.endswith(".pt") and os.path.exists(model_name):
                return model_name

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
        allow_fresh_fallback: bool = False,
    ) -> Tuple[Any, Any]:
        self.model_name = self.resolve_model_name(model_name)

        # Handle CodeMind specifically
        if "codemind" in self.model_name.lower() or self.model_name.endswith(".pt"):
            checkpoint_path = self.model_name if self.model_name.endswith(".pt") else None
            return self._load_codemind(
                checkpoint_path, allow_fresh_fallback=allow_fresh_fallback
            )
        
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
        
        # Ensure padding token is set (for models like Phi-2)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.model.config.pad_token_id = self.tokenizer.pad_token_id
        
        if load_lora and lora_path:
            self._load_lora_adapter(lora_path)
            self.is_fine_tuned = True
        
        self.logger.info(f"Model başarıyla yüklendi: {self.model_name}")
        return self.model, self.tokenizer

    def load_fresh_model(self, size: str = "125M") -> Tuple[Any, Any]:
        """Loads a completely untrained CodeMind model from scratch.
        
        Args:
            size: Model size configuration ('125M', '350M', '650M')
        """
        from src.core.codemind_adapter import CodeMindAdapter
        from src.core.tokenizer.code_tokenizer import CodeTokenizer
        from src.core.model.codemind import CodeMindConfig, CodeMindForCausalLM
        
        size = size.upper()
        self.logger.info(f"Sıfırdan (eğitilmemiş) CodeMind-{size} model oluşturuluyor...")
        
        # Architecture presets
        experts = 1
        experts_per_tok = 1
        
        if size == "350M":
            hidden_size, layers, heads = 1024, 24, 16
        elif size == "350M-MOE":
            hidden_size, layers, heads = 448, 6, 8
            experts = 4
            experts_per_tok = 2
        elif size == "650M":
            hidden_size, layers, heads = 1280, 24, 20
        else: # Default 125M
            hidden_size, layers, heads = 768, 12, 12
            
        self.is_codemind = True
        self.model_name = f"CodeMind-{size} (Sıfır Model)"
        self.is_fine_tuned = False
        
        self.codemind_adapter = CodeMindAdapter(self.config)
        
        # Resolve tokenizer exactly as standard load does
        tokenizer_path = self.codemind_adapter._codemind_path / "tokenizer"
        alt_path = self.codemind_adapter._codemind_path / "checkpoints" / "tokenizer"
                
        if tokenizer_path.exists():
            self.tokenizer = CodeTokenizer.load(str(tokenizer_path))
        elif alt_path.exists():
            self.tokenizer = CodeTokenizer.load(str(alt_path))
        else:
            self.logger.warning(f"Tokenizer bulunamadı: {tokenizer_path} veya {alt_path}, varsayılan kullanılıyor.")
            self.tokenizer = CodeTokenizer(vocab_size=16384)
            # Eğitimsiz tokenizera geçsek bile gerçek vocab size'ı model config'i için 16384'e sabitlemeliyiz ki OOM patlamasın.
            self.tokenizer.vocab_size = 16384
            
        # Flash Attention / SDPA kontrolü:
        # - "flash_attention_2" → use_flash_attention=True (flash-attn paketi kurulu olmalı)
        # - "sdpa" veya "eager" → SDPA zaten default path, use_flash_attention=False
        attn_impl = self.config.get("model.attn_implementation", "sdpa")
        use_flash_attn = attn_impl == "flash_attention_2"
        if use_flash_attn:
            self.logger.info("Flash Attention 2 etkinleştiriliyor (mevcut değilse SDPA'ya düşer).")
        else:
            self.logger.info(f"Attention backend: {attn_impl} (PyTorch SDPA kullanılacak)")

        config = CodeMindConfig(
            vocab_size=max(16384, getattr(self.tokenizer, 'vocab_size_actual', 16384)),
            hidden_size=hidden_size,
            num_hidden_layers=layers,
            num_attention_heads=heads,
            intermediate_size=hidden_size * 4,
            max_position_embeddings=4096,
            num_experts=experts,
            num_experts_per_tok=experts_per_tok,
            use_flash_attention=use_flash_attn,
        )
        self.model = CodeMindForCausalLM(config)
        
        safe_device = self.device
        self.model = self.model.to(safe_device).eval()
        
        self.codemind_adapter.model = self.model
        self.codemind_adapter.tokenizer = self.tokenizer
        
        self.logger.info(f"Sıfır {size} model başarıyla oluşturuldu.")
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

    def list_checkpoints(self) -> List[Dict[str, Any]]:
        """Discover all checkpoints from both storage formats and return a unified list.
        
        Format A: codemind/checkpoints/model_*.pt (custom CodeMind format)
        Format B: models/fine_tuned/checkpoint-*/  (HuggingFace Trainer format)
        """
        checkpoints: List[Dict[str, Any]] = []

        # Format A: .pt files
        pt_dir = self.config.get_path("codemind.checkpoint_dir", "codemind/checkpoints")
        if pt_dir.exists():
            for pt_file in sorted(pt_dir.glob("*.pt"), key=lambda p: p.stat().st_mtime, reverse=True):
                checkpoints.append({
                    "path": str(pt_file),
                    "name": pt_file.name,
                    "format": "pt",
                    "size_mb": round(pt_file.stat().st_size / (1024 * 1024), 1),
                    "modified": pt_file.stat().st_mtime,
                })

        # Format B: HuggingFace Trainer directories
        hf_dir = Path(self.config.get("training.output_dir", "models/fine_tuned"))
        if hf_dir.exists():
            # Main adapter dir itself (latest)
            adapter_cfg = hf_dir / "adapter_config.json"
            if adapter_cfg.exists():
                checkpoints.append({
                    "path": str(hf_dir),
                    "name": f"{hf_dir.name} (LoRA aktif)",
                    "format": "hf_lora",
                    "size_mb": round(sum(f.stat().st_size for f in hf_dir.rglob("*") if f.is_file()) / (1024 * 1024), 1),
                    "modified": adapter_cfg.stat().st_mtime,
                })
            # Numbered checkpoints inside
            for ckpt_dir in sorted(hf_dir.glob("checkpoint-*"), key=lambda p: p.stat().st_mtime, reverse=True):
                cfg = ckpt_dir / "adapter_config.json"
                if cfg.exists():
                    checkpoints.append({
                        "path": str(ckpt_dir),
                        "name": ckpt_dir.name,
                        "format": "hf_lora",
                        "size_mb": round(sum(f.stat().st_size for f in ckpt_dir.rglob("*") if f.is_file()) / (1024 * 1024), 1),
                        "modified": cfg.stat().st_mtime,
                    })

        # Sort all by modification time, newest first
        checkpoints.sort(key=lambda x: x["modified"], reverse=True)
        return checkpoints

    def merge_lora_into_base(self, output_dir: Optional[str] = None) -> str:
        """Merge the active LoRA adapter into the base model weights and save as .pt file.
        
        This is a one-way operation that produces a stand-alone model checkpoint 
        without any PEFT overhead. The resulting model is faster at inference time.
        
        Returns:
            Path to the saved merged checkpoint file.
        """
        if self.model is None:
            raise ValueError("Model yüklenmemiş. Önce bir model yükleyin.")

        from peft import PeftModel

        if not isinstance(self.model, PeftModel):
            raise ValueError(
                "Aktif LoRA adaptörü bulunamadı. Birleştirmek için önce bir LoRA adaptör yükleyin."
            )

        self.logger.info("LoRA adaptörü base model ile birleştiriliyor...")

        # Merge weights in-place
        merged_model = self.model.merge_and_unload()
        self.model = merged_model
        self.is_fine_tuned = False  # Now it's a merged base model

        # Determine output path
        ckpt_dir = self.config.get_path("codemind.checkpoint_dir", "codemind/checkpoints")
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        timestamp = int(time.time())
        save_path = Path(output_dir) if output_dir else ckpt_dir / f"model_merged_{timestamp}.pt"

        # Build checkpoint dict with metadata
        from src.core.checkpointing import build_checkpoint_metadata, attach_checkpoint_metadata
        state_dict = self.model.state_dict()
        checkpoint: Dict[str, Any] = {"model_state_dict": state_dict}

        config_data = getattr(getattr(self.model, "config", None), "to_dict", lambda: {})() if self.model else {}
        metadata = build_checkpoint_metadata(
            model_config=config_data,
            tokenizer=self.tokenizer,
            tokenizer_type="codemind",
            architecture_version="codemind-v2-merged",
        )
        checkpoint = attach_checkpoint_metadata(checkpoint, metadata)
        torch.save(checkpoint, save_path)

        self.logger.info(f"Birleştirilmiş model kaydedildi: {save_path}")
        return str(save_path)

    def trigger_night_shift(
        self,
        background: bool = True,
        memory_manager: Optional[Any] = None,
        trainer: Optional[Any] = None,
    ) -> None:
        """Trigger the NightShift continuous learning pipeline.
        
        Loads NightShift on demand and starts the background fine-tune loop
        that distills successful System 2 experiences into System 1.
        """
        from src.core.cognitive.continuous_learning import NightShift

        if memory_manager is None or trainer is None:
            self.logger.warning("NightShift için MemoryManager ve LoRATrainer gerekli. Atlanıyor.")
            return

        ns = NightShift(memory_manager=memory_manager, trainer=trainer)
        self.logger.info(f"NightShift başlatılıyor (arka plan={background})...")
        ns.start_night_shift(background=background)

    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 200,
        temperature: float = 0.7,
        top_p: float = 0.9,
        top_k: int = 50,
        repetition_penalty: float = 1.1,
        do_sample: bool = True,
    ) -> str:
        """Generates text from the loaded model given a prompt.

        Works with both CodeMind and HuggingFace-style models. The tokenizer
        is automatically detected (CodeTokenizer vs AutoTokenizer).
        """
        if self.model is None or self.tokenizer is None:
            raise ValueError("Model ve tokenizer yüklenmemiş. Önce load_model() çağırın.")

        # CodeMind adapter'ın kendi generate metodu varsa ona delege et
        if self.is_codemind and self.codemind_adapter and hasattr(self.codemind_adapter, "generate"):
            return self.codemind_adapter.generate(
                prompt,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                repetition_penalty=repetition_penalty,
                do_sample=do_sample,
            )

        # Generic HuggingFace / PEFT path
        self.model.eval()
        with torch.no_grad():
            # CodeTokenizer encode arayüzü
            if hasattr(self.tokenizer, "encode") and not hasattr(self.tokenizer, "__call__"):
                input_ids_list = self.tokenizer.encode(prompt)
                input_ids = torch.tensor([input_ids_list], dtype=torch.long).to(self.device)
                attention_mask = None
            else:
                encoding = self.tokenizer(
                    prompt,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                )
                input_ids = encoding["input_ids"].to(self.device)
                attention_mask = encoding.get("attention_mask", None)
                if attention_mask is not None:
                    attention_mask = attention_mask.to(self.device)

            generate_kwargs: Dict[str, Any] = dict(
                input_ids=input_ids,
                max_new_tokens=max_new_tokens,
                do_sample=do_sample,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                repetition_penalty=repetition_penalty,
                pad_token_id=getattr(self.tokenizer, "pad_token_id", None)
                    or getattr(self.tokenizer, "eos_token_id", 0),
                eos_token_id=getattr(self.tokenizer, "eos_token_id", None),
            )
            if attention_mask is not None:
                generate_kwargs["attention_mask"] = attention_mask

            output_ids = self.model.generate(**generate_kwargs)

        # Yalnızca üretilen yeni token'ları çöz
        generated_ids = output_ids[0][input_ids.shape[-1]:]

        if hasattr(self.tokenizer, "decode"):
            # CodeTokenizer veya HF tokenizer — ikisi de decode'u destekler
            try:
                result = self.tokenizer.decode(generated_ids.tolist(), skip_special_tokens=True)
            except Exception as e:
                # Decoding hatası durumunda ham token ID'lerini döndür veya hatalı kısmı temizle
                print(f"Decoding hatası: {e}")
                result = str(generated_ids.tolist())
        else:
            result = str(generated_ids.tolist())

        return result.strip()

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
