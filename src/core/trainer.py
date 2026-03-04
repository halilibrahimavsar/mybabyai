import os
from pathlib import Path
from collections import deque
from typing import Any, Callable, Dict, List, Optional, Union
import torch
from torch.utils.data import Dataset
from transformers import (
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
    TrainerCallback,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
import time
import psutil

import sys


from src.utils.config import Config
from src.utils.logger import get_logger
from src.core.model_manager import ModelManager
from src.utils.web_crawler import WebCrawler
from src.core.prompting import build_instruction_prompt
from src.core.training.meta_learning import MAMLTrainer, MAMLConfig, TaskSampler
from src.core.training.active_learning import ActiveLearner, ContinuousLearningPipeline


from src.core.callbacks import UIProgressCallback, StopCallback, NotebookProgressCallback
from src.core.datasets import TextDataset, ConversationDataset


class LoRATrainer:
    def __init__(self, model_manager: ModelManager, config: Optional[Config] = None):
        self.model_manager = model_manager
        self.config = config or Config()
        self.logger = get_logger("trainer")

        self.trainer: Optional[Trainer] = None
        self.training_args: Optional[TrainingArguments] = None
        self.is_training = False
        self.progress_callback = None
        self.should_stop = False
        self.current_training_type = "lora"

    def prepare_model_for_training(self, training_type: str = "lora", **kwargs) -> None:
        self.current_training_type = training_type
        if self.model_manager.model is None:
            raise ValueError("Model yüklenmemiş")

        model = self.model_manager.model

        if training_type == "full":
            self.logger.info("Full Training (Tam Parametre Eğitimi) başlatılıyor. LoRA kullanılmayacak.")
            # Ensure all parameters require gradients
            model.train()
            for param in model.parameters():
                param.requires_grad = True
                
            # If model was loaded in 4-bit, we can't easily do full training, warn the user
            if getattr(model, "is_loaded_in_4bit", False) or getattr(model, "is_loaded_in_8bit", False):
                self.logger.warning("DİKKAT: Model quantized (4-bit/8-bit) olarak yüklendiği için Full Training yapılamaz veya hatalı sonuç verebilir. Lütfen konfigürasyondan load_in_4bit'i kapatın.")
            
            # Print parameters for full training as well
            trainable_params, all_param = 0, 0
            for _, param in model.named_parameters():
                num_params = param.numel()
                if num_params == 0 and hasattr(param, "ds_numel"):
                    num_params = param.ds_numel
                all_param += num_params
                if param.requires_grad:
                    trainable_params += num_params
            self.logger.info(f"Full Training: trainable params: {trainable_params:,d} || all params: {all_param:,d} || trainable%: {100 * trainable_params / all_param:.4f}")
            
            self.model_manager.model = model
            return

        if self.config.get("model.load_in_4bit", False):
            model = prepare_model_for_kbit_training(model)

        if not hasattr(model, "peft_config"):
            target_modules = self.config.get(
                "model.lora.target_modules", ["q_proj", "v_proj"]
            )

            # Verify target modules exist
            verified_targets = []
            for name, module in model.named_modules():
                if any(target in name for target in target_modules):
                    if isinstance(module, torch.nn.Linear):
                        # Get the last part of the module name
                        module_name = name.split(".")[-1]
                        if module_name in target_modules and module_name not in verified_targets:
                            verified_targets.append(module_name)
            
            # If nothing found, fall back to all linear layers (common practice for small models)
            if not verified_targets:
                self.logger.warning(f"Target modules {target_modules} bulunamadı. Tüm linear katmanlar taranıyor...")
                linear_layers = set()
                for name, module in model.named_modules():
                    if isinstance(module, torch.nn.Linear):
                        module_name = name.split(".")[-1]
                        if module_name not in ["lm_head", "word_embeddings", "embed_tokens"]:
                            linear_layers.add(module_name)
                verified_targets = list(linear_layers)
                self.logger.info(f"Yedek hedef modüller {verified_targets}")

            lora_config = LoraConfig(
                r=kwargs.get("lora_r", self.config.get("model.lora.r", 16)),
                lora_alpha=kwargs.get("lora_alpha", self.config.get("model.lora.lora_alpha", 32)),
                lora_dropout=kwargs.get("lora_dropout", self.config.get("model.lora.lora_dropout", 0.05)),
                target_modules=verified_targets,
                bias="none",
                task_type="CAUSAL_LM",
            )
            model = get_peft_model(model, lora_config)
        else:
            # If it's already a PeftModel, ensure the adapters are trainable
            self.logger.info("Mevcut PEFT modeli (adapter) tespit edildi. Eğitim için parametreler çözülüyor...")
            model.train()
            for name, param in model.named_parameters():
                if "lora_" in name:
                    param.requires_grad = True
            
            # Enable input gradients for stable k-bit training if applicable
            if hasattr(model, "enable_input_require_grads"):
                model.enable_input_require_grads()
            elif hasattr(model, "get_input_embeddings"):
                def make_inputs_require_grad(module, input, output):
                    output.requires_grad_(True)
                model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)

        model.print_trainable_parameters()
        self.model_manager.model = model

    def create_training_args(
        self, output_dir: Optional[str] = None, **kwargs
    ) -> TrainingArguments:
        training_config = self.config.get("training", {})

        output_dir = output_dir or training_config.get(
            "output_dir", "models/fine_tuned"
        )

        is_codemind = self.model_manager.is_codemind
        is_full_training = getattr(self, "current_training_type", "lora") == "full"
        
        # DEFAULT optimization: 
        # CodeMind (Full): 3e-4
        # CodeMind (LoRA): 5e-5
        # Others: 2e-4
        if is_codemind:
            default_lr = 3e-4 if is_full_training else 5e-5
        else:
            default_lr = 3e-4 if is_full_training else 2e-4
        
        # CPU Optimization settings
        cpu_count = psutil.cpu_count(logical=False)
        if cpu_count and self.model_manager.device == "cpu":
            torch.set_num_threads(cpu_count)

        use_4bit = bool(self.config.get("model.load_in_4bit", False))
        gradient_checkpointing_setting = kwargs.get(
            "gradient_checkpointing",
            training_config.get("gradient_checkpointing", "auto"),
        )
        if (
            isinstance(gradient_checkpointing_setting, str)
            and gradient_checkpointing_setting.lower() == "auto"
        ):
            gradient_checkpointing = is_codemind  # Enable by default for CodeMind
        else:
            gradient_checkpointing = bool(gradient_checkpointing_setting)

        if self.model_manager.device == "cuda":
            # Always use 8-bit optimizer on GPU — saves ~50% optimizer memory
            default_optim = "paged_adamw_8bit"
        else:
            default_optim = "adamw_torch"

        use_torch_compile = kwargs.get(
            "torch_compile",
            training_config.get("torch_compile", False),
        )

        self.training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=kwargs.get(
                "num_train_epochs", training_config.get("num_train_epochs", 3)
            ),
            per_device_train_batch_size=kwargs.get(
                "per_device_train_batch_size",
                training_config.get("per_device_train_batch_size", 8 if is_codemind else 4),
            ),
            gradient_accumulation_steps=kwargs.get(
                "gradient_accumulation_steps",
                training_config.get("gradient_accumulation_steps", 1 if is_codemind else 4),
            ),
            learning_rate=kwargs.get(
                "learning_rate", training_config.get("learning_rate", default_lr)
            ),
            warmup_steps=kwargs.get(
                "warmup_steps", training_config.get("warmup_steps", 100)
            ),
            lr_scheduler_type=kwargs.get(
                "lr_scheduler_type", training_config.get("lr_scheduler_type", "cosine")
            ),
            weight_decay=kwargs.get(
                "weight_decay", training_config.get("weight_decay", 0.01)
            ),
            max_grad_norm=kwargs.get(
                "max_grad_norm", training_config.get("max_grad_norm", 0.5 if is_codemind else 1.0)
            ),
            logging_steps=kwargs.get(
                "logging_steps", training_config.get("logging_steps", 10)
            ),
            save_steps=kwargs.get("save_steps", training_config.get("save_steps", 500)),
            save_total_limit=3,
            fp16=self.model_manager.device == "cuda",
            bf16=False,
            gradient_checkpointing=gradient_checkpointing,
            optim=kwargs.get("optim", training_config.get("optim", default_optim)),
            report_to="none",
            remove_unused_columns=False,
            ddp_find_unused_parameters=False,
            dataloader_num_workers=self._get_safe_num_workers(cpu_count),
            dataloader_pin_memory=True if torch.cuda.is_available() else False,
            dataloader_prefetch_factor=2 if self._get_safe_num_workers(cpu_count) > 0 else None,
            tf32=True if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8 else False,
            torch_compile=(
                bool(use_torch_compile)
                if hasattr(torch, "compile") and sys.platform != "win32"
                else False
            ),
        )

        self.logger.info(
            "Training args: device=%s, batch=%s, grad_acc=%s, grad_ckpt=%s, optim=%s, torch_compile=%s",
            self.model_manager.device,
            self.training_args.per_device_train_batch_size,
            self.training_args.gradient_accumulation_steps,
            self.training_args.gradient_checkpointing,
            self.training_args.optim,
            self.training_args.torch_compile,
        )

        return self.training_args

    def _get_safe_num_workers(self, cpu_count: int) -> int:
        """On low-VRAM GPUs (Colab/Kaggle), use 0 workers to save RAM."""
        if torch.cuda.is_available():
            gpu_mem_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
            if gpu_mem_gb <= 16.0:
                return 0  # Multiprocessing workers eat ~1-2 GB RAM each
        return cpu_count if cpu_count else 2


    def train_from_texts(
        self,
        texts: List[str],
        formatting_func: Optional[Callable] = None,
        resume_from_checkpoint: bool = False,
        **training_kwargs,
    ) -> Dict[str, Any]:
        if self.model_manager.model is None or self.model_manager.tokenizer is None:
            raise ValueError("Model ve tokenizer yüklenmiş olmalı")

        training_type = training_kwargs.pop("training_type", "lora")
        self.prepare_model_for_training(training_type=training_type, **training_kwargs)
        
        # If resuming, we don't necessarily need to reload PeftModel if it's already one,
        # HF Trainer will handle it from checkpoint.

        max_length = int(
            training_kwargs.pop(
                "max_length",
                self.config.get(
                    "training.max_length",
                    self.config.get("memory.chunk_size", 512),
                ),
            )
        )
        pack_sequences = bool(
            training_kwargs.pop(
                "pack_sequences", self.config.get("training.pack_sequences", True)
            )
        )
        self.logger.info(
            "Text veri tokenizasyonu başlıyor: samples=%s, max_length=%s, pack_sequences=%s",
            len(texts),
            max_length,
            pack_sequences,
        )

        dataset = TextDataset(
            texts=texts,
            tokenizer=self.model_manager.tokenizer,
            max_length=max_length,
            formatting_func=formatting_func,
            pack_sequences=pack_sequences,
        )
        self.logger.info(
            "Text dataset hazırlandı: samples=%s, max_length=%s, pack_sequences=%s",
            len(dataset),
            max_length,
            pack_sequences,
        )

        return self._train(dataset, **training_kwargs)

    def train_from_conversations(
        self, conversations: List[Dict[str, str]], resume_from_checkpoint: bool = False, **training_kwargs
    ) -> Dict[str, Any]:
        if self.model_manager.model is None or self.model_manager.tokenizer is None:
            raise ValueError("Model ve tokenizer yüklenmiş olmalı")

        training_type = training_kwargs.pop("training_type", "lora")
        self.prepare_model_for_training(training_type=training_type, **training_kwargs)

        max_length = int(
            training_kwargs.pop(
                "max_length",
                self.config.get(
                    "training.max_length",
                    self.config.get("memory.chunk_size", 512),
                ),
            )
        )
        pack_sequences = bool(
            training_kwargs.pop(
                "pack_sequences", self.config.get("training.pack_sequences", True)
            )
        )
        self.logger.info(
            "Conversation veri tokenizasyonu başlıyor: samples=%s, max_length=%s, pack_sequences=%s",
            len(conversations),
            max_length,
            pack_sequences,
        )

        dataset = ConversationDataset(
            conversations=conversations,
            tokenizer=self.model_manager.tokenizer,
            max_length=max_length,
            pack_sequences=pack_sequences,
        )
        self.logger.info(
            "Conversation dataset hazırlandı: original=%s, packed=%s, max_length=%s, pack_sequences=%s",
            len(conversations),
            len(dataset),
            max_length,
            pack_sequences,
        )

        return self._train(dataset, **training_kwargs)

    def train_from_urls(
        self, urls: List[str], resume_from_checkpoint: bool = False, **training_kwargs
    ) -> Dict[str, Any]:
        """Crawls text from URLs and uses it for training."""
        self.logger.info(f"{len(urls)} adet URL'den veri çekiliyor...")
        crawler = WebCrawler()
        texts = crawler.crawl_urls(urls)
        
        if not texts:
            raise ValueError("Belirtilen URL'lerden eğitilecek metin bulunamadı")
            
        self.logger.info(f"{len(texts)} adet URL'den metin başarıyla çekildi.")
        return self.train_from_texts(texts, resume_from_checkpoint=resume_from_checkpoint, **training_kwargs)

    def train_from_hf(
        self, 
        dataset_key: str, 
        max_samples: Optional[int] = None, 
        split: str = "train",
        resume_from_checkpoint: bool = False,
        **training_kwargs
    ) -> Dict[str, Any]:
        """Downloads a dataset from Hugging Face and uses it for training."""
        from src.data.dataset_downloader import DatasetDownloader
        
        self.logger.info(f"Hugging Face dataseti indiriliyor: {dataset_key}...")
        downloader = DatasetDownloader()
        
        try:
            conversations = downloader.download_dataset(dataset_key, max_samples=max_samples, split=split)
            if not conversations:
                raise ValueError(f"Dataset {dataset_key} boş döndü veya uygun formatta değil.")
                
            self.logger.info(f"Hugging Face dataseti başarıyla indirildi: {len(conversations)} örnek.")
            return self.train_from_conversations(conversations, resume_from_checkpoint=resume_from_checkpoint, **training_kwargs)
        except Exception as e:
            self.logger.error(f"HF dataset eğitim hatası: {e}")
            raise

    def train_from_pool(
        self, pool: List[Dict[str, Any]], resume_from_checkpoint: bool = False, **training_kwargs
    ) -> Dict[str, Any]:
        if self.model_manager.model is None or self.model_manager.tokenizer is None:
            raise ValueError("Model ve tokenizer yüklenmiş olmalı")

        training_type = training_kwargs.pop("training_type", "lora")
        self.prepare_model_for_training(training_type=training_type, **training_kwargs)

        max_length = int(
            training_kwargs.pop(
                "max_length",
                self.config.get(
                    "training.max_length",
                    self.config.get("memory.chunk_size", 512),
                ),
            )
        )

        # Auto-cap max_length for low-VRAM GPUs
        # T4 = 15GB → cap at 128 (very tight for 1.4B MoE full train)
        # P100 = 16GB → cap at 256 (enough headroom with grad_ckpt + batch=1)
        # A100/V100 = 40/80 GB → no cap needed
        if torch.cuda.is_available():
            gpu_mem_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
            if gpu_mem_gb <= 12.0 and max_length > 128:
                self.logger.warning(
                    f"Düşük VRAM ({gpu_mem_gb:.0f} GB) algılandı. "
                    f"max_length {max_length} → 128 olarak kısıtlandı (OOM önleme)."
                )
                max_length = 128
            elif gpu_mem_gb <= 17.0 and max_length > 256:
                self.logger.warning(
                    f"Orta VRAM ({gpu_mem_gb:.0f} GB) algılandı. "
                    f"max_length {max_length} → 256 olarak kısıtlandı (OOM önleme)."
                )
                max_length = 256
        pack_sequences = bool(
            training_kwargs.pop(
                "pack_sequences", self.config.get("training.pack_sequences", True)
            )
        )
        
        from torch.utils.data import ConcatDataset
        datasets = []
        
        self.logger.info(f"Dataset havuzu işleniyor: {len(pool)} kaynak...")
        
        for item in pool:
            data_type = item.get("type")
            data = item.get("data")
            name = item.get("name", "Bilinmeyen")
            
            self.logger.info(f"Kaynak işleniyor: {name} ({data_type})")
            
            if data_type == "file":
                # Load from local file(s)
                from src.data.dataset_loader import DatasetLoader
                loader = DatasetLoader()
                filepath = item.get("path", "")
                if not filepath:
                    self.logger.warning(f"{name}: dosya yolu belirtilmemiş, atlanıyor.")
                    continue
                try:
                    convs = loader.load_from_file(filepath)
                    if convs:
                        ds = ConversationDataset(
                            conversations=convs,
                            tokenizer=self.model_manager.tokenizer,
                            max_length=max_length,
                            pack_sequences=pack_sequences,
                        )
                        datasets.append(ds)
                        self.logger.info(f"{name}: {len(convs)} konuşma yüklendi.")
                    else:
                        self.logger.warning(f"{name}: dosyadan veri okunamadı, atlanıyor.")
                except Exception as e:
                    self.logger.error(f"{name}: dosya yüklenirken hata: {e}")

            elif data_type in ("url", "urls"):
                # Crawl text from URL(s)
                crawler = WebCrawler()
                urls = data if isinstance(data, list) else [item.get("url", data or "")]
                urls = [u for u in urls if u]
                crawl_depth = item.get("crawl_depth", 0)
                
                if not urls:
                    self.logger.warning(f"{name}: URL belirtilmemiş, atlanıyor.")
                    continue
                    
                self.logger.info(f"{name} için {len(urls)} URL çekiliyor (derinlik: {crawl_depth})...")
                
                if crawl_depth > 0:
                    texts = crawler.crawl_with_depth(urls, depth=crawl_depth, max_pages=50)
                else:
                    texts = crawler.crawl_urls(urls)
                    
                if not texts:
                    self.logger.warning(f"{name} kaynağından metin alınamadı, atlanıyor.")
                    continue
                ds = TextDataset(
                    texts=texts,
                    tokenizer=self.model_manager.tokenizer,
                    max_length=max_length,
                    pack_sequences=pack_sequences,
                )
                datasets.append(ds)
                self.logger.info(f"{name}: {len(texts)} metin parçası çekildi.")
                
            elif data_type == "texts":
                ds = TextDataset(
                    texts=data,
                    tokenizer=self.model_manager.tokenizer,
                    max_length=max_length,
                    pack_sequences=pack_sequences,
                )
                datasets.append(ds)
                
            elif data_type == "conversations":
                ds = ConversationDataset(
                    conversations=data,
                    tokenizer=self.model_manager.tokenizer,
                    max_length=max_length,
                    pack_sequences=pack_sequences,
                )
                datasets.append(ds)
            elif data_type == "huggingface":
                from src.data.dataset_downloader import DatasetDownloader
                downloader = DatasetDownloader()
                self.logger.info(f"HF Dataset çekiliyor: {name}")
                convs = downloader.download_dataset(
                    item.get("dataset_key", data), 
                    max_samples=item.get("max_samples"),
                    split=item.get("split", "train")
                )
                if convs:
                    ds = ConversationDataset(
                        conversations=convs,
                        tokenizer=self.model_manager.tokenizer,
                        max_length=max_length,
                        pack_sequences=pack_sequences,
                    )
                    datasets.append(ds)
            else:
                self.logger.warning(f"Bilinmeyen veri tipi: {data_type}, atlanıyor.")
                
        if not datasets:
            raise ValueError("Havuzdaki verilerden hiçbir geçerli dataset oluşturulamadı.")
            
        combined_dataset = ConcatDataset(datasets)
        self.logger.info(f"Tüm kaynaklar birleştirildi. Toplam örnek sayısı: {len(combined_dataset)}")

        return self._train(combined_dataset, resume_from_checkpoint=resume_from_checkpoint, **training_kwargs)

    def _train(self, dataset: Dataset, resume_from_checkpoint: bool = False, **kwargs) -> Dict[str, Any]:
        use_notebook_callback = kwargs.pop("use_notebook_callback", False)
        self.create_training_args(**kwargs)

        # ── Aggressive memory optimization ──
        import gc
        import os as _os

        # Set PyTorch memory allocator to prevent fragmentation
        _os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            gc.collect()

        # Explicitly enable gradient checkpointing on the model for VRAM savings
        model = self.model_manager.model
        if self.training_args.gradient_checkpointing:
            if hasattr(model, "gradient_checkpointing_enable"):
                model.gradient_checkpointing_enable()
                self.logger.info("Gradient checkpointing etkinleştirildi (VRAM tasarrufu).")
            if hasattr(model, "enable_input_require_grads"):
                model.enable_input_require_grads()

        # Log VRAM status before training
        if torch.cuda.is_available():
            gpu_mem_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
            free_mem = (torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_allocated(0)) / 1e9
            self.logger.info(f"GPU VRAM: {gpu_mem_gb:.1f} GB toplam, {free_mem:.1f} GB boş")

        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.model_manager.tokenizer,
            mlm=False,
            pad_to_multiple_of=8 if self.model_manager.device == "cuda" else None,
        )

        callbacks = []
        if self.progress_callback:
            callbacks.append(UIProgressCallback(self.progress_callback))
        
        if use_notebook_callback:
            callbacks.append(NotebookProgressCallback())

        # Add stop callback
        self.should_stop = False
        callbacks.append(StopCallback(lambda: self.should_stop))

        self.trainer = Trainer(
            model=self.model_manager.model,
            args=self.training_args,
            train_dataset=dataset,
            data_collator=data_collator,
            callbacks=callbacks,
        )

        self.is_training = True
        self.logger.info("Eğitim başlatılıyor...")

        try:
            # Check if we should resume from the last checkpoint
            checkpoint_path = None
            if resume_from_checkpoint:
                output_dir = self.training_args.output_dir
                if os.path.exists(output_dir):
                    # HF Trainer can find the latest checkpoint automatically if passed True
                    # or better, it finds the specific directory if it exists.
                    checkpoint_path = True
                    self.logger.info(f"Resume from checkpoint aktif: {output_dir}")

            train_result = self.trainer.train(resume_from_checkpoint=checkpoint_path)
            self.is_training = False

            metrics = train_result.metrics
            self.trainer.log_metrics("train", metrics)
            self.trainer.save_metrics("train", metrics)

            self.trainer.save_model()
            self.trainer.save_state()
            
            # Save CodeMind format checkpoint with metadata
            try:
                import time
                from src.core.checkpointing import build_checkpoint_metadata, attach_checkpoint_metadata
                
                checkpoint_dir = self.config.get_path("codemind.checkpoint_dir", "codemind/checkpoints")
                checkpoint_dir.mkdir(parents=True, exist_ok=True)
                
                if hasattr(self.model_manager.model, "base_model"):
                    raw_state_dict = self.model_manager.model.base_model.model.state_dict()
                else:
                    raw_state_dict = self.model_manager.model.state_dict()
                
                # If full training, we want all weights. 
                # If LoRA, we might just want to save the adapters or base model depending on design.
                # Here we save base weights without lora params for CodeMind checkpoint format.
                state_dict = {k: v for k, v in raw_state_dict.items() if "lora_" not in k and "modules_to_save" not in k}
                checkpoint = {"model_state_dict": state_dict}
                
                config_dict = getattr(self.model_manager.model, "config", None)
                config_data = config_dict.to_dict() if config_dict else {}
                
                metadata = build_checkpoint_metadata(
                    model_config=config_data,
                    tokenizer=self.model_manager.tokenizer,
                    tokenizer_type="pretrained",
                    architecture_version="codemind-v2"
                )
                checkpoint = attach_checkpoint_metadata(checkpoint, metadata)
                
                timestamp = int(time.time())
                versioned_path = checkpoint_dir / f"model_final_{timestamp}.pt"
                final_path = checkpoint_dir / "model_final.pt"
                
                torch.save(checkpoint, versioned_path)
                torch.save(checkpoint, final_path)
                self.logger.info(f"CodeMind format checkpoint saved to {final_path}")
            except Exception as metadata_e:
                self.logger.error(f"CodeMind checkpoint ve metadata kaydedilemedi: {metadata_e}")
            
            # Silent update for the model manager state
            self.model_manager.is_fine_tuned = True

            self.logger.info(f"Eğitim tamamlandı: {metrics}")
            return metrics

        except Exception as e:
            self.is_training = False
            error_msg = str(e)
            if "optimizer" in error_msg.lower() and "match the size" in error_msg.lower():
                self.logger.error("HATA: Optimizer (Eğitim Hafızası) uyuşmazlığı tıkandı! \n"
                                  "Neden: Muhtemelen LoRA ve Full Training arasında geçiş yaptınız veya model yapısını değiştirdiniz.\n"
                                  "Çözüm: 'Kaldığı Yerden Devam Et (Resume)' seçeneğini kapatın veya farklı bir 'Çıktı Dizini' belirleyin.")
            else:
                self.logger.error(f"Eğitim hatası: {e}")
            raise

    def stop_training(self) -> None:
        if self.is_training:
            self.should_stop = True
            self.logger.info("Eğitim durdurma sinyali gönderildi")

    def get_training_progress(self) -> Dict[str, Any]:
        if not self.trainer or not self.is_training:
            return {"is_training": False}

        state = self.trainer.state
        return {
            "is_training": True,
            "current_step": state.global_step,
            "max_steps": state.max_steps,
            "progress_percent": state.global_step / state.max_steps * 100
            if state.max_steps
            else 0,
            "current_epoch": state.epoch,
            "loss": state.log_history[-1].get("loss", 0) if state.log_history else 0,
        }

    def train_maml(
        self, 
        task_data_path: str, 
        iterations: int = 1000, 
        save_path: Optional[str] = None
    ) -> List[Dict[str, float]]:
        """Runs Meta-Learning (MAML) training."""
        if self.model_manager.model is None or self.model_manager.tokenizer is None:
            raise ValueError("Model and tokenizer must be loaded")

        self.logger.info(f"MAML eğitimi başlatılıyor: iterations={iterations}")
        
        maml_config = MAMLConfig(
            inner_lr=self.config.get("training.meta_learning.inner_lr", 1e-3),
            outer_lr=self.config.get("training.meta_learning.outer_lr", 1e-4),
            inner_steps=self.config.get("training.meta_learning.inner_steps", 5),
            meta_batch_size=self.config.get("training.meta_learning.meta_batch_size", 8),
        )
        
        from src.core.training.meta_learning import create_task_sampler_from_data
        task_sampler = create_task_sampler_from_data(task_data_path)
        
        trainer = MAMLTrainer(
            model=self.model_manager.model,
            tokenizer=self.model_manager.tokenizer,
            config=maml_config,
            device=self.model_manager.device
        )
        
        self.is_training = True
        try:
            history = trainer.train(task_sampler, num_iterations=iterations, save_path=save_path)
            self.is_training = False
            return history
        except Exception as e:
            self.is_training = False
            self.logger.error(f"MAML eğitim hatası: {e}")
            raise

    def train_active(
        self,
        data_path: str,
        get_answer_func: Callable,
        save_path: str = "checkpoints/active"
    ) -> Dict[str, Any]:
        """Runs Active Learning pipeline."""
        if self.model_manager.model is None or self.model_manager.tokenizer is None:
            raise ValueError("Model and tokenizer must be loaded")

        self.logger.info(f"Active Learning döngüsü başlatılıyor: {data_path}")
        
        pipeline = ContinuousLearningPipeline(
            model=self.model_manager.model,
            tokenizer=self.model_manager.tokenizer,
            data_path=data_path,
            save_path=save_path
        )
        
        self.is_training = True
        try:
            result = pipeline.interactive_learning_round(get_answer_func)
            pipeline.save_progress()
            self.is_training = False
            return result
        except Exception as e:
            self.is_training = False
            self.logger.error(f"Active Learning hatası: {e}")
            raise
