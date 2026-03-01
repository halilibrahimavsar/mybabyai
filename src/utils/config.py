import os
from pathlib import Path
from typing import Any, Dict, Optional
import yaml


class Config:
    _instance: Optional["Config"] = None
    _config: Dict[str, Any] = {}

    def __new__(cls, config_path: Optional[str] = None) -> "Config":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self, config_path: Optional[str] = None) -> None:
        if self._initialized:
            return

        self._initialized = True
        self.base_dir = Path(__file__).parent.parent.parent
        self.config_path = config_path or str(self.base_dir / "configs" / "config.yaml")
        self._load_config()
        self._setup_paths()

    def _load_config(self) -> None:
        if os.path.exists(self.config_path):
            with open(self.config_path, "r", encoding="utf-8") as f:
                self._config = yaml.safe_load(f) or {}
        else:
            self._config = self._default_config()

    def _default_config(self) -> Dict[str, Any]:
        return {
            "app": {
                "name": "MyBabyAI",
                "version": "1.0.0",
                "language": "tr",
                "debug": False,
                "auto_load_model": True,
            },
            "model": {
                "default_model": "mistralai/Mistral-7B-Instruct-v0.2",
                "model_path": "models/",
                "device": "auto",
                "load_in_4bit": True,
                "max_memory": "8GB",
                "codemind": {
                    "checkpoint_dirs": ["checkpoints", "checkpoints_instruct"],
                    "min_compatibility_ratio": 0.6,
                },
                "lora": {
                    "r": 16,
                    "lora_alpha": 32,
                    "lora_dropout": 0.05,
                    "target_modules": ["q_proj", "v_proj", "k_proj", "o_proj"],
                },
            },
            "memory": {
                "vector_db_path": "data/chroma",
                "embedding_model": "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
                "chunk_size": 512,
                "chunk_overlap": 50,
                "max_history": 100,
            },
            "training": {
                "output_dir": "models/fine_tuned",
                "num_train_epochs": 3,
                "per_device_train_batch_size": 8,
                "gradient_accumulation_steps": 1,
                "gradient_checkpointing": "auto",
                "learning_rate": 2.0e-4,
                "max_length": 256,
                "pack_sequences": True,
                "warmup_steps": 100,
                "logging_steps": 10,
                "save_steps": 500,
                "optim": "adamw_torch",
                "torch_compile": False,
                "qlora": {
                    "use_qlora": True,
                    "bnb_4bit_quant_type": "nf4",
                    "bnb_4bit_compute_dtype": "float16",
                },
            },
            "gui": {
                "theme": "dark",
                "font_family": "Segoe UI",
                "font_size": 11,
                "window_size": [1200, 800],
                "sidebar_width": 300,
            },
        }

    def _setup_paths(self) -> None:
        paths = ["models", "data", "data/chroma", "logs"]
        for path in paths:
            full_path = self.base_dir / path
            full_path.mkdir(parents=True, exist_ok=True)

    def get(self, key: str, default: Any = None) -> Any:
        keys = key.split(".")
        value = self._config
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
        return value

    def get_path(self, key: str, default: Any = None) -> Path:
        val = self.get(key, default)
        if val is None:
            return self.base_dir
        p = Path(val)
        if p.is_absolute():
            return p
        return self.base_dir / p

    def validate(self) -> bool:
        required_keys = ["model.default_model", "memory.vector_db_path", "training.output_dir"]
        is_valid = True
        for key in required_keys:
            if self.get(key) is None:
                print(f"[Config Error] Eksik yapılandırma anahtarı: {key}")
                is_valid = False
        return is_valid

    def set(self, key: str, value: Any) -> None:
        keys = key.split(".")
        config = self._config
        for k in keys[:-1]:
            if k not in config:
                config[k] = {}
            config = config[k]
        config[keys[-1]] = value

    def save(self) -> None:
        with open(self.config_path, "w", encoding="utf-8") as f:
            yaml.dump(self._config, f, default_flow_style=False, allow_unicode=True)

    @property
    def app_name(self) -> str:
        return self.get("app.name", "MyBabyAI")

    @property
    def app_version(self) -> str:
        return self.get("app.version", "1.0.0")

    @property
    def debug(self) -> bool:
        return self.get("app.debug", False)

    @property
    def model_config(self) -> Dict[str, Any]:
        return self.get("model", {})

    @property
    def memory_config(self) -> Dict[str, Any]:
        return self.get("memory", {})

    @property
    def training_config(self) -> Dict[str, Any]:
        return self.get("training", {})

    @property
    def gui_config(self) -> Dict[str, Any]:
        return self.get("gui", {})
