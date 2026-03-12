import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Generator, List, Optional, Tuple, Union
import torch



from src.utils.config import Config
from src.utils.logger import get_logger
from src.core.checkpointing import (
    extract_checkpoint_metadata,
    build_checkpoint_metadata,
    attach_checkpoint_metadata,
    DEPRECATED_KEY_PATTERNS,
)
from src.core.prompting import build_codemind_code_prompt, build_instruction_prompt, extract_assistant_response


@dataclass
class CheckpointCompatibilityReport:
    checkpoint_path: str
    total_checkpoint_keys: int
    total_model_keys: int
    matched_keys: int
    adapted_keys: int
    shape_mismatches: int
    missing_in_model: int
    missing_in_checkpoint: int
    deprecated_keys_found: int
    compatibility_ratio: float
    compatibility_threshold: float
    metadata_ok: bool
    tokenizer_vocab_size: int
    model_vocab_size: int
    skipped_keys_preview: List[str]

    @property
    def is_compatible(self) -> bool:
        return self.compatibility_ratio >= self.compatibility_threshold and self.metadata_ok

    def summary(self) -> str:
        return (
            f"ratio={self.compatibility_ratio:.3f} "
            f"(threshold={self.compatibility_threshold:.3f}), "
            f"matched={self.matched_keys}, adapted={self.adapted_keys}, "
            f"shape_mismatches={self.shape_mismatches}, "
            f"missing_in_model={self.missing_in_model}, "
            f"missing_in_checkpoint={self.missing_in_checkpoint}, "
            f"deprecated_keys={self.deprecated_keys_found}"
        )


class CodeMindAdapter:
    def __init__(self, config: Optional[Config] = None):
        self.config = config or Config()
        self.logger = get_logger("codemind_adapter")

        self.model: Optional[Any] = None
        self.tokenizer: Optional[Any] = None
        self.device: str = "cuda" if torch.cuda.is_available() else "cpu"
        self.model_name: str = "CodeMind-125M"
        self.last_compatibility_report: Optional[CheckpointCompatibilityReport] = None

        self._codemind_path = self.config.base_dir / "codemind"

        configured_dirs = self.config.get("model.codemind.checkpoint_dirs", None)
        if configured_dirs and isinstance(configured_dirs, list):
            self._checkpoint_dirs = [self._codemind_path / d for d in configured_dirs]
        else:
            self._checkpoint_dirs = [
                self._codemind_path / "checkpoints",
                self._codemind_path / "checkpoints_instruct",
            ]

        self._compatibility_threshold = float(
            self.config.get("model.codemind.min_compatibility_ratio", 0.55)
        )

    def _candidate_checkpoint_files(self, checkpoint_dir: Path) -> List[Path]:
        return [
            checkpoint_dir / "model_best.pt",
            checkpoint_dir / "model_final.pt",
            checkpoint_dir / "model.pt",
        ]

    def _resolve_checkpoint_path(self) -> Optional[Path]:
        for checkpoint_dir in self._checkpoint_dirs:
            self.logger.debug(f"Searching checkpoint dir: {checkpoint_dir}")
            for candidate in self._candidate_checkpoint_files(checkpoint_dir):
                if candidate.exists() and candidate.stat().st_size > 0:
                    self.logger.info(f"Found valid checkpoint: {candidate}")
                    return candidate
                elif candidate.exists():
                    self.logger.warning(f"Skipping empty checkpoint: {candidate}")
        return None

    def is_available(self) -> bool:
        return self._resolve_checkpoint_path() is not None

    def _resolve_tokenizer_path(self, checkpoint_path: Path) -> Path:
        # 1. Try sibling directory (standard)
        sibling_tokenizer = checkpoint_path.parent / "tokenizer"
        if sibling_tokenizer.exists():
            return sibling_tokenizer
            
        # 2. Try other configured checkpoint directories
        for checkpoint_dir in self._checkpoint_dirs:
            candidate = checkpoint_dir / "tokenizer"
            if candidate.exists():
                self.logger.info(f"Using alternative tokenizer found at: {candidate}")
                return candidate
        
        # 3. Fallback to default (will likely fail if none of the above exist)
        return sibling_tokenizer

    def _build_model_from_checkpoint(self, checkpoint: Dict[str, Any], tokenizer_vocab_size: int):
        from src.core.model.codemind import CodeMindConfig, CodeMindForCausalLM

        checkpoint_config = checkpoint.get("config", {}) if isinstance(checkpoint, dict) else {}
        if not checkpoint_config and isinstance(checkpoint, dict) and "checkpoint_metadata" in checkpoint:
            checkpoint_config = checkpoint["checkpoint_metadata"].get("model_config", {})
            
        hidden_size = int(checkpoint_config.get("hidden_size", 768))
        num_hidden_layers = int(checkpoint_config.get("num_hidden_layers", 12))
        num_attention_heads = int(checkpoint_config.get("num_attention_heads", 12))
        intermediate_size = int(checkpoint_config.get("intermediate_size", hidden_size * 4))
        max_position_embeddings = int(
            checkpoint_config.get("max_position_embeddings", 2048)
        )

        # --- GQA: read num_key_value_heads from checkpoint ---
        # If the checkpoint was trained with full MHA (no GQA), num_key_value_heads
        # equals num_attention_heads. We must honour that here; otherwise we build a
        # GQA model with fewer KV dimensions than the saved weights have.
        num_key_value_heads = int(
            checkpoint_config.get(
                "num_key_value_heads",
                checkpoint_config.get("num_kv_heads", num_attention_heads),  # default = MHA
            )
        )

        # --- MoE: read expert count from checkpoint ---
        num_experts = int(checkpoint_config.get("num_experts", 1))
        num_experts_per_tok = int(checkpoint_config.get("num_experts_per_tok", 1))

        config = CodeMindConfig(
            vocab_size=int(tokenizer_vocab_size),
            hidden_size=hidden_size,
            num_hidden_layers=num_hidden_layers,
            num_attention_heads=num_attention_heads,
            num_key_value_heads=num_key_value_heads,
            intermediate_size=intermediate_size,
            max_position_embeddings=max_position_embeddings,
            num_experts=num_experts,
            num_experts_per_tok=num_experts_per_tok,
        )
        return CodeMindForCausalLM(config)

    def _adapt_checkpoint_tensor(
        self,
        key: str,
        checkpoint_tensor: torch.Tensor,
        model_tensor: torch.Tensor,
    ) -> Optional[torch.Tensor]:
        # Allow embedding/head row adaptation when checkpoint vocab is larger/smaller.
        if key not in {"model.word_embeddings.weight", "lm_head.weight"}:
            return None
        if checkpoint_tensor.ndim != 2 or model_tensor.ndim != 2:
            return None
        if checkpoint_tensor.shape[1] != model_tensor.shape[1]:
            return None

        if checkpoint_tensor.shape[0] >= model_tensor.shape[0]:
            return checkpoint_tensor[: model_tensor.shape[0], :]

        adapted = model_tensor.clone()
        adapted[: checkpoint_tensor.shape[0], :] = checkpoint_tensor
        return adapted

    def _build_compatibility_report(
        self,
        checkpoint_path: Path,
        state_dict: Dict[str, torch.Tensor],
        model_state: Dict[str, torch.Tensor],
        matched_keys: int,
        adapted_keys: int,
        shape_mismatches: int,
        missing_in_model: int,
        metadata_ok: bool,
        tokenizer_vocab_size: int,
        model_vocab_size: int,
        skipped_keys: List[str],
        missing_in_checkpoint: int,
        deprecated_keys_found: int,
    ) -> CheckpointCompatibilityReport:
        # Deprecated keys don't count towards the model's expected keys
        total_model_keys = max(len(model_state), 1)
        compatibility_ratio = (matched_keys + adapted_keys) / (total_model_keys)

        return CheckpointCompatibilityReport(
            checkpoint_path=str(checkpoint_path),
            total_checkpoint_keys=len(state_dict),
            total_model_keys=len(model_state),
            matched_keys=matched_keys,
            adapted_keys=adapted_keys,
            shape_mismatches=shape_mismatches,
            missing_in_model=missing_in_model,
            missing_in_checkpoint=missing_in_checkpoint,
            deprecated_keys_found=deprecated_keys_found,
            compatibility_ratio=compatibility_ratio,
            compatibility_threshold=self._compatibility_threshold,
            metadata_ok=metadata_ok,
            tokenizer_vocab_size=tokenizer_vocab_size,
            model_vocab_size=model_vocab_size,
            skipped_keys_preview=skipped_keys[:10],
        )

    def load_model(
        self, checkpoint_path: Optional[Union[str, Path]] = None
    ) -> Tuple[Any, Any, CheckpointCompatibilityReport]:
        if checkpoint_path is None:
            checkpoint_path = self._resolve_checkpoint_path()
        else:
            checkpoint_path = Path(checkpoint_path)
            if not checkpoint_path.exists():
                # Be forgiving if the user provided an absolute path from the project root (e.g. /codemind/...)
                rel_path = str(checkpoint_path).lstrip("/")
                potential_path = self.config.base_dir / rel_path
                if potential_path.exists():
                    checkpoint_path = potential_path

        if checkpoint_path is None or not checkpoint_path.exists():
            searched = ", ".join(str(p) for p in self._checkpoint_dirs)
            raise FileNotFoundError(
                f"CodeMind checkpoint not found at {checkpoint_path}. Searched directories: {searched}. "
                "If you are in a fresh environment (e.g. Colab), please ensure checkpoints are uploaded to 'codemind/checkpoints/' "
                "or set allow_fresh_fallback=True in ModelManager.load_model() to train from scratch."
            )

        self.logger.info(f"Loading CodeMind model from: {checkpoint_path}")

        # --- 1. Load checkpoint to extract metadata first ---
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        state_dict = checkpoint.get("model_state_dict", checkpoint)
        metadata = extract_checkpoint_metadata(checkpoint)

        from src.core.tokenizer.code_tokenizer import CodeTokenizer
        
        pretrained_tok = self.config.get("model.pretrained_tokenizer", "")
        # Fallback to checkpoint metadata if local config doesn't specify one
        if not pretrained_tok and metadata is not None and metadata.pretrained_tokenizer_name:
            pretrained_tok = metadata.pretrained_tokenizer_name

        tokenizer_path = self._resolve_tokenizer_path(checkpoint_path)

        if pretrained_tok:
            from transformers import AutoTokenizer
            self.logger.info(f"Loading HF tokenizer: {pretrained_tok}")
            self.tokenizer = AutoTokenizer.from_pretrained(pretrained_tok, trust_remote_code=True)
            special_tokens = ["<|pad|>", "<|eos|>", "<|unk|>", "<|tr|>", "<|python|>", "<|dart|>", "<|javascript|>", "▁"]
            self.tokenizer.add_tokens(special_tokens, special_tokens=True)
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = "<|pad|>" if "<|pad|>" in self.tokenizer.get_vocab() else self.tokenizer.eos_token
            tokenizer_vocab = len(self.tokenizer)
            self.tokenizer.vocab_size_actual = tokenizer_vocab
        elif tokenizer_path.exists() and (tokenizer_path / "tokenizer_config.json").exists():
            from transformers import AutoTokenizer
            self.logger.info(f"Loading HF tokenizer from path: {tokenizer_path}")
            self.tokenizer = AutoTokenizer.from_pretrained(str(tokenizer_path), trust_remote_code=True)
            tokenizer_vocab = len(self.tokenizer)
            self.tokenizer.vocab_size_actual = tokenizer_vocab
        elif tokenizer_path.exists():
            self.logger.info(f"Loading CodeTokenizer from path: {tokenizer_path}")
            self.tokenizer = CodeTokenizer.load(str(tokenizer_path))
            tokenizer_vocab = self.tokenizer.vocab_size_actual
        else:
            raise FileNotFoundError(f"Tokenizer not found at {tokenizer_path} and no pretrained_tokenizer set in config or metadata.")
        
        self.logger.info(f"Tokenizer loaded. Vocab size: {tokenizer_vocab}")
        metadata_ok = True
        if metadata is not None:
            metadata_ok = metadata.vocab_size == tokenizer_vocab
            if not metadata_ok:
                self.logger.error(
                    "Checkpoint metadata mismatch: "
                    f"metadata.vocab_size={metadata.vocab_size}, tokenizer.vocab={tokenizer_vocab}. "
                    "Note: vocab_size=228 usually indicates a checkpoint saved with an untrained (base) tokenizer."
                )

        self.model = self._build_model_from_checkpoint(checkpoint, tokenizer_vocab)
        model_state = self.model.state_dict()
        filtered_state: Dict[str, torch.Tensor] = {}
        skipped_keys: List[str] = []

        matched_keys = 0
        adapted_keys = 0
        shape_mismatches = 0
        missing_in_model = 0
        deprecated_keys_found = 0

        for og_key, value in state_dict.items():
            key = og_key
            if "lora_" in key or "modules_to_save" in key:
                continue
                
            # Strip PEFT base model prefix
            if key.startswith("base_model.model."):
                key = key[len("base_model.model."):]
            
            # Strip PEFT layer wrapping (.base_layer.)
            if ".base_layer." in key:
                key = key.replace(".base_layer.", ".")
            
            # Handle model. prefix mismatch (important for CausalLM wrapper vs raw model)
            if key not in model_state:
                if f"model.{key}" in model_state:
                    key = f"model.{key}"
                elif key.startswith("model.") and key[6:] in model_state:
                    # check if the version without prefix exists
                    if key[6:] in model_state:
                        key = key[6:]
                
            # Skip non-persistent buffers that are in checkpoint but not model
            if any(buf in key for buf in ["inv_freq", "cos_cached", "sin_cached"]):
                continue

            # Skip known deprecated keys from older architectures
            if any(pattern in key for pattern in DEPRECATED_KEY_PATTERNS):
                deprecated_keys_found += 1
                skipped_keys.append(f"{og_key} (deprecated in new architecture)")
                continue
            
            # Mapping for older MLP architectures (GELU -> SwiGLU) and MoE migration
            if "mlp.dense_h_to_4h" in key:
                # Try to map to both gate_proj and up_proj
                for sub in ["gate_proj", "up_proj"]:
                    # check if we are mapping to MoE expert 0
                    if getattr(self.model.config, "num_experts", 1) > 1:
                        new_key = key.replace("mlp.dense_h_to_4h", f"mlp.experts.0.{sub}")
                    else:
                        new_key = key.replace("mlp.dense_h_to_4h", f"mlp.{sub}")
                        
                    if new_key in model_state and new_key not in filtered_state:
                        if value.shape == model_state[new_key].shape:
                            filtered_state[new_key] = value
                            matched_keys += 1
                        else:
                            adapted = self._adapt_checkpoint_tensor(new_key, value, model_state[new_key])
                            if adapted is not None:
                                filtered_state[new_key] = adapted
                                adapted_keys += 1
                continue
            
            if "mlp.dense_4h_to_h" in key:
                if getattr(self.model.config, "num_experts", 1) > 1:
                    new_key = key.replace("mlp.dense_4h_to_h", "mlp.experts.0.down_proj")
                else:
                    new_key = key.replace("mlp.dense_4h_to_h", "mlp.down_proj")
                if new_key in model_state:
                    if value.shape == model_state[new_key].shape:
                        filtered_state[new_key] = value
                        matched_keys += 1
                    else:
                        adapted = self._adapt_checkpoint_tensor(new_key, value, model_state[new_key])
                        if adapted is not None:
                            filtered_state[new_key] = adapted
                            adapted_keys += 1
                continue
                
            # Direct SwiGLU to MoE mapping for newer checkpoints being loaded into MoE models
            if "mlp.gate_proj" in key or "mlp.up_proj" in key or "mlp.down_proj" in key:
                if getattr(self.model.config, "num_experts", 1) > 1 and "experts" not in key:
                    new_key = key.replace("mlp.", "mlp.experts.0.")
                    if new_key in model_state:
                        if value.shape == model_state[new_key].shape:
                            filtered_state[new_key] = value
                            matched_keys += 1
                        else:
                            adapted = self._adapt_checkpoint_tensor(new_key, value, model_state[new_key])
                            if adapted is not None:
                                filtered_state[new_key] = adapted
                                adapted_keys += 1
                    continue
                    
            # MHA -> GQA migration
            if "attention.query_key_value" in key:
                # Old MHA fused projection: [3 * hidden_size, hidden_size]
                # Split it into q, k, v
                d_model = self.model.config.hidden_size
                num_heads = self.model.config.num_attention_heads
                head_dim = self.model.config.head_dim
                num_kv_heads = getattr(self.model.config, "num_key_value_heads", num_heads)
                
                # Check if the shape matches the old combined projection shape
                if value.shape[0] == 3 * d_model:
                    # reshape to [3, num_heads, head_dim, d_model]
                    qkv = value.view(3, num_heads, head_dim, -1)
                    old_q = qkv[0].reshape(num_heads * head_dim, -1)
                    old_k = qkv[1].reshape(num_heads * head_dim, -1)
                    old_v = qkv[2].reshape(num_heads * head_dim, -1)
                    
                    q_key = key.replace("query_key_value", "q_proj")
                    k_key = key.replace("query_key_value", "k_proj")
                    v_key = key.replace("query_key_value", "v_proj")
                    
                    if q_key in model_state:
                        filtered_state[q_key] = old_q
                        matched_keys += 1
                        
                    if k_key in model_state:
                        # GQA adapter: if num_kv_heads < num_heads, we average the heads
                        if num_kv_heads < num_heads:
                            num_kv_groups = num_heads // num_kv_heads
                            # shape: [num_kv_heads, num_kv_groups, head_dim, d_model]
                            grouped_k = old_k.view(num_kv_heads, num_kv_groups, head_dim, -1)
                            grouped_v = old_v.view(num_kv_heads, num_kv_groups, head_dim, -1)
                            
                            # Average over the groups
                            old_k = grouped_k.mean(dim=1).reshape(num_kv_heads * head_dim, -1)
                            old_v = grouped_v.mean(dim=1).reshape(num_kv_heads * head_dim, -1)
                            
                        filtered_state[k_key] = old_k
                        filtered_state[v_key] = old_v
                        matched_keys += 2 # one for K, one for V
                else:
                    skipped_keys.append(f"{og_key} -> cannot split, shape {value.shape} != 3x{d_model}")
                    shape_mismatches += 1
                continue
                
            if "attention.dense.weight" in key:
                new_key = key.replace("attention.dense", "attention.o_proj")
                if new_key in model_state:
                    if value.shape == model_state[new_key].shape:
                        filtered_state[new_key] = value
                        matched_keys += 1
                    else:
                        adapted = self._adapt_checkpoint_tensor(new_key, value, model_state[new_key])
                        if adapted is not None:
                            filtered_state[new_key] = adapted
                            adapted_keys += 1
                continue
                
            if key not in model_state:
                missing_in_model += 1
                skipped_keys.append(f"{og_key} -> {key} (not in model)")
                continue

            if value.shape == model_state[key].shape:
                filtered_state[key] = value
                matched_keys += 1
                continue

            adapted = self._adapt_checkpoint_tensor(key, value, model_state[key])
            if adapted is not None:
                filtered_state[key] = adapted
                adapted_keys += 1
            else:
                shape_mismatches += 1
                skipped_keys.append(
                    f"{og_key} -> {key} (checkpoint: {tuple(value.shape)} vs model: {tuple(model_state[key].shape)})"
                )

        # Handle weight tying: if lm_head.weight is missing but word_embeddings.weight is present
        if "lm_head.weight" not in filtered_state and "word_embeddings.weight" in filtered_state:
            if "lm_head.weight" in model_state:
                filtered_state["lm_head.weight"] = filtered_state["word_embeddings.weight"]
                matched_keys += 1

        report = self._build_compatibility_report(
            checkpoint_path=checkpoint_path,
            state_dict=state_dict,
            model_state=model_state,
            matched_keys=matched_keys,
            adapted_keys=adapted_keys,
            shape_mismatches=shape_mismatches,
            missing_in_model=missing_in_model,
            metadata_ok=metadata_ok,
            tokenizer_vocab_size=tokenizer_vocab,
            model_vocab_size=self.model.config.vocab_size,
            skipped_keys=skipped_keys,
            missing_in_checkpoint=len(model_state) - (matched_keys + adapted_keys),
            deprecated_keys_found=deprecated_keys_found,
        )

        if not report.is_compatible:
            # Instead of crashing, we log a warning and proceed with partial weights
            self.logger.warning(
                "Checkpoint compatibility below threshold. "
                f"{report.summary()}. "
                f"Skipped preview: {report.skipped_keys_preview}\n"
                "Loading with partial weights (unmapped layers will remain randomly initialized)."
            )

        self.model.load_state_dict(filtered_state, strict=False)

        # --- CUDA OOM safety: try target device, fall back to CPU ---
        try:
            self.model = self.model.to(self.device)
        except RuntimeError as oom_err:
            if "out of memory" in str(oom_err).lower():
                self.logger.warning(
                    f"CUDA OOM while moving model to {self.device}. "
                    "Falling back to CPU. Performance will be reduced."
                )
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                self.device = "cpu"
                self.model = self.model.to("cpu")
            else:
                raise

        self.model.eval()
        self.last_compatibility_report = report

        self.logger.info(f"CodeMind loaded on {self.device}. Compatibility: {report.summary()}")
        return self.model, self.tokenizer, report

    def get_model_info(self) -> Dict[str, Any]:
        if self.model is None:
            return {"status": "not_loaded"}

        total_params = sum(p.numel() for p in self.model.parameters())
        base_info = {
            "model_name": self.model_name,
            "device": self.device,
            "total_parameters": total_params,
            "trainable_parameters": total_params,
            "trainable_percentage": "100%",
            "quantized": False,
            "vocab_size": self.tokenizer.vocab_size_actual if self.tokenizer else 0,
        }
        if self.last_compatibility_report is not None:
            base_info["checkpoint_compatibility"] = asdict(self.last_compatibility_report)
        return base_info

    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 256,
        temperature: float = 0.7,
        top_k: int = 50,
        top_p: float = 0.95,
        language: str = "python",
    ) -> str:
        if self.model is None or self.tokenizer is None:
            raise ValueError("Model not loaded")

        full_prompt = build_instruction_prompt(user=prompt, language=language, include_eos=False)
        tokens = self.tokenizer.encode(full_prompt, add_special_tokens=False)
        input_ids = torch.tensor([tokens], device=self.device)

        with torch.no_grad():
            output_ids = self.model.generate(
                input_ids,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                do_sample=True,
            )

        generated = self.tokenizer.decode(
            output_ids[0].tolist(), skip_special_tokens=True
        )
        return extract_assistant_response(generated)

    def generate_stream(
        self,
        prompt: str,
        max_new_tokens: int = 256,
        temperature: float = 0.7,
        top_k: int = 50,
        language: str = "python",
    ) -> Generator[str, None, None]:
        if self.model is None or self.tokenizer is None:
            raise ValueError("Model not loaded")

        full_prompt = build_instruction_prompt(user=prompt, language=language, include_eos=False)
        tokens = self.tokenizer.encode(full_prompt, add_special_tokens=False)
        input_ids = torch.tensor([tokens], device=self.device)

        generated_tokens: List[int] = []

        with torch.no_grad():
            for _ in range(max_new_tokens):
                outputs = self.model(
                    input_ids=torch.cat(
                        [input_ids, torch.tensor([generated_tokens], device=self.device)],
                        dim=1,
                    )
                    if generated_tokens
                    else input_ids,
                    attention_mask=None,
                )

                logits = outputs["logits"][:, -1, :] / temperature

                if top_k > 0:
                    indices_to_remove = (
                        logits < torch.topk(logits, top_k)[0][..., -1, None]
                    )
                    logits[indices_to_remove] = float("-inf")

                probs = torch.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
                token_id = next_token.item()
                generated_tokens.append(token_id)

                token_text = self.tokenizer.decode([token_id], skip_special_tokens=True)
                if token_id == self.tokenizer.token_to_id("<|eos|>"):
                    break
                yield token_text

    def save_checkpoint(self, path: Optional[Path] = None) -> bool:
        if self.model is None or self.tokenizer is None:
            self.logger.error("Model or tokenizer not loaded. Cannot save checkpoint.")
            return False

        try:
            checkpoint_dir = self.config.get_path("codemind.checkpoint_dir", "codemind/checkpoints")
            checkpoint_path = path or checkpoint_dir / "model_final.pt"
            checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

            state_dict = self.model.state_dict()
            checkpoint = {"model_state_dict": state_dict}

            config_dict = getattr(self.model, "config", None)
            config_data = config_dict.to_dict() if config_dict else {}

            metadata = build_checkpoint_metadata(
                model_config=config_data,
                tokenizer=self.tokenizer,
                tokenizer_type="pretrained" if self.config.get("model.pretrained_tokenizer") else "codemind",
                architecture_version="codemind-v2",
                pretrained_tokenizer_name=self.config.get("model.pretrained_tokenizer", "")
            )
            checkpoint = attach_checkpoint_metadata(checkpoint, metadata)

            torch.save(checkpoint, checkpoint_path)
            self.logger.info(f"Checkpoint saved to {checkpoint_path}")
            return True
        except Exception as e:
            self.logger.error(f"Failed to save checkpoint: {e}")
            return False

    def unload_model(self) -> None:
        if self.model is not None:
            del self.model
            self.model = None

        if self.tokenizer is not None:
            del self.tokenizer
            self.tokenizer = None

        self.last_compatibility_report = None

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        self.logger.info("CodeMind model unloaded")
