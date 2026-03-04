import os
from pathlib import Path
from typing import Any, Dict, Generator, List, Optional, Tuple
import torch
from transformers import (
    TextIteratorStreamer,
    StoppingCriteria,
    StoppingCriteriaList,
)
from threading import Thread

import sys


from src.utils.config import Config
from src.utils.logger import get_logger
from src.core.model_manager import ModelManager
from src.core.memory import MemoryManager
from transformers.cache_utils import DynamicCache
from src.core.prompting import TOKENS, build_chat_prompt, extract_assistant_response
from src.core.cognitive.router import CognitiveRouter
from src.core.cognitive.modes import CognitiveMode
from src.core.cognitive.reward_model import DualRewardEvaluator
from src.core.cognitive.reasoning_engine import ReasoningEngine

def _patch_dynamic_cache():
    if not hasattr(DynamicCache, "seen_tokens"):
        DynamicCache.seen_tokens = property(lambda self: self.get_seq_length())
    if not hasattr(DynamicCache, "get_max_length"):
        DynamicCache.get_max_length = lambda self, layer_idx=None: getattr(self, "_max_length", None)
    if not hasattr(DynamicCache, "get_usable_length"):
        DynamicCache.get_usable_length = lambda self, seq_length, layer_idx=0: seq_length + self.get_seq_length(layer_idx)

_patch_dynamic_cache()


class StopOnTokens(StoppingCriteria):
    def __init__(self, stop_token_ids: List[int]):
        self.stop_token_ids = stop_token_ids

    def __call__(
        self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs
    ) -> bool:
        for stop_id in self.stop_token_ids:
            if input_ids[0][-1] == stop_id:
                return True
        return False


class InferenceEngine:
    def __init__(
        self,
        model_manager: ModelManager,
        memory_manager: Optional[MemoryManager] = None,
        config: Optional[Config] = None,
    ):
        self.model_manager = model_manager
        self.memory_manager = memory_manager
        self.config = config or Config()
        self.logger = get_logger("inference")

        # Read generation parameters from config (with sensible defaults)
        self._load_settings_from_config()

        # Cognitive Engine Components
        self.router = CognitiveRouter()
        self.reward_model = DualRewardEvaluator()  # Base heuristic reward initially

        # Experience Buffer for saving successful System 2 thought chains
        self.experience_buffer: Optional[Any] = None
        if self.memory_manager:
            try:
                from src.core.cognitive.experience_buffer import ExperienceBuffer
                self.experience_buffer = ExperienceBuffer(self.memory_manager)
            except Exception:
                self.logger.debug("ExperienceBuffer başlatılamadı (isteğe bağlı).")

    def _load_settings_from_config(self) -> None:
        """Load all generation parameters from the config file."""
        default_prompt = (
            "Sen yardımsever, bilgili ve nazik bir AI asistanısın. "
            "Kullanıcılara Türkçe ve diğer dillerde yardımcı oluyorsun. "
            "Sorulara detaylı ve doğru cevaplar veriyorsun. "
            'Bilmediğin konularda dürüstçe "bilmiyorum" diyorsun.'
        )
        self.system_prompt = self.config.get("inference.system_prompt", default_prompt)
        self.max_new_tokens = self.config.get("generation.max_new_tokens", 2048)
        self.temperature = self.config.get("generation.temperature", 0.7)
        self.top_p = self.config.get("generation.top_p", 0.95)
        self.top_k = self.config.get("generation.top_k", 50)
        self.repetition_penalty = self.config.get("generation.repetition_penalty", 1.1)

    def update_settings(
        self,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        top_k: Optional[int] = None,
        max_new_tokens: Optional[int] = None,
        repetition_penalty: Optional[float] = None,
        system_prompt: Optional[str] = None,
    ) -> None:
        """Update generation parameters at runtime (called by Model Hub UI)."""
        if temperature is not None:
            self.temperature = temperature
        if top_p is not None:
            self.top_p = top_p
        if top_k is not None:
            self.top_k = top_k
        if max_new_tokens is not None:
            self.max_new_tokens = max_new_tokens
        if repetition_penalty is not None:
            self.repetition_penalty = repetition_penalty
        if system_prompt is not None:
            self.system_prompt = system_prompt
        self.logger.info("Inference ayarları canlı olarak güncellendi.")

    def _lm_generate_multiple(self, context: str, n_samples: int) -> List[str]:
        """Wrapper to generate multiple subsequent possible thoughts for MCTS."""
        # Simple implementation for expansion: repeatedly call generate or use beam search conceptually
        thoughts = []
        # For efficiency, we just do a loop with high temperature to get diverse steps.
        # Ensure we don't infinitely recurse System 2 logic by using internal generate
        for _ in range(n_samples):
             # Fast internal generation
             response = self._fast_generate(context, max_new_tokens=150, temperature=0.9)
             thoughts.append(response)
        return thoughts

    def _fast_generate(self, user_input: str, **generation_kwargs) -> str:
        """Internal fast generation completely skipping System 2."""
        prompt = self.format_prompt(user_input)
        inputs = self.model_manager.tokenizer(
            prompt, return_tensors="pt", truncation=True, max_length=4096
        )
        if self.model_manager.device == "cuda":
            inputs = {k: v.cuda() for k, v in inputs.items()}
            
        stop_tokens = [self.model_manager.tokenizer.eos_token_id]
        with torch.no_grad():
            outputs = self.model_manager.model.generate(
                **inputs,
                max_new_tokens=generation_kwargs.get("max_new_tokens", self.max_new_tokens),
                temperature=generation_kwargs.get("temperature", self.temperature),
                do_sample=True,
                pad_token_id=self.model_manager.tokenizer.pad_token_id,
            )
        generated_text = self.model_manager.tokenizer.decode(
            outputs[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True
        )
        return extract_assistant_response(generated_text)

    def format_prompt(
        self,
        user_input: str,
        context: Optional[str] = None,
        history: Optional[List[Dict[str, str]]] = None,
    ) -> str:
        return build_chat_prompt(
            user_input=user_input,
            system_prompt=self.system_prompt,
            context=context,
            history=history,
            history_turns=5,
            append_assistant_token=True,
        )

    def generate(
        self,
        user_input: str,
        use_memory: bool = True,
        history: Optional[List[Dict[str, str]]] = None,
        mode_callback: Optional[callable] = None,
        **generation_kwargs,
    ) -> str:
        if self.model_manager.model is None or self.model_manager.tokenizer is None:
            raise ValueError("Model yüklenmemiş")

        if self.model_manager.is_codemind:
            return self._generate_codemind(user_input, **generation_kwargs)

        context = None
        if use_memory and self.memory_manager:
            context = self.memory_manager.get_relevant_context(user_input)

        # -------------------------------------------------------------
        # System 1 / System 2 Routing
        # -------------------------------------------------------------
        mode_config = self.router.route(user_input)
        
        simulations = mode_config.simulations_per_step * mode_config.max_depth if mode_config.use_mcts else 0
        if mode_callback:
            mode_callback(mode_config.mode.value, simulations)

        if mode_config.use_mcts:
            self.logger.info(f"System 2 devrede: {mode_config.mode.value}. Derin düşünülüyor...")
            engine = ReasoningEngine(
                language_model_generate=self._lm_generate_multiple,
                reward_evaluator=self.reward_model.evaluate,
                max_depth=mode_config.max_depth,
                simulations_per_step=mode_config.simulations_per_step,
                branching_factor=mode_config.branching_factor
            )
            initial_state = self.format_prompt(user_input, context, history)
            result = engine.search(initial_state)

            # Save successful System 2 thoughts to ExperienceBuffer
            if self.experience_buffer and result:
                avg_reward = self.reward_model.evaluate(initial_state, result)
                self.experience_buffer.add_experience(
                    query=user_input, thought_chain=result, reward=avg_reward,
                )

            return result
            
        self.logger.info("System 1 devrede. Hızlı yanıt üretiliyor...")
        prompt = self.format_prompt(user_input, context, history)

        inputs = self.model_manager.tokenizer(
            prompt, return_tensors="pt", truncation=True, max_length=4096
        )

        if self.model_manager.device == "cuda":
            inputs = {k: v.cuda() for k, v in inputs.items()}
        elif self.model_manager.device == "mps":
            inputs = {k: v.to("mps") for k, v in inputs.items()}

        stop_tokens = [
            self.model_manager.tokenizer.eos_token_id,
            self.model_manager.tokenizer.convert_tokens_to_ids(TOKENS.user),
            self.model_manager.tokenizer.convert_tokens_to_ids(TOKENS.system),
        ]
        stop_tokens = [t for t in stop_tokens if t is not None]

        with torch.no_grad():
            outputs = self.model_manager.model.generate(
                **inputs,
                max_new_tokens=generation_kwargs.get(
                    "max_new_tokens", self.max_new_tokens
                ),
                temperature=generation_kwargs.get("temperature", self.temperature),
                top_p=generation_kwargs.get("top_p", self.top_p),
                top_k=generation_kwargs.get("top_k", self.top_k),
                repetition_penalty=generation_kwargs.get(
                    "repetition_penalty", self.repetition_penalty
                ),
                do_sample=True,
                pad_token_id=self.model_manager.tokenizer.pad_token_id,
                stopping_criteria=StoppingCriteriaList([StopOnTokens(stop_tokens)]),
            )

        generated_text = self.model_manager.tokenizer.decode(
            outputs[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True
        )

        response = extract_assistant_response(generated_text)

        if use_memory and self.memory_manager:
            self.memory_manager.add_conversation(user_input, response)

        return response

    def _generate_codemind(self, user_input: str, **generation_kwargs) -> str:
        if not self.model_manager.codemind_adapter:
            raise ValueError("CodeMind adapter not initialized")

        language = "python"
        user_lower = user_input.lower()
        if "flutter" in user_lower or "dart" in user_lower:
            language = "dart"
        elif "javascript" in user_lower or "js" in user_lower:
            language = "javascript"

        return self.model_manager.codemind_adapter.generate(
            prompt=user_input,
            max_new_tokens=generation_kwargs.get("max_new_tokens", 512),
            temperature=generation_kwargs.get("temperature", self.temperature),
            top_k=generation_kwargs.get("top_k", self.top_k),
            top_p=generation_kwargs.get("top_p", self.top_p),
            language=language,
        )

    def generate_stream(
        self,
        user_input: str,
        use_memory: bool = True,
        history: Optional[List[Dict[str, str]]] = None,
        mode_callback: Optional[callable] = None,
        **generation_kwargs,
    ) -> Generator[str, None, None]:
        if self.model_manager.model is None or self.model_manager.tokenizer is None:
            raise ValueError("Model yüklenmemiş")

        if self.model_manager.is_codemind:
            response = self._generate_codemind(user_input, **generation_kwargs)
            for char in response:
                yield char
            return

        context = None
        if use_memory and self.memory_manager:
            context = self.memory_manager.get_relevant_context(user_input)
            
        mode_config = self.router.route(user_input)
        simulations = mode_config.simulations_per_step * mode_config.max_depth if mode_config.use_mcts else 0
        if mode_callback:
            mode_callback(mode_config.mode.value, simulations)

        if mode_config.use_mcts:
            self.logger.info(f"System 2 devrede: {mode_config.mode.value}. Derin düşünülüyor...")
            engine = ReasoningEngine(
                language_model_generate=self._lm_generate_multiple,
                reward_evaluator=self.reward_model.evaluate,
                max_depth=mode_config.max_depth,
                simulations_per_step=mode_config.simulations_per_step,
                branching_factor=mode_config.branching_factor
            )
            initial_state = self.format_prompt(user_input, context, history)
            
            # Since MCTS returns the final output, stream it by yielding the whole text
            final_response = engine.search(initial_state)
            yield final_response

            # Save successful System 2 thoughts to ExperienceBuffer
            if self.experience_buffer and final_response:
                avg_reward = self.reward_model.evaluate(initial_state, final_response)
                self.experience_buffer.add_experience(
                    query=user_input, thought_chain=final_response, reward=avg_reward,
                )
            
            if use_memory and self.memory_manager:
                self.memory_manager.add_conversation(user_input, final_response.strip())
            return
            
        self.logger.info("System 1 devrede. Hızlı yanıt üretiliyor...")

        prompt = self.format_prompt(user_input, context, history)

        inputs = self.model_manager.tokenizer(
            prompt, return_tensors="pt", truncation=True, max_length=4096
        )

        if self.model_manager.device == "cuda":
            inputs = {k: v.cuda() for k, v in inputs.items()}
        elif self.model_manager.device == "mps":
            inputs = {k: v.to("mps") for k, v in inputs.items()}

        streamer = TextIteratorStreamer(
            self.model_manager.tokenizer, skip_prompt=True, skip_special_tokens=True
        )

        stop_tokens = [
            self.model_manager.tokenizer.eos_token_id,
            self.model_manager.tokenizer.convert_tokens_to_ids(TOKENS.user),
        ]
        stop_tokens = [t for t in stop_tokens if t is not None]

        generation_kwargs = {
            **inputs,
            "max_new_tokens": generation_kwargs.get(
                "max_new_tokens", self.max_new_tokens
            ),
            "temperature": generation_kwargs.get("temperature", self.temperature),
            "top_p": generation_kwargs.get("top_p", self.top_p),
            "top_k": generation_kwargs.get("top_k", self.top_k),
            "repetition_penalty": generation_kwargs.get(
                "repetition_penalty", self.repetition_penalty
            ),
            "do_sample": True,
            "streamer": streamer,
            "pad_token_id": self.model_manager.tokenizer.pad_token_id,
        }

        thread = Thread(
            target=self.model_manager.model.generate, kwargs=generation_kwargs
        )
        thread.start()

        full_response = ""
        for text in streamer:
            if TOKENS.user in text:
                text = text.split(TOKENS.user)[0]
            full_response += text
            yield text

        if use_memory and self.memory_manager:
            self.memory_manager.add_conversation(user_input, full_response.strip())

    def summarize(self, text: str, max_length: int = 200) -> str:
        prompt = f"Aşağıdaki metni {max_length} kelimeyi geçmeyecek şekilde özetle:\n\n{text}\n\nÖzet:"

        return self.generate(prompt, use_memory=False)

    def translate(
        self, text: str, source_lang: str = "auto", target_lang: str = "tr"
    ) -> str:
        prompt = f"Çevir: {source_lang} -> {target_lang}\n\n{text}\n\nÇeviri:"

        return self.generate(prompt, use_memory=False)

    def classify_text(self, text: str, categories: List[str]) -> Tuple[str, float]:
        categories_str = ", ".join(categories)
        prompt = f"""Aşağıdaki metni verilen kategorilerden birine sınıflandır.

Kategoriler: {categories_str}

Metin: {text}

Kategori:"""

        response = self.generate(prompt, use_memory=False, max_new_tokens=50)

        for category in categories:
            if category.lower() in response.lower():
                return category, 0.9

        return categories[0], 0.5

    def set_parameters(
        self,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        top_k: Optional[int] = None,
        max_new_tokens: Optional[int] = None,
        repetition_penalty: Optional[float] = None,
    ) -> None:
        if temperature is not None:
            self.temperature = temperature
        if top_p is not None:
            self.top_p = top_p
        if top_k is not None:
            self.top_k = top_k
        if max_new_tokens is not None:
            self.max_new_tokens = max_new_tokens
        if repetition_penalty is not None:
            self.repetition_penalty = repetition_penalty

        self.logger.info(
            f"Parametreler güncellendi: temp={self.temperature}, top_p={self.top_p}"
        )

    def get_parameters(self) -> Dict[str, Any]:
        return {
            "temperature": self.temperature,
            "top_p": self.top_p,
            "top_k": self.top_k,
            "max_new_tokens": self.max_new_tokens,
            "repetition_penalty": self.repetition_penalty,
        }
