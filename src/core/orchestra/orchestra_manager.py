"""
OrchestraManager — top-level facade for the Orchestra system.

Handles:
  - Orchestrator (routing model)
  - Expert loading on demand (lazy loading)
  - LRU cache: last N experts kept in RAM to avoid repeated disk I/O
  - Response generation through the selected expert(s)

Memory strategy for GTX 1650 / limited VRAM:
  - Orchestrator model (350M, quantized): stays in VRAM permanently (~1 GB)
  - Expert models: loaded to CPU, moved to VRAM only during inference,
    then moved back to CPU or evicted from LRU cache.
  - Max RAM occupied: max_cached_experts * model_size_in_ram

Usage:
    manager = OrchestraManager(
        registry=ExpertRegistry("codemind/experts"),
        orchestrator_model=router_model,
        max_cached_experts=2,
    )
    response = manager.ask("Flutter'da BLoC nasil kullanilir?")
"""

from __future__ import annotations

import gc
from collections import OrderedDict
from typing import Any, Dict, List, Optional, Tuple

import torch

from src.core.orchestra.expert_registry import ExpertDomain, ExpertInfo, ExpertRegistry
from src.core.orchestra.orchestrator import Orchestrator
from src.core.model_manager import ModelManager
from src.utils.config import Config
from src.utils.logger import get_logger

logger = get_logger("orchestra_manager")


class OrchestraManager:
    """Manages routing and lazy-loaded expert models for the Orchestra system.

    Args:
        registry:            ExpertRegistry with checkpoint paths.
        orchestrator_model:  Optional pre-loaded routing model.
        orchestrator_tokenizer: Tokenizer for the routing model.
        max_cached_experts:  Number of expert models to keep in RAM (LRU).
        device:              Primary device for inference.
        generation_kwargs:   Default kwargs passed to expert model.generate().
    """

    def __init__(
        self,
        registry: ExpertRegistry,
        orchestrator_model: Optional[torch.nn.Module] = None,
        orchestrator_tokenizer=None,
        max_cached_experts: int = 2,
        device: Optional[str] = None,
        generation_kwargs: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.registry = registry
        self.device   = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.generation_kwargs = generation_kwargs or {
            "max_new_tokens": 512,
            "temperature": 0.7,
            "do_sample": True,
            "top_p": 0.95,
        }

        # Orchestrator (always resident in memory)
        self.orchestrator = Orchestrator(
            registry=registry,
            model=orchestrator_model,
            tokenizer=orchestrator_tokenizer,
            device=self.device,
        )

        # LRU cache: OrderedDict preserves insertion order
        # key=ExpertDomain, value=(model, tokenizer)
        self._cache: OrderedDict[ExpertDomain, Tuple[torch.nn.Module, Any]] = OrderedDict()
        self._max_cached = max_cached_experts

        logger.info(
            "OrchestraManager initialized | device=%s max_cached_experts=%d",
            self.device,
            self._max_cached,
        )

    # ── Public API ──────────────────────────────────────────────────────────────

    def ask(
        self,
        query: str,
        mode: str = "single",   # "single" | "ensemble" | "pipeline"
        top_k: int = 2,
    ) -> str:
        """Route query and generate a response.

        Args:
            query: User query string.
            mode:  "single"   → one expert responds.
                   "ensemble" → top_k experts respond; answers are merged.
                   "pipeline" → expert1 refines the query, expert2 answers.
            top_k: Number of experts used in ensemble/pipeline mode.
        """
        if mode == "ensemble":
            domains = self.orchestrator.route_ensemble(query, top_k=top_k)
            responses = [self._generate(d, query) for d in domains]
            return self._merge_ensemble(responses, domains)

        if mode == "pipeline":
            domains = self.orchestrator.route_ensemble(query, top_k=2)
            # Stage 1: First expert elaborates / reformulates
            elaborated = self._generate(domains[0], query)
            # Stage 2: Second expert answers the elaborated query
            final_query = f"{query}\n\n[Context from prior analysis]:\n{elaborated}"
            return self._generate(domains[1], final_query)

        # Single mode (default)
        domain = self.orchestrator.route(query)
        logger.info("Routing: %r → %s", query[:60], domain.value)
        return self._generate(domain, query)

    def load_expert(self, domain: ExpertDomain) -> Optional[Tuple[torch.nn.Module, Any]]:
        """Explicitly load an expert into the LRU cache."""
        return self._get_expert(domain)

    def evict_expert(self, domain: ExpertDomain) -> None:
        """Remove an expert from the cache (free RAM)."""
        if domain in self._cache:
            model, _ = self._cache.pop(domain)
            del model
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            logger.info("Expert evicted from cache: %s", domain.value)

    def cache_info(self) -> Dict[str, Any]:
        """Returns current cache state for debugging."""
        return {
            "cached_experts": [d.value for d in self._cache],
            "max_cached":     self._max_cached,
            "device":         self.device,
        }

    # ── Private helpers ─────────────────────────────────────────────────────────

    def _get_expert(
        self, domain: ExpertDomain
    ) -> Optional[Tuple[torch.nn.Module, Any]]:
        """Return (model, tokenizer) for domain. Loads and caches on first access."""
        if domain in self._cache:
            # Move to end = mark as most recently used
            self._cache.move_to_end(domain)
            return self._cache[domain]

        info: ExpertInfo = self.registry.get(domain)
        if not info.available:
            logger.warning(
                "Expert '%s' not available (no checkpoint). Falling back to GENERAL.",
                domain.value,
            )
            if domain != ExpertDomain.GENERAL:
                return self._get_expert(ExpertDomain.GENERAL)
            return None

        # Load the expert via ModelManager (handles .pt + HF formats)
        logger.info("Loading expert: %s from %s", domain.value, info.checkpoint_path)
        config = Config()
        config.set("model.device", self.device)
        config.set("model.load_in_4bit", False)
        manager = ModelManager(config)

        try:
            manager.load_model(model_path=info.checkpoint_path)
            entry = (manager.model, manager.tokenizer)

            # Evict LRU if over capacity
            if len(self._cache) >= self._max_cached:
                evicted_domain, (evicted_model, _) = self._cache.popitem(last=False)
                logger.info("LRU evict: %s", evicted_domain.value)
                del evicted_model
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            self._cache[domain] = entry
            logger.info("Expert cached: %s | cache size=%d", domain.value, len(self._cache))
            return entry

        except Exception as exc:
            logger.error("Failed to load expert '%s': %s", domain.value, exc)
            return None

    def _generate(self, domain: ExpertDomain, query: str) -> str:
        """Generate a response from the given expert domain."""
        entry = self._get_expert(domain)
        if entry is None:
            return f"[Orchestra] {domain.value} uzmanı şu an mevcut değil."

        model, tokenizer = entry
        model.eval()

        try:
            inputs = tokenizer(
                query,
                return_tensors="pt",
                truncation=True,
                max_length=1024,
            )
            input_ids = inputs["input_ids"].to(self.device)

            with torch.inference_mode():
                output_ids = model.generate(
                    input_ids,
                    **self.generation_kwargs,
                    pad_token_id=tokenizer.eos_token_id,
                )

            generated = tokenizer.decode(
                output_ids[0][input_ids.shape[1]:],
                skip_special_tokens=True,
            )
            return generated.strip()

        except Exception as exc:
            logger.error("Expert '%s' generation failed: %s", domain.value, exc)
            return f"[Orchestra] {domain.value} uzmanından yanıt alınamadı: {exc}"

    def _merge_ensemble(
        self,
        responses: List[str],
        domains: List[ExpertDomain],
    ) -> str:
        """Simple ensemble merge strategy: concatenate with attribution."""
        if len(responses) == 1:
            return responses[0]
        parts = []
        for domain, response in zip(domains, responses):
            parts.append(f"[{domain.value}]\n{response}")
        return "\n\n---\n\n".join(parts)
