"""
Orchestrator — routes incoming queries to the best expert(s).

Two routing modes:
  1. LM-based routing (preferred): orchestrator model classifies the query
     and returns an ExpertDomain label.
  2. Keyword fallback: fast rule-based routing used when no model is loaded.
"""

from __future__ import annotations

import re
from typing import List, Optional

import torch

from src.core.orchestra.expert_registry import ExpertDomain, ExpertRegistry
from src.utils.logger import get_logger

logger = get_logger("orchestrator")

# Prompt template for the routing model
_ROUTE_PROMPT = (
    "[SYSTEM] Sen bir AI yönlendirme sistemisin. "
    "Kullanicinin sorusunu analiz et ve en uygun uzman kategorisini sec.\n"
    "Kategoriler: python_code, dart_flutter, sql_database, linux_system, "
    "turkish_chat, mathematics, analysis, creative, security, general\n"
    "Sadece kategori adini yaz, baska hicbir sey yazma.\n\n"
    "[USER] {query}\n"
    "[ASSISTANT]"
)


class Orchestrator:
    """Routes queries to the correct expert domain.

    Args:
        registry:         ExpertRegistry to use for keyword fallback.
        model:            Optional language model for LM-based routing.
        tokenizer:        Tokenizer for the routing model.
        device:           Device for routing model inference.
        confidence_thresh: Below this softmax confidence, fall back to keywords.
    """

    def __init__(
        self,
        registry: ExpertRegistry,
        model: Optional[torch.nn.Module] = None,
        tokenizer=None,
        device: Optional[str] = None,
        confidence_thresh: float = 0.5,
    ) -> None:
        self.registry           = registry
        self.model              = model
        self.tokenizer          = tokenizer
        self.device             = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.confidence_thresh  = confidence_thresh

        if self.model is not None:
            self.model.to(self.device).eval()
            logger.info("Orchestrator initialized with LM-based routing.")
        else:
            logger.info("Orchestrator initialized with keyword-only routing.")

    # ── Public API ─────────────────────────────────────────────────────────────

    def route(self, query: str) -> ExpertDomain:
        """Return the best expert domain for the query."""
        if self.model is not None and self.tokenizer is not None:
            domain = self._lm_route(query)
            if domain is not None:
                return domain
        # Fallback to keyword routing
        domain = self.registry.keyword_route(query)
        logger.debug("Keyword route: %s → %s", query[:40], domain.value)
        return domain

    def route_ensemble(self, query: str, top_k: int = 2) -> List[ExpertDomain]:
        """Return multiple experts (for ensemble / pipeline mode).

        The primary domain is always first; secondary domains are selected
        by keyword scores on remaining categories.
        """
        primary = self.route(query)
        query_lower = query.lower()
        scores = {}
        for info in self.registry.list_all():
            if info.domain == primary:
                continue
            score = sum(1 for kw in info.keywords if kw in query_lower)
            if score > 0:
                scores[info.domain] = score

        secondaries = sorted(scores, key=lambda d: scores[d], reverse=True)[: top_k - 1]
        return [primary] + secondaries

    # ── Private helpers ────────────────────────────────────────────────────────

    @torch.inference_mode()
    def _lm_route(self, query: str) -> Optional[ExpertDomain]:
        """Use the routing LM to classify the query. Returns None on failure."""
        prompt = _ROUTE_PROMPT.format(query=query[:500])
        try:
            inputs = self.tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=512,
            ).to(self.device)
            out = self.model.generate(
                **inputs,
                max_new_tokens=10,
                do_sample=False,
                temperature=1.0,
                pad_token_id=self.tokenizer.eos_token_id,
            )
            # Decode only the new tokens
            generated = self.tokenizer.decode(
                out[0][inputs["input_ids"].shape[1]:],
                skip_special_tokens=True,
            ).strip().lower()

            # Extract first word and match to domain
            first_word = re.split(r"[\s,\.\n]", generated)[0]
            for domain in ExpertDomain:
                if domain.value == first_word or domain.value.replace("_", "") == first_word.replace("_", ""):
                    logger.debug("LM route: %s → %s (raw: %r)", query[:40], domain.value, generated)
                    return domain

            logger.warning("LM route: could not parse domain from %r", generated)
            return None

        except Exception as exc:
            logger.warning("LM routing failed: %s", exc)
            return None
