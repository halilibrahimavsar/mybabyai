"""
Latent Orchestra — the main conductor that orchestrates experts, router, and bridge.

This is the top-level API for the Latent Expert Orchestra system.
It manages the full inference pipeline:

  1. Tokenize user prompt
  2. Route via NeuralRouter → RoutingDecision
  3. Execute in SINGLE or CASCADE mode
  4. Return generated text

Execution Modes:
  - SINGLE: Route to best expert, generate directly.
  - CASCADE: Expert A produces hidden states → Bridge projects to
    Expert B's space → Expert B generates conditioned on bridge output.

Usage:
    orchestra = LatentOrchestra(tokenizer=shared_tokenizer)
    orchestra.load_expert("turkish", "checkpoints/turkish_expert.pt")
    orchestra.load_expert("code", "checkpoints/code_expert.pt")
    response = orchestra.generate("Bana Python'da Fibonacci yaz ve açıkla")
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import torch

from src.core.orchestra.expert_config import (
    ExpertConfig,
    ExpertDomain,
    EXPERT_CONFIGS,
    SHARED_HIDDEN_SIZE,
)
from src.core.orchestra.expert_model import ExpertModel
from src.core.orchestra.latent_bridge import BridgePool
from src.core.orchestra.neural_router import (
    ExecutionMode,
    KeywordRouterFallback,
    NeuralRouter,
    RoutingDecision,
)

logger = logging.getLogger(__name__)


class LatentOrchestra:
    """Main conductor for the Latent Expert Orchestra.

    Manages experts, router, and bridge to produce coordinated responses
    from specialized models communicating via latent representations.

    Args:
        tokenizer: Shared tokenizer used by all experts and the router.
        device: Target device for all models.
        use_neural_router: If True, use the transformer-based neural router.
            If False, fall back to keyword-based routing.
        bridge_bottleneck: Bottleneck dimension for the latent bridge.
        router_d_model: Hidden dimension of the neural router.
        cascade_threshold: Weight gap threshold for triggering CASCADE mode.
    """

    def __init__(
        self,
        tokenizer: Any,
        device: Optional[str] = None,
        use_neural_router: bool = True,
        bridge_bottleneck: int = 512,
        router_d_model: int = 256,
        cascade_threshold: float = 0.3,
    ) -> None:
        self.tokenizer = tokenizer
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.use_neural_router = use_neural_router

        # Expert registry: domain → ExpertModel
        self._experts: Dict[ExpertDomain, ExpertModel] = {}

        # Bridge pool (initialized lazily when experts are loaded)
        self._bridge: Optional[BridgePool] = None
        self._bridge_bottleneck = bridge_bottleneck

        # Router
        self._keyword_router = KeywordRouterFallback()
        self._neural_router: Optional[NeuralRouter] = None
        self._router_d_model = router_d_model
        self._cascade_threshold = cascade_threshold

        logger.info(
            "LatentOrchestra initialized (device=%s, neural_router=%s)",
            self.device, use_neural_router,
        )

    # ── Expert Management ──────────────────────────────────────────────────────

    def create_expert(
        self,
        domain: ExpertDomain,
        config: Optional[ExpertConfig] = None,
    ) -> ExpertModel:
        """Create a fresh (untrained) expert model.

        Args:
            domain: Expert domain.
            config: Optional override config. Uses EXPERT_CONFIGS default if None.

        Returns:
            The created ExpertModel instance.
        """
        cfg = config or EXPERT_CONFIGS[domain]
        expert = ExpertModel(config=cfg, device=self.device)
        self._experts[domain] = expert
        self._update_bridge_and_router()
        logger.info("Expert[%s] created and registered.", domain.value)
        return expert

    def load_expert(
        self,
        domain: ExpertDomain,
        checkpoint_path: Union[str, Path],
    ) -> ExpertModel:
        """Load trained expert from checkpoint.

        Args:
            domain: Expert domain identifier.
            checkpoint_path: Path to .pt file or HF-style directory.

        Returns:
            The loaded ExpertModel instance.
        """
        expert = ExpertModel.from_checkpoint(
            domain=domain,
            checkpoint_path=checkpoint_path,
            device=self.device,
        )
        self._experts[domain] = expert
        self._update_bridge_and_router()
        logger.info("Expert[%s] loaded from %s", domain.value, checkpoint_path)
        return expert

    def get_expert(self, domain: ExpertDomain) -> Optional[ExpertModel]:
        """Get a loaded expert by domain."""
        return self._experts.get(domain)

    def list_experts(self) -> List[ExpertDomain]:
        """List all loaded expert domains."""
        return list(self._experts.keys())

    # ── Router Management ──────────────────────────────────────────────────────

    def load_router(self, checkpoint_path: Union[str, Path]) -> None:
        """Load a pre-trained neural router from checkpoint.

        Args:
            checkpoint_path: Path to router .pt checkpoint.
        """
        if self._neural_router is None:
            self._initialize_neural_router()

        state_dict = torch.load(
            str(checkpoint_path), map_location="cpu", weights_only=False,
        )
        if "model_state_dict" in state_dict:
            state_dict = state_dict["model_state_dict"]
        self._neural_router.load_state_dict(state_dict, strict=False)
        self._neural_router.to(self.device)
        self._neural_router.eval()
        logger.info("Neural router loaded from %s", checkpoint_path)

    # ── Bridge Management ──────────────────────────────────────────────────────

    def load_bridge(self, checkpoint_path: Union[str, Path]) -> None:
        """Load a pre-trained bridge from checkpoint.

        Args:
            checkpoint_path: Path to bridge .pt checkpoint.
        """
        if self._bridge is None:
            self._initialize_bridge()

        state_dict = torch.load(
            str(checkpoint_path), map_location="cpu", weights_only=False,
        )
        if "model_state_dict" in state_dict:
            state_dict = state_dict["model_state_dict"]
        self._bridge.load_state_dict(state_dict, strict=False)
        self._bridge.to(self.device)
        logger.info("Bridge pool loaded from %s", checkpoint_path)

    # ── Main Inference Pipeline ────────────────────────────────────────────────

    @torch.inference_mode()
    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 256,
        temperature: float = 0.7,
        top_k: int = 50,
        top_p: float = 0.95,
        force_mode: Optional[ExecutionMode] = None,
        force_expert: Optional[ExpertDomain] = None,
    ) -> Dict[str, Any]:
        """Generate a response for the given prompt.

        Full pipeline: tokenize → route → execute → decode.

        Args:
            prompt: User's input text.
            max_new_tokens: Maximum tokens to generate.
            temperature: Sampling temperature.
            top_k: Top-K sampling.
            top_p: Top-P sampling.
            force_mode: Override router's execution mode decision.
            force_expert: Override router's expert selection.

        Returns:
            Dict with keys:
                - "text": Generated response text.
                - "routing": RoutingDecision details.
                - "execution_mode": Actual execution mode used.
                - "experts_used": List of expert domains that participated.
                - "latent_bridge_used": Whether bridge was used.
        """
        if not self._experts:
            raise RuntimeError("No experts loaded. Call load_expert() first.")

        # 1. Tokenize
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=1024,
            padding=True,
        )
        input_ids = inputs["input_ids"].to(self.device)
        attention_mask = inputs.get("attention_mask")
        if attention_mask is not None:
            attention_mask = attention_mask.to(self.device)

        # 2. Route
        routing = self._route(prompt, input_ids, attention_mask)

        # Apply overrides
        if force_expert is not None:
            routing.primary_expert = force_expert
            routing.execution_mode = ExecutionMode.SINGLE
        if force_mode is not None:
            routing.execution_mode = force_mode

        # 3. Execute
        mode = routing.execution_mode
        if mode == ExecutionMode.CASCADE and self._bridge is not None:
            result = self._execute_cascade(
                input_ids, attention_mask, routing,
                max_new_tokens, temperature, top_k, top_p,
            )
        else:
            # Fall back to SINGLE if bridge not available
            if mode == ExecutionMode.CASCADE:
                logger.warning(
                    "CASCADE requested but bridge not loaded. "
                    "Falling back to SINGLE mode."
                )
            result = self._execute_single(
                input_ids, attention_mask, routing,
                max_new_tokens, temperature, top_k, top_p,
            )

        return result

    # ── Text Relay Baseline ────────────────────────────────────────────────────

    @torch.inference_mode()
    def generate_text_relay(
        self,
        prompt: str,
        max_new_tokens: int = 256,
        temperature: float = 0.7,
    ) -> Dict[str, Any]:
        """Generate using text-to-text relay (baseline for comparison).

        Expert A generates text → Expert B receives text as input.
        This is the baseline against which latent bridge is measured.
        """
        if not self._experts:
            raise RuntimeError("No experts loaded.")

        inputs = self.tokenizer(
            prompt, return_tensors="pt", truncation=True, max_length=1024,
        )
        input_ids = inputs["input_ids"].to(self.device)
        attention_mask = inputs.get("attention_mask")
        if attention_mask is not None:
            attention_mask = attention_mask.to(self.device)

        routing = self._route(prompt, input_ids, attention_mask)

        if routing.execution_mode == ExecutionMode.CASCADE and routing.secondary_expert:
            # Step 1: Primary expert generates text
            primary = self._experts.get(routing.primary_expert)
            if primary is None:
                raise RuntimeError(f"Expert {routing.primary_expert} not loaded.")

            gen_ids = primary.generate(
                input_ids, attention_mask,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
            )
            intermediate_text = self.tokenizer.decode(
                gen_ids[0], skip_special_tokens=True,
            )

            # Step 2: Secondary expert receives intermediate text + original prompt
            combined_prompt = f"{prompt}\n\n{intermediate_text}"
            combined_inputs = self.tokenizer(
                combined_prompt, return_tensors="pt",
                truncation=True, max_length=1024,
            )
            combined_ids = combined_inputs["input_ids"].to(self.device)
            combined_mask = combined_inputs.get("attention_mask")
            if combined_mask is not None:
                combined_mask = combined_mask.to(self.device)

            secondary = self._experts.get(routing.secondary_expert)
            if secondary is None:
                raise RuntimeError(f"Expert {routing.secondary_expert} not loaded.")

            final_ids = secondary.generate(
                combined_ids, combined_mask,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
            )
            final_text = self.tokenizer.decode(
                final_ids[0], skip_special_tokens=True,
            )

            return {
                "text": final_text,
                "routing": routing,
                "execution_mode": "text_relay_cascade",
                "experts_used": [routing.primary_expert, routing.secondary_expert],
                "latent_bridge_used": False,
                "intermediate_text": intermediate_text,
            }
        else:
            return self._execute_single(
                input_ids, attention_mask, routing,
                max_new_tokens, temperature, 50, 0.95,
            )

    # ── Private Execution Methods ──────────────────────────────────────────────

    def _route(
        self,
        prompt: str,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
    ) -> RoutingDecision:
        """Route the prompt to appropriate expert(s)."""
        if self.use_neural_router and self._neural_router is not None:
            routing = self._neural_router.route(input_ids, attention_mask)
            logger.info(
                "Neural routing: %s (confidence=%.2f, mode=%s)",
                routing.primary_expert.value,
                routing.confidence,
                routing.execution_mode.value,
            )
            return routing

        # Keyword fallback
        domain = self._keyword_router.route(prompt)
        available = list(self._experts.keys())

        # If routed domain is not loaded, fall back to first available
        if domain not in available and available:
            domain = available[0]

        # Determine secondary expert
        secondary = None
        for d in available:
            if d != domain:
                secondary = d
                break

        return RoutingDecision(
            expert_weights={domain: 1.0},
            primary_expert=domain,
            secondary_expert=secondary,
            execution_mode=ExecutionMode.SINGLE,
            confidence=1.0,
        )

    def _execute_single(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        routing: RoutingDecision,
        max_new_tokens: int,
        temperature: float,
        top_k: int,
        top_p: float,
    ) -> Dict[str, Any]:
        """Execute with a single expert."""
        expert = self._experts.get(routing.primary_expert)
        if expert is None:
            # Fall back to any available expert
            if self._experts:
                expert = next(iter(self._experts.values()))
            else:
                raise RuntimeError("No experts available.")

        gen_ids = expert.generate(
            input_ids, attention_mask,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
        )

        text = self.tokenizer.decode(gen_ids[0], skip_special_tokens=True)

        return {
            "text": text,
            "routing": routing,
            "execution_mode": ExecutionMode.SINGLE.value,
            "experts_used": [routing.primary_expert],
            "latent_bridge_used": False,
        }

    def _execute_cascade(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        routing: RoutingDecision,
        max_new_tokens: int,
        temperature: float,
        top_k: int,
        top_p: float,
    ) -> Dict[str, Any]:
        """Execute in CASCADE mode: Expert A → Bridge → Expert B.

        This is the core latent communication pipeline:
          1. Primary expert processes the prompt and outputs hidden states
          2. Bridge projects hidden states from primary → secondary space
          3. Secondary expert generates conditioned on bridge output
        """
        primary_expert = self._experts.get(routing.primary_expert)
        secondary_domain = routing.secondary_expert
        if secondary_domain is None:
            # No secondary — fall back to single
            return self._execute_single(
                input_ids, attention_mask, routing,
                max_new_tokens, temperature, top_k, top_p,
            )

        secondary_expert = self._experts.get(secondary_domain)
        if primary_expert is None or secondary_expert is None:
            logger.warning(
                "CASCADE: missing expert(s). Falling back to SINGLE."
            )
            return self._execute_single(
                input_ids, attention_mask, routing,
                max_new_tokens, temperature, top_k, top_p,
            )

        # Step 1: Extract hidden states from primary expert
        hidden_states = primary_expert.extract_hidden_states(
            input_ids, attention_mask, layer_index=-1,
        )

        # Step 2: Project through bridge
        source_id = self._domain_to_bridge_id(routing.primary_expert)
        target_id = self._domain_to_bridge_id(secondary_domain)
        bridge_output = self._bridge(
            hidden_states,
            source_domain_id=source_id,
            target_domain_id=target_id,
            attention_mask=attention_mask,
        )

        # Step 3: Secondary expert generates conditioned on bridge output
        gen_ids = secondary_expert.generate_conditioned(
            input_ids=input_ids,
            bridge_hidden_states=bridge_output,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
        )

        text = self.tokenizer.decode(gen_ids[0], skip_special_tokens=True)

        return {
            "text": text,
            "routing": routing,
            "execution_mode": ExecutionMode.CASCADE.value,
            "experts_used": [routing.primary_expert, secondary_domain],
            "latent_bridge_used": True,
        }

    # ── Internal Helpers ───────────────────────────────────────────────────────

    def _update_bridge_and_router(self) -> None:
        """Rebuild bridge and router when expert set changes."""
        num_experts = len(self._experts)
        if num_experts >= 2 and self._bridge is None:
            self._initialize_bridge()
        if self.use_neural_router and self._neural_router is None and num_experts >= 2:
            self._initialize_neural_router()

    def _initialize_bridge(self) -> None:
        """Create the bridge pool for current expert set."""
        num_domains = max(len(self._experts), 2)
        self._bridge = BridgePool(
            num_domains=num_domains,
            hidden_size=SHARED_HIDDEN_SIZE,
            bottleneck_size=self._bridge_bottleneck,
        )
        self._bridge.to(self.device)
        logger.info(
            "BridgePool initialized: %d domains, %d params",
            num_domains, self._bridge.num_parameters,
        )

    def _initialize_neural_router(self) -> None:
        """Create the neural router for current expert set."""
        domains = list(self._experts.keys())
        self._neural_router = NeuralRouter(
            d_model=self._router_d_model,
            num_experts=len(domains),
            expert_domains=domains,
            cascade_threshold=self._cascade_threshold,
        )
        self._neural_router.to(self.device)
        logger.info(
            "NeuralRouter initialized: %d experts, %d params",
            len(domains), self._neural_router.num_parameters,
        )

    def _domain_to_bridge_id(self, domain: ExpertDomain) -> int:
        """Map domain to bridge embedding index."""
        domains = list(self._experts.keys())
        if domain in domains:
            return domains.index(domain)
        return 0

    # ── Status & Debugging ─────────────────────────────────────────────────────

    def system_info(self) -> Dict[str, Any]:
        """Get full system status for debugging."""
        expert_info = {}
        for domain, expert in self._experts.items():
            expert_info[domain.value] = {
                "params": expert.num_parameters,
                "trainable_params": expert.num_trainable_parameters,
                "has_lora": expert._has_lora,
                "is_frozen": expert._is_frozen,
            }

        return {
            "device": self.device,
            "num_experts": len(self._experts),
            "experts": expert_info,
            "bridge": {
                "loaded": self._bridge is not None,
                "params": self._bridge.num_parameters if self._bridge else 0,
            },
            "router": {
                "type": "neural" if self._neural_router else "keyword",
                "params": self._neural_router.num_parameters if self._neural_router else 0,
            },
            "total_params": sum(e.num_parameters for e in self._experts.values())
                + (self._bridge.num_parameters if self._bridge else 0)
                + (self._neural_router.num_parameters if self._neural_router else 0),
        }
