"""Latent Expert Orchestra — Modular Brain Architecture

A next-generation expert orchestration system where specialized models
communicate via latent representations instead of text relay.

Architecture:
  ┌──────────────┐
  │  User Prompt  │
  └──────┬───────┘
         │
  ┌──────▼───────┐
  │ Neural Router │  ← Classifies prompt → expert(s) + execution mode
  └──────┬───────┘
         │
    ┌────┴────┐
    │         │
 SINGLE    CASCADE
    │         │
    ▼         ▼
  Expert   Expert A ──hidden──▶ Bridge ──projected──▶ Expert B
    │         │                                         │
    ▼         ▼                                         ▼
 Response  (latent context)                         Response

Components:
  - ExpertModel: Wrapper around CodeMindForCausalLM with latent extraction
  - NeuralRouter: Transformer-based expert selector (~2.5M params)
  - LatentBridge/BridgePool: MLP bridge for inter-expert communication
  - LatentOrchestra: Top-level conductor managing the pipeline
  - Training: BridgeTrainer, RouterTrainer for phased training
"""

# ── Expert Configuration ──────────────────────────────────────────────────────
from src.core.orchestra.expert_config import (
    ExpertConfig,
    ExpertDomain,
    EXPERT_CONFIGS,
    SHARED_HIDDEN_SIZE,
    SHARED_VOCAB_SIZE,
)

# ── Expert Model ──────────────────────────────────────────────────────────────
from src.core.orchestra.expert_model import ExpertModel

# ── Latent Bridge ─────────────────────────────────────────────────────────────
from src.core.orchestra.latent_bridge import BridgePool, LatentBridge

# ── Neural Router ─────────────────────────────────────────────────────────────
from src.core.orchestra.neural_router import (
    ExecutionMode,
    KeywordRouterFallback,
    NeuralRouter,
    RoutingDecision,
)

# ── Latent Orchestra (main conductor) ─────────────────────────────────────────
from src.core.orchestra.latent_orchestra import LatentOrchestra

# ── Legacy (backward compat) ──────────────────────────────────────────────────
from src.core.orchestra.orchestrator import Orchestrator
from src.core.orchestra.expert_registry import ExpertRegistry, ExpertInfo
from src.core.orchestra.orchestra_manager import OrchestraManager

__all__ = [
    # New Architecture
    "ExpertConfig",
    "ExpertDomain",
    "EXPERT_CONFIGS",
    "SHARED_HIDDEN_SIZE",
    "SHARED_VOCAB_SIZE",
    "ExpertModel",
    "LatentBridge",
    "BridgePool",
    "NeuralRouter",
    "ExecutionMode",
    "RoutingDecision",
    "KeywordRouterFallback",
    "LatentOrchestra",
    # Legacy
    "Orchestrator",
    "ExpertRegistry",
    "ExpertInfo",
    "OrchestraManager",
]
