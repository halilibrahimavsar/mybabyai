"""Orchestra System — Expert Registry & Routing

10 Specialized Expert Models + 1 Orchestrator

Architecture:
  - Orchestrator: 350M model, fine-tuned only for routing decisions
  - Expert 1-10: Domain-specific 350M models, each fine-tuned for one specialty
  - Lazy loading: only the selected expert is in VRAM at any time
  - LRU cache: last N used experts are kept in RAM for quick reload
"""
from src.core.orchestra.orchestrator import Orchestrator, ExpertDomain
from src.core.orchestra.expert_registry import ExpertRegistry, ExpertInfo
from src.core.orchestra.orchestra_manager import OrchestraManager

__all__ = [
    "Orchestrator",
    "ExpertDomain",
    "ExpertRegistry",
    "ExpertInfo",
    "OrchestraManager",
]
