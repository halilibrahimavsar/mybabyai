"""
Expert Registry — tracks available specialist models and their metadata.

Each "expert" is a fine-tuned CodeMind checkpoint specializing in one domain.
The registry maps domain → checkpoint path, enabling the OrchestraManager
to lazy-load experts on demand.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional

from src.utils.logger import get_logger

logger = get_logger("expert_registry")


class ExpertDomain(str, Enum):
    """The 10 specialized domains in the Orchestra."""

    PYTHON_CODE    = "python_code"      # Python programming, scripting
    DART_FLUTTER   = "dart_flutter"     # Flutter/Dart mobile development
    SQL_DATABASE   = "sql_database"     # SQL, NoSQL, database design
    LINUX_SYSTEM   = "linux_system"     # Linux, shell, system administration
    TURKISH_CHAT   = "turkish_chat"     # Turkish-language conversation
    MATHEMATICS    = "mathematics"      # Math, algebra, statistics, proofs
    ANALYSIS       = "analysis"         # Research, summarization, document analysis
    CREATIVE       = "creative"         # Creative writing, storytelling
    SECURITY       = "security"         # Cybersecurity, CTF, penetration testing
    GENERAL        = "general"          # General knowledge fallback


@dataclass
class ExpertInfo:
    """Metadata for a single expert model."""

    domain: ExpertDomain
    # Path to model checkpoint (.pt or HF model dir)
    checkpoint_path: Optional[str] = None
    # Human-readable description shown in routing logs
    description: str = ""
    # Keywords used by rule-based routing fallback
    keywords: List[str] = field(default_factory=list)
    # Whether this expert is currently available (checkpoint exists)
    available: bool = False

    def __post_init__(self) -> None:
        if self.checkpoint_path:
            self.available = Path(self.checkpoint_path).exists()


class ExpertRegistry:
    """Central registry of all expert models.

    Initialize with optional checkpoint_root to auto-discover checkpoints:
        registry = ExpertRegistry(checkpoint_root="codemind/experts")

    Expert directories are expected as:
        codemind/experts/python_code/model_final.pt
        codemind/experts/dart_flutter/model_final.pt
        ...
    """

    # Default keyword sets for rule-based routing (used when no orchestrator model is loaded)
    _DEFAULT_KEYWORDS: Dict[ExpertDomain, List[str]] = {
        ExpertDomain.PYTHON_CODE: [
            "python", "pip", "django", "flask", "fastapi", "pandas", "numpy",
            "matplotlib", "asyncio", "pydantic",
        ],
        ExpertDomain.DART_FLUTTER: [
            "dart", "flutter", "widget", "stateful", "bloc", "riverpod",
            "pubspec", "material", "cupertino",
        ],
        ExpertDomain.SQL_DATABASE: [
            "sql", "mysql", "postgresql", "sqlite", "query", "join", "select",
            "insert", "mongodb", "firestore",
        ],
        ExpertDomain.LINUX_SYSTEM: [
            "linux", "bash", "shell", "terminal", "ubuntu", "debian", "grep",
            "awk", "sed", "systemd", "docker", "nginx",
        ],
        ExpertDomain.TURKISH_CHAT: [
            "merhaba", "nasılsın", "teşekkür", "türkçe", "sohbet", "konuş"
            "anlatır mısın", "bana söyle",
        ],
        ExpertDomain.MATHEMATICS: [
            "matematik", "calculus", "integral", "türev", "matris", "istatistik",
            "olasılık", "algebra", "lineer", "proof",
        ],
        ExpertDomain.ANALYSIS: [
            "analiz", "özetle", "summarize", "araştır", "karşılaştır", "değerlendir",
            "rapor", "explain", "compare",
        ],
        ExpertDomain.CREATIVE: [
            "hikaye", "şiir", "yaz", "roman", "senaryo", "story", "creative",
            "poem", "narra",
        ],
        ExpertDomain.SECURITY: [
            "güvenlik", "ctf", "exploit", "pentest", "xss", "sql injection",
            "encryption", "hash", "vulnerability",
        ],
        ExpertDomain.GENERAL: [],  # catch-all
    }

    def __init__(self, checkpoint_root: Optional[str] = None) -> None:
        self._experts: Dict[ExpertDomain, ExpertInfo] = {}
        self._checkpoint_root = Path(checkpoint_root) if checkpoint_root else None
        self._init_defaults()
        if self._checkpoint_root:
            self._auto_discover()

    def _init_defaults(self) -> None:
        """Register all 10 domains with default metadata (no checkpoints yet)."""
        descriptions = {
            ExpertDomain.PYTHON_CODE:  "Python programlama ve scripting uzmanı",
            ExpertDomain.DART_FLUTTER: "Flutter/Dart mobil geliştirme uzmanı",
            ExpertDomain.SQL_DATABASE: "SQL ve veritabanı tasarım uzmanı",
            ExpertDomain.LINUX_SYSTEM: "Linux ve sistem yönetimi uzmanı",
            ExpertDomain.TURKISH_CHAT: "Türkçe sohbet ve genel yardım uzmanı",
            ExpertDomain.MATHEMATICS:  "Matematik ve istatistik uzmanı",
            ExpertDomain.ANALYSIS:     "Araştırma ve analiz uzmanı",
            ExpertDomain.CREATIVE:     "Yaratıcı yazarlık uzmanı",
            ExpertDomain.SECURITY:     "Siber güvenlik ve CTF uzmanı",
            ExpertDomain.GENERAL:      "Genel bilgi ve fallback uzmanı",
        }
        for domain, desc in descriptions.items():
            self._experts[domain] = ExpertInfo(
                domain=domain,
                description=desc,
                keywords=self._DEFAULT_KEYWORDS[domain],
            )

    def _auto_discover(self) -> None:
        """Scan checkpoint_root for existing expert checkpoints."""
        for domain in ExpertDomain:
            ckpt = self._checkpoint_root / domain.value / "model_final.pt"
            if ckpt.exists():
                self._experts[domain].checkpoint_path = str(ckpt)
                self._experts[domain].available = True
                logger.info("Expert discovered: %s → %s", domain.value, ckpt)

    def register(
        self,
        domain: ExpertDomain,
        checkpoint_path: str,
        description: str = "",
        keywords: Optional[List[str]] = None,
    ) -> None:
        """Manually register an expert checkpoint."""
        info = self._experts[domain]
        info.checkpoint_path = checkpoint_path
        info.available = Path(checkpoint_path).exists()
        if description:
            info.description = description
        if keywords:
            info.keywords = keywords
        logger.info(
            "Expert registered: %s available=%s path=%s",
            domain.value, info.available, checkpoint_path,
        )

    def get(self, domain: ExpertDomain) -> ExpertInfo:
        return self._experts[domain]

    def list_available(self) -> List[ExpertDomain]:
        return [d for d, e in self._experts.items() if e.available]

    def list_all(self) -> List[ExpertInfo]:
        return list(self._experts.values())

    def keyword_route(self, query: str) -> ExpertDomain:
        """Simple keyword-based routing — used when no orchestrator LM is loaded."""
        query_lower = query.lower()
        scores: Dict[ExpertDomain, int] = {d: 0 for d in ExpertDomain}
        for domain, info in self._experts.items():
            for kw in info.keywords:
                if kw in query_lower:
                    scores[domain] += 1
        # Pick highest score; fall back to GENERAL
        best = max(scores, key=lambda d: scores[d])
        if scores[best] == 0:
            return ExpertDomain.GENERAL
        return best
