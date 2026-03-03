import re
import logging
from typing import Optional

from src.core.cognitive.modes import CognitiveMode, MODE_CONFIGS, ModeConfig

logger = logging.getLogger(__name__)

class CognitiveRouter:
    """
    Determines whether a given prompt requires System 1 (fast intuitively) 
    or System 2 (slow, MCTS reasoning) thinking, and which specific mode to use.
    """
    
    # Keywords indicating a need for deep logic or mathematical reasoning
    DEEPTHINK_KEYWORDS = [
        "calculate", "prove", "theorem", "math", "equation", "logic puzzle",
        "solve for", "integral", "derivative", "optimize", "algorithm complexity"
    ]
    
    # Keywords indicating a sequence of steps or complex code architecture
    PLAN_KEYWORDS = [
        "plan", "step by step", "architecture", "design", "refactor",
        "how to build", "strategy", "structure"
    ]
    
    # Keywords indicating a need to act upon the environment
    AGENT_KEYWORDS = [
        "run command", "terminal", "execute", "read file", "write to file",
        "search the web", "look up", "fix this error in my workspace"
    ]

    def __init__(self, force_mode: Optional[CognitiveMode] = None):
        """
        Args:
            force_mode: Override dynamic routing and always use this mode.
        """
        self.force_mode = force_mode

    def route(self, prompt: str) -> ModeConfig:
        """
        Analyzes the prompt and returns the appropriate ModeConfig.
        """
        if self.force_mode is not None:
            logger.info(f"Router forced to use mode: {self.force_mode.value}")
            return MODE_CONFIGS[self.force_mode]
            
        mode = self._classify_prompt(prompt.lower())
        logger.info(f"Router classified prompt as: {mode.value}")
        
        return MODE_CONFIGS[mode]

    def _classify_prompt(self, prompt: str) -> CognitiveMode:
        """
        Heuristic classification based on keywords and prompt length.
        """
        # 1. Agent Mode Check (Environment Interaction)
        if any(kw in prompt for kw in self.AGENT_KEYWORDS):
            return CognitiveMode.SYSTEM_2_AGENT
            
        # 2. DeepThink Check (Logic, Math, Hard problems)
        if any(kw in prompt for kw in self.DEEPTHINK_KEYWORDS):
            return CognitiveMode.SYSTEM_2_DEEPTHINK
            
        # 3. Plan Check (Complex reasoning, coding)
        if any(kw in prompt for kw in self.PLAN_KEYWORDS):
            return CognitiveMode.SYSTEM_2_PLAN
            
        # 4. Length/Complexity heuristic feature
        # If the prompt is very long, it likely requires more than a simple System 1 response
        word_count = len(prompt.split())
        if word_count > 100:
             return CognitiveMode.SYSTEM_2_PLAN
             
        # Fallback: Fast System 1 thinking
        return CognitiveMode.SYSTEM_1
