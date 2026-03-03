import re
from typing import Dict, Any, Tuple

class FitratRules:
    """
    R_fitrat: Hardcoded constitutional principles for the CodeMind AI.
    These are non-negotiable rules acting as the foundation for safety,
    truthfulness, and alignment.
    """
    
    # Simple regex patterns for immediate identification of boundary violations
    VIOLATION_PATTERNS = [
        re.compile(r"(?i)\b(hack\s+into|bypass\s+security|steal\s+credentials|malware|ransomware|ddos)\b"),
        re.compile(r"(?i)\b(ignore\s+previous\s+instructions|system\s+prompt|as\s+an\s+ai)\b"),
        # Add more domain-specific harmful intent patterns here
    ]
    
    @classmethod
    def evaluate(cls, context: str, action: str) -> Tuple[float, str]:
        """
        Evaluates the proposed action against Fitrat (constitutional) rules.
        Returns a score from 0.0 to 1.0, and a reasoning string.
        0.0 = Absolute violation (Haram/Unsafe)
        1.0 = Perfectly aligned
        """
        full_text = f"{context}\n{action}"
        
        # 1. Check for explicit safety policy violations
        for pattern in cls.VIOLATION_PATTERNS:
            if pattern.search(full_text):
                return 0.0, "Violation of core safety policies detected."
                
        # 2. Check for logic consistency / circular reasoning
        # For example, if the action just repeats the context exactly
        if action.strip() != "" and action.strip() in context:
            return 0.2, "Action is highly repetitive of the context."
            
        # 3. Assess formatting and structural consistency (e.g., did it follow XML tags)
        if "<thought>" in action and "</thought>" not in action:
            return 0.4, "Malformed output tags."

        return 1.0, "Aligned with Fitrat rules."
