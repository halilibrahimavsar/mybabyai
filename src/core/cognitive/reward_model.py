import torch
from transformers import PreTrainedModel, PreTrainedTokenizer
from typing import Tuple

from src.core.cognitive.fitrat import FitratRules
from src.core.model.codemind import CodeMindConfig, CodeMindForCausalLM

class DualRewardEvaluator:
    """
    Combines R_fitrat (Safety & Constitution) with R_nefis (Helpfulness & Speed).
    R_total = alpha * R_fitrat + (1 - alpha) * R_nefis
    """
    def __init__(
        self, 
        evaluator_model: PreTrainedModel = None, 
        evaluator_tokenizer: PreTrainedTokenizer = None,
        alpha: float = 0.6
    ):
        """
        Args:
            evaluator_model: The 50M parameter Reward Model. If None, uses a heuristic fallback.
            evaluator_tokenizer: Tokenizer for the reward model.
            alpha: Weight of the Fitrat (Safety) score vs Nefis (Helpfulness) score.
        """
        self.model = evaluator_model
        self.tokenizer = evaluator_tokenizer
        self.alpha = alpha

    def evaluate(self, context: str, action: str) -> float:
        """
        Calculates the combined reward score for a given state-action pair.
        """
        # 1. R_fitrat Evaluation
        r_fitrat, _ = FitratRules.evaluate(context, action)
        
        # Immediate short-circuit if safety is violated
        if r_fitrat == 0.0:
             return 0.0
             
        # 2. R_nefis Evaluation
        r_nefis = self._score_nefis(context, action)
        
        # 3. Combine
        total_reward = (self.alpha * r_fitrat) + ((1.0 - self.alpha) * r_nefis)
        
        return float(total_reward)
        
    def _score_nefis(self, context: str, action: str) -> float:
        """
        Scores the helpfulness and execution speed preference.
        In a full implementation, this runs through the 50M Reward Model.
        """
        if self.model is None or self.tokenizer is None:
            # Heuristic fallback if no model is loaded
            return self._heuristic_nefis(action)
            
        # TODO: Implement actual model forward pass for Preference Scoring (R_nefis)
        # prompt = f"Context: {context}\nAction: {action}\nIs this helpful? Score 0.0 to 1.0:"
        # return model_score
        
        return 0.5

    def _heuristic_nefis(self, action: str) -> float:
        """A simple heuristic for R_nefis when the neural evaluator isn't available."""
        score = 0.5
        # Prefer actions with actual content
        if len(action.strip()) > 10:
             score += 0.2
        # Prefer actions that sound decisive vs uncertain
        if "I think" in action or "maybe" in action.lower():
             score -= 0.1
        if "def " in action or "class " in action: # Prefer code generation
             score += 0.2
             
        return max(0.0, min(1.0, score))


def get_reward_model_config() -> CodeMindConfig:
    """Returns the config for the 50M parameter Evaluator/Critic Model."""
    return CodeMindConfig(
        vocab_size=32768,
        hidden_size=512,         
        num_hidden_layers=8,
        num_attention_heads=8,
        num_key_value_heads=2,
        intermediate_size=2048,
        max_position_embeddings=2048,
        num_experts=1,           # Reward model doesn't need MoE usually
        num_experts_per_tok=1
    )
