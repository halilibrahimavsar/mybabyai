from enum import Enum
from dataclasses import dataclass

class CognitiveMode(Enum):
    """
    Defines the level of cognitive effort and interaction the AI will employ.
    """
    SYSTEM_1 = "system_1"                 # Fast, intuitive, direct LM generation
    SYSTEM_2_PLAN = "system_2_plan"       # Step-by-step thinking before answering
    SYSTEM_2_DEEPTHINK = "system_2_deepthink" # High-budget MCTS for complex logic/math
    SYSTEM_2_AGENT = "system_2_agent"     # Autonomous iteration with environment tools

@dataclass
class ModeConfig:
    """
    Budget and parameters associated with a specific Cognitive Mode.

    compute_budget (1-10): multiplier for simulations_per_step.
    Adjustable at runtime from UI — 1x=fast, 10x=deep.
    """
    mode: CognitiveMode
    use_mcts: bool
    max_depth: int
    simulations_per_step: int
    branching_factor: int
    # 2026: Test-Time Compute Scaling — multiply simulations by this factor
    compute_budget: int = 1

    def effective_simulations(self) -> int:
        """Actual simulations after applying compute_budget multiplier."""
        return self.simulations_per_step * self.compute_budget

# Pre-defined MCTS budgets for System 2 modes
MODE_CONFIGS = {
    CognitiveMode.SYSTEM_1: ModeConfig(
        mode=CognitiveMode.SYSTEM_1,
        use_mcts=False,
        max_depth=0,
        simulations_per_step=0,
        branching_factor=0,
        compute_budget=1,
    ),
    CognitiveMode.SYSTEM_2_PLAN: ModeConfig(
        mode=CognitiveMode.SYSTEM_2_PLAN,
        use_mcts=True,
        max_depth=5,
        simulations_per_step=5,
        branching_factor=2,
        compute_budget=1,
    ),
    CognitiveMode.SYSTEM_2_DEEPTHINK: ModeConfig(
        mode=CognitiveMode.SYSTEM_2_DEEPTHINK,
        use_mcts=True,
        max_depth=10,
        simulations_per_step=20,
        branching_factor=3,
        compute_budget=1,
    ),
    CognitiveMode.SYSTEM_2_AGENT: ModeConfig(
        mode=CognitiveMode.SYSTEM_2_AGENT,
        use_mcts=True,
        max_depth=8,
        simulations_per_step=10,
        branching_factor=2,
        compute_budget=1,
    )
}
