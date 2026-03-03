from typing import List, Optional, Any, Dict
import numpy as np

class ThoughtNode:
    """
    Represents a single state/action in the Monte Carlo Tree Search (MCTS)
    process for System 2 reasoning.
    """
    def __init__(
        self,
        state: str,
        action: Optional[str] = None,
        parent: Optional['ThoughtNode'] = None,
        is_terminal: bool = False
    ):
        self.state = state  # The current context/prompt accumulated so far
        self.action = action  # The specific step/thought that led to this state
        self.parent = parent
        self.children: List['ThoughtNode'] = []
        
        # MCTS Statistics
        self.visits: int = 0
        self.value: float = 0.0
        self.reward: Optional[float] = None
        
        self.is_terminal = is_terminal
        self.metadata: Dict[str, Any] = {}

    @property
    def is_fully_expanded(self) -> bool:
        """
        Determines if this node has been fully expanded.
        In an LLM context, 'fully expanded' is subjective as action space is continuous/infinite.
        We will define this conceptually based on a fixed branching factor budget in the reasoning engine.
        """
        # This will be managed by the reasoning engine, but structurally it means 
        # we have explored enough child actions for this state.
        return len(self.children) > 0 # Placeholder logic, engine overrides this

    def add_child(self, child_node: 'ThoughtNode') -> None:
        """Adds a child node to the current node."""
        self.children.append(child_node)

    def update(self, reward: float) -> None:
        """
        Backpropagates the reward up the tree.
        Updates the visit count and the accumulated value of the node.
        """
        self.visits += 1
        self.value += reward

    def uct_score(self, exploration_weight: float = 1.414) -> float:
        """
        Calculates the Upper Confidence Bound applied to Trees (UCT) score.
        Used to balance exploration vs. exploitation during the Selection phase.
        """
        if self.visits == 0:
            return float('inf')  # Prioritize unvisited nodes
        
        exploitation_term = self.value / self.visits
        
        if self.parent is None or self.parent.visits == 0:
             exploration_term = 0.0
        else:
             exploration_term = exploration_weight * np.sqrt(np.log(self.parent.visits) / self.visits)
             
        return exploitation_term + exploration_term

    def get_path_to_root(self) -> List['ThoughtNode']:
        """Returns the path from the root to this node."""
        path = []
        current = self
        while current is not None:
            path.append(current)
            current = current.parent
        return path[::-1] # Reverse to get Root -> Node

    def __repr__(self):
        v = self.value / self.visits if self.visits > 0 else 0
        return f"<ThoughtNode(visits={self.visits}, val={v:.2f}, children={len(self.children)})>"
