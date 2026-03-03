import math
import logging
from typing import List, Optional, Any, Callable, Dict
import random

from src.core.cognitive.thought_node import ThoughtNode

logger = logging.getLogger(__name__)

class ReasoningEngine:
    """
    Monte Carlo Tree Search (MCTS) Engine for System 2.
    It governs the deep thinking process by exploring possible thoughts
    before generating a final output.
    """
    def __init__(
        self,
        language_model_generate: Callable[[str, int], List[str]],
        reward_evaluator: Callable[[str, str], float],
        max_depth: int = 5,
        simulations_per_step: int = 10,
        exploration_weight: float = 1.414,
        branching_factor: int = 3
    ):
        """
        Args:
            language_model_generate: A function that takes (context, num_samples) and returns possible next steps.
            reward_evaluator: A function that takes (context, step) and returns a float score [0, 1].
            max_depth: Deepest reasoning chain allowed.
            simulations_per_step: Number of MCTS loops to run per active decision.
            exploration_weight: UCT exploration constant (c).
            branching_factor: How many alternative thoughts to generate at each node.
        """
        self.lm_generate = language_model_generate
        self.reward_evaluator = reward_evaluator
        self.max_depth = max_depth
        self.simulations_per_step = simulations_per_step
        self.exploration_weight = exploration_weight
        self.branching_factor = branching_factor

    def search(self, initial_state: str) -> str:
        """
        Executes the MCTS over the initial state and returns the best 
        synthesized sequence of thoughts as the final answer.
        """
        root = ThoughtNode(state=initial_state)
        
        # Run MCTS loops
        for i in range(self.simulations_per_step):
            logger.debug(f"--- MCTS Simulation {i+1}/{self.simulations_per_step} ---")
            
            # 1. Selection
            leaf = self._select(root)
            
            # 2. Expansion
            # If leaf is not a terminal state and hasn't reached max depth
            depth = len(leaf.get_path_to_root()) - 1
            if not leaf.is_terminal and depth < self.max_depth:
                self._expand(leaf)
                
                # After expanding, select a random child to simulate from
                if leaf.children:
                    leaf = random.choice(leaf.children)
            
            # 3. Simulation (Evaluation)
            # In LLM-agent logic, instead of random rollouts to the end of a game,
            # we typically evaluate the immediate quality of the node state using the Reward Model.
            reward_score = self._simulate(leaf)
            
            # 4. Backpropagation
            self._backpropagate(leaf, reward_score)
            
        # Select best immediate action from root based on visits (robust) or highest value
        best_child = max(root.children, key=lambda node: node.visits) if root.children else None
        
        if not best_child:
             return "I could not formulate a complex thought."
             
        # Optional: Instead of returning just the next step, we follow the best path down
        # or recursively search. For now, we trace the greedy path of the highest value children.
        final_thought_process = self._extract_best_path(root)
        return final_thought_process

    def _select(self, node: ThoughtNode) -> ThoughtNode:
        """Traverses the tree to find the best leaf node using UCT."""
        current = node
        while current.children:
            # Check if current node is fully expanded conceptually.
            # If it has fewer children than branching factor, maybe consider it a leaf to expand more
            if len(current.children) < self.branching_factor:
                 return current
            
            # Otherwise, pick child with highest UCT score
            current = max(current.children, key=lambda n: n.uct_score(self.exploration_weight))
        return current

    def _expand(self, node: ThoughtNode) -> None:
        """Generates possible next thoughts from the LM and adds them as children."""
        # Query System 1 to generate `branching_factor` different ways forward
        possible_thoughts = self.lm_generate(node.state, self.branching_factor)
        
        for thought in possible_thoughts:
            # The new state is the accumulation of the previous state + the new thought
            # Depending on implementation, we might separate context and new thought
            new_state = f"{node.state}\n<thought>{thought}</thought>"
            child = ThoughtNode(
                state=new_state,
                action=thought,
                parent=node,
                is_terminal=False # determining terminal state is tricky, usually ends with <answer>tag
            )
            
            # Very basic heuristic for terminal: if LM generated an answer formatted block
            if "<answer>" in thought.lower() or "final answer:" in thought.lower():
                child.is_terminal = True
                
            node.add_child(child)

    def _simulate(self, node: ThoughtNode) -> float:
        """
        Evaluates the "goodness" of the current thought state.
        Instead of a random rollout to the end of a game (which is infinite in LLMs),
        we use the Reward Model (Evaluator) to score the node's action in context.
        """
        # If no parent, it's root, reward is neutral.
        if node.parent is None or not node.action:
            return 0.5 
            
        # Score the action taken to reach this leaf
        # We pass Context (parent's state) and the newly generated thought (action)
        score = self.reward_evaluator(node.parent.state, node.action)
        return score

    def _backpropagate(self, node: ThoughtNode, reward: float) -> None:
        """Passes the reward signal back up the tree to the root."""
        current = node
        while current is not None:
             current.update(reward)
             current = current.parent

    def _extract_best_path(self, root: ThoughtNode) -> str:
        """Walks down the tree following the most visited nodes to form the final chain."""
        path_texts = []
        current = root
        
        while current.children:
            # Exploitation: purely use visits or total value
            current = max(current.children, key=lambda n: n.visits)
            if current.action:
                path_texts.append(current.action)
                
        # Join the thoughts together clearly
        full_thought = "\n".join([f"Step {i+1}: {t}" for i, t in enumerate(path_texts)])
        return full_thought
