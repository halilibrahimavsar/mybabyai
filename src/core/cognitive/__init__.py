"""
Cognitive Module for CodeMind.
This module implements System 2 (slow, deliberate) thinking using Monte Carlo Tree Search (MCTS)
and Reinforcement Learning principles to enhance the base System 1 (fast, intuitive) LLM.
"""

from .thought_node import ThoughtNode

__all__ = ["ThoughtNode"]
