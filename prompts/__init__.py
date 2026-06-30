"""Prompt Engineering techniques package."""

from prompts.simple_prompt import SimplePrompt
from prompts.zero_shot_prompt import ZeroShotPrompt
from prompts.few_shot_prompt import FewShotPrompt
from prompts.chain_of_thought_prompt import ChainOfThoughtPrompt
from prompts.self_consistency_prompt import SelfConsistencyPrompt
from prompts.role_prompt import RolePrompt
from prompts.step_back_prompt import StepBackPrompt
from prompts.tree_of_thoughts_prompt import TreeOfThoughtsPrompt

__all__ = [
    "SimplePrompt",
    "ZeroShotPrompt",
    "FewShotPrompt",
    "ChainOfThoughtPrompt",
    "SelfConsistencyPrompt",
    "RolePrompt",
    "StepBackPrompt",
    "TreeOfThoughtsPrompt",
]
