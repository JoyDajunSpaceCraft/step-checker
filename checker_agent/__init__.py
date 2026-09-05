"""StepGap-Agent: an isolated checker for tool-use agent traces."""

from .checker import AgentStepGapChecker
from .schema import Action, AgentCheckResult, AgentStep, PolicyRule, ToolSpec

__all__ = [
    "Action", "AgentCheckResult", "AgentStep", "AgentStepGapChecker",
    "PolicyRule", "ToolSpec",
]
