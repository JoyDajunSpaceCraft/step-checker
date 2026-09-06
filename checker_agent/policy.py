from __future__ import annotations

from typing import Iterable, Optional, Tuple

from .normalize import explicit_confirmation
from .schema import AgentStep, NormativeLabel, PolicyRule


def check_policy(step: AgentStep, rules: Iterable[PolicyRule]) -> Optional[Tuple[str, str, str]]:
    if step.action is None:
        return None
    for rule in rules:
        if rule.operation != step.action.tool:
            continue
        if rule.requires_confirmation and not explicit_confirmation(step.user_turns):
            return NormativeLabel.POLICY.value, "Required explicit confirmation is absent", rule.clause
        for key, expected in rule.required_state.items():
            actual = step.known_state.get(key)
            if actual != expected:
                return NormativeLabel.POLICY.value, f"Required {key}={expected!r}; observed {actual!r}", rule.clause
    return None
