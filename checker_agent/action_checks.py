from __future__ import annotations

from typing import Dict, Optional, Tuple

from .normalize import canonical_action, observation_record_id, referenced_ids
from .schema import AgentStep, NormativeLabel, ToolSpec


def check_wrong_tool(step: AgentStep, specs: Dict[str, ToolSpec]) -> Optional[Tuple[str, str]]:
    if step.action is None:
        return None
    spec = specs.get(step.action.tool)
    if spec is None:
        return NormativeLabel.WRONG_TOOL.value, f"Unknown tool: {step.action.tool}"
    status = step.known_state.get("status")
    if spec.allowed_statuses and status is not None and str(status) not in spec.allowed_statuses:
        return NormativeLabel.WRONG_TOOL.value, f"Tool requires status in {spec.allowed_statuses}, observed {status}"
    if spec.object_id_argument:
        target = step.action.arguments.get(spec.object_id_argument)
        user_ids = referenced_ids("\n".join(step.user_turns))
        known_id = observation_record_id(step.known_state)
        allowed = {x.lstrip("#") for x in user_ids}
        if target is not None and allowed and str(target).lstrip("#") not in allowed and str(target) != known_id:
            return NormativeLabel.WRONG_TOOL.value, f"Action targets {target}, outside user-mentioned objects"
    return None


def check_redundant(step: AgentStep) -> Optional[Tuple[str, str]]:
    if step.action is None:
        return None
    current = canonical_action(step.action)
    for prior in reversed(step.prior_steps):
        if prior.action and prior.action.is_write:
            break
        if prior.action and canonical_action(prior.action) == current:
            if not (isinstance(prior.observation, dict) and prior.observation.get("error")):
                return NormativeLabel.REDUNDANT.value, "Same tool and arguments repeated without an intervening write"
    return None
