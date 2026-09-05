from __future__ import annotations

from typing import Optional, Tuple

from .schema import AgentStep, NormativeLabel


def check_termination(step: AgentStep) -> Optional[Tuple[str, str]]:
    if step.response is None or not step.expected_final_state:
        return None
    for key, expected in step.expected_final_state.items():
        actual = step.known_state.get(key)
        if actual != expected:
            return NormativeLabel.PREMATURE.value, f"Response emitted before {key} reached {expected!r}; observed {actual!r}"
    return None
