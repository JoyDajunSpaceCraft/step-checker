from __future__ import annotations

import json
import re
from typing import Any, Tuple

from .schema import Action


def canonical_action(action: Action) -> Tuple[str, str]:
    args = json.dumps(action.arguments, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return action.tool.strip().lower(), args


def observation_record_id(observation: Any) -> str | None:
    if not isinstance(observation, dict):
        return None
    for key, value in observation.items():
        if key.endswith("_id") or key in {"id", "record"}:
            return str(value)
    return None


def referenced_ids(text: str) -> set[str]:
    return set(re.findall(r"#?[A-Za-z]+\d+", text or ""))


def explicit_confirmation(user_turns: list[str]) -> bool:
    patterns = (r"\b(confirm|confirmed|yes[, ]+proceed|go ahead|do it|i agree)\b", r"确认|同意|可以执行|继续操作")
    joined = "\n".join(user_turns).lower()
    return any(re.search(pattern, joined, flags=re.I) for pattern in patterns)
