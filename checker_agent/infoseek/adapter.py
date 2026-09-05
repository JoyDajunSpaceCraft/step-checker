from __future__ import annotations

import hashlib
import re
from typing import Any


BLOCK = re.compile(
    r"<(think|search|information|answer)>\s*(.*?)\s*</\1>",
    re.IGNORECASE | re.DOTALL,
)


def parse_assistant_content(content: str, trace_uid: str, question: str) -> dict[str, Any]:
    """Convert an InfoSeek RFT assistant message into replayable search steps."""
    blocks = [(kind.lower(), text.strip()) for kind, text in BLOCK.findall(content)]
    steps: list[dict[str, Any]] = []
    pending_thought = ""
    final_response = ""
    index = 0
    while index < len(blocks):
        kind, text = blocks[index]
        if kind == "think":
            pending_thought = text
            index += 1
            continue
        if kind == "search":
            observation = ""
            if index + 1 < len(blocks) and blocks[index + 1][0] == "information":
                observation = blocks[index + 1][1]
                index += 1
            step_index = len(steps)
            steps.append({
                "uid": f"{trace_uid}::step{step_index:04d}",
                "step_index": step_index,
                "step_type": "search",
                "thought": pending_thought,
                "action": {"tool": "search", "arguments": {"query": text}, "is_write": False},
                "observation": observation,
                "response": None,
                "user_turns": [question],
            })
            pending_thought = ""
        elif kind == "answer":
            final_response = text
            step_index = len(steps)
            steps.append({
                "uid": f"{trace_uid}::step{step_index:04d}",
                "step_index": step_index,
                "step_type": "answer",
                "thought": pending_thought,
                "action": None,
                "observation": None,
                "response": text,
                "user_turns": [question],
            })
            pending_thought = ""
        index += 1
    return {
        "schema_version": "infoseek.stepgap.v1",
        "trace_uid": trace_uid,
        "task_uid": trace_uid.split("::rft", 1)[0],
        "source": "InfoSeek/Trajectory-RFT-17K",
        "steps": steps,
        "final_response": final_response,
        "parse": {
            "block_count": len(blocks),
            "complete": bool(final_response) and all(
                step["observation"] for step in steps if step["step_type"] == "search"
            ),
        },
    }


def convert_rft_record(record: dict[str, Any], source_index: int) -> dict[str, Any]:
    conversation = record.get("conversation") or []
    user = "\n".join(str(turn.get("content", "")) for turn in conversation if turn.get("role") == "user")
    assistant = "\n".join(
        str(turn.get("content", "")) for turn in conversation if turn.get("role") == "assistant"
    )
    digest = hashlib.sha256(user.encode("utf-8")).hexdigest()[:16]
    trace_uid = f"infoseek::{digest}::rft{source_index:05d}"
    result = parse_assistant_content(assistant, trace_uid, user)
    result["source_index"] = source_index
    return result
