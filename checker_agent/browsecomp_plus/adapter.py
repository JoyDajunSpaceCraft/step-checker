from __future__ import annotations

import hashlib
import json
from typing import Any, Dict, Iterable, List, MutableMapping, Optional, Tuple

from .schema import (
    SCHEMA_VERSION,
    BrowseAction,
    BrowseObservation,
    BrowseStep,
    BrowseTrajectory,
    DocumentHit,
)


def _parse_json(value: Any, expected_type: type) -> Any:
    parsed = json.loads(value) if isinstance(value, str) else value
    if not isinstance(parsed, expected_type):
        raise ValueError(f"Expected {expected_type.__name__}, got {type(parsed).__name__}")
    return parsed


def _reasoning_text(output: Any) -> str:
    if isinstance(output, list):
        return "\n".join(str(x) for x in output if str(x).strip()).strip()
    return str(output or "").strip()


def _tool_call_counts(record: Dict[str, Any]) -> Dict[str, int]:
    value = record.get("tool_call_counts")
    if isinstance(value, dict):
        return {str(k): int(v) for k, v in value.items()}
    value = record.get("search_counts")
    if isinstance(value, dict):
        return {str(k): int(v) for k, v in value.items()}
    if isinstance(value, int):
        return {"search": value}
    return {}


def _observation(tool: str, arguments: Dict[str, Any], output: Any) -> BrowseObservation:
    documents: List[DocumentHit] = []
    error: Optional[str] = None
    try:
        rows = _parse_json(output, list)
        for rank, row in enumerate(rows, start=1):
            if not isinstance(row, dict):
                continue
            score = row.get("score")
            documents.append(
                DocumentHit(
                    docid=str(row.get("docid", "")),
                    rank=rank,
                    snippet=str(row.get("snippet", "")),
                    score=float(score) if score is not None else None,
                )
            )
    except (ValueError, TypeError, json.JSONDecodeError) as exc:
        error = f"unparsed_tool_output:{type(exc).__name__}"

    canonical = {
        "tool": tool,
        "arguments": arguments,
        "documents": [
            {"docid": d.docid, "rank": d.rank, "snippet": d.snippet, "score": d.score}
            for d in documents
        ],
        "error": error,
    }
    digest = hashlib.sha256(
        json.dumps(canonical, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return BrowseObservation(f"obs::sha256::{digest}", tool, arguments, documents, error)


def convert_official_record(
    record: Dict[str, Any],
    *,
    source_run: str,
    model: str,
    retriever: str,
    benchmark_version: str,
    seed: Optional[int] = None,
    observation_store: Optional[MutableMapping[str, BrowseObservation]] = None,
) -> Tuple[BrowseTrajectory, MutableMapping[str, BrowseObservation]]:
    """Convert one official normalized run without copying prior observations.

    Reasoning events are accumulated until the following tool call.  Any
    remaining reasoning is attached to a final-response step.  Observations
    are content-addressed so identical tool results are stored only once.
    """

    query_id = str(record["query_id"])
    trace_uid = f"bcp::{query_id}::{model}::{retriever}::seed{seed if seed is not None else 'na'}"
    store: MutableMapping[str, BrowseObservation] = observation_store if observation_store is not None else {}
    steps: List[BrowseStep] = []
    pending_reasoning: List[str] = []
    final_outputs: List[str] = []
    retrieved: List[str] = []

    for item in record.get("result", []):
        event_type = item.get("type")
        if event_type == "reasoning":
            text = _reasoning_text(item.get("output"))
            if text:
                pending_reasoning.append(text)
            continue
        if event_type == "tool_call":
            tool = str(item.get("tool_name") or "")
            arguments = _parse_json(item.get("arguments") or {}, dict)
            obs = _observation(tool, arguments, item.get("output"))
            store.setdefault(obs.observation_uid, obs)
            for doc in obs.documents:
                if doc.docid and doc.docid not in retrieved:
                    retrieved.append(doc.docid)
            index = len(steps)
            steps.append(
                BrowseStep(
                    step_uid=f"{trace_uid}::step{index:04d}",
                    step_index=index,
                    thought="\n".join(pending_reasoning).strip(),
                    action=BrowseAction(tool, arguments),
                    observation_uid=obs.observation_uid,
                )
            )
            pending_reasoning.clear()
            continue
        if event_type == "output_text":
            text = _reasoning_text(item.get("output"))
            if text:
                final_outputs.append(text)

    if pending_reasoning or final_outputs:
        index = len(steps)
        steps.append(
            BrowseStep(
                step_uid=f"{trace_uid}::step{index:04d}",
                step_index=index,
                thought="\n".join(pending_reasoning).strip(),
                action=None,
                observation_uid=None,
            )
        )

    trajectory = BrowseTrajectory(
        schema_version=SCHEMA_VERSION,
        trace_uid=trace_uid,
        task_uid=f"bcp::{query_id}",
        query_id=query_id,
        source_run=source_run,
        model=model,
        retriever=retriever,
        seed=seed,
        status=str(record.get("status", "unknown")),
        steps=steps,
        final_response="\n".join(final_outputs).strip(),
        retrieved_docids=retrieved,
        usage={
            "tool_call_counts": _tool_call_counts(record),
            "event_count": len(record.get("result", [])),
        },
        benchmark_version=benchmark_version,
    )
    return trajectory, store


def convert_records(records: Iterable[Dict[str, Any]], **kwargs: Any):
    store: Dict[str, BrowseObservation] = {}
    for record in records:
        trajectory, _ = convert_official_record(record, observation_store=store, **kwargs)
        yield trajectory
