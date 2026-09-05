from __future__ import annotations

import argparse
import hashlib
import json
import re
import threading
from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Dict, List, Optional

from .adapter import _observation, _parse_json
from .schema import BrowseAction, BrowseObservation, BrowseStep


SemanticChecker = Callable[[str, str, List[BrowseObservation], bool], Dict[str, Any]]


def canonical_query(query: str) -> str:
    return " ".join(re.findall(r"[a-z0-9]+", query.lower()))


@dataclass
class AuditSession:
    trace_uid: str
    task_uid: str
    query: str
    steps: List[BrowseStep] = field(default_factory=list)
    observations: Dict[str, BrowseObservation] = field(default_factory=dict)
    final_response: str = ""


class BrowseCompAuditEngine:
    """Stateful observer called by an agent runner after retrieval events.

    This engine never exposes benchmark answers or relevance labels.  Without
    a semantic backend it performs only deterministic checks and reports
    semantic_status=not_evaluated rather than treating an unchecked claim as
    no-gap.
    """

    def __init__(self, semantic_checker: Optional[SemanticChecker] = None):
        self.semantic_checker = semantic_checker
        self.sessions: Dict[str, AuditSession] = {}
        self._lock = threading.RLock()

    def start_trace(self, task_uid: str, query: str, run_id: str = "run0") -> Dict[str, Any]:
        digest = hashlib.sha256(f"{task_uid}\0{run_id}".encode()).hexdigest()[:16]
        trace_uid = f"{task_uid}::audit::{digest}"
        with self._lock:
            if trace_uid in self.sessions:
                raise ValueError(f"Trace already exists: {trace_uid}")
            self.sessions[trace_uid] = AuditSession(trace_uid, task_uid, query)
        return {"trace_uid": trace_uid, "status": "started"}

    def record_search(self, trace_uid: str, thought: str, query: str, observation_json: str) -> Dict[str, Any]:
        with self._lock:
            session = self._session(trace_uid)
            arguments = {"query": query}
            observation = _observation("search", arguments, observation_json)
            session.observations.setdefault(observation.observation_uid, observation)
            prior_queries = [
                canonical_query(str(step.action.arguments.get("query", "")))
                for step in session.steps if step.action and step.action.tool == "search"
            ]
            duplicate = bool(canonical_query(query) and canonical_query(query) in prior_queries)
            index = len(session.steps)
            step = BrowseStep(
                step_uid=f"{trace_uid}::step{index:04d}", step_index=index,
                thought=thought.strip(), action=BrowseAction("search", arguments),
                observation_uid=observation.observation_uid,
            )
            session.steps.append(step)
            deterministic = {
                "label": "redundant_action" if duplicate else "no_deterministic_gap",
                "has_gap": duplicate,
                "repair": "reuse prior observation" if duplicate else None,
            }
            semantic = self._semantic(session, thought, is_final=False)
            return {
                "trace_uid": trace_uid, "step_uid": step.step_uid,
                "deterministic": deterministic, "semantic": semantic,
                "pipeline_path": [
                    "search:recorded", "redundancy:duplicate" if duplicate else "redundancy:pass",
                    f"semantic:{semantic['status']}",
                ],
            }

    def finish_trace(self, trace_uid: str, response: str) -> Dict[str, Any]:
        with self._lock:
            session = self._session(trace_uid)
            session.final_response = response.strip()
            no_evidence = not any(obs.documents for obs in session.observations.values())
            semantic = self._semantic(session, response, is_final=True)
            return {
                "trace_uid": trace_uid,
                "status": "finished",
                "deterministic": {
                    "label": "premature_answer_candidate" if response.strip() and no_evidence else "no_deterministic_gap",
                    "has_gap": bool(response.strip() and no_evidence),
                    "requires_semantic_confirmation": True,
                },
                "semantic": semantic,
                "step_count": len(session.steps),
            }

    def export_trace(self, trace_uid: str) -> Dict[str, Any]:
        with self._lock:
            session = self._session(trace_uid)
            return {
                "trace_uid": session.trace_uid, "task_uid": session.task_uid,
                "query": session.query, "steps": [asdict(step) for step in session.steps],
                "observations": [asdict(session.observations[key]) for key in sorted(session.observations)],
                "final_response": session.final_response,
            }

    def _semantic(self, session: AuditSession, claim_text: str, is_final: bool) -> Dict[str, Any]:
        if self.semantic_checker is None:
            return {"status": "not_evaluated", "reason": "No claim-decomposition/evidence-verification backend configured"}
        return self.semantic_checker(session.query, claim_text, list(session.observations.values()), is_final)

    def _session(self, trace_uid: str) -> AuditSession:
        try:
            return self.sessions[trace_uid]
        except KeyError as exc:
            raise KeyError(f"Unknown trace_uid: {trace_uid}") from exc


def build_mcp(engine: Optional[BrowseCompAuditEngine] = None):
    from fastmcp import FastMCP

    audit = engine or BrowseCompAuditEngine()
    mcp = FastMCP(name="stepgap-browsecomp-auditor")

    @mcp.tool()
    def start_trace(task_uid: str, query: str, run_id: str = "run0") -> Dict[str, Any]:
        """Start an audit session. Do not pass benchmark answers or gold document labels."""
        return audit.start_trace(task_uid, query, run_id)

    @mcp.tool()
    def audit_search(trace_uid: str, thought: str, query: str, observation_json: str) -> Dict[str, Any]:
        """Record one completed search and audit its reasoning/evidence step."""
        return audit.record_search(trace_uid, thought, query, observation_json)

    @mcp.tool()
    def audit_answer(trace_uid: str, response: str) -> Dict[str, Any]:
        """Audit and finalize an answer after all retrieval actions are complete."""
        return audit.finish_trace(trace_uid, response)

    @mcp.tool()
    def get_audit_trace(trace_uid: str) -> Dict[str, Any]:
        """Export the replayable audit trace for logging or process-reward computation."""
        return audit.export_trace(trace_uid)

    return mcp


def main() -> None:
    parser = argparse.ArgumentParser(description="BrowseComp-Plus StepGap audit MCP")
    parser.add_argument("--transport", choices=["stdio", "streamable-http", "sse"], default="stdio")
    parser.add_argument("--port", type=int, default=8011)
    args = parser.parse_args()
    mcp = build_mcp()
    if args.transport == "stdio":
        mcp.run(transport="stdio")
    else:
        mcp.run(transport=args.transport, path="/mcp", port=args.port)


if __name__ == "__main__":
    main()
