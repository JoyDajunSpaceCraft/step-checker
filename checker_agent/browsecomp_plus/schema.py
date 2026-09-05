from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional


SCHEMA_VERSION = "browsecomp_plus.stepgap.v1"


@dataclass(frozen=True)
class DocumentHit:
    docid: str
    rank: int
    snippet: str
    score: Optional[float] = None


@dataclass(frozen=True)
class BrowseObservation:
    observation_uid: str
    tool: str
    arguments: Dict[str, Any]
    documents: List[DocumentHit] = field(default_factory=list)
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class BrowseAction:
    tool: str
    arguments: Dict[str, Any]


@dataclass(frozen=True)
class BrowseStep:
    step_uid: str
    step_index: int
    thought: str
    action: Optional[BrowseAction]
    observation_uid: Optional[str]
    citations: List[str] = field(default_factory=list)


@dataclass(frozen=True)
class BrowseTrajectory:
    schema_version: str
    trace_uid: str
    task_uid: str
    query_id: str
    source_run: str
    model: str
    retriever: str
    seed: Optional[int]
    status: str
    steps: List[BrowseStep]
    final_response: str
    retrieved_docids: List[str]
    usage: Dict[str, Any]
    benchmark_version: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
