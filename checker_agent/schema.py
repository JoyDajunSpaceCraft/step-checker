from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional


class CognitiveLabel(str, Enum):
    NO_GAP = "no_gap"
    CC = "contradicted_claim"
    IE = "irrelevant_evidence"
    MB = "missing_bridge"


class NormativeLabel(str, Enum):
    NO_GAP = "no_gap"
    POLICY = "policy_violation"
    WRONG_TOOL = "wrong_tool"
    REDUNDANT = "redundant_action"
    PREMATURE = "premature_commit"


@dataclass(frozen=True)
class Action:
    tool: str
    arguments: Dict[str, Any] = field(default_factory=dict)
    is_write: bool = False

    @classmethod
    def from_dict(cls, value: Optional[Dict[str, Any]]) -> Optional["Action"]:
        if value is None:
            return None
        return cls(str(value.get("tool", "")), dict(value.get("arguments") or {}), bool(value.get("is_write", False)))


@dataclass(frozen=True)
class ToolSpec:
    name: str
    is_write: bool = False
    allowed_statuses: List[str] = field(default_factory=list)
    object_id_argument: Optional[str] = None

    @classmethod
    def from_dict(cls, value: Dict[str, Any]) -> "ToolSpec":
        return cls(str(value["name"]), bool(value.get("is_write", False)),
                   [str(v) for v in value.get("allowed_statuses", [])], value.get("object_id_argument"))


@dataclass(frozen=True)
class PolicyRule:
    rule_id: str
    operation: str
    clause: str
    requires_confirmation: bool = False
    required_state: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, value: Dict[str, Any]) -> "PolicyRule":
        return cls(str(value["rule_id"]), str(value["operation"]), str(value.get("clause", "")),
                   bool(value.get("requires_confirmation", False)), dict(value.get("required_state") or {}))


@dataclass
class AgentStep:
    uid: str
    thought: str = ""
    action: Optional[Action] = None
    observation: Any = None
    response: Optional[str] = None
    user_turns: List[str] = field(default_factory=list)
    known_state: Dict[str, Any] = field(default_factory=dict)
    expected_final_state: Dict[str, Any] = field(default_factory=dict)
    prior_steps: List["AgentStep"] = field(default_factory=list)

    @classmethod
    def from_dict(cls, value: Dict[str, Any]) -> "AgentStep":
        return cls(
            uid=str(value["uid"]), thought=str(value.get("thought", "")),
            action=Action.from_dict(value.get("action")), observation=value.get("observation"),
            response=value.get("response"), user_turns=[str(v) for v in value.get("user_turns", [])],
            known_state=dict(value.get("known_state") or {}),
            expected_final_state=dict(value.get("expected_final_state") or {}),
            prior_steps=[cls.from_dict(v) for v in value.get("prior_steps", [])],
        )


@dataclass
class AxisDecision:
    label: str
    has_gap: bool
    confidence: float
    rationale: str
    repair: Optional[str] = None
    evidence: Optional[str] = None


@dataclass
class AgentCheckResult:
    uid: str
    cognitive: AxisDecision
    normative: AxisDecision
    primary_label: str
    pipeline_path: List[str]
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
