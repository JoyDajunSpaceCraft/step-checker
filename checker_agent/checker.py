from __future__ import annotations

from typing import Dict, Iterable, Optional

from .action_checks import check_redundant, check_wrong_tool
from .cognitive import CognitiveChecker
from .models import DeterministicVerbalizer, HeuristicNLI, NLIModel, Verbalizer
from .policy import check_policy
from .schema import AgentCheckResult, AgentStep, AxisDecision, PolicyRule, ToolSpec
from .termination import check_termination


REPAIRS = {
    "policy_violation": "satisfy policy precondition",
    "wrong_tool": "re-target tool or object",
    "redundant_action": "reuse prior observation",
    "premature_commit": "execute required state change before responding",
}

PRIORITY = {
    "policy_violation": 70,
    "wrong_tool": 60,
    "premature_commit": 50,
    "contradicted_claim": 40,
    "irrelevant_evidence": 30,
    "missing_bridge": 20,
    "redundant_action": 10,
    "no_gap": 0,
}


class AgentStepGapChecker:
    """Run cognitive and normative axes independently and preserve both."""

    def __init__(
        self,
        tool_specs: Iterable[ToolSpec] = (),
        policy_rules: Iterable[PolicyRule] = (),
        verbalizer: Optional[Verbalizer] = None,
        nli: Optional[NLIModel] = None,
    ):
        self.tool_specs: Dict[str, ToolSpec] = {x.name: x for x in tool_specs}
        self.policy_rules = list(policy_rules)
        self.cognitive = CognitiveChecker(
            verbalizer or DeterministicVerbalizer(), nli or HeuristicNLI()
        )

    def check(self, step: AgentStep) -> AgentCheckResult:
        path = ["normalize:success"]
        cognitive = self.cognitive.check(step)
        path.append(f"cognitive:{cognitive.label}")

        normative = AxisDecision(
            "no_gap", False, 1.0, "No normative violation detected"
        )
        checks = (
            ("wrong_tool", check_wrong_tool(step, self.tool_specs)),
            ("policy", check_policy(step, self.policy_rules)),
            ("redundancy", check_redundant(step)),
            ("termination", check_termination(step)),
        )
        for stage, result in checks:
            if result is None:
                path.append(f"{stage}:pass")
                continue
            label, rationale, *evidence = result
            normative = AxisDecision(
                label,
                True,
                1.0,
                rationale,
                REPAIRS[label],
                evidence[0] if evidence else None,
            )
            path.append(f"{stage}:{label}")
            break

        primary = max(
            (cognitive.label, normative.label),
            key=lambda label: PRIORITY.get(label, 0),
        )
        return AgentCheckResult(
            uid=step.uid,
            cognitive=cognitive,
            normative=normative,
            primary_label=primary,
            pipeline_path=path,
            metadata={"multi_axis": cognitive.has_gap and normative.has_gap},
        )
