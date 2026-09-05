from __future__ import annotations

from .models import NLIModel, Verbalizer
from .normalize import observation_record_id, referenced_ids
from .schema import AgentStep, AxisDecision, CognitiveLabel


REPAIRS = {"contradicted_claim": "retract", "irrelevant_evidence": "re-target observation/tool", "missing_bridge": "obtain missing observation"}


class CognitiveChecker:
    def __init__(self, verbalizer: Verbalizer, nli: NLIModel):
        self.verbalizer, self.nli = verbalizer, nli

    def check(self, step: AgentStep) -> AxisDecision:
        if not step.thought or step.observation is None:
            return AxisDecision("no_gap", False, 1.0, "No claim-observation pair to verify")
        record_id = observation_record_id(step.observation)
        thought_ids = referenced_ids(step.thought)
        if record_id and thought_ids and record_id.lstrip("#") not in {x.lstrip("#") for x in thought_ids}:
            label = CognitiveLabel.IE.value
            return AxisDecision(label, True, 0.95, f"Observation is for {record_id}, thought refers to {sorted(thought_ids)}", REPAIRS[label])
        premise = self.verbalizer.verbalize(step.observation, step.thought)
        label, confidence = self.nli.classify(premise, step.thought)
        if label == "entailment":
            return AxisDecision("no_gap", False, confidence, "Observation entails the thought", evidence=premise)
        if label == "contradiction":
            gap = CognitiveLabel.CC.value
            return AxisDecision(gap, True, confidence, "Observation contradicts the thought", REPAIRS[gap], premise)
        for prior in reversed(step.prior_steps):
            if prior.observation is None:
                continue
            old = self.verbalizer.verbalize(prior.observation, step.thought)
            old_label, old_confidence = self.nli.classify(old, step.thought)
            if old_label == "entailment":
                return AxisDecision("no_gap", False, old_confidence, "A prior observation entails the thought", evidence=old)
        gap = CognitiveLabel.MB.value
        return AxisDecision(gap, True, confidence, "Relevant observation does not entail the thought", REPAIRS[gap], premise)
