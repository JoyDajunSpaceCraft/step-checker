from __future__ import annotations

import json
from typing import Any


CLAIM_ITEM = {
    "type": "object",
    "properties": {
        "claim": {"type": "string"},
        "evidence_relevant": {"type": "boolean"},
        "evidence_spans": {"type": "array", "items": {"type": "string"}, "maxItems": 3},
    },
    "required": ["claim", "evidence_relevant", "evidence_spans"],
    "additionalProperties": False,
}

ANALYSIS_SCHEMA = {
    "name": "infoseek_step_analysis_v2",
    "strict": True,
    "schema": {
        "type": "object",
        "properties": {
            "thought_type": {
                "type": "string",
                "enum": ["procedural_plan", "factual_assertion", "mixed", "no_claim"],
            },
            "query_aligned": {"type": "boolean"},
            "atomic_claims": {"type": "array", "items": CLAIM_ITEM, "maxItems": 6},
            "reason": {"type": "string"},
        },
        "required": ["thought_type", "query_aligned", "atomic_claims", "reason"],
        "additionalProperties": False,
    },
}


class GPT54Analyzer:
    def __init__(self, model: str = "gpt-5.4"):
        from openai import OpenAI
        self.client = OpenAI()
        self.model = model

    def analyze(self, *, question: str, step: dict[str, Any], evidence: str) -> tuple[dict[str, Any], dict[str, int]]:
        action = step.get("action") or {}
        query = (action.get("arguments") or {}).get("query", "")
        prompt = f"""Audit one step of a search agent. Do not answer the task and do not judge entailment.

QUESTION:\n{question[:4000]}
STEP TYPE: {step.get('step_type')}
THOUGHT:\n{str(step.get('thought') or '')[:5000]}
SEARCH QUERY:\n{str(query)[:2000]}
FINAL RESPONSE:\n{str(step.get('response') or '')[:3000]}
AVAILABLE EVIDENCE (only evidence available by this step):\n{evidence[:16000]}

Rules:
1. Classify the thought as procedural_plan, factual_assertion, mixed, or no_claim. A proposed search strategy, information need, question, uncertainty, or intention is procedural—not a factual claim.
2. query_aligned asks whether a search query targets a useful entity/relation. Retrieval failure is not query drift. For answer steps set true.
3. Decompose only factual content actually asserted into minimal atomic claims. Do not turn a procedural plan into a claim. For mixed thoughts omit procedural content.
   CRITICAL FOR ANSWER STEPS: ignore auxiliary/background assertions in THOUGHT. Construct only the smallest proposition(s) required to express and justify FINAL RESPONSE relative to QUESTION. Usually this is one identity proposition; split it only when the question itself requires multiple independent constraints.
4. For each atomic claim, mark whether any available evidence addresses its entity and relation.
5. For each atomic claim, copy zero to three SHORT, CONTIGUOUS, VERBATIM spans from AVAILABLE EVIDENCE. Never join spans with ellipses and never paraphrase. Separate supporting facts belong in separate array items.
6. Do not output CC/IE/MB/no-gap and do not perform entailment yourself.
"""
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_schema", "json_schema": ANALYSIS_SCHEMA},
            max_completion_tokens=1800,
        )
        value = json.loads(response.choices[0].message.content)
        rejected = 0
        for claim in value["atomic_claims"]:
            raw = list(claim["evidence_spans"])
            claim["evidence_spans"] = [span for span in raw if span and span in evidence]
            claim["rejected_nonverbatim_spans"] = [span for span in raw if span and span not in evidence]
            rejected += len(claim["rejected_nonverbatim_spans"])
        value["rejected_nonverbatim_count"] = rejected
        usage = response.usage
        return value, {
            "input_tokens": int(getattr(usage, "prompt_tokens", 0) or 0),
            "output_tokens": int(getattr(usage, "completion_tokens", 0) or 0),
        }


def evaluate_analysis(analysis: dict[str, Any], nli: Any) -> dict[str, Any]:
    path = ["gpt54:v2_structured_ok"]
    if not analysis["query_aligned"]:
        return _decision("irrelevant_evidence", True, "reformulate query", path + ["alignment:off_target"], analysis)
    path.append("alignment:pass")
    claims = analysis["atomic_claims"]
    if analysis["thought_type"] in {"procedural_plan", "no_claim"} and not claims:
        return _decision("not_applicable", False, None, path + ["claim:not_eligible"], analysis)
    if not claims:
        return _decision("extraction_failure", False, "review claim extraction", path + ["claim:extraction_failure"], analysis)

    outcomes = []
    for index, item in enumerate(claims):
        spans = item["evidence_spans"]
        if item["evidence_relevant"] and not spans:
            outcomes.append({"claim_index": index, "status": "extraction_failure", "nli": []})
            continue
        if not item["evidence_relevant"]:
            outcomes.append({"claim_index": index, "status": "irrelevant", "nli": []})
            continue
        scores = []
        for span in spans:
            label, confidence = nli.classify(span, item["claim"])
            scores.append({"mode": "single", "label": label, "confidence": confidence})
        if any(score["label"] == "entailment" for score in scores):
            status = "entailed"
        else:
            if len(spans) > 1:
                label, confidence = nli.classify(" ".join(spans), item["claim"])
                scores.append({"mode": "joint", "label": label, "confidence": confidence})
            if any(score["label"] == "entailment" for score in scores):
                status = "entailed_joint"
            elif any(score["label"] == "contradiction" for score in scores):
                status = "contradicted"
            else:
                status = "insufficient"
        outcomes.append({"claim_index": index, "status": status, "nli": scores})

    analysis["claim_outcomes"] = outcomes
    statuses = {outcome["status"] for outcome in outcomes}
    path.append("claims:" + ",".join(sorted(statuses)))
    if "contradicted" in statuses:
        return _decision("contradicted_claim", True, "retract", path, analysis)
    if "irrelevant" in statuses:
        return _decision("irrelevant_evidence", True, "re-search", path, analysis)
    if "insufficient" in statuses:
        return _decision("missing_bridge", True, "bridging search", path, analysis)
    if "extraction_failure" in statuses:
        return _decision("extraction_failure", False, "review evidence extraction", path, analysis)
    return _decision("no_gap", False, None, path, analysis)


def _decision(label: str, has_gap: bool, repair: str | None, path: list[str], analysis: dict[str, Any]) -> dict[str, Any]:
    return {"label": label, "has_gap": has_gap, "repair": repair, "pipeline_path": path, "analysis": analysis}
