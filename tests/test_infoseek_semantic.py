from checker_agent.infoseek.semantic import evaluate_analysis


class FakeNLI:
    def classify(self, premise, hypothesis):
        if "supports" in premise:
            return "entailment", .9
        if "opposes" in premise:
            return "contradiction", .8
        return "neutral", .7


def analysis(kind="factual_assertion", claims=None):
    return {"thought_type": kind, "query_aligned": True, "atomic_claims": claims or [], "reason": "test"}


def claim(spans, relevant=True):
    return {"claim": "X is Y", "evidence_relevant": relevant, "evidence_spans": spans,
            "rejected_nonverbatim_spans": []}


def test_procedural_is_not_applicable():
    assert evaluate_analysis(analysis("procedural_plan"), FakeNLI())["label"] == "not_applicable"


def test_extraction_failure_is_not_ie():
    result = evaluate_analysis(analysis(claims=[claim([])]), FakeNLI())
    assert result["label"] == "extraction_failure" and not result["has_gap"]


def test_atomic_and_joint_routes():
    assert evaluate_analysis(analysis(claims=[claim(["supports X"])]), FakeNLI())["label"] == "no_gap"
    assert evaluate_analysis(analysis(claims=[claim(["opposes X"])]), FakeNLI())["label"] == "contradicted_claim"
    assert evaluate_analysis(analysis(claims=[claim(["partial one", "supports jointly"])]), FakeNLI())["label"] == "no_gap"
    assert evaluate_analysis(analysis(claims=[claim(["partial"])]), FakeNLI())["label"] == "missing_bridge"
