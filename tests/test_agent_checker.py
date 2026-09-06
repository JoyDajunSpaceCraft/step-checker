import unittest

from checker_agent import Action, AgentStep, AgentStepGapChecker, PolicyRule, ToolSpec


class StubNLI:
    def classify(self, premise, hypothesis):
        p, h = premise.lower(), hypothesis.lower()
        if "delivered" in p and "pending" in h:
            return "contradiction", 0.99
        if "delivered" in p and "delivered" in h:
            return "entailment", 0.98
        if "cancelled" in p and "cancelled" in h:
            return "entailment", 0.98
        return "neutral", 0.80


SPECS = [
    ToolSpec("get_order_details", False, [], "order_id"),
    ToolSpec("modify_pending_order_items", True, ["pending"], "order_id"),
    ToolSpec("exchange_delivered_order_items", True, ["delivered"], "order_id"),
    ToolSpec("cancel_pending_order", True, ["pending"], "order_id"),
]
RULES = [
    PolicyRule("exchange-confirm", "exchange_delivered_order_items", "Confirm before exchange.", True, {"status": "delivered"}),
]


class AgentCheckerTests(unittest.TestCase):
    def setUp(self):
        self.checker = AgentStepGapChecker(SPECS, RULES, nli=StubNLI())

    def test_twenty_targeted_cases(self):
        read = lambda oid="W123": Action("get_order_details", {"order_id": oid}, False)
        exchange = lambda oid="W123": Action("exchange_delivered_order_items", {"order_id": oid}, True)
        modify = lambda oid="W123": Action("modify_pending_order_items", {"order_id": oid}, True)
        prior_read = AgentStep("prior", action=read(), observation={"order_id": "W123", "status": "delivered"})
        cases = [
            (AgentStep("01", "Order W123 is delivered.", read(), {"order_id": "W123", "status": "delivered"}), "no_gap", "no_gap"),
            (AgentStep("02", "Order W123 is pending.", read(), {"order_id": "W123", "status": "delivered"}), "contradicted_claim", "no_gap"),
            (AgentStep("03", "Order W999 is delivered.", read(), {"order_id": "W123", "status": "delivered"}), "irrelevant_evidence", "no_gap"),
            (AgentStep("04", "Order W123 is refundable.", read(), {"order_id": "W123", "status": "delivered"}), "missing_bridge", "no_gap"),
            (AgentStep("05", "Order W123 is delivered.", exchange(), {"order_id": "W123", "status": "delivered"}, user_turns=["Exchange W123"], known_state={"status": "delivered"}), "no_gap", "policy_violation"),
            (AgentStep("06", "Order W123 is delivered.", exchange(), {"order_id": "W123", "status": "delivered"}, user_turns=["Exchange W123", "Yes, confirm"], known_state={"status": "delivered"}), "no_gap", "no_gap"),
            (AgentStep("07", action=modify(), user_turns=["Modify W123"], known_state={"status": "delivered"}), "no_gap", "wrong_tool"),
            (AgentStep("08", action=exchange(), user_turns=["Exchange W123", "confirm"], known_state={"status": "pending"}), "no_gap", "wrong_tool"),
            (AgentStep("09", action=Action("missing_tool", {}, False)), "no_gap", "wrong_tool"),
            (AgentStep("10", action=read(), user_turns=["Read W123"], prior_steps=[prior_read]), "no_gap", "redundant_action"),
            (AgentStep("11", action=read(), user_turns=["Read W123"], prior_steps=[AgentStep("p", action=read(), observation={"error": "timeout"})]), "no_gap", "no_gap"),
            (AgentStep("12", action=read(), user_turns=["Read W123"], prior_steps=[prior_read, AgentStep("w", action=modify(), known_state={"status": "pending"})]), "no_gap", "no_gap"),
            (AgentStep("13", response="Done", known_state={"status": "pending"}, expected_final_state={"status": "cancelled"}), "no_gap", "premature_commit"),
            (AgentStep("14", response="Done", known_state={"status": "cancelled"}, expected_final_state={"status": "cancelled"}), "no_gap", "no_gap"),
            (AgentStep("15", thought="Need more information"), "no_gap", "no_gap"),
            (AgentStep("16", thought="Order W123 is delivered.", observation={"order_id": "W123", "status": "unknown"}, prior_steps=[prior_read]), "no_gap", "no_gap"),
            (AgentStep("17", action=read("W456"), user_turns=["Read W123"], known_state={"order_id": "W123"}), "no_gap", "wrong_tool"),
            (AgentStep("18", action=read("W123"), user_turns=["Read W123"], known_state={"order_id": "W123"}), "no_gap", "no_gap"),
            (AgentStep("19", thought="Order W123 is pending.", action=exchange(), observation={"order_id": "W123", "status": "delivered"}, user_turns=["Exchange W123"], known_state={"status": "delivered"}), "contradicted_claim", "policy_violation"),
            (AgentStep("20", thought="Order W123 is refundable.", action=modify(), observation={"order_id": "W123", "status": "delivered"}, user_turns=["Modify W123"], known_state={"status": "delivered"}), "missing_bridge", "wrong_tool"),
        ]
        self.assertEqual(len(cases), 20)
        for step, cognitive, normative in cases:
            with self.subTest(uid=step.uid):
                result = self.checker.check(step)
                self.assertEqual(result.cognitive.label, cognitive)
                self.assertEqual(result.normative.label, normative)
                self.assertTrue(result.pipeline_path)

    def test_multi_axis_preserved(self):
        step = AgentStep(
            "both", "Order W123 is pending.",
            Action("exchange_delivered_order_items", {"order_id": "W123"}, True),
            {"order_id": "W123", "status": "delivered"},
            user_turns=["Exchange W123"], known_state={"status": "delivered"},
        )
        result = self.checker.check(step)
        self.assertTrue(result.metadata["multi_axis"])
        self.assertEqual(result.primary_label, "policy_violation")


if __name__ == "__main__":
    unittest.main()
