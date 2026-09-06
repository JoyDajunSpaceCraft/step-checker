import json
import unittest

from checker_agent.browsecomp_plus.mcp_bridge import BrowseCompAuditEngine


class BrowseCompAuditEngineTest(unittest.TestCase):
    def setUp(self):
        self.engine = BrowseCompAuditEngine()
        self.trace = self.engine.start_trace("bcp::1", "Find the answer", "test")["trace_uid"]
        self.output = json.dumps([{"docid": "d1", "score": 1.0, "snippet": "Evidence text."}])

    def test_unconfigured_semantics_are_not_no_gap(self):
        result = self.engine.record_search(self.trace, "Need evidence", "alpha beta", self.output)
        self.assertEqual(result["semantic"]["status"], "not_evaluated")
        self.assertEqual(result["deterministic"]["label"], "no_deterministic_gap")

    def test_duplicate_query_is_deterministic(self):
        self.engine.record_search(self.trace, "First", "Alpha, beta!", self.output)
        result = self.engine.record_search(self.trace, "Again", "alpha beta", self.output)
        self.assertEqual(result["deterministic"]["label"], "redundant_action")
        self.assertTrue(result["deterministic"]["has_gap"])
        exported = self.engine.export_trace(self.trace)
        self.assertEqual(len(exported["steps"]), 2)
        self.assertEqual(len(exported["observations"]), 2)

    def test_answer_without_evidence_is_only_candidate(self):
        fresh = self.engine.start_trace("bcp::2", "Question", "test")["trace_uid"]
        result = self.engine.finish_trace(fresh, "An answer")
        self.assertEqual(result["deterministic"]["label"], "premature_answer_candidate")
        self.assertTrue(result["deterministic"]["requires_semantic_confirmation"])


if __name__ == "__main__":
    unittest.main()
