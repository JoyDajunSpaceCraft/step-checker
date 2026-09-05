import json
import unittest

from checker_agent.browsecomp_plus.adapter import convert_official_record


class BrowseCompPlusAdapterTest(unittest.TestCase):
    def test_reasoning_search_and_final_are_preserved(self):
        output = json.dumps([
            {"docid": "d1", "score": 3.0, "snippet": "Alpha supports the first fact."},
            {"docid": "d2", "score": 2.0, "snippet": "Beta is unrelated."},
        ])
        record = {
            "query_id": "7", "status": "completed", "search_counts": {"search": 1},
            "result": [
                {"type": "reasoning", "tool_name": None, "arguments": None, "output": ["Find alpha."]},
                {"type": "tool_call", "tool_name": "search", "arguments": json.dumps({"query": "alpha"}), "output": output},
                {"type": "reasoning", "tool_name": None, "arguments": None, "output": ["The evidence is sufficient."]},
                {"type": "output_text", "tool_name": None, "arguments": None, "output": "Final answer"},
            ],
        }
        trajectory, observations = convert_official_record(
            record, source_run="synthetic", model="m", retriever="r", benchmark_version="v", seed=1,
        )
        self.assertEqual(trajectory.trace_uid, "bcp::7::m::r::seed1")
        self.assertEqual(len(trajectory.steps), 2)
        self.assertEqual(trajectory.steps[0].thought, "Find alpha.")
        self.assertEqual(trajectory.final_response, "Final answer")
        self.assertEqual(trajectory.retrieved_docids, ["d1", "d2"])
        self.assertEqual(len(observations), 1)

    def test_observations_are_content_addressed(self):
        tool = {"type": "tool_call", "tool_name": "search", "arguments": '{"query":"x"}', "output": "[]"}
        record = {"query_id": "1", "status": "completed", "result": [tool, tool]}
        trajectory, observations = convert_official_record(
            record, source_run="s", model="m", retriever="r", benchmark_version="v"
        )
        self.assertEqual(len(trajectory.steps), 2)
        self.assertEqual(len(observations), 1)
        self.assertEqual(trajectory.steps[0].observation_uid, trajectory.steps[1].observation_uid)


if __name__ == "__main__":
    unittest.main()
