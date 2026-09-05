from checker_agent.infoseek.adapter import parse_assistant_content


def test_parse_search_and_answer():
    content = """<think>need a fact</think><search>alpha</search>
    <information>alpha evidence</information><think>done</think><answer>A</answer>"""
    trajectory = parse_assistant_content(content, "infoseek::x::rft00000", "question")
    assert trajectory["parse"]["complete"] is True
    assert [step["step_type"] for step in trajectory["steps"]] == ["search", "answer"]
    assert trajectory["steps"][0]["action"]["arguments"]["query"] == "alpha"
    assert trajectory["steps"][0]["observation"] == "alpha evidence"
    assert trajectory["final_response"] == "A"


def test_missing_information_is_incomplete():
    content = "<think>x</think><search>q</search><think>y</think><answer>a</answer>"
    trajectory = parse_assistant_content(content, "infoseek::x::rft00001", "q")
    assert trajectory["parse"]["complete"] is False
