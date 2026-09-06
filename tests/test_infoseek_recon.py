import importlib.util
from pathlib import Path

MODULE_PATH = Path(__file__).parents[1] / "scripts" / "infoseek_recon.py"
SPEC = importlib.util.spec_from_file_location("infoseek_recon", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def test_walk_and_title_parsing():
    tree = {"claims": ["a"], "children": [{"claims": ["b"], "children": []}]}
    assert [depth for depth, _ in MODULE.walk(tree)] == [0, 1]
    assert MODULE.title_from_href("https://en.wikipedia.org/wiki/New_York_City#x") == "New York City"
    assert MODULE.title_from_href("New%20York%20City") == "New York City"


def test_schema_does_not_copy_text():
    result = MODULE.schema({"question": "private benchmark text", "values": [1]})
    assert result == {"question": "str", "values": {"type": "list", "item": "int"}}
    assert "private benchmark text" not in str(result)
