"""Tool-argument sanitizing: nothing unparseable ever reaches upstream."""
import json

from cc_codex_router.sanitize import sanitize_tool_arguments


def test_valid_json_passes_through():
    assert sanitize_tool_arguments('{"a": 1}') == '{"a": 1}'


def test_empty_becomes_empty_object():
    assert sanitize_tool_arguments("") == "{}"
    assert sanitize_tool_arguments(None) == "{}"


def test_template_leak_tail_is_cut_and_json_closed():
    raw = '{"cmd": "ls -la"<tool_call>garbage tail'
    out = sanitize_tool_arguments(raw, "Bash")
    assert json.loads(out) == {"cmd": "ls -la"}


def test_truncated_json_is_closed():
    out = sanitize_tool_arguments('{"a": {"b": [1, 2', "F")
    assert json.loads(out) == {"a": {"b": [1, 2]}}


def test_open_string_closed():
    out = sanitize_tool_arguments('{"a": "unfinished', "F")
    assert json.loads(out) == {"a": "unfinished"}


def test_hopeless_input_collapses_to_empty_object():
    assert sanitize_tool_arguments("not json at all ]}", "F") == "{}"


def test_non_string_input_serialized():
    assert sanitize_tool_arguments({"x": 2}) == '{"x": 2}'
