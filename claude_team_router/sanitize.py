"""Repair tool-call argument JSON before it reaches an upstream tokenizer.

A model that leaks chat-template markers mid-arguments or stops mid-JSON can
poison the whole conversation: the broken arguments string gets echoed back
in history forever and every later request fails to parse. So arguments are
never forwarded raw — valid JSON passes through; repairable JSON is closed;
hopeless input collapses to "{}" and a warning.
"""
from __future__ import annotations

import json
import logging

log = logging.getLogger("claude_team_router.sanitize")

_TEMPLATE_LEAK_MARKERS = (
    "</think>", "<think>", "<tool_call>", "</tool_call>",
    "<arg_key>", "<arg_value>", "<|im_start|>", "<|im_end|>",
)


def _close_open_json(text: str):
    """Close an unterminated JSON string/object/array; None if unfixable."""
    stack = []
    in_string = False
    escaped = False
    for ch in text:
        if in_string:
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
        elif ch in "{[":
            stack.append(ch)
        elif ch in "}]":
            if not stack:
                return None
            stack.pop()
    repaired = text[:-1] if escaped else text
    if in_string:
        repaired += '"'
    repaired += "".join("}" if ch == "{" else "]" for ch in reversed(stack))
    return repaired


def sanitize_tool_arguments(raw, call_name: str = "") -> str:
    """Return valid JSON text for a tool call's `arguments` field."""
    text = raw if isinstance(raw, str) else json.dumps(raw or {})
    if not text.strip():
        return "{}"
    try:
        json.loads(text)
        return text
    except (json.JSONDecodeError, ValueError):
        pass
    cut = min((text.find(m) for m in _TEMPLATE_LEAK_MARKERS if m in text),
              default=-1)
    candidate = text[:cut] if cut > 0 else text
    repaired = _close_open_json(candidate)
    if repaired:
        try:
            if isinstance(json.loads(repaired), (dict, list)):
                log.warning("[tool-args] repaired broken arguments name=%r (%d -> %d chars)",
                            call_name or "?", len(text), len(repaired))
                return repaired
        except (json.JSONDecodeError, ValueError):
            pass
    log.warning("[tool-args] arguments unfixable name=%r (%d chars) -> {}",
                call_name or "?", len(text))
    return "{}"
