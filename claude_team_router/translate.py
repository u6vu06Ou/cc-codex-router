"""Anthropic Messages request -> OpenAI chat/completions request.

Ports the production proxy's inbound fixes:

- tool_use ids re-minted as unique `toolu_...` (some backends emit
  deterministic ids like "Read:0"; replayed ids break strict clients).
- thinking-only assistant turns never vanish: reasoning preserved for native
  reasoning models, otherwise a placeholder keeps role alternation intact.
- user turns that contain only <system-reminder> wrappers are hoisted to the
  system role (non-Anthropic models otherwise think the user said nothing).
- degenerate empty user/assistant turns are stripped.
- tool_result image blocks are not dropped: vision models get them as
  image_url in a follow-up user message; other models get an honest note.
- mid-conversation system messages are merged into the leading system
  message for backends that demand system-first ordering (always done when
  the request carries an image; opt-in per model otherwise).
- tools: input_schema -> parameters; tool_choice mapped to OpenAI's shape.
"""
from __future__ import annotations

import json
import re
import secrets

from .config import ModelCfg
from .sanitize import sanitize_tool_arguments

NO_CONTENT_PLACEHOLDER = "(no content)"
THINKING_ONLY_PLACEHOLDER = "(This turn contained only internal reasoning; no visible output was produced.)"
IMAGE_OMITTED_NOTE = "[image omitted: this model has no vision support]"
TOOL_IMAGE_INTRO = "(The following images were returned by the tool call above.)"
TOOL_IMAGE_HINT = "[The tool returned image data; it is shown in the next message.]"

_REMINDER_RE = re.compile(r"<system-reminder>.*?</system-reminder>", re.DOTALL)


def unique_tool_id() -> str:
    return "toolu_" + secrets.token_hex(12)


def _is_reminder_only(blocks: list) -> bool:
    """True when every text block of the turn is wrapped in <system-reminder>."""
    saw_reminder = False
    for b in blocks:
        if not isinstance(b, dict) or b.get("type") != "text":
            return False
        text = b.get("text", "")
        if not text.strip():
            continue
        stripped = _REMINDER_RE.sub("", text).strip()
        if stripped:
            return False
        if _REMINDER_RE.search(text):
            saw_reminder = True
    return saw_reminder


def _is_blank(blocks: list) -> bool:
    for b in blocks:
        if isinstance(b, dict) and b.get("type") == "text" and b.get("text", "").strip():
            return False
    return True


def _strip_degenerate_turns(messages: list) -> list:
    """Drop user/assistant turns with no content at all.

    Agent loops sometimes leave empty assistant + "(no content)" user pairs
    behind; shipping them makes some backends answers hallucinate a silent
    user. Removing whole empty turns keeps role alternation paired.
    """
    out = []
    for m in messages:
        role = m.get("role")
        if role not in ("user", "assistant"):
            out.append(m)
            continue
        if m.get("tool_calls"):
            out.append(m)
            continue
        c = m.get("content")
        if isinstance(c, list):
            texts = [b.get("text", "") for b in c
                     if isinstance(b, dict) and b.get("type") == "text"]
            imgs = [b for b in c
                    if isinstance(b, dict) and b.get("type") == "image_url"]
            if imgs:
                out.append(m)
                continue
            real = [t for t in texts
                    if t.strip() and t.strip() != NO_CONTENT_PLACEHOLDER]
            if real:
                out.append(m)
            continue
        if isinstance(c, str) and c.strip() and c.strip() != NO_CONTENT_PLACEHOLDER:
            out.append(m)
            continue
        if m.get("reasoning_content"):
            out.append(m)
            continue
        # empty turn: drop
    return out


def _msg_has_image(m: dict) -> bool:
    c = m.get("content")
    return isinstance(c, list) and any(
        isinstance(b, dict) and b.get("type") == "image_url" for b in c)


def hoist_mid_system_messages(messages: list) -> list:
    """Merge any system message after position 0 into the leading system one.

    Some chat backends reject "System message must be at the beginning." if a
    system message appears mid-conversation (which hoisted reminders create).
    """
    system_parts = [str(m.get("content", "")) for m in messages
                    if m.get("role") == "system"]
    if len(system_parts) <= 1:
        return messages
    out = [{"role": "system", "content": "\n\n".join(system_parts)}]
    out.extend(m for m in messages if m.get("role") != "system")
    return out


def _flatten_system(system) -> str:
    if isinstance(system, list):
        return "\n\n".join(b.get("text", "") for b in system
                           if isinstance(b, dict) and b.get("type") == "text")
    return str(system or "")


def translate_anthropic_to_openai(anthropic_req: dict, model: ModelCfg) -> dict:
    out_messages = []
    system_prompt = _flatten_system(anthropic_req.get("system", ""))
    if system_prompt.strip():
        out_messages.append({"role": "system", "content": system_prompt})

    preserve_reasoning = model.reasoning == "native"

    for msg in anthropic_req.get("messages", []):
        role = msg.get("role")
        content = msg.get("content")

        if not isinstance(content, list):
            if content:
                out_messages.append({"role": role, "content": content})
            continue

        blocks = []
        tool_calls = []
        has_thinking = False
        reasoning_parts = []
        pending_tool_images = []

        for item in content:
            if not isinstance(item, dict):
                continue
            t = item.get("type")
            if t == "text":
                blocks.append({"type": "text", "text": item.get("text", "")})
            elif t in ("thinking", "redacted_thinking"):
                has_thinking = True
                if t == "thinking" and item.get("thinking"):
                    reasoning_parts.append(item["thinking"])
            elif t == "image":
                source = item.get("source") or {}
                if model.vision and source.get("data"):
                    blocks.append({
                        "type": "image_url",
                        "image_url": {"url": f"data:{source.get('media_type', 'image/png')};base64,{source['data']}"},
                    })
                else:
                    blocks.append({"type": "text", "text": IMAGE_OMITTED_NOTE})
            elif t == "tool_use":
                tool_calls.append({
                    "id": item.get("id") or unique_tool_id(),
                    "type": "function",
                    "function": {
                        "name": item.get("name", ""),
                        "arguments": sanitize_tool_arguments(
                            json.dumps(item.get("input") or {}), item.get("name", "")),
                    },
                })
            elif t == "tool_result":
                result_content = item.get("content", "")
                tr_images = []
                if isinstance(result_content, list):
                    tr_images = [
                        r for r in result_content
                        if isinstance(r, dict) and r.get("type") == "image"
                        and isinstance(r.get("source"), dict) and r["source"].get("data")
                    ]
                    result_content = " ".join(
                        r.get("text", "") for r in result_content
                        if isinstance(r, dict) and r.get("type") == "text")
                result_text = str(result_content)
                if tr_images and not result_text.strip():
                    result_text = TOOL_IMAGE_HINT
                out_messages.append({
                    "role": "tool",
                    "tool_call_id": item.get("tool_use_id", ""),
                    "content": result_text,
                })
                pending_tool_images.extend(tr_images)

        # A user turn whose entire payload is <system-reminder> wrappers is a
        # system instruction borrowed into a user shell — relabel it.
        if role == "user" and not tool_calls and _is_reminder_only(blocks):
            role = "system"

        if blocks or tool_calls:
            msg_obj = {"role": role, "content": blocks if blocks else ""}
            if tool_calls:
                msg_obj["tool_calls"] = tool_calls
            if role == "assistant" and preserve_reasoning and reasoning_parts:
                msg_obj["reasoning_content"] = "".join(reasoning_parts)
            out_messages.append(msg_obj)
        elif role == "assistant" and has_thinking:
            if preserve_reasoning and reasoning_parts:
                out_messages.append({
                    "role": "assistant",
                    "content": "",
                    "reasoning_content": "".join(reasoning_parts),
                })
            else:
                out_messages.append({
                    "role": "assistant",
                    "content": THINKING_ONLY_PLACEHOLDER,
                })

        # Images inside tool_result ride in a follow-up user message (OpenAI
        # tool messages only take string content).
        if pending_tool_images:
            img_blocks = [{"type": "text", "text": TOOL_IMAGE_INTRO}]
            for r in pending_tool_images:
                source = r.get("source") or {}
                if model.vision:
                    img_blocks.append({
                        "type": "image_url",
                        "image_url": {"url": f"data:{source.get('media_type', 'image/png')};base64,{source['data']}"},
                    })
                else:
                    img_blocks.append({"type": "text", "text": IMAGE_OMITTED_NOTE})
            out_messages.append({"role": "user", "content": img_blocks})

    out_tools = []
    for tool in anthropic_req.get("tools", []):
        out_tools.append({
            "type": "function",
            "function": {
                "name": tool.get("name", ""),
                "description": tool.get("description", ""),
                "parameters": tool.get("input_schema") or tool.get("parameters") or {},
            },
        })

    out_messages = _strip_degenerate_turns(out_messages)
    if model.strict_system_position or any(_msg_has_image(m) for m in out_messages):
        out_messages = hoist_mid_system_messages(out_messages)

    body = {
        "model": model.upstream_name(),
        "messages": out_messages,
        "stream": bool(anthropic_req.get("stream", False)),
        "max_tokens": int(anthropic_req.get("max_tokens") or 4096),
    }
    if "temperature" in anthropic_req:
        body["temperature"] = anthropic_req["temperature"]
    elif preserve_reasoning:
        body["temperature"] = 1.0
        body["top_p"] = anthropic_req.get("top_p", 1.0)
    if preserve_reasoning and "top_p" in anthropic_req:
        body["top_p"] = anthropic_req["top_p"]
    if anthropic_req.get("stop_sequences"):
        body["stop"] = anthropic_req["stop_sequences"]
    if preserve_reasoning:
        body["reasoning_effort"] = anthropic_req.get("reasoning_effort") \
            or model.reasoning_effort or "high"
    if model.extra_body:
        body.update(model.extra_body)
    if out_tools:
        body["tools"] = out_tools
        tc = anthropic_req.get("tool_choice")
        if isinstance(tc, dict):
            t = tc.get("type")
            if t == "auto":
                body["tool_choice"] = "auto"
            elif t == "any":
                body["tool_choice"] = "required"
            elif t == "tool" and tc.get("name"):
                body["tool_choice"] = {"type": "function",
                                       "function": {"name": tc["name"]}}
    return body
