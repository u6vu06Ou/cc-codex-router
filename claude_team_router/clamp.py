"""Context-window protection (the clamp suite), ported from production.

Why this exists: agent clients re-send the whole conversation every turn, and
nothing else in the chain knows the backend's true context hard wall. If the
prompt overflows, the backend 400s and the session bricks. Estimation is
deliberately biased to over-count (clamp early = safe).

Kept behaviors:
- budget = context_window * (1 - safety margin) - reserved output tokens
- tool schemas count toward the budget, capped at half (a pathological MCP
  tool list must not silently erase all history — a loud upstream 400 beats a
  silent amnesia loop)
- any single oversized message is cut to head 60% + tail 40%
- fat tool results are shrunk in place before whole turns are dropped
- the first user message (the original task) is pinned; if it alone exceeds
  half the budget it is shrunk in place rather than dropping everything else
- turns drop on user/assistant boundaries only (tool_call/tool pairs stay
  intact), from the oldest forward
- drop points are quantized to a fixed step so consecutive requests present a
  byte-identical prefix to the backend (prefix-cache stays warm; measured 9x
  prefill speedup)
- when anything was dropped, a fixed-wording notice is inserted so the model
  knows history is missing (an unaware model fills the gap with confident
  invention instead of saying "I forgot")
- requests with no text-bearing user message get a wrap-up placeholder
  (backends that demand a user query otherwise 400)
- max_tokens adapts down to fit the remaining budget — and never rises above
  what the client asked for
- reactive trim: if the backend still answers 400 "context length", drop the
  oldest third on a boundary and retry (the backend is the judge)
"""
from __future__ import annotations

import json
import logging

from .config import Config, ModelCfg


log = logging.getLogger("claude_team_router.clamp")

CTX_SAFETY_MARGIN = 0.10
MIN_OUTPUT_TOKENS = 16_384
TOOL_RESULT_SHRINK_CHARS = 8_000
IMAGE_EST_TOKENS = 1_600   # flat per-image cost, over-estimated on purpose
CLAMP_DROP_QUANTUM = 64    # quantize drop counts (prefix-cache stability)
CLAMP_MIN_KEEP_MSGS = 40   # never let quantization eat the last 40 messages


def est_msg_tokens(msg: dict) -> int:
    total = 0
    c = msg.get("content", "") or ""
    if isinstance(c, list):
        parts = []
        for b in c:
            if isinstance(b, dict) and b.get("type") == "image_url":
                total += IMAGE_EST_TOKENS
            else:
                parts.append(str(b))
        s = "".join(parts)
    else:
        s = str(c)
    for tc in (msg.get("tool_calls") or []):
        fn = tc.get("function", {})
        s += str(fn.get("name", "")) + str(fn.get("arguments", ""))
    s += str(msg.get("reasoning_content", "") or "")
    # CJK ~1 tok/char, ASCII ~1/3 tok/char; over-estimate = clamp early = safe.
    if s.isascii():
        ascii_n = len(s)
    else:
        nbytes = len(s.encode("utf-8", "ignore"))
        non_ascii = min(len(s), (nbytes - len(s) + 1) // 2)
        ascii_n = len(s) - non_ascii
    # Floor of 1/msg: a zero estimate would silently break the auto-compact
    # water-level signal downstream.
    return total + max(ascii_n // 3 + (len(s) - ascii_n), 1)


def est_tokens(messages: list) -> int:
    return sum(est_msg_tokens(m) for m in messages)


def est_tool_tokens(openai_req: dict) -> int:
    tools = openai_req.get("tools") or []
    if not tools:
        return 0
    serialized = json.dumps(tools, ensure_ascii=False, separators=(",", ":"))
    return est_msg_tokens({"role": "system", "content": serialized})


def est_request_tokens(openai_req: dict) -> int:
    return est_tokens(openai_req.get("messages", [])) + est_tool_tokens(openai_req)


def ctx_budget(model: ModelCfg, reserve: int) -> int:
    return int(model.context_window * (1 - CTX_SAFETY_MARGIN)) - reserve


def _cut(s: str, limit: int) -> str:
    """Keep head 60% + tail 40% of a string, marking what was removed."""
    if len(s) <= limit:
        return s
    h = int(limit * 0.6)
    return s[:h] + f"\n...[proxy trimmed {len(s) - limit} chars]...\n" + s[-(limit - h):]


def _shrink_msg_to(m: dict, max_tokens: int) -> None:
    """Cut ONE message down to roughly max_tokens, in place."""
    c = m.get("content")
    if isinstance(c, str):
        m["content"] = _cut(c, max_tokens)
        return
    if not isinstance(c, list):
        return
    texts = [b for b in c if isinstance(b, dict) and b.get("type") == "text"]
    per = max(max_tokens // max(len(texts), 1), 200)
    for b in texts:
        b["text"] = _cut(str(b.get("text", "")), per)
    # Still over? Then it is images. Drop oldest first, always keep the last.
    while est_msg_tokens(m) > max_tokens:
        imgs = [b for b in c if isinstance(b, dict) and b.get("type") == "image_url"]
        if len(imgs) <= 1:
            break
        c.remove(imgs[0])


def _shrink_tool_results(msgs: list, budget: int) -> None:
    """Shrink fat tool outputs in place before any whole turn is dropped."""
    toks = [est_msg_tokens(m) for m in msgs]
    running = sum(toks)
    if running <= budget:
        return
    shrunk = 0
    for i, m in enumerate(msgs):
        if running <= budget:
            break
        if m.get("role") != "tool":
            continue
        c = m.get("content")
        if not isinstance(c, str) or len(c) <= TOOL_RESULT_SHRINK_CHARS:
            continue
        h = int(TOOL_RESULT_SHRINK_CHARS * 0.6)
        m["content"] = (c[:h]
                        + f"\n...[proxy truncated {len(c) - TOOL_RESULT_SHRINK_CHARS} chars]...\n"
                        + c[h - TOOL_RESULT_SHRINK_CHARS:])
        running -= toks[i] - est_msg_tokens(m)
        shrunk += 1
    if shrunk:
        log.info("[ctx-clamp] shrank %d fat tool result(s) in place", shrunk)


def _user_has_text(m: dict) -> bool:
    if m.get("role") != "user":
        return False
    c = m.get("content")
    if isinstance(c, str):
        return bool(c.strip())
    if isinstance(c, list):
        return any(isinstance(b, dict) and b.get("type") == "text"
                   and str(b.get("text", "")).strip() for b in c)
    return False


def adaptive_max_tokens(openai_req: dict, model: ModelCfg) -> dict:
    """Shrink output cap to fit remaining budget; never raise above the ask."""
    client = int(openai_req.get("max_tokens") or 4096)
    usable = ctx_budget(model, 0)
    est_in = est_request_tokens(openai_req)
    shrunk = min(client, max(MIN_OUTPUT_TOKENS, usable - est_in))
    if model.max_output:
        shrunk = min(shrunk, model.max_output)
    if shrunk != client:
        log.info("[ctx-clamp] adaptive max_tokens %d -> %d (est input ~%d tok)",
                 client, shrunk, est_in)
    openai_req["max_tokens"] = shrunk
    return openai_req


def clamp_context(openai_req: dict, model: ModelCfg, cfg: Config) -> dict:
    msgs = openai_req.get("messages", [])
    budget = ctx_budget(model, MIN_OUTPUT_TOKENS)

    # Tool schemas ride the request, not messages — deduct first.
    tool_tok = est_tool_tokens(openai_req)
    if tool_tok:
        cap = max(budget // 2, 0)
        if tool_tok > cap:
            log.warning("[ctx-clamp] tool schemas ~%d tok > half the budget; "
                        "deduction capped", tool_tok)
            budget -= cap
        else:
            budget -= tool_tok

    # (1) cap any single oversized message (huge file read / paste).
    for m in msgs:
        if m.get("role") == "system":
            continue
        c = m.get("content")
        if isinstance(c, str) and len(c) > model.max_message_chars:
            m["content"] = _cut(c, model.max_message_chars)

    # (1.5) shrink fat tool results in place before dropping whole turns.
    _shrink_tool_results(msgs, budget)

    # (2) drop oldest WHOLE turns on role boundaries if still over.
    if est_tokens(msgs) <= budget:
        return adaptive_max_tokens(openai_req, model)

    i = 0
    while i < len(msgs) and msgs[i].get("role") == "system":
        i += 1
    head, body = msgs[:i], msgs[i:]

    def _drop_to_boundary():
        while body and body[0].get("role") not in ("user", "assistant"):
            body.pop(0)

    pinned = []
    if body and body[0].get("role") == "user":
        pinned = [body[0]]
        body = body[1:]
        _drop_to_boundary()

    # A pinned task that eats the budget alone gets shrunk, not kept whole.
    pin_cap = max(budget - est_tokens(head), 2000) // 2
    if pinned and est_tokens(pinned) > pin_cap:
        log.warning("[ctx-clamp] pinned task over cap; trimming it so history fits")
        _shrink_msg_to(pinned[0], pin_cap)

    body_tok = [est_msg_tokens(m) for m in body]
    running = est_tokens(head + pinned) + sum(body_tok)
    drop = 0
    while drop < len(body) and running > budget:
        running -= body_tok[drop]
        drop += 1
    if drop:
        quantized = min(len(body), -(-drop // CLAMP_DROP_QUANTUM) * CLAMP_DROP_QUANTUM)
        if quantized <= len(body) - CLAMP_MIN_KEEP_MSGS:
            drop = quantized
        while drop < len(body) and body[drop].get("role") not in ("user", "assistant"):
            drop += 1
    body = body[drop:]

    if not body and len(msgs) - len(head) - len(pinned) > 2:
        log.warning("[ctx-clamp] body drained to EMPTY (task-only context)")

    if not any(_user_has_text(m) for m in pinned + body):
        from .config import WRAP_UP_PLACEHOLDER
        body = [{"role": "user", "content": WRAP_UP_PLACEHOLDER}]

    dropped = len(msgs) - len(head) - len(pinned) - len(body)
    if dropped:
        body.insert(0, {"role": "user", "content": cfg.clamp_notice})
        log.info("[ctx-clamp] dropped %d oldest msgs (task pinned, notice inserted)",
                 dropped)
    openai_req["messages"] = head + pinned + body
    return adaptive_max_tokens(openai_req, model)


def shrink_messages(openai_req: dict) -> bool:
    """Reactive trim: drop the oldest ~1/3 of non-system turns at a boundary.

    Used when the backend itself rejects with context overflow — the backend
    is the judge, so this works however badly estimation misjudged tokens.
    Returns True when it actually shrank something.
    """
    msgs = openai_req.get("messages", [])
    i = 0
    while i < len(msgs) and msgs[i].get("role") == "system":
        i += 1
    head, body = msgs[:i], msgs[i:]
    if len(body) <= 1:
        return False
    cut = max(1, len(body) // 3)
    while cut < len(body) and body[cut].get("role") not in ("user", "assistant"):
        cut += 1
    if cut >= len(body):
        return False
    pinned = [body[0]] if body[0].get("role") == "user" and cut > 0 else []
    openai_req["messages"] = head + pinned + body[cut:]
    return True
