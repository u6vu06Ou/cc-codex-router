"""OpenAI streaming -> Anthropic SSE, with retry, failover and empty-reply fixes.

Queue-based design (ported from production): a producer task reads upstream
and pushes SSE frames; a keepalive task emits comment lines so idle streams
survive impatient middleboxes; the generator drains the queue. All exit paths
(error, exception, normal) close the last block and send message_stop.

Kept behaviors:
- a delta carrying reasoning AND text emits both (the removed `continue`)
- the finish chunk does NOT end reading: the trailing usage chunk replaces the
  local token estimate, but only with non-zero values
- transport errors retry only while nothing has been emitted; after partial
  output an honest dropped-connection note is appended instead
- 429/500/502/503 retry with backoff, then fail over down the fallback chain
  (with a cooldown circuit breaker)
- an empty answer after the thinking budget burned out retries once with
  lowered reasoning; if still empty, a visible placeholder is emitted instead
  of a silent dead turn
- reactive context trim: backend 400 "context length" drops the oldest third
  and retries (the backend is the judge)
"""
from __future__ import annotations

import asyncio
import json
import logging
import os

import httpx

from .clamp import est_request_tokens, shrink_messages
from .translate import unique_tool_id


log = logging.getLogger("cc_codex_router.stream")

RETRY_CODES = {429, 500, 502, 503}
THINK_RUNAWAY_CHARS = 64_000  # thinking with zero visible output past this = cut + retry

EMPTY_RETRY_NOTE = ("[proxy] The backend produced reasoning but no visible answer "
                    "twice in a row; nothing to show for this turn. Please re-issue "
                    "the instruction, or try a different model.")
TRANSPORT_CUT_NOTE = ("\n[proxy] Upstream connection dropped mid-answer; the reply "
                      "above may be incomplete — please ask again.")
TRANSPORT_DEAD_NOTE = ("[proxy] Upstream unreachable ({kind}); retry in a moment.")


def _sse(event: str, payload: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(payload, ensure_ascii=False)}\n\n"


def _stop_of(finish: str) -> str:
    return {"tool_calls": "tool_use", "stop": "end_turn"}.get(finish, finish or "end_turn")


class UpstreamError(Exception):
    def __init__(self, status: int, body: str = ""):
        super().__init__(f"upstream {status}: {body[:200]}")
        self.status = status
        self.body = body


async def relay_stream(openai_req: dict, alias: str, rt, req_id: str = ""):
    """Yield Anthropic SSE frames for one /v1/messages streaming request."""
    cfg = rt.cfg
    msg_id = "msg_" + os.urandom(8).hex()
    queue: asyncio.Queue = asyncio.Queue()
    stop_event = asyncio.Event()
    _DONE = object()
    est_usage_in = est_request_tokens(openai_req)

    async def _keepalive():
        while not stop_event.is_set():
            try:
                await asyncio.wait_for(stop_event.wait(), timeout=cfg.keepalive_secs)
            except asyncio.TimeoutError:
                await queue.put(": keepalive\n\n")

    async def _produce():
        idx = 0
        in_text = in_tool = in_think = False
        stop_reason = "end_turn"
        t_start = asyncio.get_event_loop().time()
        content_len = 0
        think_chars = 0
        usage_in = est_usage_in
        usage_out = 0

        async def close_block():
            nonlocal idx, in_text, in_tool, in_think
            if in_text or in_tool or in_think:
                await queue.put(_sse("content_block_stop",
                                     {"type": "content_block_stop", "index": idx}))
                idx += 1
                in_text = in_tool = in_think = False

        async def open_text():
            nonlocal in_text
            await close_block()
            await queue.put(_sse("content_block_start", {
                "type": "content_block_start", "index": idx,
                "content_block": {"type": "text", "text": ""}}))
            in_text = True

        async def emit_text(text: str):
            nonlocal content_len
            if not in_text:
                await open_text()
            content_len += len(text)
            await queue.put(_sse("content_block_delta", {
                "type": "content_block_delta", "index": idx,
                "delta": {"type": "text_delta", "text": text}}))

        try:
            current_alias = alias
            ctx_trims = 0
            attempt = 0
            while attempt < cfg.retry_times + 1:  # +1: first try is not a retry
                model = rt.cfg.models[current_alias]
                req = dict(openai_req)
                req["model"] = model.upstream_name()
                req["max_tokens"] = max(int(req.get("max_tokens") or 4096), 8192)
                if model.max_output:
                    req["max_tokens"] = min(req["max_tokens"], model.max_output)
                req["stream_options"] = {"include_usage": True}
                t0 = asyncio.get_event_loop().time()
                try:
                    async with rt.backend_stream(model, req) as resp:
                        if resp.status_code in RETRY_CODES:
                            if attempt < cfg.retry_times:
                                await asyncio.sleep(cfg.retry_backoff * (2 ** attempt))
                                attempt += 1
                                continue
                            nxt = rt.next_fallback(current_alias)
                            if not (in_text or in_tool or in_think or idx > 0) and nxt:
                                rt.mark_down(current_alias)
                                current_alias = nxt
                                attempt = 0
                                log.warning("[%s] [failover] %s retries exhausted -> %s",
                                            req_id, resp.status_code, nxt)
                                continue
                        if resp.status_code != 200:
                            err = (await resp.aread()).decode(errors="replace")[:300]
                            if (resp.status_code == 400 and "context length" in err
                                    and ctx_trims < 8 and shrink_messages(openai_req)):
                                ctx_trims += 1
                                attempt = 0
                                log.info("[ctx-retry] backend 400 context length -> "
                                         "dropped oldest turns, retry #%d", ctx_trims)
                                continue
                            # Surface as visible text, not a silent empty turn.
                            hint = ""
                            if resp.status_code >= 500 or "Connection error" in err:
                                hint = (f" The {current_alias} backend looks temporarily "
                                        f"down; retry in a moment, or switch models.")
                            await emit_text(f"[proxy] upstream {resp.status_code}.{hint}")
                            stop_reason = "end_turn"
                            return

                        async for line in resp.aiter_lines():
                            if not line.startswith("data: "):
                                continue
                            raw = line[6:].strip()
                            if raw == "[DONE]":
                                break
                            try:
                                data = json.loads(raw)
                            except json.JSONDecodeError:
                                continue

                            choices = data.get("choices")
                            if not choices:
                                u = data.get("usage")
                                if u:
                                    # Only a non-zero reading may overwrite the
                                    # local estimate — a zero would re-break
                                    # auto-compact's water-level signal.
                                    if u.get("prompt_tokens"):
                                        usage_in = u["prompt_tokens"]
                                    if u.get("completion_tokens"):
                                        usage_out = u["completion_tokens"]
                                continue

                            delta = choices[0].get("delta", {})
                            finish = choices[0].get("finish_reason")

                            reasoning = delta.get("reasoning_content") or delta.get("reasoning")
                            if reasoning:
                                think_chars += len(reasoning)
                                if model.reasoning != "native" and not in_tool \
                                        and content_len == 0 \
                                        and think_chars > THINK_RUNAWAY_CHARS:
                                    log.warning("[%s] thinking runaway (%d chars, 0 visible); "
                                                "cutting stream for low-reasoning retry",
                                                req_id, think_chars)
                                    break
                                if not in_think:
                                    await close_block()
                                    await queue.put(_sse("content_block_start", {
                                        "type": "content_block_start", "index": idx,
                                        "content_block": {"type": "thinking", "thinking": ""}}))
                                    in_think = True
                                await queue.put(_sse("content_block_delta", {
                                    "type": "content_block_delta", "index": idx,
                                    "delta": {"type": "thinking_delta", "thinking": reasoning}}))
                                # NOTE: no `continue` here — a single delta may carry
                                # the last reasoning AND the first content. Skipping
                                # ahead eats the answer's opening characters.

                            text = delta.get("content")
                            if text:
                                await emit_text(text)

                            for tc in delta.get("tool_calls") or []:
                                if tc.get("id"):
                                    await close_block()
                                    func = tc.get("function", {})
                                    await queue.put(_sse("content_block_start", {
                                        "type": "content_block_start", "index": idx,
                                        "content_block": {"type": "tool_use",
                                                          "id": unique_tool_id(),
                                                          "name": func.get("name", ""),
                                                          "input": {}}}))
                                    in_tool = True
                                args_chunk = (tc.get("function") or {}).get("arguments", "")
                                if args_chunk:
                                    await queue.put(_sse("content_block_delta", {
                                        "type": "content_block_delta", "index": idx,
                                        "delta": {"type": "input_json_delta",
                                                  "partial_json": args_chunk}}))

                            if finish:
                                stop_reason = _stop_of(finish)
                                # Do NOT break: the usage chunk rides AFTER the
                                # finish chunk on some backends.
                                continue
                    break  # success
                except httpx.TransportError as e:
                    have_output = in_text or in_tool or in_think or idx > 0 or content_len
                    if not have_output and attempt < cfg.retry_times:
                        await asyncio.sleep(cfg.retry_backoff * (2 ** attempt))
                        attempt += 1
                        log.warning("[%s] transport %s attempt %d: %r",
                                    req_id, type(e).__name__, attempt, e)
                        continue
                    if not have_output:
                        nxt = rt.next_fallback(current_alias)
                        if nxt:
                            rt.mark_down(current_alias)
                            current_alias = nxt
                            attempt = 0
                            log.warning("[%s] [failover] transport exhausted -> %s",
                                        req_id, nxt)
                            continue
                    log.error("[%s] transport error (output started=%s): %r",
                              req_id, bool(have_output), e)
                    await close_block()
                    note = TRANSPORT_CUT_NOTE if have_output else \
                        TRANSPORT_DEAD_NOTE.format(kind=type(e).__name__)
                    await emit_text(note)
                    stop_reason = "end_turn"
                    break

            # Empty reply after reasoning burned the budget: retry once with
            # lowered reasoning, then fall back to a visible placeholder.
            if stop_reason in ("length", "end_turn") and content_len == 0 \
                    and not in_tool:
                log.warning("[%s] empty stream (finish=%s, think_chars=%d, %.1fs); "
                            "low-reasoning retry",
                            req_id, stop_reason, think_chars,
                            asyncio.get_event_loop().time() - t_start)
                try:
                    retry_req = dict(openai_req)
                    retry_req["_low_reasoning"] = True
                    retry = await collect_json(retry_req, current_alias, rt)
                    for block in retry.get("content", []):
                        if block.get("type") == "text":
                            await emit_text(block.get("text", ""))
                        elif block.get("type") == "tool_use":
                            await close_block()
                            await queue.put(_sse("content_block_start", {
                                "type": "content_block_start", "index": idx,
                                "content_block": {"type": "tool_use",
                                                  "id": block.get("id", ""),
                                                  "name": block.get("name", ""),
                                                  "input": {}}}))
                            in_tool = True
                            await queue.put(_sse("content_block_delta", {
                                "type": "content_block_delta", "index": idx,
                                "delta": {"type": "input_json_delta",
                                          "partial_json": json.dumps(block.get("input", {}))}}))
                        content_len += len(block.get("text", "") or "")
                    stop_reason = retry.get("stop_reason", "end_turn")
                    ru = retry.get("usage") or {}
                    if ru.get("input_tokens"):
                        usage_in = ru["input_tokens"]
                    if ru.get("output_tokens"):
                        usage_out = ru["output_tokens"]
                except Exception as e:  # noqa: BLE001 - never let retry kill the stream
                    log.error("[%s] low-reasoning retry failed: %r", req_id, e)
                if content_len == 0 and not in_tool:
                    log.warning("[%s] retry also empty; emitting visible fallback", req_id)
                    await emit_text(EMPTY_RETRY_NOTE)
                    stop_reason = "end_turn"

        except Exception:  # noqa: BLE001
            log.exception("[%s] stream error", req_id)
        finally:
            elapsed = asyncio.get_event_loop().time() - t_start
            log.info("[%s] stream done | stop=%s | chars=%d | %.1fs",
                     req_id, stop_reason, content_len, elapsed)
            await close_block()
            await queue.put(_sse("message_delta", {
                "type": "message_delta",
                "delta": {"stop_reason": stop_reason, "stop_sequence": None},
                "usage": {"input_tokens": usage_in, "output_tokens": usage_out}}))
            await queue.put(_sse("message_stop", {"type": "message_stop"}))
            await queue.put(_DONE)

    ka_task = asyncio.create_task(_keepalive())
    prod_task = asyncio.create_task(_produce())

    yield _sse("message_start", {
        "type": "message_start",
        "message": {"id": msg_id, "type": "message", "role": "assistant",
                    "content": [], "model": alias, "stop_reason": None,
                    "stop_sequence": None,
                    "usage": {"input_tokens": est_usage_in, "output_tokens": 0}}})

    try:
        while True:
            chunk = await queue.get()
            if chunk is _DONE:
                break
            yield chunk
    finally:
        stop_event.set()
        ka_task.cancel()
        prod_task.cancel()


async def collect_json(openai_req: dict, alias: str, rt) -> dict:
    """One /v1/messages-shaped completion, returned as an Anthropic JSON body.

    Retries transport errors and 429/5xx, then walks the fallback chain.
    Raises UpstreamError when nothing works.
    """
    cfg = rt.cfg
    current_alias = alias
    low_reasoning = bool(openai_req.pop("_low_reasoning", False))
    ctx_trims = 0
    attempt = 0
    last_err = None
    while True:
        model = cfg.models[current_alias]
        req = dict(openai_req)
        req["model"] = model.upstream_name()
        req["stream"] = False
        if low_reasoning and model.reasoning == "native":
            req.pop("reasoning_effort", None)
        try:
            resp = await rt.backend_post(model, req)
        except httpx.TransportError as e:
            last_err = UpstreamError(502, f"transport: {type(e).__name__}")
            if attempt < cfg.retry_times:
                attempt += 1
                await asyncio.sleep(cfg.retry_backoff * (2 ** (attempt - 1)))
                continue
            nxt = rt.next_fallback(current_alias)
            if nxt:
                rt.mark_down(current_alias)
                current_alias, attempt = nxt, 0
                continue
            raise last_err
        if resp.status_code in RETRY_CODES:
            last_err = UpstreamError(resp.status_code, resp.text[:300])
            if attempt < cfg.retry_times:
                attempt += 1
                await asyncio.sleep(cfg.retry_backoff * (2 ** (attempt - 1)))
                continue
            nxt = rt.next_fallback(current_alias)
            if nxt:
                rt.mark_down(current_alias)
                current_alias, attempt = nxt, 0
                continue
            raise last_err
        if resp.status_code == 400 and "context length" in resp.text \
                and ctx_trims < 8 and shrink_messages(openai_req):
            ctx_trims += 1
            attempt = 0
            continue
        if resp.status_code != 200:
            raise UpstreamError(resp.status_code, resp.text[:300])

        data = resp.json()
        choice = (data.get("choices") or [{}])[0]
        msg = choice.get("message", {}) or {}
        content = []
        reasoning = msg.get("reasoning_content")
        if reasoning:
            content.append({"type": "thinking", "thinking": reasoning})
        if msg.get("content"):
            content.append({"type": "text", "text": msg["content"]})
        stop_reason = _stop_of(choice.get("finish_reason") or "stop")
        for tc in msg.get("tool_calls") or []:
            func = tc.get("function", {})
            args = func.get("arguments", "{}")
            try:
                parsed = json.loads(args) if isinstance(args, str) else (args or {})
            except json.JSONDecodeError:
                parsed = {}
            content.append({"type": "tool_use", "id": unique_tool_id(),
                            "name": func.get("name", ""), "input": parsed})
            stop_reason = "tool_use"
        usage = data.get("usage") or {}
        return {
            "id": "msg_" + os.urandom(8).hex(),
            "type": "message",
            "role": "assistant",
            "model": current_alias,
            "content": content,
            "stop_reason": stop_reason,
            "stop_sequence": None,
            "usage": {"input_tokens": usage.get("prompt_tokens") or est_request_tokens(openai_req),
                      "output_tokens": usage.get("completion_tokens") or 0},
        }
