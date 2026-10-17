"""OpenAI Responses API (/v1/responses) <-> chat/completions, for Codex CLI.

Codex speaks the Responses dialect: input items (message / function_call /
function_call_output / reasoning), `instructions`, flat or namespaced tools,
and a streaming event sequence (response.created, output_item.added,
output_text.delta, function_call_arguments.delta/done, output_item.done,
response.completed). Backends here only speak chat-completions, so this
module translates in both directions and replays the hard-won production
fixes:

- function_call arguments arrive as RAW STRINGS and may be broken JSON (a
  truncated string, or chat-template markers leaked mid-arguments). Codex
  stores them verbatim and re-sends them every turn, so one broken parse
  poisons the session forever. Everything from history passes through
  sanitize_tool_arguments before it reaches an upstream tokenizer.
- function_call_output with {call_id: null, id: null} still yields "" for
  the tool_call_id (dict.get returns None, not the default).
- developer/system messages that appear mid-conversation are collected and
  prepended as ONE leading system message (many backends 400 on a late
  system role).
- reasoning models burn the whole output budget on thinking when max_tokens
  is small: native-reasoning models floor at 32k output tokens, everyone
  else at 8k (the same floor the Messages path uses).
- tool_choice is only forwarded when tools are actually present (codex
  sometimes sends tool_choice on turns whose tools were all filtered out,
  and backends then 400 with "tools must be set").
- streaming only retries / fails over while nothing user-visible has been
  emitted; afterwards the connection is honestly failed so the client can
  retry the turn itself.
- an upstream finish_reason of "length" closes with status "incomplete",
  never "completed" — otherwise the client parses a truncated patch or
  truncated tool arguments as if they were whole.
- usage.details (reasoning_tokens / cached_tokens) pass through so the
  client's context water-level accounting stays real.

Tool types other than function/namespace (e.g. hosted web_search) are
skipped with a warning rather than hard-failing the whole request.
"""
from __future__ import annotations

import asyncio
import json
import logging
import secrets
import time

import httpx

from .clamp import shrink_messages
from .config import ModelCfg
from .sanitize import sanitize_tool_arguments
from .stream import RETRY_CODES, UpstreamError


log = logging.getLogger("claude_team_router.responses")

# A small output cap + a reasoning model = the answer never arrives: the
# thinking eats the entire budget first. Native-reasoning models therefore
# floor at 32k; everyone else gets the Messages path's 8k floor.
NATIVE_REASONING_MIN_OUTPUT = 32_768
DEFAULT_MIN_OUTPUT = 8_192

EMPTY_RETRY_NOTE = ("[router] The backend produced no visible answer twice "
                    "in a row; nothing to show for this turn. Please re-issue "
                    "the instruction, or try a different model.")


def _rid(prefix: str) -> str:
    return f"{prefix}_{secrets.token_hex(8)}"


# ── request translation: Responses -> chat/completions ──────────────────────

def _responses_text(content) -> str:
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return ""
    return "\n".join(
        item.get("text", "")
        for item in content
        if isinstance(item, dict)
        and item.get("type") in {"input_text", "output_text", "text"}
    )


def _responses_reasoning_text(item: dict) -> str:
    """Recover reasoning text our own adapter (or the client) stored in an item."""
    for key in ("content", "summary"):
        value = item.get(key)
        if isinstance(value, str) and value:
            return value
        if isinstance(value, list):
            text = "\n".join(
                part.get("text", "")
                for part in value
                if isinstance(part, dict)
                and part.get("type") in {"reasoning_text", "summary_text", "text"}
            )
            if text:
                return text
    return ""


def _requested_reasoning_effort(body: dict) -> str:
    effort = body.get("reasoning_effort")
    if not effort and isinstance(body.get("reasoning"), dict):
        effort = body["reasoning"].get("effort")
    return str(effort or "")


def responses_to_chat(body: dict, model: ModelCfg) -> dict:
    """One Responses request -> one OpenAI chat/completions request.

    Raises ValueError on anything the caller should surface as a 400.
    """
    messages = []
    system_parts = []
    preserve_reasoning = model.reasoning == "native"
    pending_reasoning = ""
    assistant_group = None
    if instructions := _responses_text(body.get("instructions")):
        system_parts.append(instructions)

    input_items = body.get("input", [])
    if isinstance(input_items, str):
        input_items = [{"type": "message", "role": "user",
                        "content": input_items}]
    for item in input_items:
        if not isinstance(item, dict):
            continue
        item_type = item.get("type", "message")
        if item_type == "reasoning":
            if preserve_reasoning:
                pending_reasoning += _responses_reasoning_text(item)
            continue
        if item_type == "message":
            role = "system" if item.get("role") == "developer" \
                else item.get("role", "user")
            content = _responses_text(item.get("content"))
            # Many chat-completions backends reject any system message after
            # normal conversation turns. Codex emits later developer messages
            # during multi-turn tool loops, so collect all system-like content
            # and prepend it as one leading system message.
            if role == "system":
                if content:
                    system_parts.append(content)
                assistant_group = None
            else:
                msg_obj = {"role": role, "content": content}
                if role == "assistant" and preserve_reasoning:
                    if pending_reasoning:
                        msg_obj["reasoning_content"] = pending_reasoning
                        pending_reasoning = ""
                    assistant_group = msg_obj
                else:
                    assistant_group = None
                messages.append(msg_obj)
        elif item_type == "function_call":
            call_id = item.get("call_id") or item.get("id") \
                or _rid("call")
            name = item.get("name", "")
            if item.get("namespace"):
                name = f"{item['namespace']}__{name}"
            tool_call = {
                "id": call_id,
                "type": "function",
                "function": {
                    "name": name,
                    # raw-string arguments are the session poison: never
                    # forward them without the sanitizer.
                    "arguments": sanitize_tool_arguments(
                        item.get("arguments"), name),
                },
            }
            if preserve_reasoning:
                if assistant_group is None:
                    assistant_group = {"role": "assistant", "content": "",
                                       "tool_calls": []}
                    if pending_reasoning:
                        assistant_group["reasoning_content"] = pending_reasoning
                        pending_reasoning = ""
                    messages.append(assistant_group)
                assistant_group.setdefault("tool_calls", []).append(tool_call)
            else:
                messages.append({"role": "assistant", "content": "",
                                 "tool_calls": [tool_call]})
        elif item_type == "function_call_output":
            output = item.get("output", "")
            if not isinstance(output, str):
                output = json.dumps(output, ensure_ascii=False)
            messages.append({
                "role": "tool",
                # {call_id: null, id: null} must still give "" — dict.get
                # returns None, not the default, when the key exists.
                "tool_call_id": item.get("call_id") or item.get("id") or "",
                "content": output,
            })
            assistant_group = None

    if preserve_reasoning and pending_reasoning:
        messages.append({"role": "assistant", "content": "",
                         "reasoning_content": pending_reasoning})

    tools = []
    tool_names = {}   # flat "namespace__name" -> [namespace, name]
    for tool in body.get("tools") or []:
        if tool.get("type") in {"web_search", "web_search_preview",
                                "image_generation"}:
            continue  # hosted tools no chat-completions backend can serve
        ttype = tool.get("type")
        if ttype == "function":
            nested_tools = [(None, tool)]
        elif ttype == "namespace":
            nested_tools = [(tool.get("name", ""), nested)
                            for nested in tool.get("tools", [])]
        else:
            log.warning("[responses] skipping unsupported tool type %r", ttype)
            continue
        for namespace, nested in nested_tools:
            if nested.get("type") != "function":
                log.warning("[responses] skipping unsupported namespace tool %r",
                            nested.get("type"))
                continue
            name = nested.get("name", "")
            flat_name = f"{namespace}__{name}" if namespace else name
            if namespace:
                tool_names[flat_name] = [namespace, name]
            tools.append({
                "type": "function",
                "function": {
                    "name": flat_name,
                    "description": nested.get(
                        "description", tool.get("description", "")),
                    "parameters": nested.get("parameters", {}),
                },
            })

    if system_parts:
        messages.insert(0, {"role": "system",
                            "content": "\n\n".join(system_parts)})

    asked = body.get("max_output_tokens")
    try:
        asked = int(asked) if asked else 0
    except (TypeError, ValueError):
        asked = 0
    floor = NATIVE_REASONING_MIN_OUTPUT if preserve_reasoning \
        else DEFAULT_MIN_OUTPUT
    max_tokens = max(asked, floor)
    if model.max_output:
        max_tokens = min(max_tokens, model.max_output)

    request = {
        "model": model.upstream_name(),
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": body.get("temperature",
                                1.0 if preserve_reasoning else 0.0),
    }
    if preserve_reasoning:
        effort = _requested_reasoning_effort(body) \
            or model.reasoning_effort or "high"
        request["reasoning_effort"] = effort
        if "top_p" in body:
            request["top_p"] = body["top_p"]
    if tools:
        request["tools"] = tools
    # tool_choice only rides when tools exist: a bare tool_choice 400s on
    # backends with "When using tool_choice, tools must be set".
    if body.get("tool_choice") is not None and tools:
        request["tool_choice"] = body["tool_choice"]
    if body.get("parallel_tool_calls") is not None:
        request["parallel_tool_calls"] = body["parallel_tool_calls"]
    if tool_names:
        request["_responses_tool_names"] = tool_names
    return request


# ── response translation: chat/completions chunks -> Responses SSE events ───

def _event(state: dict, event_type: str, **payload) -> dict:
    event = {"type": event_type,
             "sequence_number": state.setdefault("sequence", 0), **payload}
    state["sequence"] += 1
    return event


def response_snapshot(state: dict, status: str) -> dict:
    usage = state.get("usage", {})
    input_tokens = usage.get("prompt_tokens", 0)
    output_tokens = usage.get("completion_tokens", 0)
    usage_out = {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": input_tokens + output_tokens,
    }
    completion_details = usage.get("completion_tokens_details") or {}
    if (reasoning := completion_details.get("reasoning_tokens")) is not None:
        usage_out["output_tokens_details"] = {"reasoning_tokens": reasoning}
    prompt_details = usage.get("prompt_tokens_details") or {}
    if (cached := prompt_details.get("cached_tokens")) is not None:
        usage_out["input_tokens_details"] = {"cached_tokens": cached}
    return {
        "id": state["response_id"],
        "object": "response",
        "created_at": state["created_at"],
        "status": status,
        "model": state.get("model", "unknown"),
        "output": state.setdefault("output", []),
        "usage": usage_out,
        "error": state.get("error"),
        "incomplete_details": state.get("incomplete_details"),
    }


def _final_status(state: dict) -> str:
    if state.get("error"):
        return "failed"
    if state.get("finish_reason") == "length":
        return "incomplete"
    return "completed"


def start_responses_stream(state: dict) -> list[dict]:
    if state.get("started"):
        return []
    state.update({
        "started": True,
        "response_id": state.get("response_id", _rid("resp")),
        "created_at": state.get("created_at", int(time.time())),
        "output": [],
        "tools": {},
    })
    snapshot = response_snapshot(state, "in_progress")
    return [
        _event(state, "response.created", response=snapshot),
        _event(state, "response.in_progress", response=snapshot),
    ]


def translate_chat_chunk(state: dict, chunk: dict) -> list[dict]:
    """One OpenAI streaming chunk -> zero or more Responses events."""
    events = start_responses_stream(state)
    if chunk.get("usage"):
        state["usage"] = chunk["usage"]
    choices = chunk.get("choices") or []
    if not choices:
        return events
    delta = choices[0].get("delta", {})

    # The upstream finish_reason decides completed vs incomplete at the end.
    if fr := choices[0].get("finish_reason"):
        state["finish_reason"] = fr

    # Reasoning streams first on native-reasoning backends; forward it so the
    # client shows progress during the thinking phase instead of going idle.
    if reasoning_text := (delta.get("reasoning_content")
                          or delta.get("reasoning")):
        reasoning = state.get("reasoning")
        if reasoning is None:
            reasoning = {"id": _rid("rs"), "type": "reasoning",
                         "summary": [], "content": []}
            state["reasoning"] = reasoning
            state["reasoning_text"] = ""
            state["output"].append(reasoning)
            state["reasoning_index"] = len(state["output"]) - 1
            events.append(_event(
                state, "response.output_item.added",
                output_index=state["reasoning_index"], item=reasoning))
        state["reasoning_text"] += reasoning_text
        events.append(_event(
            state, "response.reasoning_text.delta", item_id=reasoning["id"],
            output_index=state["reasoning_index"], content_index=0,
            delta=reasoning_text))

    if text := delta.get("content"):
        if "message" not in state:
            item = {"id": _rid("msg"), "type": "message",
                    "status": "in_progress", "role": "assistant",
                    "content": [{"type": "output_text", "text": "",
                                 "annotations": []}]}
            state["message"] = item
            state["output"].append(item)
            index = len(state["output"]) - 1
            state["message_index"] = index
            events += [
                _event(state, "response.output_item.added",
                       output_index=index, item=item),
                _event(state, "response.content_part.added",
                       item_id=item["id"], output_index=index,
                       content_index=0, part=item["content"][0]),
            ]
        state["message"]["content"][0]["text"] += text
        events.append(_event(
            state, "response.output_text.delta",
            item_id=state["message"]["id"],
            output_index=state["message_index"], content_index=0, delta=text))

    for tool_delta in delta.get("tool_calls") or []:
        tool_index = tool_delta.get("index", 0)
        function = tool_delta.get("function") or {}
        tool = state["tools"].get(tool_index)
        if tool is None:
            call_id = tool_delta.get("id") or _rid("call")
            flat_name = function.get("name", "")
            namespace, name = state.get("tool_names", {}).get(
                flat_name, [None, flat_name])
            tool = {"id": _rid("fc"), "type": "function_call",
                    "status": "in_progress", "call_id": call_id,
                    "name": name, "arguments": ""}
            if namespace:
                tool["namespace"] = namespace
            state["tools"][tool_index] = tool
            state["output"].append(tool)
            tool["output_index"] = len(state["output"]) - 1
            events.append(_event(
                state, "response.output_item.added",
                output_index=tool["output_index"], item=tool))
        if function.get("name"):
            namespace, name = state.get("tool_names", {}).get(
                function["name"], [None, function["name"]])
            tool["name"] = name
            if namespace:
                tool["namespace"] = namespace
        if arguments := function.get("arguments"):
            tool["arguments"] += arguments
            events.append(_event(
                state, "response.function_call_arguments.delta",
                item_id=tool["id"], output_index=tool["output_index"],
                delta=arguments))
    return events


def finish_responses_stream(state: dict, error: str | None = None) -> list[dict]:
    """Close every open item and emit the terminal event."""
    events = start_responses_stream(state)
    if reasoning := state.get("reasoning"):
        text = state.get("reasoning_text", "")
        reasoning["content"] = [{"type": "reasoning_text", "text": text}]
        events += [
            _event(state, "response.reasoning_text.done",
                   item_id=reasoning["id"],
                   output_index=state["reasoning_index"], content_index=0,
                   text=text),
            _event(state, "response.output_item.done",
                   output_index=state["reasoning_index"], item=reasoning),
        ]
    if message := state.get("message"):
        message["status"] = "completed"
        text = message["content"][0]["text"]
        events += [
            _event(state, "response.output_text.done", item_id=message["id"],
                   output_index=state["message_index"], content_index=0,
                   text=text),
            _event(state, "response.content_part.done", item_id=message["id"],
                   output_index=state["message_index"], content_index=0,
                   part=message["content"][0]),
            _event(state, "response.output_item.done",
                   output_index=state["message_index"], item=message),
        ]
    for tool in state.get("tools", {}).values():
        tool["status"] = "completed"
        events += [
            _event(state, "response.function_call_arguments.done",
                   item_id=tool["id"], output_index=tool["output_index"],
                   arguments=tool["arguments"]),
            _event(state, "response.output_item.done",
                   output_index=tool["output_index"], item=tool),
        ]
        tool.pop("output_index", None)
    if error:
        state["error"] = {"code": "upstream_error", "message": error}
    elif state.get("finish_reason") == "length":
        # A truncated answer must say "incomplete" — as "completed" the
        # client would parse half a patch or half the tool arguments.
        state["incomplete_details"] = {"reason": "max_output_tokens"}
    state["final"] = response_snapshot(state, _final_status(state))
    terminal = "response.failed" if error else "response.completed"
    return events + [_event(state, terminal, response=state["final"])]


def chat_result_to_response(data: dict, alias: str,
                            tool_names: dict) -> dict:
    """A non-streaming chat/completions JSON body -> a completed Response."""
    choice = (data.get("choices") or [{}])[0]
    msg = choice.get("message") or {}
    delta = {}
    if msg.get("reasoning_content"):
        delta["reasoning_content"] = msg["reasoning_content"]
    if msg.get("content"):
        delta["content"] = msg["content"]
    if msg.get("tool_calls"):
        delta["tool_calls"] = msg["tool_calls"]
    chunk = {"choices": [{"index": 0, "delta": delta,
                          "finish_reason": choice.get("finish_reason")}],
             "usage": data.get("usage")}
    state = {"model": alias, "tool_names": tool_names or {}}
    start_responses_stream(state)
    translate_chat_chunk(state, chunk)
    finish_responses_stream(state)
    return state["final"]


# ── transport: retries, failover, reactive context trim ─────────────────────

def _ready_for_attempt(openai_req: dict, model: ModelCfg) -> dict:
    req = dict(openai_req)
    req["model"] = model.upstream_name()
    return req


async def relay_responses_stream(openai_req: dict, alias: str, rt,
                                 req_id: str = ""):
    """Yield Responses SSE frames for one streaming /v1/responses request.

    Same hardening policy as the Messages path: backoff on 429/5xx, failover
    down the alias chain and transport reconnects only while nothing
    user-visible has been emitted, reactive context trim on backend 400.
    """
    cfg = rt.cfg
    queue: asyncio.Queue = asyncio.Queue()
    stop_event = asyncio.Event()
    _DONE = object()
    tool_names = openai_req.pop("_responses_tool_names", {})
    state = {"model": alias, "tool_names": tool_names}

    async def _keepalive():
        while not stop_event.is_set():
            try:
                await asyncio.wait_for(stop_event.wait(),
                                       timeout=cfg.keepalive_secs)
            except asyncio.TimeoutError:
                await queue.put(": keepalive\n\n")

    async def _emit(events):
        for event in events:
            await queue.put(f"event: {event['type']}\n"
                            f"data: {json.dumps(event, ensure_ascii=False)}\n\n")

    def _safe_to_resend() -> bool:
        # Reasoning replay is harmless; a re-sent message/tool call is not.
        return not state.get("message") and not state.get("tools")

    async def _produce():
        try:
            await _emit(start_responses_stream(state))
            current_alias = alias
            ctx_trims = 0       # backend-rejected context overflows
            attempt = 0         # backoff counter for 429/5xx + transport
            empty_retries = 0   # reasoning burned the budget, no visible reply
            while attempt < cfg.retry_times + 1:
                model = cfg.models[current_alias]
                req = _ready_for_attempt(openai_req, model)
                req["stream"] = True
                req["stream_options"] = {"include_usage": True}
                try:
                    async with rt.backend_stream(model, req) as resp:
                        if resp.status_code in RETRY_CODES:
                            if attempt < cfg.retry_times:
                                attempt += 1
                                await asyncio.sleep(
                                    cfg.retry_backoff * (2 ** (attempt - 1)))
                                continue
                            nxt = rt.next_fallback(current_alias)
                            if _safe_to_resend() and nxt:
                                rt.mark_down(current_alias)
                                current_alias = nxt
                                attempt = 0
                                log.warning("[%s] [failover] /v1/responses %s "
                                            "retries exhausted -> %s",
                                            req_id, resp.status_code, nxt)
                                continue
                        if resp.status_code != 200:
                            err = (await resp.aread()).decode(
                                errors="replace")[:300]
                            if (resp.status_code == 400
                                    and "context length" in err
                                    and ctx_trims < 8
                                    and shrink_messages(openai_req)):
                                ctx_trims += 1
                                attempt = 0
                                log.info("[%s] [ctx-retry] backend 400 context "
                                         "length -> dropped oldest turns, "
                                         "retry #%d", req_id, ctx_trims)
                                continue
                            await _emit(finish_responses_stream(
                                state, f"upstream {resp.status_code}: {err}"))
                            return
                        async for line in resp.aiter_lines():
                            if not line.startswith("data: "):
                                continue
                            raw = line[6:].strip()
                            if raw == "[DONE]":
                                break
                            try:
                                chunk = json.loads(raw)
                            except json.JSONDecodeError:
                                continue
                            await _emit(translate_chat_chunk(state, chunk))
                        # Empty turn: reasoning burned the budget before any
                        # visible answer. Retry once with the effort pin
                        # removed; if still empty, say so out loud instead of
                        # completing a blank turn.
                        if _safe_to_resend() and empty_retries < 1:
                            empty_retries += 1
                            if cfg.models[current_alias].reasoning == "native":
                                openai_req.pop("reasoning_effort", None)
                            openai_req["max_tokens"] = max(
                                int(openai_req.get("max_tokens") or 0),
                                NATIVE_REASONING_MIN_OUTPUT)
                            attempt = 0
                            log.warning("[%s] empty stream (no message/tool); "
                                        "low-reasoning retry %d/1",
                                        req_id, empty_retries)
                            continue
                        if _safe_to_resend():
                            await _emit(translate_chat_chunk(state, {
                                "choices": [{"index": 0,
                                             "delta": {"content":
                                                       EMPTY_RETRY_NOTE},
                                             "finish_reason": "stop"}]}))
                        await _emit(finish_responses_stream(state))
                        return
                except httpx.TransportError as e:
                    if _safe_to_resend() and attempt < cfg.retry_times:
                        attempt += 1
                        log.warning("[%s] transport %s attempt %d: %r",
                                    req_id, type(e).__name__, attempt, e)
                        await asyncio.sleep(
                            cfg.retry_backoff * (2 ** (attempt - 1)))
                        continue
                    nxt = rt.next_fallback(current_alias)
                    if _safe_to_resend() and nxt:
                        rt.mark_down(current_alias)
                        current_alias = nxt
                        attempt = 0
                        log.warning("[%s] [failover] /v1/responses transport "
                                    "exhausted -> %s", req_id, nxt)
                        continue
                    log.error("[%s] transport error, giving up: %r",
                              req_id, e)
                    await _emit(finish_responses_stream(
                        state, f"upstream transport error: "
                               f"{type(e).__name__}"))
                    return
        except Exception as e:  # noqa: BLE001 - the client must get a terminal event
            log.exception("[%s] responses stream error", req_id)
            try:
                await _emit(finish_responses_stream(state, str(e)))
            except Exception:  # noqa: BLE001
                pass
        finally:
            await queue.put(_DONE)

    ka_task = asyncio.create_task(_keepalive())
    prod_task = asyncio.create_task(_produce())
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


async def collect_responses_json(openai_req: dict, alias: str, rt) -> dict:
    """One non-streaming /v1/responses call, returned as a Response object.

    Retries transport errors and 429/5xx, then walks the fallback chain.
    Raises UpstreamError when nothing works.
    """
    cfg = rt.cfg
    tool_names = openai_req.pop("_responses_tool_names", {})
    current_alias = alias
    ctx_trims = 0
    attempt = 0
    last_err = None
    while True:
        model = cfg.models[current_alias]
        req = _ready_for_attempt(openai_req, model)
        req["stream"] = False
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
        return chat_result_to_response(resp.json(), current_alias, tool_names)
