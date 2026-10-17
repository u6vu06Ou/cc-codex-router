"""Responses API (/v1/responses): translation, event order, failover."""
import json

import httpx
from starlette.requests import Request

from cc_codex_router.app import Runtime, create_app, responses_endpoint
from cc_codex_router.responses import (DEFAULT_MIN_OUTPUT,
                                          NATIVE_REASONING_MIN_OUTPUT,
                                          responses_to_chat)

from conftest import (chunk_finish, chunk_text, chunk_toolcall, chunk_usage,
                      make_config, parse_sse, run)


def make_request(app, body, path="/v1/responses"):
    sent = {"done": False}

    async def receive():
        if not sent["done"]:
            sent["done"] = True
            return {"type": "http.request", "body": body, "more_body": False}
        return {"type": "http.disconnect"}

    scope = {"type": "http", "method": "POST", "path": path, "app": app,
             "headers": [], "query_string": b""}
    return Request(scope, receive)


def app_with(upstream, **model_over):
    cfg = make_config(upstream, **model_over)
    app = create_app(cfg)
    app.state.rt = Runtime(cfg, httpx.AsyncClient(timeout=httpx.Timeout(10.0)))
    return app


def call(endpoint, req):
    async def go():
        resp = await endpoint(req)
        if hasattr(resp, "body_iterator"):
            parts = [c async for c in resp.body_iterator]
            body = b"".join(p if isinstance(p, bytes) else p.encode()
                            for p in parts)
        else:
            body = resp.body
        return resp.status_code, body
    return run(go())


def responses_req(model="main", text="hi", stream=True, **extra):
    body = {"model": model, "stream": stream,
            "input": [{"type": "message", "role": "user",
                       "content": [{"type": "input_text", "text": text}]}]}
    body.update(extra)
    return json.dumps(body).encode()


def event_types(events):
    return [ev for ev, _ in events]


# ── request translation ─────────────────────────────────────────────────────

def test_input_string_and_instructions_become_leading_system(upstream):
    model = make_config(upstream).models["main"]
    chat = responses_to_chat(
        {"model": "main", "instructions": "Be brief.",
         "input": "hello"}, model)
    assert chat["messages"][0] == {"role": "system", "content": "Be brief."}
    assert chat["messages"][1] == {"role": "user", "content": "hello"}


def test_developer_messages_hoisted_into_one_system(upstream):
    model = make_config(upstream).models["main"]
    chat = responses_to_chat(
        {"model": "main",
         "instructions": "Rules.",
         "input": [
             {"type": "message", "role": "user",
              "content": [{"type": "input_text", "text": "q1"}]},
             {"type": "message", "role": "developer",
              "content": [{"type": "input_text", "text": "More rules."}]},
             {"type": "message", "role": "user",
              "content": [{"type": "input_text", "text": "q2"}]},
         ]}, model)
    assert chat["messages"][0] == {"role": "system",
                                   "content": "Rules.\n\nMore rules."}
    roles = [m["role"] for m in chat["messages"]]
    assert roles == ["system", "user", "user"]


def test_tool_call_history_sanitizes_raw_string_arguments(upstream):
    model = make_config(upstream).models["main"]
    chat = responses_to_chat(
        {"model": "main",
         "input": [
             {"type": "function_call", "call_id": "c1", "name": "shell",
              "arguments": '{"cmd": "ls -la</think><tool_call>'},
             {"type": "function_call", "call_id": "c2", "name": "shell",
              "arguments": '"unterminated'},
             {"type": "function_call_output", "call_id": None, "id": None,
              "output": "ok"},
         ]}, model)
    calls = [tc for m in chat["messages"] if m.get("tool_calls")
             for tc in m["tool_calls"]]
    args = [tc["function"]["arguments"] for tc in calls]
    assert json.loads(args[0]) == {"cmd": "ls -la"}   # leak cut + JSON closed
    assert json.loads(args[1]) == {}                  # unfixable -> {}
    tools = [m for m in chat["messages"] if m["role"] == "tool"]
    assert tools[0]["tool_call_id"] == ""   # None ids fall through, not "None"


def test_tool_choice_dropped_when_all_tools_filtered(upstream):
    model = make_config(upstream).models["main"]
    chat = responses_to_chat(
        {"model": "main", "input": "hi",
         "tools": [{"type": "web_search"}],
         "tool_choice": "auto"}, model)
    assert "tools" not in chat
    assert "tool_choice" not in chat


def test_max_tokens_floor_by_reasoning_kind(upstream):
    native = make_config(upstream).models["main"]           # reasoning native
    plain = make_config(upstream, reasoning="none").models["main"]
    assert responses_to_chat({"model": "main", "input": "hi"},
                             native)["max_tokens"] == NATIVE_REASONING_MIN_OUTPUT
    assert responses_to_chat({"model": "main", "input": "hi"},
                             plain)["max_tokens"] == DEFAULT_MIN_OUTPUT
    bigger = responses_to_chat({"model": "main", "input": "hi",
                                "max_output_tokens": 99_999}, plain)
    assert bigger["max_tokens"] == 99_999


def test_reasoning_history_grouped_for_native_models(upstream):
    model = make_config(upstream).models["main"]
    chat = responses_to_chat(
        {"model": "main",
         "input": [
             {"type": "reasoning",
              "summary": [{"type": "summary_text", "text": "thoughts"}]},
             {"type": "function_call", "call_id": "c1", "name": "ls",
              "arguments": "{}"},
         ]}, model)
    assistant = chat["messages"][0]
    assert assistant["role"] == "assistant"
    assert assistant["reasoning_content"] == "thoughts"
    assert assistant["tool_calls"][0]["function"]["name"] == "ls"
    assert chat["reasoning_effort"] == "high"


# ── endpoint: happy paths ───────────────────────────────────────────────────

def test_non_streaming_text_reply(upstream):
    app = app_with(upstream)
    upstream.push_json({"choices": [{"index": 0, "finish_reason": "stop",
                                     "message": {"content": "hello there"}}],
                        "usage": {"prompt_tokens": 5, "completion_tokens": 2,
                                  "completion_tokens_details":
                                  {"reasoning_tokens": 1}}})
    req = make_request(app, responses_req(stream=False))
    status, body = call(responses_endpoint, req)
    assert status == 200
    resp = json.loads(body)
    assert resp["object"] == "response"
    assert resp["status"] == "completed"
    assert resp["output"][0]["type"] == "message"
    assert resp["output"][0]["content"][0]["text"] == "hello there"
    assert resp["usage"]["input_tokens"] == 5
    assert resp["usage"]["output_tokens"] == 2
    assert resp["usage"]["output_tokens_details"] == {"reasoning_tokens": 1}
    seen = upstream.requests[0]["json"]
    assert seen["model"] == "up-main"
    assert seen["stream"] is False


def test_streaming_event_order(upstream):
    app = app_with(upstream)
    upstream.push_sse([chunk_text("par"), chunk_text("tial"),
                       chunk_finish("stop"), chunk_usage(7, 3)])
    req = make_request(app, responses_req())
    status, body = call(responses_endpoint, req)
    assert status == 200
    events = parse_sse(body.decode())
    types = event_types(events)
    assert types[:4] == ["response.created", "response.in_progress",
                         "response.output_item.added",
                         "response.content_part.added"]
    deltas = [d["delta"] for ev, d in events
              if ev == "response.output_text.delta"]
    assert "".join(deltas) == "partial"
    assert types[-4:] == ["response.output_text.done",
                          "response.content_part.done",
                          "response.output_item.done",
                          "response.completed"]
    final = [d for ev, d in events if ev == "response.completed"][0]
    assert final["response"]["status"] == "completed"
    assert final["response"]["usage"] == {"input_tokens": 7,
                                          "output_tokens": 3,
                                          "total_tokens": 10}
    seqs = [d["sequence_number"] for _, d in events]
    assert seqs == list(range(len(seqs)))

def test_streaming_reasoning_then_text(upstream):
    app = app_with(upstream)
    upstream.push_sse([
        {"choices": [{"index": 0,
                      "delta": {"reasoning_content": "thinking"}}]},
        chunk_text("answer"), chunk_finish("stop"),
    ])
    req = make_request(app, responses_req())
    _, body = call(responses_endpoint, req)
    events = parse_sse(body.decode())
    types = event_types(events)
    assert "response.reasoning_text.delta" in types
    text_at = types.index("response.output_item.added", types.index(
        "response.reasoning_text.delta"))
    assert text_at > 0
    final = [d for ev, d in events if ev == "response.completed"][0]
    kinds = [item["type"] for item in final["response"]["output"]]
    assert kinds == ["reasoning", "message"]


def test_length_finish_reports_incomplete(upstream):
    app = app_with(upstream)
    upstream.push_sse([chunk_text("half"), chunk_finish("length")])
    req = make_request(app, responses_req())
    _, body = call(responses_endpoint, req)
    events = parse_sse(body.decode())
    final = [d for ev, d in events if ev == "response.completed"][0]
    assert final["response"]["status"] == "incomplete"
    assert final["response"]["incomplete_details"] == {
        "reason": "max_output_tokens"}


def test_unknown_model_404_lists_configured(upstream):
    app = app_with(upstream)
    req = make_request(app, responses_req(model="nope"))
    status, body = call(responses_endpoint, req)
    assert status == 404
    err = json.loads(body)["error"]
    assert err["code"] == "model_not_found"
    assert "main" in err["message"] and "spare" in err["message"]


# ── tool call round trip ────────────────────────────────────────────────────

SHELL_TOOL = {"type": "function", "name": "shell",
              "description": "run a command",
              "parameters": {"type": "object",
                             "properties": {"cmd": {"type": "string"}}}}


def test_tool_call_streams_then_round_trip(upstream):
    app = app_with(upstream)
    upstream.push_sse([
        chunk_toolcall("call_1", "shell", '{"cmd":'),
        chunk_toolcall(None, None, '"ls"}'),
        chunk_finish("tool_calls"),
        chunk_usage(11, 4),
    ])
    req = make_request(app, responses_req(tools=[SHELL_TOOL]))
    _, body = call(responses_endpoint, req)
    events = parse_sse(body.decode())
    added = [d for ev, d in events if ev == "response.output_item.added"
             and d["item"]["type"] == "function_call"][0]
    assert added["item"]["name"] == "shell"
    assert added["item"]["call_id"] == "call_1"
    deltas = [d["delta"] for ev, d in events
              if ev == "response.function_call_arguments.delta"]
    assert "".join(deltas) == '{"cmd":"ls"}'
    done = [d for ev, d in events
            if ev == "response.function_call_arguments.done"][0]
    assert json.loads(done["arguments"]) == {"cmd": "ls"}
    final = [d for ev, d in events if ev == "response.completed"][0]
    assert final["response"]["status"] == "completed"
    assert final["response"]["usage"]["output_tokens"] == 4
    # upstream got the tool schema in chat-completions shape
    sent_tools = upstream.requests[-1]["json"]["tools"]
    assert sent_tools[0]["function"]["name"] == "shell"

    # turn 2: the client sends the call + its output back as history
    upstream.push_sse([chunk_text("3 files"), chunk_finish("stop")])
    turn2 = {"model": "main", "stream": True, "tools": [SHELL_TOOL],
             "input": [
                 {"type": "message", "role": "user",
                  "content": [{"type": "input_text", "text": "list files"}]},
                 {"type": "function_call", "call_id": "call_1",
                  "name": "shell", "arguments": '{"cmd":"ls"}'},
                 {"type": "function_call_output", "call_id": "call_1",
                  "output": "a\nb\nc"},
                 {"type": "message", "role": "user",
                  "content": [{"type": "input_text", "text": "how many?"}]},
             ]}
    _, body2 = call(responses_endpoint,
                    make_request(app, json.dumps(turn2).encode()))
    events2 = parse_sse(body2.decode())
    texts = [d["delta"] for ev, d in events2
             if ev == "response.output_text.delta"]
    assert "".join(texts) == "3 files"
    sent = upstream.requests[-1]["json"]["messages"]
    assistant = [m for m in sent if m.get("tool_calls")][0]
    assert assistant["tool_calls"][0]["id"] == "call_1"
    tool_msg = [m for m in sent if m["role"] == "tool"][0]
    assert tool_msg["tool_call_id"] == "call_1"
    assert tool_msg["content"] == "a\nb\nc"


def test_namespaced_tools_flatten_out_and_back(upstream):
    app = app_with(upstream)
    upstream.push_sse([
        chunk_toolcall("call_9", "agent__spawn", "{}"),
        chunk_finish("tool_calls"),
    ])
    ns_tool = {"type": "namespace", "name": "agent",
               "tools": [{"type": "function", "name": "spawn",
                          "description": "spawn a subagent",
                          "parameters": {"type": "object"}}]}
    req = make_request(app, responses_req(tools=[ns_tool]))
    _, body = call(responses_endpoint, req)
    events = parse_sse(body.decode())
    sent_names = [t["function"]["name"]
                  for t in upstream.requests[-1]["json"]["tools"]]
    assert sent_names == ["agent__spawn"]          # flat on the wire
    fc = [d["item"] for ev, d in events
          if ev == "response.output_item.added"
          and d["item"]["type"] == "function_call"][0]
    assert fc["name"] == "spawn"                    # split back for the client
    assert fc["namespace"] == "agent"


# ── resilience ──────────────────────────────────────────────────────────────

def test_streaming_failover_after_retries_exhausted(upstream):
    app = app_with(upstream)
    for _ in range(3):                      # initial try + 2 retries on main
        upstream.push_text(500, "blip")
    upstream.push_sse([chunk_text("spare says hi"), chunk_finish("stop")])
    req = make_request(app, responses_req())
    status, body = call(responses_endpoint, req)
    assert status == 200
    events = parse_sse(body.decode())
    assert [ev for ev, _ in events][-1] == "response.completed"
    models_hit = [r["json"].get("model") for r in upstream.requests]
    assert models_hit == ["up-main"] * 3 + ["up-spare"]


def test_transport_cut_before_output_fails_cleanly(upstream):
    app = app_with(upstream, fallbacks=[])
    for _ in range(3):
        upstream.push_close()
    req = make_request(app, responses_req())
    status, body = call(responses_endpoint, req)
    assert status == 200   # SSE channel stays up; the failure is an event
    events = parse_sse(body.decode())
    failed = [d for ev, d in events if ev == "response.failed"]
    assert failed and failed[0]["response"]["status"] == "failed"
    assert "transport" in failed[0]["response"]["error"]["message"]


def test_upstream_error_status_surfaces_as_failed_event(upstream):
    app = app_with(upstream, fallbacks=[])
    upstream.push_text(400, "bad request: nope")
    req = make_request(app, responses_req())
    _, body = call(responses_endpoint, req)
    events = parse_sse(body.decode())
    failed = [d for ev, d in events if ev == "response.failed"]
    assert failed and "400" in failed[0]["response"]["error"]["message"]
