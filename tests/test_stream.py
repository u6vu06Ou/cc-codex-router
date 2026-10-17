"""Streaming hardening: the kept production fixes, one test each."""
import json

from cc_codex_router.stream import (EMPTY_RETRY_NOTE, TRANSPORT_CUT_NOTE,
                                       collect_json, relay_stream)

from conftest import (basic_anthropic_req, chunk_both, chunk_finish,
                      chunk_reasoning, chunk_text, chunk_toolcall,
                      chunk_usage, collect_stream, make_rt, parse_sse, run)


def text_of(events):
    return "".join(d["delta"]["text"] for ev, d in events
                   if ev == "content_block_delta"
                   and d.get("delta", {}).get("type") == "text_delta")


def test_stream_start_not_eaten(upstream):
    """A delta with reasoning AND content must emit both."""
    rt = make_rt(upstream)
    upstream.push_sse([
        chunk_reasoning("let me think"),
        chunk_both("...so", "ZEPHYR-4417-QUARTZ is the answer"),
        chunk_finish("stop"),
        chunk_usage(1200, 30),
    ])
    raw = collect_stream(relay_stream(
        {"model": "up-main", "stream": True, "max_tokens": 1024,
         "messages": [{"role": "user", "content": "code?"}]},
        "main", rt))
    events = parse_sse(run(raw))
    assert "ZEPHYR-4417-QUARTZ" in text_of(events)  # opening chars intact
    thinking = [d for ev, d in events if ev == "content_block_delta"
                and d.get("delta", {}).get("type") == "thinking_delta"]
    assert thinking  # reasoning still streamed as thinking blocks


def test_usage_chunk_after_finish_is_read(upstream):
    rt = make_rt(upstream)
    upstream.push_sse([
        chunk_text("hi"), chunk_finish("stop"),
        chunk_usage(43210, 22),
    ])
    raw = collect_stream(relay_stream(
        {"model": "up-main", "stream": True, "max_tokens": 1024,
         "messages": [{"role": "user", "content": "hi"}]}, "main", rt))
    events = parse_sse(run(raw))
    md = [d for ev, d in events if ev == "message_delta"][0]
    assert md["usage"]["input_tokens"] == 43210   # real value overwrote estimate
    assert md["usage"]["output_tokens"] == 22


def test_zero_usage_never_overwrites_estimate(upstream):
    rt = make_rt(upstream)
    upstream.push_sse([
        chunk_text("hi"), chunk_finish("stop"),
        chunk_usage(0, 0),
    ])
    raw = collect_stream(relay_stream(
        {"model": "up-main", "stream": True, "max_tokens": 1024,
         "messages": [{"role": "user", "content": "hi"}]}, "main", rt))
    events = parse_sse(run(raw))
    md = [d for ev, d in events if ev == "message_delta"][0]
    assert md["usage"]["input_tokens"] > 0  # estimate kept


def test_unique_tool_ids_minted_on_outgoing(upstream):
    rt = make_rt(upstream)
    upstream.push_sse([
        chunk_toolcall("Read:0", "Read", '{"path":'),
        chunk_toolcall(None, None, '"/tmp/x"}'),
        chunk_toolcall("Read:0", "Read", '{"path":'),   # same upstream id again
        chunk_toolcall(None, None, '"/tmp/y"}'),
        chunk_finish("tool_calls"),
        chunk_usage(100, 20),
    ])
    raw = collect_stream(relay_stream(
        {"model": "up-main", "stream": True, "max_tokens": 1024,
         "messages": [{"role": "user", "content": "read two files"}]},
        "main", rt))
    events = parse_sse(run(raw))
    starts = [d["content_block"] for ev, d in events
              if ev == "content_block_start"
              and d.get("content_block", {}).get("type") == "tool_use"]
    assert len(starts) == 2
    ids = [s["id"] for s in starts]
    assert ids[0] != ids[1] and all(i.startswith("toolu_") for i in ids)
    md = [d for ev, d in events if ev == "message_delta"][0]
    assert md["delta"]["stop_reason"] == "tool_use"


def test_retry_on_429_then_success(upstream):
    rt = make_rt(upstream)
    upstream.push_text(429, "rate limited")
    upstream.push_sse([chunk_text("recovered"), chunk_finish("stop")])
    raw = collect_stream(relay_stream(
        {"model": "up-main", "stream": True, "max_tokens": 1024,
         "messages": [{"role": "user", "content": "hi"}]}, "main", rt))
    events = parse_sse(run(raw))
    assert "recovered" in text_of(events)


def test_failover_after_retries_exhausted(upstream):
    rt = make_rt(upstream)
    for _ in range(5):  # enough for every attempt of the primary
        upstream.push_text(503, "down")
    upstream.push_sse([chunk_text("spare answered"), chunk_finish("stop")])
    # give the fake backend room to serve spare through the same server
    raw = collect_stream(relay_stream(
        {"model": "up-main", "stream": True, "max_tokens": 1024,
         "messages": [{"role": "user", "content": "hi"}]}, "main", rt))
    events = parse_sse(run(raw))
    assert "spare answered" in text_of(events)
    models_called = [r["json"].get("model") for r in upstream.requests]
    assert "up-spare" in models_called


def test_transport_error_pre_emission_is_retried(upstream):
    rt = make_rt(upstream)
    upstream.push_close()
    upstream.push_sse([chunk_text("second try worked"), chunk_finish("stop")])
    raw = collect_stream(relay_stream(
        {"model": "up-main", "stream": True, "max_tokens": 1024,
         "messages": [{"role": "user", "content": "hi"}]}, "main", rt))
    events = parse_sse(run(raw))
    assert "second try worked" in text_of(events)


def test_transport_error_mid_stream_appends_honest_note(upstream):
    rt = make_rt(upstream)
    upstream.push_sse([chunk_text("partial answer so far")],
                      close_without_done=True, hang_len=999999)
    # partial body + bogus length => transport error after content emitted
    raw = collect_stream(relay_stream(
        {"model": "up-main", "stream": True, "max_tokens": 1024,
         "messages": [{"role": "user", "content": "hi"}]}, "main", rt))
    events = parse_sse(run(raw))
    text = text_of(events)
    assert "partial answer so far" in text
    assert TRANSPORT_CUT_NOTE in text          # no silent truncation
    assert events[-1][0] == "message_stop"     # stream closed cleanly


def test_empty_stream_retries_with_lower_reasoning(upstream):
    rt = make_rt(upstream)  # main is reasoning: native
    # 1st answer: reasoning burned everything, finish=length, zero content
    upstream.push_sse([chunk_reasoning("still thinking", finish="length")])
    # retry (non-stream JSON) must go up WITHOUT the original effort...
    upstream.push_json({
        "choices": [{"index": 0, "finish_reason": "stop",
                     "message": {"role": "assistant",
                                 "content": "the real answer"}}],
        "usage": {"prompt_tokens": 50, "completion_tokens": 9}})
    raw = collect_stream(relay_stream(
        {"model": "up-main", "stream": True, "max_tokens": 1024,
         "reasoning_effort": "high",
         "messages": [{"role": "user", "content": "hi"}]}, "main", rt))
    events = parse_sse(run(raw))
    assert "the real answer" in text_of(events)
    retry_body = upstream.requests[1]["json"]
    assert retry_body.get("reasoning_effort") != "high"
    assert retry_body.get("stream") is False


def test_empty_twice_emits_visible_placeholder(upstream):
    rt = make_rt(upstream)
    upstream.push_sse([chunk_reasoning("x", finish="length")])
    upstream.push_json({
        "choices": [{"index": 0, "finish_reason": "length",
                     "message": {"role": "assistant",
                                 "reasoning_content": "", "content": ""}}],
        "usage": {"prompt_tokens": 50, "completion_tokens": 0}})
    raw = collect_stream(relay_stream(
        {"model": "up-main", "stream": True, "max_tokens": 1024,
         "messages": [{"role": "user", "content": "hi"}]}, "main", rt))
    events = parse_sse(run(raw))
    assert EMPTY_RETRY_NOTE in text_of(events)


def test_backend_400_context_length_trims_and_retries(upstream):
    rt = make_rt(upstream)
    upstream.push_text(400, "This model maximum context length is 230000")
    upstream.push_sse([chunk_text("after trim"), chunk_finish("stop")])
    long_hist = [{"role": "user", "content": "task"}] + \
        [{"role": "assistant", "content": f"a{i}"} for i in range(60)] + \
        [{"role": "user", "content": "now"}]
    raw = collect_stream(relay_stream(
        {"model": "up-main", "stream": True, "max_tokens": 1024,
         "messages": long_hist}, "main", rt))
    events = parse_sse(run(raw))
    assert "after trim" in text_of(events)
    first, second = upstream.requests[0]["json"], upstream.requests[1]["json"]
    assert len(second["messages"]) < len(first["messages"])


def test_collect_json_tool_calls_and_usage(upstream):
    rt = make_rt(upstream)
    upstream.push_json({
        "choices": [{"index": 0, "finish_reason": "tool_calls",
                     "message": {"role": "assistant", "content": None,
                                 "tool_calls": [{
                                     "id": "Read:0", "type": "function",
                                     "function": {"name": "Read",
                                                  "arguments": '{"p": 1}'}}]}}],
        "usage": {"prompt_tokens": 700, "completion_tokens": 40}})
    out = run(collect_json({"model": "up-main", "max_tokens": 1024,
                            "messages": [{"role": "user", "content": "go"}]},
                           "main", rt))
    assert out["stop_reason"] == "tool_use"
    assert out["content"][0]["type"] == "tool_use"
    assert out["content"][0]["id"].startswith("toolu_")
    assert out["content"][0]["input"] == {"p": 1}
    assert out["usage"]["input_tokens"] == 700
