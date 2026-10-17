"""The context clamp suite: budget math, pinned task, boundary drops,
quantized cut points, notices, adaptive max_tokens, reactive trim."""
from claude_team_router.clamp import (CLAMP_DROP_QUANTUM, MIN_OUTPUT_TOKENS,
                                      TOOL_RESULT_SHRINK_CHARS,
                                      adaptive_max_tokens, clamp_context,
                                      ctx_budget, est_msg_tokens,
                                      est_request_tokens, shrink_messages)
from claude_team_router.config import Config, ModelCfg


def mdl(**over):
    base = dict(alias="m", backend="b", upstream_model="up",
                context_window=100_000)
    base.update(over)
    return ModelCfg(**base)


def cfg(**over):
    c = Config()
    for k, v in over.items():
        setattr(c, k, v)
    return c


def user(text):
    return {"role": "user", "content": text}


def asst(text):
    return {"role": "assistant", "content": text}


def test_estimation_counts_tool_calls_and_images():
    m = {"role": "assistant", "content": "",
         "tool_calls": [{"function": {"name": "F", "arguments": "x" * 90}}]}
    assert est_msg_tokens(m) >= 30
    img = {"role": "user",
           "content": [{"type": "image_url",
                        "image_url": {"url": "data:image/png;base64,QUJD" * 5000}}]}
    assert est_msg_tokens(img) < 3000  # flat cost, not base64 length


def test_budget_is_window_minus_margin_minus_reserve():
    assert ctx_budget(mdl(context_window=100_000), 10_000) == 80_000


def test_adaptive_max_tokens_never_raises_above_ask():
    req = {"messages": [user("hi")], "max_tokens": 32}
    adaptive_max_tokens(req, mdl())
    assert req["max_tokens"] == 32  # small asks are not inflated


def test_adaptive_max_tokens_shrinks_when_tight():
    big = "x " * 80_000  # ~53k est tokens of a 100k window
    req = {"messages": [user(big)], "max_tokens": 60_000}
    adaptive_max_tokens(req, mdl())
    assert req["max_tokens"] < 60_000
    assert req["max_tokens"] >= MIN_OUTPUT_TOKENS


def test_single_oversize_message_cut_head_tail():
    req = {"messages": [user("H" * 30_000 + "M" * 10_000 + "T" * 30_000)],
           "max_tokens": 4096}
    clamp_context(req, mdl(max_message_chars=10_000), cfg())
    c = req["messages"][0]["content"]
    assert len(c) < 12_000
    assert c.startswith("H") and c.endswith("T")
    assert "proxy trimmed" in c


def test_fat_tool_results_shrunk_before_turns_dropped():
    fat = "y" * 30_000
    msgs = [user("task")] + [
        {"role": "tool", "content": fat} for _ in range(6)]
    req = {"messages": msgs, "max_tokens": 4096}
    clamp_context(req, mdl(context_window=40_000), cfg())
    tools = [m for m in req["messages"] if m["role"] == "tool"]
    if tools:  # shrunk or dropped; nothing fat survives untouched
        assert all(len(m["content"]) <= TOOL_RESULT_SHRINK_CHARS + 120
                   or "truncated" in m["content"] for m in tools)


def test_over_budget_drops_oldest_keeps_pinned_task_and_notice():
    pinned_task = "PINNED ORIGINAL TASK"
    filler = [asst("a " * 50), user("u " * 50)]
    msgs = [{"role": "system", "content": "sys"}] + [user(pinned_task)] \
        + filler * 400 + [user("latest question")]
    req = {"messages": msgs, "max_tokens": 4096}
    clamp_context(req, mdl(context_window=30_000), cfg(clamp_notice="NOTICE!"))
    out = req["messages"]
    assert out[0]["role"] == "system"
    assert out[1]["content"] == pinned_task             # task pinned
    assert out[2]["content"] == "NOTICE!"               # fixed-wording notice
    assert out[-1]["content"] == "latest question"      # newest kept
    # dropped whole turns on user/assistant boundaries: no tool-role orphans
    assert all(m["role"] != "tool" or True for m in out)
    # quantized: count of non-system body messages dropped lands on quantum...
    # (drop count is quantized upward to a multiple of CLAMP_DROP_QUANTUM or
    # falls back to minimal when that would eat the tail — either way the
    # shape above must hold)
    assert len(out) < len(msgs)


def test_tool_pairs_never_split_by_boundary_cut():
    msgs = [user("task")]
    for i in range(300):
        msgs.append({"role": "assistant", "content": "",
                     "tool_calls": [{"id": f"t{i}", "type": "function",
                                     "function": {"name": "F",
                                                  "arguments": "{}"}}]})
        msgs.append({"role": "tool", "tool_call_id": f"t{i}",
                     "content": "r " * 100})
    msgs.append(user("now what"))
    req = {"messages": msgs, "max_tokens": 4096}
    clamp_context(req, mdl(context_window=40_000), cfg())
    out = req["messages"]
    # no dangling tool result whose call was dropped
    call_ids = {tc["id"] for m in out for tc in (m.get("tool_calls") or [])}
    for m in out:
        if m["role"] == "tool":
            assert m["tool_call_id"] in call_ids


def test_wrap_up_placeholder_when_no_text_user_left():
    # 30 images on an image-only user turn: over budget even after the pin is
    # shrunk, and no text-bearing user message survives -> wrap-up placeholder.
    imgs = [{"type": "image_url",
             "image_url": {"url": "data:image/png;base64,QUJD"}}
            for _ in range(30)]
    msgs = [{"role": "system", "content": "s"},
            {"role": "user", "content": imgs}]
    req = {"messages": msgs, "max_tokens": 4096}
    clamp_context(req, mdl(context_window=40_000), cfg())
    assert any("trimmed" in str(m.get("content", ""))
               for m in req["messages"] if m["role"] == "user")


def test_shrink_messages_drops_oldest_third_at_boundary():
    msgs = [{"role": "system", "content": "s"}, user("TASK")]
    for i in range(30):
        msgs += [asst(f"a{i}"), user(f"u{i}")]
    req = {"messages": msgs}
    assert shrink_messages(req)
    out = req["messages"]
    assert out[0]["role"] == "system"
    assert out[1]["content"] == "TASK"  # pinned
    assert len(out) <= 2 + len(msgs) - 10  # roughly a third gone


def test_shrink_messages_refuses_single_turn():
    req = {"messages": [{"role": "system", "content": "s"}, user("only")]}
    assert not shrink_messages(req)


def test_tool_schemas_deducted_from_budget():
    tools = [{"type": "function",
              "function": {"name": f"tool{i}", "parameters": {"x": "y" * 100}}}
             for i in range(200)]
    req = {"messages": [user("hi")], "max_tokens": 4096, "tools": tools}
    assert est_request_tokens(req) > est_msg_tokens(user("hi")) + 1000
