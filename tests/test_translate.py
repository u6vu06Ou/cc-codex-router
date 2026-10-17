"""Inbound translation fixes: reminders, degenerate turns, thinking history,
unique tool ids, tool_result images, system hoisting, tools mapping."""
import json

from claude_team_router.config import ModelCfg
from claude_team_router.translate import (THINKING_ONLY_PLACEHOLDER,
                                          translate_anthropic_to_openai)


def mdl(**over):
    base = dict(alias="m", backend="b", upstream_model="up", vision=False,
                reasoning="none")
    base.update(over)
    return ModelCfg(**base)


def test_system_string_and_list_flattened():
    req = {"model": "m", "messages": [],
           "system": [{"type": "text", "text": "a "},
                      {"type": "text", "text": "b"}]}
    out = translate_anthropic_to_openai(req, mdl())
    assert out["messages"][0] == {"role": "system", "content": "a \n\nb"}


def test_reminder_only_user_turn_becomes_system():
    req = {"model": "m",
           "messages": [
               {"role": "user", "content": [
                   {"type": "text",
                    "text": "<system-reminder>keep going</system-reminder>"}]}]}
    out = translate_anthropic_to_openai(req, mdl())
    assert out["messages"][0]["role"] == "system"


def test_reminder_plus_text_stays_user():
    req = {"model": "m",
           "messages": [
               {"role": "user", "content": [
                   {"type": "text",
                    "text": "<system-reminder>x</system-reminder> real words"}]}]}
    out = translate_anthropic_to_openai(req, mdl())
    assert out["messages"][0]["role"] == "user"


def test_degenerate_empty_turns_stripped():
    req = {"model": "m",
           "messages": [
               {"role": "assistant", "content": ""},
               {"role": "user", "content": [
                   {"type": "text", "text": "(no content)"}]},
               {"role": "user", "content": "real"},
           ]}
    out = translate_anthropic_to_openai(req, mdl())
    contents = [m["content"] for m in out["messages"]]
    assert contents == ["real"] or contents == [[{"type": "text", "text": "real"}]]


def test_thinking_only_assistant_turn_preserved_placeholder():
    req = {"model": "m",
           "messages": [
               {"role": "assistant", "content": [
                   {"type": "thinking", "thinking": "hmm"}]},
               {"role": "user", "content": "next"},
           ]}
    out = translate_anthropic_to_openai(req, mdl())
    assert out["messages"][0]["role"] == "assistant"
    assert out["messages"][0]["content"] == THINKING_ONLY_PLACEHOLDER


def test_native_reasoning_history_carries_reasoning_content():
    req = {"model": "m",
           "messages": [
               {"role": "assistant", "content": [
                   {"type": "thinking", "thinking": "chain"},
                   {"type": "text", "text": "answer"}]},
           ]}
    out = translate_anthropic_to_openai(req, mdl(reasoning="native"))
    assert out["messages"][0]["reasoning_content"] == "chain"
    assert "reasoning_effort" in out  # native reasoning models get an effort


def test_tool_use_ids_in_history_pass_through():
    tu = {"type": "tool_use", "id": "Read:0", "name": "Read", "input": {}}
    req = {"model": "m",
           "messages": [
               {"role": "assistant", "content": [dict(tu), dict(tu)]}]}
    out = translate_anthropic_to_openai(req, mdl())
    ids = [tc["id"] for tc in out["messages"][0]["tool_calls"]]
    assert ids == ["Read:0", "Read:0"]
    # fresh unique ids are minted on OUTGOING turns (covered in test_stream);
    # arguments are always valid JSON text
    assert out["messages"][0]["tool_calls"][0]["function"]["arguments"] == "{}"


def test_broken_tool_arguments_in_history_never_sent_raw():
    tu = {"type": "tool_use", "id": "x", "name": "Run",
          "input": {"cmd": "ls"}}
    req = {"model": "m",
           "messages": [{"role": "assistant", "content": [tu]}]}
    out = translate_anthropic_to_openai(req, mdl())
    args = out["messages"][0]["tool_calls"][0]["function"]["arguments"]
    assert json.loads(args) == {"cmd": "ls"}


def test_tool_result_image_vision_model_gets_image_url():
    img = {"type": "image",
           "source": {"media_type": "image/png", "data": "QUJD"}}
    req = {"model": "m",
           "messages": [{"role": "user", "content": [
               {"type": "tool_result", "tool_use_id": "t1",
                "content": [img]}]}]}
    out = translate_anthropic_to_openai(req, mdl(vision=True))
    roles = [m["role"] for m in out["messages"]]
    assert roles == ["tool", "user"]
    kinds = [b["type"] for b in out["messages"][1]["content"]]
    assert "image_url" in kinds
    assert out["messages"][0]["content"]  # tool result text not empty


def test_tool_result_image_non_vision_gets_honest_note():
    img = {"type": "image",
           "source": {"media_type": "image/png", "data": "QUJD"}}
    req = {"model": "m",
           "messages": [{"role": "user", "content": [
               {"type": "tool_result", "tool_use_id": "t1",
                "content": [img]}]}]}
    out = translate_anthropic_to_openai(req, mdl(vision=False))
    joined = json.dumps(out["messages"], ensure_ascii=False)
    assert "[image omitted" in joined  # never silently dropped


def test_image_in_request_triggers_system_hoist():
    req = {"model": "m",
           "system": "lead",
           "messages": [
               {"role": "user", "content": [
                   {"type": "text",
                    "text": "<system-reminder>mid</system-reminder>"}]},
               {"role": "user", "content": [
                   {"type": "image",
                    "source": {"media_type": "image/png", "data": "QUJD"}}]},
           ]}
    out = translate_anthropic_to_openai(req, mdl(vision=True))
    sys_msgs = [i for i, m in enumerate(out["messages"])
                if m["role"] == "system"]
    assert sys_msgs == [0]
    assert "mid" in out["messages"][0]["content"]


def test_strict_system_position_hoists_without_images():
    req = {"model": "m",
           "system": "lead",
           "messages": [
               {"role": "user", "content": [
                   {"type": "text",
                    "text": "<system-reminder>mid</system-reminder>"}]},
               {"role": "user", "content": "hi"},
           ]}
    out = translate_anthropic_to_openai(req, mdl(strict_system_position=True))
    roles = [m["role"] for m in out["messages"]]
    assert roles[0] == "system" and "system" not in roles[1:]


def test_tools_and_tool_choice_mapped():
    req = {"model": "m", "messages": [{"role": "user", "content": "x"}],
           "tools": [{"name": "Bash", "description": "run",
                      "input_schema": {"type": "object", "properties": {}}}],
           "tool_choice": {"type": "tool", "name": "Bash"}}
    out = translate_anthropic_to_openai(req, mdl())
    assert out["tools"][0]["function"]["parameters"] == \
        {"type": "object", "properties": {}}
    assert out["tool_choice"] == {"type": "function",
                                  "function": {"name": "Bash"}}
