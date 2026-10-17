"""Router behavior end to end: lead-lane passthrough vs proxy pipeline."""
import asyncio
import json

import httpx
from starlette.requests import Request

from claude_team_router.app import (Runtime, count_tokens_endpoint,
                                    create_app, health_endpoint,
                                    messages_endpoint, models_endpoint)
from claude_team_router.config import Config

from conftest import (FakeUpstream, chunk_finish, chunk_text, make_config,
                      parse_sse, run)


def make_request(app, method, path, body=b"", headers=None):
    hdrs = [(k.lower().encode(), v.encode()) for k, v in (headers or {}).items()]
    sent = {"done": False}

    async def receive():
        if not sent["done"]:
            sent["done"] = True
            return {"type": "http.request", "body": body, "more_body": False}
        return {"type": "http.disconnect"}

    scope = {"type": "http", "method": method, "path": path, "app": app,
             "headers": hdrs, "query_string": b""}
    return Request(scope, receive)


def app_with(upstream, anthropic_upstream_url=""):
    cfg = make_config(upstream)
    cfg.anthropic_upstream = anthropic_upstream_url
    app = create_app(cfg)
    app.state.rt = Runtime(cfg, httpx.AsyncClient(timeout=httpx.Timeout(10.0)))
    return app


def call(endpoint, req):
    """Run the endpoint AND consume its streaming body in one event loop."""
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


def test_unknown_model_404_with_known_list(upstream):
    app = app_with(upstream)
    req = make_request(app, "POST", "/v1/messages",
                       json.dumps({"model": "nope", "messages": []}).encode())
    status, body = call(messages_endpoint, req)
    assert status == 404
    assert b"main" in body


def test_lead_lane_passthrough_keeps_auth_and_identity(upstream):
    app = app_with(upstream, anthropic_upstream_url=upstream.url)
    upstream.push_text(200, "lead-lane-answer")
    req = make_request(
        app, "POST", "/v1/messages",
        json.dumps({"model": "claude-x", "messages": []}).encode(),
        headers={"authorization": "Bearer unit-key", "x-api-key": "unit-key",
                 "accept-encoding": "gzip"})
    status, body = call(messages_endpoint, req)
    assert status == 200
    assert body == b"lead-lane-answer"
    seen = upstream.requests[0]["headers"]
    assert seen["authorization"] == "Bearer unit-key"   # auth kept for real API
    assert seen["accept-encoding"] == "identity"        # no gzip over SSE


def test_proxy_lane_strips_client_auth_and_streams(upstream):
    app = app_with(upstream, anthropic_upstream_url="http://127.0.0.1:1")
    upstream.push_sse([chunk_text("proxied"), chunk_finish("stop")])
    req = make_request(
        app, "POST", "/v1/messages",
        json.dumps({"model": "main", "stream": True, "max_tokens": 100,
                    "messages": [{"role": "user", "content": "hi"}]}).encode(),
        headers={"authorization": "Bearer unit-key"})
    status, body = call(messages_endpoint, req)
    assert status == 200
    events = parse_sse(body.decode())
    texts = [d["delta"]["text"] for ev, d in events
             if ev == "content_block_delta"
             and d.get("delta", {}).get("type") == "text_delta"]
    assert "".join(texts) == "proxied"
    seen = upstream.requests[0]["headers"]
    assert "authorization" not in seen  # client key never reaches the backend


def test_backend_key_injected_from_env(upstream, monkeypatch):
    cfg = make_config(upstream)
    cfg.backends["fake"].api_key_env = "UNIT_BACKEND_KEY"
    monkeypatch.setenv("UNIT_BACKEND_KEY", "ubk")
    app = create_app(cfg)
    app.state.rt = Runtime(cfg, httpx.AsyncClient(timeout=httpx.Timeout(10.0)))
    upstream.push_json({"choices": [{"index": 0, "finish_reason": "stop",
                                     "message": {"content": "ok"}}],
                        "usage": {}})
    req = make_request(app, "POST", "/v1/messages",
                       json.dumps({"model": "main", "max_tokens": 10,
                                   "messages": [{"role": "user",
                                                 "content": "hi"}]}).encode())
    resp = run(messages_endpoint(req))
    assert resp.status_code == 200
    seen = upstream.requests[0]["headers"]
    assert seen["authorization"] == "Bearer ubk"


def test_count_tokens_local_and_passthrough(upstream):
    app = app_with(upstream, anthropic_upstream_url=upstream.url)
    upstream.push_text(200, '{"input_tokens": 42}')
    lead = make_request(app, "POST", "/v1/messages/count_tokens",
                        json.dumps({"model": "claude-x",
                                    "messages": []}).encode())
    _, body = call(count_tokens_endpoint, lead)
    assert json.loads(body)["input_tokens"] == 42

    local = make_request(app, "POST", "/v1/messages/count_tokens",
                         json.dumps({"model": "main", "max_tokens": 10,
                                     "messages": [{"role": "user",
                                                   "content": "x" * 300}]
                                     }).encode())
    _, body = call(count_tokens_endpoint, local)
    assert json.loads(body)["input_tokens"] >= 100


def test_models_and_health(upstream):
    app = app_with(upstream)
    _, body = call(models_endpoint, make_request(app, "GET", "/v1/models"))
    ids = [m["id"] for m in json.loads(body)["data"]]
    assert ids == ["main", "spare"]
    _, body = call(health_endpoint, make_request(app, "GET", "/health"))
    data = json.loads(body)
    assert data["ok"] and data["models"] == ["main", "spare"]


def test_extra_headers(monkeypatch):
    """Backend extra_headers: literal pass-through, env: reads, Bearer merge."""
    from claude_team_router.config import BackendCfg

    cfg = Config()
    cfg.backends["b"] = BackendCfg(
        name="b", base_url="http://127.0.0.1:9/v1",
        api_key_env="EXTRA_HEADERS_TEST_KEY",
        extra_headers={"x-org-id": "team-42",
                       "x-alt-key": "env:EXTRA_HEADERS_TEST_ALT",
                       "x-missing": "env:EXTRA_HEADERS_TEST_UNSET"})
    monkeypatch.setenv("EXTRA_HEADERS_TEST_KEY", "k1")
    monkeypatch.setenv("EXTRA_HEADERS_TEST_ALT", "k2")
    rt = Runtime(cfg, httpx.AsyncClient(timeout=10.0))
    hdrs = rt.backend_headers(cfg.backends["b"])
    assert hdrs["Authorization"] == "Bearer k1"   # from api_key_env
    assert hdrs["x-org-id"] == "team-42"          # literal, non-secret
    assert hdrs["x-alt-key"] == "k2"              # env:NAME read
    assert "x-missing" not in hdrs                # unset variable omitted

    cfg.backends["c"] = BackendCfg(
        name="c", base_url="http://127.0.0.1:9/v1",
        api_key_env="EXTRA_HEADERS_TEST_KEY",
        extra_headers={"Authorization": "Bearer k3"})  # intentional override
    hdrs = rt.backend_headers(cfg.backends["c"])
    assert hdrs["Authorization"] == "Bearer k3"
