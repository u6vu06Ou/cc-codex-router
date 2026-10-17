"""The server: team routing + the hardened proxy pipeline, one port.

Routing rule: a request whose model starts with the configured claude prefix
(default "claude-"), or has no model at all, is forwarded untouched to the
configured Anthropic-compatible upstream (its auth headers kept). Any other
model name is looked up in the config's `models:` table and run through the
proxy pipeline (translate -> clamp -> hardened stream/JSON), with the
request's auth headers stripped and the target backend's key injected.

Two API dialects are served: /v1/messages (Anthropic Messages, what Claude
Code speaks) and /v1/responses (OpenAI Responses, what Codex CLI speaks).
The Responses dialect has no lead lane — Anthropic's own API has no
/v1/responses, so an unconfigured model there is a plain 404.
"""
from __future__ import annotations

import asyncio
import json
import logging
import time
from contextlib import asynccontextmanager

import httpx
import uvicorn
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse, StreamingResponse
from starlette.routing import Route

from .clamp import clamp_context, est_request_tokens
from .config import Config, load_config, validate
from .responses import (collect_responses_json, relay_responses_stream,
                        responses_to_chat)
from .stream import UpstreamError, collect_json, relay_stream
from .translate import translate_anthropic_to_openai


log = logging.getLogger("cc_codex_router")

# hop-by-hop headers, plus anything httpx recomputes
DROP_REQ = {"host", "content-length", "connection", "accept-encoding",
            "transfer-encoding"}
DROP_RESP = {"content-length", "connection", "transfer-encoding",
             "content-encoding"}
AUTH_HEADERS = {"authorization", "x-api-key"}


class Runtime:
    """Shared state: config, http client, per-backend semaphores, breakers."""

    def __init__(self, cfg: Config, client: httpx.AsyncClient):
        self.cfg = cfg
        self.client = client
        self.semaphores = {
            name: asyncio.Semaphore(b.max_concurrent)
            for name, b in cfg.backends.items() if b.max_concurrent > 0
        }
        self.down_until = {}  # alias -> monotonic deadline

    def backend_of(self, model_cfg):
        return self.cfg.backends[model_cfg.backend]

    def backend_headers(self, backend) -> dict:
        key = backend.api_key()
        hdrs = {"content-type": "application/json"}
        if key:
            hdrs["Authorization"] = f"Bearer {key}"
        hdrs.update(backend.resolved_headers())
        return hdrs

    def chat_url(self, backend) -> str:
        return backend.base_url + "/chat/completions"

    def backend_stream(self, model_cfg, req):
        """Async context manager: POST a streaming chat/completions request."""
        backend = self.backend_of(model_cfg)
        return self.client.stream("POST", self.chat_url(backend), json=req,
                                  headers=self.backend_headers(backend))

    async def backend_post(self, model_cfg, req):
        backend = self.backend_of(model_cfg)
        return await self.client.post(self.chat_url(backend), json=req,
                                      headers=self.backend_headers(backend))

    def mark_down(self, alias: str):
        cooldown = max(self.cfg.retry_backoff * 10, 30.0)
        self.down_until[alias] = time.monotonic() + cooldown

    def is_down(self, alias: str) -> bool:
        return time.monotonic() < self.down_until.get(alias, 0.0)

    def next_fallback(self, alias: str):
        model = self.cfg.models.get(alias)
        if not model:
            return None
        for fb in model.fallbacks:
            if fb in self.cfg.models and not self.is_down(fb):
                return fb
        return None


def anthropic_error(status: int, message: str,
                    err_type: str = "api_error") -> JSONResponse:
    return JSONResponse(
        {"type": "error", "error": {"type": err_type, "message": message}},
        status_code=status)


def responses_error(status: int, message: str, code: str | None = None,
                    err_type: str = "invalid_request_error") -> JSONResponse:
    """OpenAI/Responses-style error body (what Codex prints on failure)."""
    return JSONResponse(
        {"error": {"message": message, "type": err_type,
                   "param": None, "code": code}},
        status_code=status)


async def _passthrough(request: Request, target_base: str):
    """Forward the request byte-for-byte (auth kept) to a real upstream."""
    rt: Runtime = request.app.state.rt
    url = target_base + request.url.path
    if request.url.query:
        url += "?" + request.url.query
    headers = {k: v for k, v in request.headers.items()
               if k.lower() not in DROP_REQ}
    # gzip-encoded SSE breaks pure-passthrough streaming: ask for identity.
    headers["accept-encoding"] = "identity"
    body = await request.body()
    try:
        req = rt.client.build_request(request.method, url, headers=headers,
                                      content=body)
        resp = await rt.client.send(req, stream=True)
    except httpx.HTTPError as e:
        return anthropic_error(502, f"anthropic upstream unreachable: {e!r}")
    out_headers = {k: v for k, v in resp.headers.items()
                   if k.lower() not in DROP_RESP}

    async def stream():
        try:
            async for chunk in resp.aiter_bytes():
                yield chunk
        finally:
            await resp.aclose()

    return StreamingResponse(stream(), status_code=resp.status_code,
                             headers=out_headers)


def _wants_lead_lane(body: bytes, cfg: Config) -> bool:
    if not cfg.anthropic_upstream:
        return False
    try:
        model = str(json.loads(body or b"{}").get("model") or "")
    except (json.JSONDecodeError, ValueError):
        model = ""
    return not model or model.startswith(cfg.claude_prefix)


async def _run_pipeline(request: Request, body_bytes: bytes):
    rt: Runtime = request.app.state.rt
    try:
        anthropic_req = json.loads(body_bytes)
    except (json.JSONDecodeError, ValueError):
        return anthropic_error(400, "request body is not valid JSON",
                               "invalid_request_error")
    alias = str(anthropic_req.get("model") or "")
    model = rt.cfg.models.get(alias)
    if not model:
        known = ", ".join(sorted(rt.cfg.models)) or "(none configured)"
        return anthropic_error(404, f"unknown model {alias!r}; configured: {known}",
                               "not_found_error")
    backend = rt.backend_of(model)
    if not backend.base_url:
        return anthropic_error(500, f"backend {backend.name!r} has no base_url")

    openai_req = translate_anthropic_to_openai(anthropic_req, model)
    openai_req = clamp_context(openai_req, model, rt.cfg)

    sem = rt.semaphores.get(backend.name)
    if sem is None:
        return await _dispatch(request, rt, alias, openai_req, anthropic_req)

    # Per-backend queue with a visible heartbeat so long waits don't look dead.
    if not anthropic_req.get("stream", False):
        async with sem:
            return await _dispatch(request, rt, alias, openai_req, anthropic_req)

    async def gated():
        if sem.locked():
            yield ": queued — waiting for a backend slot\n\n"
        async with sem:
            async for frame in relay_stream(openai_req, alias, rt):
                yield frame

    return StreamingResponse(gated(), media_type="text/event-stream",
                             headers={"cache-control": "no-cache"})


async def _dispatch(request: Request, rt: Runtime, alias: str,
                    openai_req: dict, anthropic_req: dict):
    if anthropic_req.get("stream", False):
        return StreamingResponse(relay_stream(openai_req, alias, rt),
                                 media_type="text/event-stream",
                                 headers={"cache-control": "no-cache"})
    try:
        return JSONResponse(await collect_json(openai_req, alias, rt))
    except UpstreamError as e:
        return anthropic_error(e.status if e.status != 400 else 400,
                               f"model {alias}: upstream {e.status}: {e.body[:200]}")


async def responses_endpoint(request: Request):
    """POST /v1/responses — the OpenAI Responses dialect Codex CLI speaks.

    There is no lead lane here (nothing routes to the Anthropic upstream):
    the model field must name a configured alias; anything else is a 404
    that lists what IS configured.
    """
    body_bytes = await request.body()
    rt: Runtime = request.app.state.rt
    try:
        responses_req = json.loads(body_bytes)
    except (json.JSONDecodeError, ValueError):
        return responses_error(400, "request body is not valid JSON")
    if not isinstance(responses_req, dict):
        return responses_error(400, "request body must be a JSON object")
    alias = str(responses_req.get("model") or "")
    model = rt.cfg.models.get(alias)
    if not model:
        known = ", ".join(sorted(rt.cfg.models)) or "(none configured)"
        return responses_error(
            404, f"unknown model {alias!r}; configured: {known}",
            code="model_not_found")
    backend = rt.backend_of(model)
    if not backend.base_url:
        return responses_error(500, f"backend {backend.name!r} has no base_url",
                               err_type="server_error")
    try:
        openai_req = responses_to_chat(responses_req, model)
    except ValueError as e:
        return responses_error(400, str(e))
    openai_req = clamp_context(openai_req, model, rt.cfg)

    sem = rt.semaphores.get(backend.name)
    if sem is None:
        return await _dispatch_responses(rt, alias, responses_req, openai_req)

    # Per-backend queue with a visible heartbeat so long waits don't look dead.
    async def gated():
        if sem.locked():
            yield ": queued — waiting for a backend slot\n\n"
        async with sem:
            async for frame in relay_responses_stream(openai_req, alias, rt):
                yield frame

    if responses_req.get("stream", False):
        return StreamingResponse(gated(), media_type="text/event-stream",
                                 headers={"cache-control": "no-cache"})
    async with sem:
        return await _dispatch_responses(rt, alias, responses_req, openai_req)


async def _dispatch_responses(rt: Runtime, alias: str, responses_req: dict,
                              openai_req: dict):
    if responses_req.get("stream", False):
        return StreamingResponse(relay_responses_stream(openai_req, alias, rt),
                                 media_type="text/event-stream",
                                 headers={"cache-control": "no-cache"})
    try:
        return JSONResponse(
            await collect_responses_json(openai_req, alias, rt))
    except UpstreamError as e:
        return responses_error(e.status if e.status != 400 else 400,
                               f"model {alias}: upstream {e.status}: "
                               f"{e.body[:200]}",
                               err_type="server_error" if e.status >= 500
                               else "invalid_request_error")


async def messages_endpoint(request: Request):
    body = await request.body()
    rt: Runtime = request.app.state.rt
    if _wants_lead_lane(body, rt.cfg):
        log.info("POST /v1/messages -> anthropic upstream (lead lane)")
        return await _passthrough(request, rt.cfg.anthropic_upstream)
    return await _run_pipeline(request, body)


async def count_tokens_endpoint(request: Request):
    body = await request.body()
    rt: Runtime = request.app.state.rt
    if _wants_lead_lane(body, rt.cfg):
        return await _passthrough(request, rt.cfg.anthropic_upstream)
    try:
        anthropic_req = json.loads(body)
    except (json.JSONDecodeError, ValueError):
        return anthropic_error(400, "request body is not valid JSON",
                               "invalid_request_error")
    alias = str(anthropic_req.get("model") or "")
    model = rt.cfg.models.get(alias)
    if not model:
        return anthropic_error(404, f"unknown model {alias!r}",
                               "not_found_error")
    openai_req = translate_anthropic_to_openai(anthropic_req, model)
    return JSONResponse({"input_tokens": est_request_tokens(openai_req)})


async def models_endpoint(request: Request):
    rt: Runtime = request.app.state.rt
    data = [{"id": alias, "object": "model", "owned_by": m.backend}
            for alias, m in sorted(rt.cfg.models.items())]
    return JSONResponse({"object": "list", "data": data})


async def health_endpoint(request: Request):
    rt: Runtime = request.app.state.rt
    return JSONResponse({
        "ok": True,
        "models": sorted(rt.cfg.models),
        "anthropic_upstream": bool(rt.cfg.anthropic_upstream),
    })


async def fallback_endpoint(request: Request):
    body = await request.body()
    rt: Runtime = request.app.state.rt
    if _wants_lead_lane(body, rt.cfg):
        return await _passthrough(request, rt.cfg.anthropic_upstream)
    return anthropic_error(404, f"no route for {request.url.path}",
                           "not_found_error")


def create_app(cfg: Config) -> Starlette:
    @asynccontextmanager
    async def lifespan(app):
        app.state.rt = Runtime(cfg, httpx.AsyncClient(
            timeout=httpx.Timeout(connect=15, read=900, write=60, pool=30)))
        problems = validate(cfg)
        for p in problems:
            log.warning("config: %s", p)
        yield
        await app.state.rt.client.aclose()

    return Starlette(
        routes=[
            Route("/v1/messages", messages_endpoint, methods=["POST"]),
            Route("/v1/messages/count_tokens", count_tokens_endpoint,
                  methods=["POST"]),
            Route("/v1/responses", responses_endpoint, methods=["POST"]),
            Route("/v1/models", models_endpoint, methods=["GET"]),
            Route("/health", health_endpoint, methods=["GET"]),
            Route("/{path:path}", fallback_endpoint,
                  methods=["GET", "POST", "PUT", "DELETE", "PATCH"]),
        ],
        lifespan=lifespan,
    )


def serve(config_path: str):
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s %(message)s")
    cfg = load_config(config_path)
    host, _, port = cfg.listen.rpartition(":")
    app = create_app(cfg)
    uvicorn.run(app, host=host or "127.0.0.1", port=int(port or 4003),
                log_level="warning")
