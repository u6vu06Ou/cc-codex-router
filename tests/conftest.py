"""Shared fixtures: a scripted fake OpenAI-compatible upstream (stdlib only).

The fake server records every request (path, headers, JSON body) and answers
from a queue of scripted responses pushed by each test:

  push_json(obj, status=200)               one JSON response
  push_sse(chunks, status=200, hang_len=..) SSE data: lines; hang_len sends a
                                           bogus Content-Length so a short
                                           close surfaces as a transport error
  push_close()                             close with no response at all
"""
from __future__ import annotations

import asyncio
import json
import threading
from collections import deque
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import httpx
import pytest

from cc_codex_router.app import Runtime
from cc_codex_router.config import BackendCfg, Config, ModelCfg


def chunk_text(text, finish=None):
    d = {"choices": [{"index": 0, "delta": {"content": text},
                      "finish_reason": finish}]}
    return d


def chunk_reasoning(text, finish=None):
    return {"choices": [{"index": 0, "delta": {"reasoning_content": text},
                         "finish_reason": finish}]}


def chunk_both(reasoning, text):
    return {"choices": [{"index": 0,
                         "delta": {"reasoning_content": reasoning,
                                   "content": text},
                         "finish_reason": None}]}


def chunk_finish(reason="stop"):
    return {"choices": [{"index": 0, "delta": {}, "finish_reason": reason}]}


def chunk_usage(prompt_tokens, completion_tokens):
    return {"choices": [], "usage": {"prompt_tokens": prompt_tokens,
                                     "completion_tokens": completion_tokens}}


def chunk_toolcall(call_id, name, args=None):
    delta = {}
    if call_id is not None:
        delta = {"id": call_id, "type": "function",
                 "function": {"name": name, "arguments": args or ""}}
    elif args is not None:
        delta = {"function": {"arguments": args}}
    return {"choices": [{"index": 0, "delta": {"tool_calls": [delta]},
                         "finish_reason": None}]}


class FakeUpstream:
    def __init__(self):
        self.queue = deque()
        self.requests = []
        self._lock = threading.Lock()

        outer = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *a):
                pass

            def _handle(self):
                length = int(self.headers.get("content-length") or 0)
                body = self.rfile.read(length) if length else b""
                try:
                    parsed = json.loads(body) if body else {}
                except json.JSONDecodeError:
                    parsed = {"_raw": body.decode("utf-8", "replace")}
                with outer._lock:
                    outer.requests.append({
                        "path": self.path,
                        "headers": {k.lower(): v for k, v in self.headers.items()},
                        "json": parsed,
                    })
                    step = outer.queue.popleft() if outer.queue else None
                if step is None:
                    self.send_response(500)
                    self.send_header("content-type", "text/plain")
                    self.end_headers()
                    self.wfile.write(b"fake upstream: no scripted response")
                    return
                step(self)

            do_POST = _handle
            do_GET = _handle

        self.httpd = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.httpd.daemon_threads = True
        self.thread = threading.Thread(target=self.httpd.serve_forever,
                                       daemon=True)
        self.thread.start()
        self.url = f"http://127.0.0.1:{self.httpd.server_address[1]}"

    def stop(self):
        self.httpd.shutdown()
        self.httpd.server_close()

    # ---- scripting helpers -------------------------------------------------
    def push_json(self, obj, status=200):
        def step(h):
            payload = json.dumps(obj).encode()
            h.send_response(status)
            h.send_header("content-type", "application/json")
            h.send_header("content-length", str(len(payload)))
            h.end_headers()
            h.wfile.write(payload)
        self.queue.append(step)

    def push_text(self, status, text):
        def step(h):
            payload = text.encode()
            h.send_response(status)
            h.send_header("content-type", "text/plain")
            h.send_header("content-length", str(len(payload)))
            h.end_headers()
            h.wfile.write(payload)
        self.queue.append(step)

    def push_sse(self, chunks, status=200, close_without_done=False,
                 hang_len=None):
        def step(h):
            h.send_response(status)
            h.send_header("content-type", "text/event-stream")
            if hang_len:
                h.send_header("content-length", str(hang_len))
            h.end_headers()
            try:
                for c in chunks:
                    line = c if isinstance(c, str) else \
                        f"data: {json.dumps(c)}\n\n"
                    if not line.endswith("\n\n"):
                        line += "\n"
                    h.wfile.write(line.encode())
                    h.wfile.flush()
                if not close_without_done:
                    h.wfile.write(b"data: [DONE]\n\n")
                    h.wfile.flush()
            except (BrokenPipeError, ConnectionResetError):
                pass
            if close_without_done or hang_len:
                h.close_connection = True
        self.queue.append(step)

    def push_close(self):
        """Server accepts then drops with zero bytes — transport error."""
        def step(h):
            h.close_connection = True
            h.connection.shutdown(2)
            h.connection.close()
        self.queue.append(step)


@pytest.fixture()
def upstream():
    server = FakeUpstream()
    yield server
    server.stop()


def make_config(server: FakeUpstream, **model_over) -> Config:
    cfg = Config()
    cfg.retry_backoff = 0.001
    cfg.keepalive_secs = 0.05
    cfg.backends["fake"] = BackendCfg(name="fake",
                                      base_url=server.url + "/v1",
                                      api_key_env="", max_concurrent=0)
    main = ModelCfg(alias="main", backend="fake", upstream_model="up-main",
                    context_window=100_000, vision=True, reasoning="native",
                    fallbacks=["spare"])
    spare = ModelCfg(alias="spare", backend="fake", upstream_model="up-spare",
                     context_window=100_000)
    for k, v in model_over.items():
        setattr(main, k, v)
    cfg.models["main"] = main
    cfg.models["spare"] = spare
    return cfg


def make_rt(server: FakeUpstream, **model_over) -> Runtime:
    cfg = make_config(server, **model_over)
    return Runtime(cfg, httpx.AsyncClient(timeout=httpx.Timeout(10.0)))


def run(coro):
    return asyncio.run(coro)


def basic_anthropic_req(model="main", text="hello", stream=True):
    return {"model": model, "max_tokens": 1024, "stream": stream,
            "messages": [{"role": "user", "content": text}]}


def parse_sse(text: str):
    """[(event, data_dict)] from an SSE payload string."""
    out = []
    for block in text.split("\n\n"):
        block = block.strip()
        if not block or block.startswith(":"):
            continue
        ev = None
        data = None
        for line in block.splitlines():
            if line.startswith("event: "):
                ev = line[7:]
            elif line.startswith("data: "):
                try:
                    data = json.loads(line[6:])
                except json.JSONDecodeError:
                    data = line[6:]
        out.append((ev, data))
    return out


async def collect_stream(agen) -> str:
    parts = []
    async for frame in agen:
        parts.append(frame)
    return "".join(parts)
