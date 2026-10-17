# cc-codex-router — design

One local Anthropic-compatible endpoint that routes Claude Code traffic:

- model starts with `claude-` (or is empty) -> forwarded untouched to your real Anthropic API account (auth headers kept).
- any other model name -> goes through the proxy pipeline to **your own** OpenAI-compatible backend (`/chat/completions`), with the hard-won fixes below applied in transit.

Config: one YAML (or JSON) file. `backends:` hold `base_url` + `api_key_env` (the **name** of the env var holding the key — keys never live in the file); `models:` are user-defined aliases with `backend`, `upstream_model`, `context_window`, `vision`, `reasoning`, `fallbacks`. No model names are baked into code.

## Battle fixes kept (porting from a long-running production proxy)

- Stream start eaten: a delta carrying reasoning **and** text must emit both — the old `continue` dropped the answer's first words.
- Thinking burned the whole budget -> empty reply: auto retry once with lowered reasoning effort; if still empty, emit a visible, honest placeholder instead of silence. Thinking-only assistant turns get a placeholder in history so they never evaporate.
- Context clamp suite: per-model window, safety margin, tool schemas counted (capped at half the budget), first user task pinned, whole turns dropped on role boundaries from the middle, cut points quantized (keeps upstream prefix-cache warm), tool results shrunk, adaptive `max_tokens`, fixed-wording notice so the model knows context was cut. Reactive trim if the backend still answers 400.
- Failover chain per model with cooldown circuit breaker; retryable statuses 429/500/502/503, counts/backoff in config.
- Transport retry only while nothing has been streamed yet; after partial output, close the stream cleanly with an honest dropped-connection note.
- Unique tool-call ids (`toolu_` + random), degenerate empty turns stripped, mid-conversation system-reminder turns hoisted, broken tool-call JSON repaired or dropped (never sent upstream raw).
- Usage pass-through: keep reading after the finish chunk, overwrite local token estimates only with non-zero upstream numbers; plus `/v1/messages/count_tokens`.
- Per-backend concurrency semaphore with an SSE "queued" heartbeat; SSE keepalive comments so idle streams don't get killed by impatient middleboxes.

## Responses API dialect (Codex CLI)

`/v1/responses` serves the OpenAI Responses dialect on top of the same
config, clamp, retries and failover. Ported fixes that are specific to it:

- function_call arguments arrive from history as raw strings and may be broken JSON (truncated, or chat-template markers leaked mid-arguments); they are sanitized before re-entering an upstream tokenizer, because one broken parse otherwise bricks the session on every later turn.
- `{call_id: null, id: null}` outputs still produce `tool_call_id: ""`.
- developer/system content appearing mid-conversation is collected into ONE leading system message.
- reasoning models burn the whole output budget on thinking when the cap is small: native-reasoning aliases floor `max_tokens` at 32k on this path (8k elsewhere).
- `tool_choice` is forwarded only when tools survive filtering; upstream `finish_reason: "length"` closes as `incomplete`, never `completed`; reasoning/cached usage details pass through for the client's context accounting.
- streaming retries/failovers are allowed only before any user-visible item was emitted; afterwards the turn fails honestly (`response.failed`) so the client retries it.
- unknown tool types are skipped with a warning instead of hard-failing the request.

## Dropped (machine- or fleet-specific)

- Watchdogs, scheduled restarts, tunnel accounting, hardcoded alias tables, datacenter special cases.
- External vision-describe chain (non-vision models get an honest "[image omitted]" note instead of silently losing the image).
- Response-side turn-loop guard and auto context probing (set `context_window` explicitly).

## Layout

`cc_codex_router/`: `config.py`, `translate.py`, `clamp.py`, `sanitize.py`, `stream.py`, `responses.py` (Codex's Responses dialect), `app.py` (router + proxy), `teamgen.py` (config -> `agents/*-mate.md`), `fanout.py` (cross-platform parallel dispatch CLI). `tests/`: stdlib fake upstream, one test per kept fix.
