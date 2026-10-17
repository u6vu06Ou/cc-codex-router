---
name: setup
description: Install and wire cc-codex-router so Claude Code can split an agent team between the official Anthropic API and your own OpenAI-compatible model backends.
---

# cc-codex-router setup

Goal: one local endpoint, mixed team. Lead = your official Anthropic account,
teammates = model APIs you already have (self-hosted or hosted).

## 1. Install

```bash
pip install .[yaml]        # from the repo root; yaml extra is optional but
                           # recommended (JSON config works without it)
```

## 2. Connect your own API

1. `cp config.example.yaml config.yaml`
2. Under `backends:` add one entry per model API: `base_url` plus
   `api_key_env` — the NAME of an environment variable that holds the key.
   The key itself never goes in the config.
3. Under `models:` give each model an alias (this is the `model` name Claude
   Code will send), its `backend`, the `upstream_model` name the backend
   expects, and `context_window`. Set `vision: true` only if the model
   actually reads images, `reasoning: native` for reasoning-content models.
4. Export the env var: `export LOCAL_BACKEND_API_KEY="..."` (or set it in
   your service manager).

## 3. Start

```bash
cc-codex-router serve --config config.yaml
# listening on 127.0.0.1:4003 by default; verify:
# curl http://127.0.0.1:4003/health   # -> {"ok": true, "models": [...]}
```

## 4. Point Claude Code at the proxy

```bash
export ANTHROPIC_BASE_URL=http://127.0.0.1:4003
claude
```

- Any `claude-*` model (or empty) passes through to
  `https://api.anthropic.com` with your usual auth — nothing changes for the
  lead.
- Any other model name is looked up in your config and proxied through the
  translator to your backend.

## 5. Generate teammates

```bash
cc-codex-router teamgen --config config.yaml --out ~/.claude/agents
```

One `<alias>-mate.md` per model, with a `description: TODO` for you to fill
in (Claude Code shows it when picking agents). Existing files are kept;
`--overwrite` regenerates.

## 6. Verify

```bash
curl -s http://127.0.0.1:4003/health
curl -s http://127.0.0.1:4003/v1/models | head
curl -s -X POST http://127.0.0.1:4003/v1/messages/count_tokens \
  -H 'content-type: application/json' \
  -d '{"model": "<your-alias>", "messages": [{"role": "user", "content": "hi"}]}'
```

Then in Claude Code: switch a teammate to the alias and give it a tiny task.
Watch the proxy log — you should see the request land on your backend.

## Troubleshooting

- `defs not found / 404` from the proxy: the model name is not in `models:`.
- Backend auth errors: the env var named by `api_key_env` is unset in the
  proxy's environment (export it in the same shell / service unit).
- Empty replies from reasoning models: the router retries with lowered
  effort and marks the reply; check `reasoning_effort` in config.
