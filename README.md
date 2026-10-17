# claude-team-router

One local endpoint that lets Claude Code run *mixed* agent teams: the lead
stays on your official Anthropic account, and teammates run on any
OpenAI-compatible model API you already have — self-hosted or hosted.

It answers `/v1/messages` like the Anthropic API does, then routes by model
name: anything starting with `claude-` (or empty) is forwarded untouched to
the real Anthropic API; every other model is translated to OpenAI
chat-completions and sent to the backend you configured for it.

## What you get

- **Team routing** — `ANTHROPIC_BASE_URL=http://127.0.0.1:4003` once; the lead
  lane passes through to Anthropic with auth intact, teammates go to your own
  models via the built-in translator.
- **Battle-tested hardening** — stream fixes (opening deltas never eaten,
  usage pass-through), thinking-history preservation, context clamping with
  pinned task + boundary drops, failover chains with cooldown, retries, and
  honest markers instead of silent truncation.
- **teamgen** — generates one `agents/<name>-mate.md` per model in your
  config, ready for Claude Code agent teams.
- **fanout** — run one prompt over an item list across models in parallel,
  with retries and per-item output files.

## Quick start

```bash
pip install ".[yaml]"
cp config.example.yaml config.yaml   # edit: your backends + models
export LOCAL_BACKEND_API_KEY=...     # only if your config NAMES this var
claude-team-router serve --config config.yaml
claude-team-router teamgen --config config.yaml --out-dir ~/.claude/agents
```

Then start Claude Code with `ANTHROPIC_BASE_URL=http://127.0.0.1:4003` and
use `/model main` (your lead) plus the generated `*-mate` teammates.

Parallel batch work: `claude-team-router fanout run --task "Review: {item}"
--items items.txt --out runs/r1 --model my-workhorse`, then read the results
with `claude-team-router fanout status --out runs/r1`.

## Common backends

All backends are plain OpenAI-compatible endpoints. The router always POSTs
to `{base_url}/chat/completions` with `Authorization: Bearer <key>`; a
backend with no `api_key_env` set (or the variable unset) sends no auth
header — which is what local servers want.

**Ollama** — serve any pulled model, no key needed:

```yaml
backends:
  ollama:
    base_url: "http://localhost:11434/v1"

models:
  local-coder:
    backend: ollama
    upstream_model: "qwen2.5:14b"   # names as shown by `ollama list`
```

**LM Studio** — start the developer server, then load models into it:

```yaml
backends:
  lmstudio:
    base_url: "http://localhost:1234/v1"

models:
  local-fast:
    backend: lmstudio
    upstream_model: "qwen2.5-7b-instruct"   # names as shown in the server tab
```

**vLLM** — `vllm serve <model>`; the served model name is the one you passed:

```yaml
backends:
  vllm:
    base_url: "http://127.0.0.1:8000/v1"   # default host/port for vllm serve

models:
  vllm-big:
    backend: vllm
    upstream_model: "Qwen/Qwen2.5-14B-Instruct"  # exact served-model name
    # vllm serve --api-key ...? then: api_key_env: "VLLM_API_KEY"
```

**OpenRouter** — one key, many hosted models:

```yaml
backends:
  openrouter:
    base_url: "https://openrouter.ai/api/v1"
    api_key_env: "OPENROUTER_API_KEY"
    # the two attribution headers below are optional and publicly documented
    extra_headers:
      HTTP-Referer: "https://your-app.example.com"
      X-Title: "your-app"

models:
  any-hosted:
    backend: openrouter
    upstream_model: "deepseek/deepseek-chat"   # format: vendor/model
```

**Generic cloud OpenAI-compatible API** — most providers follow the same
shape; store the key in an environment variable and name the variable, never
the key:

```yaml
backends:
  cloud:
    base_url: "https://api.your-provider.com/v1"   # check the provider docs
    api_key_env: "CLOUD_API_KEY"

models:
  cloud-model:
    backend: cloud
    upstream_model: "the-name-your-provider-expects"
```

`api_key_env` holds the *name* of an environment variable — create it once
with `export CLOUD_API_KEY=...` (or your shell's equivalent). Never put the
key itself in the config.

## How this differs from other tools

Routing Claude Code between Claude and non-Claude models is a solved problem
in several mature projects: [claude-code-router](https://github.com/musistudio/claude-code-router),
[LiteLLM](https://docs.litellm.ai/), and OpenRouter's Anthropic-compatible
endpoint all do model routing, with far more providers, throughput, and
observability than this repo. If you need that breadth, they are the
established tools; this project is not trying to replace them.

What this repo adds is a narrow one-stop setup:

- one YAML describes everything — the lead's passthrough plus every teammate
  backend and model — instead of configuring the gateway and the model list
  separately;
- a zero-config routing convention instead of rule sets: a model name
  starting with `claude-` goes to the official API untouched, everything else
  routes by config lookup;
- **teamgen**, which turns that same config into Claude Code teammate
  definition files, so teams don't hand-write `agents/*.md`;
- **fanout**, a standalone parallel dispatcher (one task × item list across
  models, per-item output files).

**Risk note:** Anthropic does not officially support routing Claude Code to
non-Claude models. This works today because Claude Code only speaks HTTP
over `ANTHROPIC_BASE_URL`; treat it as your own risk, not an endorsed
setup, and expect no guarantees if that behavior ever changes.

## Notes

- Config field reference lives in `config.example.yaml`; design decisions in
  `docs/DESIGN.md`. Keys are never stored in the config — only env var names.
- The pre-commit hook runs an OSS-sensitive-content scan; hits are blocked.
  `.oss-scan-allow` (not versioned) lists scan exceptions.

## 繁中速覽

把這個 proxy 裝在本機，`ANTHROPIC_BASE_URL` 指過來：lead 用你官方
Anthropic 帳號直連，隊員走你設定的任何 OpenAI 相容後端（自架或買的）。
`config.example.yaml` 改模型清單→ `teamgen` 生隊員定義 → `fanout`
平行跑批次。常見後端（Ollama、LM Studio、vLLM、OpenRouter）的填法見
上方「Common backends」一節；與其他路由工具的差別見
「How this differs from other tools」。詳見 `skills/setup/SKILL.md`。

## License

MIT — see [LICENSE](LICENSE).
