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
pip install .[yaml]
cp config.example.yaml config.yaml   # edit: your backends + models
export LOCAL_BACKEND_API_KEY=...     # the env var your config NAMES
claude-team-router serve --config config.yaml
claude-team-router teamgen --config config.yaml --out ~/.claude/agents
```

Then start Claude Code with `ANTHROPIC_BASE_URL=http://127.0.0.1:4003` and
use `/model main` (your lead) plus the generated `*-mate` teammates.

Parallel batch work: `claude-team-router fanout run --task "Review: {item}"
--items items.txt --out runs/r1 --model my-workhorse`. See `fanout status`.

## Notes

- Config field reference lives in `config.example.yaml`; design decisions in
  `docs/DESIGN.md`. Keys are never stored in the config — only env var names.
- The pre-commit hook runs an OSS-sensitive-content scan; hits are blocked.
  `.oss-scan-allow` (not versioned) lists scan exceptions.

## 繁中速覽

把這個 proxy 裝在本機，`ANTHROPIC_BASE_URL` 指過來：lead 用你官方
Anthropic 帳號直連，隊員走你設定的任何 OpenAI 相容後端（自架或買的）。
`config.example.yaml` 改模型清單→ `teamgen` 生隊員定義 → `fanout`
平行跑批次。詳見 `skills/setup/SKILL.md`。

## License

MIT — see [LICENSE](LICENSE).
