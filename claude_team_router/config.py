"""Config loading for claude-team-router.

One YAML (or JSON) file describes backends and model aliases. API keys are
never stored in the file: a backend names the environment variable that holds
its key (`api_key_env`).

Example:

    listen: "127.0.0.1:4003"
    anthropic_upstream: "https://api.anthropic.com"

    backends:
      mine:
        base_url: "http://127.0.0.1:8000/v1"
        api_key_env: "MY_BACKEND_API_KEY"
        max_concurrent: 4

    models:
      my-workhorse:
        backend: mine
        upstream_model: "llama-3.3-70b-instruct"
        context_window: 131072
        vision: false
        reasoning: none            # or "native" for reasoning_content models
        fallbacks: [my-backup]
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field


DEFAULT_CLAMP_NOTICE = (
    "Note: earlier conversation history was trimmed to fit this model's "
    "context window. The original task (above) still applies — do not "
    "restart it; continue from the latest state."
)
WRAP_UP_PLACEHOLDER = (
    "(earlier context was trimmed; summarize what you already have, give "
    "your final answer, and stop — do NOT restart the task)"
)


@dataclass
class BackendCfg:
    name: str
    base_url: str
    api_key_env: str = ""
    max_concurrent: int = 0  # 0 = unlimited
    # Optional extra request headers for backends that need more than the
    # default Bearer auth. Each value is either a literal (use for plain,
    # non-secret headers such as an org/project id) or "env:NAME" to read the
    # value from that environment variable. Applied after the Bearer header,
    # so "Authorization" here intentionally overrides it.
    extra_headers: dict = field(default_factory=dict)

    def api_key(self) -> str:
        return os.environ.get(self.api_key_env, "") if self.api_key_env else ""

    def resolved_headers(self) -> dict:
        out = {}
        for name, value in self.extra_headers.items():
            if isinstance(value, str) and value.startswith("env:"):
                value = os.environ.get(value[4:], "")
            if value:
                out[name] = value
        return out


@dataclass
class ModelCfg:
    alias: str
    backend: str
    upstream_model: str = ""
    context_window: int = 262_144
    max_output: int = 0            # 0 = no cap
    vision: bool = False
    reasoning: str = "none"        # "none" | "native"
    reasoning_effort: str = ""     # e.g. "high"/"low"; native models only
    strict_system_position: bool = False
    max_message_chars: int = 24_000
    extra_body: dict = field(default_factory=dict)
    fallbacks: list = field(default_factory=list)  # list of aliases

    def upstream_name(self) -> str:
        return self.upstream_model or self.alias


@dataclass
class Config:
    listen: str = "127.0.0.1:4003"
    anthropic_upstream: str = ""   # empty = lead lane disabled
    claude_prefix: str = "claude-"
    retry_times: int = 2
    retry_backoff: float = 1.0     # seconds; doubled per attempt
    keepalive_secs: float = 20.0
    queued_heartbeat_secs: float = 5.0
    clamp_notice: str = DEFAULT_CLAMP_NOTICE
    backends: dict = field(default_factory=dict)   # name -> BackendCfg
    models: dict = field(default_factory=dict)     # alias -> ModelCfg

    def model(self, alias: str):
        return self.models.get(alias)


def _load_raw(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as fh:
        text = fh.read()
    if path.endswith(".json"):
        return json.loads(text)
    try:
        import yaml  # optional dependency; config files are usually YAML
    except ImportError:
        # YAML is a JSON superset — plain JSON files still load.
        return json.loads(text)
    return yaml.safe_load(text)


def load_config(path: str) -> Config:
    raw = _load_raw(path) or {}
    cfg = Config()
    cfg.listen = raw.get("listen", cfg.listen)
    cfg.anthropic_upstream = raw.get("anthropic_upstream", cfg.anthropic_upstream) or ""
    cfg.claude_prefix = raw.get("claude_prefix", cfg.claude_prefix)
    cfg.retry_times = int(raw.get("retry_times", cfg.retry_times))
    cfg.retry_backoff = float(raw.get("retry_backoff", cfg.retry_backoff))
    cfg.keepalive_secs = float(raw.get("keepalive_secs", cfg.keepalive_secs))
    cfg.clamp_notice = raw.get("clamp_notice", cfg.clamp_notice)

    for name, b in (raw.get("backends") or {}).items():
        cfg.backends[name] = BackendCfg(
            name=name,
            base_url=(b.get("base_url") or "").rstrip("/"),
            api_key_env=b.get("api_key_env", "") or "",
            max_concurrent=int(b.get("max_concurrent", 0) or 0),
            extra_headers=dict(b.get("extra_headers") or {}),
        )
    for alias, m in (raw.get("models") or {}).items():
        cfg.models[alias] = ModelCfg(
            alias=alias,
            backend=m.get("backend", ""),
            upstream_model=m.get("upstream_model", "") or "",
            context_window=int(m.get("context_window", 262_144)),
            max_output=int(m.get("max_output", 0) or 0),
            vision=bool(m.get("vision", False)),
            reasoning=m.get("reasoning", "none"),
            reasoning_effort=m.get("reasoning_effort", "") or "",
            strict_system_position=bool(m.get("strict_system_position", False)),
            max_message_chars=int(m.get("max_message_chars", 24_000)),
            extra_body=dict(m.get("extra_body") or {}),
            fallbacks=list(m.get("fallbacks") or []),
        )
    return cfg


def validate(cfg: Config) -> list:
    """Return a list of config problems (empty = OK)."""
    problems = []
    for alias, m in cfg.models.items():
        if m.backend not in cfg.backends:
            problems.append(f"model {alias!r}: unknown backend {m.backend!r}")
        for fb in m.fallbacks:
            if fb not in cfg.models:
                problems.append(f"model {alias!r}: unknown fallback {fb!r}")
            elif fb == alias:
                problems.append(f"model {alias!r}: fallback loop to itself")
    if cfg.anthropic_upstream and not cfg.anthropic_upstream.startswith("http"):
        problems.append(f"anthropic_upstream {cfg.anthropic_upstream!r} is not a URL")
    return problems
