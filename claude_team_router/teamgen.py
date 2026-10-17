"""teamgen: generate Claude Code teammate definitions from your config.

Each alias in the config's `models:` table becomes one `agents/<alias>-mate.md`
file (frontmatter: name / description / model). The description is left for
you to fill in — the generator cannot know what each of your models is good
at; the body is a short sane-default teammate brief.
"""
from __future__ import annotations

import re
from pathlib import Path

from .config import Config, load_config

TEMPLATE = """---
name: {name}
description: TODO — say what this model is good at and when to call it. \
Model alias: {alias} (via the team router).
model: {alias}
---

You are a teammate in a Claude Code agent team, running on the model alias
`{alias}` through the team router.

- Stay inside the task you were given; report back with concrete results.
- Verify your own work before reporting done; never claim success you did not observe.
- If you are blocked, say what is missing instead of guessing.
"""


def _agent_name(alias: str) -> str:
    base = re.sub(r"[^A-Za-z0-9]+", "-", alias).strip("-").lower()
    return (base or "model") + "-mate"


def generate(cfg: Config, out_dir: str, overwrite: bool = False) -> list:
    """Write one agents/<alias>-mate.md per configured model. Returns file paths."""
    out = []
    target = Path(out_dir)
    target.mkdir(parents=True, exist_ok=True)
    for alias in sorted(cfg.models):
        path = target / (_agent_name(alias) + ".md")
        if path.exists() and not overwrite:
            out.append(f"SKIP (exists): {path}")
            continue
        path.write_text(TEMPLATE.format(name=_agent_name(alias), alias=alias),
                        encoding="utf-8", newline="\n")
        out.append(str(path))
    return out


def main(argv=None):
    import argparse

    ap = argparse.ArgumentParser(
        prog="teamgen",
        description="Generate Claude Code teammate definitions from config models")
    ap.add_argument("--config", required=True, help="router config (YAML/JSON)")
    ap.add_argument("--out-dir", required=True,
                    help="e.g. the agents/ directory of your Claude config")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args(argv)

    cfg = load_config(args.config)
    for line in generate(cfg, args.out_dir, overwrite=args.overwrite):
        print(line)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
