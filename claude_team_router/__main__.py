"""CLI entry: python -m claude_team_router {serve|teamgen|fanout} ..."""
from __future__ import annotations

import sys


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or argv[0] in ("-h", "--help"):
        print(__doc__)
        print("\n  serve   --config config.yaml          start the router/proxy")
        print("  teamgen --config config.yaml --out-dir <dir>   write agents/*.md")
        print("  fanout  run|status ...                parallel task dispatch")
        return 0
    sub, rest = argv[0], argv[1:]
    if sub == "serve":
        import argparse
        from .app import serve
        ap = argparse.ArgumentParser(prog="claude-team-router serve")
        ap.add_argument("--config", required=True)
        args = ap.parse_args(rest)
        serve(args.config)
        return 0
    if sub == "teamgen":
        from .teamgen import main as teamgen_main
        return teamgen_main(rest)
    if sub == "fanout":
        from .fanout import main as fanout_main
        return fanout_main(rest)
    print(f"unknown subcommand: {sub}", file=sys.stderr)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
