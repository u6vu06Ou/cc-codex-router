"""fanout: run one task over many items in parallel subprocesses (any OS).

Each item gets its own slug directory entry in --out:
  <slug>.prompt   the exact prompt sent
  <slug>.raw      raw process stdout
  <slug>.md       the extracted answer text
  <slug>.status   one-line JSON: ok/fail, chars, duration, retries

The worker command is a template (default: the Claude Code CLI). The prompt
is fed on stdin. `{model}` and `{slug}` are substituted into the command.

Retries: an answer shorter than --min-chars counts as a failed attempt.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import shlex
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

DEFAULT_CMD = "claude -p --output-format json --model {model}"
CONTRACT = ("\n\n---\nOutput contract: produce a substantive answer "
            "(at least {min_chars} characters). Plain text; no preamble.")


def slugify(item: str, used: set) -> str:
    base = re.sub(r"[^A-Za-z0-9]+", "-", item)[:24].strip("-").lower() or "item"
    digest = hashlib.sha1(item.encode("utf-8")).hexdigest()[:6]
    slug = f"{base}-{digest}"
    n = 2
    while slug in used:
        slug = f"{base}-{digest}-{n}"
        n += 1
    used.add(slug)
    return slug


def extract_answer(raw: str) -> str:
    """Claude CLI --output-format json carries the answer in .result."""
    try:
        data = json.loads(raw)
        if isinstance(data, dict) and isinstance(data.get("result"), str):
            return data["result"]
    except (json.JSONDecodeError, ValueError):
        pass
    return raw


def run_one(item: str, slug: str, args, task_text: str) -> dict:
    out_dir = Path(args.out)
    prompt = task_text.replace("{item}", item)
    if "{item}" not in task_text:
        prompt = task_text.rstrip() + "\n\nITEM:\n" + item
    prompt += CONTRACT.format(min_chars=args.min_chars)
    (out_dir / f"{slug}.prompt").write_text(prompt, encoding="utf-8")

    cmd = args.cmd.format(model=args.model, slug=slug)
    env = dict(os.environ)
    for kv in args.env or []:
        k, _, v = kv.partition("=")
        env[k] = v

    attempt = 0
    last = {"slug": slug, "ok": False, "chars": 0, "note": "not run"}
    while attempt < args.retries + 1:
        t0 = time.time()
        try:
            proc = subprocess.run(
                shlex.split(cmd), input=prompt, capture_output=True,
                text=True, timeout=args.timeout, env=env)
            raw = proc.stdout or ""
            (out_dir / f"{slug}.raw").write_text(raw, encoding="utf-8")
            answer = extract_answer(raw)
            note = "" if proc.returncode == 0 else f"exit {proc.returncode}"
        except subprocess.TimeoutExpired:
            answer, raw, note = "", "", f"timeout {args.timeout}s"
        except OSError as e:
            answer, raw, note = "", "", f"spawn failed: {e}"
        (out_dir / f"{slug}.md").write_text(answer, encoding="utf-8")
        ok = bool(answer) and len(answer) >= args.min_chars and not note
        last = {"slug": slug, "ok": ok, "chars": len(answer),
                "attempt": attempt + 1, "secs": round(time.time() - t0, 1),
                "note": note}
        if ok:
            break
        attempt += 1
    (out_dir / f"{slug}.status").write_text(
        json.dumps(last, ensure_ascii=False) + "\n", encoding="utf-8")
    return last


def cmd_run(args) -> int:
    task_text = Path(args.task).read_text(encoding="utf-8") \
        if Path(args.task).is_file() else args.task
    items = [ln.strip() for ln in Path(args.items).read_text(encoding="utf-8").splitlines()
             if ln.strip()]
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    used = set()
    jobs = [(item, slugify(item, used)) for item in items]
    results = []
    with ThreadPoolExecutor(max_workers=args.parallel) as pool:
        futures = [pool.submit(run_one, item, slug, args, task_text)
                   for item, slug in jobs]
        for f in futures:
            results.append(f.result())
            r = results[-1]
            print(f"[{'OK' if r['ok'] else 'FAIL'}] {r['slug']} "
                  f"chars={r['chars']} attempt={r['attempt']} {r.get('note', '')}")
    ok = sum(1 for r in results if r["ok"])
    print(f"fanout done: {ok}/{len(results)} ok; out={out_dir}")
    return 0 if ok == len(results) else 1


def cmd_status(args) -> int:
    out_dir = Path(args.out)
    rows = []
    for p in sorted(out_dir.glob("*.status")):
        try:
            rows.append(json.loads(p.read_text(encoding="utf-8")))
        except (json.JSONDecodeError, ValueError):
            rows.append({"slug": p.stem, "ok": False, "note": "unreadable status"})
    ok = sum(1 for r in rows if r.get("ok"))
    for r in rows:
        print(f"[{'OK' if r.get('ok') else 'FAIL'}] {r.get('slug')} "
              f"chars={r.get('chars', 0)} {r.get('note', '')}")
    print(f"{ok}/{len(rows)} ok")
    return 0 if rows and ok == len(rows) else 1


def main(argv=None) -> int:
    import argparse

    ap = argparse.ArgumentParser(prog="fanout",
                                 description="Parallel one-task-many-items dispatch")
    sub = ap.add_subparsers(dest="sub", required=True)

    run = sub.add_parser("run")
    run.add_argument("--task", required=True,
                     help="task text, or a path to a file containing it; "
                          "{item} marks where the item goes (appended otherwise)")
    run.add_argument("--items", required=True, help="file with one item per line")
    run.add_argument("--out", required=True, help="output directory")
    run.add_argument("--model", default="", help="model alias for {model}")
    run.add_argument("--cmd", default=DEFAULT_CMD,
                     help=f"worker command template (default: {DEFAULT_CMD!r})")
    run.add_argument("--parallel", type=int, default=4)
    run.add_argument("--min-chars", type=int, default=200)
    run.add_argument("--retries", type=int, default=2)
    run.add_argument("--timeout", type=int, default=900)
    run.add_argument("--env", action="append",
                     help="KEY=VALUE added to each worker's environment")
    run.set_defaults(func=cmd_run)

    st = sub.add_parser("status")
    st.add_argument("--out", required=True)
    st.set_defaults(func=cmd_status)

    args = ap.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
