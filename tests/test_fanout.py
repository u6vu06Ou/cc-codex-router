"""fanout: parallel dispatch, slug collision safety, min-chars retry."""
import json
import sys
from pathlib import Path

from cc_codex_router.fanout import (extract_answer, main as fanout_main,
                                       slugify)

ECHO_SHIM = r'''
import json, sys
prompt = sys.stdin.read()
# long on retry (env-counted), short on first try when told to be
import os
counter = os.environ.get("FANOUT_SHIM_COUNTER", "")
n = 0
if counter:
    try:
        n = int(open(counter).read() or 0)
    except OSError:
        n = 0
    with open(counter, "w") as fh:
        fh.write(str(n + 1))
short_mode = os.environ.get("FANOUT_SHIM_SHORT_FIRST") == "1"
if short_mode and n == 0:
    print("tiny")
else:
    print(json.dumps({"result": "A" * 500}))
'''


def shim_cmd(tmp_path, short_first=False):
    shim = tmp_path / "shim.py"
    shim.write_text(ECHO_SHIM, encoding="utf-8")
    return f'"{sys.executable}" "{shim}" {{model}}'


def test_slug_collision_safe():
    used = set()
    a = "the quick brown fox jumps over the lazy dog A"
    b = "the quick brown fox jumps over the lazy dog B"
    s1, s2 = slugify(a, used), slugify(b, used)
    assert s1 != s2  # shared 24-char prefix must not overwrite
    assert s1 == slugify(a, set())  # deterministic


def test_extract_answer_json_and_raw():
    assert extract_answer('{"result": "hi"}') == "hi"
    assert extract_answer("plain text") == "plain text"


def test_fanout_run_and_status(tmp_path):
    items = tmp_path / "items.txt"
    items.write_text("task alpha one\ntask beta two\n", encoding="utf-8")
    out = tmp_path / "run1"
    rc = fanout_main(["run", "--task", "Do: {item}",
                      "--items", str(items), "--out", str(out),
                      "--cmd", shim_cmd(tmp_path), "--model", "main",
                      "--parallel", "2", "--min-chars", "100",
                      "--retries", "1", "--timeout", "60"])
    assert rc == 0
    statuses = sorted(out.glob("*.status"))
    assert len(statuses) == 2
    for s in statuses:
        row = json.loads(s.read_text(encoding="utf-8"))
        assert row["ok"] and row["chars"] >= 100
    rc = fanout_main(["status", "--out", str(out)])
    assert rc == 0


def test_fanout_retries_short_answers(tmp_path, monkeypatch):
    counter = tmp_path / "counter.txt"
    monkeypatch.setenv("FANOUT_SHIM_SHORT_FIRST", "1")
    items = tmp_path / "items.txt"
    items.write_text("only item\n", encoding="utf-8")
    out = tmp_path / "run2"
    rc = fanout_main(["run", "--task", "go",
                      "--items", str(items), "--out", str(out),
                      "--cmd", shim_cmd(tmp_path), "--model", "main",
                      "--min-chars", "100", "--retries", "2",
                      "--timeout", "60",
                      "--env", f"FANOUT_SHIM_COUNTER={counter}"])
    assert rc == 0
    row = json.loads(next(out.glob("*.status")).read_text(encoding="utf-8"))
    assert row["ok"] and row["attempt"] == 2  # tiny first try was retried
