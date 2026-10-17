"""CLI entry (`python -m claude_team_router ...` / installed script).

These run the real subcommand entry points in a subprocess — not just the
internal functions — so a missing import on any path (e.g. calling
load_config from a subcommand main) fails these tests too.
"""
import json
import socket
import subprocess
import sys
import time
from pathlib import Path

import httpx

REPO = Path(__file__).resolve().parents[1]


def run_cli(args, timeout=120):
    return subprocess.run(
        [sys.executable, "-m", "claude_team_router", *args],
        cwd=REPO, capture_output=True, text=True, timeout=timeout)


def test_top_level_help():
    res = run_cli(["--help"])
    assert res.returncode == 0
    assert "serve" in res.stdout and "teamgen" in res.stdout


def test_teamgen_cli_writes_definitions(tmp_path):
    """Travels the real serve->teamgen CLI path: parser, load_config, write."""
    cfg = REPO / "config.example.yaml"
    out = tmp_path / "agents"
    res = run_cli(["teamgen", "--config", str(cfg), "--out-dir", str(out)])
    assert res.returncode == 0, res.stderr
    assert "my-workhorse-mate.md" in res.stdout
    assert "my-thinker-mate.md" in res.stdout
    text = (out / "my-thinker-mate.md").read_text(encoding="utf-8")
    assert "model: my-thinker" in text


def test_fanout_cli_runs_against_shim(tmp_path):
    shim = tmp_path / "shim.py"
    shim.write_text(
        "import json, sys\nsys.stdin.read()\n"
        "print(json.dumps({'result': 'A' * 300}))\n", encoding="utf-8")
    items = tmp_path / "items.txt"
    items.write_text("item one\nitem two\n", encoding="utf-8")
    res = run_cli(["fanout", "run", "--task", "Do: {item}",
                   "--items", str(items), "--out", str(tmp_path / "run"),
                   "--cmd", f'"{sys.executable}" "{shim}" {{model}}',
                   "--model", "main", "--parallel", "2",
                   "--min-chars", "100", "--retries", "1",
                   "--timeout", "60"])
    assert res.returncode == 0, res.stderr
    statuses = list((tmp_path / "run").glob("*.status"))
    assert len(statuses) == 2
    assert all(json.loads(s.read_text(encoding="utf-8"))["ok"]
               for s in statuses)
    res = run_cli(["fanout", "status", "--out", str(tmp_path / "run")])
    assert res.returncode == 0
    assert "2/2 ok" in res.stdout


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def test_serve_cli_answers_health_then_stops(tmp_path):
    """Boots the actual `serve` entry, curls /health, then shuts it down."""
    port = _free_port()
    cfg = tmp_path / "server.json"
    cfg.write_text(json.dumps({
        "listen": f"127.0.0.1:{port}",
        "backends": {"fake": {"base_url": "http://127.0.0.1:1/v1"}},
        "models": {"probe-model": {"backend": "fake"}},
    }), encoding="utf-8")
    proc = subprocess.Popen(
        [sys.executable, "-m", "claude_team_router", "serve",
         "--config", str(cfg)],
        cwd=REPO, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    try:
        deadline = time.time() + 60
        health = None
        while time.time() < deadline:
            try:
                health = httpx.get(f"http://127.0.0.1:{port}/health",
                                   timeout=2.0)
                break
            except (httpx.HTTPError, OSError):
                if proc.poll() is not None:
                    break
                time.sleep(0.2)
        assert health is not None, (proc.poll(), proc.stderr.read())
        assert health.json()["ok"] is True
        assert "probe-model" in health.json()["models"]
        listed = httpx.get(f"http://127.0.0.1:{port}/v1/models", timeout=5.0)
        assert [m["id"] for m in listed.json()["data"]] == ["probe-model"]
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=15)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=15)
