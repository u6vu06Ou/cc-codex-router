"""teamgen: config model list -> Claude Code agents/*.md definitions."""
from pathlib import Path

from claude_team_router.config import BackendCfg, Config, ModelCfg
from claude_team_router.teamgen import _agent_name, generate


def cfg():
    c = Config()
    c.backends["b"] = BackendCfg(name="b", base_url="http://127.0.0.1:9/v1")
    c.models["alpha-fast"] = ModelCfg(alias="alpha-fast", backend="b")
    c.models["beta.deep.v2"] = ModelCfg(alias="beta.deep.v2", backend="b")
    return c


def test_agent_names_sanitized():
    assert _agent_name("beta.deep.v2") == "beta-deep-v2-mate"
    assert _agent_name("___") == "model-mate"


def test_one_file_per_model_with_frontmatter(tmp_path):
    written = generate(cfg(), str(tmp_path))
    assert len(written) == 2
    for p in written:
        text = Path(p).read_text(encoding="utf-8")
        assert text.startswith("---")
        assert "description: TODO" in text  # left for the user to fill in
    alpha = (tmp_path / "alpha-fast-mate.md").read_text(encoding="utf-8")
    assert "model: alpha-fast" in alpha
    assert "name: alpha-fast-mate" in alpha


def test_existing_files_not_overwritten(tmp_path):
    generate(cfg(), str(tmp_path))
    p = tmp_path / "alpha-fast-mate.md"
    p.write_text("customized", encoding="utf-8")
    out = generate(cfg(), str(tmp_path))
    assert any("SKIP" in line for line in out)
    assert p.read_text(encoding="utf-8") == "customized"
    generate(cfg(), str(tmp_path), overwrite=True)
    assert "customized" not in p.read_text(encoding="utf-8")
