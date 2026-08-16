"""The import tools end-to-end against the fake stores — dry-run discipline,
apply paths through faked host modules, and the license refusal."""

from __future__ import annotations

import json
import secrets
import sys
import types
from pathlib import Path

import pytest

from tests.conftest import PROJECT_DIR, SECRET


@pytest.fixture
def import_tools(fake_home):
    from claude_bridge.tools_import import build_import_tools

    return {t.name: t for t in build_import_tools(fake_home)}


@pytest.fixture
def fake_host(monkeypatch, tmp_path):
    """Fake every host seam the importer touches; return the capture dict."""
    captured: dict = {"applied": []}

    target = tmp_path / "user-skills"
    target.mkdir()
    captured["skills_root"] = target
    monkeypatch.setitem(sys.modules, "infra.paths", types.ModuleType("infra.paths"))
    sys.modules["infra.paths"].user_skills_dir = lambda: target
    monkeypatch.setitem(sys.modules, "infra", types.ModuleType("infra"))

    state_mod = types.ModuleType("runtime.state")
    state_mod.STATE = types.SimpleNamespace(skills_index=None)
    monkeypatch.setitem(sys.modules, "runtime.state", state_mod)
    monkeypatch.setitem(sys.modules, "runtime", types.ModuleType("runtime"))

    host_mod = types.ModuleType("graph.plugins.host")

    class _Host:
        config = staticmethod(
            lambda: types.SimpleNamespace(
                mcp_servers=[{"name": "existing", "transport": "stdio", "command": "x"}],
                plugin_config={"claude_bridge": {}},
            )
        )

        @staticmethod
        def apply_settings(patch):
            captured["applied"].append(patch)
            return True, ["reloaded"]

    host_mod.HOST = _Host
    monkeypatch.setitem(sys.modules, "graph.plugins.host", host_mod)

    sdk_mod = types.ModuleType("graph.sdk")

    async def _knowledge_add(content, *, domain="general", heading=None, epoch=None):
        captured.setdefault("chunks", []).append((domain, heading, content))
        return len(captured["chunks"])

    sdk_mod.knowledge_add = _knowledge_add
    monkeypatch.setitem(sys.modules, "graph.sdk", sdk_mod)
    monkeypatch.setitem(sys.modules, "graph", types.ModuleType("graph"))
    monkeypatch.setitem(sys.modules, "graph.plugins", types.ModuleType("graph.plugins"))
    return captured


def test_scan_finds_everything_and_excludes_anthropic(import_tools):
    out = import_tools["claude_import_scan"].invoke({"project_dir": PROJECT_DIR})
    assert "demo-skill" in out
    assert "my-writing-style" in out
    assert "docx" in out and "excluded (Anthropic-licensed" in out
    assert "standup" in out and "reviewer" in out and "github" in out
    assert "memory: 1 topic" in out


async def test_skill_import_refuses_symlinked_source_escape(fake_home, fake_host, tmp_path):
    from claude_bridge.tools_import import build_import_tools

    marker = secrets.token_urlsafe(24)
    outside = tmp_path / "outside-skill"
    outside.mkdir()
    (outside / "SKILL.md").write_text(f"---\nname: outside\ndescription: {marker}\n---\n\nOutside.\n")
    skills = tmp_path / "dot-claude" / "skills"
    (skills / "linked").symlink_to(outside, target_is_directory=True)
    tool = {item.name: item for item in build_import_tools(fake_home)}["claude_import_skills"]

    out = await tool.ainvoke({"names": "all", "source": "user", "apply": True})

    assert marker not in out
    assert not (fake_host["skills_root"] / "outside").exists()


async def test_skill_import_honors_configured_read_cap(fake_home, fake_host, tmp_path):
    from claude_bridge.tools_import import build_import_tools

    marker = secrets.token_urlsafe(24)
    skill_md = tmp_path / "dot-claude" / "skills" / "demo-skill" / "SKILL.md"
    skill_md.write_text("---\nname: demo-skill\ndescription: Demo.\n---\n\n" + marker + "\n")
    capped = dict(fake_home, max_read_bytes=16)
    tool = {item.name: item for item in build_import_tools(capped)}["claude_import_skills"]

    out = await tool.ainvoke({"names": "demo-skill", "source": "user", "apply": True})

    assert marker not in out
    assert not (fake_host["skills_root"] / "demo-skill").exists()
    assert "max_read_bytes" in out


async def test_skills_dry_run_writes_nothing(import_tools, fake_host):
    out = await import_tools["claude_import_skills"].ainvoke({"names": "all", "source": "cowork"})
    assert "DRY RUN" in out and "my-writing-style" in out
    assert not list(fake_host["skills_root"].iterdir())


async def test_skills_apply_imports_user_authored_only(import_tools, fake_host):
    out = await import_tools["claude_import_skills"].ainvoke({"names": "all", "source": "cowork", "apply": True})
    assert "imported skill 'my-writing-style'" in out
    assert (fake_host["skills_root"] / "my-writing-style" / "SKILL.md").is_file()
    # the Anthropic-licensed cowork skill never entered the candidate list
    assert "docx" not in out
    assert not (fake_host["skills_root"] / "docx").exists()


async def test_skills_apply_never_overwrites(import_tools, fake_host):
    (fake_host["skills_root"] / "my-writing-style").mkdir()
    out = await import_tools["claude_import_skills"].ainvoke({"names": "all", "source": "cowork", "apply": True})
    assert "skipped 'my-writing-style'" in out


async def test_commands_become_slash_skills(import_tools, fake_host):
    out = await import_tools["claude_import_commands"].ainvoke({"names": "standup", "apply": True})
    assert "imported skill 'standup'" in out
    text = (fake_host["skills_root"] / "standup" / "SKILL.md").read_text()
    assert "slash: standup" in text and "user_facing: true" in text


async def test_subagents_persist_via_plugin_config(import_tools, fake_host):
    out = await import_tools["claude_import_subagents"].ainvoke({"names": "reviewer", "apply": True})
    assert "reviewer" in out and "next config reload" in out
    patch = fake_host["applied"][-1]
    entry = patch["claude_bridge"]["imported_subagents"][0]
    assert entry["name"] == "reviewer"
    assert entry["tools"] == ["read_file", "run_command", "search_files"]


async def test_mcp_dry_run_redacts_and_apply_merges(import_tools, fake_host):
    dry = await import_tools["claude_import_mcp"].ainvoke({"names": "github"})
    assert "DRY RUN" in dry and SECRET not in dry and "env keys=['GH_TOKEN']" in dry

    out = await import_tools["claude_import_mcp"].ainvoke({"names": "github", "apply": True})
    assert "applied and reloaded" in out
    servers = fake_host["applied"][-1]["mcp"]["servers"]
    names = [s["name"] for s in servers]
    assert "existing" in names and "github" in names  # merge, not replace-all


async def test_mcp_reports_strip_url_credentials_without_changing_applied_config(import_tools, fake_home, fake_host):
    marker = secrets.token_urlsafe(24)
    full_url = f"https://person:{marker}@example.test/mcp?token={marker}#fragment-{marker}"
    settings = Path(fake_home["cli_root"]) / "settings.json"
    data = json.loads(settings.read_text())
    data["mcpServers"]["remote"] = {"type": "http", "url": full_url}
    settings.write_text(json.dumps(data))

    dry = await import_tools["claude_import_mcp"].ainvoke({"names": "remote"})
    applied = await import_tools["claude_import_mcp"].ainvoke({"names": "remote", "apply": True})
    configured = next(row for row in fake_host["applied"][-1]["mcp"]["servers"] if row["name"] == "remote")

    assert marker not in dry
    assert marker not in applied
    assert "https://example.test/mcp" in dry
    assert configured["url"] == full_url


async def test_mcp_reports_hide_stdio_argument_values_without_changing_applied_config(
    import_tools, fake_home, fake_host
):
    marker = secrets.token_urlsafe(24)
    full_args = ["serve", "--token", marker]
    settings = Path(fake_home["cli_root"]) / "settings.json"
    data = json.loads(settings.read_text())
    data["mcpServers"]["argumented"] = {"command": "example-mcp", "args": full_args}
    settings.write_text(json.dumps(data))

    dry = await import_tools["claude_import_mcp"].ainvoke({"names": "argumented"})
    applied = await import_tools["claude_import_mcp"].ainvoke({"names": "argumented", "apply": True})
    configured = next(row for row in fake_host["applied"][-1]["mcp"]["servers"] if row["name"] == "argumented")

    assert marker not in dry
    assert marker not in applied
    assert "args=3" in dry
    assert configured["args"] == full_args


async def test_memory_import_ingests_with_provenance(import_tools, fake_host):
    dry = await import_tools["claude_import_memory"].ainvoke({"directory": PROJECT_DIR})
    assert "DRY RUN" in dry and "1 topic" in dry
    out = await import_tools["claude_import_memory"].ainvoke({"directory": PROJECT_DIR, "apply": True})
    assert "ingested 1/1" in out
    domain, heading, content = fake_host["chunks"][0]
    assert domain == "claude-import" and "imported from claude-code" in content


async def test_claude_md_import_ingests_operating_instructions(import_tools, fake_host, tmp_path):
    # CLAUDE.md lives at the REPO root (a real file), not under ~/.claude — read from tmp_path.
    (tmp_path / "CLAUDE.md").write_text(
        "# CLAUDE.md\n\nRun `python -m server`. Pre-PR gate: `ruff check .`.", encoding="utf-8"
    )
    d = str(tmp_path)
    dry = await import_tools["claude_import_claude_md"].ainvoke({"directory": d})
    assert "DRY RUN" in dry and "CLAUDE.md" in dry
    out = await import_tools["claude_import_claude_md"].ainvoke({"directory": d, "apply": True})
    assert "ingested CLAUDE.md" in out
    domain, heading, content = fake_host["chunks"][0]
    assert domain == "claude-import"
    assert "Operating instructions" in heading
    assert "python -m server" in content and "imported from claude-code CLAUDE.md" in content


async def test_claude_md_missing_is_a_clean_noop(import_tools, fake_host, tmp_path):
    out = await import_tools["claude_import_claude_md"].ainvoke({"directory": str(tmp_path)})
    assert "no CLAUDE.md" in out
    assert not fake_host.get("chunks")


async def test_memory_import_default_limit_is_unbounded(import_tools, fake_host):
    # Default limit 0 imports every topic (regression: the old default capped at 100).
    out = await import_tools["claude_import_memory"].ainvoke({"directory": PROJECT_DIR, "apply": True})
    assert "ingested 1/1" in out  # the fixture memory has 1 topic; none dropped by a cap


def test_hooks_are_report_only(import_tools):
    out = import_tools["claude_hooks_report"].invoke({})
    assert "PreToolUse" in out and "NOT translated" in out


def test_imported_subagents_register_on_load(registry):
    import claude_bridge

    sub_mod = types.ModuleType("graph.subagents.config")

    class SubagentConfig:
        def __init__(self, **kw):
            self.__dict__.update(kw)

    sub_mod.SubagentConfig = SubagentConfig
    sys.modules["graph.subagents.config"] = sub_mod
    sys.modules.setdefault("graph.subagents", types.ModuleType("graph.subagents"))
    sys.modules.setdefault("graph", types.ModuleType("graph"))

    captured = []
    registry.register_subagent = lambda cfg: captured.append(cfg)
    registry.config = {
        "imported_subagents": [{"name": "reviewer", "description": "d", "system_prompt": "p", "tools": []}]
    }
    try:
        claude_bridge.register(registry)
    finally:
        for mod in ("graph.subagents.config", "graph.subagents", "graph"):
            sys.modules.pop(mod, None)
    assert captured and captured[0].name == "reviewer"
