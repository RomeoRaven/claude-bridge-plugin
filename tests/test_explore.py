"""Each explore tool exercised against the fake three-store home."""

from __future__ import annotations

import secrets
from pathlib import Path

from tests.conftest import COWORK_SESSION, PROJECT_DIR, SECRET, SESSION, SLUG


def test_projects_lists_and_resolves(tools):
    listing = tools["claude_projects"].invoke({"query": ""})
    assert SLUG in listing and "memory=yes" in listing

    resolved = tools["claude_projects"].invoke({"query": PROJECT_DIR})
    assert f"project: {SLUG}" in resolved
    assert "sessions: 1" in resolved
    assert "a-fact.md" in resolved


def test_projects_handles_unknown_dir(tools):
    out = tools["claude_projects"].invoke({"query": "/no/such/dir"})
    assert out.startswith("error:")


def test_memory_index_and_file(tools):
    index = tools["claude_memory"].invoke({"directory": PROJECT_DIR})
    assert "# Memory index" in index and "a-fact.md" in index

    fact = tools["claude_memory"].invoke({"directory": PROJECT_DIR, "file": "a-fact.md"})
    assert "The fact body." in fact


def test_memory_file_cannot_escape(tools):
    out = tools["claude_memory"].invoke({"directory": PROJECT_DIR, "file": "../../../settings.json"})
    assert out.startswith("error:")


def test_sessions_list_and_tail(tools):
    listing = tools["claude_sessions"].invoke({"directory": PROJECT_DIR})
    assert SESSION in listing

    tail = tools["claude_sessions"].invoke({"directory": PROJECT_DIR, "session_id": SESSION})
    assert "hello there" in tail
    assert "hi — answering" in tail
    assert "[tool_use: Bash]" in tail


def test_scratchpad_list_files_read(tools):
    sessions = tools["claude_scratchpad"].invoke({"directory": PROJECT_DIR})
    assert SESSION in sessions

    files = tools["claude_scratchpad"].invoke({"directory": PROJECT_DIR, "session_id": SESSION})
    assert "notes.md" in files

    body = tools["claude_scratchpad"].invoke({"directory": PROJECT_DIR, "session_id": SESSION, "path": "notes.md"})
    assert "scratch notes content" in body


def test_cowork_list_detail_read(tools):
    listing = tools["claude_cowork"].invoke({})
    assert COWORK_SESSION in listing and "organize my files" in listing

    detail = tools["claude_cowork"].invoke({"session_id": COWORK_SESSION})
    assert "claude-opus-4-7" in detail
    assert "outputs/report.md" in detail
    assert "audit tail:" in detail

    body = tools["claude_cowork"].invoke({"session_id": COWORK_SESSION, "path": "outputs/report.md"})
    assert "cowork report body" in body


def test_inventory_lists_and_redacts_secrets(tools):
    out = tools["claude_inventory"].invoke({})
    assert "demo-skill" in out
    assert "helper" in out
    assert "some-marketplace/tool-pack" in out
    assert "github" in out and "GH_TOKEN" in out
    assert SECRET not in out, "MCP env values must never be echoed"


def test_inventory_honors_configured_read_cap_for_settings(fake_home):
    from claude_bridge.explore import build_explore_tools

    marker = secrets.token_urlsafe(24)
    settings = Path(fake_home["cli_root"]) / "settings.json"
    settings.write_text('{"model": "' + marker + '"}')
    capped = dict(fake_home, max_read_bytes=16)
    inventory = {tool.name: tool for tool in build_explore_tools(capped)}["claude_inventory"]

    out = inventory.invoke({})

    assert marker not in out
    assert "truncated" in out


def test_inventory_refuses_symlinked_skill_escape(tools, tmp_path):
    marker = secrets.token_urlsafe(24)
    project = tmp_path / "project"
    skills = project / ".claude" / "skills"
    outside = tmp_path / "outside-skill"
    skills.mkdir(parents=True)
    outside.mkdir()
    (outside / "SKILL.md").write_text(f"---\nname: outside\ndescription: {marker}\n---\n\nOutside.\n")
    (skills / "linked").symlink_to(outside, target_is_directory=True)

    out = tools["claude_inventory"].invoke({"project_dir": str(project)})

    assert marker not in out
    assert "outside" not in out


def test_inventory_reports_symlinked_user_skills_root(fake_home, tmp_path):
    from claude_bridge.explore import build_explore_tools

    skills = Path(fake_home["cli_root"]) / "skills"
    skills.rename(Path(fake_home["cli_root"]) / "skills-original")
    outside = tmp_path / "outside-skills"
    outside.mkdir()
    skills.symlink_to(outside, target_is_directory=True)
    inventory = {tool.name: tool for tool in build_explore_tools(fake_home)}["claude_inventory"]

    out = inventory.invoke({})

    assert "user skills" in out
    assert "outside declared root" in out


def test_inventory_project_level(tools, tmp_path):
    marker = secrets.token_urlsafe(24)
    proj = tmp_path / "someproj"
    (proj / ".claude" / "agents").mkdir(parents=True)
    (proj / ".claude" / "agents" / "local.md").write_text(
        "---\nname: local\ndescription: Project-scoped agent.\n---\n\nPrompt.\n"
    )
    (proj / ".claude" / "settings.json").write_text('{"hooks": {"PreToolUse": []}}')
    (proj / ".mcp.json").write_text(
        '{"mcpServers": {"docs": {"url": '
        f'"https://person:{marker}@x.test/mcp?opaque={marker}#fragment-{marker}"'
        "}}}"
    )
    (proj / "CLAUDE.md").write_text("# My project\n")

    out = tools["claude_inventory"].invoke({"project_dir": str(proj)})
    assert "local" in out
    assert "PreToolUse" in out
    assert "docs" in out and marker not in out
    assert "url=https://x.test/mcp" in out
    assert "CLAUDE.md: present" in out


def test_inventory_accepts_relative_project_dir_without_losing_user_sections(tools, tmp_path, monkeypatch):
    proj = tmp_path / "relative-project"
    (proj / ".claude" / "agents").mkdir(parents=True)
    (proj / ".claude" / "agents" / "local.md").write_text(
        "---\nname: local\ndescription: Relative project agent.\n---\n\nPrompt.\n"
    )
    monkeypatch.chdir(proj)

    out = tools["claude_inventory"].invoke({"project_dir": "."})

    assert not out.startswith("error:")
    assert "demo-skill" in out
    assert "local" in out
