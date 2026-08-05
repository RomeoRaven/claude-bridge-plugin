"""register() wires everything through the registry seam, host-free."""

from __future__ import annotations

import claude_bridge

EXPECTED_TOOLS = {
    "claude_projects",
    "claude_memory",
    "claude_sessions",
    "claude_scratchpad",
    "claude_cowork",
    "claude_inventory",
    "claude_import_scan",
    "claude_import_skills",
    "claude_import_commands",
    "claude_import_subagents",
    "claude_import_mcp",
    "claude_import_memory",
    "claude_import_claude_md",
    "claude_hooks_report",
    # v0.4 — the export half: a translated setup as an ADR 0091 agent snapshot.
    "claude_export_snapshot",
}


def test_register_contributes_tools_and_skills(registry):
    claude_bridge.register(registry)
    assert {t.name for t in registry.tools} == EXPECTED_TOOLS
    assert registry.skill_dirs == ["skills"]


def test_every_tool_has_a_description(registry):
    claude_bridge.register(registry)
    for t in registry.tools:
        assert t.description and len(t.description) > 20, f"{t.name} ships without a description"


def test_register_survives_a_broken_registry():
    class Broken:
        config = {}

        def register_tools(self, tools):
            raise RuntimeError("boom")

        def register_skill_dir(self, path):
            raise RuntimeError("boom")

    claude_bridge.register(Broken())  # must not raise
