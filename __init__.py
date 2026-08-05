"""claude_bridge — explore Claude Code's on-disk state from protoAgent.

Read-only tools over three stores (the ~/.claude CLI store, per-session
scratchpads, and Cowork/desktop local-agent-mode sessions), so the agent can
answer "what does Claude Code know about this directory?" — memory, sessions,
scratchpads, and the installed skill/agent/plugin/MCP inventory.

Host-only imports stay lazy so the test suite runs with no protoAgent host.
"""

from __future__ import annotations

import logging

log = logging.getLogger("protoagent.plugins.claude_bridge")


def register(registry) -> None:
    cfg = dict(registry.config or {})

    try:
        from .explore import build_explore_tools

        registry.register_tools(build_explore_tools(cfg))
    except Exception:  # noqa: BLE001 — one failing group must not sink the rest
        log.exception("[claude_bridge] registering explore tools failed")

    try:
        from .tools_import import build_import_tools

        registry.register_tools(build_import_tools(cfg))
    except Exception:  # noqa: BLE001
        log.exception("[claude_bridge] registering import tools failed")

    try:
        from .tools_export import build_export_tools

        registry.register_tools(build_export_tools(cfg))
    except Exception:  # noqa: BLE001
        log.exception("[claude_bridge] registering export tools failed")

    try:
        from .importer import register_imported_subagents

        n = register_imported_subagents(registry, cfg)
        if n:
            log.info("[claude_bridge] registered %d imported subagent(s)", n)
    except Exception:  # noqa: BLE001
        log.exception("[claude_bridge] registering imported subagents failed")

    try:
        registry.register_skill_dir("skills")
    except Exception:  # noqa: BLE001
        log.exception("[claude_bridge] registering skills failed")

    log.info("[claude_bridge] registered")
