"""The export tool: a translated Claude Code setup → a protoAgent agent snapshot.

The mirror of ``tools_import``. Those tools fold Claude Code state into the agent they run
inside; this one emits it as a **file** another instance can stand up — the same translators,
a different destination.

Keeps this pack's convention: **dry run by default.** `claude_export_snapshot` reports the
bundle plan and writes nothing until `apply=True`. Here that matters for a second reason —
the bundle leaves the machine, and the plan is where an operator sees which credentials were
nulled and whether a knowledge seed would ride along.
"""

from __future__ import annotations

from pathlib import Path

from langchain_core.tools import tool

from . import export_bundle as eb
from . import translate as tr
from .stores import ClaudeStores
from .tools_import import _pick, _skill_sources


def build_export_tools(cfg: dict) -> list:
    stores = ClaudeStores(cfg)

    @tool
    def claude_export_snapshot(
        project_dir: str = "",
        name: str = "claude-import",
        skills: str = "all",
        include_memory: bool = False,
        out: str = "",
        apply: bool = False,
    ) -> str:
        """Package this machine's Claude Code setup as a protoAgent **agent snapshot** —
        a portable, secret-free zip another protoAgent instance can import.

        Translates skills, slash commands, subagents, MCP servers and CLAUDE.md into ADR
        0091's snapshot format, so `protoagent agent import <file>` consumes it directly
        (there is no second format to learn). MCP credential values are **nulled** and
        listed as `required_secrets`, so the importer is prompted rather than handed a
        broken server — your tokens stay on this machine.

        Dry run by default: reports the plan and writes nothing. Pass `apply=True` to write.

        Args:
            project_dir: a Claude Code project to include (its `.claude/`, CLAUDE.md and
                memory). Blank uses user-level state only.
            name: the agent name recorded in the snapshot.
            skills: comma-separated skill names, or "all".
            include_memory: ALSO carry project memory as a knowledge seed. Off by default —
                it holds no credentials but may be private, and including it means the file
                is no longer safe to publish.
            out: where to write the zip (a directory is fine). Blank writes into the
                current working directory.
            apply: actually write the file.
        """
        proj = Path(project_dir).expanduser() if project_dir.strip() else None

        translated_skills = []
        candidates, _excluded = _skill_sources(stores, "user")
        by_name = {p.name: (p, label) for p, label in candidates}
        for picked in _pick(skills, sorted(by_name)):
            src, label = by_name[picked]
            got = tr.translate_skill_dir(src, source=label)
            if got is not None:
                translated_skills.append(got)

        subagents = []
        for root in _agent_dirs(stores, proj):
            for md in sorted(root.glob("*.md")):
                subagents.append(tr.translate_subagent_md(md))

        servers: list[dict] = []
        warnings: list[str] = []
        raw = _claude_mcp_servers(stores, proj)
        if raw:
            servers, warnings = tr.translate_mcp_servers(raw)

        claude_md = ""
        if proj is not None:
            cm = proj / "CLAUDE.md"
            if cm.is_file():
                claude_md = cm.read_text(encoding="utf-8", errors="replace")

        memory: list[tuple[str, str]] = []
        if include_memory and proj is not None:
            mem_dir = proj / ".claude" / "memory"
            if mem_dir.is_dir():
                memory = tr.memory_chunks(mem_dir)

        data, plan = eb.build_bundle(
            agent_name=name,
            skills=translated_skills,
            subagents=subagents,
            mcp_servers=servers,
            claude_md=claude_md,
            memory=memory,
        )
        plan.warnings.extend(warnings)

        report = plan.render()
        if not apply:
            return f"{report}\n\nDry run — nothing written. Re-run with apply=True to write the zip."
        written = eb.write_bundle(data, Path(out).expanduser() if out.strip() else Path.cwd())
        tail = (
            "\n\nThis bundle carries a knowledge seed — it is NOT publishable. Read REVIEW.md inside."
            if plan.carries_knowledge
            else "\n\nRead REVIEW.md inside before sharing."
        )
        return f"{report}\n\nWrote {written} ({len(data)} bytes).{tail}"

    return [claude_export_snapshot]


def _claude_mcp_servers(stores: ClaudeStores, proj: Path | None) -> dict:
    """The same two sources ``claude_import_mcp`` reads — user `settings.json` then the
    project's `.mcp.json`, project winning. Duplicated deliberately rather than refactored
    out of the import tool: that one is async and threads through an apply path, and export
    only needs the read. If a third caller appears, hoist it into ``stores``."""
    import json

    out: dict = {}
    settings = stores.cli.root / "settings.json"
    if settings.is_file():
        try:
            out.update(json.loads(settings.read_text(encoding="utf-8")).get("mcpServers") or {})
        except (ValueError, OSError):
            pass
    if proj is not None:
        mcp_json = proj / ".mcp.json"
        if mcp_json.is_file():
            try:
                out.update(json.loads(mcp_json.read_text(encoding="utf-8")).get("mcpServers") or {})
            except (ValueError, OSError):
                pass
    return out


def _agent_dirs(stores: ClaudeStores, proj: Path | None) -> list[Path]:
    """Every directory that may hold Claude Code subagent markdown — user-level, plus the
    project's own when one was given."""
    out: list[Path] = []
    user = stores.cli.root / "agents"
    if user.is_dir():
        out.append(user)
    if proj is not None:
        p = proj / ".claude" / "agents"
        if p.is_dir():
            out.append(p)
    return out
