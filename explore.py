"""Read-only explore tools over the Claude Code stores.

Every tool returns a bounded, human-readable string; errors come back as
"error: ..." text instead of raising, so a bad path never kills the turn.
Secrets discipline: MCP server env/header VALUES are never echoed — key names only.
"""

from __future__ import annotations

import json
from pathlib import Path
from langchain_core.tools import tool

from .boundaries import credential_free_url
from .stores import ClaudeStores, FencedRoot, project_slug_candidates

_LIST_CAP = 40
_SNIPPET = 200


def _snippet(text: str, limit: int = _SNIPPET) -> str:
    text = " ".join(str(text).split())
    return text if len(text) <= limit else text[: limit - 1] + "…"


def _capped(lines: list[str], cap: int = _LIST_CAP) -> str:
    if len(lines) > cap:
        return "\n".join(lines[:cap] + [f"… {len(lines) - cap} more"])
    return "\n".join(lines)


def _mtime(path: Path) -> str:
    from datetime import datetime

    try:
        return datetime.fromtimestamp(path.stat().st_mtime).strftime("%Y-%m-%d %H:%M")
    except OSError:
        return "?"


def _when(value) -> str:
    """Render a timestamp that may be ISO text or epoch seconds/milliseconds."""
    try:
        from datetime import datetime

        n = float(value)
        if n > 1e12:
            n /= 1000.0
        return datetime.fromtimestamp(n).strftime("%Y-%m-%d %H:%M")
    except (TypeError, ValueError, OSError):
        return _snippet(str(value), 30)


def _frontmatter(text: str) -> dict:
    """Best-effort YAML frontmatter parse; returns {} on any failure."""
    if not text.startswith("---"):
        return {}
    try:
        import yaml

        _, fm, _ = text.split("---", 2)
        data = yaml.safe_load(fm)
        return data if isinstance(data, dict) else {}
    except Exception:  # noqa: BLE001
        return {}


def _redact_mcp(servers) -> list[str]:
    """Describe MCP server entries without echoing env values, headers, or URLs' secrets."""
    out = []
    items = servers.items() if isinstance(servers, dict) else enumerate(servers or [])
    for name, entry in items:
        if not isinstance(entry, dict):
            out.append(f"- {name}")
            continue
        kind = entry.get("type") or entry.get("transport") or ("url" if entry.get("url") else "stdio")
        bits = [f"- {name} ({kind})"]
        if entry.get("command"):
            bits.append(f"command={entry['command']} args={len(entry.get('args') or [])}")
        if entry.get("url"):
            bits.append(f"url={credential_free_url(entry['url'], invalid='[redacted]')}")
        env_keys = sorted((entry.get("env") or {}).keys())
        if env_keys:
            bits.append(f"env keys={env_keys}")
        header_keys = sorted((entry.get("headers") or {}).keys())
        if header_keys:
            bits.append(f"header keys={header_keys}")
        out.append(" ".join(bits))
    return out


def _transcript_tail(path: Path, tail: int, max_bytes: int) -> list[str]:
    """Summarize the last `tail` records of a session JSONL transcript."""
    size = path.stat().st_size
    with path.open("rb") as fh:
        fh.seek(max(0, size - max(max_bytes * 4, 262144)))
        chunk = fh.read().decode("utf-8", errors="replace")
    lines = chunk.splitlines()
    if size > len(chunk.encode("utf-8", errors="replace")) and lines:
        lines = lines[1:]  # drop the partial first line of the tail window
    out = []
    for line in lines[-tail:]:
        try:
            obj = json.loads(line)
        except ValueError:
            continue
        kind = obj.get("type") or obj.get("role") or "?"
        text = ""
        msg = obj.get("message")
        if isinstance(msg, dict):
            content = msg.get("content")
            if isinstance(content, str):
                text = content
            elif isinstance(content, list):
                for part in content:
                    if isinstance(part, dict) and part.get("type") == "text" and part.get("text"):
                        text = part["text"]
                        break
                    if isinstance(part, dict) and part.get("type") == "tool_use":
                        text = f"[tool_use: {part.get('name', '?')}]"
                        break
        elif obj.get("summary"):
            text = f"[summary] {obj['summary']}"
        out.append(f"- {kind}: {_snippet(text)}" if text else f"- {kind}")
    return out


def build_explore_tools(cfg: dict) -> list:
    stores = ClaudeStores(cfg)

    def _project_dir(directory: str) -> tuple[str, Path]:
        found = stores.find_project(directory)
        if not found:
            raise ValueError(f"no Claude Code project found for {directory!r} under {stores.cli.root / 'projects'}")
        return found

    @tool
    def claude_projects(query: str = "") -> str:
        """List Claude Code projects on this machine, or resolve a directory to its project.

        Pass a directory path (starting with / or ~) to look up that project's entry:
        session count, memory presence, last activity. Pass a plain substring to filter
        the project list, or nothing for the full list (most recently active first).
        """
        try:
            projects = stores.cli.resolve("projects")
            if not projects.is_dir():
                return f"error: no projects dir at {projects}"
            if query.strip().startswith(("/", "~")):
                slug, pdir = _project_dir(query.strip())
                sessions = sorted(pdir.glob("*.jsonl"))
                memory = pdir / "memory"
                mem_files = sorted(p.name for p in memory.glob("*.md")) if memory.is_dir() else []
                lines = [
                    f"project: {slug}",
                    f"sessions: {len(sessions)} (latest: {_mtime(max(sessions, key=lambda p: p.stat().st_mtime)) if sessions else '-'})",
                    f"memory: {'yes — ' + str(len(mem_files)) + ' files' if mem_files else 'none'}",
                ]
                if mem_files:
                    lines.append("memory files: " + ", ".join(mem_files[:20]))
                return "\n".join(lines)
            entries = []
            for pdir in projects.iterdir():
                if not pdir.is_dir() or (query and query.lower() not in pdir.name.lower()):
                    continue
                n_sessions = len(list(pdir.glob("*.jsonl")))
                has_memory = (pdir / "memory" / "MEMORY.md").exists()
                entries.append(
                    (
                        pdir.stat().st_mtime,
                        f"- {pdir.name}  sessions={n_sessions}" + ("  memory=yes" if has_memory else ""),
                    )
                )
            entries.sort(reverse=True)
            if not entries:
                return f"no projects match {query!r}"
            return _capped([line for _, line in entries])
        except Exception as exc:  # noqa: BLE001
            return f"error: {exc}"

    @tool
    def claude_memory(directory: str, file: str = "") -> str:
        """Read Claude Code's persistent memory for a directory's project.

        With no file: returns the MEMORY.md index plus the list of memory files.
        With file (e.g. 'some-fact.md'): returns that memory file's content.
        """
        try:
            slug, pdir = _project_dir(directory)
            memory_rel = f"projects/{slug}/memory"
            if file:
                text, truncated = stores.cli.read_text(f"{memory_rel}/{file}", stores.max_read_bytes)
                return text + ("\n[truncated]" if truncated else "")
            memory = pdir / "memory"
            if not memory.is_dir():
                return f"project {slug} has no memory directory"
            files = sorted(p.name for p in memory.glob("*.md") if p.name != "MEMORY.md")
            index = memory / "MEMORY.md"
            head = ""
            if index.exists():
                head, truncated = stores.cli.read_text(f"{memory_rel}/MEMORY.md", stores.max_read_bytes)
                head += "\n[truncated]" if truncated else ""
            return head + "\n\nmemory files: " + (", ".join(files) if files else "(none)")
        except Exception as exc:  # noqa: BLE001
            return f"error: {exc}"

    @tool
    def claude_sessions(directory: str, session_id: str = "") -> str:
        """List Claude Code sessions for a directory's project, or tail one transcript.

        With no session_id: lists recent session transcripts (id, size, last modified).
        With session_id: summarizes the last messages of that session's transcript.
        """
        try:
            slug, pdir = _project_dir(directory)
            if session_id:
                transcript = stores.cli.resolve(f"projects/{slug}/{session_id}.jsonl")
                if not transcript.is_file():
                    return f"error: no transcript {session_id}.jsonl in {slug}"
                lines = _transcript_tail(transcript, stores.transcript_tail, stores.max_read_bytes)
                return f"last {len(lines)} records of {session_id}:\n" + "\n".join(lines)
            sessions = sorted(pdir.glob("*.jsonl"), key=lambda p: p.stat().st_mtime, reverse=True)
            if not sessions:
                return f"project {slug} has no session transcripts"
            rows = [f"- {p.stem}  {p.stat().st_size // 1024}KB  {_mtime(p)}" for p in sessions]
            return _capped(rows, cap=20)
        except Exception as exc:  # noqa: BLE001
            return f"error: {exc}"

    @tool
    def claude_scratchpad(directory: str, session_id: str = "", path: str = "") -> str:
        """Browse a project's ephemeral Claude Code scratchpads (wiped on reboot).

        With no session_id: lists sessions that have a scratchpad for the directory's
        project. With session_id: lists that scratchpad's files. With session_id and
        path: reads one file.
        """
        try:
            slug = None
            for candidate in project_slug_candidates(directory):
                if (stores.scratchpad.root / candidate).is_dir():
                    slug = candidate
                    break
            if slug is None:
                return f"no scratchpads for {directory!r} under {stores.scratchpad.root}"
            if session_id and path:
                text, truncated = stores.scratchpad.read_text(
                    f"{slug}/{session_id}/scratchpad/{path}", stores.max_read_bytes
                )
                return text + ("\n[truncated]" if truncated else "")
            if session_id:
                pad = stores.scratchpad.resolve(f"{slug}/{session_id}/scratchpad")
                if not pad.is_dir():
                    return f"session {session_id} has no scratchpad"
                rows = [
                    f"- {p.relative_to(pad)}  {p.stat().st_size}B  {_mtime(p)}"
                    for p in sorted(pad.rglob("*"))
                    if p.is_file()
                ]
                return _capped(rows) if rows else "scratchpad is empty"
            rows = []
            for sess in sorted((stores.scratchpad.root / slug).iterdir()):
                pad = sess / "scratchpad"
                if pad.is_dir():
                    n = sum(1 for p in pad.rglob("*") if p.is_file())
                    rows.append(f"- {sess.name}  files={n}  {_mtime(sess)}")
            return _capped(rows) if rows else f"no scratchpads for {slug}"
        except Exception as exc:  # noqa: BLE001
            return f"error: {exc}"

    @tool
    def claude_cowork(session_id: str = "", path: str = "") -> str:
        """Browse Claude Cowork / desktop local-agent-mode sessions.

        With nothing: lists sessions (model, dates, initial message). With session_id:
        shows that session's metadata, outputs listing, and audit-log tail. With
        session_id and path: reads a file from the session's directory (e.g.
        'outputs/report.md'). Cowork transcripts stay inside the sandbox VM — metadata,
        outputs, and the audit log are what exists on the host.
        """
        try:
            root = stores.cowork.root
            if not root.is_dir():
                return f"error: no Cowork store at {root}"
            sidecars = sorted(root.glob("*/*/local_*.json"))
            if not session_id:
                rows = []
                for sc in sidecars:
                    try:
                        meta = json.loads(sc.read_text(encoding="utf-8"))
                    except ValueError:
                        continue
                    sid = sc.stem
                    rows.append(
                        f"- {sid}  model={meta.get('model', '?')}  created={_when(meta.get('createdAt', '?'))}"
                        + ("  archived" if meta.get("isArchived") else "")
                        + f"\n    {_snippet(str(meta.get('initialMessage', '')), 120)}"
                    )
                return _capped(rows, cap=25) if rows else "no Cowork sessions found"
            match = next((sc for sc in sidecars if sc.stem == session_id or session_id in sc.stem), None)
            if match is None:
                return f"error: no Cowork session matching {session_id!r}"
            rel = match.relative_to(root)
            session_dir = match.parent / match.stem
            if path:
                text, truncated = stores.cowork.read_text(str(rel.parent / match.stem / path), stores.max_read_bytes)
                return text + ("\n[truncated]" if truncated else "")
            meta = json.loads(match.read_text(encoding="utf-8"))
            keep = [
                "createdAt",
                "lastActivityAt",
                "model",
                "permissionMode",
                "cwd",
                "egressAllowedDomains",
                "memoryEnabled",
                "pluginsEnabled",
                "isArchived",
                "cliSessionId",
                "initialMessage",
            ]
            lines = [
                f"{k}: {_when(meta[k]) if k in ('createdAt', 'lastActivityAt') else _snippet(str(meta[k]), 160)}"
                for k in keep
                if k in meta
            ]
            outputs = session_dir / "outputs"
            if outputs.is_dir():
                files = [str(p.relative_to(session_dir)) for p in sorted(outputs.rglob("*")) if p.is_file()]
                lines.append(
                    "files: " + (", ".join(files[:50]) + ("…" if len(files) > 50 else "") if files else "(none)")
                )
            audit = session_dir / "audit.jsonl"
            if audit.is_file():
                tail = _transcript_tail(audit, 10, stores.max_read_bytes)
                lines += ["audit tail:"] + tail
            return "\n".join(lines)
        except Exception as exc:  # noqa: BLE001
            return f"error: {exc}"

    @tool
    def claude_inventory(project_dir: str = "") -> str:
        """Inventory the Claude Code customizations installed on this machine.

        User level: skills, subagents, slash commands, plugins, MCP servers (secret
        values redacted), and settings highlights. Pass project_dir to also inventory
        that directory's project-level .claude/ (agents, commands, hooks), .mcp.json,
        and CLAUDE.md.
        """
        try:
            cli = stores.cli.root
            sections: list[str] = []

            def _inventory_text(fence: FencedRoot, rel: str, label: str) -> str | None:
                try:
                    text, truncated = fence.read_text(rel, stores.max_read_bytes)
                except (OSError, ValueError):
                    sections.append(f"{label}: (unreadable or outside declared root)")
                    return None
                if truncated:
                    sections.append(f"{label}: (truncated at max_read_bytes={stores.max_read_bytes})")
                    return None
                return text

            def _md_listing(folder: Path, kind: str, fence: FencedRoot) -> None:
                if not folder.is_dir():
                    return
                rows = []
                for p in sorted(folder.iterdir()):
                    rel = ""
                    if p.is_dir() and (p / "SKILL.md").is_file():
                        rel = str((p / "SKILL.md").relative_to(fence.root))
                    elif p.suffix == ".md":
                        rel = str(p.relative_to(fence.root))
                    else:
                        continue
                    try:
                        text, truncated = fence.read_text(rel, stores.max_read_bytes)
                    except (OSError, ValueError):
                        continue
                    fm = _frontmatter(text)
                    name = fm.get("name") or p.stem
                    description = _snippet(str(fm.get("description", "")), 110)
                    rows.append(f"- {name}: {description}" + (" [truncated]" if truncated else ""))
                if rows:
                    sections.append(f"{kind} ({len(rows)}):\n" + _capped(rows))

            _md_listing(cli / "skills", "user skills", stores.cli)
            _md_listing(cli / "agents", "user subagents", stores.cli)
            _md_listing(cli / "commands", "user commands", stores.cli)

            installed = cli / "plugins" / "installed_plugins.json"
            if installed.is_file():
                try:
                    raw = _inventory_text(stores.cli, "plugins/installed_plugins.json", "installed plugins")
                    data = json.loads(raw) if raw is not None else {}
                    plugins = data.get("plugins")
                    names = sorted(plugins.keys()) if isinstance(plugins, dict) else []
                    if raw is not None:
                        sections.append(f"installed plugins ({len(names)}): " + ", ".join(names))
                except ValueError:
                    sections.append("installed plugins: (unreadable)")

            settings = cli / "settings.json"
            if settings.is_file():
                try:
                    raw = _inventory_text(stores.cli, "settings.json", "settings")
                    data = json.loads(raw) if raw is not None else {}
                    bits = []
                    if data.get("model"):
                        bits.append(f"model={data['model']}")
                    if isinstance(data.get("enabledPlugins"), (list, dict)):
                        bits.append(f"enabledPlugins={len(data['enabledPlugins'])}")
                    if isinstance(data.get("hooks"), dict):
                        bits.append(f"hooks={sorted(data['hooks'].keys())}")
                    if raw is not None:
                        sections.append("settings: " + (", ".join(bits) if bits else "(no highlights)"))
                    if raw is not None and isinstance(data.get("mcpServers"), dict) and data["mcpServers"]:
                        sections.append("user MCP servers:\n" + "\n".join(_redact_mcp(data["mcpServers"])))
                except ValueError:
                    sections.append("settings: (unreadable)")

            if project_dir:
                proj = Path(project_dir).expanduser().resolve()
                project_fence = FencedRoot("project", proj)
                sections.append(f"— project level: {proj} —")
                dot = proj / ".claude"
                _md_listing(dot / "skills", "project skills", project_fence)
                _md_listing(dot / "agents", "project subagents", project_fence)
                _md_listing(dot / "commands", "project commands", project_fence)
                psettings = dot / "settings.json"
                if psettings.is_file():
                    try:
                        raw = _inventory_text(project_fence, ".claude/settings.json", "project settings")
                        data = json.loads(raw) if raw is not None else {}
                        hooks = data.get("hooks")
                        if raw is not None and isinstance(hooks, dict):
                            sections.append(f"project hooks: {sorted(hooks.keys())}")
                    except ValueError:
                        sections.append("project settings: (unreadable)")
                mcp = proj / ".mcp.json"
                if mcp.is_file():
                    try:
                        raw = _inventory_text(project_fence, ".mcp.json", "project .mcp.json")
                        data = json.loads(raw) if raw is not None else {}
                        if raw is not None:
                            sections.append(
                                "project MCP servers:\n" + "\n".join(_redact_mcp(data.get("mcpServers") or {}))
                            )
                    except ValueError:
                        sections.append("project .mcp.json: (unreadable)")
                if (proj / "CLAUDE.md").is_file():
                    raw = _inventory_text(project_fence, "CLAUDE.md", "CLAUDE.md")
                    if raw is not None:
                        heading = next((ln for ln in raw.splitlines() if ln.strip()), "")
                        sections.append(f"CLAUDE.md: present ({_snippet(heading, 80)})")

            return "\n\n".join(sections) if sections else "nothing found"
        except Exception as exc:  # noqa: BLE001
            return f"error: {exc}"

    return [
        claude_projects,
        claude_memory,
        claude_sessions,
        claude_scratchpad,
        claude_cowork,
        claude_inventory,
    ]
