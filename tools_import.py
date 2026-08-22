"""The v0.2 import tools: translate Claude Code artifacts into protoAgent.

Every tool defaults to a DRY RUN (`apply=False`) — it reports exactly what
would be imported (and what gets skipped, and why) so the operator approves
before anything writes. License rule enforced throughout: Anthropic-authored
skills (Cowork's `creatorType: anthropic` + license-text detection) are never
imported — this pack's originals replace them.
"""

from __future__ import annotations

import json
from pathlib import Path

from langchain_core.tools import tool

from .boundaries import complete_text, credential_free_url
from .stores import ClaudeStores, FencedRoot
from . import translate as tr
from . import importer


def _bounded_json(fence: FencedRoot, rel: str, max_bytes: int) -> dict:
    data = json.loads(complete_text(fence, rel, max_bytes))
    return data if isinstance(data, dict) else {}


def _safe_resolved_paths(
    fence: FencedRoot, paths, *, directories: bool, problems: list[str] | None = None, label: str = ""
) -> list[Path]:
    """Resolve each entry inside the fence; name the ones that escape it.

    A symlinked skill/command/agent that points outside the store (a common
    dotfiles setup) is refused, never followed — but silently dropping it
    leaves the operator guessing why their skill is missing, so the refusal is
    reported through ``problems`` when the caller collects them."""
    safe: list[Path] = []
    for path in paths:
        try:
            candidate = fence.resolve(str(path.relative_to(fence.root)))
        except (OSError, ValueError):
            if problems is not None:
                problems.append(f"{label or fence.name}: {path.name}: REFUSED (outside declared root)")
            continue
        matches_kind = candidate.is_dir() if directories else candidate.is_file()
        if matches_kind:
            safe.append(candidate)
    return safe


def _safe_markdown_files(
    fence: FencedRoot, rel_dir: str, problems: list[str] | None = None, *, label: str = ""
) -> list[Path]:
    try:
        root = fence.resolve(rel_dir)
    except (OSError, ValueError):
        if problems is not None:
            problems.append(f"{label or rel_dir}: REFUSED (unreadable or outside declared root)")
        return []
    if not root.is_dir():
        return []
    return _safe_resolved_paths(fence, sorted(root.glob("*.md")), directories=False, problems=problems, label=label)


def _cowork_manifest_types(stores: ClaudeStores) -> tuple[dict[str, str], set[Path]]:
    """Return creator types plus account roots whose manifest cannot be trusted."""
    out: dict[str, str] = {}
    unverifiable: set[Path] = set()
    for manifest in stores.cowork.root.glob("skills-plugin/*/*/manifest.json"):
        try:
            rel = str(manifest.relative_to(stores.cowork.root))
            data = _bounded_json(stores.cowork, rel, stores.max_read_bytes)
        except (ValueError, OSError):
            unverifiable.add(manifest.parent.resolve())
            continue
        for s in data.get("skills") or []:
            if isinstance(s, dict) and s.get("name"):
                out[s["name"]] = str(s.get("creatorType") or "")
    return out, unverifiable


def _skill_sources(stores: ClaudeStores, source: str) -> tuple[list[tuple[Path, str]], list[str], list[str]]:
    """Return importable candidates, license exclusions, and source problems.

    Exclusions are license-driven and non-negotiable: Cowork's manifest marks
    Anthropic-authored skills (`creatorType: anthropic`) — those never enter
    the candidate list, only the excluded report."""

    if source == "user":
        try:
            root = stores.cli.resolve("skills")
        except (OSError, ValueError):
            return [], [], ["user skills root: REFUSED (unreadable or outside declared root)"]
        if not root.is_dir():
            return [], [], []
        problems: list[str] = []
        safe = _safe_resolved_paths(
            stores.cli, sorted(root.iterdir()), directories=True, problems=problems, label="user skills"
        )
        return [(p, "claude-code:user") for p in safe], [], problems
    if source == "cowork":
        types, unverifiable = _cowork_manifest_types(stores)
        candidates, excluded = [], []
        problems: list[str] = []
        paths = _safe_resolved_paths(
            stores.cowork,
            sorted(stores.cowork.root.glob("skills-plugin/*/*/skills/*")),
            directories=True,
            problems=problems,
            label="cowork skills",
        )
        for p in paths:
            if p.parent.parent.resolve() in unverifiable:
                excluded.append(f"{p.name} (unverifiable manifest)")
            elif types.get(p.name, "").lower() == "anthropic":
                excluded.append(p.name)
            else:
                candidates.append((p, "claude-cowork"))
        return candidates, excluded, problems
    # anything else is a project directory
    project = FencedRoot("project", Path(source).expanduser())
    try:
        root = project.resolve(".claude/skills")
    except (OSError, ValueError):
        return [], [], [f"project skills root: REFUSED (unreadable or outside declared root): {source}"]
    if not root.is_dir():
        return [], [], []
    problems: list[str] = []
    safe = _safe_resolved_paths(
        project, sorted(root.iterdir()), directories=True, problems=problems, label="project skills"
    )
    return [(p, f"claude-code:{source}") for p in safe], [], problems


def _pick(names: str, available: list[str]) -> list[str]:
    if not names.strip() or names.strip().lower() == "all":
        return available
    wanted = {n.strip() for n in names.split(",") if n.strip()}
    return [n for n in available if n in wanted]


def build_import_tools(cfg: dict) -> list:
    stores = ClaudeStores(cfg)

    @tool
    def claude_import_scan(project_dir: str = "") -> str:
        """Inventory what could be imported from Claude Code into this agent:
        user-authored skills (Anthropic-licensed ones are listed as excluded),
        slash commands, subagents, MCP servers, and project memory. Dry-run
        only — nothing is written. Pass project_dir to include that
        directory's project-level artifacts.
        """
        try:
            lines: list[str] = []
            for source in ["user", "cowork"] + ([project_dir] if project_dir else []):
                candidates, excluded, source_problems = _skill_sources(stores, source)
                names = []
                refused = list(source_problems)
                for d, _ in candidates:
                    try:
                        skill_fence = FencedRoot("skill", d)
                        meta, _body = tr.parse_frontmatter(
                            complete_text(skill_fence, "SKILL.md", stores.max_read_bytes)
                            if (d / "SKILL.md").is_file()
                            else ""
                        )
                        (excluded if tr.is_anthropic_material(d, meta, stores.max_read_bytes) else names).append(d.name)
                    except (OSError, ValueError) as exc:
                        refused.append(f"{d.name}: REFUSED ({exc})")
                if names or excluded or refused:
                    lines.append(f"skills [{source}]: {', '.join(names) or '(none)'}")
                    if excluded:
                        lines.append(
                            f"  excluded (Anthropic-licensed or unverifiable, never imported): {', '.join(excluded)}"
                        )
                    if refused:
                        lines.append(f"  refused: {', '.join(refused)}")
            markdown_problems: list[str] = []
            cmds = [
                p.stem
                for p in _safe_markdown_files(stores.cli, "commands", markdown_problems, label="user commands root")
            ]
            project_fence = FencedRoot("project", Path(project_dir).expanduser()) if project_dir else None
            if project_fence is not None:
                cmds += [
                    p.stem
                    for p in _safe_markdown_files(
                        project_fence, ".claude/commands", markdown_problems, label="project commands root"
                    )
                ]
            if cmds:
                lines.append(f"commands: {', '.join(cmds)}")
            agents = [
                p.stem for p in _safe_markdown_files(stores.cli, "agents", markdown_problems, label="user agents root")
            ]
            if project_fence is not None:
                agents += [
                    p.stem
                    for p in _safe_markdown_files(
                        project_fence, ".claude/agents", markdown_problems, label="project agents root"
                    )
                ]
            if agents:
                lines.append(f"subagents: {', '.join(agents)}")
            lines += [f"refused: {problem}" for problem in markdown_problems]
            mcp_names: list[str] = []
            mcp_problems: list[str] = []
            settings = stores.cli.root / "settings.json"
            if settings.is_file():
                try:
                    mcp_names += sorted(
                        (
                            _bounded_json(stores.cli, "settings.json", stores.max_read_bytes).get("mcpServers") or {}
                        ).keys()
                    )
                except ValueError as exc:
                    mcp_problems.append(f"settings.json: REFUSED ({exc})")
            if project_fence is not None:
                if (project_fence.root / ".mcp.json").is_file():
                    try:
                        mcp_names += sorted(
                            (
                                _bounded_json(project_fence, ".mcp.json", stores.max_read_bytes).get("mcpServers") or {}
                            ).keys()
                        )
                    except ValueError as exc:
                        mcp_problems.append(f"project .mcp.json: REFUSED ({exc})")
            if mcp_names:
                lines.append(f"mcp servers: {', '.join(mcp_names)}")
            lines += [f"mcp: {problem}" for problem in mcp_problems]
            if project_dir:
                found = stores.find_project(project_dir)
                if found:
                    memory_problems: list[str] = []
                    n = (
                        len(tr.memory_chunks(found[1] / "memory", stores.max_read_bytes, memory_problems))
                        if (found[1] / "memory").is_dir()
                        else 0
                    )
                    lines.append(f"memory: {n} topic files for {project_dir}")
                    lines += [f"  refused: {problem}" for problem in memory_problems]
            return "\n".join(lines) if lines else "nothing importable found"
        except Exception as exc:  # noqa: BLE001
            return f"error: {exc}"

    @tool
    async def claude_import_skills(names: str = "all", source: str = "user", apply: bool = False) -> str:
        """Import user-authored Claude Code skills as protoAgent skills.
        source: 'user' (~/.claude/skills), 'cowork' (user-authored Cowork
        skills), or a project directory path. names: comma-separated or 'all'.
        Dry-run by default — set apply=True only after the operator approves.
        Anthropic-licensed skills are always refused (their license prohibits
        redistribution); existing skills are never overwritten.
        """
        try:
            candidates, excluded, source_problems = _skill_sources(stores, source)
            chosen = _pick(names, [d.name for d, _ in candidates])
            results: list[str] = list(source_problems)
            target = importer.skills_target_root() if apply else None
            for d, label in candidates:
                if d.name not in chosen:
                    continue
                try:
                    translated = tr.translate_skill_dir(d, source=label, max_bytes=stores.max_read_bytes)
                except (OSError, ValueError) as exc:
                    results.append(f"- {d.name}: REFUSED ({exc})")
                    continue
                if translated is None:
                    results.append(f"- {d.name}: REFUSED (Anthropic-licensed or unreadable)")
                    continue
                for w in translated.warnings:
                    results.append(f"  note ({translated.name}): {w}")
                if apply and target is not None:
                    results.append(f"- {importer.write_skill(translated, target)}")
                else:
                    results.append(
                        f"- would import {d.name!r} as skill {translated.name!r} ({len(translated.files)} file(s))"
                    )
            if not results:
                return f"no matching skills in source {source!r}"
            header = "" if apply else "DRY RUN — re-run with apply=True after the operator approves:\n"
            return header + "\n".join(results)
        except Exception as exc:  # noqa: BLE001
            return f"error: {exc}"

    @tool
    async def claude_import_commands(names: str = "all", project_dir: str = "", apply: bool = False) -> str:
        """Import Claude Code slash commands as protoAgent slash skills
        (user_facing + /name invocation preserved). names: comma-separated or
        'all'. Dry-run by default; existing skills are never overwritten.
        """
        try:
            results: list[str] = []
            files = _safe_markdown_files(stores.cli, "commands", results, label="user commands root")
            if project_dir:
                project_fence = FencedRoot("project", Path(project_dir).expanduser())
                files += _safe_markdown_files(project_fence, ".claude/commands", results, label="project commands root")
            chosen = _pick(names, [p.stem for p in files])
            target = importer.skills_target_root() if apply else None
            for p in files:
                if p.stem not in chosen:
                    continue
                try:
                    translated = tr.translate_command_md(p, max_bytes=stores.max_read_bytes)
                except (OSError, ValueError) as exc:
                    results.append(f"- {p.stem}: REFUSED ({exc})")
                    continue
                for w in translated.warnings:
                    results.append(f"  note ({translated.name}): {w}")
                if apply and target is not None:
                    results.append(f"- {importer.write_skill(translated, target)}")
                else:
                    results.append(f"- would import /{p.stem} as slash skill /{translated.slash}")
            if not results:
                return "no matching commands found"
            header = "" if apply else "DRY RUN — re-run with apply=True after the operator approves:\n"
            return header + "\n".join(results)
        except Exception as exc:  # noqa: BLE001
            return f"error: {exc}"

    @tool
    async def claude_import_subagents(names: str = "all", project_dir: str = "", apply: bool = False) -> str:
        """Import Claude Code subagents as protoAgent subagents (registered
        via this plugin's config; live after the next config reload). Tool
        names map to protoAgent equivalents — unmapped ones are omitted and
        flagged; the model is not carried (the subagent inherits the instance
        model). Dry-run by default.
        """
        try:
            results: list[str] = []
            files = _safe_markdown_files(stores.cli, "agents", results, label="user agents root")
            if project_dir:
                project_fence = FencedRoot("project", Path(project_dir).expanduser())
                files += _safe_markdown_files(project_fence, ".claude/agents", results, label="project agents root")
            chosen = _pick(names, [p.stem for p in files])
            subs = []
            for p in files:
                if p.stem not in chosen:
                    continue
                try:
                    s = tr.translate_subagent_md(p, max_bytes=stores.max_read_bytes)
                except (OSError, ValueError) as exc:
                    results.append(f"- {p.stem}: REFUSED ({exc})")
                    continue
                subs.append(s)
                results.append(f"- {s.name}: tools={s.tools or '(text-only)'}")
                results += [f"  note: {w}" for w in s.warnings]
            if not subs:
                return "\n".join(results) if results else "no matching subagents found"
            if apply:
                ok, messages = await importer.apply_subagents(subs)
                results += [f"- {m}" for m in messages]
                if not ok:
                    results.append("- APPLY FAILED — nothing persisted")
                return "\n".join(results)
            return "DRY RUN — re-run with apply=True after the operator approves:\n" + "\n".join(results)
        except Exception as exc:  # noqa: BLE001
            return f"error: {exc}"

    @tool
    async def claude_import_mcp(names: str = "all", project_dir: str = "", apply: bool = False) -> str:
        """Import Claude Code MCP server configs into protoAgent's mcp.servers
        (replace-by-name merge; the config reload reconnects). Sources:
        ~/.claude/settings.json mcpServers + the project's .mcp.json. Dry-run
        by default — the dry run redacts env/header values.
        """
        try:
            cc: dict = {}
            source_problems: list[str] = []
            settings = stores.cli.root / "settings.json"
            if settings.is_file():
                try:
                    cc.update(_bounded_json(stores.cli, "settings.json", stores.max_read_bytes).get("mcpServers") or {})
                except ValueError as exc:
                    source_problems.append(f"settings.json: REFUSED ({exc})")
            if project_dir:
                project_fence = FencedRoot("project", Path(project_dir).expanduser())
                mcp_json = Path(project_dir).expanduser() / ".mcp.json"
                if mcp_json.is_file():
                    try:
                        cc.update(
                            _bounded_json(project_fence, ".mcp.json", stores.max_read_bytes).get("mcpServers") or {}
                        )
                    except ValueError as exc:
                        source_problems.append(f"project .mcp.json: REFUSED ({exc})")
            chosen = _pick(names, sorted(cc.keys()))
            entries, warnings = tr.translate_mcp_servers({k: v for k, v in cc.items() if k in chosen})
            if not entries and not warnings:
                return "\n".join(source_problems) if source_problems else "no matching MCP servers found"
            results = list(source_problems)
            for e in entries:
                safe = {k: v for k, v in e.items() if k not in ("env", "headers", "args")}
                if safe.get("url"):
                    safe["url"] = credential_free_url(safe["url"], invalid="[redacted]")
                extras = []
                if e.get("args"):
                    extras.append(f"args={len(e['args'])}")
                if e.get("env"):
                    extras.append(f"env keys={sorted(e['env'])}")
                if e.get("headers"):
                    extras.append(f"header keys={sorted(e['headers'])}")
                results.append(f"- {safe}" + (f"  ({', '.join(extras)})" if extras else ""))
            results += [f"  note: {w}" for w in warnings]
            if apply and entries:
                ok, messages = await importer.apply_mcp_entries(entries)
                results += [f"- {m}" for m in messages]
                results.append("- applied and reloaded" if ok else "- APPLY FAILED — nothing persisted")
                return "\n".join(results)
            return "DRY RUN — re-run with apply=True after the operator approves:\n" + "\n".join(results)
        except Exception as exc:  # noqa: BLE001
            return f"error: {exc}"

    @tool
    async def claude_import_memory(directory: str, apply: bool = False, limit: int = 0) -> str:
        """Ingest a directory's Claude Code project memory into this agent's
        knowledge graph (domain 'claude-import', provenance-tagged; undo with
        knowledge_purge('claude-import')). Dry-run by default lists the topic
        files and sizes. `limit` 0 (the default) imports EVERY topic; a positive
        value caps it (a large project memory can hold 100+ topics).
        """
        try:
            found = stores.find_project(directory)
            if not found or not (found[1] / "memory").is_dir():
                return f"no Claude Code memory for {directory!r}"
            read_problems: list[str] = []
            chunks = tr.memory_chunks(found[1] / "memory", stores.max_read_bytes, read_problems)
            if int(limit) > 0:
                chunks = chunks[: int(limit)]
            if not chunks:
                out = "memory directory has no importable topics"
                if read_problems:
                    out += "\nproblems:\n" + "\n".join(f"- {p}" for p in read_problems[:10])
                return out
            if not apply:
                listing = "\n".join(f"- {h} ({len(c)} chars)" for h, c in chunks[:40])
                out = (
                    f"DRY RUN — {len(chunks)} topic file(s) would be ingested into knowledge "
                    f"domain 'claude-import' (re-run with apply=True after the operator approves):\n{listing}"
                )
                if read_problems:
                    out += "\nproblems:\n" + "\n".join(f"- {p}" for p in read_problems[:10])
                return out
            added, ingest_problems = await importer.ingest_memory(chunks, source_label=f"claude-code {directory}")
            out = f"ingested {added}/{len(chunks)} memory topics into domain 'claude-import'"
            problems = read_problems + ingest_problems
            if problems:
                out += "\nproblems:\n" + "\n".join(f"- {p}" for p in problems[:10])
            return out
        except Exception as exc:  # noqa: BLE001
            return f"error: {exc}"

    @tool
    async def claude_import_claude_md(directory: str, apply: bool = False) -> str:
        """Ingest a repo's CLAUDE.md — its *operating instructions* (run commands,
        pre-PR gates, the gotchas that recur) — into this agent's knowledge graph
        (domain 'claude-import', undo with knowledge_purge('claude-import')), so the
        agent can recall how the repo wants to be worked. Dry-run by default.

        CLAUDE.md lives at the repo root (not under ~/.claude). It's *instructions*,
        not a persona: this lands it in knowledge (retrievable). If you want it always
        in context, promote the translated text into the agent's SOUL.md yourself.
        """
        try:
            project = Path(directory).expanduser()
            claude_md = project / "CLAUDE.md"
            if not claude_md.is_file():
                return f"no CLAUDE.md in {directory!r}"
            project_fence = FencedRoot("project", project)
            try:
                content = complete_text(project_fence, "CLAUDE.md", stores.max_read_bytes).strip()
            except (OSError, ValueError) as exc:
                return f"CLAUDE.md: REFUSED ({exc})"
            if not content:
                return "CLAUDE.md is empty"
            heading = f"Operating instructions (CLAUDE.md) — {project.name}"
            if not apply:
                return (
                    f"DRY RUN — CLAUDE.md ({len(content)} chars) would be ingested into knowledge "
                    f"domain 'claude-import' as {heading!r} (re-run with apply=True after the "
                    f"operator approves)."
                )
            added, problems = await importer.ingest_memory(
                [(heading, content)], source_label=f"claude-code CLAUDE.md {directory}"
            )
            out = "ingested CLAUDE.md into domain 'claude-import'" if added else "nothing ingested"
            if problems:
                out += "\nproblems:\n" + "\n".join(f"- {p}" for p in problems[:10])
            return out
        except Exception as exc:  # noqa: BLE001
            return f"error: {exc}"

    @tool
    def claude_hooks_report(project_dir: str = "") -> str:
        """Report Claude Code hooks (PreToolUse/PostToolUse/etc.) — REPORT
        ONLY: protoAgent has no declarative shell-hook table (its equivalent
        is plugin middleware), so hooks are listed with their commands for the
        operator to port deliberately, never auto-translated.
        """
        try:
            sections: list[str] = []
            source_problems: list[str] = []
            sources: list[tuple[str, FencedRoot, str]] = [("user", stores.cli, "settings.json")]
            if project_dir:
                sources.append(
                    (
                        "project",
                        FencedRoot("project", Path(project_dir).expanduser()),
                        ".claude/settings.json",
                    )
                )
            for label, fence, rel in sources:
                if not (fence.root / rel).is_file():
                    continue
                try:
                    hooks = _bounded_json(fence, rel, stores.max_read_bytes).get("hooks") or {}
                except ValueError as exc:
                    source_problems.append(f"{label} {rel}: REFUSED ({exc})")
                    continue
                if hooks:
                    sections.append(f"[{label}] {json.dumps(hooks, indent=1)[:1500]}")
            if not sections:
                return "\n".join(source_problems) if source_problems else "no hooks configured"
            return (
                "Hooks found (NOT translated — protoAgent's equivalent is plugin middleware, "
                "ADR 0032; port these deliberately):\n" + "\n".join(sections + source_problems)
            )
        except Exception as exc:  # noqa: BLE001
            return f"error: {exc}"

    return [
        claude_import_scan,
        claude_import_skills,
        claude_import_commands,
        claude_import_subagents,
        claude_import_mcp,
        claude_import_memory,
        claude_import_claude_md,
        claude_hooks_report,
    ]
