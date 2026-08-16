"""Pure translators: Claude Code artifacts → protoAgent equivalents.

No host imports and no writes here — every function maps source material to a
plain result object the importer layer (importer.py) applies. Keeping this
layer pure is what lets the test suite cover every mapping host-free.

License rule (protoAgent ADR 0083 D3/D4): Anthropic's bundled skills are
all-rights-reserved — anything Anthropic-authored is detected and REFUSED,
only user-authored material translates.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import yaml

from .stores import FencedRoot

# Agent Skills spec (agentskills.io/specification): 1-64 chars, lowercase
# alphanumerics + hyphens, no leading/trailing/consecutive hyphens, name == dir.
SPEC_NAME_RE = re.compile(r"^[a-z0-9]+(-[a-z0-9]+)*$")
DESCRIPTION_MAX = 1024
NAME_MAX = 64

# Claude Code tool names → protoAgent equivalents. None = no counterpart; the
# translator flags it instead of guessing. Advisory only (protoAgent skill
# `tools:` is a hint; subagent tool lists are enforced allowlists).
TOOL_MAP: dict[str, str | None] = {
    "Bash": "run_command",
    "Read": "read_file",
    "Grep": "search_files",
    "Glob": "find_files",
    "LS": "list_dir",
    "WebFetch": "fetch_url",
    "WebSearch": "web_search",
    "Write": None,
    "Edit": None,
    "NotebookEdit": None,
    "Task": None,  # protoAgent subagents can't recurse into task (recursion guard)
    "TodoWrite": None,
    "Skill": "load_skill",
}


def parse_frontmatter(text: str) -> tuple[dict, str]:
    """Split a ``---``-fenced YAML frontmatter document into (meta, body).
    A document without frontmatter returns ({}, whole text)."""
    if not text.startswith("---"):
        return {}, text
    try:
        _, fm, body = text.split("---", 2)
        meta = yaml.safe_load(fm)
        return (meta if isinstance(meta, dict) else {}), body.lstrip("\n")
    except (ValueError, yaml.YAMLError):
        return {}, text


def slugify_name(raw: str) -> str:
    """Coerce any label to a spec-valid skill name."""
    slug = re.sub(r"[^a-z0-9]+", "-", str(raw).strip().lower()).strip("-")
    slug = re.sub(r"-{2,}", "-", slug)
    return slug[:NAME_MAX].rstrip("-") or "imported-skill"


def provenance(source: str) -> dict:
    """Spec-compliant metadata block marking where an import came from."""
    return {
        "imported-from": source,
        "imported-at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }


def is_anthropic_material(skill_dir: Path, meta: dict, max_bytes: int = 65536) -> bool:
    """True when a skill directory is Anthropic-authored/licensed material.

    Checks the frontmatter ``license``, any bundled LICENSE/NOTICE file, and is
    used alongside the Cowork manifest's ``creatorType`` (checked by the
    scanner, which knows the manifest)."""
    lic = str(meta.get("license", ""))
    if "anthropic" in lic.lower():
        return True
    fence = FencedRoot("skill", skill_dir)
    for candidate in ("LICENSE.txt", "LICENSE", "LICENSE.md", "NOTICE"):
        if (skill_dir / candidate).is_file():
            try:
                head, truncated = fence.read_text(candidate, min(max_bytes, 2000))
            except (OSError, ValueError):
                return True  # unreadable or escaped license = do not import
            if truncated or "anthropic" in head.lower():
                return True
    return False


@dataclass
class TranslatedSkill:
    name: str
    files: dict[str, bytes] = field(default_factory=dict)  # relpath -> content
    description: str = ""
    user_facing: bool = False
    slash: str = ""
    warnings: list[str] = field(default_factory=list)

    @property
    def skill_md(self) -> str:
        return self.files.get("SKILL.md", b"").decode("utf-8", errors="replace")


def _render_frontmatter(meta: dict) -> str:
    return "---\n" + yaml.safe_dump(meta, sort_keys=False, allow_unicode=True).strip() + "\n---\n"


def _normalized_meta(name: str, description: str, source: str, extra: dict | None = None) -> dict:
    meta: dict = {"name": name, "description": description[:DESCRIPTION_MAX]}
    if extra:
        meta.update(extra)
    meta["metadata"] = provenance(source)
    return meta


def _bounded_document(path: Path, max_bytes: int) -> str:
    fence = FencedRoot("document", path.parent)
    text, truncated = fence.read_text(path.name, max_bytes)
    if truncated:
        raise ValueError(f"{path.name} exceeds max_read_bytes={max_bytes}")
    return text


def translate_skill_dir(src: Path, source: str = "claude-code", max_bytes: int = 65536) -> TranslatedSkill | None:
    """A Claude Code skill directory → a protoAgent skill directory.

    Near 1:1 (same open SKILL.md standard). Normalizations: spec-slugged name
    (must equal the target dir), description capped at 1024, ``allowed-tools``
    string → advisory ``tools`` list (mapped where a counterpart exists),
    provenance metadata. Returns None for Anthropic-licensed material."""
    if src.is_symlink():
        return None
    fence = FencedRoot("skill", src)
    skill_md = src / "SKILL.md"
    if not skill_md.is_file():
        return None
    try:
        skill_text, skill_truncated = fence.read_text("SKILL.md", max_bytes)
    except (OSError, ValueError):
        return None
    if skill_truncated:
        raise ValueError(f"SKILL.md exceeds max_read_bytes={max_bytes}")
    meta, body = parse_frontmatter(skill_text)
    if is_anthropic_material(src, meta, max_bytes):
        return None

    out = TranslatedSkill(name=slugify_name(meta.get("name") or src.name))
    raw_desc = str(meta.get("description") or "").strip() or f"Imported Claude Code skill {out.name}."
    if len(raw_desc) > DESCRIPTION_MAX:
        out.warnings.append(f"description truncated to {DESCRIPTION_MAX} chars")
    out.description = raw_desc[:DESCRIPTION_MAX]

    extra: dict = {}
    allowed = meta.get("allowed-tools") or meta.get("tools")
    if allowed:
        raw_tools = allowed.split() if isinstance(allowed, str) else [str(t) for t in allowed]
        mapped = []
        for t in raw_tools:
            base = t.split("(", 1)[0]
            counterpart = TOOL_MAP.get(base, base)
            if counterpart is None:
                out.warnings.append(f"tool {t!r} has no protoAgent counterpart (dropped from hints)")
            else:
                mapped.append(counterpart)
        if mapped:
            extra["tools"] = sorted(set(mapped))
    if meta.get("compatibility"):
        extra["compatibility"] = str(meta["compatibility"])[:500]

    new_meta = _normalized_meta(out.name, out.description, source, extra)
    out.files["SKILL.md"] = (_render_frontmatter(new_meta) + "\n" + body.strip() + "\n").encode("utf-8")

    # Carry supporting material (scripts/references/assets + loose files) verbatim.
    for p in sorted(src.rglob("*")):
        rel = p.relative_to(src)
        if p.is_file() and str(rel) != "SKILL.md":
            try:
                content, truncated = fence.read_bytes(str(rel), max_bytes)
            except (OSError, ValueError):
                continue
            if truncated:
                raise ValueError(f"{rel} exceeds max_read_bytes={max_bytes}")
            out.files[str(rel)] = content
    return out


def translate_command_md(path: Path, source: str = "claude-code", max_bytes: int = 65536) -> TranslatedSkill:
    """A Claude Code slash command (``.claude/commands/<name>.md``) → a
    protoAgent user-facing slash skill (``user_facing: true`` + ``slash:``,
    ADR 0052). The body carries over verbatim; ``$ARGUMENTS`` semantics are
    documented in a translator note so the prompt still reads correctly."""
    meta, body = parse_frontmatter(_bounded_document(path, max_bytes))
    out = TranslatedSkill(name=slugify_name(path.stem), user_facing=True, slash=slugify_name(path.stem))
    desc = str(meta.get("description") or "").strip()
    if not desc:
        first = next((ln.strip() for ln in body.splitlines() if ln.strip()), "")
        desc = first[:200] or f"Imported /{out.name} command."
    out.description = desc[:DESCRIPTION_MAX]

    new_meta = _normalized_meta(out.name, out.description, source)
    new_meta["user_facing"] = True
    new_meta["slash"] = out.slash

    note = ""
    if "$ARGUMENTS" in body or re.search(r"\$\d", body):
        note = (
            "\n\n> Translator note: `$ARGUMENTS` (and `$1`, `$2`, …) below refer to the text "
            "the operator types after the slash command — substitute it when following this skill.\n"
        )
        out.warnings.append("command uses $ARGUMENTS — carried with an interpretation note")
    out.files["SKILL.md"] = (_render_frontmatter(new_meta) + "\n" + body.strip() + note).encode("utf-8")
    return out


@dataclass
class TranslatedSubagent:
    name: str
    description: str
    system_prompt: str
    tools: list[str] = field(default_factory=list)
    original_model: str = ""
    warnings: list[str] = field(default_factory=list)


def translate_subagent_md(path: Path, max_bytes: int = 65536) -> TranslatedSubagent:
    """A Claude Code subagent (``.claude/agents/<name>.md``) → the fields of a
    protoAgent ``SubagentConfig``. Tool names map through TOOL_MAP (unmapped →
    flagged, never guessed); the model is recorded but left blank so the
    subagent inherits the instance's aux/main model (gateway aliases differ
    per instance)."""
    meta, body = parse_frontmatter(_bounded_document(path, max_bytes))
    out = TranslatedSubagent(
        name=slugify_name(meta.get("name") or path.stem),
        description=str(meta.get("description") or "").strip()[:DESCRIPTION_MAX]
        or f"Imported Claude Code subagent {path.stem}.",
        system_prompt=body.strip(),
    )
    raw_tools = meta.get("tools")
    if isinstance(raw_tools, str):
        raw_tools = [t.strip() for t in raw_tools.split(",") if t.strip()]
    for t in raw_tools or []:
        counterpart = TOOL_MAP.get(t, None if t[:1].isupper() else t)
        if counterpart is None:
            out.warnings.append(f"tool {t!r} has no protoAgent counterpart (omitted)")
        else:
            out.tools.append(counterpart)
    out.tools = sorted(set(out.tools))
    if meta.get("model"):
        out.original_model = str(meta["model"])
        out.warnings.append(f"model {out.original_model!r} not carried over — the subagent inherits the instance model")
    return out


def translate_mcp_servers(cc_servers: dict) -> tuple[list[dict], list[str]]:
    """Claude Code ``mcpServers`` JSON → protoAgent ``mcp.servers`` entries.

    Near shape-identical: stdio keeps command/args/env; http/sse keep
    url/headers. Follows the same normalization rules as the console's
    add-server form (name + transport required, blank-stripped)."""
    entries: list[dict] = []
    warnings: list[str] = []
    for name, cc in (cc_servers or {}).items():
        if not isinstance(cc, dict):
            warnings.append(f"{name}: unrecognized entry shape (skipped)")
            continue
        cc_type = str(cc.get("type") or cc.get("transport") or ("stdio" if cc.get("command") else "http")).lower()
        entry: dict = {"name": str(name).strip()}
        if cc_type == "stdio":
            if not cc.get("command"):
                warnings.append(f"{name}: stdio entry without a command (skipped)")
                continue
            entry["transport"] = "stdio"
            entry["command"] = str(cc["command"]).strip()
            args = cc.get("args")
            if isinstance(args, list) and args:
                entry["args"] = [str(a) for a in args]
        else:
            if not cc.get("url"):
                warnings.append(f"{name}: {cc_type} entry without a url (skipped)")
                continue
            entry["transport"] = "sse" if cc_type == "sse" else "http"
            entry["url"] = str(cc["url"]).strip()
            headers = cc.get("headers")
            if isinstance(headers, dict) and headers:
                entry["headers"] = {str(k): str(v) for k, v in headers.items()}
        env = cc.get("env")
        if isinstance(env, dict) and env:
            entry["env"] = {str(k): str(v) for k, v in env.items()}
        entries.append(entry)
    return entries, warnings


def memory_chunks(memory_dir: Path, max_bytes: int = 65536) -> list[tuple[str, str]]:
    """A Claude Code project memory dir → (heading, content) chunks for
    knowledge ingestion. One chunk per topic file; MEMORY.md (the index) is
    derivative and skipped. Frontmatter is folded into the heading."""
    chunks: list[tuple[str, str]] = []
    fence = FencedRoot("memory", memory_dir)
    for f in sorted(memory_dir.glob("*.md")):
        if f.name == "MEMORY.md":
            continue
        text, truncated = fence.read_text(f.name, max_bytes)
        if truncated:
            raise ValueError(f"{f.name} exceeds max_read_bytes={max_bytes}")
        meta, body = parse_frontmatter(text)
        heading = str(meta.get("description") or meta.get("name") or f.stem)
        content = body.strip()
        if content:
            chunks.append((heading, content))
    return chunks
