"""Export a translated Claude Code setup as a protoAgent **agent snapshot** (ADR 0091).

The mirror of ``tools_import``: the same translators, aimed at a *file* instead of the live
agent. Where import says "fold this Claude Code state into the agent I'm running in", export
says "emit it as a portable recipe someone else can stand up".

**One artifact schema, not two.** The obvious thing to build here was a bespoke
"translated bundle" format. protoAgent already has one — ADR 0091's agent snapshot, which
`protoagent agent import` consumes, with its plan/consent gate, its `required_secrets`
inventory, and its review document. Inventing a second shape would have meant a second
importer, a second set of safety properties to get right, and a divergence to maintain. So
this emits `snapshot_version: 1` / `kind: agent-snapshot` and gets all of that for free.

What maps onto what:

===========================  ==========================================================
Claude Code                  Snapshot
===========================  ==========================================================
``skills/``, slash commands  ``skills/<name>/SKILL.md`` — re-seeded from disk at boot
``.claude/agents/*.md``      ``config.subagents``
``mcpServers``               ``config.mcp.servers`` (values nulled — see below)
``CLAUDE.md``                a knowledge doc — *instructions*, NOT the persona
memory topics                an **opt-in** knowledge seed
===========================  ==========================================================

Two properties inherited from ADR 0091 that this file must not break:

* **Secret-free.** MCP entries keep their env KEYS and null the values, and every nulled key
  is inventoried in ``required_secrets`` so the importer is prompted rather than handed a
  silently broken server. A Claude Code ``mcpServers`` block routinely holds live tokens;
  this is the one place they could leak into a file meant to be shared.
* **The MEMORY seed is opt-in and NOT publishable.** Memory topics are private project
  notes — no credentials, and quite possibly the last thing you want public. Including them
  flips the artifact from "shareable recipe" to "shareable like the source documents", and
  the review says so rather than leaving the reader with the earlier promise. ``CLAUDE.md``
  does **not** trigger that warning: it is normally committed to the repo it describes, and
  warning on it would make the warning wallpaper.

``CLAUDE.md`` deliberately does NOT become ``SOUL.md``. ``claude_import_claude_md`` already
made that call for the import direction — *"it's instructions, not a persona: this lands it
in knowledge (retrievable)"* — and routing repo doctrine into a persona is the accretion
protoAgent treats as a defect (ADR 0079/0081). Export follows the same rule, so a bundle
round-trips to the same place the import tools would put it. A snapshot with no persona
leaves the imported agent on its default SOUL, which is the honest outcome.

``plugins`` is always empty: a Claude Code setup has no protoAgent plugins to pin. That is
worth stating rather than leaving implicit — it means importing one of these bundles
installs and runs **no** third-party code, so the import gate stays on its light path.
"""

from __future__ import annotations

import io
import zipfile
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from .boundaries import credential_free_url

SNAPSHOT_MANIFEST = "agent.snapshot.yaml"
#: The ADR 0091 schema version this emits. Bumping protoAgent's version without bumping
#: this is how the two silently drift — an importer refuses an unknown version outright, so
#: a mismatch is loud rather than subtly wrong.
SNAPSHOT_VERSION = 1


@dataclass
class BundlePlan:
    """What a bundle would contain — reportable without writing anything, so the plugin's
    dry-run-by-default convention holds for export exactly as it does for import."""

    agent_name: str = "claude-import"
    skills: list[str] = field(default_factory=list)
    subagents: list[str] = field(default_factory=list)
    mcp_servers: list[str] = field(default_factory=list)
    required_secrets: list[dict] = field(default_factory=list)
    knowledge_topics: int = 0
    has_claude_md: bool = False
    has_soul: bool = False
    warnings: list[str] = field(default_factory=list)

    @property
    def carries_knowledge(self) -> bool:
        return self.knowledge_topics > 0

    def render(self) -> str:
        lines = [f"Bundle plan for {self.agent_name!r} (protoAgent agent snapshot, ADR 0091):"]
        lines.append(
            f"  SOUL.md          {'yes' if self.has_soul else 'none (CLAUDE.md is instructions, not a persona)'}"
        )
        if self.has_claude_md:
            lines.append("  CLAUDE.md        → knowledge (retrievable operating instructions)")
        lines.append(
            f"  skills           {len(self.skills)}" + (f" — {', '.join(self.skills[:6])}" if self.skills else "")
        )
        lines.append(
            f"  subagents        {len(self.subagents)}"
            + (f" — {', '.join(self.subagents[:6])}" if self.subagents else "")
        )
        lines.append(
            f"  MCP servers      {len(self.mcp_servers)}"
            + (f" — {', '.join(self.mcp_servers[:6])}" if self.mcp_servers else "")
        )
        lines.append("  plugins          0 (a Claude Code setup pins none — importing runs no third-party code)")
        if self.required_secrets:
            lines.append(f"  credentials the importer must supply ({len(self.required_secrets)}):")
            for r in self.required_secrets:
                lines.append(f"    {r['name']}")
        if self.carries_knowledge:
            lines.append(f"  knowledge seed   {self.knowledge_topics} topic(s) — NOT publishable, see REVIEW.md")
        else:
            lines.append("  knowledge seed   none (pass include_memory=True to add one)")
        for w in self.warnings:
            lines.append(f"  ! {w}")
        return "\n".join(lines)


def _null_mcp_secrets(servers: list[dict]) -> tuple[list[dict], list[dict]]:
    """Strip values from credential-shaped MCP env/header entries, keeping the KEYS.

    The importer needs to know *which* variables a server wants; it must not receive their
    values. Returns (clean servers, required_secrets rows)."""
    import copy

    clean = copy.deepcopy(servers)
    required: list[dict] = []
    for entry in clean:
        if not isinstance(entry, dict):
            continue
        name = str(entry.get("name") or "?")
        args = entry.get("args")
        if isinstance(args, list) and args:
            had = any(str(value or "").strip() for value in args)
            entry["args"] = []
            required.append(
                {
                    "name": f"mcp.{name}.args",
                    "kind": "mcp_args",
                    "description": f"Command arguments for MCP server `{name}` (removed on export).",
                    "was_set": had,
                }
            )
        if entry.get("url"):
            original_url = str(entry["url"])
            entry["url"] = credential_free_url(original_url)
            if entry["url"] != original_url:
                required.append(
                    {
                        "name": f"mcp.{name}.url",
                        "kind": "mcp_url",
                        "description": f"Credential-bearing URL parts for MCP server `{name}` (removed on export).",
                        "was_set": True,
                    }
                )
        for field_name in ("env", "headers"):
            values = entry.get(field_name)
            if not isinstance(values, dict):
                continue
            for var in list(values):
                had = bool(str(values.get(var) or "").strip())
                values[var] = ""
                required.append(
                    {
                        "name": f"mcp.{name}.{field_name}.{var}",
                        "kind": "mcp_env" if field_name == "env" else "mcp_header",
                        "description": f"`{var}` for MCP server `{name}` (value nulled on export).",
                        "was_set": had,
                    }
                )
    return clean, required


def _review(plan: BundlePlan, exported_at: str) -> str:
    """The disclosure that travels INSIDE the zip, mirroring protoAgent's own export.

    In the artifact rather than only in a tool reply, so whoever opens this file later can
    still see what was stripped and what they must supply."""
    lines = [
        f"# Snapshot review — {plan.agent_name}",
        "",
        f"Translated from a Claude Code setup and exported {exported_at}.",
        "",
        "A **recipe, not a backup**: persona, skills, subagents and MCP definitions. No",
        "conversation history and no credentials. Importing yields a *fresh* agent.",
        "",
        "**No plugins are pinned**, so importing this installs and runs no third-party code.",
        "",
    ]
    if plan.carries_knowledge:
        lines += [
            "> [!WARNING]",
            "> **This bundle carries a knowledge seed — do NOT treat it as publishable.**",
            f"> {plan.knowledge_topics} memory topic(s) travel with it, as text. They hold no",
            "> credentials and may still be private: project detail, client names, internal",
            "> notes. Share this the way you would share those documents themselves.",
            "",
        ]
    lines += ["## Credentials the importer must supply", ""]
    if plan.required_secrets:
        lines += ["| Credential | Set in the source |", "| --- | --- |"]
        lines += [f"| `{r['name']}` | {'yes' if r['was_set'] else 'no'} |" for r in plan.required_secrets]
        lines += ["", "*Names only — no values travel. Your Claude Code `mcpServers` values stayed behind.*"]
    else:
        lines.append("*None — no credential-shaped MCP variables were found.*")
    if plan.warnings:
        lines += ["", "## Translation notes", ""] + [f"- {w}" for w in plan.warnings]
    lines += [
        "",
        "## Import it",
        "",
        "```bash",
        "protoagent agent import <this-file>.zip --dry-run   # the plan; changes nothing",
        "protoagent agent import <this-file>.zip --name my-agent --yes",
        "```",
        "",
    ]
    return "\n".join(lines)


def build_bundle(
    *,
    agent_name: str,
    skills: list,
    subagents: list,
    mcp_servers: list[dict] | None = None,
    claude_md: str = "",
    memory: list[tuple[str, str]] | None = None,
    now: datetime | None = None,
) -> tuple[bytes, BundlePlan]:
    """Build the snapshot zip in memory. Returns ``(zip_bytes, plan)``.

    Takes already-translated pieces rather than paths so it stays pure — the same reason
    ``translate.py`` is separate from ``tools_import.py``, and what lets the shape be tested
    without a Claude Code install.
    """
    import yaml

    stamp = (now or datetime.now(timezone.utc)).isoformat()
    plan = BundlePlan(agent_name=agent_name, has_claude_md=bool(claude_md.strip()))

    clean_servers, required = _null_mcp_secrets(list(mcp_servers or []))
    plan.mcp_servers = [str(s.get("name") or "?") for s in clean_servers if isinstance(s, dict)]
    plan.required_secrets = required

    config: dict = {"identity": {"name": agent_name}}
    if clean_servers:
        config["mcp"] = {"servers": clean_servers}
    if subagents:
        config["subagents"] = [
            {
                "name": s.name,
                "description": s.description,
                "system_prompt": s.system_prompt,
                "tools": list(s.tools),
            }
            for s in subagents
        ]
        plan.subagents = [s.name for s in subagents]
        for s in subagents:
            plan.warnings.extend(f"subagent {s.name}: {w}" for w in s.warnings)

    for sk in skills:
        plan.skills.append(sk.name)
        plan.warnings.extend(f"skill {sk.name}: {w}" for w in sk.warnings)

    topics = list(memory or [])
    plan.knowledge_topics = len(topics)
    # Two knowledge docs, tracked separately because they answer different questions.
    # CLAUDE.md is repo-committed instructions; memory topics are private notes, and only
    # THEY make the bundle unshareable.
    kb_docs: dict[str, str] = {}
    if claude_md.strip():
        kb_docs["claude-md"] = f"## Operating instructions (CLAUDE.md)\n\n{claude_md.strip()}"
    if topics:
        kb_docs["claude-import"] = "\n\n".join(f"## {h}\n\n{c}" for h, c in topics)

    manifest = {
        "snapshot_version": SNAPSHOT_VERSION,
        "kind": "agent-snapshot",
        "exported_at": stamp,
        "agent": {"name": agent_name},
        # Always empty — a Claude Code setup pins no protoAgent plugins. Import's
        # code-execution gate therefore stays on its light path.
        "plugins": [],
        "config": config,
        "soul": None,
        "required_secrets": required,
        "knowledge": (
            {"domains": {k: (len(topics) if k == "claude-import" else 1) for k in kb_docs}} if kb_docs else None
        ),
        "excludes": {
            "runtime_state": "none carried — this yields a FRESH agent",
            "credentials": ["Claude Code mcpServers values"],
            "persona": "CLAUDE.md is operating INSTRUCTIONS, so it rides knowledge — not SOUL.md (ADR 0079/0081)",
        },
        "translated_from": "claude-code",
    }

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr(SNAPSHOT_MANIFEST, yaml.safe_dump(manifest, sort_keys=False, allow_unicode=True))
        for sk in skills:
            for rel, body in sk.files.items():
                # `skills/<label>/<name>/…` is the layout protoAgent's importer collapses
                # into the new agent's own skills dir.
                zf.writestr(f"skills/claude/{sk.name}/{rel}", body)
        for domain, body in sorted(kb_docs.items()):
            zf.writestr(f"knowledge/{domain}.md", body)
        zf.writestr("REVIEW.md", _review(plan, stamp))
    return buf.getvalue(), plan


def write_bundle(data: bytes, out: Path) -> Path:
    """Write the zip, treating a non-`.zip` target as a directory to write into.

    Keyed on the SUFFIX rather than ``is_dir()``: an operator naming an output directory
    that doesn't exist yet is the common case (`out=~/bundles`), and ``is_dir()`` is False
    for it — so the earlier check silently produced a *file* called `bundles` with a zip
    inside it. Anything ending in `.zip` is taken as the filename the caller wants.
    """
    out = out.expanduser()
    if out.suffix.lower() != ".zip":
        out = out / "claude-import-snapshot.zip"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_bytes(data)
    return out


def manifest_of(data: bytes) -> dict:
    """Read back the manifest — used by the tests to assert the emitted SHAPE, which is the
    actual contract with protoAgent's importer."""
    import yaml

    with zipfile.ZipFile(io.BytesIO(data)) as zf:
        return yaml.safe_load(zf.read(SNAPSHOT_MANIFEST).decode("utf-8")) or {}


__all__ = ["BundlePlan", "build_bundle", "write_bundle", "manifest_of", "SNAPSHOT_VERSION"]
