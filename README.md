# claude-bridge-plugin

A [protoAgent](https://github.com/protoLabsAI/protoAgent) plugin that bridges
**Claude Code** state into protoAgent, for seamless pivoting between the two:
ask your agent what Claude Code knows about a project — its persistent memory,
past sessions, scratchpads, Cowork runs, and installed customizations.

Pairs with protoAgent's ACP runtime (ADR 0033/0082): ACP lets protoAgent *run*
Claude Code as a coding agent; this plugin lets it *read* Claude Code's
accumulated state.

## The three stores

| Store | Where | What |
|---|---|---|
| CLI | `~/.claude` | `projects/<slug>/` — session transcripts + persistent `memory/`; user `skills/`, `agents/`, `commands/`, `plugins/`, `settings.json` |
| Scratchpads | `/private/tmp/claude-<uid>` | per-session working files (ephemeral, wiped on reboot) |
| Cowork | `~/Library/Application Support/Claude/local-agent-mode-sessions` | desktop local-agent-mode sessions: metadata, `outputs/`, `uploads/`, `audit.jsonl` (transcripts stay in the sandbox VM) |

## Import tools (v0.2+ — translators)

All importers are **dry-run by default** (`apply=True` writes, after the
operator approves); existing material is never overwritten, and re-imports
skip + report.

- `claude_import_scan` — inventory everything importable (skills/commands/subagents/MCP/memory)
- `claude_import_skills` — user-authored skills → protoAgent skills (same open SKILL.md standard; spec-normalized names, provenance metadata)
- `claude_import_commands` — slash commands → `user_facing`/`slash` skills
- `claude_import_subagents` — subagent markdown → `SubagentConfig` (tools mapped, unmapped flagged; model inherited)
- `claude_import_mcp` — `mcpServers` configs → `mcp.servers` (replace-by-name merge)
- `claude_import_memory` — project memory → knowledge domain `claude-import` (undo: `knowledge_purge('claude-import')`). `limit=0` (default) imports **every** topic; a large project memory can hold 100+.
- `claude_import_claude_md` — a repo's `CLAUDE.md` (its *operating instructions* — run commands, gates, gotchas) → knowledge domain `claude-import`, so the agent can recall how the repo wants to be worked. It's instructions, not a persona — promote it into `SOUL.md` yourself if you want it always in context.
- `claude_hooks_report` — hooks are **report-only** (protoAgent's equivalent is middleware)

**License rule:** Anthropic-authored skills (Cowork `creatorType: anthropic`,
or Anthropic license text in the skill dir) are always refused — their license
prohibits redistribution. Only user-authored material migrates.

## Tools (v0.1 — explore, read-only)

- `claude_projects` — list projects or resolve a directory to its project
- `claude_memory` — MEMORY.md index + individual memory files for a directory
- `claude_sessions` — recent session transcripts; tail one by id
- `claude_scratchpad` — list/read a project's session scratchpads
- `claude_cowork` — list Cowork sessions; metadata + outputs + audit tail; read files
- `claude_inventory` — user- and project-level skills/subagents/commands/plugins/MCP
  servers (secret values redacted) and hooks

All reads go through a path fence (no absolute paths, no `..`, symlink escapes
refused) and are byte-capped. MCP env/header **values are never echoed** — key
names only.

## Roadmap

- v0.2 — translators: Claude skills → protoAgent skills (same open SKILL.md
  standard), slash commands → `user_facing`/`slash` skills, MCP servers →
  `mcp.servers` via `apply_settings`, subagent markdown → `SubagentConfig`;
  hooks report-only (protoAgent's equivalent is middleware, not a hook table).
- v0.3 — `claude_import_claude_md` (repo operating instructions → knowledge), and
  `claude_import_memory` now imports **all** topics by default (the old `limit=100`
  capped large project memories).
- v0.4 — `claude_export_snapshot`: package a translated setup as a **protoAgent agent
  snapshot** (ADR 0091) that `protoagent agent import` consumes directly. One artifact
  schema, not two — the bundle inherits that format's plan/consent gate, its
  `required_secrets` inventory and its REVIEW.md. MCP credential values are nulled and
  listed; memory is an opt-in seed that marks the bundle non-publishable; `CLAUDE.md` rides
  knowledge, not SOUL (it is instructions, not a persona).
- Later — console rail view for browsing memory; a `SOUL.md`-addendum mode for CLAUDE.md
  (always-in-context instead of retrievable).

## Install

```
plugin install https://github.com/protoLabsAI/claude-bridge-plugin
```

Then enable `claude_bridge` in the console (ships disabled — enabling is the
operator's trust decision). No pip deps, no network access.

## Development

```
python3 -m venv .venv && .venv/bin/pip install -r requirements-dev.txt ruff
.venv/bin/pytest -q          # host-free: no protoAgent checkout required
.venv/bin/ruff check .
```
