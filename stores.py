"""Store adapters over Claude Code's on-disk state.

Three stores, each fenced to its root:

- cli:        ~/.claude — projects/<slug>/ (session transcripts + memory/),
              skills/, agents/, commands/, plugins/, todos/, plans/, settings.json
- scratchpad: /private/tmp/claude-<uid>/<slug>/<session>/scratchpad — ephemeral
- cowork:     ~/Library/Application Support/Claude/local-agent-mode-sessions/
              <account>/<org>/local_<session>/ (outputs/ + uploads/ + audit.jsonl
              + sidecar local_<session>.json). Cowork transcripts live inside the
              sandbox VM, not on the host — metadata/outputs/audit is all there is.

Claude Code keys the CLI store by a project *slug*: the absolute path with every
non-alphanumeric character replaced by "-" (so both "/" and "." collapse to "-").
"""

from __future__ import annotations

import os
import re
import sys
from pathlib import Path


def project_slug_candidates(directory: str) -> list[str]:
    """Slugs a directory may be stored under, most likely first.

    The observed encoding maps every non-alphanumeric char to "-"; keep an
    underscore-preserving variant as a fallback in case "_" survives.
    """
    raw = str(Path(directory).expanduser())
    strict = re.sub(r"[^A-Za-z0-9]", "-", raw)
    loose = re.sub(r"[^A-Za-z0-9_]", "-", raw)
    return [strict] if strict == loose else [strict, loose]


def default_scratchpad_root() -> Path:
    uid = os.getuid() if hasattr(os, "getuid") else 0
    base = Path("/private/tmp") if sys.platform == "darwin" else Path("/tmp")
    return base / f"claude-{uid}"


class FencedRoot:
    """Resolve relative paths under a fixed root, refusing every escape.

    Same contract as protoAgent's operator fs fence (ADR 0007): no absolute
    paths, no "..", and the resolved target (symlinks followed) must stay
    inside the resolved root.
    """

    def __init__(self, name: str, root: str | Path):
        self.name = name
        self.root = Path(root).expanduser().resolve()

    def exists(self) -> bool:
        return self.root.is_dir()

    def resolve(self, rel: str = "") -> Path:
        rel = (rel or "").strip()
        if rel.startswith(("/", "~")) or ".." in Path(rel).parts:
            raise ValueError(f"path escapes the {self.name} store: {rel!r}")
        target = (self.root / rel).resolve() if rel else self.root
        if target != self.root and not target.is_relative_to(self.root):
            raise ValueError(f"path escapes the {self.name} store: {rel!r}")
        return target

    def read_text(self, rel: str, max_bytes: int) -> tuple[str, bool]:
        """Read a fenced file, capped at max_bytes. Returns (text, truncated)."""
        data, truncated = self.read_bytes(rel, max_bytes)
        return data.decode("utf-8", errors="replace"), truncated

    def read_bytes(self, rel: str, max_bytes: int) -> tuple[bytes, bool]:
        """Read at most max_bytes plus one sentinel byte from a fenced file."""
        target = self.resolve(rel)
        with target.open("rb") as handle:
            data = handle.read(max_bytes + 1)
        truncated = len(data) > max_bytes
        return data[:max_bytes], truncated


class ClaudeStores:
    """The three fenced roots, built from plugin config."""

    def __init__(self, cfg: dict | None = None):
        cfg = cfg or {}
        self.max_read_bytes = int(cfg.get("max_read_bytes") or 65536)
        self.transcript_tail = int(cfg.get("transcript_tail") or 20)
        self.cli = FencedRoot("cli", cfg.get("cli_root") or "~/.claude")
        self.scratchpad = FencedRoot("scratchpad", cfg.get("scratchpad_root") or default_scratchpad_root())
        self.cowork = FencedRoot(
            "cowork",
            cfg.get("cowork_root") or "~/Library/Application Support/Claude/local-agent-mode-sessions",
        )

    def find_project(self, directory: str) -> tuple[str, Path] | None:
        """Map a directory to its fenced ~/.claude/projects entry, if one exists."""
        for slug in project_slug_candidates(directory):
            try:
                candidate = self.cli.resolve(f"projects/{slug}")
            except (OSError, ValueError):
                continue
            if candidate.is_dir():
                return slug, candidate
        return None
