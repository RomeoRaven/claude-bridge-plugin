"""Shared helpers for security and bounded-read boundaries."""

from __future__ import annotations

from urllib.parse import urlsplit, urlunsplit

from .stores import FencedRoot


def complete_text(fence: FencedRoot, rel: str, max_bytes: int) -> str:
    """Read one complete fenced text file or refuse a truncated result."""
    text, truncated = fence.read_text(rel, max_bytes)
    if truncated:
        raise ValueError(f"{rel} exceeds max_read_bytes={max_bytes}")
    return text


def credential_free_url(value: object, *, invalid: str = "") -> str:
    """Keep an endpoint location while removing credential-bearing URL parts."""
    raw = str(value or "").strip()
    try:
        parsed = urlsplit(raw)
        host = parsed.hostname or ""
        if ":" in host and not host.startswith("["):
            host = f"[{host}]"
        port = parsed.port
        netloc = host + (f":{port}" if port is not None else "")
        if not parsed.scheme or not netloc:
            return invalid
        return urlunsplit((parsed.scheme, netloc, parsed.path, "", ""))
    except ValueError:
        return invalid
