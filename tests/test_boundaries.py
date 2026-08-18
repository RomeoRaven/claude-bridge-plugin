from __future__ import annotations

from claude_bridge.boundaries import credential_free_url, complete_text
from claude_bridge.stores import FencedRoot


def test_credential_free_url_strips_credentials_query_and_fragment():
    value = "https://person:secret@example.test:8443/mcp?token=secret#fragment"

    assert credential_free_url(value) == "https://example.test:8443/mcp"


def test_credential_free_url_uses_caller_selected_invalid_sentinel():
    malformed = "https://example.test:invalid/mcp"

    assert credential_free_url(malformed) == ""
    assert credential_free_url(malformed, invalid="[redacted]") == "[redacted]"


def test_complete_text_accepts_complete_content_and_refuses_truncation(tmp_path):
    (tmp_path / "short.txt").write_text("complete")
    (tmp_path / "long.txt").write_text("x" * 9)
    fence = FencedRoot("test", tmp_path)

    assert complete_text(fence, "short.txt", 8) == "complete"

    try:
        complete_text(fence, "long.txt", 8)
    except ValueError as exc:
        assert str(exc) == "long.txt exceeds max_read_bytes=8"
    else:
        raise AssertionError("truncated content was accepted")
