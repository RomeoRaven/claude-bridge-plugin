"""The pure translation layer — every mapping and the license refusal."""

from __future__ import annotations

import secrets

import pytest

from claude_bridge import translate as tr


def test_slugify_enforces_spec_rules():
    assert tr.slugify_name("My Cool Skill!") == "my-cool-skill"
    assert tr.slugify_name("--weird--name--") == "weird-name"
    assert tr.slugify_name("x" * 100).startswith("x") and len(tr.slugify_name("x" * 100)) <= tr.NAME_MAX
    assert tr.SPEC_NAME_RE.match(tr.slugify_name("PDF_Processing (v2)"))


def test_translate_skill_normalizes(tmp_path):
    d = tmp_path / "My_Skill"
    (d / "scripts").mkdir(parents=True)
    (d / "SKILL.md").write_text(
        "---\nname: My_Skill\ndescription: "
        + "d" * 1100
        + "\nallowed-tools: Bash(git:*) Read TodoWrite\n---\n\nDo the thing.\n"
    )
    (d / "scripts" / "run.py").write_text("print('hi')\n")

    out = tr.translate_skill_dir(d, source="test")
    assert out is not None and out.name == "my-skill"
    assert len(out.description) == tr.DESCRIPTION_MAX
    meta, body = tr.parse_frontmatter(out.skill_md)
    assert meta["name"] == "my-skill"
    assert meta["tools"] == ["read_file", "run_command"]  # TodoWrite dropped, Bash(git:*) → run_command
    assert meta["metadata"]["imported-from"] == "test"
    assert "Do the thing." in body
    assert "scripts/run.py" in out.files
    assert any("truncated" in w for w in out.warnings)
    assert any("TodoWrite" in w for w in out.warnings)


def test_translate_skill_refuses_anthropic_material(tmp_path):
    d = tmp_path / "docx"
    d.mkdir()
    (d / "SKILL.md").write_text("---\nname: docx\ndescription: Word.\n---\n\nBody.\n")
    (d / "LICENSE.txt").write_text("© 2025 Anthropic, PBC. All rights reserved.")
    assert tr.translate_skill_dir(d) is None

    lic = tmp_path / "lic"
    lic.mkdir()
    (lic / "SKILL.md").write_text(
        "---\nname: lic\ndescription: X.\nlicense: Anthropic terms of service\n---\n\nBody.\n"
    )
    assert tr.translate_skill_dir(lic) is None


def test_translate_skill_accepts_long_non_anthropic_license(tmp_path):
    skill = tmp_path / "licensed-skill"
    skill.mkdir()
    (skill / "SKILL.md").write_text("---\nname: licensed-skill\ndescription: Licensed.\n---\n\nBody.\n")
    (skill / "LICENSE").write_text("MIT License\n" + "Permission is hereby granted.\n" * 100)

    assert tr.translate_skill_dir(skill) is not None


def test_translate_skill_refuses_anthropic_text_after_license_preview(tmp_path):
    skill = tmp_path / "licensed-skill"
    skill.mkdir()
    (skill / "SKILL.md").write_text("---\nname: licensed-skill\ndescription: Licensed.\n---\n\nBody.\n")
    (skill / "LICENSE").write_text("x" * 2100 + "Anthropic material")

    assert tr.translate_skill_dir(skill, max_bytes=4096) is None


def test_translate_skill_refuses_truncated_license_notice(tmp_path):
    skill = tmp_path / "licensed-skill"
    skill.mkdir()
    (skill / "SKILL.md").write_text("---\nname: licensed-skill\ndescription: Licensed.\n---\n\nBody.\n")
    (skill / "NOTICE").write_text("x" * 128 + "Anthropic material")

    assert tr.translate_skill_dir(skill, max_bytes=128) is None


def test_translate_skill_refuses_symlinked_root_escape(tmp_path):
    marker = secrets.token_urlsafe(24)
    outside = tmp_path / "outside-skill"
    outside.mkdir()
    (outside / "SKILL.md").write_text(f"---\nname: outside\ndescription: {marker}\n---\n\n{marker}\n")
    linked = tmp_path / "linked-skill"
    linked.symlink_to(outside, target_is_directory=True)

    assert tr.translate_skill_dir(linked) is None


def test_translate_skill_refuses_symlinked_supporting_file_escape(tmp_path):
    marker = secrets.token_urlsafe(24)
    skill = tmp_path / "skill"
    skill.mkdir()
    (skill / "SKILL.md").write_text("---\nname: safe-skill\ndescription: Safe.\n---\n\nBody.\n")
    outside = tmp_path / "outside.txt"
    outside.write_text(marker)
    (skill / "linked.txt").symlink_to(outside)

    translated = tr.translate_skill_dir(skill)

    assert translated is not None
    assert marker.encode() not in translated.files.values()
    assert "linked.txt" not in translated.files


def test_translate_skill_refuses_truncated_supporting_file(tmp_path):
    skill = tmp_path / "skill"
    (skill / "scripts").mkdir(parents=True)
    (skill / "SKILL.md").write_text("---\nname: safe-skill\ndescription: Safe.\n---\n\nBody.\n")
    (skill / "scripts" / "run.py").write_bytes(b"x" * 129)

    with pytest.raises(ValueError, match=r"scripts/run\.py exceeds max_read_bytes=128"):
        tr.translate_skill_dir(skill, max_bytes=128)


def test_translate_command_becomes_slash_skill(tmp_path):
    p = tmp_path / "standup.md"
    p.write_text("---\ndescription: Progress report.\nallowed-tools: Bash\n---\n\nReport on $ARGUMENTS now.\n")
    out = tr.translate_command_md(p)
    meta, body = tr.parse_frontmatter(out.skill_md)
    assert meta["user_facing"] is True and meta["slash"] == "standup"
    assert meta["description"] == "Progress report."
    assert "$ARGUMENTS" in body and "Translator note" in body


def test_translate_subagent_maps_tools_and_drops_model(tmp_path):
    p = tmp_path / "reviewer.md"
    p.write_text(
        "---\nname: Reviewer\ndescription: Reviews.\ntools: Read, Grep, Bash, TodoWrite\nmodel: opus\n---\n\nYou review.\n"
    )
    s = tr.translate_subagent_md(p)
    assert s.name == "reviewer" and s.system_prompt == "You review."
    assert s.tools == ["read_file", "run_command", "search_files"]
    assert s.original_model == "opus"
    assert any("TodoWrite" in w for w in s.warnings) and any("model" in w for w in s.warnings)


def test_translate_mcp_shapes():
    entries, warnings = tr.translate_mcp_servers(
        {
            "gh": {"command": "gh-mcp", "args": ["serve"], "env": {"TOKEN": "s3cret"}},
            "docs": {"type": "http", "url": "https://x.test/mcp", "headers": {"Authorization": "Bearer t"}},
            "events": {"type": "sse", "url": "https://y.test/sse"},
            "broken": {"type": "http"},
        }
    )
    by = {e["name"]: e for e in entries}
    assert by["gh"] == {
        "name": "gh",
        "transport": "stdio",
        "command": "gh-mcp",
        "args": ["serve"],
        "env": {"TOKEN": "s3cret"},
    }
    assert by["docs"]["transport"] == "http" and by["docs"]["headers"]["Authorization"] == "Bearer t"
    assert by["events"]["transport"] == "sse"
    assert "broken" not in by and any("broken" in w for w in warnings)


def test_memory_chunks_skips_index(tmp_path):
    (tmp_path / "MEMORY.md").write_text("# index\n")
    (tmp_path / "fact.md").write_text("---\nname: fact\ndescription: A fact hook\n---\n\nThe body.\n")
    (tmp_path / "empty.md").write_text("---\nname: empty\n---\n\n")
    chunks = tr.memory_chunks(tmp_path)
    assert chunks == [("A fact hook", "The body.")]
