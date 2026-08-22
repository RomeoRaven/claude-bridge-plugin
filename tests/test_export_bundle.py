"""Export a translated Claude Code setup as a protoAgent agent snapshot (v0.4).

The contract this file defends is **the format**: protoAgent's `agent import` refuses a
manifest whose `snapshot_version` or `kind` it doesn't recognize, so a drift here is a
bundle nobody can open. And the property that makes such a bundle shareable at all is that
your Claude Code `mcpServers` tokens stay on your machine — the conftest fixture puts a real
one in `settings.json`, and the litmus greps the built zip's bytes for it.
"""

from __future__ import annotations

import io
import secrets
import zipfile
from pathlib import Path

import claude_bridge  # noqa: F401 — registers the synthetic package (conftest)
import pytest
import yaml
from claude_bridge.export_bundle import SNAPSHOT_VERSION, build_bundle, manifest_of
from claude_bridge.stores import project_slug_candidates
from claude_bridge.tools_export import build_export_tools
from claude_bridge.translate import TranslatedSkill, TranslatedSubagent

from tests.conftest import PROJECT_DIR, SECRET


def _skill(name="demo-skill"):
    return TranslatedSkill(name=name, files={"SKILL.md": b"---\nname: demo\n---\n\nBody.\n"}, description="d")


def _subagent(name="helper"):
    return TranslatedSubagent(name=name, description="A helper.", system_prompt="You help.", tools=["read_file"])


# ── the format contract with protoAgent's importer ───────────────────────────────────


class TestSnapshotShape:
    def test_emits_the_version_and_kind_the_importer_requires(self):
        """`inspect_snapshot` refuses an unknown version or kind outright — so these two
        keys ARE the contract. Anything else in the manifest is detail."""
        m = manifest_of(build_bundle(agent_name="x", skills=[], subagents=[])[0])
        assert m["snapshot_version"] == SNAPSHOT_VERSION == 1
        assert m["kind"] == "agent-snapshot"

    def test_the_manifest_is_the_first_member_and_correctly_named(self):
        data, _ = build_bundle(agent_name="x", skills=[], subagents=[])
        with zipfile.ZipFile(io.BytesIO(data)) as zf:
            assert "agent.snapshot.yaml" in zf.namelist()

    def test_pins_no_plugins(self):
        """A Claude Code setup has none to pin — which is why importing one of these
        bundles installs and runs no third-party code, keeping the import gate light."""
        m = manifest_of(build_bundle(agent_name="x", skills=[], subagents=[])[0])
        assert m["plugins"] == []

    def test_records_where_it_came_from(self):
        m = manifest_of(build_bundle(agent_name="x", skills=[], subagents=[])[0])
        assert m["translated_from"] == "claude-code"

    def test_skills_land_where_the_importer_looks(self):
        """protoAgent's importer collapses `skills/<label>/…` into the new agent's own
        skills dir — the path shape matters, not just the presence."""
        data, _ = build_bundle(agent_name="x", skills=[_skill()], subagents=[])
        with zipfile.ZipFile(io.BytesIO(data)) as zf:
            assert "skills/claude/demo-skill/SKILL.md" in zf.namelist()

    def test_subagents_ride_the_config(self):
        m = manifest_of(build_bundle(agent_name="x", skills=[], subagents=[_subagent()])[0])
        sub = m["config"]["subagents"][0]
        assert sub["name"] == "helper" and sub["tools"] == ["read_file"]

    def test_claude_md_rides_knowledge_NOT_the_persona(self):
        """`claude_import_claude_md` already decided this for the import direction — it is
        "instructions, not a persona". Routing repo doctrine into SOUL is the accretion
        protoAgent treats as a defect (ADR 0079/0081), so export follows the same rule and
        a bundle round-trips to where the import tools would have put it."""
        data, plan = build_bundle(agent_name="x", skills=[], subagents=[], claude_md="# Rules\n\nBe careful.\n")
        assert plan.has_claude_md is True
        assert plan.has_soul is False
        assert manifest_of(data)["soul"] is None
        with zipfile.ZipFile(io.BytesIO(data)) as zf:
            assert "SOUL.md" not in zf.namelist()
            assert "Be careful." in zf.read("knowledge/claude-md.md").decode()

    def test_claude_md_alone_does_NOT_make_the_bundle_unpublishable(self):
        """CLAUDE.md is normally committed to the repo it describes. Warning on it would
        make the warning wallpaper — the flag is for PRIVATE memory topics."""
        data, _ = build_bundle(agent_name="x", skills=[], subagents=[], claude_md="# Rules\n")
        with zipfile.ZipFile(io.BytesIO(data)) as zf:
            assert "publishable" not in zf.read("REVIEW.md").decode()


# ── the credential boundary ──────────────────────────────────────────────────────────


class TestSecretsStayBehind:
    def _servers(self):
        return [
            {
                "name": "github",
                "transport": "stdio",
                "command": "gh-mcp",
                "args": ["serve", "--token", SECRET],
                "env": {"GH_TOKEN": SECRET, "GH_HOST": "github.com"},
            },
            {
                "name": "vendor",
                "transport": "http",
                "url": "https://v.example",
                "headers": {"Authorization": f"Bearer {SECRET}"},
            },
        ]

    def test_no_mcp_token_survives_anywhere_in_the_zip(self):
        """The litmus, greping the raw bytes — a Claude Code `mcpServers` block routinely
        holds live tokens, and this bundle is meant to be handed to someone."""
        data, _ = build_bundle(agent_name="x", skills=[], subagents=[], mcp_servers=self._servers())
        assert SECRET.encode() not in data

    def test_decompressed_snapshot_strips_every_mcp_value_and_url_credential_component(self):
        marker = secrets.token_urlsafe(24)
        servers = [
            {
                "name": "private-endpoint",
                "transport": "http",
                "url": f"https://person:{marker}@example.test/mcp?opaque={marker}#fragment-{marker}",
                "env": {"MODE": marker},
                "headers": {"X-Custom": marker},
            }
        ]

        data, _ = build_bundle(agent_name="x", skills=[], subagents=[], mcp_servers=servers)
        manifest = manifest_of(data)
        rendered = yaml.safe_dump(manifest)
        exported = manifest["config"]["mcp"]["servers"][0]

        assert marker not in rendered
        assert exported["url"] == "https://example.test/mcp"
        assert exported["env"] == {"MODE": ""}
        assert exported["headers"] == {"X-Custom": ""}

    def test_the_keys_survive_so_the_importer_knows_what_to_set(self):
        m = manifest_of(build_bundle(agent_name="x", skills=[], subagents=[], mcp_servers=self._servers())[0])
        github = next(s for s in m["config"]["mcp"]["servers"] if s["name"] == "github")
        assert github["env"] == {"GH_TOKEN": "", "GH_HOST": ""}
        assert github["command"] == "gh-mcp"

    def test_stdio_arguments_stay_behind_and_are_inventoried(self):
        m = manifest_of(build_bundle(agent_name="x", skills=[], subagents=[], mcp_servers=self._servers())[0])
        github = next(s for s in m["config"]["mcp"]["servers"] if s["name"] == "github")
        names = {row["name"] for row in m["required_secrets"]}

        assert github["args"] == []
        assert "mcp.github.args" in names

    def test_every_env_value_stays_behind_while_its_key_survives(self):
        m = manifest_of(build_bundle(agent_name="x", skills=[], subagents=[], mcp_servers=self._servers())[0])
        github = next(s for s in m["config"]["mcp"]["servers"] if s["name"] == "github")
        assert github["env"]["GH_HOST"] == ""

    def test_every_nulled_key_is_inventoried(self):
        _, plan = build_bundle(agent_name="x", skills=[], subagents=[], mcp_servers=self._servers())
        names = {r["name"] for r in plan.required_secrets}
        assert "mcp.github.env.GH_TOKEN" in names
        assert "mcp.github.env.GH_HOST" in names
        assert "mcp.vendor.headers.Authorization" in names
        assert all(r["was_set"] for r in plan.required_secrets)

    def test_the_inventory_carries_no_values(self):
        _, plan = build_bundle(agent_name="x", skills=[], subagents=[], mcp_servers=self._servers())
        assert SECRET not in yaml.safe_dump(plan.required_secrets)


# ── the knowledge seed ───────────────────────────────────────────────────────────────


class TestKnowledgeSeed:
    TOPICS = [("A fact", "The fact body."), ("Another", "More detail.")]

    def test_absent_by_default(self):
        data, plan = build_bundle(agent_name="x", skills=[], subagents=[])
        assert plan.carries_knowledge is False
        with zipfile.ZipFile(io.BytesIO(data)) as zf:
            assert not any(n.startswith("knowledge/") for n in zf.namelist())

    def test_included_when_asked_for(self):
        data, plan = build_bundle(agent_name="x", skills=[], subagents=[], memory=self.TOPICS)
        assert plan.knowledge_topics == 2
        with zipfile.ZipFile(io.BytesIO(data)) as zf:
            body = zf.read("knowledge/claude-import.md").decode()
        assert "The fact body." in body and "## A fact" in body

    def test_review_retracts_publishability(self):
        """Mirrors protoAgent's own rule: memory topics hold no credentials and may still
        be private, so the artifact stops being safe to publish."""
        data, _ = build_bundle(agent_name="x", skills=[], subagents=[], memory=self.TOPICS)
        with zipfile.ZipFile(io.BytesIO(data)) as zf:
            review = zf.read("REVIEW.md").decode()
        assert "do NOT treat it as publishable" in review

    def test_memory_topics_DO_make_it_unpublishable(self):
        data, _ = build_bundle(agent_name="x", skills=[], subagents=[], memory=self.TOPICS)
        with zipfile.ZipFile(io.BytesIO(data)) as zf:
            assert "do NOT treat it as publishable" in zf.read("REVIEW.md").decode()

    def test_a_seedless_review_makes_no_such_claim(self):
        """The warning must not become wallpaper — it appears only when true."""
        data, _ = build_bundle(agent_name="x", skills=[], subagents=[])
        with zipfile.ZipFile(io.BytesIO(data)) as zf:
            review = zf.read("REVIEW.md").decode()
        assert "publishable" not in review

    def test_manifest_declares_the_domain_for_the_importer(self):
        m = manifest_of(build_bundle(agent_name="x", skills=[], subagents=[], memory=self.TOPICS)[0])
        assert m["knowledge"]["domains"] == {"claude-import": 2}


# ── the tool ─────────────────────────────────────────────────────────────────────────


class TestExportTool:
    def _tool(self, fake_home):
        return build_export_tools(fake_home)[0]

    def test_dry_run_is_the_default_and_writes_nothing(self, fake_home, tmp_path):
        out = tmp_path / "bundles"
        res = self._tool(fake_home).invoke({"out": str(out)})
        assert "Dry run" in res
        assert not out.exists()

    def test_the_plan_names_what_would_travel(self, fake_home):
        res = self._tool(fake_home).invoke({})
        assert "demo-skill" in res  # the fixture's user skill
        assert "standup" in res  # the fixture's slash command translated as a user-facing skill
        assert "helper" in res  # the fixture's subagent
        assert "mcp.github.env.GH_TOKEN" in res  # the credential the importer must supply

    def test_apply_writes_a_bundle_with_the_token_stripped(self, fake_home, tmp_path):
        out = tmp_path / "bundles"
        res = self._tool(fake_home).invoke({"out": str(out), "apply": True})
        assert "Wrote" in res
        written = next(out.glob("*.zip"))
        assert SECRET.encode() not in written.read_bytes()
        assert manifest_of(written.read_bytes())["kind"] == "agent-snapshot"

    def test_include_memory_uses_canonical_claude_project_store(self, fake_home, tmp_path):
        out = tmp_path / "bundles"

        result = self._tool(fake_home).invoke(
            {
                "project_dir": PROJECT_DIR,
                "include_memory": True,
                "out": str(out),
                "apply": True,
            }
        )
        written = next(out.glob("*.zip"))
        with zipfile.ZipFile(written) as archive:
            memory = archive.read("knowledge/claude-import.md").decode()

        assert "The fact body." in memory
        assert "NOT publishable" in result

    def test_include_memory_accepts_relative_project_dir(self, fake_home, tmp_path, monkeypatch):
        project = tmp_path / "project"
        project.mkdir()
        slug = project_slug_candidates(str(project.resolve()))[0]
        memory = Path(fake_home["cli_root"]) / "projects" / slug / "memory"
        memory.mkdir(parents=True)
        (memory / "fact.md").write_text("---\nname: relative-fact\n---\n\nRelative memory.\n")
        out = tmp_path / "bundles"
        monkeypatch.chdir(project)

        result = self._tool(fake_home).invoke(
            {"project_dir": ".", "include_memory": True, "out": str(out), "apply": True}
        )

        written = next(out.glob("*.zip"))
        with zipfile.ZipFile(written) as archive:
            exported = archive.read("knowledge/claude-import.md").decode()
        assert "Relative memory." in exported
        assert "NOT publishable" in result

    def test_export_honors_configured_read_cap(self, fake_home, tmp_path):
        marker = secrets.token_urlsafe(24)
        skill_md = Path(fake_home["cli_root"]) / "skills" / "demo-skill" / "SKILL.md"
        skill_md.write_text("---\nname: demo-skill\ndescription: Demo.\n---\n\n" + marker + "\n")
        capped = dict(fake_home, max_read_bytes=16)
        out = tmp_path / "bundles"

        result = build_export_tools(capped)[0].invoke({"out": str(out), "apply": True})
        written = next(out.glob("*.zip"))
        with zipfile.ZipFile(written) as archive:
            decompressed = b"".join(archive.read(name) for name in archive.namelist())

        assert marker.encode() not in decompressed
        assert "max_read_bytes" in result

    def test_oversized_mcp_settings_are_reported_not_silently_omitted(self, fake_home):
        settings = Path(fake_home["cli_root"]) / "settings.json"
        settings.write_text(settings.read_text() + " " * 129)
        capped = dict(fake_home, max_read_bytes=128)

        result = build_export_tools(capped)[0].invoke({})

        assert "settings.json exceeds max_read_bytes=128" in result
        assert "MCP servers      0" in result

    def test_symlinked_user_skills_root_is_reported_not_silently_omitted(self, fake_home, tmp_path):
        skills = Path(fake_home["cli_root"]) / "skills"
        skills.rename(Path(fake_home["cli_root"]) / "skills-original")
        outside = tmp_path / "outside-skills"
        outside.mkdir()
        skills.symlink_to(outside, target_is_directory=True)

        result = self._tool(fake_home).invoke({})

        assert "user skills root" in result
        assert "outside declared root" in result

    def test_symlinked_user_commands_root_is_reported_not_silently_omitted(self, fake_home, tmp_path):
        commands = Path(fake_home["cli_root"]) / "commands"
        commands.rename(Path(fake_home["cli_root"]) / "commands-original")
        outside = tmp_path / "outside-commands"
        outside.mkdir()
        commands.symlink_to(outside, target_is_directory=True)

        result = self._tool(fake_home).invoke({})

        assert "user commands root" in result
        assert "outside declared root" in result

    def test_the_reply_tells_the_operator_to_read_the_review(self, fake_home, tmp_path):
        res = self._tool(fake_home).invoke({"out": str(tmp_path), "apply": True})
        assert "REVIEW.md" in res


@pytest.mark.parametrize("name", ["", "  "])
def test_blank_project_dir_is_user_level_only(fake_home, name):
    """Passing no project must not crash on a missing CLAUDE.md / memory dir."""
    res = build_export_tools(fake_home)[0].invoke({"project_dir": name})
    assert "Bundle plan" in res
