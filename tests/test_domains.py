"""Domain resolution and prompt packs (lca | generic)."""

from __future__ import annotations

import json

import pytest

from auto_lca.process.sog.domains import (
    extract_behavior_defaults,
    prompts_for_domain,
    resolve_domain,
)
from auto_lca.scripts import agent_sog_runner


def test_resolve_prefers_cli_over_config_and_entry():
    assert (
        resolve_domain(
            cli_domain="generic",
            config_defaults={"domain": "lca"},
            entry_default="lca",
        )
        == "generic"
    )


def test_resolve_uses_config_when_cli_absent():
    assert (
        resolve_domain(
            cli_domain=None,
            config_defaults={"domain": "generic"},
            entry_default="lca",
        )
        == "generic"
    )


def test_resolve_falls_back_to_entry_default():
    assert resolve_domain(entry_default="generic") == "generic"
    assert resolve_domain(entry_default="lca") == "lca"


def test_extract_behavior_defaults():
    lca = extract_behavior_defaults("lca")
    generic = extract_behavior_defaults("generic")
    assert lca["extract_sections"] is False
    assert lca["include_pdf_tables"] is False
    assert lca["allow_additional_fields"] is False
    assert generic["extract_sections"] is False
    assert generic["include_pdf_tables"] is True
    assert generic["allow_additional_fields"] is False
    assert resolve_domain() == "lca"


def test_unknown_domain_raises():
    with pytest.raises(ValueError, match="Unknown domain"):
        resolve_domain(cli_domain="fashion")


def test_prompt_packs_differ():
    lca = prompts_for_domain("lca")
    gen = prompts_for_domain("generic")
    assert "LCA" in lca["system_extract_concept"] or "lifecycle" in lca[
        "system_extract_concept"
    ].lower()
    assert "LCA" not in gen["system_extract_concept"]
    assert "LCA study" in lca["prompt_extract_scenarios"]
    assert "LCA study" not in gen["prompt_extract_scenarios"]
    assert lca["system_extract_scenarios"] != gen["system_extract_scenarios"]


def test_auto_extract_entry_defaults_domain_generic(monkeypatch):
    seen = {}

    def fake_run(args):
        seen["domain"] = getattr(args, "domain", None)
        seen["entry"] = args.entry_default_domain
        seen["config"] = args.config

    monkeypatch.setattr(agent_sog_runner, "run_extract", fake_run)
    monkeypatch.setattr(
        agent_sog_runner, "_resolve_config", lambda c: c
    )
    agent_sog_runner.main_generic(
        [
            "extract",
            "--pdfs",
            "a.pdf",
            "--config",
            "c.json",
            "--output",
            "out",
        ]
    )
    assert seen["entry"] == "generic"
    assert seen["domain"] is None  # resolved later inside AgentExtractor


def test_sample_config_json_writes_domain(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    agent_sog_runner.main(
        ["sample-config", "--format", "json", "--domain", "generic"]
    )
    doc = json.loads((tmp_path / "sample_config_generic.json").read_text())
    assert doc["defaults"]["domain"] == "generic"
