"""full_extract end to end, with a fake LLM in place of litellm.

Section detection (which loads a sentence-transformers model) is replaced by a
plain full-text section, so these tests stay offline and fast.
"""

import hashlib
import re

import pytest

from auto_lca.process.sog import agent_sog

USAGE = {"prompt_tokens": 100, "completion_tokens": 10, "total_tokens": 110}

CANDIDATES = [
    {"name": "Farm A", "description": "Pasture farm"},
    {"name": "Farm B", "description": "Confined farm"},
    {"name": "Literature farm", "description": "From another study"},
    {"name": "Noise", "description": "Not a system"},
]
VERDICTS = [
    {"scenario_id": 0, "category": "own_system_modelled", "grouping_dimension": "farm"},
    {"scenario_id": 1, "category": "own_system_modelled", "grouping_dimension": "farm"},
    {"scenario_id": 2, "category": "external_reference"},
    {"scenario_id": 3, "category": "other"},
]


class FakeLLM:
    """Answers each tool call the extractor makes and records the calls."""

    def __init__(self, candidates=CANDIDATES, verdicts=VERDICTS, fail_on=()):
        self.candidates = candidates
        self.verdicts = verdicts
        self.fail_on = set(fail_on)
        self.calls = []

    async def __call__(self, messages, tools, max_tokens=16000, prompt_cache_key=None, model=None):
        tool = tools[0]["function"]
        name = tool["name"]
        prompt = messages[-1]["content"]
        scope = re.search(r'SPECIFICALLY for the scenario "([^"]+)"', prompt)
        scenario = scope.group(1) if scope else None
        self.calls.append({"tool": name, "model": model, "cache_key": prompt_cache_key,
                           "scenario": scenario, "prompt": prompt})
        if name in self.fail_on or (scenario and scenario in self.fail_on):
            raise RuntimeError(f"fake failure: {name} {scenario}")
        if name == "record_scenarios":
            return {"scenarios": [dict(c) for c in self.candidates]}, USAGE
        if name == "record_scenario_verdicts":
            return {"verdicts": self.verdicts}, USAGE
        # record_extraction: one value per field, tagged with where it came from.
        args = {
            concept: {field: f"{scenario or 'study'}:{field}" for field in schema["properties"]}
            for concept, schema in tool["parameters"]["properties"].items()
        }
        return args, USAGE

    def tools_called(self, tool):
        return [c for c in self.calls if c["tool"] == tool]


@pytest.fixture(autouse=True)
def no_section_model(monkeypatch):
    monkeypatch.setattr(
        agent_sog, "build_sections_dict",
        lambda text: {"full_text": {"section_title": "full_text", "section_text": text}},
    )


@pytest.fixture
def run(make_extractor, make_pdf):
    """Run full_extract with a FakeLLM; returns (output, fake)."""

    def _run(defaults=None, fake=None, pdf_text="A dairy LCA comparing farms.", **kwargs):
        ex = make_extractor(defaults)
        fake = fake or FakeLLM()
        ex._acall = fake
        return ex.full_extract(make_pdf(pdf_text), **kwargs), fake

    return _run


def _groups(out):
    """{system name: {concept or 'Scenario': fields}}"""
    result = {}
    for group in out["results"]:
        merged = {k: v for entry in group for k, v in entry.items()}
        result[merged["Scenario"]["Scenario Name"]] = merged
    return result


def test_output_shape(run):
    out, _ = run()
    assert set(out) == {"metadata", "tokens", "results", "sections_dict", "extracted_from"}
    assert out["extracted_from"] == "full_text (agent, per-scenario concurrent chunks)"
    assert out["metadata"]["pid"] == "paper"
    assert out["metadata"]["extractor"] == "agent_sog.AgentExtractor"
    assert out["metadata"]["model"] == "mistral/mistral-small-test"
    assert out["metadata"]["scenario_model"] == "mistral/mistral-large-test"


def test_metadata_records_the_config(run):
    out, _ = run()
    meta = out["metadata"]
    assert meta["config_path"].endswith("config.json")
    with open(meta["config_path"], "rb") as f:
        assert meta["config_sha256"] == hashlib.sha256(f.read()).hexdigest()


def test_systems_are_labelled_and_noise_dropped(run):
    groups = _groups(run()[0])
    assert list(groups) == ["Farm A", "Farm B", "Literature farm"]
    cats = {n: g["Scenario"]["scenario_category"] for n, g in groups.items()}
    assert cats == {"Farm A": "own_system_modelled", "Farm B": "own_system_modelled",
                    "Literature farm": "external_reference"}


def test_modelled_systems_get_their_own_fields_plus_study_wide(run):
    groups = _groups(run()[0])
    farm_a = groups["Farm A"]
    assert farm_a["Scenario"]["fields_extracted"] is True
    assert farm_a["Scenario"]["definition_only"] is False
    assert farm_a["Inventory"] == {"Milk yield": "Farm A:Milk yield", "Feed": "Farm A:Feed"}
    assert farm_a["Goal and Scope"]["Functional Unit"] == "study:Functional Unit"
    assert groups["Farm B"]["Inventory"]["Feed"] == "Farm B:Feed"


def test_external_reference_is_a_shell_without_fields(run):
    lit = _groups(run()[0])["Literature farm"]
    assert set(lit) == {"Scenario"}
    assert lit["Scenario"]["fields_extracted"] is False
    assert lit["Scenario"]["definition_only"] is True


def test_calls_use_the_right_models_and_one_cache_key(run):
    _, fake = run()
    for tool in ("record_scenarios", "record_scenario_verdicts"):
        assert [c["model"] for c in fake.tools_called(tool)] == ["mistral/mistral-large-test"]
    extraction = fake.tools_called("record_extraction")
    # 1 study-wide call + 1 per modelled system (each concept fits one chunk).
    assert len(extraction) == 3
    assert sorted(c["scenario"] or "" for c in extraction) == ["", "Farm A", "Farm B"]
    assert {c["model"] for c in extraction} == {None}  # None -> extractor's own model
    assert {c["cache_key"] for c in extraction} == {"test:paper:full_text"}


def test_token_usage_is_split_by_model(run):
    tokens = run()[0]["tokens"]
    assert tokens["scenario_tokens"]["total_tokens"] == 2 * 110
    assert tokens["scenario_tokens"]["model"] == "mistral/mistral-large-test"
    assert tokens["field_tokens"]["total_tokens"] == 3 * 110
    assert tokens["field_tokens"]["model"] == "mistral/mistral-small-test"


def test_extract_non_primary_fills_external_systems(run):
    out, fake = run({"extract_non_primary": True})
    lit = _groups(out)["Literature farm"]
    assert lit["Scenario"]["fields_extracted"] is True
    assert lit["Inventory"]["Feed"] == "Literature farm:Feed"
    assert len(fake.tools_called("record_extraction")) == 4


def test_scenarios_only_makes_no_extraction_calls(run):
    out, fake = run(scenarios_only=True)
    assert fake.tools_called("record_extraction") == []
    assert out["extracted_from"] == "full_text (agent, scenarios only)"
    for group in _groups(out).values():
        assert set(group) == {"Scenario"}


def test_paper_level_mode_skips_system_identification(run):
    out, fake = run({"extract_production_systems": False})
    assert fake.tools_called("record_scenarios") == []
    assert fake.tools_called("record_scenario_verdicts") == []
    groups = _groups(out)
    assert list(groups) == ["Paper"]
    assert groups["Paper"]["Inventory"]["Feed"] == "Paper:Feed"
    assert out["metadata"]["extract_scenarios"] is False


def test_no_systems_found_falls_back_to_one_system(run):
    out, _ = run(fake=FakeLLM(candidates=[]))
    groups = _groups(out)
    assert list(groups) == ["Own system"]
    assert groups["Own system"]["Inventory"]["Feed"] == "Own system:Feed"


def test_validator_failure_keeps_candidates_as_unlabelled_shells(run):
    out, fake = run(fake=FakeLLM(fail_on={"record_scenario_verdicts"}))
    groups = _groups(out)
    assert list(groups) == ["Farm A", "Farm B", "Literature farm", "Noise"]
    assert {g["Scenario"]["scenario_category"] for g in groups.values()} == {"unclassified"}
    # Unclassified systems get no fields by default; only study-wide is extracted.
    assert [c["scenario"] for c in fake.tools_called("record_extraction")] == [None]


def test_failed_chunk_does_not_lose_the_paper(run):
    out, _ = run(fake=FakeLLM(fail_on={"Farm B"}))
    groups = _groups(out)
    assert groups["Farm A"]["Inventory"]["Feed"] == "Farm A:Feed"
    assert "Inventory" not in groups["Farm B"]
    assert "Goal and Scope" in groups["Farm B"]


def test_bibliography_is_stripped_before_the_llm_sees_it(run):
    body = "Life cycle assessment of dairy farms in Portugal. " * 6
    # The PDF fixture writes one line, so build a multi-line page.
    text = "\n".join([body[i:i + 90] for i in range(0, len(body), 90)] * 6)
    text += "\nReferences\nSmith, J. 2019. Milk footprints.\nDoe, A. 2020. Feed.\n"
    out, fake = run(pdf_text=text)
    assert out["metadata"]["bibliography_stripped"] is True
    assert "Smith, J." not in fake.tools_called("record_scenarios")[0]["prompt"]
    assert "Smith, J." not in out["sections_dict"]["full_text"]["section_text"]


def test_extra_text_is_appended(make_extractor, make_pdf):
    ex = make_extractor()
    fake = FakeLLM()
    ex._acall = fake
    out = ex.full_extract(make_pdf(), extra_text="Supplementary table S1: feed 12 kg")
    assert "Supplementary table S1" in fake.tools_called("record_scenarios")[0]["prompt"]
    assert out["metadata"]["included_supplementary"] is True
    assert out["extracted_from"] == "full_text+SI (agent, per-scenario concurrent chunks)"
