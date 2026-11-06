import json
from argparse import Namespace

import pytest

from auto_lca.process.sog import extract_runner


class FakeExtractor:
    """Stands in for AgentExtractor; records which PDFs it was asked to extract."""

    instances = []
    tokens_key = "field_tokens"

    def __init__(self, config_path, model=None, scenario_model=None):
        self.config_path = config_path
        self.model = model or "mistral/fake-small"
        self.scenario_model = scenario_model or "mistral/fake-large"
        self.cache_namespace = "test"
        self.extracted = []
        FakeExtractor.instances.append(self)

    def full_extract(self, pdf_path, scenarios_only=False):
        self.extracted.append((pdf_path, scenarios_only))
        if any(name in pdf_path for name in FAIL_FOR):
            raise RuntimeError("rate limited")
        return {
            "metadata": {"pid": pdf_path},
            "tokens": {self.tokens_key: {"prompt_tokens": 100, "cached_tokens": 40}},
            "results": [[{"Scenario": {"Scenario Name": "Farm A"}}]],
            "sections_dict": {"Methods": {"section_title": "Methods", "section_text": "m"}},
            "extracted_from": "fake",
        }


FAIL_FOR: set[str] = set()


@pytest.fixture(autouse=True)
def fake_extractor(monkeypatch):
    FakeExtractor.instances.clear()
    FAIL_FOR.clear()
    monkeypatch.setattr(extract_runner, "AgentExtractor", FakeExtractor)


def _args(tmp_path, pdfs, **kw):
    defaults = dict(config="cfg.json", model=None, scenario_model=None,
                    output=str(tmp_path / "out"), pdfs=pdfs, overwrite=False,
                    scenarios_only=False)
    return Namespace(**{**defaults, **kw})


def _pdfs(tmp_path, *names):
    paths = []
    for name in names:
        path = tmp_path / f"{name}.pdf"
        path.write_bytes(b"%PDF-1.4")
        paths.append(str(path))
    return paths


def test_writes_one_json_and_sections_per_paper(tmp_path):
    run_args = _args(tmp_path, _pdfs(tmp_path, "p1", "p2"))
    extract_runner.run_extract(run_args)
    out = tmp_path / "out"
    for pid in ("p1", "p2"):
        data = json.loads((out / f"{pid}.json").read_text())
        assert data["results"][0][0]["Scenario"]["Scenario Name"] == "Farm A"
        assert data["metadata"]["exec_time"] >= 0
        sections = json.loads((out / "extracted_sections" / f"{pid}.json").read_text())
        assert sections["Methods"]["section_text"] == "m"


def test_existing_results_are_skipped(tmp_path):
    pdfs = _pdfs(tmp_path, "p1", "p2")
    (tmp_path / "out").mkdir()
    (tmp_path / "out" / "p1.json").write_text("{}")
    extract_runner.run_extract(_args(tmp_path, pdfs))
    (ex,) = FakeExtractor.instances
    assert [p for p, _ in ex.extracted] == [pdfs[1]]
    assert (tmp_path / "out" / "p1.json").read_text() == "{}"


def test_overwrite_reextracts(tmp_path):
    pdfs = _pdfs(tmp_path, "p1")
    (tmp_path / "out").mkdir()
    (tmp_path / "out" / "p1.json").write_text("{}")
    extract_runner.run_extract(_args(tmp_path, pdfs, overwrite=True))
    assert json.loads((tmp_path / "out" / "p1.json").read_text())["results"]


def test_failed_paper_is_skipped_and_the_rest_continue(tmp_path, capsys):
    FAIL_FOR.add("bad")
    extract_runner.run_extract(_args(tmp_path, _pdfs(tmp_path, "bad", "good")))
    assert not (tmp_path / "out" / "bad.json").exists()
    assert (tmp_path / "out" / "good.json").exists()
    assert "skipping this paper" in capsys.readouterr().out


def test_cli_options_reach_the_extractor(tmp_path):
    pdfs = _pdfs(tmp_path, "p1")
    extract_runner.run_extract(_args(tmp_path, pdfs, model="openai/a",
                                     scenario_model="openai/b", scenarios_only=True))
    (ex,) = FakeExtractor.instances
    assert (ex.config_path, ex.model, ex.scenario_model) == ("cfg.json", "openai/a", "openai/b")
    assert ex.extracted == [(pdfs[0], True)]


@pytest.mark.parametrize("key", ["field_tokens", "aspect_tokens"])  # aspect_tokens: pre-rename
def test_cache_summary_is_printed(tmp_path, capsys, monkeypatch, key):
    monkeypatch.setattr(FakeExtractor, "tokens_key", key)
    extract_runner.run_extract(_args(tmp_path, _pdfs(tmp_path, "p1")))
    assert "40/100 prompt tokens were cache hits (40.0%)" in capsys.readouterr().out
