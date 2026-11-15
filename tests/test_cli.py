import shutil
import sys
from pathlib import Path

import pytest

from auto_lca.process.preprocess.config_parser import ensure_json_config
from auto_lca.process.sog.concept import build_concepts_from_config
from auto_lca.scripts import agent_sog_runner

SAMPLE_FILES = [
    f"sample_config_{domain}.{ext}"
    for domain in ("lca", "generic")
    for ext in ("json", "xlsx")
]


def _main(monkeypatch, *argv):
    monkeypatch.setattr(sys, "argv", ["auto-lca", *argv])
    agent_sog_runner.main()


@pytest.fixture
def in_tmp(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    return tmp_path


def test_version(monkeypatch, capsys):
    with pytest.raises(SystemExit) as exit_info:
        _main(monkeypatch, "--version")
    assert exit_info.value.code == 0
    assert capsys.readouterr().out.startswith("auto-lca ")


def test_help_lists_every_command(monkeypatch, capsys):
    with pytest.raises(SystemExit):
        _main(monkeypatch, "--help")
    out = capsys.readouterr().out
    for cmd in (
        "sample-config",
        "extract",
        "reclassify-scenarios",
        "extract-sections",
        "compare",
        "enrich-excel",
        "regenerate-stats",
    ):
        assert cmd in out


def test_command_is_required(monkeypatch):
    with pytest.raises(SystemExit) as exit_info:
        _main(monkeypatch)
    assert exit_info.value.code == 2


def test_extract_requires_its_options(monkeypatch, capsys):
    with pytest.raises(SystemExit):
        _main(monkeypatch, "extract", "--pdfs", "a.pdf")
    assert "--config" in capsys.readouterr().err


@pytest.mark.parametrize("fmt", ["json", "xlsx"])
@pytest.mark.parametrize("domain", ["lca", "generic"])
def test_sample_config_copies_bundled_file(
    monkeypatch, in_tmp, sample_config_dir, fmt, domain
):
    _main(monkeypatch, "sample-config", "--format", fmt, "--domain", domain)
    name = f"sample_config_{domain}.{fmt}"
    assert (in_tmp / name).read_bytes() == (sample_config_dir / name).read_bytes()


def test_sample_config_custom_output(monkeypatch, in_tmp):
    _main(monkeypatch, "sample-config", "--domain", "lca", "--output", "mine.json")
    assert (in_tmp / "mine.json").is_file()


def test_sample_config_refuses_to_overwrite(monkeypatch, in_tmp):
    (in_tmp / "sample_config_lca.json").write_text("my edits")
    with pytest.raises(SystemExit, match="already exists"):
        _main(monkeypatch, "sample-config", "--domain", "lca")
    assert (in_tmp / "sample_config_lca.json").read_text() == "my edits"
    _main(monkeypatch, "sample-config", "--domain", "lca", "--overwrite")
    assert (in_tmp / "sample_config_lca.json").read_text() != "my edits"


def test_sample_config_csv_removed(monkeypatch, in_tmp, capsys):
    with pytest.raises(SystemExit) as exc:
        _main(monkeypatch, "sample-config", "--format", "csv", "--domain", "lca")
    assert exc.value.code == 2
    err = capsys.readouterr().err
    assert "invalid choice: 'csv'" in err
    assert "json" in err and "xlsx" in err


def test_extract_command_dispatches_to_runner(monkeypatch, in_tmp):
    seen = {}
    monkeypatch.setattr(
        agent_sog_runner, "run_extract", lambda args: seen.update(vars(args))
    )
    (in_tmp / "c.csv").write_text("my config")
    _main(
        monkeypatch,
        "extract",
        "--pdfs",
        "a.pdf",
        "b.pdf",
        "--config",
        "c.csv",
        "--output",
        "out",
        "--scenarios-only",
    )
    assert seen["pdfs"] == ["a.pdf", "b.pdf"]
    assert (seen["config"], seen["output"], seen["scenarios_only"]) == (
        "c.csv",
        "out",
        True,
    )


@pytest.mark.parametrize("name", SAMPLE_FILES)
def test_extract_copies_missing_sample_config(
    monkeypatch, in_tmp, sample_config_dir, name
):
    monkeypatch.setattr(agent_sog_runner, "run_extract", lambda args: None)
    _main(monkeypatch, "extract", "--pdfs", "a.pdf", "--config", name, "--output", "out")
    assert (in_tmp / name).read_bytes() == (sample_config_dir / name).read_bytes()


def test_extract_keeps_existing_sample_config(monkeypatch, in_tmp):
    monkeypatch.setattr(agent_sog_runner, "run_extract", lambda args: None)
    (in_tmp / "sample_config_lca.json").write_text("my edits")
    _main(
        monkeypatch,
        "extract",
        "--pdfs",
        "a.pdf",
        "--config",
        "sample_config_lca.json",
        "--output",
        "out",
    )
    assert (in_tmp / "sample_config_lca.json").read_text() == "my edits"


def test_extract_missing_config_points_to_sample_config(monkeypatch, in_tmp):
    monkeypatch.setattr(agent_sog_runner, "run_extract", lambda args: None)
    with pytest.raises(SystemExit, match="auto-lca sample-config"):
        _main(
            monkeypatch,
            "extract",
            "--pdfs",
            "a.pdf",
            "--config",
            "mine.csv",
            "--output",
            "out",
        )
    assert not (in_tmp / "mine.csv").exists()


@pytest.mark.parametrize("name", SAMPLE_FILES)
def test_every_bundled_sample_config_loads(tmp_path, sample_config_dir, name):
    path = tmp_path / name
    shutil.copy(sample_config_dir / name, path)
    study_wide, by_system = build_concepts_from_config(ensure_json_config(str(path)))
    assert study_wide or by_system
    assert all(c.structure for c in study_wide + by_system)


def test_lca_xlsx_results_use_string_fields_matching_json_names(sample_config_dir):
    """Excel-first LCA: impact fields are str | None, same names as JSON."""
    import json
    import tempfile

    from openpyxl import load_workbook

    from auto_lca.process.preprocess.config_parser import parse_xlsx_config

    doc = json.loads((sample_config_dir / "sample_config_lca.json").read_text())
    json_results = doc["field_sets"][1]["fields"]
    assert set(json_results) == {
        "Carbon footprint",
        "Acidification Impact",
        "Eutrophication Impact",
    }

    xlsx = sample_config_dir / "sample_config_lca.xlsx"
    wb = load_workbook(xlsx)
    rows = [
        r
        for r in wb["Field sets"].iter_rows(min_row=2, values_only=True)
        if r[0] == "Results"
    ]
    assert {r[3] for r in rows} == set(json_results)
    assert all(r[4] == "str | None" for r in rows)

    converted = parse_xlsx_config(
        str(xlsx), tempfile.mktemp(suffix=".json")
    )
    cfg = json.loads(Path(converted).read_text())
    x_fields = cfg["field_sets"][1]["fields"]
    assert x_fields == {
        name: "str | None" for name in json_results
    }


def test_generic_json_and_xlsx_samples_diverge(sample_config_dir):
    """JSON: Models[] + scenarios off. XLSX: flat fields + scenarios on."""
    import json

    from openpyxl import load_workbook

    from auto_lca.process.sog.pipeline_defaults import pipeline_rows_for_domain

    doc = json.loads((sample_config_dir / "sample_config_generic.json").read_text())
    assert doc["defaults"]["extract_production_systems"] is False
    assert doc["defaults"]["by_production_system"] is False
    models = doc["field_sets"][1]["fields"]["Models"]
    assert isinstance(models, list) and isinstance(models[0], dict)

    rows = {
        r["key"]: r["value"]
        for r in pipeline_rows_for_domain("generic", sample_format="xlsx")
    }
    assert rows["extract_production_systems"] is True
    assert rows["by_production_system"] is True

    wb = load_workbook(sample_config_dir / "sample_config_generic.xlsx")
    pipe = {
        r[0]: r[1]
        for r in wb["Pipeline"].iter_rows(min_row=2, values_only=True)
    }
    assert pipe["extract_production_systems"] is True
    assert pipe["by_production_system"] is True
    model_rows = [
        r
        for r in wb["Field sets"].iter_rows(min_row=2, values_only=True)
        if r[0] == "Model or Experiment"
    ]
    assert model_rows
    assert all(r[2] is True for r in model_rows)  # by_production_system
    assert {r[3] for r in model_rows} == {
        "Name",
        "Key metric",
        "Score",
        "Description",
    }


@pytest.mark.parametrize("name", SAMPLE_FILES)
def test_sample_configs_are_packaged(sample_config_dir, name):
    assert (agent_sog_runner._SAMPLE_CONFIG_DIR / name).is_file()
