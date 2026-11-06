import shutil
import sys

import pytest

from auto_lca.process.preprocess.config_parser import ensure_json_config
from auto_lca.process.sog.concept import build_concepts_from_config
from auto_lca.scripts import agent_sog_runner

SAMPLE_FILES = [f"{name}.{ext}" for name in ("sample_config", "sample_config_mini")
                for ext in ("json", "csv", "xlsx")]


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
    for cmd in ("sample-config", "extract", "reclassify-scenarios", "extract-sections",
                "compare", "enrich-excel", "regenerate-stats"):
        assert cmd in out


def test_command_is_required(monkeypatch):
    with pytest.raises(SystemExit) as exit_info:
        _main(monkeypatch)
    assert exit_info.value.code == 2


def test_extract_requires_its_options(monkeypatch, capsys):
    with pytest.raises(SystemExit):
        _main(monkeypatch, "extract", "--pdfs", "a.pdf")
    assert "--config" in capsys.readouterr().err


@pytest.mark.parametrize("fmt", ["json", "csv", "xlsx"])
@pytest.mark.parametrize("mini", [False, True])
def test_sample_config_copies_bundled_file(monkeypatch, in_tmp, sample_config_dir, fmt, mini):
    _main(monkeypatch, "sample-config", "--format", fmt, *(["--mini"] if mini else []))
    name = f"{'sample_config_mini' if mini else 'sample_config'}.{fmt}"
    assert (in_tmp / name).read_bytes() == (sample_config_dir / name).read_bytes()


def test_sample_config_custom_output(monkeypatch, in_tmp):
    _main(monkeypatch, "sample-config", "--output", "mine.json")
    assert (in_tmp / "mine.json").is_file()


def test_sample_config_refuses_to_overwrite(monkeypatch, in_tmp):
    (in_tmp / "sample_config.json").write_text("my edits")
    with pytest.raises(SystemExit, match="already exists"):
        _main(monkeypatch, "sample-config")
    assert (in_tmp / "sample_config.json").read_text() == "my edits"
    _main(monkeypatch, "sample-config", "--overwrite")
    assert (in_tmp / "sample_config.json").read_text() != "my edits"


def test_extract_command_dispatches_to_runner(monkeypatch, in_tmp):
    seen = {}
    monkeypatch.setattr(agent_sog_runner, "run_extract", lambda args: seen.update(vars(args)))
    (in_tmp / "c.csv").write_text("my config")
    _main(monkeypatch, "extract", "--pdfs", "a.pdf", "b.pdf", "--config", "c.csv",
          "--output", "out", "--scenarios-only")
    assert seen["pdfs"] == ["a.pdf", "b.pdf"]
    assert (seen["config"], seen["output"], seen["scenarios_only"]) == ("c.csv", "out", True)


@pytest.mark.parametrize("name", SAMPLE_FILES)
def test_extract_copies_missing_sample_config(monkeypatch, in_tmp, sample_config_dir, name):
    monkeypatch.setattr(agent_sog_runner, "run_extract", lambda args: None)
    _main(monkeypatch, "extract", "--pdfs", "a.pdf", "--config", name, "--output", "out")
    assert (in_tmp / name).read_bytes() == (sample_config_dir / name).read_bytes()


def test_extract_keeps_existing_sample_config(monkeypatch, in_tmp):
    monkeypatch.setattr(agent_sog_runner, "run_extract", lambda args: None)
    (in_tmp / "sample_config.json").write_text("my edits")
    _main(monkeypatch, "extract", "--pdfs", "a.pdf", "--config", "sample_config.json",
          "--output", "out")
    assert (in_tmp / "sample_config.json").read_text() == "my edits"


def test_extract_missing_config_points_to_sample_config(monkeypatch, in_tmp):
    monkeypatch.setattr(agent_sog_runner, "run_extract", lambda args: None)
    with pytest.raises(SystemExit, match="auto-lca sample-config"):
        _main(monkeypatch, "extract", "--pdfs", "a.pdf", "--config", "mine.csv",
              "--output", "out")
    assert not (in_tmp / "mine.csv").exists()


@pytest.mark.parametrize("name", SAMPLE_FILES)
def test_every_bundled_sample_config_loads(tmp_path, sample_config_dir, name):
    # Regression: sample_config_mini.xlsx once lacked the required field-sets sheet.
    path = tmp_path / name
    shutil.copy(sample_config_dir / name, path)
    study_wide, by_system = build_concepts_from_config(ensure_json_config(str(path)))
    assert study_wide and by_system
    assert all(c.structure for c in study_wide + by_system)


@pytest.mark.parametrize("name", SAMPLE_FILES)
def test_sample_configs_are_packaged(sample_config_dir, name):
    # pyproject.toml ships these via package-data; the CLI copies them.
    assert (agent_sog_runner._SAMPLE_CONFIG_DIR / name).is_file()
