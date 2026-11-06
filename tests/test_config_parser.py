import json

import openpyxl
import pytest

from auto_lca.process.preprocess.config_parser import (
    _enrich_config,
    ensure_json_config,
    parse_csv_config,
    parse_xlsx_config,
)

CSV_HEADER = "field_set,description,by_production_system,field,data_type\n"


def _write(tmp_path, name, text):
    path = tmp_path / name
    path.write_text(text)
    return str(path)


def _load(path):
    with open(path) as f:
        return json.load(f)


# --- CSV -------------------------------------------------------------------


def test_csv_rows_are_grouped_into_field_sets(tmp_path):
    csv = _write(tmp_path, "c.csv", CSV_HEADER
                 + "Goal,Setup,FALSE,Functional Unit,str\n"
                 + "Goal,Setup,FALSE,System Boundary,str | None\n"
                 + "Results,Impacts,TRUE,GWP,float | None\n")
    config = _load(parse_csv_config(csv))
    field_sets = {c["name"]: c for c in config["field_sets"]}
    assert list(field_sets) == ["Goal", "Results"]
    assert field_sets["Goal"]["fields"] == {
        "Functional Unit": "str", "System Boundary": "str | None"}
    assert field_sets["Goal"]["by_production_system"] is False
    assert field_sets["Results"]["by_production_system"] is True


def test_csv_writes_json_next_to_csv_by_default(tmp_path):
    csv = _write(tmp_path, "c.csv", CSV_HEADER + "Goal,Setup,FALSE,FU,str\n")
    assert parse_csv_config(csv) == str(tmp_path / "c.json")
    assert (tmp_path / "c.json").is_file()


def test_csv_gets_defaults_from_sample_config(tmp_path):
    csv = _write(tmp_path, "c.csv", CSV_HEADER + "Goal,Setup,FALSE,FU,str\n")
    defaults = _load(parse_csv_config(csv))["defaults"]
    assert defaults["model"]
    assert defaults["production_system_model"]


@pytest.mark.parametrize("raw,expected", [
    ("TRUE", True), ("true", True), ("1", True), ("yes", True),
    ("FALSE", False), ("0", False), ("no", False),
])
def test_csv_boolean_spellings(tmp_path, raw, expected):
    csv = _write(tmp_path, "c.csv", CSV_HEADER + f"Goal,Setup,{raw},FU,str\n")
    (field_set,) = _load(parse_csv_config(csv))["field_sets"]
    assert field_set["by_production_system"] is expected


def test_csv_legacy_column_names_and_aspect_column_are_accepted(tmp_path):
    csv = _write(tmp_path, "c.csv",
                 "name,description,aspect,by_scenario,structure_title,structure_data_type\n"
                 "Goal,Setup,Methods,FALSE,FU,str\n")
    (field_set,) = _load(parse_csv_config(csv))["field_sets"]
    assert field_set["by_production_system"] is False
    assert field_set["fields"] == {"FU": "str"}
    assert "aspect" not in field_set


def test_csv_missing_required_column(tmp_path):
    csv = _write(tmp_path, "c.csv", "field_set,description,field\nGoal,Setup,FU\n")
    with pytest.raises(ValueError, match="Missing required columns"):
        parse_csv_config(csv)


def test_csv_inconsistent_description(tmp_path):
    csv = _write(tmp_path, "c.csv", CSV_HEADER
                 + "Goal,Setup,FALSE,FU,str\n"
                 + "Goal,Different,FALSE,SB,str\n")
    with pytest.raises(ValueError, match="Inconsistent description"):
        parse_csv_config(csv)


def test_csv_field_set_without_fields(tmp_path):
    csv = _write(tmp_path, "c.csv", CSV_HEADER + "Goal,Setup,FALSE,,\n")
    with pytest.raises(ValueError, match="has no fields"):
        parse_csv_config(csv)


# --- XLSX ------------------------------------------------------------------


def _xlsx(tmp_path, sheets: dict[str, list[list]]):
    wb = openpyxl.Workbook()
    wb.remove(wb.active)
    for title, rows in sheets.items():
        ws = wb.create_sheet(title)
        for row in rows:
            ws.append(row)
    path = tmp_path / "c.xlsx"
    wb.save(path)
    return str(path)


FIELD_ROWS = [
    ["field_set", "description", "by_production_system", "field", "data_type"],
    ["Goal", "Setup", False, "FU", "str"],
    ["Results", "Impacts", True, "GWP", "float | None"],
]


def test_xlsx_field_sets_and_settings(tmp_path):
    path = _xlsx(tmp_path, {
        "Field sets": FIELD_ROWS,
        "Settings": [["section", "key", "value"],
                     ["defaults", "model", "mistral-small-custom"],
                     ["defaults", "strip_bibliography", "false"]],
    })
    config = _load(parse_xlsx_config(path))
    assert [c["name"] for c in config["field_sets"]] == ["Goal", "Results"]
    assert config["defaults"]["model"] == "mistral-small-custom"
    assert config["defaults"]["strip_bibliography"] is False
    # Keys not in Settings come from the sample config.
    assert config["defaults"]["production_system_model"]


def test_xlsx_legacy_concepts_sheet_and_columns(tmp_path):
    legacy_rows = [["concept_name", "description", "by_production_system",
                    "structure_field", "structure_type"]] + FIELD_ROWS[1:]
    config = _load(parse_xlsx_config(_xlsx(tmp_path, {"Concepts": legacy_rows})))
    assert [c["name"] for c in config["field_sets"]] == ["Goal", "Results"]
    assert config["field_sets"][1]["fields"] == {"GWP": "float | None"}


def test_xlsx_legacy_aspects_sheet_is_ignored(tmp_path):
    path = _xlsx(tmp_path, {
        "Field sets": FIELD_ROWS,
        "Aspects": [["name", "description"], ["Methods", "Methods section"]],
    })
    assert "aspects" not in _load(parse_xlsx_config(path))


def test_xlsx_without_field_sets_sheet(tmp_path):
    path = _xlsx(tmp_path, {"Sheet1": FIELD_ROWS})
    with pytest.raises(ValueError, match="missing required sheet 'Field sets'"):
        parse_xlsx_config(path)


# --- ensure_json_config / enrichment ------------------------------------------


def test_json_is_passed_through(tmp_path):
    path = _write(tmp_path, "c.json", "{}")
    assert ensure_json_config(path) == path


def test_csv_and_xlsx_are_converted(tmp_path):
    csv = _write(tmp_path, "a.csv", CSV_HEADER + "Goal,Setup,FALSE,FU,str\n")
    assert ensure_json_config(csv).endswith("a.json")
    xlsx = _xlsx(tmp_path, {"Field sets": FIELD_ROWS})
    assert ensure_json_config(xlsx).endswith("c.json")


def test_missing_config_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        ensure_json_config(str(tmp_path / "nope.json"))


def test_unsupported_config_format(tmp_path):
    path = _write(tmp_path, "c.txt", "")
    with pytest.raises(ValueError, match="Unsupported config format"):
        ensure_json_config(path)


def test_enrich_merges_defaults_and_drops_aspects():
    base = {"defaults": {"model": "base", "strip_bibliography": True},
            "field_sets": [{"name": "Base"}]}
    config = {"defaults": {"model": "mine"}, "aspects": [{"name": "Methods"}]}
    result = _enrich_config(config, base)
    assert result["defaults"] == {"model": "mine", "strip_bibliography": True}
    assert result["field_sets"] == [{"name": "Base"}]
    assert "aspects" not in result


def test_enrich_keeps_provided_field_sets():
    result = _enrich_config({"field_sets": [{"name": "Mine"}]},
                            {"field_sets": [{"name": "Base"}]})
    assert result["field_sets"] == [{"name": "Mine"}]
