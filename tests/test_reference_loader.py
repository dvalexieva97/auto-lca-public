"""Reference (ground-truth) workbook loading: formulas and Excel error values."""

import openpyxl
import pytest

import pandas as pd

from auto_lca.statistics.compare import _drop_formula_cells, load_manual_data

HEADER = [["METADATA", None, "STUDY DETAILS"], [None, None, None],
          ["PID", "DOI", "Scenario Name"]]


def _workbook(tmp_path, rows):
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = "GT"
    for row in HEADER + rows:
        ws.append(row)
    for row in ws.iter_rows():
        for cell in row:
            if isinstance(cell.value, str) and cell.value.startswith("#"):
                cell.data_type = "s"  # stored as text, as Google Sheets exports it
    path = tmp_path / "gt.xlsx"
    wb.save(path)
    return str(path)


def test_formula_without_stored_result_stops_loading(tmp_path):
    # openpyxl saves formulas without results, like the export that hid
    # Sharma's scenario names.
    path = _workbook(tmp_path, [["p1", "d", '=CONCAT("Farm", " A")']])
    with pytest.raises(ValueError, match="C4.*no stored result|no stored result.*C4"):
        load_manual_data(path, "GT")


def test_excel_error_values_read_as_empty(tmp_path, capsys):
    # e.g. =AVERAGE(#REF!), whose stored result is the text "#REF!".
    path = _workbook(tmp_path, [["p1", "#REF!", "Farm A"], ["p2", "#DIV/0!", "Farm B"]])
    df = load_manual_data(path, "GT")
    assert df["SCENARIO NAME"].tolist() == ["Farm A", "Farm B"]
    assert df["DOI"].isna().all()
    assert "2 Excel error value(s)" in capsys.readouterr().out


def test_plain_values_load_unchanged(tmp_path):
    path = _workbook(tmp_path, [["p1", "10.1/x", "Farm A"]])
    df = load_manual_data(path, "GT")
    assert df.loc[0, "PID"] == "p1"
    assert df.loc[0, "DOI"] == "10.1/x"



TWO_ROW_HEADER = [[None, None, None], ["METADATA", None, "IMPACTS"], ["PID", "DOI", "GWP"]]


def _two_row_workbook(tmp_path, rows):
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = "GT"
    for row in TWO_ROW_HEADER + rows:
        ws.append(row)
    path = tmp_path / "gt2.xlsx"
    wb.save(path)
    return str(path)


def test_typed_only_rule_drops_formula_cells(tmp_path):
    path = _two_row_workbook(tmp_path, [["p1", "d1", "1.2 kg CO2e"], ["p2", "d2", "=C4"]])
    buf = _drop_formula_cells(path, "GT", [1, 2], ["IMPACTS_GWP"])
    df = pd.read_excel(buf, "GT", header=[1, 2])
    assert df.iloc[:, 2].tolist()[0] == "1.2 kg CO2e"
    assert pd.isna(df.iloc[1, 2])


def test_typed_only_rule_rejects_unknown_columns(tmp_path):
    path = _two_row_workbook(tmp_path, [["p1", "d1", "1.2 kg CO2e"]])
    with pytest.raises(ValueError, match="not found"):
        load_manual_data(path, "GT", [1, 2], typed_only_columns=["IMPACTS_NOPE"])
