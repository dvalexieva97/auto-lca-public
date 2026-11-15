"""Join extract JSON → extracted_results.xlsx."""

from __future__ import annotations

import json
from pathlib import Path

from auto_lca.process.sog.results_table import (
    load_extract_rows,
    write_extract_results_xlsx,
)


def _write_extract(folder: Path, pid: str, results: list) -> None:
    (folder / f"{pid}.json").write_text(
        json.dumps(
            {
                "metadata": {"pid": pid, "pdf_path": f"{pid}.pdf"},
                "results": results,
            }
        ),
        encoding="utf-8",
    )


def test_nested_and_list_fields_flatten(tmp_path):
    _write_extract(
        tmp_path,
        "p1",
        [
            [
                {
                    "Scenario": {
                        "Scenario Name": "BASE",
                        "scenario_category": "own_system_modelled",
                    }
                },
                {
                    "Results": {
                        "Carbon Impact": {"amount": 1.43, "unit": "kg CO2e"},
                        "Impacts": [{"name": "GWP", "amount": 1.43}],
                    }
                },
            ]
        ],
    )
    rows = load_extract_rows(tmp_path)
    assert len(rows) == 1
    row = rows[0]
    assert row[("Meta", "PID")] == "p1"
    assert row[("Meta", "Scenario Name")] == "BASE"
    assert row[("Results", "Carbon Impact", "amount")] == 1.43
    assert row[("Results", "Carbon Impact", "unit")] == "kg CO2e"
    assert '"GWP"' in row[("Results", "Impacts")]


def test_write_xlsx(tmp_path):
    _write_extract(
        tmp_path,
        "p1",
        [
            [
                {"Scenario": {"Scenario Name": "A"}},
                {"Results": {"Carbon Impact": {"amount": 1.0, "unit": "kg"}}},
            ]
        ],
    )
    _write_extract(
        tmp_path,
        "p2",
        [[{"Study Design": {"Objective": "test"}}]],
    )
    path = write_extract_results_xlsx(tmp_path)
    assert path is not None
    assert path.name == "extracted_results.xlsx"
    assert path.is_file()

    from openpyxl import load_workbook

    ws = load_workbook(path).active
    headers_l0 = [ws.cell(1, c).value for c in range(1, ws.max_column + 1)]
    assert "Meta" in headers_l0
    assert "Results" in headers_l0
    # Nested leaf under Results
    assert any(
        ws.cell(ws.max_row and 3, c).value == "amount"
        for c in range(1, ws.max_column + 1)
    )
    assert ws.max_row >= 4  # 3 header levels + ≥1 data row
