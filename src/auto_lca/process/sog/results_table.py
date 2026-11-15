"""Join per-paper extract JSON files into one Excel table (one row per scenario)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd


def _json_result_paths(output_folder: str | Path) -> list[Path]:
    root = Path(output_folder)
    if not root.is_dir():
        return []
    skip_dirs = {"extracted_sections"}
    paths = []
    for path in sorted(root.glob("*.json")):
        if path.parent.name in skip_dirs:
            continue
        if path.name.startswith("extracted_results"):
            continue
        paths.append(path)
    return paths


def _flatten_fields(
    prefix: tuple[str, ...], value: Any, out: dict[tuple[str, ...], Any]
) -> None:
    """Nested dicts → nested header keys; lists → one JSON cell."""
    if isinstance(value, dict):
        if not value:
            out[prefix] = None
            return
        for key, child in value.items():
            _flatten_fields(prefix + (str(key),), child, out)
        return
    if isinstance(value, list):
        out[prefix] = json.dumps(value, ensure_ascii=False)
        return
    out[prefix] = value


def _row_from_result_group(
    group: list[dict], *, pid: str, pdf_path: str | None
) -> dict[tuple[str, ...], Any]:
    row: dict[tuple[str, ...], Any] = {
        ("Meta", "PID"): pid,
        ("Meta", "PDF path"): pdf_path or "",
        ("Meta", "Scenario Name"): "",
    }
    for entry in group or []:
        if not isinstance(entry, dict):
            continue
        for concept_name, payload in entry.items():
            if concept_name == "Scenario" and isinstance(payload, dict):
                row[("Meta", "Scenario Name")] = payload.get("Scenario Name") or ""
                for sk, sv in payload.items():
                    if sk == "Scenario Name":
                        continue
                    row[("Scenario", sk)] = sv
                continue
            if isinstance(payload, dict):
                _flatten_fields((concept_name,), payload, row)
            else:
                row[(concept_name,)] = payload
    return row


def load_extract_rows(output_folder: str | Path) -> list[dict[tuple[str, ...], Any]]:
    """One flattened row per result group across all ``{pid}.json`` extracts."""
    rows: list[dict[tuple[str, ...], Any]] = []
    for path in _json_result_paths(output_folder):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if not isinstance(data, dict) or "results" not in data:
            continue
        meta = data.get("metadata") or {}
        stem = path.stem
        pid = meta.get("pid") or stem
        pdf_path = meta.get("pdf_path")
        results = data.get("results") or []
        if not results:
            rows.append(
                {
                    ("Meta", "PID"): pid,
                    ("Meta", "PDF path"): pdf_path or "",
                    ("Meta", "Scenario Name"): "",
                }
            )
            continue
        for group in results:
            if isinstance(group, list):
                rows.append(
                    _row_from_result_group(group, pid=pid, pdf_path=pdf_path)
                )
            elif isinstance(group, dict):
                rows.append(
                    _row_from_result_group([group], pid=pid, pdf_path=pdf_path)
                )
    return rows


def _rows_to_dataframe(rows: list[dict[tuple[str, ...], Any]]) -> pd.DataFrame:
    if not rows:
        return pd.DataFrame()
    # Stable column order: Meta first, then Scenario, then others alphabetically
    # by tuple, preserving longest-prefix nesting.
    all_keys: set[tuple[str, ...]] = set()
    for row in rows:
        all_keys.update(row)
    meta_order = {
        ("Meta", "PID"): 0,
        ("Meta", "PDF path"): 1,
        ("Meta", "Scenario Name"): 2,
    }
    meta = sorted(
        (k for k in all_keys if k and k[0] == "Meta"),
        key=lambda k: (meta_order.get(k, 99), k),
    )
    scenario = sorted(k for k in all_keys if k and k[0] == "Scenario")
    rest = sorted(k for k in all_keys if k and k[0] not in ("Meta", "Scenario"))
    columns = meta + scenario + rest
    max_depth = max((len(c) for c in columns), default=1)
    # Pad tuples so MultiIndex is rectangular.
    padded = [c + ("",) * (max_depth - len(c)) for c in columns]
    data = []
    for row in rows:
        data.append([row.get(c) for c in columns])
    df = pd.DataFrame(data, columns=pd.MultiIndex.from_tuples(padded))
    # Drop all-empty trailing levels from the MultiIndex names display
    while df.columns.nlevels > 1 and all(
        str(c[-1]).strip() == "" for c in df.columns
    ):
        df.columns = df.columns.droplevel(-1)
    return df


def _excel_cell_value(value: Any) -> Any:
    """Values openpyxl can store; leave scalars, stringify oddballs."""
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    return json.dumps(value, ensure_ascii=False)


def _write_dataframe_xlsx(df: pd.DataFrame, out: Path) -> None:
    """Write ``df`` with MultiIndex (or flat) column headers and no row index."""
    from openpyxl import Workbook
    from openpyxl.styles import Alignment, Font
    from openpyxl.utils import get_column_letter

    wb = Workbook()
    ws = wb.active
    ws.title = "Extracted"

    if isinstance(df.columns, pd.MultiIndex):
        columns = [tuple("" if pd.isna(x) else str(x) for x in col) for col in df.columns]
        nlevels = df.columns.nlevels
    else:
        columns = [(str(c),) for c in df.columns]
        nlevels = 1

    header_font = Font(bold=True)
    center = Alignment(horizontal="center", vertical="center", wrap_text=True)

    for level in range(nlevels):
        for col_idx, col in enumerate(columns, start=1):
            label = col[level] if level < len(col) else ""
            cell = ws.cell(
                row=level + 1,
                column=col_idx,
                value=label or None,
            )
            cell.font = header_font
            cell.alignment = center

    # Merge consecutive equal labels on each header level (nested groups).
    for level in range(nlevels):
        start = 1
        prev = columns[0][level] if columns else ""
        for col_idx in range(2, len(columns) + 2):
            cur = columns[col_idx - 1][level] if col_idx <= len(columns) else None
            if cur == prev:
                continue
            if prev and col_idx - 1 > start:
                ws.merge_cells(
                    start_row=level + 1,
                    start_column=start,
                    end_row=level + 1,
                    end_column=col_idx - 1,
                )
            start = col_idx
            prev = cur

    # Shallow columns: stretch the deepest non-empty label down through blank levels.
    if nlevels > 1:
        for col_idx, col in enumerate(columns, start=1):
            last_nonempty = max(
                (i for i, part in enumerate(col) if str(part).strip()),
                default=None,
            )
            if last_nonempty is None or last_nonempty >= nlevels - 1:
                continue
            if any(str(col[i]).strip() for i in range(last_nonempty + 1, nlevels)):
                continue
            ws.merge_cells(
                start_row=last_nonempty + 1,
                start_column=col_idx,
                end_row=nlevels,
                end_column=col_idx,
            )

    for r_idx, row in enumerate(df.itertuples(index=False), start=nlevels + 1):
        for c_idx, val in enumerate(row, start=1):
            ws.cell(row=r_idx, column=c_idx, value=_excel_cell_value(val))

    ws.freeze_panes = f"A{nlevels + 1}"
    for col_idx in range(1, len(columns) + 1):
        ws.column_dimensions[get_column_letter(col_idx)].width = 18

    wb.save(out)


def write_extract_results_xlsx(
    output_folder: str | Path,
    xlsx_path: str | Path | None = None,
) -> Path | None:
    """Write ``extracted_results.xlsx`` (one row per scenario) under ``output_folder``.

    Nested objects become nested header levels; list fields are one JSON cell.
    Returns the path written, or None if there were no extract JSON files.
    """
    rows = load_extract_rows(output_folder)
    if not rows:
        return None
    df = _rows_to_dataframe(rows)
    out = Path(xlsx_path) if xlsx_path else Path(output_folder) / "extracted_results.xlsx"
    out.parent.mkdir(parents=True, exist_ok=True)
    _write_dataframe_xlsx(df, out)
    return out
