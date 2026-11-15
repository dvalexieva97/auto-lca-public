"""
Excel ground-truth enrichment (dairy): verify system-only gaps against the
source PDF and write confirmed values into a copy of the GT workbook.

This is NOT the AI scenario matcher. Scenario matching happens in
`statistics.comparison_runner.run_comparison` (ScenarioMapper). Enrichment
runs *after* a comparison workbook already exists, using its Joined Data
sheet + gap_analysis.find_gaps.
"""

from __future__ import annotations

import json
import os
import shutil
import time

import openpyxl
import pandas as pd

from auto_lca.process.sog.agent_sog import AgentExtractor
from auto_lca.statistics.gap_analysis import find_gaps


def _pdf_path_for_pid(pdfs_dir: str, pid: str) -> str | None:
    p = os.path.join(pdfs_dir, f"{pid}.pdf")
    return p if os.path.exists(p) else None


def _build_excel_index(
    ws, header_rows=(2, 3), pid_col_header=("METADATA", "PID"), data_start_row=4
):
    """Walk the 2-row-header sheet once: returns
    (col_index: {(CONCEPT_UPPER, FIELD_UPPER): col}, pid_col, scenario_col, last_row).
    Concept cells are merged-forward (only set on the first column of each
    concept block), exactly like Dairy LCI-3.xlsx's real layout."""
    concept_row, field_row = header_rows
    col_index: dict[tuple[str, str], int] = {}
    pid_col = None
    scenario_col = None
    concept = None
    for c in range(1, ws.max_column + 1):
        v_concept = ws.cell(row=concept_row, column=c).value
        v_field = ws.cell(row=field_row, column=c).value
        if v_concept is not None:
            concept = str(v_concept).strip()
        if concept is None or v_field is None:
            continue
        field = str(v_field).strip()
        col_index[(concept.upper(), field.upper())] = c
        if (concept.upper(), field.upper()) == (
            pid_col_header[0].upper(),
            pid_col_header[1].upper(),
        ):
            pid_col = c
        if concept.upper() == "STUDY DETAILS" and field.upper() == "SCENARIO NAME":
            scenario_col = c

    last_row = data_start_row - 1
    if pid_col:
        for r in range(data_start_row, ws.max_row + 1):
            if ws.cell(row=r, column=pid_col).value:
                last_row = r
    return col_index, pid_col, scenario_col, last_row


def run_enrich_excel(args):
    """Backfill system-only gaps into a COPY of a 2-row-header Excel GT.

    Never touches --manual-xlsx itself.
    """
    df = pd.read_excel(args.joined, sheet_name="Joined Data")
    report = find_gaps(df)
    by_pid = report.by_pid()

    log_path = args.output_xlsx.replace(".xlsx", "_enrichment_log.json")
    if os.path.exists(args.output_xlsx):
        print(f"resuming: {args.output_xlsx} already exists")
    else:
        shutil.copy(args.manual_xlsx, args.output_xlsx)
        print(
            f"created enriched copy: {args.output_xlsx} (source untouched: {args.manual_xlsx})"
        )

    # Two views of the same file: `wb` (formulas intact) is what actually
    # gets edited and saved -- writes only ever touch cells that were
    # already blank, so no formula cell is ever at risk. `wb_read`
    # (data_only=True) is used ONLY to read values, because many Scenario
    # Name cells are formulas (e.g. `=CONCAT(I176," - ",F176)`) and
    # loading without data_only=True returns that formula string instead
    # of its actual computed value, breaking every row match for those
    # rows (confirmed live: 555/1932 gaps mismatched before this fix).
    wb = openpyxl.load_workbook(args.output_xlsx)
    ws = wb[args.manual_sheet]
    ws_read = openpyxl.load_workbook(args.output_xlsx, data_only=True)[
        args.manual_sheet
    ]

    # Bake every formula cell (e.g. Scenario Name built via
    # `=CONCAT(...)`) into its cached literal value BEFORE any further
    # save. openpyxl has no way to write a formula *and* its cached
    # result together -- once a formula-carrying workbook opened with
    # data_only=False is saved back out, the cached value is dropped
    # from the file entirely, so a later data_only=True read (exactly
    # what compare.py's load_manual_data does) gets None instead of the
    # real text. Confirmed live: this silently broke every scenario
    # match for one 29-scenario paper whose Scenario Name column is all
    # CONCAT formulas, even though enrichment never touched that column
    # -- the mere act of opening+resaving the workbook was enough.
    baked = 0
    for row in ws.iter_rows():
        for cell in row:
            v = cell.value
            if isinstance(v, str) and v.strip().startswith("="):
                cell.value = ws_read.cell(row=cell.row, column=cell.column).value
                baked += 1
    if baked:
        print(
            f"baked {baked} formula cell(s) to their cached literal value before saving"
        )
        wb.save(
            args.output_xlsx
        )  # persist immediately, independent of the per-paper save() below

    col_index, pid_col, scenario_col, last_row = _build_excel_index(ws_read)
    if not pid_col or not scenario_col:
        raise RuntimeError(
            f"could not locate PID/Scenario Name columns in {args.manual_sheet!r} "
            f"(pid_col={pid_col}, scenario_col={scenario_col})"
        )

    # pid -> {scenario_name: row}. Scenario names come straight from this
    # same sheet's own cells (not uppercased -- only load_manual_data's
    # column headers get uppercased, not data values), so they match
    # gap.scenario_manual (sourced from the joined sheet's manual column,
    # itself sourced from this same workbook) verbatim.
    row_index: dict[str, dict[str, int]] = {}
    for r in range(4, last_row + 1):
        pid = ws_read.cell(row=r, column=pid_col).value
        if not pid:
            continue
        scen = ws_read.cell(row=r, column=scenario_col).value
        row_index.setdefault(str(pid).strip(), {})[
            str(scen).strip() if scen else ""
        ] = r

    extractor = AgentExtractor(config_path=args.config, model=args.model)

    run_log = []
    if os.path.exists(log_path):
        with open(log_path) as f:
            run_log = json.load(f)
    done_pids = {entry["pid"] for entry in run_log if entry.get("status") == "ok"}

    def save():
        wb.save(args.output_xlsx)
        with open(log_path, "w") as f:
            json.dump(run_log, f, indent=2, ensure_ascii=False)

    total_confirmed = 0
    total_rejected = 0
    total_unmatched = 0
    for pid, gaps in by_pid.items():
        if pid in done_pids:
            print(f"[{pid[:8]}] already done (see {log_path}), skipping")
            continue
        system_only = [g for g in gaps if g.category == "system_only"]
        if not system_only:
            continue
        pdf_path = _pdf_path_for_pid(args.pdfs_dir, pid)
        if not pdf_path:
            print(f"⚠️  skipping {pid[:8]}: no PDF found in {args.pdfs_dir}")
            continue
        rows_for_pid = row_index.get(pid)
        if not rows_for_pid:
            print(f"⚠️  skipping {pid[:8]}: no matching PID in {args.manual_sheet}")
            continue

        print(f"\n[{pid[:8]}] verifying {len(system_only)} gap(s)...")
        t0 = time.time()
        try:
            result = extractor.enrich_gaps(pdf_path=pdf_path, pid=pid, gaps=system_only)
        except Exception as e:
            print(f"  ❌ failed after retries, skipping this paper: {e}")
            run_log.append({"pid": pid, "status": "failed", "error": str(e)})
            save()
            continue

        n_added = sum(len(f) for c in result.additions.values() for f in c.values())
        print(
            f"  -> {n_added} confirmed, {len(result.rejected)} rejected "
            f"({time.time() - t0:.0f}s, {result.tokens.get('total_tokens', '?')} tokens)"
        )
        total_confirmed += n_added
        total_rejected += len(result.rejected)

        n_unmatched = 0
        for scenario_name, concepts in result.additions.items():
            row = rows_for_pid.get(str(scenario_name).strip() if scenario_name else "")
            if row is None:
                # Fall back to the only row for this pid if there's just one.
                if len(rows_for_pid) == 1:
                    row = next(iter(rows_for_pid.values()))
                else:
                    n_unmatched += sum(len(f) for f in concepts.values())
                    print(
                        f"  ⚠️  no row match for scenario {scenario_name!r}, skipping its additions"
                    )
                    continue
            for concept_name, fields in concepts.items():
                for field_name, val in fields.items():
                    col = col_index.get((concept_name.upper(), field_name.upper()))
                    if col is None:
                        n_unmatched += 1
                        print(
                            f"  ⚠️  no column for {concept_name!r} / {field_name!r}, skipping"
                        )
                        continue
                    existing = ws.cell(row=row, column=col).value
                    if existing not in (None, ""):
                        continue  # never overwrite an existing manual value
                    ws.cell(row=row, column=col).value = val
        total_unmatched += n_unmatched

        run_log.append(
            {
                "pid": pid,
                "status": "ok",
                "confirmed": n_added,
                "unmatched": n_unmatched,
                "rejected": result.rejected,
                "tokens": result.tokens,
            }
        )
        save()

    print(f"\n{'=' * 60}")
    print(f"Total confirmed additions this run: {total_confirmed}")
    print(f"Total rejected this run: {total_rejected}")
    print(f"Total confirmed-but-unmatched (no row/column found): {total_unmatched}")
    print(f"Enriched workbook: {args.output_xlsx}")
    print(f"Run log (resumable): {log_path}")
