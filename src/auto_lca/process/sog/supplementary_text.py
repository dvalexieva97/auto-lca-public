"""Load supplementary materials (PDF / Excel / CSV / DOCX / text) as extractable text."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from auto_lca.process.nlp.extract import TextExtractor

_TABLE_EXTS = {".xlsx", ".xls", ".csv", ".tsv"}
_TEXT_EXTS = {".txt", ".md"}
_DOC_EXTS = {".docx"}


def _excel_to_text(path: Path, max_rows_per_sheet: int = 200) -> str:

    chunks: list[str] = []
    # .xls needs xlrd; .xlsx uses openpyxl (pandas default).
    engine = "xlrd" if path.suffix.lower() == ".xls" else None
    xl = pd.ExcelFile(path, engine=engine) if engine else pd.ExcelFile(path)
    for sheet in xl.sheet_names:
        df = pd.read_excel(path, sheet_name=sheet, header=None, engine=engine)
        if df.empty:
            continue
        if len(df) > max_rows_per_sheet:
            df = df.iloc[:max_rows_per_sheet]
        table = df.fillna("").astype(str)
        chunks.append(f"### Sheet: {sheet}\n{table.to_csv(index=False, header=False)}")
    return "\n\n".join(chunks)


def _docx_to_text(path: Path) -> str:
    from docx import Document

    doc = Document(str(path))
    parts: list[str] = []
    for p in doc.paragraphs:
        t = (p.text or "").strip()
        if t:
            parts.append(t)
    for i, table in enumerate(doc.tables):
        rows = []
        for row in table.rows:
            rows.append([("" if c.text is None else c.text.strip()) for c in row.cells])
        if not rows:
            continue
        lines = [",".join('"' + c.replace('"', '""') + '"' for c in r) for r in rows]
        parts.append(f"### Table {i+1}\n" + "\n".join(lines))
    return "\n\n".join(parts)


def materialize_supplementary_text(
    paths: list[Path] | Path,
    *,
    max_chars: int | None = 120_000,
) -> str:
    """Concatenate SI files into one text block for appending to paper text.

    PDFs are text-extracted; spreadsheets become CSV-like tables; DOCX
    paragraphs + tables are flattened to text.
    """
    allowed = {".pdf", *_TABLE_EXTS, *_TEXT_EXTS, *_DOC_EXTS}
    if isinstance(paths, (str, Path)):
        root = Path(paths)
        files = sorted(
            p
            for p in root.rglob("*")
            if p.is_file()
            and not p.name.startswith(".")
            and p.suffix.lower() in allowed
        )
    else:
        files = [Path(p) for p in paths]

    parts: list[str] = []
    for path in files:
        ext = path.suffix.lower()
        try:
            if ext == ".pdf":
                body = TextExtractor.extract_text_from_pdf(str(path))
            elif ext in {".xlsx", ".xls"}:
                body = _excel_to_text(path)
            elif ext == ".docx":
                body = _docx_to_text(path)
            elif ext in {".csv", ".tsv", *_TEXT_EXTS}:
                body = path.read_text(encoding="utf-8", errors="ignore")
            else:
                continue
        except Exception as e:  # noqa: BLE001 -- SI is best-effort
            parts.append(f"## Supplementary file: {path.name}\n[failed to read: {e}]")
            continue
        body = (body or "").strip()
        if not body:
            continue
        parts.append(f"## Supplementary file: {path.name}\n{body}")

    if not parts:
        return ""

    text = (
        "\n\n===== SUPPLEMENTARY INFORMATION (provided with the paper) =====\n\n"
        + "\n\n".join(parts)
        + "\n\n===== END SUPPLEMENTARY INFORMATION =====\n"
    )
    if max_chars is not None and len(text) > max_chars:
        text = text[: max_chars - 80] + "\n\n[... supplementary text truncated ...]\n"
    return text
