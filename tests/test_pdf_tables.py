"""Structured PDF table extraction for LLM prompts."""

from pathlib import Path

import pymupdf

from auto_lca.process.nlp.extract import TextExtractor


def test_format_pdf_tables_missing_file():
    text, n = TextExtractor.format_pdf_tables_as_text("/no/such/file.pdf")
    assert text == ""
    assert n == 0


def test_format_pdf_tables_from_attention_paper(tmp_path: Path):
    """Synthetic grid PDF (sample papers are not shipped in CI)."""
    path = tmp_path / "table.pdf"
    doc = pymupdf.open()
    page = doc.new_page()
    rows = [
        ["Model", "Metric", "Score"],
        ["Base", "BLEU", ""],
        ["Big", "BLEU", "28.4"],
    ]
    x0, y0, cw, rh = 72, 72, 100, 20
    for r in range(len(rows) + 1):
        y = y0 + r * rh
        page.draw_line(pymupdf.Point(x0, y), pymupdf.Point(x0 + 3 * cw, y))
    for c in range(4):
        x = x0 + c * cw
        page.draw_line(pymupdf.Point(x, y0), pymupdf.Point(x, y0 + 3 * rh))
    for r, row in enumerate(rows):
        for c, val in enumerate(row):
            page.insert_text((x0 + 4 + c * cw, y0 + 14 + r * rh), val, fontsize=10)
    doc.save(path)
    doc.close()

    text, n = TextExtractor.format_pdf_tables_as_text(str(path))
    assert n >= 1
    assert "EXTRACTED TABLES" in text
    assert "### Table (page" in text
    # Blank cells should survive as empty CSV fields, not invented numbers.
    assert ",," in text or text.count("\n") > 5
