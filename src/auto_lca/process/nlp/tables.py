"""
Table extraction from PDFs with Tabula and Docling.

Needs the optional extra: pip install "auto-lca[tables]" (Tabula also needs
a Java runtime). Not used by the extraction pipeline, which reads PDF text
with PyMuPDF.
"""

from typing import List

import pandas as pd
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import (
    PdfPipelineOptions,
    TableFormerMode,
    TableStructureOptions,
)
from docling.document_converter import DocumentConverter, PdfFormatOption
from tabula.io import read_pdf

from auto_lca.process.nlp.extract import TextExtractor


def extract_tables_from_pdf(pdf_path: str) -> List[pd.DataFrame]:
    """Tabula first, PyMuPDF as fallback."""
    tables = extract_tables_from_pdf_tabula(pdf_path)
    if not tables:
        tables = TextExtractor.extract_tables_from_pdf_pymupdf(pdf_path)
    if not tables:
        print("Warning: no tables extracted from pdf.")
    return tables


def extract_tables_from_pdf_docling(pdf_path: str, table_mode: str = "fast"):
    """Returns (docling conversion result or None, list of table DataFrames)."""
    print("[Docling] table extraction...")
    converted = None
    try:
        mode = (
            TableFormerMode.FAST
            if table_mode.lower() == "fast"
            else TableFormerMode.ACCURATE
        )
        pipeline_options = PdfPipelineOptions(
            table_structure_options=TableStructureOptions(mode=mode),
        )
        converter = DocumentConverter(
            format_options={
                InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
            }
        )
        converted = converter.convert(pdf_path)
        doc = converted.document
        tables: List[pd.DataFrame] = []
        for table in doc.tables:
            df = table.export_to_dataframe(doc)
            if df is None or df.empty:
                continue
            tables.append(df)
        return converted, tables
    except Exception as e:
        print(f"[Docling] Table extraction failed for {pdf_path}: {e}")  # TODO Log, not print
        return converted, []


def extract_tables_from_pdf_tabula(pdf_path: str) -> List[pd.DataFrame]:
    """One DataFrame per detected table; [] on failure or if no tables."""
    try:
        dfs = read_pdf(pdf_path, pages="all", multiple_tables=True)
        if dfs is None:
            return []
        if isinstance(dfs, pd.DataFrame):
            dfs = [dfs]
        return [df for df in dfs if isinstance(df, pd.DataFrame) and not df.empty]
    except Exception as e:
        print(f"[Tabula] Table extraction failed for {pdf_path}: {e}")
        return []
