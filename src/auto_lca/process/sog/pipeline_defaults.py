"""Documented pipeline knobs shown in sample configs (JSON + XLSX Pipeline sheet)."""

from __future__ import annotations

from typing import Any

# (key, default_for_lca_sample, default_for_generic_sample, description)
PIPELINE_SETTINGS: list[tuple[str, Any, Any, str]] = [
    (
        "domain",
        "lca",
        "generic",
        "Prompt pack: lca or generic (also set by auto-lca vs auto-extract).",
    ),
    (
        "extract_production_systems",
        True,
        False,
        "Identify product systems / scenarios. false = one paper-level extraction.",
    ),
    (
        "extract_non_primary_production_systems",
        False,
        False,
        "Also extract fields for non-primary systems (e.g. literature comparators).",
    ),
    (
        "by_production_system",
        True,
        False,
        "Default grain for field sets that omit by_production_system.",
    ),
    (
        "strip_bibliography",
        True,
        True,
        "Drop the reference list from PDF text before extraction.",
    ),
    (
        "extract_sections",
        False,
        False,
        "Write Methods/Results/… split + extracted_sections/{pid}.json.",
    ),
    (
        "include_pdf_tables",
        False,
        True,
        "Append PyMuPDF table grids (CSV) after body text.",
    ),
    (
        "allow_additional_fields",
        False,
        True,
        "Keep model keys beyond the configured schema (exploratory).",
    ),
    (
        "model",
        "",
        "",
        "Optional field-extraction model override (blank = Mistral plan default).",
    ),
    (
        "production_system_model",
        "",
        "",
        "Optional scenario-ID model override (blank = plan default).",
    ),
    (
        "scenario_mapper_model",
        "",
        "",
        "Optional compare/mapper model override (blank = plan default).",
    ),
]


# Generic XLSX sample only: scenarios on so flat Name/Score fields fan out
# one row per identified case. JSON sample stays scenarios-off + Models[].
_GENERIC_XLSX_PIPELINE_OVERRIDES: dict[str, Any] = {
    "extract_production_systems": True,
    "by_production_system": True,
}


def defaults_for_domain(domain: str) -> dict[str, Any]:
    """Return sample-config ``defaults`` for ``lca`` or ``generic`` (no blank model keys).

    Matches the **JSON** sample for generic (scenarios off). The XLSX sample
    uses different Pipeline values — see ``pipeline_rows_for_domain(..., sample_format="xlsx")``.
    """
    use_generic = str(domain).strip().lower() == "generic"
    out: dict[str, Any] = {}
    for key, lca_val, generic_val, _desc in PIPELINE_SETTINGS:
        val = generic_val if use_generic else lca_val
        if key in ("model", "production_system_model", "scenario_mapper_model") and val == "":
            continue
        out[key] = val
    return out


def pipeline_rows_for_domain(
    domain: str, *, sample_format: str = "json"
) -> list[dict[str, Any]]:
    """Rows for the XLSX Pipeline sheet: key, value, description.

    ``sample_format="xlsx"`` applies generic XLSX overrides (scenarios on).
    """
    use_generic = str(domain).strip().lower() == "generic"
    fmt = str(sample_format).strip().lower()
    rows = []
    for key, lca_val, generic_val, desc in PIPELINE_SETTINGS:
        val = generic_val if use_generic else lca_val
        if use_generic and fmt == "xlsx" and key in _GENERIC_XLSX_PIPELINE_OVERRIDES:
            val = _GENERIC_XLSX_PIPELINE_OVERRIDES[key]
        rows.append({"key": key, "value": val, "description": desc})
    return rows
