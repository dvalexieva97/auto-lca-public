"""
CLI for agent-based SOG extraction, comparison, enrichment, and stats regen.

Installed as the ``auto-lca`` command (``python -m auto_lca`` also works).

Subcommands
-----------
sample-config
    Copy a bundled sample config (json/csv/xlsx) into the current folder.
extract
    Run AgentExtractor on PDFs → per-paper {pid}.json
reclassify-scenarios
    Re-tag scenario_category on existing extracts (no field re-extract).
    Uses candidate names/descriptions + first ~3k chars of PDF text only.
compare
    Join extracts vs ground truth (AI scenario match by default) → stats/joined*.xlsx
    This is where ScenarioMapper / field cosine scoring live — same path as the
    pre-agent sog-compare / sog_runner --compare flow.
enrich-excel
    After a comparison exists: verify system-only gaps against the PDF and
    backfill an Excel ground truth (dairy). NOT the AI scenario matcher.
regenerate-stats
    Recompute stats sheets from an existing Joined Data workbook (same
    `_calculate_comprehensive_statistics` used inside compare).

Examples
--------
    auto-lca sample-config --mini --format csv
    auto-lca extract --pdfs baldini2020.pdf --config sample_config_mini.csv --output results

    python -m auto_lca.scripts.agent_sog_runner extract \\
        --pdfs ___cross_domain/dairy/input-papers/*.pdf \\
        --config ___cross_domain/dairy/configs/config_dairy.json \\
        --output ___cross_domain/dairy/outputs-agent-40-final

    python -m auto_lca.scripts.agent_sog_runner reclassify-scenarios \\
        --extracts-dir ___cross_domain/dairy/outputs-agent-40-final \\
        --config ___cross_domain/dairy/configs/config_dairy.json \\
        --pdf-dir ___cross_domain/dairy/input-papers

    python -m auto_lca.scripts.agent_sog_runner compare \\
        --output-folder ___cross_domain/dairy/outputs-agent-40-final \\
        --manual-xlsx ___cross_domain/dairy/ground_truth/dairy_ground_truth.xlsx \\
        --manual-sheet "Final Data - Dianka edits with " \\
        --dairy-ignore
"""

from __future__ import annotations

import argparse
import json
import shutil
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

from dotenv import load_dotenv

from auto_lca.process.sog.agent_sog import DEFAULT_MODEL
from auto_lca.process.sog.extract_runner import run_extract
from auto_lca.process.sog.extract_sections_runner import run_extract_sections
from auto_lca.process.sog.reclassify_runner import run_reclassify_scenarios
from auto_lca.statistics.comparison_runner import (
    DEFAULT_COLUMNS_TO_SKIP,
    regenerate_statistics,
    run_comparison,
)
from auto_lca.statistics.enrichment import run_enrich_excel
from auto_lca.statistics.provenance_checker import load_audited_pids


def _cmd_compare(args):
    ignore = list(DEFAULT_COLUMNS_TO_SKIP) if args.dairy_ignore else []
    excluded_fields = None
    if args.excluded_fields:
        with open(args.excluded_fields) as f:
            excluded_fields = json.load(f)

    category_crosswalk = None
    category_field = args.category_field
    if args.category_crosswalk:
        with open(args.category_crosswalk) as f:
            crosswalk_doc = json.load(f)
        category_crosswalk = {
            "auto": crosswalk_doc.get("auto", {}),
            "manual": crosswalk_doc.get("manual", {}),
        }
        category_field = category_field or crosswalk_doc.get("category_field")

    audited_pids = None
    if args.audit_samples:
        audited_pids = load_audited_pids(args.audit_samples)

    run_comparison(
        output_folder=args.output_folder,
        manual_xlsx=args.manual_xlsx,
        manual_sheet_name=args.manual_sheet,
        llm_judge_enabled=args.llm_judge,
        ignore_columns=ignore,
        scenario_match_mode=args.scenario_match_fields,
        mapping_only=args.mapping_only,
        excluded_fields=excluded_fields,
        category_field=category_field,
        category_crosswalk=category_crosswalk,
        check_provenance=args.check_provenance,
        audited_pids=audited_pids,
        mapper_model=getattr(args, "mapper_model", None),
        model_profile=getattr(args, "model_profile", None),
    )


def _cmd_regenerate_stats(args):
    regenerate_statistics(
        input_path=args.joined,
        output_excel_path=args.output,
        scenario_similarity_threshold=args.threshold,
        include_execution_stats=bool(args.execution_stats_path),
        execution_stats_path=args.execution_stats_path,
    )


_SAMPLE_CONFIG_DIR = Path(__file__).resolve().parent.parent / "process" / "sog"


def _cmd_sample_config(args):
    name = "sample_config_mini" if args.mini else "sample_config"
    src = _SAMPLE_CONFIG_DIR / f"{name}.{args.format}"
    dest = Path(args.output or src.name)
    if dest.exists() and not args.overwrite:
        raise SystemExit(f"{dest} already exists (use --overwrite to replace it)")
    shutil.copyfile(src, dest)
    print(f"Wrote {dest}. Edit the field sets, then run:")
    print(f"  auto-lca extract --pdfs <your PDFs> --config {dest} --output results")


def _resolve_config(config: str) -> str:
    """Return the config path; copy a bundled sample here if one is named but missing."""
    path = Path(config)
    if path.exists():
        return config
    bundled = _SAMPLE_CONFIG_DIR / path.name
    if path.parent == Path(".") and path.name.startswith("sample_config") and bundled.exists():
        shutil.copyfile(bundled, path)
        print(f"Copied the bundled {path.name} into this folder.")
        return config
    raise SystemExit(
        f"Config file not found: {config}\n"
        "Create one with `auto-lca sample-config`, or pass the path to your own."
    )


def _cmd_extract(args):
    args.config = _resolve_config(args.config)
    run_extract(args)


def _version() -> str:
    try:
        return version("auto-lca")
    except PackageNotFoundError:
        return "unknown (not installed)"


def main():
    # MISTRAL_API_KEY etc. from a .env in the current folder, if present.
    load_dotenv(Path.cwd() / ".env")

    parser = argparse.ArgumentParser(
        prog="auto-lca",
        description="Extract structured LCA data from papers, and compare it to ground truth.",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {_version()}")
    sub = parser.add_subparsers(dest="cmd", required=True)

    # --- sample-config ---
    p_sample = sub.add_parser(
        "sample-config", help="Copy a bundled sample config into the current folder"
    )
    p_sample.add_argument("--format", choices=["json", "csv", "xlsx"], default="json")
    p_sample.add_argument(
        "--mini", action="store_true", help="Two-field-set config for a quick first run"
    )
    p_sample.add_argument("--output", default=None, help="Destination file path")
    p_sample.add_argument("--overwrite", action="store_true")
    p_sample.set_defaults(func=_cmd_sample_config)

    # --- extract ---
    p_extract = sub.add_parser(
        "extract", help="Extract fields from PDFs via AgentExtractor"
    )
    p_extract.add_argument("--pdfs", nargs="+", required=True, help="PDF files")
    p_extract.add_argument(
        "--config", required=True, help="Domain config (.json, .csv or .xlsx)"
    )
    p_extract.add_argument(
        "--output", required=True, help="Folder for {pid}.json results"
    )
    p_extract.add_argument(
        "--model",
        default=None,
        help="Override field-extraction model (checked against your Mistral plan)",
    )
    p_extract.add_argument(
        "--scenario-model",
        default=None,
        help="Override scenario model (checked against your Mistral plan)",
    )
    p_extract.add_argument(
        "--model-profile",
        choices=["auto", "paid", "free"],
        default="auto",
        help=(
            "Plan defaults from model_profiles.json. auto (default) uses Mistral "
            "plan_name (FREE→free, else paid). paid/free force that pair. "
            "Config/CLI model overrides still win and are validated."
        ),
    )
    p_extract.add_argument("--overwrite", action="store_true")
    p_extract.add_argument(
        "--scenarios-only",
        action="store_true",
        help="Identify/validate scenarios only (no field extraction)",
    )
    p_extract.set_defaults(func=_cmd_extract)

    # --- reclassify-scenarios (in-place category refresh) ---
    p_re = sub.add_parser(
        "reclassify-scenarios",
        help="Re-tag scenario_category on existing extracts (no field re-extract)",
    )
    p_re.add_argument("--extracts-dir", required=True)
    p_re.add_argument("--config", required=True)
    p_re.add_argument(
        "--pdf-dir",
        default=None,
        help="PDF folder for paper-context prefix (falls back to metadata.pdf_path)",
    )
    p_re.add_argument("--model", default=None)
    p_re.add_argument("--scenario-model", default=None)
    p_re.add_argument(
        "--model-profile",
        choices=["auto", "paid", "free"],
        default="auto",
        help="Plan default models (see extract --help)",
    )
    p_re.add_argument("--dry-run", action="store_true")
    p_re.add_argument("--limit", type=int, default=None, help="Max papers (smoke test)")
    p_re.add_argument(
        "--refresh-sections",
        action="store_true",
        help="Rebuild extracted_sections/{pid}.json from PDF before classifying",
    )
    p_re.set_defaults(func=run_reclassify_scenarios)

    # --- extract-sections (cheap non-AI) ---
    p_sec = sub.add_parser(
        "extract-sections",
        help="Rebuild extracted_sections/{pid}.json (header split; no LLM)",
    )
    p_sec.add_argument("--extracts-dir", required=True)
    p_sec.add_argument("--pdf-dir", default=None)
    p_sec.add_argument("--overwrite", action="store_true")
    p_sec.add_argument("--limit", type=int, default=None)
    p_sec.set_defaults(func=run_extract_sections)

    # --- compare (AI scenario match + field scoring) ---
    p_cmp = sub.add_parser(
        "compare",
        help="Join extracts vs GT (AI scenario mapper by default) and write stats",
    )
    p_cmp.add_argument("--output-folder", required=True)
    p_cmp.add_argument("--manual-xlsx", required=True)
    p_cmp.add_argument("--manual-sheet", default="Ground Truth")
    p_cmp.add_argument(
        "--scenario-match-fields",
        choices=["name", "name_description", "ai"],
        default="ai",
    )
    p_cmp.add_argument("--mapping-only", action="store_true")
    p_cmp.add_argument(
        "--dairy-ignore",
        action="store_true",
        help="Skip dairy-specific columns historically ignored in scoring",
    )
    p_cmp.add_argument("--llm-judge", action="store_true")
    p_cmp.add_argument(
        "--mapper-model",
        default=None,
        help="Override AI scenario-mapper model (checked against your Mistral plan)",
    )
    p_cmp.add_argument(
        "--model-profile",
        choices=["auto", "paid", "free"],
        default="auto",
        help="Plan default mapper model (see extract --help)",
    )
    p_cmp.add_argument("--excluded-fields", help="JSON list of {field, source, reason}")
    p_cmp.add_argument("--category-field")
    p_cmp.add_argument("--category-crosswalk")
    p_cmp.add_argument("--check-provenance", action="store_true")
    p_cmp.add_argument("--audit-samples", nargs="*", default=[])
    p_cmp.set_defaults(func=_cmd_compare)

    # --- enrich-excel (dairy xlsx GT) ---
    p_enrich_xlsx = sub.add_parser(
        "enrich-excel",
        help="Backfill Excel GT gaps verified against the PDF (dairy 2-row header)",
    )
    p_enrich_xlsx.add_argument("--joined", required=True)
    p_enrich_xlsx.add_argument("--manual-xlsx", required=True)
    p_enrich_xlsx.add_argument("--manual-sheet", required=True)
    p_enrich_xlsx.add_argument("--output-xlsx", required=True)
    p_enrich_xlsx.add_argument("--pdfs-dir", required=True)
    p_enrich_xlsx.add_argument(
        "--config",
        default="___cross_domain/dairy/configs/config_dairy.json",
    )
    p_enrich_xlsx.add_argument("--model", default=DEFAULT_MODEL)
    p_enrich_xlsx.set_defaults(func=run_enrich_excel)

    # --- regenerate-stats ---
    p_regen = sub.add_parser(
        "regenerate-stats",
        help="Recompute stats sheets from an existing Joined Data workbook",
    )
    p_regen.add_argument("--joined", required=True, help="Existing joined CSV/XLSX")
    p_regen.add_argument("--output", required=True, help="Output Excel path")
    p_regen.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="Optional scenario similarity threshold",
    )
    p_regen.add_argument(
        "--execution-stats-path",
        default=None,
        help="Folder of extract JSONs for execution-stats sheet",
    )
    p_regen.set_defaults(func=_cmd_regenerate_stats)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
