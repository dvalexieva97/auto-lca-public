"""
CLI for agent-based SOG extraction, comparison, enrichment, and stats regen.

Installed as the ``auto-lca`` command (``python -m auto_lca`` also works).

Subcommands
-----------
sample-config
    Copy a bundled sample config (lca|generic × json|xlsx) into the current folder.
extract
    Run AgentExtractor on PDFs → per-paper {pid}.json + extracted_results.xlsx
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
    auto-lca sample-config --domain lca --format xlsx
    auto-lca extract --pdfs baldini2020.pdf --config sample_config_lca.xlsx --output results

    auto-lca reclassify-scenarios \\
        --extracts-dir results \\
        --config sample_config_lca.json \\
        --pdf-dir papers/

    auto-lca compare \\
        --output-folder results \\
        --manual-xlsx reference.xlsx \\
        --manual-sheet Sheet1
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

# Before any import that pulls in torch / sentence-transformers (macOS note).
logging.getLogger(
    "torch.distributed.elastic.multiprocessing.redirects"
).setLevel(logging.ERROR)

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
    entry = getattr(args, "entry_default_domain", "lca")
    domain = getattr(args, "domain", None) or entry
    # Legacy --mini mapped to the LCA starter (same field sets as before).
    if getattr(args, "mini", False) and not getattr(args, "domain", None):
        domain = "lca"
    name = f"sample_config_{domain}"
    src = _SAMPLE_CONFIG_DIR / f"{name}.{args.format}"
    if not src.is_file():
        raise SystemExit(f"Bundled sample not found: {src.name}")
    dest = Path(args.output or src.name)
    if dest.exists() and not args.overwrite:
        raise SystemExit(f"{dest} already exists (use --overwrite to replace it)")
    shutil.copyfile(src, dest)
    prog = "auto-extract" if entry == "generic" else "auto-lca"
    print(f"Wrote {dest}. Edit the Pipeline / Field sets, then run:")
    print(f"  {prog} extract --pdfs <your PDFs> --config {dest} --output results")
    if args.format == "xlsx":
        print(
            "Note: XLSX Field sets only support scalar data_type strings. "
            "Nested objects/lists require the JSON sample config."
        )
        if domain == "generic":
            print(
                "Generic XLSX sample: scenarios on + flat Name/Score per case. "
                "For a Models[] list (scenarios off), use --format json."
            )


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


def main(argv=None, *, entry_default_domain: str = "lca", prog: str | None = None):
    # MISTRAL_API_KEY etc. from a .env in the current folder, if present.
    load_dotenv(Path.cwd() / ".env")
    # Mute litellm's "Give Feedback / Get Help" banner (and chatter) on errors.
    try:
        import logging

        import litellm

        litellm.suppress_debug_info = True
        logging.getLogger("LiteLLM").setLevel(logging.ERROR)
        logging.getLogger("litellm").setLevel(logging.ERROR)
    except Exception:
        pass

    is_generic_entry = entry_default_domain == "generic"
    parser = argparse.ArgumentParser(
        prog=prog or ("auto-extract" if is_generic_entry else "auto-lca"),
        description=(
            "Extract structured data from research papers (generic scholarly domain)."
            if is_generic_entry
            else "Extract structured LCA data from papers, and compare it to ground truth."
        ),
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {_version()}")
    parser.set_defaults(entry_default_domain=entry_default_domain)
    sub = parser.add_subparsers(dest="cmd", required=True)

    # --- sample-config ---
    p_sample = sub.add_parser(
        "sample-config",
        help="Copy a bundled sample config (lca or generic) into the current folder",
    )
    p_sample.add_argument(
        "--format",
        choices=["json", "xlsx"],
        default="json",
        help="json or xlsx",
    )
    p_sample.add_argument(
        "--domain",
        choices=["lca", "generic"],
        default=None,
        help="Which sample to copy (default: this entry's domain)",
    )
    p_sample.add_argument(
        "--mini",
        action="store_true",
        help=argparse.SUPPRESS,  # legacy alias → LCA sample
    )
    p_sample.add_argument("--output", default=None, help="Destination file path")
    p_sample.add_argument("--overwrite", action="store_true")
    p_sample.set_defaults(func=_cmd_sample_config)

    # --- extract ---
    p_extract = sub.add_parser(
        "extract", help="Extract fields from PDFs via AgentExtractor"
    )
    p_extract.add_argument(
        "--pdfs",
        nargs="+",
        required=True,
        help="PDF file(s) and/or folder(s) of PDFs",
    )
    p_extract.add_argument(
        "--config",
        required=True,
        help="Domain config (.json or .xlsx; legacy .csv still converts)",
    )
    p_extract.add_argument(
        "--output", required=True, help="Folder for {pid}.json results"
    )
    p_extract.add_argument(
        "--domain",
        choices=["lca", "generic"],
        default=None,
        help=(
            "Prompt pack: lca or generic. Overrides defaults.domain in the config. "
            f"If neither is set, this entry defaults to {entry_default_domain!r}."
        ),
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
        "--domain",
        choices=["lca", "generic"],
        default=None,
        help="Prompt pack for reclassify (same precedence as extract)",
    )
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
        default="sample_config_lca.json",
        help="Field-set config (default: sample_config_lca.json in the cwd)",
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

    args = parser.parse_args(argv)
    args.func(args)


def main_generic(argv=None):
    """Entry point for ``auto-extract``: same CLI, default domain ``generic``."""
    main(argv, entry_default_domain="generic", prog="auto-extract")


if __name__ == "__main__":
    main()
