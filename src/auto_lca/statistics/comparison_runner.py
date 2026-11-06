"""
Compare extracted SOG JSON outputs against ground truth and write joined reports.

This is the shared comparison entry point used by:
  - `python -m auto_lca.scripts.agent_sog_runner compare`
  - `___cross_domain/run_domain_compare.py`
  - `python -m auto_lca.scripts.regenerate_statistics` (stats-only recompute)

AI scenario matching (ScenarioMapper) lives here when scenario_match_mode="ai".
That is separate from enrichment (`statistics.enrichment`), which backfills
system-only gaps into ground truth after a comparison already exists.
"""

from __future__ import annotations

import datetime
import os
from typing import Any

import json

import pandas as pd

from auto_lca.statistics.reuse import (
    ReplayScenarioMapper,
    judge_verdicts,
    load_previous_run,
    summary,
)

# Dairy workbook columns that sog-compare historically skipped.
DEFAULT_COLUMNS_TO_SKIP = [
    "METADATA_EXCLUDED DUE TO NO DATA TRACEABILITY",
    "METADATA_DOI",
    "METADATA_REFERENCE",
    "EXTRA INFO_UNNAMED: 39_LEVEL_1",
    "EMISSIONS (DETAILED)_OTHERS",
    "MIDPOINT IMPACT CATEGORIES_FINAL GWP (KG CO2 E/KG FPCM)",
    "MIDPOINT IMPACT CATEGORIES_TYPE OF ALLOCATION.1",
    # Synthetic join columns (also ALWAYS_IGNORE_COLUMNS in compare.py).
    "PID",
    "_scenario_match_text",
]


def run_comparison(
    output_folder: str,
    manual_xlsx: str,
    manual_sheet_name: str,
    llm_judge_enabled: bool = False,
    ignore_columns: list | None = None,
    scenario_match_mode: str = "ai",
    mapping_only: bool = False,
    excluded_fields: list | None = None,
    category_field: str | None = None,
    category_crosswalk: dict | None = None,
    check_provenance: bool = False,
    audited_pids: set | None = None,
    include_literature_comparators: bool = False,
    stats_subdir: str = "stats",
    run_label: str | None = None,
    typed_only_columns: list[str] | None = None,
    reuse_from: str | None = None,
    mapper_model: str | None = None,
    model_profile: str | None = None,
) -> str | None:
    """
    Compare extracted JSON results in `output_folder` against manual GT.

    Writes `output_folder/<stats_subdir>/joined*.xlsx` (or
    `joined_<run_label>_*.xlsx` when ``run_label`` is set).

    Default scenario join is the AI 1-to-1 mapper. Pass scenario_match_mode=
    "name" for the cosine-on-title baseline.
    """
    from auto_lca.process.nlp.document_rank import DocumentRanker
    from auto_lca.statistics.compare import (
        _calculate_comprehensive_statistics,
        _calculate_execution_stats,
        build_scenario_matching,
        compare_data,
        create_llm_meaning_judge,
        load_all_extracted_data,
        load_manual_data,
        save_results_to_excel,
    )

    ignore_columns = list(ignore_columns or [])
    extracted_jsons = [
        os.path.join(output_folder, x)
        for x in os.listdir(output_folder)
        if x.endswith(".json")
    ]
    if not extracted_jsons:
        print("No JSON files found in output folder, skipping comparison.")
        return None

    print("Loading all extracted JSON files...")
    df_auto_all = load_all_extracted_data(extracted_jsons)

    print(f"Loading manual data from: {manual_xlsx} (sheet: {manual_sheet_name!r})")
    df_manual, _ = load_manual_data(
        manual_xlsx, manual_sheet_name, [1, 2], as_nested=True,
        typed_only_columns=typed_only_columns,
    )

    print("Initializing ranker for fuzzy scenario matching (this may take a moment)...")
    ranker = DocumentRanker()
    _ = ranker._get_bi_encoder()

    llm_judge = None
    if llm_judge_enabled:
        # Pin a dated snapshot so paper-table LLM metrics stay reproducible.
        # Override with LLM_JUDGE_MODEL if needed (alias or snapshot id).
        llm_judge_model = os.environ.get(
            "LLM_JUDGE_MODEL", "gpt-5.4-mini-2026-03-17"
        )
        print(
            "Initializing LLM judge for meaning equality evaluation "
            f"(model={llm_judge_model})..."
        )
        llm_judge = create_llm_meaning_judge(
            model=llm_judge_model,
            api_key=os.environ.get("OPENAI_API_KEY"),
            temperature=0.0,
            max_workers=int(os.environ.get("LLM_JUDGE_WORKERS", "32")),
        )
        print(
            f"LLM judge ready (model={llm_judge_model}, parallel workers="
            f"{getattr(llm_judge, 'max_workers', 32)})"
        )
    else:
        print(
            "LLM judge disabled (pass --llm-judge to enable; requires OPENAI_API_KEY)."
        )

    scenario_mapper = None
    if scenario_match_mode == "ai":
        from auto_lca.process.sog.model_profiles import resolve_profile_models
        from auto_lca.shared.model_preflight import preflight_user_models
        from auto_lca.statistics.scenario_mapper import ScenarioMapper

        resolved = resolve_profile_models(
            model_profile,
            mapper_model=mapper_model,
        )
        mapper_id = resolved["mapper_model"]
        preflight_user_models(
            mapper_model=mapper_id,
            plan_profile=resolved.get("plan_profile"),
            plan_defaults=resolved.get("plan_defaults"),
            has_user_override=bool(resolved.get("has_user_override")),
        )
        print(
            f"Initializing AI scenario mapper ({mapper_id}; "
            f"profile={resolved['profile']})..."
        )
        scenario_mapper = ScenarioMapper(model=mapper_id)

    # Reuse an earlier run's matching and judge verdicts where inputs are unchanged.
    if reuse_from:
        previous_matching, previous_joined = load_previous_run(reuse_from)
        if scenario_mapper is not None:
            scenario_mapper = ReplayScenarioMapper(scenario_mapper, previous_matching)
        if llm_judge is not None:
            n = llm_judge.seed(judge_verdicts(previous_joined))
            print(f"LLM judge: reusing {n} verdicts from {reuse_from}")

    lit_note = (
        ", include literature comparators (TEST)"
        if include_literature_comparators
        else ""
    )
    print(
        "Comparing extracted data against manual ground truth "
        f"(scenario match mode={scenario_match_mode}"
        f"{', mapping only' if mapping_only else ''}{lit_note})..."
    )
    excluded_fields = excluded_fields or []
    excluded_field_names = {
        item["field"] for item in excluded_fields if item.get("field")
    }
    df_joined = compare_data(
        df_auto_all,
        df_manual,
        ranker,
        ignore_columns=list(set(ignore_columns) | excluded_field_names),
        max_workers=os.cpu_count(),
        llm_judge=llm_judge,
        scenario_match_mode=scenario_match_mode,
        scenario_mapper=scenario_mapper,
        mapping_only=mapping_only,
        include_literature_comparators=include_literature_comparators,
    )

    print("Calculating execution statistics...")
    df_execution_stats = _calculate_execution_stats(extracted_jsons)

    empty = pd.DataFrame()
    if mapping_only or df_joined.empty:
        df_stats_by_scenario = empty
        df_stats_by_paper = empty
        df_stats_by_field_scenario = empty
        df_stats_by_field_paper = empty
        df_stats_by_field = empty
        df_summary_stats = empty
        df_sensitivity = empty
        df_f1 = empty
    else:
        print("Calculating comprehensive statistics...")
        (
            df_stats_by_scenario,
            df_stats_by_paper,
            df_stats_by_field_scenario,
            df_stats_by_field_paper,
            df_stats_by_field,
            df_summary_stats,
            df_sensitivity,
            df_f1,
        ) = _calculate_comprehensive_statistics(df_joined, None)

    stats_dir = os.path.join(output_folder, stats_subdir)
    os.makedirs(stats_dir, exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    prefix = f"joined_{run_label}_" if run_label else "joined"
    output_excel_path = os.path.join(stats_dir, f"{prefix}{timestamp}.xlsx")

    save_results_to_excel(
        output_excel_path,
        df_auto_all,
        df_joined,
        df_stats_by_scenario,
        df_stats_by_paper,
        df_stats_by_field_scenario,
        df_stats_by_field_paper,
        df_stats_by_field,
        df_summary_stats,
        df_sensitivity,
        df_f1,
        df_execution_stats,
        excluded_fields=excluded_fields,
        category_field=category_field,
        category_crosswalk=category_crosswalk,
        check_provenance=check_provenance,
        audited_pids=audited_pids,
    )

    if reuse_from:
        record = {"reuse_from": reuse_from, **summary(
            scenario_mapper if isinstance(scenario_mapper, ReplayScenarioMapper) else None,
            llm_judge)}
        if isinstance(scenario_mapper, ReplayScenarioMapper):
            record["matched_live"] = scenario_mapper.matched_live
        with open(os.path.splitext(output_excel_path)[0] + ".reuse.json", "w") as f:
            json.dump(record, f, indent=2)
        print(f"Reuse: {record}")

    print(f"\nComparison complete! Results saved to: {output_excel_path}")
    if not df_summary_stats.empty:
        print("\nSummary stats:")
        print(df_summary_stats.to_string(index=False))
    else:
        _, match_stats = build_scenario_matching(
            df_joined, category_field=category_field
        )
        if not match_stats.empty:
            print("\nScenario matching stats:")
            print(match_stats.tail(1).to_string(index=False))

    return output_excel_path


def load_joined_data(input_path: str) -> pd.DataFrame:
    """Load Joined Data from an existing Excel or CSV comparison report."""
    print(f"Loading joined data from: {input_path}")
    file_ext = os.path.splitext(input_path)[1].lower()

    if file_ext == ".csv":
        df_joined = pd.read_csv(input_path)
        print(f"Loaded {len(df_joined)} rows from CSV file")
        return df_joined
    if file_ext in [".xlsx", ".xls"]:
        try:
            df_joined = pd.read_excel(input_path, sheet_name="Joined Data")
            print(f"Loaded {len(df_joined)} rows from 'Joined Data' sheet")
            return df_joined
        except ValueError as e:
            if "Worksheet named 'Joined Data' not found" in str(e):
                print("'Joined Data' sheet not found, trying first sheet...")
                df_joined = pd.read_excel(input_path)
                print(f"Loaded {len(df_joined)} rows from first sheet")
                return df_joined
            raise
    raise ValueError(
        f"Unsupported file format: {file_ext}. Please use .csv, .xlsx, or .xls files."
    )


def regenerate_statistics(
    input_path: str,
    output_excel_path: str,
    scenario_similarity_threshold: float | None = None,
    include_execution_stats: bool = False,
    execution_stats_path: str | None = None,
) -> None:
    """
    Recompute stats sheets from an existing Joined Data file.

    Same `_calculate_comprehensive_statistics` / `save_results_to_excel` path
    that `run_comparison` uses after joining — useful when join is already
    done and only scoring/aggregation code changed.
    """
    from auto_lca.statistics.compare import (
        _calculate_comprehensive_statistics,
        _calculate_execution_stats,
        save_results_to_excel,
    )

    df_joined = load_joined_data(input_path)
    if df_joined.empty:
        print("Error: No data found in the input file")
        return

    print("Calculating comprehensive statistics...")
    if scenario_similarity_threshold is not None:
        print(f"Using scenario similarity threshold: {scenario_similarity_threshold}")

    (
        df_stats_by_scenario,
        df_stats_by_paper,
        df_stats_by_field_scenario,
        df_stats_by_field_paper,
        df_stats_by_field,
        df_summary_stats,
        df_sensitivity,
        df_f1,
    ) = _calculate_comprehensive_statistics(df_joined, scenario_similarity_threshold)

    df_execution_stats = None
    if include_execution_stats and execution_stats_path:
        print("Calculating execution statistics...")
        try:
            if os.path.isdir(execution_stats_path):
                extracted_jsons = [
                    os.path.join(execution_stats_path, f)
                    for f in os.listdir(execution_stats_path)
                    if f.endswith(".json")
                ]
                df_execution_stats = _calculate_execution_stats(extracted_jsons)
            else:
                print(
                    f"Warning: Execution stats path {execution_stats_path} is not a directory"
                )
        except Exception as e:
            print(f"Warning: Could not calculate execution stats: {e}")

    print("Saving results to Excel...")
    save_results_to_excel(
        output_excel_path,
        None,
        df_joined,
        df_stats_by_scenario,
        df_stats_by_paper,
        df_stats_by_field_scenario,
        df_stats_by_field_paper,
        df_stats_by_field,
        df_summary_stats,
        df_sensitivity,
        df_f1,
        df_execution_stats,
    )
    print(f"Statistics regeneration complete! Results saved to: {output_excel_path}")
