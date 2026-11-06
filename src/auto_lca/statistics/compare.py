"""
Module for comparing auto-extracted data with manual data.

This module provides functionality to:
- Load extracted data from JSON files
- Compare with manual CSV data using fuzzy scenario name matching
- Generate comparison reports in Excel format with statistics
"""

import io
import json
import os
import re
import threading
import time
from collections.abc import Callable, Iterable
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import openpyxl
import pandas as pd

from auto_lca.process.nlp.document_rank import DocumentRanker, stopwords
from auto_lca.shared.util import flatten_extracted
from auto_lca.statistics.field_decomposition import build_field_error_decomposition
from auto_lca.statistics.provenance_checker import (
    compute_field_provenance,
    compute_paper_status_table,
)
from auto_lca.statistics.similarity import DataSimilarity

try:
    from litellm import completion

    LITELLM_AVAILABLE = True
except (ImportError, AttributeError):
    # AttributeError can occur when litellm has an internal circular import
    # (e.g. litellm_core_utils before package init finishes)
    LITELLM_AVAILABLE = False
    completion = None

try:
    from bert_score import score as bert_score_score

    BERT_SCORE_AVAILABLE = True
except ImportError:
    BERT_SCORE_AVAILABLE = False
    bert_score_score = None
# Constants
SCENARIO_NAME_COL = "SCENARIO NAME"
PID_COL = "PID"
SCENARIO_CATEGORY_COL = "SCENARIO_SCENARIO_CATEGORY"
SCENARIO_DEFINITION_ONLY_COL = "SCENARIO_DEFINITION_ONLY"
SCENARIO_DESCRIPTION_COL = "SCENARIO_SCENARIO_DESCRIPTION"
SCENARIO_SIMILARITY_COL = "_scenario_name_similarity_score"
SCENARIO_MATCH_TEXT_AUTO_COL = "_scenario_match_text_auto"
SCENARIO_MATCH_TEXT_MANUAL_COL = "_scenario_match_text_manual"
SCENARIO_MATCH_MODE_NAME = "name"
SCENARIO_MATCH_MODE_NAME_DESCRIPTION = "name_description"
SCENARIO_MATCH_MODE_AI = "ai"
SCENARIO_AI_CONFIDENCE_COL = "_scenario_ai_confidence"
SCENARIO_AI_REASON_COL = "_scenario_ai_reason"
SCENARIO_COSINE_PARTNER_COL = "_scenario_cosine_partner"
SCENARIO_COLLAPSED_INTO_COL = "_scenario_collapsed_into"
SCENARIO_REVIEW_COLLAPSED_COL = "_scenario_review_collapsed"
SCENARIO_PAPER_SCENARIO_COUNT_COL = "_scenario_paper_scenario_count"
# For unmatched autos with no informative mapper reason: was the auto
# scenario name found in the source PDF text?
SCENARIO_TEXT_MENTIONED_COL = "_scenario_text_mentioned"
SCENARIO_TEXT_MENTION_SNIPPET_COL = "_scenario_text_mention_snippet"
# When GT has no prose description, these identity fields stand in.
_GT_IDENTITY_COLS = (
    "SCENARIO CHARACTERIZATION_PRODUCT",
    "SCENARIO CHARACTERIZATION_EMERGING TECHNOLOGY",
    "FEEDSTOCK_FEEDSTOCK",
    "SCENARIO AND PRODUCT IDENTITY_CARBONATION ROUTE",
    "SCENARIO AND PRODUCT IDENTITY_SPECIFIC ROUTE NAME",
    "SCENARIO AND PRODUCT IDENTITY_MAIN PRODUCT",
    "ALKALINE FEEDSTOCK AND MATERIALS_ALKALINE FEEDSTOCK",
)
MATCH_SUFFIX = "_match"
FUZZY_FLAG = True
NULL_KEYWORDS = ["null", "none", "missing", "missing value"]

col_rename_dict = {
    # Older references call the downloaded flag "Scraped".
    "METADATA_SCRAPED": "METADATA_DOWNLOADED",
    "STUDY DETAILS_SCENARIO NAME": "SCENARIO NAME",
    # Non-dairy schemas may group scenario identity under a differently
    # named concept (e.g. chemicals' "Scenario and Product Identity",
    # plastics/fashion's "Scenario Characterization") rather than "Study
    # details" -- add aliases here rather than assuming "Study details"
    # is where every domain's Scenario Name lives.
    "SCENARIO AND PRODUCT IDENTITY_SCENARIO NAME": "SCENARIO NAME",
    "SCENARIO CHARACTERIZATION_SCENARIO NAME": "SCENARIO NAME",
    "METADATA_PID": "PID",
}
# Join-level flags: kept on the comparison sheet, dropped from accuracy.
EXCLUDE_FROM_ACCURACY_COL = "EXCLUDE_FROM_ACCURACY"
# Matched shells without LCI payloads (e.g. alternate_dimension when
# extract_non_primary=false): count toward scenario match / GT pair rates,
# but skip field-level cosine / BERTScore.
EXCLUDE_FROM_FIELD_METRICS_COL = "EXCLUDE_FROM_FIELD_METRICS"
EXCLUSION_REASON_COL = "EXCLUSION_REASON"
DOWNLOADED_COL = "METADATA_DOWNLOADED"
REVIEW_CREATED_COL = "METADATA_REVIEW CREATED"
GT_EXCLUSION_REASON_COL = "METADATA_EXCLUSION REASON"
REASON_LITERATURE_COMPARATOR = "literature comparator (not this paper's system)"
REASON_OWN_SYSTEM_MENTIONED = (
    "mentioned only (named/hinted; not fully modeled with inventory/results)"
)
REASON_SUBCATEGORY_BREAKDOWN = "subcategory breakdown (split inside one system)"
REASON_REVIEW_CREATED = "review-created (not in source paper)"
REASON_NOT_DOWNLOADED = "paper not downloaded"
REASON_REVIEW_COLLAPSED = "paper variant of a review-collapsed pathway"
REASON_NO_FIELD_EXTRACTION = "labeled shell (no field extraction)"
SCENARIO_FIELDS_EXTRACTED_COL = "SCENARIO_FIELDS_EXTRACTED"
ALWAYS_IGNORE_COLUMNS = {
    "PDF_PATH",
    # Synthetic join / identity columns — never score these (inflate agreement).
    PID_COL,
    PID_COL.upper(),
    "METADATA_PID",
    # Bibliographic citation key from the review (Author Year), not an LCI
    # extraction target. PID identifies the paper; scoring Reference inflates
    # top-field tables (near-perfect string match to the review label).
    "STUDY DETAILS_REFERENCE",
    "STUDY DETAILS_Reference",
    "REFERENCE",
    "Reference",
    "_scenario_match_text",
    "_SCENARIO_MATCH_TEXT",
    "SCENARIO_SCENARIO_CATEGORY",
    "SCENARIO_DEFINITION_ONLY",
    "SCENARIO_SCENARIO_DESCRIPTION",
    SCENARIO_MATCH_TEXT_AUTO_COL,
    SCENARIO_MATCH_TEXT_MANUAL_COL,
    SCENARIO_MATCH_TEXT_AUTO_COL.upper(),
    SCENARIO_MATCH_TEXT_MANUAL_COL.upper(),
    SCENARIO_AI_CONFIDENCE_COL,
    SCENARIO_AI_REASON_COL,
    SCENARIO_COSINE_PARTNER_COL,
    SCENARIO_AI_CONFIDENCE_COL.upper(),
    SCENARIO_AI_REASON_COL.upper(),
    SCENARIO_COSINE_PARTNER_COL.upper(),
    SCENARIO_COLLAPSED_INTO_COL,
    SCENARIO_REVIEW_COLLAPSED_COL,
    SCENARIO_PAPER_SCENARIO_COUNT_COL,
    SCENARIO_COLLAPSED_INTO_COL.upper(),
    SCENARIO_REVIEW_COLLAPSED_COL.upper(),
    SCENARIO_PAPER_SCENARIO_COUNT_COL.upper(),
    SCENARIO_TEXT_MENTIONED_COL,
    SCENARIO_TEXT_MENTION_SNIPPET_COL,
    SCENARIO_TEXT_MENTIONED_COL.upper(),
    SCENARIO_TEXT_MENTION_SNIPPET_COL.upper(),
    DOWNLOADED_COL,
    REVIEW_CREATED_COL,
    GT_EXCLUSION_REASON_COL,
    EXCLUDE_FROM_ACCURACY_COL,
    EXCLUDE_FROM_FIELD_METRICS_COL,
    EXCLUSION_REASON_COL,
    SCENARIO_FIELDS_EXTRACTED_COL,
    "SCENARIO_FIELDS_EXTRACTED",
}


def _is_scored_field_base(base_name: str) -> bool:
    """False for synthetic join/identity fields that must not enter accuracy stats."""
    if not base_name:
        return False
    return base_name.upper() not in ALWAYS_IGNORE_COLUMNS and base_name not in ALWAYS_IGNORE_COLUMNS

ds = DataSimilarity()
_number_commas = re.compile(r"(?<=\d),(?=\d{3}\b)")


@dataclass
class MetricConfig:
    """Configuration for a metric type."""

    name: str
    column_suffix: str
    display_name: str

    def get_column_name(self, base_field: str) -> str:
        """Get the column name for this metric and field."""
        return f"{base_field}_{MATCH_SUFFIX}_{self.column_suffix}"

    def get_columns_from_dataframe(self, df: pd.DataFrame) -> list[str]:
        """Get all columns for this metric from the dataframe."""
        return [
            c
            for c in df.columns
            if MATCH_SUFFIX in c and self.column_suffix in c.lower()
        ]


@dataclass
class StatisticsConfig:
    """Configuration for statistics calculation."""

    metrics: list[MetricConfig]
    stat_types: list[str]  # 'mean', 'min', 'max', 'count'

    def __post_init__(self):
        # Default metric configurations if not provided
        if not self.metrics:
            self.metrics = [
                MetricConfig("cosine", "cosine", "Cosine Similarity"),
                MetricConfig("fuzzy_cosine", "fuzzy_cosine", "Fuzzy Cosine Similarity"),
                MetricConfig(
                    "llm_meaning_equal", "llm_meaning_equal", "LLM Meaning Equal"
                ),
            ]
        if not self.stat_types:
            self.stat_types = ["mean", "min", "max"]


# Global statistics configuration
STATS_CONFIG = StatisticsConfig(
    metrics=[
        MetricConfig("cosine", "cosine", "Cosine Similarity"),
        MetricConfig("fuzzy_cosine", "fuzzy_cosine", "Fuzzy Cosine Similarity"),
        MetricConfig(
            "fuzzy_threshold_equal",
            "fuzzy_threshold_equal",
            "Fuzzy Cosine Threshold Equal",
        ),
        MetricConfig("llm_meaning_equal", "llm_meaning_equal", "LLM Meaning Equal"),
    ],
    stat_types=["mean", "min", "max"],
)


def _collect_metric_values(
    df_group: pd.DataFrame,
    metric_config: MetricConfig,
    base_field: str,
    auto_has_value: bool = True,
    manual_has_value: bool = True,
) -> tuple[pd.Series, pd.Series]:
    """
    Collect metric values for a specific field and metric.

    Returns:
        Tuple of (all_values, non_empty_values) Series
    """
    column_name = metric_config.get_column_name(base_field)

    if column_name not in df_group.columns:
        return pd.Series(dtype=float), pd.Series(dtype=float)

    # Get all valid values
    all_values = df_group[column_name].dropna().astype(float)
    all_values = all_values[(all_values >= 0) & (all_values <= 1)]

    # Get non-empty values (only if both auto and manual have values)
    if auto_has_value and manual_has_value:
        non_empty_values = all_values.copy()
    else:
        non_empty_values = pd.Series(dtype=float)

    return all_values, non_empty_values


def _generate_metric_stats_dict(
    metric_config: MetricConfig,
    all_values: pd.Series,
    non_empty_values: pd.Series,
    include_non_empty: bool = True,
) -> dict[str, Any]:
    """
    Generate statistics dictionary for a metric using configuration.

    Args:
        metric_config: Configuration for the metric
        all_values: All metric values
        non_empty_values: Non-empty metric values
        include_non_empty: Whether to include non-empty statistics

    Returns:
        Dictionary with metric statistics based on configuration
    """
    stats = {}

    # Generate statistics based on configuration
    for stat_type in STATS_CONFIG.stat_types:
        stat_key = f"{metric_config.display_name.replace(' ', '_')}_{stat_type}"

        if not all_values.empty:
            if stat_type == "mean":
                stats[stat_key] = float(all_values.mean())
            elif stat_type == "min":
                stats[stat_key] = float(all_values.min())
            elif stat_type == "max":
                stats[stat_key] = float(all_values.max())
        else:
            stats[stat_key] = None

    # Non-empty statistics if requested
    if include_non_empty:
        for stat_type in STATS_CONFIG.stat_types:
            stat_key = (
                f"{metric_config.display_name.replace(' ', '_')}_{stat_type}_NonEmpty"
            )

            if not non_empty_values.empty:
                if stat_type == "mean":
                    stats[stat_key] = float(non_empty_values.mean())
                elif stat_type == "min":
                    stats[stat_key] = float(non_empty_values.min())
                elif stat_type == "max":
                    stats[stat_key] = float(non_empty_values.max())
            else:
                stats[stat_key] = None

        # Add count for non-empty values
        count_key = f"{metric_config.display_name.replace(' ', '_')}_Count_NonEmpty"
        stats[count_key] = len(non_empty_values) if not non_empty_values.empty else 0

    return stats


def _calculate_empty_field_percentages(
    df_group: pd.DataFrame, manual_cols: list[str], base_field: str
) -> tuple[float, float, int]:
    """
    Calculate empty field percentages for a specific field.

    Returns:
        Tuple of (empty_manual_nonempty_auto_pct, empty_auto_nonempty_manual_pct, total_comparisons)
    """
    auto_col = (
        f"{base_field}_auto" if f"{base_field}_auto" in df_group.columns else base_field
    )
    manual_col = f"{base_field}_manual"

    if manual_col not in df_group.columns or auto_col not in df_group.columns:
        return 0.0, 0.0, 0

    empty_manual_nonempty_auto_count = 0
    empty_auto_nonempty_manual_count = 0
    total_comparisons = 0

    for _, row in df_group.iterrows():
        manual_val = row[manual_col]
        auto_val = row[auto_col]

        manual_is_empty = DataSimilarity._is_null(manual_val)
        auto_is_empty = DataSimilarity._is_null(auto_val)
        auto_has_value = not auto_is_empty
        manual_has_value = not manual_is_empty

        total_comparisons += 1

        if auto_has_value and manual_is_empty:
            empty_manual_nonempty_auto_count += 1
        elif auto_is_empty and manual_has_value:
            empty_auto_nonempty_manual_count += 1

    empty_manual_nonempty_auto_pct = (
        (empty_manual_nonempty_auto_count / total_comparisons * 100)
        if total_comparisons > 0
        else 0.0
    )
    empty_auto_nonempty_manual_pct = (
        (empty_auto_nonempty_manual_count / total_comparisons * 100)
        if total_comparisons > 0
        else 0.0
    )

    return (
        empty_manual_nonempty_auto_pct,
        empty_auto_nonempty_manual_pct,
        total_comparisons,
    )


def create_llm_meaning_judge(
    model: str = "gpt-5.4-mini-2026-03-17",
    api_key: str | None = None,
    temperature: float = 0.0,
    max_workers: int | None = None,
) -> Callable[[str, str], float | None]:
    """
    Create an LLM judge function that evaluates semantic meaning equality between two strings.

    The judge determines if two strings have the same meaning, even if they differ in specificity
    or wording. For example, "Simapro software v8.0.5" and "SimaPro" would be considered equal
    in meaning.

    Calls are cached and ``judge.judge_many(pairs)`` runs unique pairs concurrently
    (default 32 workers; override via ``max_workers`` or env ``LLM_JUDGE_WORKERS``).

    Args:
        model: LLM model identifier (default: dated gpt-5.4-mini snapshot)
        api_key: API key for the LLM provider (if None, uses environment variables)
        temperature: Temperature for LLM generation (default: 0.0 for deterministic output)
        max_workers: Parallel API workers for ``judge_many`` (default 32)

    Returns:
        A callable function that takes two strings and returns 1.0 if they have the same meaning,
        0.0 otherwise. Returns float in [0,1] format. Also exposes ``judge_many``.
    """
    if not LITELLM_AVAILABLE:
        raise ImportError(
            "litellm is required for LLM judge functionality. "
            "Install it with: pip install litellm"
        )

    # Required schema: output must be {"same_meaning": bool}
    BOOL_SCHEMA = {
        "type": "json_schema",
        "json_schema": {
            "name": "semantic_equivalence",
            "strict": True,
            "schema": {
                "type": "object",
                "properties": {"same_meaning": {"type": "boolean"}},
                "required": ["same_meaning"],
                "additionalProperties": False,
            },
        },
    }

    workers = max_workers
    if workers is None:
        env_w = os.environ.get("LLM_JUDGE_WORKERS")
        workers = int(env_w) if env_w and env_w.isdigit() else 32
    workers = max(1, int(workers))

    cache: dict[tuple[str, str], float | None] = {}
    cache_lock = threading.Lock()
    stats = {"calls": 0, "hits": 0, "errors": 0}
    stats_lock = threading.Lock()

    def _call_api(val1: str, val2: str) -> float | None:
        prompt = f"""You are a semantic equivalence judge. Determine if two values have the same meaning.

SAME MEANING (return True) when:
- Abbreviation vs full form: "CO2" and "carbon dioxide", "kg" and "kilogram"
- More vs less specific: "Simapro software v8.0.5" and "SimaPro"
- Values with vs without unit: "1" and "1 kg of FPCM" (same numeric value, unit added)
- Partially contained: "522" and "Grass grown: 522 ton DM/year" (one is embedded in the other)
- Partially contained: "Bulgaria" and "Bulgaria, 20 dairy farms" (location same, extra context)
- Partially contained: "Economic allocation" and "Economic" (one abbreviates or contains the other)
- Less vs more detailed: "Farm surveys" and "20 farms were surveyed and modelled" (same activity, different detail)

DIFFERENT MEANING (return False) when:
- Completely different things: "Apple" and "Orange"
- Different numeric values: "1.5" and "2.0"
- Different semantics: "Mass allocation" vs "Economic allocation"

Value 1: "{val1}"
Value 2: "{val2}"

Do these two values have the same meaning?"""

        max_attempts = int(os.environ.get("LLM_JUDGE_RETRIES", "6"))
        for attempt in range(max_attempts):
            try:
                response = completion(
                    model=model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=temperature,
                    api_key=api_key,
                    response_format=BOOL_SCHEMA,
                )

                content = response.choices[0].message.content
                if not content:
                    return None
                data = json.loads(content)
                return 1.0 if data.get("same_meaning", False) else 0.0
            except Exception as e:
                err = str(e).lower()
                is_rate = (
                    "ratelimit" in err
                    or "rate limit" in err
                    or "rate_limit" in err
                    or "429" in err
                )
                if is_rate and attempt + 1 < max_attempts:
                    # Prefer server-suggested wait; fall back to exponential backoff.
                    wait_s = 0.25 * (2**attempt)
                    m = re.search(r"try again in\s+([\d.]+)\s*(ms|s)", str(e), re.I)
                    if m:
                        wait_s = float(m.group(1))
                        if m.group(2).lower() == "ms":
                            wait_s /= 1000.0
                        wait_s = max(wait_s, 0.05) + 0.05
                    time.sleep(wait_s)
                    continue
                with stats_lock:
                    stats["errors"] += 1
                print(f"Warning: LLM judge error: {e}")
                return None
        return None

    def judge(val1: str, val2: str) -> float | None:
        """Judge if two values have the same meaning (cached)."""
        key = (str(val1), str(val2))
        with cache_lock:
            if key in cache:
                with stats_lock:
                    stats["hits"] += 1
                return cache[key]
        with stats_lock:
            stats["calls"] += 1
        result = _call_api(val1, val2)
        with cache_lock:
            cache[key] = result
        return result

    def judge_many(
        pairs: list[tuple[str, str]], *, max_workers: int | None = None
    ) -> dict[tuple[str, str], float | None]:
        """Judge unique pairs concurrently; return map (val1, val2) → score."""
        unique = list(dict.fromkeys(pairs))
        if not unique:
            return {}
        # Fill from cache first
        pending: list[tuple[str, str]] = []
        out: dict[tuple[str, str], float | None] = {}
        with cache_lock:
            for pair in unique:
                if pair in cache:
                    out[pair] = cache[pair]
                    with stats_lock:
                        stats["hits"] += 1
                else:
                    pending.append(pair)
        if not pending:
            return out

        n_workers = max(1, min(max_workers or workers, len(pending)))
        print(
            f"LLM judge: {len(pending)} unique pairs "
            f"({len(unique) - len(pending)} cache hits) with {n_workers} workers..."
        )
        done = 0
        with ThreadPoolExecutor(max_workers=n_workers) as executor:
            futures = {
                executor.submit(judge, a, b): (a, b) for a, b in pending
            }
            for fut in as_completed(futures):
                pair = futures[fut]
                try:
                    out[pair] = fut.result()
                except Exception as e:  # noqa: BLE001
                    print(f"Warning: LLM judge worker error: {e}")
                    out[pair] = None
                done += 1
                if done % 50 == 0 or done == len(pending):
                    print(f"  LLM judge progress: {done}/{len(pending)}")
        return out

    def seed(verdicts: dict[tuple[str, str], float]) -> int:
        """Pre-fill the cache with earlier verdicts; returns how many were added."""
        added = 0
        with cache_lock:
            for key, value in verdicts.items():
                if key not in cache:
                    cache[key] = value
                    added += 1
        return added

    judge.judge_many = judge_many  # type: ignore[attr-defined]
    judge.seed = seed  # type: ignore[attr-defined]
    judge.cache_stats = stats  # type: ignore[attr-defined]
    judge.max_workers = workers  # type: ignore[attr-defined]
    return judge


def _sanitize_for_excel(value):
    """
    Sanitize a value for Excel by removing illegal characters.

    Excel doesn't allow control characters (ASCII 0-31) except tab (9),
    newline (10), and carriage return (13).

    Args:
        value: Value to sanitize (can be any type)

    Returns:
        Sanitized value
    """
    if DataSimilarity._is_null(value):
        return value

    if not isinstance(value, str):
        return value

    # Remove control characters except tab, newline, and carriage return
    # Excel allows: tab (9), newline (10), carriage return (13)
    allowed_control_chars = {9, 10, 13}
    sanitized = "".join(
        char if ord(char) >= 32 or ord(char) in allowed_control_chars else " "
        for char in value
    )

    return sanitized


def _sanitize_dataframe_for_excel(df: pd.DataFrame) -> pd.DataFrame:
    """
    Sanitize all string values in a DataFrame for Excel export.

    Args:
        df: DataFrame to sanitize

    Returns:
        Sanitized DataFrame
    """
    if df.empty:
        return df

    df_sanitized = df.copy()

    # Apply sanitization to all string columns
    for col in df_sanitized.columns:
        df_col = df_sanitized[col]
        if df_col.dtype == "object":
            df_sanitized[col] = df_sanitized[col].apply(_sanitize_for_excel)

    return df_sanitized


def load_extracted_json(json_path: str) -> tuple[dict, list]:
    """
    Load extracted data from JSON file.

    Args:
        json_path: Path to the JSON file containing extracted data

    Returns:
        Tuple of (metadata dict, list of results with concepts)
    """
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    metadata = data.get("metadata", {})
    results = data.get("results", [])

    return metadata, results


def extract_scenario_name_from_concepts(concepts: list) -> str | None:
    """
    Extract scenario name from concepts data.

    Args:
        concepts: List of concept dictionaries

    Returns:
        Scenario name string if found, None otherwise
    """
    for concept_dict in concepts:
        if "Scenario" in concept_dict:
            scenario_data = concept_dict["Scenario"]
            if "Scenario Name" in scenario_data:
                return scenario_data["Scenario Name"]
    return None


def process_extracted_data(
    concepts: list, pid: str | None = None, scenario_name: str | None = None
) -> pd.DataFrame:
    """
    Process and flatten extracted concepts into a DataFrame.

    Args:
        concepts: List of concept dictionaries
        pid: Optional paper ID to add to the DataFrame
        scenario_name: Optional scenario name to add to the DataFrame

    Returns:
        DataFrame with flattened extracted data
    """
    # Convert concepts to the format expected by flatten_extracted
    extracted = []
    for concept_dict in concepts:
        for concept_name, concept_data in concept_dict.items():
            if isinstance(concept_data, dict):
                extracted.append({concept_name: concept_data})

    # Flatten the extracted data
    df = flatten_extracted(extracted, keep_last_key_only=False)

    # Normalize column names to uppercase
    df = df.rename(columns=lambda x: x.upper() if isinstance(x, str) else x)
    df = df.rename(columns=col_rename_dict)

    # Add pid if provided (overwrite if already exists)
    if pid:
        df[PID_COL] = pid

    # Add scenario name if provided and not already present
    # If it exists from flattened concepts, prefer the manually extracted one if provided
    if scenario_name:
        df[SCENARIO_NAME_COL] = scenario_name

    return df


def load_all_extracted_data(json_paths: list[str]) -> pd.DataFrame:
    """
    Load and combine all extracted JSON files into a single dataframe.

    Args:
        json_paths: List of paths to JSON files containing extracted data

    Returns:
        DataFrame with all extracted data combined, one row per scenario
    """
    df_auto_list = []

    for json_path in json_paths:
        print(f"Loading extracted data from: {json_path}")
        metadata, results = load_extracted_json(json_path)
        # Prefer sha256 hex stem of the extract file over metadata.pid. Fashion
        # extracts were renamed to DOI-hash PIDs while metadata.pid still held
        # publisher PDF stems, which silently broke GT joins (Matched collapsed).
        stem = Path(json_path).stem
        meta_pid = metadata.get("pid")
        if re.fullmatch(r"[0-9a-f]{64}", stem):
            pid = stem
        else:
            pid = meta_pid or stem
        pdf_path = metadata.get("pdf_path")

        # Process each result (scenario) separately
        for result in results:
            # concepts = result.get("concepts", [])
            concepts = result
            if not concepts:
                continue

            # Extract scenario name
            scenario_name = extract_scenario_name_from_concepts(concepts)

            # Process extracted data for this scenario
            df_scenario = process_extracted_data(concepts, pid, scenario_name)

            # Add PDF path if available
            if pdf_path:
                df_scenario["PDF_PATH"] = pdf_path

            df_auto_list.append(df_scenario)

    if not df_auto_list:
        print("Warning: No concepts found in any extracted data.")
        return pd.DataFrame()

    # Combine all scenarios into one dataframe
    return pd.concat(df_auto_list, ignore_index=True)


def join_all_json_results(
    output_folder: str, output_csv_path: str | None = None
) -> pd.DataFrame:
    """
    Join all JSON result files from a folder into a single CSV with one row per scenario.

    Args:
        output_folder: Folder containing JSON result files
        output_csv_path: Optional path to save the joined CSV.
                        If None, saves as 'joined_results.csv' in output_folder.

    Returns:
        DataFrame with all joined results
    """
    import os

    # Find all JSON files
    json_paths = [
        os.path.join(output_folder, f)
        for f in os.listdir(output_folder)
        if f.endswith(".json") and f != "joined_results.csv"
    ]

    if not json_paths:
        print("No JSON files found to join!")
        return pd.DataFrame()

    print(f"\nJoining {len(json_paths)} JSON files into CSV...")

    # Use the shared function to load all data
    df_joined = load_all_extracted_data(json_paths)

    if df_joined.empty:
        return df_joined

    # Save to CSV
    if output_csv_path is None:
        output_csv_path = os.path.join(output_folder, "joined_results.csv")

    df_joined.to_csv(output_csv_path, index=False)
    print(f"Joined {len(df_joined)} scenarios from {len(json_paths)} PDFs")
    print(f"Joined results saved to: {output_csv_path}")

    return df_joined


# Results a spreadsheet stores when a formula fails (e.g. =AVERAGE(#REF!)).
# Google Sheets exports them as text, so pandas returns the text.
EXCEL_ERROR_VALUES = {"#REF!", "#N/A", "#DIV/0!", "#VALUE!", "#NAME?", "#NUM!", "#NULL!"}


def _check_reference_formulas(xlsx_path: str, sheet_name) -> None:
    """Stop when a formula cell has no stored result.

    Python reads only the result a spreadsheet app saved with the file. A file
    written without results (some exports do this) would make every formula
    cell read as empty, silently dropping reference values.
    """
    formulas = openpyxl.load_workbook(xlsx_path, read_only=True)
    values = openpyxl.load_workbook(xlsx_path, read_only=True, data_only=True)
    name = sheet_name if isinstance(sheet_name, str) else formulas.sheetnames[sheet_name or 0]
    missing = []
    for row_f, row_v in zip(formulas[name].iter_rows(), values[name].iter_rows()):
        for cf, cv in zip(row_f, row_v):
            if isinstance(cf.value, str) and cf.value.startswith("=") and cv.value is None:
                missing.append(cf.coordinate)
    if missing:
        raise ValueError(
            f"{len(missing)} formula cell(s) in {xlsx_path} [{name}] have no stored "
            f"result, so they would read as empty: {', '.join(missing[:15])}"
            + (" ..." if len(missing) > 15 else "")
            + ". Open the file in Excel or Google Sheets and save/download it again."
        )


def _blank_excel_errors(df: pd.DataFrame, xlsx_path: str) -> pd.DataFrame:
    """Read failed-formula results (#REF!, #N/A, ...) as empty, and say where."""
    is_error = df.apply(
        lambda col: col.map(lambda v: isinstance(v, str) and v.strip() in EXCEL_ERROR_VALUES)
    )
    if is_error.values.any():
        hits = [
            f"row {i + 1}: {col}" for col in df.columns
            for i in np.flatnonzero(is_error[col].to_numpy())
        ]
        print(
            f"Warning: {len(hits)} Excel error value(s) in {xlsx_path} read as empty "
            f"({'; '.join(hits[:5])}{' ...' if len(hits) > 5 else ''})"
        )
        df = df.mask(is_error)
    return df


def _drop_formula_cells(
    xlsx_path: str, sheet_name, header_row: int | list[int], columns: list[str]
) -> io.BytesIO:
    """Copy of the workbook where formula cells in ``columns`` are empty.

    ``columns`` use the comparison's column names (header rows joined with
    "_", upper case, e.g. "MIDPOINT IMPACT CATEGORIES_GWP (KG CO2-EQ/...)").
    Used when a reference column mixes values typed from the paper with
    formulas that copy or derive values from other columns.
    """
    rows = header_row if isinstance(header_row, list) else [header_row]
    formulas = openpyxl.load_workbook(xlsx_path)
    values = openpyxl.load_workbook(xlsx_path, data_only=True)
    name = sheet_name if isinstance(sheet_name, str) else formulas.sheetnames[sheet_name or 0]
    wf, wv = formulas[name], values[name]
    labels, carry = {}, {}
    for col in range(1, wf.max_column + 1):
        parts = []
        for level, r in enumerate(rows):
            v = wf.cell(r + 1, col).value
            if v not in (None, "") or level == len(rows) - 1:
                carry[level] = str(v).strip() if v is not None else ""
            parts.append(carry.get(level, ""))
        labels["_".join(parts).upper()] = col
    unknown = [c for c in columns if c.upper() not in labels]
    if unknown:
        raise ValueError(f"typed-only columns not found in {xlsx_path} [{name}]: {unknown}")
    first_data_row = rows[-1] + 2
    for column in columns:
        col = labels[column.upper()]
        n = 0
        for r in range(first_data_row, wf.max_row + 1):
            v = wf.cell(r, col).value
            if isinstance(v, str) and v.startswith("="):
                wv.cell(r, col).value = None
                n += 1
        print(f"Reference rule: {column}: {n} formula cell(s) read as empty (typed values only)")
    buf = io.BytesIO()
    values.save(buf)
    buf.seek(0)
    return buf


def load_manual_data(
    xlsx_path: str,
    sheet_name=str,
    header_row: int | list[int] = 2,
    as_nested: bool = False,
    typed_only_columns: list[str] | None = None,
) -> tuple[pd.DataFrame, dict] | pd.DataFrame:
    """
    Load manual data from excel file.

    Args:
        csv_path: Path to the xlsx file containing manual data
        sheet_name =: Worksheet
        header_row: Row number to use as header (0-indexed).
        Headers will get concatenated.

    Returns:
        DataFrame with manual data
    """
    _check_reference_formulas(xlsx_path, sheet_name)
    source = (
        _drop_formula_cells(xlsx_path, sheet_name, header_row, typed_only_columns)
        if typed_only_columns
        else xlsx_path
    )
    df = pd.read_excel(source, sheet_name, header=header_row)
    df = _blank_excel_errors(df, xlsx_path)
    # keep MultiIndex columns to preserve nested structure and avoid collisions
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [tuple(str(c).upper() for c in col) for col in df.columns]
    else:
        df.columns = [str(c).upper() for c in df.columns]

    # Always build a flattened view to maintain backward compatibility
    df_flat = df.copy()
    df_flat.columns = [
        "_".join(col) if isinstance(col, tuple) else col for col in df_flat.columns
    ]

    # Apply renaming after flattening so flattened column names are properly renamed
    df_flat = df_flat.rename(columns=lambda x: col_rename_dict.get(x, x))

    if not as_nested:
        return df_flat

    def _row_to_nested(row) -> dict:
        nested: dict = {}
        for col, val in row.items():
            parts: Iterable[str]
            if isinstance(col, tuple):
                parts = list(col)
            else:
                parts = str(col).split("_")
            cursor = nested
            for i, part in enumerate(parts):
                part = part.strip()
                if i == len(parts) - 1:
                    cursor[part] = val
                else:
                    cursor = cursor.setdefault(part, {})
        return nested

    # Build nested dict keyed by PID then scenario to avoid column collisions
    # Use df_flat (after renaming) instead of df to ensure proper column names
    nested_by_pid: dict = {}
    for _, row in df_flat.iterrows():
        nested_row = _row_to_nested(row)
        pid = nested_row.get(PID_COL) or row.get(PID_COL)
        scenario = nested_row.get(SCENARIO_NAME_COL) or row.get(SCENARIO_NAME_COL)
        pid_key = str(pid) if pid is not None else f"row_{_}"
        scenario_key = (
            str(scenario)
            if scenario is not None and str(scenario).strip() != ""
            else "UNKNOWN_SCENARIO"
        )

        pid_bucket = nested_by_pid.setdefault(pid_key, {})
        pid_bucket[scenario_key] = nested_row

    return df_flat, nested_by_pid


def _truthy(val, default: bool = False) -> bool:
    """Coerce Excel/JSON flag cells to bool. Missing columns stay `default`."""
    if val is None:
        return default
    try:
        if pd.isna(val):
            return default
    except (TypeError, ValueError):
        pass
    if isinstance(val, bool):
        return val
    if isinstance(val, (int, float)) and not isinstance(val, bool):
        return bool(val)
    s = str(val).strip().lower()
    if s in {"true", "1", "yes", "y"}:
        return True
    if s in {"false", "0", "no", "n", ""}:
        return False
    return default


def _combine_reasons(*reasons: str) -> str:
    seen = []
    for reason in reasons:
        if reason and reason not in seen:
            seen.append(reason)
    return "; ".join(seen)


def _auto_exclusion(
    row: pd.Series,
    *,
    include_literature_comparators: bool = False,
) -> str:
    """Extracted scenarios that are not this paper's own fully-modeled system.

    When ``include_literature_comparators`` is True (test runs), mentioned /
    external / definition-only shells stay eligible for pairing.
    """
    if include_literature_comparators:
        return ""
    cat = str(row.get(SCENARIO_CATEGORY_COL) or "").strip().lower()
    if cat == "own_system_mentioned":
        return REASON_OWN_SYSTEM_MENTIONED
    if cat == "external_reference":
        return REASON_LITERATURE_COMPARATOR
    if cat == "subcategory_breakdown":
        return REASON_SUBCATEGORY_BREAKDOWN
    if _truthy(row.get(SCENARIO_DEFINITION_ONLY_COL)):
        return REASON_LITERATURE_COMPARATOR
    return ""


def _manual_exclusion(row: pd.Series) -> str:
    """GT rows the review invented, or papers we never downloaded."""
    reasons = []
    # Paper not downloaded: default True (downloaded) when the column is absent
    # so dairy / older ground-truth files are unchanged.
    if DOWNLOADED_COL in row.index and not _truthy(row.get(DOWNLOADED_COL), default=True):
        reasons.append(REASON_NOT_DOWNLOADED)
    if _truthy(row.get(REVIEW_CREATED_COL)):
        reasons.append(REASON_REVIEW_CREATED)
    stated = row.get(GT_EXCLUSION_REASON_COL)
    if stated is not None and not (isinstance(stated, float) and pd.isna(stated)):
        text = str(stated).strip()
        if text and text.lower() not in {"nan", "none"}:
            reasons.append(text)
    return _combine_reasons(*reasons)


def _first_row_for_name(df: pd.DataFrame, name) -> pd.Series | None:
    if df is None or df.empty or SCENARIO_NAME_COL not in df.columns:
        return None
    hit = df[df[SCENARIO_NAME_COL] == name]
    if hit.empty:
        return None
    return hit.iloc[0]


def _cell_text(val) -> str:
    if val is None:
        return ""
    try:
        if pd.isna(val):
            return ""
    except (TypeError, ValueError):
        pass
    s = str(val).strip()
    if not s or s.lower() in {"nan", "none", "null"}:
        return ""
    return s


def _scenario_match_text(row: pd.Series | None, mode: str) -> str:
    """Text used to pair scenarios. Names stay the join keys."""
    if row is None:
        return ""
    name = _cell_text(row.get(SCENARIO_NAME_COL))
    if mode != SCENARIO_MATCH_MODE_NAME_DESCRIPTION:
        return name
    parts = [name]
    desc = _cell_text(row.get(SCENARIO_DESCRIPTION_COL))
    if desc:
        parts.append(desc)
    else:
        # Review GT has no prose description; use coded identity slots
        # that are not already in the constructed name.
        name_l = name.lower()
        for col in _GT_IDENTITY_COLS:
            val = _cell_text(row.get(col))
            if val and val.lower() not in name_l:
                parts.append(val)
    return ". ".join(p for p in parts if p)


def _load_pdf_text_cached(pdf_path: str | None, cache: dict[str, str]) -> str:
    """Load PDF text once per path (best-effort; empty on failure)."""
    if not pdf_path:
        return ""
    key = str(pdf_path)
    if key in cache:
        return cache[key]
    text = ""
    try:
        from pathlib import Path as _Path

        from auto_lca.process.nlp.extract import TextExtractor

        p = _Path(key)
        if p.is_file():
            text = TextExtractor.extract_text_from_pdf(str(p)) or ""
    except Exception as e:  # noqa: BLE001
        print(f"⚠️ PDF text load failed for {pdf_path}: {e}")
        text = ""
    cache[key] = text
    return text


def _scenario_name_mentioned_in_text(
    name: str, text: str, *, snippet_radius: int = 60
) -> tuple[bool, str]:
    """Fuzzy regex: auto scenario name tokens with flexible whitespace in PDF text.

    Tries (1) full name, (2) head before a parenthetical gloss the extractor often
    invents, e.g. ``Pellets FR LV - NL C (Forest Residues …)`` → ``Pellets FR LV - NL C``.

    Returns (mentioned, snippet). Uses word boundaries so ``Farm 1`` does not
    match inside ``Farm 10``. Empty / very short names return False.
    """
    name = (name or "").strip()
    if not name or not text:
        return False, ""

    candidates = [name]
    # Strip extractor gloss in parentheses / brackets.
    head = re.split(r"\s*[\(\[]", name, maxsplit=1)[0].strip(" -–—\t")
    if head and head.lower() != name.lower():
        candidates.append(head)
    # Also try a compact alphanumeric code if present (e.g. FR LV, SW US).
    code = re.search(
        r"\b([A-Za-z]{2,}(?:\s+[A-Za-z]{1,3}){0,3}\s*[-–—]\s*[A-Za-z]{1,3}"
        r"(?:\s+[A-Za-z]{1,3})?)\b",
        head or name,
    )
    if code:
        candidates.append(code.group(1).strip())

    def _search(candidate: str) -> tuple[bool, str]:
        tokens = [t for t in re.split(r"\s+", candidate) if t]
        if not tokens:
            return False, ""
        if len(tokens) == 1 and len(tokens[0]) <= 2 and not tokens[0].isdigit():
            return False, ""
        pattern = r"\b" + r"\s+".join(re.escape(t) for t in tokens) + r"\b"
        try:
            m = re.search(pattern, text, flags=re.IGNORECASE)
        except re.error:
            return False, ""
        if not m:
            alt_tokens = [
                re.sub(r"[-_/]+", r"[-_/\\s]+", re.escape(t)) for t in tokens
            ]
            alt = r"\b" + r"\s+".join(alt_tokens) + r"\b"
            try:
                m = re.search(alt, text, flags=re.IGNORECASE)
            except re.error:
                return False, ""
        if not m:
            return False, ""
        start = max(0, m.start() - snippet_radius)
        end = min(len(text), m.end() + snippet_radius)
        snippet = re.sub(r"\s+", " ", text[start:end]).strip()
        if start > 0:
            snippet = "…" + snippet
        if end < len(text):
            snippet = snippet + "…"
        return True, snippet

    seen: set[str] = set()
    for cand in candidates:
        key = cand.lower()
        if not cand or key in seen:
            continue
        seen.add(key)
        hit, snip = _search(cand)
        if hit:
            return True, snip
    return False, ""


def _ai_reason_is_uninformative(reason: str | None) -> bool:
    s = (reason or "").strip().lower()
    return s in ("", "nan", "none", "null", "unmatched")


def _fuzzy_variants(text: str) -> list[str]:
    """Full string plus content words (same expansion as DocumentRanker fuzzy)."""
    words = text.split()
    return [text] + list(set(words) - stopwords)


def _encode_strings_once(
    values: list[str],
    ranker: DocumentRanker,
    *,
    batch_size: int = 256,
    include_fuzzy_variants: bool = True,
) -> dict[str, np.ndarray]:
    """Encode unique strings sequentially in large batches.

    Do NOT thread this: HuggingFace tokenizers set TOKENIZERS_PARALLELISM=false
    (see document_rank.py) and the shared SentenceTransformer is not safe under
    concurrent encode(). One big sequential encode is faster than 100× per-column
    encodes with a fake thread pool.
    """
    to_encode: set[str] = set()
    for v in values:
        if not v:
            continue
        to_encode.add(v)
        if include_fuzzy_variants:
            to_encode.update(_fuzzy_variants(v))
    ordered = sorted(to_encode)
    if not ordered:
        return {}
    print(
        f"Encoding {len(ordered)} unique strings "
        f"(batch_size={batch_size}, sequential — tokenizer not parallel-safe)..."
    )
    out: dict[str, np.ndarray] = {}
    for i in range(0, len(ordered), batch_size):
        batch = ordered[i : i + batch_size]
        embs = ranker.encode_documents(batch, normalize_embeddings=True)
        for val, emb in zip(batch, embs):
            out[val] = np.asarray(emb, dtype=np.float32)
        done = min(i + batch_size, len(ordered))
        if done % (batch_size * 4) == 0 or done == len(ordered):
            print(f"  encoded {done}/{len(ordered)}")
    return out


def _fuzzy_cosine_from_cache(
    str1: str, str2: str, embeddings_dict: dict[str, np.ndarray]
) -> float | None:
    """Same math as DocumentRanker.compute_fuzzy_cosine_similarity, cache-only."""
    a_words = str1.split()
    b_words = str2.split()
    if len(a_words) == 1 and len(b_words) == 1:
        return None
    variants_a = _fuzzy_variants(str1)
    variants_b = _fuzzy_variants(str2)
    try:
        embs_a = np.stack([embeddings_dict[v] for v in variants_a])
        embs_b = np.stack([embeddings_dict[v] for v in variants_b])
    except KeyError:
        # Missing variant — fall back to live encode (should be rare after global pass)
        return ds.ranker.compute_fuzzy_cosine_similarity(str1, str2)
    sims = np.dot(embs_a, embs_b.T)
    full_word = float(sims[0, 0])
    best = float(sims.max())
    return float(np.average([best, full_word]))


_AGGREGATE_NAME_RE = re.compile(
    r"\b(average|avg\.?|mean|median|aggregate|aggregated|overall)\b",
    re.IGNORECASE,
)
_UNIT_NAME_RE = re.compile(
    r"\b(farm|site|case|plant|facility|unit|scenario|farmstead)\s*#?\s*\d+\b",
    re.IGNORECASE,
)
_COLLAPSE_AI_RE = re.compile(
    r"(average of|mean of|avg of|aggregat|collaps|review row .+ average|"
    r"represents the average|primary representative of the average|"
    r"not represented as a distinct|variant of)",
    re.IGNORECASE,
)


def _is_aggregate_scenario_name(name: str | None) -> bool:
    s = (name or "").strip()
    return bool(s) and bool(_AGGREGATE_NAME_RE.search(s))


def _is_unit_scenario_name(name: str | None) -> bool:
    s = (name or "").strip()
    return bool(s) and bool(_UNIT_NAME_RE.search(s))


def _ai_reason_implies_collapse(reason: str | None) -> bool:
    s = (reason or "").strip()
    if not s or _ai_reason_is_uninformative(s):
        return False
    return bool(_COLLAPSE_AI_RE.search(s))


def _pick_aggregate_gt(
    manual_countable: list[str], preferred: str | None = None
) -> str | None:
    """Prefer an explicit Average/mean GT name; else None."""
    if preferred and preferred in manual_countable and _is_aggregate_scenario_name(
        preferred
    ):
        return preferred
    aggs = [n for n in manual_countable if _is_aggregate_scenario_name(n)]
    if len(aggs) == 1:
        return aggs[0]
    if preferred and preferred in aggs:
        return preferred
    return aggs[0] if aggs else None


def _infer_structural_collapses(
    *,
    auto_countable: list[str],
    manual_countable: list[str],
    scenario_mapping: dict,
    collapsed_by_auto: dict[str, tuple[str, str]],
    ai_meta: dict[str, tuple[float, str]],
    cosine_mapping: dict,
) -> None:
    """Mark unit-like autos as review-collapsed into an aggregate GT when appropriate.

    Mutates ``collapsed_by_auto``, ``scenario_mapping``, and ``ai_meta`` in place.
    Does not override an existing primary pair. Prefer exact-name unit GT rows
    when present (Farm 3 ↔ Farm 3); only collapse when the unit has no GT twin
    and an Average/mean row exists, or when the AI reason already describes
    collapse into that average.
    """
    manual_set = set(manual_countable)
    manual_l = {n.lower().strip(): n for n in manual_countable}
    paired_autos = {
        n for n, (m, _) in scenario_mapping.items() if m is not None
    } | set(collapsed_by_auto)

    for auto_name in auto_countable:
        if auto_name in paired_autos:
            continue
        # Exact / casefold name already in GT → leave unmatched for caller;
        # exact join should have paired; if not, still don't collapse.
        if auto_name.lower().strip() in manual_l:
            continue

        ai_conf, ai_reason = ai_meta.get(auto_name, (0.0, ""))
        cosine_partner, _ = cosine_mapping.get(auto_name, (None, 0.0))
        agg = _pick_aggregate_gt(manual_countable, preferred=cosine_partner)

        unit_like = _is_unit_scenario_name(auto_name)
        reason_collapse = _ai_reason_implies_collapse(ai_reason)
        partner_is_agg = bool(
            cosine_partner and _is_aggregate_scenario_name(str(cosine_partner))
        )

        if not agg:
            continue
        # Collapse when: unit farm/site with no GT twin + aggregate GT exists,
        # or AI prose already describes average/collapse, or cosine points at Average.
        if not (unit_like or reason_collapse or partner_is_agg):
            continue
        if unit_like and not reason_collapse and not partner_is_agg:
            # Only collapse unit→Average when there is no same-name GT unit
            # and at least one aggregate GT row (typical review coding).
            if not any(_is_aggregate_scenario_name(n) for n in manual_countable):
                continue

        why = (ai_reason or "").strip() or (
            f"structural: {auto_name!r} is a paper unit of aggregate GT {agg!r}"
        )
        collapsed_by_auto[auto_name] = (agg, why)
        scenario_mapping[auto_name] = (agg, 0.0)
        ai_meta[auto_name] = (float(ai_conf or 0.0), why)
        paired_autos.add(auto_name)


def _pair_similarity(
    auto_text: str,
    manual_text: str,
    ranker: DocumentRanker,
    use_fuzzy: bool,
) -> float:
    a = str(auto_text or "")
    b = str(manual_text or "")
    if use_fuzzy:
        try:
            similarity = ranker.compute_fuzzy_cosine_similarity(a, b)
            if similarity is None:
                similarity = ranker.compute_cosine_similarity(a, b)
            return similarity
        except Exception:
            return ranker.compute_cosine_similarity(a, b)
    return ranker.compute_cosine_similarity(a, b)


def _annotate_exclusion(combined: dict, auto_reason: str, manual_reason: str) -> dict:
    reason = _combine_reasons(auto_reason, manual_reason)
    combined[EXCLUDE_FROM_ACCURACY_COL] = bool(reason)
    combined[EXCLUSION_REASON_COL] = reason or None
    combined[EXCLUDE_FROM_FIELD_METRICS_COL] = _exclude_from_field_metrics(combined)
    return combined


def _exclude_from_field_metrics(row: pd.Series | dict) -> bool:
    """True for labeled shells kept for pairing but without LCI field payloads.

    Prefer explicit ``SCENARIO_FIELDS_EXTRACTED`` from the extract JSON.
    Fallback: ``alternate_dimension`` rows (kept when extract_non_primary
    is false) are treated as shells.
    """
    get = row.get if hasattr(row, "get") else lambda k, d=None: row[k] if k in row else d
    fe = get(SCENARIO_FIELDS_EXTRACTED_COL)
    if fe is None:
        fe = get("SCENARIO_FIELDS_EXTRACTED")
    if fe is not None and not (isinstance(fe, float) and pd.isna(fe)):
        # Explicit false → shell; explicit true → score fields even if
        # category is alternate_dimension (extract_non_primary=true).
        return not _truthy(fe, default=True)
    cat = str(get(SCENARIO_CATEGORY_COL) or "").strip().lower()
    return cat == "alternate_dimension"


def _field_score_mask(df_joined: pd.DataFrame) -> pd.Series:
    """Matched, accuracy-counted rows that also have extractable fields."""
    scenario_manual_col = f"{SCENARIO_NAME_COL}_manual"
    matched_mask = df_joined[SCENARIO_NAME_COL].notna() & (
        df_joined[SCENARIO_NAME_COL].astype(str).str.strip() != ""
    )
    if scenario_manual_col in df_joined.columns:
        matched_mask = (
            matched_mask
            & df_joined[scenario_manual_col].notna()
            & (df_joined[scenario_manual_col].astype(str).str.strip() != "")
        )
    if EXCLUDE_FROM_ACCURACY_COL in df_joined.columns:
        matched_mask = matched_mask & ~df_joined[EXCLUDE_FROM_ACCURACY_COL].map(
            lambda v: _truthy(v, default=False)
        )
    if EXCLUDE_FROM_FIELD_METRICS_COL in df_joined.columns:
        matched_mask = matched_mask & ~df_joined[EXCLUDE_FROM_FIELD_METRICS_COL].map(
            lambda v: _truthy(v, default=False)
        )
    elif SCENARIO_CATEGORY_COL in df_joined.columns:
        # Older joined sheets without the column: still skip alternate shells.
        matched_mask = matched_mask & ~df_joined.apply(
            _exclude_from_field_metrics, axis=1
        )
    return matched_mask


def _build_scenario_mapping(
    auto_scenarios: list,
    manual_scenarios: list,
    ranker: DocumentRanker,
    auto_texts: dict | None = None,
    manual_texts: dict | None = None,
    use_fuzzy: bool = True,
    out_matrix: dict | None = None,
) -> dict:
    """
    Build a 1-to-1 mapping of auto scenarios to manual scenarios based on similarity.

    Args:
        auto_scenarios: List of auto scenario names (join keys)
        manual_scenarios: List of manual scenario names (join keys)
        ranker: DocumentRanker instance for similarity computation
        auto_texts: Optional name -> match-text for auto scenarios
        manual_texts: Optional name -> match-text for manual scenarios
        use_fuzzy: Word-level fuzzy cosine (names). Off for long descriptions.

    Returns:
        Dictionary mapping auto_scenario -> (manual_scenario, similarity_score)
    """
    mapping = {}
    used_manual = set()
    auto_texts = auto_texts or {}
    manual_texts = manual_texts or {}

    # Compute similarity matrix
    similarity_matrix = {}
    for auto_scenario in auto_scenarios:
        auto_text = auto_texts.get(auto_scenario, str(auto_scenario))
        for manual_scenario in manual_scenarios:
            manual_text = manual_texts.get(manual_scenario, str(manual_scenario))
            similarity_matrix[(auto_scenario, manual_scenario)] = _pair_similarity(
                auto_text, manual_text, ranker, use_fuzzy=use_fuzzy
            )
    if out_matrix is not None:
        out_matrix.update(similarity_matrix)

    # Greedy matching: match highest similarity pairs first
    auto_list = list(auto_scenarios)
    manual_list = list(manual_scenarios)

    while auto_list and manual_list:
        best_sim = -1
        best_auto = None
        best_manual = None

        for auto_scenario in auto_list:
            for manual_scenario in manual_list:
                if manual_scenario in used_manual:
                    continue
                sim = similarity_matrix.get((auto_scenario, manual_scenario), 0.0)
                if sim > best_sim:
                    best_sim = sim
                    best_auto = auto_scenario
                    best_manual = manual_scenario

        if best_auto is not None and best_manual is not None:
            mapping[best_auto] = (best_manual, best_sim)
            auto_list.remove(best_auto)
            manual_list.remove(best_manual)
            used_manual.add(best_manual)
        else:
            break

    # Map remaining auto scenarios to None (unmatched)
    for auto_scenario in auto_list:
        mapping[auto_scenario] = (None, 0.0)

    return mapping


def _ai_scenario_mapping(
    df_auto_filtered: pd.DataFrame,
    df_manual_filtered: pd.DataFrame,
    auto_countable: list,
    manual_countable: list,
    cosine_mapping: dict,
    cosine_matrix: dict,
    scenario_mapper,
) -> tuple[dict, dict, dict]:
    """Replace cosine greedy pairing with an LLM 1-to-1 join.

    Returns (primary mapping, ai_meta, collapsed auto_name -> (gt_name, reason)).
    """
    from auto_lca.statistics.scenario_mapper import (
        card_from_auto_row,
        card_from_gt_row,
        cosine_hints_from_mapping,
    )

    extracted_cards = []
    for i, name in enumerate(auto_countable, 1):
        row = _first_row_for_name(df_auto_filtered, name)
        if row is None:
            continue
        extracted_cards.append(card_from_auto_row(row, f"E{i}"))
    gt_cards = []
    for i, name in enumerate(manual_countable, 1):
        row = _first_row_for_name(df_manual_filtered, name)
        if row is None:
            continue
        gt_cards.append(card_from_gt_row(row, f"G{i}"))

    hints = cosine_hints_from_mapping(extracted_cards, gt_cards, cosine_matrix)
    paper_map = scenario_mapper.map_paper(
        extracted_cards,
        gt_cards,
        cosine_hints=hints,
        cosine_fallback=cosine_mapping,
    )
    mapping = {name: (None, 0.0) for name in auto_countable}
    ai_meta: dict[str, tuple[float, str]] = {}
    collapsed: dict[str, tuple[str, str]] = {}
    for auto_name, (manual_name, conf, reason) in paper_map.pairs.items():
        mapping[auto_name] = (manual_name, float(conf or 0.0))
        ai_meta[auto_name] = (float(conf or 0.0), reason)
    for auto_name, (manual_name, reason) in paper_map.collapsed.items():
        if auto_name in paper_map.pairs:
            continue
        collapsed[auto_name] = (manual_name, reason)
        ai_meta[auto_name] = (0.0, reason)
    for auto_name, reason in paper_map.unmatched_extracted.items():
        if auto_name not in paper_map.pairs and auto_name not in collapsed:
            mapping[auto_name] = (None, 0.0)
            ai_meta[auto_name] = (0.0, reason)
    return mapping, ai_meta, collapsed


def _fuzzy_match_scenarios(
    df_auto: pd.DataFrame,
    df_manual: pd.DataFrame,
    ranker: DocumentRanker,
    pid: str,
    match_mode: str = SCENARIO_MATCH_MODE_NAME,
    scenario_mapper=None,
    include_literature_comparators: bool = False,
) -> pd.DataFrame:
    """
    Match scenarios using fuzzy cosine similarity with outer join.

    Literature comparators (extracted definition-only / external_reference)
    and review-created or not-downloaded GT rows stay on the joined sheet but
    are not used as pairing candidates and are flagged
    EXCLUDE_FROM_ACCURACY — unless ``include_literature_comparators`` is True
    (test runs that count mentioned/external shells in matching).
    """
    df_auto_filtered = df_auto[df_auto[PID_COL] == pid].copy()
    df_manual_filtered = df_manual[df_manual[PID_COL] == pid].copy()

    if df_auto_filtered.empty and df_manual_filtered.empty:
        return pd.DataFrame()

    auto_scenarios = (
        list(df_auto_filtered[SCENARIO_NAME_COL].dropna().unique())
        if not df_auto_filtered.empty and SCENARIO_NAME_COL in df_auto_filtered.columns
        else []
    )
    manual_scenarios = (
        list(df_manual_filtered[SCENARIO_NAME_COL].dropna().unique())
        if not df_manual_filtered.empty
        and SCENARIO_NAME_COL in df_manual_filtered.columns
        else []
    )

    auto_reason_by_name = {}
    for name in auto_scenarios:
        row = _first_row_for_name(df_auto_filtered, name)
        auto_reason_by_name[name] = (
            _auto_exclusion(
                row, include_literature_comparators=include_literature_comparators
            )
            if row is not None
            else ""
        )
    manual_reason_by_name = {}
    for name in manual_scenarios:
        row = _first_row_for_name(df_manual_filtered, name)
        manual_reason_by_name[name] = _manual_exclusion(row) if row is not None else ""
        # Paper-level "not downloaded" applies even when the scenario row
        # itself has no Review-created flag.
        if (
            not df_manual_filtered.empty
            and DOWNLOADED_COL in df_manual_filtered.columns
            and not _truthy(df_manual_filtered.iloc[0].get(DOWNLOADED_COL), default=True)
        ):
            manual_reason_by_name[name] = _combine_reasons(
                manual_reason_by_name[name], REASON_NOT_DOWNLOADED
            )

    auto_countable = [n for n in auto_scenarios if not auto_reason_by_name.get(n)]
    auto_excluded = [n for n in auto_scenarios if auto_reason_by_name.get(n)]
    manual_countable = [n for n in manual_scenarios if not manual_reason_by_name.get(n)]
    manual_excluded = [n for n in manual_scenarios if manual_reason_by_name.get(n)]

    auto_text_by_name = {
        n: _scenario_match_text(_first_row_for_name(df_auto_filtered, n), match_mode)
        for n in auto_scenarios
    }
    manual_text_by_name = {
        n: _scenario_match_text(_first_row_for_name(df_manual_filtered, n), match_mode)
        for n in manual_scenarios
    }
    # AI mode still needs a cosine baseline (hints + fallback + diagnostic).
    use_fuzzy = match_mode != SCENARIO_MATCH_MODE_NAME_DESCRIPTION
    cosine_matrix: dict = {}

    if auto_countable and manual_countable:
        cosine_mapping = _build_scenario_mapping(
            auto_countable,
            manual_countable,
            ranker,
            auto_texts=auto_text_by_name,
            manual_texts=manual_text_by_name,
            use_fuzzy=use_fuzzy,
            out_matrix=cosine_matrix,
        )
    else:
        cosine_mapping = {name: (None, 0.0) for name in auto_countable}

    ai_meta: dict[str, tuple[float, str]] = {}
    collapsed_by_auto: dict[str, tuple[str, str]] = {}
    if match_mode == SCENARIO_MATCH_MODE_AI and scenario_mapper is not None:
        try:
            scenario_mapping, ai_meta, collapsed_by_auto = _ai_scenario_mapping(
                df_auto_filtered,
                df_manual_filtered,
                auto_countable,
                manual_countable,
                cosine_mapping,
                cosine_matrix,
                scenario_mapper,
            )
        except Exception as e:  # noqa: BLE001
            print(f"⚠️ AI mapping failed for {pid}: {e}; using cosine greedy")
            scenario_mapping = cosine_mapping
            ai_meta = {
                name: (sim, "fallback_cosine")
                for name, (_, sim) in cosine_mapping.items()
            }
            collapsed_by_auto = {}
    else:
        scenario_mapping = cosine_mapping

    # Exact-name pairs the AI missed (e.g. Farm 3 ↔ Farm 3 after GT naming).
    paired_manuals = {
        m for m, _ in scenario_mapping.values() if m is not None
    } | {m for m, _ in collapsed_by_auto.values()}
    manual_by_l = {n.lower().strip(): n for n in manual_countable}
    for auto_name in auto_countable:
        cur, _ = scenario_mapping.get(auto_name, (None, 0.0))
        if cur is not None or auto_name in collapsed_by_auto:
            continue
        gt_exact = manual_by_l.get(auto_name.lower().strip())
        if not gt_exact or gt_exact in paired_manuals:
            continue
        scenario_mapping[auto_name] = (gt_exact, 1.0)
        ai_meta[auto_name] = (
            1.0,
            ai_meta.get(auto_name, (0.0, ""))[1] or "exact_name",
        )
        paired_manuals.add(gt_exact)

    _infer_structural_collapses(
        auto_countable=auto_countable,
        manual_countable=manual_countable,
        scenario_mapping=scenario_mapping,
        collapsed_by_auto=collapsed_by_auto,
        ai_meta=ai_meta,
        cosine_mapping=cosine_mapping,
    )

    # Collapsed autos link to a GT row for sheet layout but are excluded from
    # match-rate via EXCLUSION_REASON (review-collapsed).
    for auto_name, (gt_name, reason) in list(collapsed_by_auto.items()):
        if not gt_name:
            continue
        scenario_mapping[auto_name] = (gt_name, 0.0)
        auto_reason_by_name[auto_name] = _combine_reasons(
            auto_reason_by_name.get(auto_name, ""), REASON_REVIEW_COLLAPSED
        )
        prev = ai_meta.get(auto_name, (0.0, ""))
        ai_meta[auto_name] = (prev[0] if prev else 0.0, reason or "paper_variant")

    gt_paper_counts: dict[str, int] = {}
    for auto_name, (gt_name, _) in scenario_mapping.items():
        if gt_name:
            gt_paper_counts[gt_name] = gt_paper_counts.get(gt_name, 0) + 1

    empty_manual_cols = (
        list(df_manual_filtered.columns) if not df_manual_filtered.empty else []
    )
    empty_auto_cols = (
        list(df_auto_filtered.columns) if not df_auto_filtered.empty else []
    )

    joined_rows = []
    pdf_text_cache: dict[str, str] = {}
    # Prefer PDF_PATH from any auto row for this paper.
    paper_pdf = None
    if not df_auto_filtered.empty and "PDF_PATH" in df_auto_filtered.columns:
        for v in df_auto_filtered["PDF_PATH"].dropna().tolist():
            s = str(v).strip()
            if s and s.lower() not in ("nan", "none"):
                paper_pdf = s
                break
    paper_text = _load_pdf_text_cached(paper_pdf, pdf_text_cache)

    def emit_auto(auto_scenario, manual_scenario, similarity):
        auto_rows = df_auto_filtered[
            df_auto_filtered[SCENARIO_NAME_COL] == auto_scenario
        ]
        auto_reason = auto_reason_by_name.get(auto_scenario, "")
        if manual_scenario is not None:
            manual_rows = df_manual_filtered[
                df_manual_filtered[SCENARIO_NAME_COL] == manual_scenario
            ]
            manual_reason = manual_reason_by_name.get(manual_scenario, "")
        else:
            manual_rows = pd.DataFrame()
            manual_reason = ""
        for _, auto_row in auto_rows.iterrows():
            combined = auto_row.to_dict()
            cosine_partner, cosine_sim = cosine_mapping.get(auto_scenario, (None, 0.0))
            ai_conf, ai_reason = ai_meta.get(auto_scenario, (0.0, ""))
            pair_cosine = 0.0
            if manual_scenario is not None:
                pair_cosine = float(
                    cosine_matrix.get((auto_scenario, manual_scenario), 0.0) or 0.0
                )
            combined[SCENARIO_COSINE_PARTNER_COL] = cosine_partner
            combined[SCENARIO_AI_CONFIDENCE_COL] = ai_conf or None
            combined[SCENARIO_AI_REASON_COL] = ai_reason or None
            collapsed_into = (
                collapsed_by_auto[auto_scenario][0]
                if auto_scenario in collapsed_by_auto
                else None
            )
            count_key = manual_scenario or collapsed_into
            paper_count = gt_paper_counts.get(count_key, 1) if count_key else None
            combined[SCENARIO_COLLAPSED_INTO_COL] = collapsed_into
            combined[SCENARIO_REVIEW_COLLAPSED_COL] = bool(
                paper_count and paper_count > 1
            )
            combined[SCENARIO_PAPER_SCENARIO_COUNT_COL] = paper_count
            # Text-mention probe for unexplained unmatched autos.
            mentioned = None
            snippet = None
            if (
                manual_scenario is None
                and not auto_reason
                and _ai_reason_is_uninformative(ai_reason)
                and paper_text
            ):
                hit, snip = _scenario_name_mentioned_in_text(
                    str(auto_scenario), paper_text
                )
                mentioned = bool(hit)
                snippet = snip or None
            combined[SCENARIO_TEXT_MENTIONED_COL] = mentioned
            combined[SCENARIO_TEXT_MENTION_SNIPPET_COL] = snippet
            if not manual_rows.empty:
                for _, manual_row in manual_rows.iterrows():
                    for col in manual_row.index:
                        combined[f"{col}_manual"] = manual_row[col]
                    # Keep cosine of the chosen pair as the diagnostic
                    # similarity; AI confidence lives in its own column.
                    combined["_scenario_name_similarity_score"] = (
                        pair_cosine if ai_meta else similarity
                    )
                    combined[SCENARIO_MATCH_TEXT_AUTO_COL] = auto_text_by_name.get(
                        auto_scenario, ""
                    )
                    combined[SCENARIO_MATCH_TEXT_MANUAL_COL] = manual_text_by_name.get(
                        manual_scenario, ""
                    )
                    joined_rows.append(
                        _annotate_exclusion(combined, auto_reason, manual_reason)
                    )
                    break
            else:
                for col in empty_manual_cols:
                    combined[f"{col}_manual"] = None
                combined["_scenario_name_similarity_score"] = 0.0
                combined[SCENARIO_MATCH_TEXT_AUTO_COL] = auto_text_by_name.get(
                    auto_scenario, ""
                )
                combined[SCENARIO_MATCH_TEXT_MANUAL_COL] = ""
                joined_rows.append(
                    _annotate_exclusion(combined, auto_reason, manual_reason)
                )

    def emit_manual_only(manual_scenario):
        manual_rows = df_manual_filtered[
            df_manual_filtered[SCENARIO_NAME_COL] == manual_scenario
        ]
        manual_reason = manual_reason_by_name.get(manual_scenario, "")
        for _, manual_row in manual_rows.iterrows():
            combined = {col: None for col in empty_auto_cols}
            for col in manual_row.index:
                combined[f"{col}_manual"] = manual_row[col]
            if PID_COL not in empty_auto_cols or combined.get(PID_COL) is None:
                combined[PID_COL] = pid
            combined["_scenario_name_similarity_score"] = 0.0
            combined[SCENARIO_MATCH_TEXT_AUTO_COL] = ""
            combined[SCENARIO_MATCH_TEXT_MANUAL_COL] = manual_text_by_name.get(
                manual_scenario, ""
            )
            combined[SCENARIO_COSINE_PARTNER_COL] = None
            combined[SCENARIO_AI_CONFIDENCE_COL] = None
            combined[SCENARIO_AI_REASON_COL] = None
            paper_count = gt_paper_counts.get(manual_scenario)
            combined[SCENARIO_COLLAPSED_INTO_COL] = None
            combined[SCENARIO_REVIEW_COLLAPSED_COL] = bool(
                paper_count and paper_count > 1
            )
            combined[SCENARIO_PAPER_SCENARIO_COUNT_COL] = paper_count
            joined_rows.append(_annotate_exclusion(combined, "", manual_reason))

    for auto_scenario, (manual_scenario, similarity) in scenario_mapping.items():
        emit_auto(auto_scenario, manual_scenario, similarity)

    for auto_scenario in auto_excluded:
        emit_auto(auto_scenario, None, 0.0)

    matched_manual = {
        mapping[0] for mapping in scenario_mapping.values() if mapping[0] is not None
    }
    for manual_scenario in manual_countable:
        if manual_scenario not in matched_manual:
            emit_manual_only(manual_scenario)
    for manual_scenario in manual_excluded:
        emit_manual_only(manual_scenario)

    if not joined_rows:
        return pd.DataFrame()

    return pd.DataFrame(joined_rows)


def _merge_dataframes_fuzzy(
    df_auto: pd.DataFrame,
    df_manual: pd.DataFrame,
    ranker: DocumentRanker,
    max_workers: int | None = None,
    match_mode: str = SCENARIO_MATCH_MODE_NAME,
    scenario_mapper=None,
    include_literature_comparators: bool = False,
) -> pd.DataFrame:
    """Merge auto and manual dataframes using fuzzy scenario matching."""
    # Remove duplicate columns
    df_auto = df_auto.loc[:, ~df_auto.columns.duplicated()]

    # Check if PID exists in both
    auto_has_pid = PID_COL in df_auto.columns
    manual_has_pid = PID_COL in df_manual.columns

    if not auto_has_pid and not manual_has_pid:
        error_msg = "PID column not found in both dataframes.\n"
        raise ValueError(error_msg)
    elif not auto_has_pid or not manual_has_pid:
        error_msg = "PID column not found in one of the dataframes.\n"
        if not auto_has_pid:
            error_msg += f"Auto dataframe missing '{PID_COL}'. PID: {df_manual[PID_COL].unique()} Available columns: {list(df_auto.columns)}\n"
        if not manual_has_pid:
            error_msg += f"Manual dataframe missing '{PID_COL}'. PID: {df_auto[PID_COL].unique()} Available columns: {list(df_manual.columns)}\n"
        raise ValueError(error_msg)

    # Group by PID and match scenarios. Include not-downloaded GT papers
    # (Downloaded=False) even when this output folder has no extraction for
    # them; other manual-only PIDs stay out so a 10-paper sample is not
    # scored against the rest of the corpus.
    all_joined = []
    auto_pids = set(df_auto[PID_COL].dropna().unique())
    not_downloaded_pids = set()
    if DOWNLOADED_COL in df_manual.columns:
        not_downloaded_mask = ~df_manual[DOWNLOADED_COL].map(lambda v: _truthy(v, default=True))
        not_downloaded_pids = set(df_manual.loc[not_downloaded_mask, PID_COL].dropna().unique())
    pids = list(auto_pids | not_downloaded_pids)

    def _match_for_pid(pid_val):
        return _fuzzy_match_scenarios(
            df_auto,
            df_manual,
            ranker,
            pid_val,
            match_mode=match_mode,
            scenario_mapper=scenario_mapper,
            include_literature_comparators=include_literature_comparators,
        )

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        for matched in executor.map(_match_for_pid, pids):
            if not matched.empty:
                all_joined.append(matched)

    if not all_joined:
        return pd.DataFrame()

    # Filter out empty DataFrames to avoid FutureWarning
    non_empty = [df for df in all_joined if not df.empty]
    if not non_empty:
        return pd.DataFrame()

    # Ensure all DataFrames have the same columns to avoid FutureWarning
    if len(non_empty) > 1:
        # Get all unique columns from all DataFrames
        all_columns = set()
        for df in non_empty:
            all_columns.update(df.columns)

        # Reindex each DataFrame to have all columns (fills missing with NaN)
        non_empty = [df.reindex(columns=sorted(all_columns)) for df in non_empty]

    return pd.concat(non_empty, ignore_index=True, sort=False)


def _parse_numeric_value(val) -> float | None:
    """Parse a cell to float, stripping unit / note suffixes (see numeric_parse)."""
    from auto_lca.statistics.numeric_parse import parse_numeric_value

    return parse_numeric_value(val)


def _parse_numeric_value_strict(val) -> float | None:
    """Bare float() parse only — no unit stripping."""
    if isinstance(val, (pd.Series, np.ndarray)):
        if len(val) == 0:
            return None
        val = val.iloc[0] if isinstance(val, pd.Series) else val.flat[0]

    if ds._is_null(val):
        return None

    if isinstance(val, (int, float)):
        return float(val)
    try:
        return float(str(val).strip())
    except (ValueError, TypeError):
        return None


def _compute_metrics_fast(
    val1,
    val2,
    embeddings_dict: dict,
    llm_judge: Callable[[str, str], float | bool] | None = None,
    *,
    defer_llm: bool = False,
) -> dict:
    """Fast metric computation using pre-computed embeddings.

    Numeric fields use binary ``numeric_exact`` plus ``numeric_outcome``
    labels (unit_format_only / unit_convertible / value_mismatch /
    review_harmonized_diff / …). When both sides parse as scalars, cosine
    follows equality (1/0) — not relative-error.

    If ``defer_llm`` is True and an LLM call would be needed, sets
    ``llm_meaning_equal`` to None and ``_llm_pair`` to ``(str1, str2)`` so the
    caller can batch/parallelize judge calls.
    """
    from auto_lca.statistics.numeric_parse import classify_numeric_pair

    metrics: dict[str, float | None | tuple | str] = {}

    if ds._is_null(val1) and ds._is_null(val2):
        return {
            "cosine": 1.0,
            "fuzzy_cosine": 1.0,
            "clean_string_equal": 1.0,
            "numeric_exact": None,
            "numeric_distance": None,
            "numeric_outcome": "not_numeric",
            "llm_meaning_equal": 1.0 if llm_judge else None,
            "fuzzy_threshold_equal": 1.0,
        }

    if ds._is_null(val1) or ds._is_null(val2):
        return {
            "cosine": 0.0,
            "fuzzy_cosine": 0.0,
            "clean_string_equal": 0.0,
            "numeric_exact": None,
            "numeric_distance": None,
            "numeric_outcome": "not_numeric",
            "llm_meaning_equal": 0.0 if llm_judge else None,
            "fuzzy_threshold_equal": 0.0,
        }

    str1 = ds._safe_str(val1)
    str2 = ds._safe_str(val2)

    num_info = classify_numeric_pair(val1, val2)
    both_numeric = (
        num_info.get("numeric_exact") is not None
        and num_info.get("auto_value") is not None
        and num_info.get("manual_value") is not None
    )

    if both_numeric:
        cosine_sim = 1.0 if num_info.get("numeric_exact") == 1.0 else 0.0
    else:
        clean_str1 = str1.strip().lower()
        clean_str2 = str2.strip().lower()
        if clean_str1 == clean_str2 and clean_str1:
            cosine_sim = 1.0
        else:
            lookup_str1 = (
                str1
                if str1 in embeddings_dict
                else (clean_str1 if clean_str1 in embeddings_dict else str1)
            )
            lookup_str2 = (
                str2
                if str2 in embeddings_dict
                else (clean_str2 if clean_str2 in embeddings_dict else str2)
            )
            if lookup_str1 in embeddings_dict and lookup_str2 in embeddings_dict:
                emb1 = embeddings_dict[lookup_str1]
                emb2 = embeddings_dict[lookup_str2]
                cosine_sim = float(np.dot(emb1, emb2))
                if lookup_str1 == lookup_str2:
                    cosine_sim = 1.0
            else:
                cosine_sim = ds.ranker.compute_cosine_similarity(str1, str2)

    metrics["cosine"] = cosine_sim

    if both_numeric:
        fuzzy_sim = cosine_sim
    elif cosine_sim == 1.0 or str1.lower() == str2.lower():
        fuzzy_sim = 1.0
    else:
        fuzzy_sim = _fuzzy_cosine_from_cache(str1, str2, embeddings_dict)
        if fuzzy_sim is None:
            fuzzy_sim = cosine_sim

    metrics["fuzzy_cosine"] = fuzzy_sim
    metrics["fuzzy_threshold_equal"] = 1.0 if fuzzy_sim >= 0.7 else 0.0

    clean_str1 = str1.lower().strip()
    clean_str2 = str2.lower().strip()
    metrics["clean_string_equal"] = (
        1.0 if clean_str1 == clean_str2 and clean_str1 else 0.0
    )

    metrics["numeric_exact"] = num_info.get("numeric_exact")
    metrics["numeric_distance"] = num_info.get("numeric_distance")
    metrics["numeric_outcome"] = num_info.get("outcome")

    if llm_judge:
        both_non_null = not ds._is_null(val1) and not ds._is_null(val2)
        cosine_below_threshold = cosine_sim is not None and cosine_sim < 0.95
        if both_non_null and cosine_below_threshold:
            if defer_llm:
                metrics["llm_meaning_equal"] = None
                metrics["_llm_pair"] = (str1, str2)
            else:
                try:
                    judge_val = llm_judge(str1, str2)
                    if judge_val is None:
                        metrics["llm_meaning_equal"] = None
                    elif isinstance(judge_val, (int, float)):
                        metrics["llm_meaning_equal"] = float(judge_val)
                    elif isinstance(judge_val, bool):
                        metrics["llm_meaning_equal"] = 1.0 if judge_val else 0.0
                    else:
                        metrics["llm_meaning_equal"] = 1.0 if judge_val else 0.0
                except Exception:
                    metrics["llm_meaning_equal"] = None
        else:
            metrics["llm_meaning_equal"] = 1.0
    else:
        metrics["llm_meaning_equal"] = None

    return metrics



def _create_comparison_columns(
    df_auto: pd.DataFrame,
    joined: pd.DataFrame,
    join_keys: list[str],
    ignore_columns: list[str] | None = None,
    max_workers: int | None = None,
    llm_judge: Callable[[str, str], float | bool] | None = None,
) -> tuple[list[str], pd.DataFrame]:
    """Create comparison columns and return sorted column list."""
    ignore_set = {c.upper() for c in (ignore_columns or [])} | ALWAYS_IGNORE_COLUMNS
    sorted_cols = []
    # Add PID first
    if PID_COL in joined.columns:
        sorted_cols.append(PID_COL)

    # Add both scenario name columns (auto and manual) with similarity score
    if SCENARIO_NAME_COL in joined.columns:
        sorted_cols.append(SCENARIO_NAME_COL)
    scenario_manual_col = f"{SCENARIO_NAME_COL}_manual"
    if scenario_manual_col in joined.columns:
        sorted_cols.append(scenario_manual_col)
    # Add similarity score after scenario names
    if "_scenario_name_similarity_score" in joined.columns:
        sorted_cols.append("_scenario_name_similarity_score")
    if EXCLUDE_FROM_ACCURACY_COL in joined.columns:
        sorted_cols.append(EXCLUDE_FROM_ACCURACY_COL)
    if EXCLUDE_FROM_FIELD_METRICS_COL in joined.columns:
        sorted_cols.append(EXCLUDE_FROM_FIELD_METRICS_COL)
    if EXCLUSION_REASON_COL in joined.columns:
        sorted_cols.append(EXCLUSION_REASON_COL)
    if SCENARIO_MATCH_TEXT_AUTO_COL in joined.columns:
        sorted_cols.append(SCENARIO_MATCH_TEXT_AUTO_COL)
    if SCENARIO_MATCH_TEXT_MANUAL_COL in joined.columns:
        sorted_cols.append(SCENARIO_MATCH_TEXT_MANUAL_COL)
    if SCENARIO_AI_CONFIDENCE_COL in joined.columns:
        sorted_cols.append(SCENARIO_AI_CONFIDENCE_COL)
    if SCENARIO_AI_REASON_COL in joined.columns:
        sorted_cols.append(SCENARIO_AI_REASON_COL)
    if SCENARIO_COSINE_PARTNER_COL in joined.columns:
        sorted_cols.append(SCENARIO_COSINE_PARTNER_COL)
    if SCENARIO_COLLAPSED_INTO_COL in joined.columns:
        sorted_cols.append(SCENARIO_COLLAPSED_INTO_COL)
    if SCENARIO_REVIEW_COLLAPSED_COL in joined.columns:
        sorted_cols.append(SCENARIO_REVIEW_COLLAPSED_COL)
    if SCENARIO_PAPER_SCENARIO_COUNT_COL in joined.columns:
        sorted_cols.append(SCENARIO_PAPER_SCENARIO_COUNT_COL)

    metric_cols: list[str] = []
    column_pairs = []

    for col in df_auto.columns:
        # Skip join keys (we already added them)
        if col in join_keys:
            continue
        if col.upper() in ignore_set or not _is_scored_field_base(col):
            continue

        auto_col = f"{col}_auto" if f"{col}_auto" in joined.columns else col
        manual_col = f"{col}_manual" if f"{col}_manual" in joined.columns else None

        if auto_col not in joined.columns:
            continue

        sorted_cols.append(auto_col)
        if manual_col and manual_col in joined.columns:
            sorted_cols.append(manual_col)
            column_pairs.append((col, auto_col, manual_col))

    # Encode all unique strings once, then score columns via cache lookups.
    # (Per-column re-encode + threaded encode was both slow and tokenizer-unsafe.)
    print(
        f"Computing metrics for {len(column_pairs)} column pairs across {len(joined)} rows..."
    )

    # Encode ALL unique cell strings (+ fuzzy word variants) once. Do not
    # parallelize encode threads — TOKENIZERS_PARALLELISM=false / shared ST model.
    # Re-encoding per column was the 103× slowdown; shared cache removes it.
    all_unique: set[str] = set()
    for _, a_col, m_col in column_pairs:
        if a_col not in joined.columns or m_col not in joined.columns:
            continue
        for v in joined[a_col].tolist():
            if not ds._is_null(v):
                all_unique.add(ds._safe_str(v))
        for v in joined[m_col].tolist():
            if not ds._is_null(v):
                all_unique.add(ds._safe_str(v))
    embeddings_dict = _encode_strings_once(
        list(all_unique), ds.ranker, batch_size=256, include_fuzzy_variants=True
    )

    row_metric_dicts: list[dict] = [{} for _ in range(len(joined))]
    # Deferred LLM jobs: (row_list_idx, llm_col_name, str1, str2)
    pending_llm: list[tuple[int, str, str, str]] = []
    defer_llm = bool(llm_judge)
    row_indices = list(joined.index)

    # Precompute unmatched rows once (same for every column).
    unmatched_rows = [False] * len(joined)
    for list_idx, row_idx in enumerate(row_indices):
        is_unmatched = False
        if "_scenario_name_similarity_score" in joined.columns:
            sim_score = joined.loc[row_idx, "_scenario_name_similarity_score"]
            is_unmatched = pd.isna(sim_score) or sim_score == 0.0
        else:
            auto_scenario = (
                joined.loc[row_idx, SCENARIO_NAME_COL]
                if SCENARIO_NAME_COL in joined.columns
                else None
            )
            manual_scenario = (
                joined.loc[row_idx, f"{SCENARIO_NAME_COL}_manual"]
                if f"{SCENARIO_NAME_COL}_manual" in joined.columns
                else None
            )
            is_unmatched = DataSimilarity._is_null(
                auto_scenario
            ) or DataSimilarity._is_null(manual_scenario)
        unmatched_rows[list_idx] = is_unmatched

    c = 0
    for base_col, a_col, m_col in column_pairs:
        c += 1
        if c == 1 or c % 20 == 0 or c == len(column_pairs):
            print(f"Scoring column pair {c} of {len(column_pairs)} (cache lookups only)")
        if a_col not in joined.columns or m_col not in joined.columns:
            continue

        auto_vals = joined[a_col].tolist()
        manual_vals = joined[m_col].tolist()

        for list_idx, (val_a, val_m) in enumerate(zip(auto_vals, manual_vals)):
            if unmatched_rows[list_idx]:
                for metric_name in [
                    "cosine",
                    "fuzzy_cosine",
                    "clean_string_equal",
                    "numeric_exact",
                    "numeric_distance",
                    "numeric_outcome",
                    "llm_meaning_equal",
                ]:
                    col_name = f"{base_col}_{MATCH_SUFFIX}_{metric_name}"
                    row_metric_dicts[list_idx][col_name] = None
                    if col_name not in metric_cols:
                        metric_cols.append(col_name)
            else:
                metrics = _compute_metrics_fast(
                    val_a,
                    val_m,
                    embeddings_dict,
                    llm_judge=llm_judge,
                    defer_llm=defer_llm,
                )
                llm_pair = metrics.pop("_llm_pair", None)
                for metric_name, metric_val in metrics.items():
                    col_name = f"{base_col}_{MATCH_SUFFIX}_{metric_name}"
                    row_metric_dicts[list_idx][col_name] = metric_val
                    if col_name not in metric_cols:
                        metric_cols.append(col_name)
                if llm_pair is not None:
                    llm_col = f"{base_col}_{MATCH_SUFFIX}_llm_meaning_equal"
                    pending_llm.append(
                        (list_idx, llm_col, llm_pair[0], llm_pair[1])
                    )
                    if llm_col not in metric_cols:
                        metric_cols.append(llm_col)

    # Parallel LLM judge over unique deferred pairs (cross-column cache).
    if pending_llm and llm_judge is not None:
        pairs = [(a, b) for _, _, a, b in pending_llm]
        judge_many = getattr(llm_judge, "judge_many", None)
        # Prefer the judge's configured API concurrency over CPU count.
        llm_workers = (
            getattr(llm_judge, "max_workers", None) or max_workers or 32
        )
        if callable(judge_many):
            judged = judge_many(pairs, max_workers=llm_workers)
        else:
            # Fallback: still parallelize even if judge lacks judge_many
            unique = list(dict.fromkeys(pairs))
            judged = {}
            n_workers = max(1, min(int(llm_workers or 32), len(unique)))
            print(
                f"LLM judge: {len(unique)} unique pairs with {n_workers} workers..."
            )
            with ThreadPoolExecutor(max_workers=n_workers) as executor:
                fut_map = {
                    executor.submit(llm_judge, a, b): (a, b) for a, b in unique
                }
                done = 0
                for fut in as_completed(fut_map):
                    pair = fut_map[fut]
                    try:
                        judged[pair] = fut.result()
                    except Exception:
                        judged[pair] = None
                    done += 1
                    if done % 50 == 0 or done == len(unique):
                        print(f"  LLM judge progress: {done}/{len(unique)}")

        for list_idx, llm_col, a, b in pending_llm:
            val = judged.get((a, b))
            if val is None:
                row_metric_dicts[list_idx][llm_col] = None
            elif isinstance(val, (int, float)):
                row_metric_dicts[list_idx][llm_col] = float(val)
            elif isinstance(val, bool):
                row_metric_dicts[list_idx][llm_col] = 1.0 if val else 0.0
            else:
                row_metric_dicts[list_idx][llm_col] = 1.0 if val else 0.0
        stats = getattr(llm_judge, "cache_stats", None)
        if isinstance(stats, dict):
            print(
                f"LLM judge done: api_calls={stats.get('calls')} "
                f"cache_hits={stats.get('hits')} errors={stats.get('errors')}"
            )

    if row_metric_dicts:
        metrics_df = pd.DataFrame(row_metric_dicts)
        joined = pd.concat([joined.reset_index(drop=True), metrics_df], axis=1)

    # Reorder columns: for each datapoint, group auto, manual, and metrics together
    # Build new column order
    reordered_cols = []

    # Add PID and scenario columns first
    if PID_COL in joined.columns:
        reordered_cols.append(PID_COL)
    if SCENARIO_NAME_COL in joined.columns:
        reordered_cols.append(SCENARIO_NAME_COL)
    scenario_manual_col = f"{SCENARIO_NAME_COL}_manual"
    if scenario_manual_col in joined.columns:
        reordered_cols.append(scenario_manual_col)
    if "_scenario_name_similarity_score" in joined.columns:
        reordered_cols.append("_scenario_name_similarity_score")
    if EXCLUDE_FROM_ACCURACY_COL in joined.columns:
        reordered_cols.append(EXCLUDE_FROM_ACCURACY_COL)
    if EXCLUDE_FROM_FIELD_METRICS_COL in joined.columns:
        reordered_cols.append(EXCLUDE_FROM_FIELD_METRICS_COL)
    if EXCLUSION_REASON_COL in joined.columns:
        reordered_cols.append(EXCLUSION_REASON_COL)
    for base_col, a_col, m_col in column_pairs:
        if a_col in joined.columns:
            reordered_cols.append(a_col)
        if m_col in joined.columns:
            reordered_cols.append(m_col)

        # Add metrics in order: cosine, fuzzy_cosine, fuzzy_threshold_equal, llm_meaning_equal
        cosine_col = f"{base_col}_{MATCH_SUFFIX}_cosine"
        fuzzy_cosine_col = f"{base_col}_{MATCH_SUFFIX}_fuzzy_cosine"
        fuzzy_threshold_col = f"{base_col}_{MATCH_SUFFIX}_fuzzy_threshold_equal"
        llm_col = f"{base_col}_{MATCH_SUFFIX}_llm_meaning_equal"

        if cosine_col in joined.columns:
            reordered_cols.append(cosine_col)
        if fuzzy_cosine_col in joined.columns:
            reordered_cols.append(fuzzy_cosine_col)
        if fuzzy_threshold_col in joined.columns:
            reordered_cols.append(fuzzy_threshold_col)
        if llm_col in joined.columns:
            reordered_cols.append(llm_col)

    # Add any remaining columns that weren't in the reordered list
    remaining_cols = [col for col in joined.columns if col not in reordered_cols]
    reordered_cols.extend(remaining_cols)

    # Filter to only columns that exist in joined
    reordered_cols = [col for col in reordered_cols if col in joined.columns]

    return reordered_cols, joined[reordered_cols]


def _calculate_statistics(df_joined: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate statistics per paper-scenario combo.

    Returns:
        DataFrame with statistics for each PID-Scenario combination
    """
    if df_joined.empty:
        return pd.DataFrame()

    stats_rows = []

    # Group by PID and both scenario names (auto and manual)
    # Use a combination key since scenario names might differ
    scenario_manual_col = f"{SCENARIO_NAME_COL}_manual"

    # Create a unique key for grouping
    if scenario_manual_col in df_joined.columns:
        group_keys = [PID_COL, SCENARIO_NAME_COL, scenario_manual_col]
        df_joined[scenario_manual_col] = df_joined[scenario_manual_col].fillna("")
    else:
        group_keys = [PID_COL, SCENARIO_NAME_COL]

    # Group by the keys
    for key_tuple, group in df_joined.groupby(group_keys):
        if len(group_keys) == 3:
            pid, scenario_auto, scenario_manual = key_tuple
        else:
            pid = key_tuple[0]
            scenario_auto = key_tuple[1] if len(key_tuple) > 1 else None
            scenario_manual = None
        stats = {PID_COL: pid}
        stats[f"{SCENARIO_NAME_COL}_auto"] = scenario_auto
        if scenario_manual is not None:
            stats[f"{SCENARIO_NAME_COL}_manual"] = scenario_manual

        # Add similarity score if available
        similarity_score = None
        if "_scenario_name_similarity_score" in group.columns:
            similarity_score = (
                group["_scenario_name_similarity_score"].iloc[0]
                if len(group) > 0
                else None
            )
            stats["Scenario_Name_Similarity_Score"] = similarity_score

        # Check if scenario is unmatched (similarity score is 0 or None)
        is_unmatched = similarity_score is None or similarity_score == 0.0

        # Get match columns - only cosine similarity columns (not distance, not other metrics)
        match_cols = [
            col
            for col in group.columns
            if col.endswith(MATCH_SUFFIX)
            and ("cosine" in col.lower() or "match_ratio" in col.lower())
            and "distance" not in col.lower()
            and "fuzzy" not in col.lower()
        ]

        if match_cols:
            # Flatten all values, filter out None/NaN, and compute average
            all_values = []
            for col in match_cols:
                values = group[col].dropna().astype(float)
                # Filter to ensure values are between 0 and 1 (cosine similarity range)
                values = values[(values >= 0) & (values <= 1)]
                all_values.extend(values.tolist())

            if all_values:
                stats["Average_Match_Ratio"] = float(np.mean(all_values))
            else:
                stats["Average_Match_Ratio"] = None
        else:
            stats["Average_Match_Ratio"] = None

        # Find columns with manual but not auto
        manual_cols = [
            col
            for col in group.columns
            if col.endswith("_manual") and col != scenario_manual_col
        ]
        columns_manual_not_auto = []
        for manual_col in manual_cols:
            base_name = manual_col.replace("_manual", "")
            auto_col = (
                f"{base_name}_auto"
                if f"{base_name}_auto" in group.columns
                else base_name
            )
            if auto_col in group.columns:
                # Check if any row has manual value but auto is empty
                for idx, row in group.iterrows():
                    manual_val = row[manual_col]
                    auto_val = row[auto_col]
                    manual_has_value = not DataSimilarity._is_null(manual_val)
                    auto_has_value = not DataSimilarity._is_null(auto_val)

                    if manual_has_value and not auto_has_value:
                        if base_name not in columns_manual_not_auto:
                            columns_manual_not_auto.append(base_name)
                        break

        stats["Columns_Manual_Not_Auto_Count"] = len(columns_manual_not_auto)
        stats["Columns_Manual_Not_Auto_Names"] = (
            ", ".join(columns_manual_not_auto) if columns_manual_not_auto else ""
        )

        # Calculate % empty in manual and auto (percentage of cells that are empty)
        manual_empty_count = 0
        manual_total = 0
        auto_empty_count = 0
        auto_total = 0

        for manual_col in manual_cols:
            base_name = manual_col.replace("_manual", "")
            auto_col = (
                f"{base_name}_auto"
                if f"{base_name}_auto" in group.columns
                else base_name
            )
            if auto_col in group.columns:
                # Count empty values
                manual_empty = (
                    group[manual_col].isna()
                    | (group[manual_col].astype(str).str.strip() == "")
                ).sum()
                auto_empty = (
                    group[auto_col].isna()
                    | (group[auto_col].astype(str).str.strip() == "")
                ).sum()
                manual_total += len(group)
                auto_total += len(group)
                manual_empty_count += manual_empty
                auto_empty_count += auto_empty

        stats["Manual_Empty_Percent"] = (
            (manual_empty_count / manual_total * 100) if manual_total > 0 else 0.0
        )
        stats["Auto_Empty_Percent"] = (
            (auto_empty_count / auto_total * 100) if auto_total > 0 else 0.0
        )

        # Calculate % of columns where manual is empty but auto has value
        empty_manual_nonempty_auto_count = 0
        total_column_pairs = 0
        for manual_col in manual_cols:
            base_name = manual_col.replace("_manual", "")
            auto_col = (
                f"{base_name}_auto"
                if f"{base_name}_auto" in group.columns
                else base_name
            )
            if auto_col in group.columns:
                total_column_pairs += 1
                # Check if for this column pair, manual is empty but auto has value in any row
                for idx, row in group.iterrows():
                    manual_val = row[manual_col]
                    auto_val = row[auto_col]
                    manual_is_empty = DataSimilarity._is_null(manual_val)
                    auto_has_value = not DataSimilarity._is_null(auto_val)
                    if manual_is_empty and auto_has_value:
                        empty_manual_nonempty_auto_count += 1
                        break

        stats["Empty_Manual_NonEmpty_Auto_Pct"] = (
            (empty_manual_nonempty_auto_count / total_column_pairs * 100)
            if total_column_pairs > 0
            else 0.0
        )

        # Calculate % of columns where auto is empty but manual has value
        empty_auto_nonempty_manual_count = 0
        for manual_col in manual_cols:
            base_name = manual_col.replace("_manual", "")
            auto_col = (
                f"{base_name}_auto"
                if f"{base_name}_auto" in group.columns
                else base_name
            )
            if auto_col in group.columns:
                # Check if for this column pair, auto is empty but manual has value in any row
                for idx, row in group.iterrows():
                    manual_val = row[manual_col]
                    auto_val = row[auto_col]
                    auto_is_empty = DataSimilarity._is_null(auto_val)
                    manual_has_value = not DataSimilarity._is_null(manual_val)
                    if auto_is_empty and manual_has_value:
                        empty_auto_nonempty_manual_count += 1
                        break

        stats["Empty_Auto_NonEmpty_Manual_Pct"] = (
            (empty_auto_nonempty_manual_count / total_column_pairs * 100)
            if total_column_pairs > 0
            else 0.0
        )

        # If unmatched, set all stats to None (except basic identifiers and similarity score)
        if is_unmatched:
            stats["Average_Match_Ratio"] = None
            stats["Columns_Manual_Not_Auto_Count"] = None
            stats["Columns_Manual_Not_Auto_Names"] = None
            stats["Manual_Empty_Percent"] = None
            stats["Auto_Empty_Percent"] = None
            stats["Empty_Manual_NonEmpty_Auto_Pct"] = None
            stats["Empty_Auto_NonEmpty_Manual_Pct"] = None

        stats_rows.append(stats)

    return pd.DataFrame(stats_rows)


def _calculate_sensitivity_analysis(
    df_joined: pd.DataFrame,
    thresholds: list[float] = None,
) -> pd.DataFrame:
    """
    Calculate sensitivity analysis for fuzzy cosine thresholds.

    Args:
        df_joined: DataFrame with joined auto and manual data
        thresholds: List of thresholds to analyze (default: [0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95])

    Returns:
        DataFrame with sensitivity analysis results
    """
    if thresholds is None:
        thresholds = [0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]

    # Get all fuzzy cosine columns
    fuzzy_cosine_cols = [
        c
        for c in df_joined.columns
        if MATCH_SUFFIX in c and "fuzzy_cosine" in c.lower()
    ]

    if not fuzzy_cosine_cols:
        return pd.DataFrame(
            columns=["Threshold", "Overall_Pass_Rate", "Non_Empty_Pass_Rate"]
        )

    # Collect all fuzzy cosine values
    all_fuzzy_values = []
    non_empty_fuzzy_values = []

    for col in fuzzy_cosine_cols:
        # Get all values
        values = df_joined[col].dropna().astype(float)
        values = values[(values >= 0) & (values <= 1)]
        all_fuzzy_values.extend(values.tolist())

        # Get non-empty values (both auto and manual have values)
        base_name = col.replace(f"_{MATCH_SUFFIX}_fuzzy_cosine", "")
        auto_col = (
            f"{base_name}_auto"
            if f"{base_name}_auto" in df_joined.columns
            else base_name
        )
        manual_col = f"{base_name}_manual"

        if auto_col in df_joined.columns and manual_col in df_joined.columns:
            # Filter for non-empty values (parentheses required: != has higher precedence than &)
            auto_non_empty = df_joined[auto_col].notna() & (
                df_joined[auto_col].astype(str).str.strip() != ""
            )
            manual_non_empty = df_joined[manual_col].notna() & (
                df_joined[manual_col].astype(str).str.strip() != ""
            )
            non_empty_mask = auto_non_empty & manual_non_empty
            non_empty_values = df_joined.loc[non_empty_mask, col].dropna().astype(float)
            non_empty_values = non_empty_values[
                (non_empty_values >= 0) & (non_empty_values <= 1)
            ]
            non_empty_fuzzy_values.extend(non_empty_values.tolist())

    # Calculate pass rates for each threshold
    results = []
    for threshold in thresholds:
        if all_fuzzy_values:
            overall_pass_rate = (
                sum(1 for v in all_fuzzy_values if v >= threshold)
                / len(all_fuzzy_values)
                * 100
            )
        else:
            overall_pass_rate = 0.0

        if non_empty_fuzzy_values:
            non_empty_pass_rate = (
                sum(1 for v in non_empty_fuzzy_values if v >= threshold)
                / len(non_empty_fuzzy_values)
                * 100
            )
        else:
            non_empty_pass_rate = 0.0

        results.append(
            {
                "Threshold": threshold,
                "Overall_Pass_Rate": overall_pass_rate,
                "Non_Empty_Pass_Rate": non_empty_pass_rate,
            }
        )

    return pd.DataFrame(results)


BERTSCORE_NULL_PLACEHOLDER = "DUMMY"
_BERTSCORE_SKIP_BASES = {
    c.upper()
    for c in (
        *ALWAYS_IGNORE_COLUMNS,
        PID_COL,
        SCENARIO_NAME_COL,
        EXCLUDE_FROM_ACCURACY_COL,
        EXCLUDE_FROM_FIELD_METRICS_COL,
        EXCLUSION_REASON_COL,
        SCENARIO_FIELDS_EXTRACTED_COL,
    )
}


def _bertscore_empty_rows() -> list[dict]:
    return [
        {"Statistic": "BERTScore_Placeholder_Precision", "Value": None},
        {"Statistic": "BERTScore_Placeholder_Recall", "Value": None},
        {"Statistic": "BERTScore_Placeholder_F1", "Value": None},
        {"Statistic": "BERTScore_Placeholder_N_pairs", "Value": 0},
        {"Statistic": "BERTScore_NonEmpty_Precision", "Value": None},
        {"Statistic": "BERTScore_NonEmpty_Recall", "Value": None},
        {"Statistic": "BERTScore_NonEmpty_F1", "Value": None},
        {"Statistic": "BERTScore_NonEmpty_N_pairs", "Value": 0},
        {"Statistic": "BERTScore_hash", "Value": None},
    ]


def _is_bertscore_field(base_name: str) -> bool:
    if not base_name or base_name.startswith("_"):
        return False
    upper = base_name.upper()
    if upper in _BERTSCORE_SKIP_BASES or upper.startswith("METADATA"):
        return False
    return True


def _collect_bertscore_pairs(
    df_matched: pd.DataFrame,
) -> tuple[list[str], list[str], list[bool]]:
    """Collect auto/manual text pairs for BERTScore.

    Null cells are filled with DUMMY so one-sided empties still score as
    mismatches. The NonEmpty mask drops only pairs that are empty on both
    sides (DUMMY vs DUMMY would otherwise look like a perfect match).
    """
    cands: list[str] = []
    refs: list[str] = []
    nonempty_mask: list[bool] = []
    scenario_manual_col = f"{SCENARIO_NAME_COL}_manual"

    for manual_col in df_matched.columns:
        if not manual_col.endswith("_manual") or manual_col == scenario_manual_col:
            continue
        base_name = manual_col[: -len("_manual")]
        if not _is_bertscore_field(base_name):
            continue
        auto_col = (
            f"{base_name}_auto"
            if f"{base_name}_auto" in df_matched.columns
            else base_name
        )
        if auto_col not in df_matched.columns or auto_col == manual_col:
            continue

        auto_null = df_matched[auto_col].map(DataSimilarity._is_null)
        manual_null = df_matched[manual_col].map(DataSimilarity._is_null)
        auto_text = [
            BERTSCORE_NULL_PLACEHOLDER if is_null else DataSimilarity._safe_str(val)
            for val, is_null in zip(df_matched[auto_col].tolist(), auto_null.tolist())
        ]
        manual_text = [
            BERTSCORE_NULL_PLACEHOLDER if is_null else DataSimilarity._safe_str(val)
            for val, is_null in zip(
                df_matched[manual_col].tolist(), manual_null.tolist()
            )
        ]
        # Drop shared blanks; keep one-sided empties as failures.
        not_both_empty = (~auto_null | ~manual_null).tolist()
        cands.extend(auto_text)
        refs.extend(manual_text)
        nonempty_mask.extend(not_both_empty)

    return cands, refs, nonempty_mask


def _bertscore_mean_slice(tensor, mask: list[bool] | None = None) -> float | None:
    if tensor is None or len(tensor) == 0:
        return None
    if mask is None:
        return float(tensor.mean().item())
    selected = [score for score, keep in zip(tensor.tolist(), mask) if keep]
    if not selected:
        return None
    return float(sum(selected) / len(selected))


# bert-score's default for English, named so it can't change under us.
BERTSCORE_MODEL = "roberta-large"


def _calculate_bert_score(
    df_matched: pd.DataFrame,
    lang: str = "en",
) -> pd.DataFrame:
    """
    Compute BERTScore for auto vs manual field text pairs.

    Placeholder: every matched field cell; nulls replaced with "DUMMY".
    NonEmpty: drop cells that are empty on both sides. One-sided empties
    remain and score as mismatches (dummy vs real text).
    """
    empty_result = pd.DataFrame(
        columns=["Statistic", "Value"],
        data=_bertscore_empty_rows(),
    )
    if df_matched.empty:
        return empty_result
    if not BERT_SCORE_AVAILABLE:
        print("Warning: bert-score not installed. Run: pip install bert-score")
        return empty_result

    cands, refs, nonempty_mask = _collect_bertscore_pairs(df_matched)
    results = []
    hashname = None

    if cands and refs:
        try:
            (P, R, F), hashname = bert_score_score(
                cands,
                refs,
                lang=lang,
                model_type=BERTSCORE_MODEL,
                return_hash=True,
                verbose=False,
            )
            n_nonempty = int(sum(nonempty_mask))
            results.extend(
                [
                    {
                        "Statistic": "BERTScore_Placeholder_Precision",
                        "Value": _bertscore_mean_slice(P),
                    },
                    {
                        "Statistic": "BERTScore_Placeholder_Recall",
                        "Value": _bertscore_mean_slice(R),
                    },
                    {
                        "Statistic": "BERTScore_Placeholder_F1",
                        "Value": _bertscore_mean_slice(F),
                    },
                    {
                        "Statistic": "BERTScore_Placeholder_N_pairs",
                        "Value": len(cands),
                    },
                    {
                        "Statistic": "BERTScore_NonEmpty_Precision",
                        "Value": _bertscore_mean_slice(P, nonempty_mask),
                    },
                    {
                        "Statistic": "BERTScore_NonEmpty_Recall",
                        "Value": _bertscore_mean_slice(R, nonempty_mask),
                    },
                    {
                        "Statistic": "BERTScore_NonEmpty_F1",
                        "Value": _bertscore_mean_slice(F, nonempty_mask),
                    },
                    {
                        "Statistic": "BERTScore_NonEmpty_N_pairs",
                        "Value": n_nonempty,
                    },
                ]
            )
        except Exception as e:
            print(f"Warning: BERTScore failed: {e}")
            results.extend(row for row in _bertscore_empty_rows()[:-1])
    else:
        results.extend(_bertscore_empty_rows()[:-1])

    results.append({"Statistic": "BERTScore_hash", "Value": hashname})
    return pd.DataFrame(results)


def _calculate_comprehensive_statistics(
    df_joined: pd.DataFrame,
    scenario_similarity_threshold: float = None,
) -> tuple[
    pd.DataFrame,  # Statistics by scenario
    pd.DataFrame,  # Statistics by paper
    pd.DataFrame,  # Statistics by field and scenario
    pd.DataFrame,  # Statistics by field and paper
    pd.DataFrame,  # Statistics by field
    pd.DataFrame,  # Summary statistics
    pd.DataFrame,  # Sensitivity analysis
    pd.DataFrame,  # BERTScore (Precision, Recall, F1)
]:
    """
    Calculate comprehensive statistics at multiple levels with all required metrics.

    Args:
        df_joined: DataFrame with joined auto and manual data
        scenario_similarity_threshold: Optional threshold to filter scenarios by similarity score
                                    (only include scenarios with score >= threshold)

    Returns:
        Tuple of (stats_by_scenario, stats_by_paper, stats_by_field_scenario,
                stats_by_field_paper, stats_by_field, summary_stats, sensitivity_stats)
    """
    if df_joined.empty:
        return (
            pd.DataFrame(),
            pd.DataFrame(),
            pd.DataFrame(),
            pd.DataFrame(),
            pd.DataFrame(),
            pd.DataFrame(),
            pd.DataFrame(),
            _calculate_bert_score(pd.DataFrame()),
        )

    # Filter by scenario similarity threshold if provided
    if scenario_similarity_threshold is not None:
        scenario_similarity_col = f"{SCENARIO_NAME_COL}_similarity_score"
        if scenario_similarity_col in df_joined.columns:
            initial_count = len(df_joined)
            df_joined = df_joined[
                df_joined[scenario_similarity_col] >= scenario_similarity_threshold
            ]
            filtered_count = len(df_joined)
            print(
                f"Filtered scenarios by similarity >= {scenario_similarity_threshold}: {filtered_count}/{initial_count} scenarios retained"
            )
        else:
            print(
                f"Warning: Scenario similarity column '{scenario_similarity_col}' not found. Skipping filter."
            )

    # Filter to only matched scenarios with field payloads for cosine / BERT.
    matched_mask = _field_score_mask(df_joined)

    df_matched = df_joined.loc[matched_mask].copy()
    df_unmatched = df_joined.loc[~matched_mask].copy()

    scenario_manual_col = f"{SCENARIO_NAME_COL}_manual"
    # Get manual columns (excluding scenario name and synthetic join fields)
    manual_cols = [
        col
        for col in df_matched.columns
        if col.endswith("_manual")
        and col != scenario_manual_col
        and _is_scored_field_base(col[: -len("_manual")])
    ]

    # Initialize statistics containers
    stats_by_scenario = []
    stats_by_paper = []
    stats_by_field_scenario = []
    stats_by_field_paper = []
    stats_by_field = []

    # Collect overall statistics across all levels
    overall_metrics = {
        metric.name: pd.Series(dtype=float) for metric in STATS_CONFIG.metrics
    }
    overall_non_empty_metrics = {
        metric.name: pd.Series(dtype=float) for metric in STATS_CONFIG.metrics
    }
    overall_empty_manual_nonempty_auto_pct = []
    overall_empty_auto_nonempty_manual_pct = []

    # Track field-level statistics
    field_stats_dict = {}

    # Process data by paper, then scenario, then field
    for pid, paper_group in df_matched.groupby(PID_COL):
        # Paper-level metrics collection
        paper_metrics = {
            metric.name: pd.Series(dtype=float) for metric in STATS_CONFIG.metrics
        }
        paper_non_empty_metrics = {
            metric.name: pd.Series(dtype=float) for metric in STATS_CONFIG.metrics
        }
        paper_empty_manual_nonempty_auto_pct = []
        paper_empty_auto_nonempty_manual_pct = []

        # Process each scenario within the paper
        for scenario_key, scenario_group in paper_group.groupby(
            [SCENARIO_NAME_COL, scenario_manual_col]
        ):
            scenario_auto, scenario_manual = scenario_key

            # Scenario-level metrics collection
            scenario_metrics = {
                metric.name: pd.Series(dtype=float) for metric in STATS_CONFIG.metrics
            }
            scenario_non_empty_metrics = {
                metric.name: pd.Series(dtype=float) for metric in STATS_CONFIG.metrics
            }

            scenario_empty_manual_nonempty_auto_count = 0
            scenario_empty_auto_nonempty_manual_count = 0
            scenario_total_comparisons = 0

            # Process each field in this scenario
            for manual_col in manual_cols:
                base_name = manual_col.replace("_manual", "")
                auto_col = (
                    f"{base_name}_auto"
                    if f"{base_name}_auto" in scenario_group.columns
                    else base_name
                )

                if auto_col not in scenario_group.columns:
                    continue

                # Calculate empty field percentages for this field
                empty_manual_pct, empty_auto_pct, total_comparisons = (
                    _calculate_empty_field_percentages(
                        scenario_group, manual_cols, base_name
                    )
                )

                if total_comparisons > 0:
                    scenario_empty_manual_nonempty_auto_count += int(
                        empty_manual_pct * total_comparisons / 100
                    )
                    scenario_empty_auto_nonempty_manual_count += int(
                        empty_auto_pct * total_comparisons / 100
                    )
                    scenario_total_comparisons += total_comparisons

                # Collect field-scenario statistics for all metrics
                field_scenario_stats = {
                    PID_COL: pid,
                    f"{SCENARIO_NAME_COL}_auto": scenario_auto,
                    f"{SCENARIO_NAME_COL}_manual": scenario_manual,
                    "Field": base_name,
                    "Empty_Manual_NonEmpty_Auto_Pct": empty_manual_pct,
                    "Empty_Auto_NonEmpty_Manual_Pct": empty_auto_pct,
                }
                if "_scenario_name_similarity_score" in scenario_group.columns:
                    field_scenario_stats["_scenario_name_similarity_score"] = (
                        scenario_group["_scenario_name_similarity_score"].iloc[0]
                    )

                field_scenario_metrics = {
                    metric.name: pd.Series(dtype=float)
                    for metric in STATS_CONFIG.metrics
                }
                field_scenario_non_empty_metrics = {
                    metric.name: pd.Series(dtype=float)
                    for metric in STATS_CONFIG.metrics
                }

                # Collect actual auto and manual datapoints for this field-scenario
                auto_values_collected = []
                manual_values_collected = []

                # Process each row in this scenario
                for _, row in scenario_group.iterrows():
                    manual_val = row[manual_col]
                    auto_val = row[auto_col]

                    auto_values_collected.append(auto_val)
                    manual_values_collected.append(manual_val)

                    manual_is_empty = DataSimilarity._is_null(manual_val)
                    auto_is_empty = DataSimilarity._is_null(auto_val)
                    auto_has_value = not auto_is_empty
                    manual_has_value = not manual_is_empty

                    # Collect values for each metric
                    for metric_config in STATS_CONFIG.metrics:
                        column_name = metric_config.get_column_name(base_name)

                        if column_name in row and not pd.isna(row[column_name]):
                            val = float(row[column_name])
                            if 0 <= val <= 1:
                                # Add to all levels
                                scenario_metrics[metric_config.name] = pd.concat(
                                    [
                                        scenario_metrics[metric_config.name],
                                        pd.Series([val]),
                                    ]
                                )
                                paper_metrics[metric_config.name] = pd.concat(
                                    [
                                        paper_metrics[metric_config.name],
                                        pd.Series([val]),
                                    ]
                                )
                                overall_metrics[metric_config.name] = pd.concat(
                                    [
                                        overall_metrics[metric_config.name],
                                        pd.Series([val]),
                                    ]
                                )
                                field_scenario_metrics[metric_config.name] = pd.concat(
                                    [
                                        field_scenario_metrics[metric_config.name],
                                        pd.Series([val]),
                                    ]
                                )

                                # Add to non-empty if both values exist
                                if auto_has_value and manual_has_value:
                                    scenario_non_empty_metrics[metric_config.name] = (
                                        pd.concat(
                                            [
                                                scenario_non_empty_metrics[
                                                    metric_config.name
                                                ],
                                                pd.Series([val]),
                                            ]
                                        )
                                    )
                                    paper_non_empty_metrics[metric_config.name] = (
                                        pd.concat(
                                            [
                                                paper_non_empty_metrics[
                                                    metric_config.name
                                                ],
                                                pd.Series([val]),
                                            ]
                                        )
                                    )
                                    overall_non_empty_metrics[metric_config.name] = (
                                        pd.concat(
                                            [
                                                overall_non_empty_metrics[
                                                    metric_config.name
                                                ],
                                                pd.Series([val]),
                                            ]
                                        )
                                    )
                                    field_scenario_non_empty_metrics[
                                        metric_config.name
                                    ] = pd.concat(
                                        [
                                            field_scenario_non_empty_metrics[
                                                metric_config.name
                                            ],
                                            pd.Series([val]),
                                        ]
                                    )

                # Add actual auto and manual datapoints to field-scenario stats
                field_scenario_stats["Auto_Values"] = " | ".join(
                    str(v)
                    for v in auto_values_collected
                    if not DataSimilarity._is_null(v)
                )
                field_scenario_stats["Manual_Values"] = " | ".join(
                    str(v)
                    for v in manual_values_collected
                    if not DataSimilarity._is_null(v)
                )

                # Generate field-scenario statistics
                for metric_config in STATS_CONFIG.metrics:
                    all_values = field_scenario_metrics[metric_config.name]
                    non_empty_values = field_scenario_non_empty_metrics[
                        metric_config.name
                    ]

                    metric_stats = _generate_metric_stats_dict(
                        metric_config,
                        all_values,
                        non_empty_values,
                        include_non_empty=True,
                    )
                    field_scenario_stats.update(metric_stats)

                if total_comparisons > 0:
                    stats_by_field_scenario.append(field_scenario_stats)

                # Update field-level statistics
                if base_name not in field_stats_dict:
                    field_stats_dict[base_name] = {
                        f"{metric.name}_values": pd.Series(dtype=float)
                        for metric in STATS_CONFIG.metrics
                    }
                    field_stats_dict[base_name].update(
                        {
                            f"{metric.name}_non_empty_values": pd.Series(dtype=float)
                            for metric in STATS_CONFIG.metrics
                        }
                    )
                    field_stats_dict[base_name].update(
                        {
                            "empty_manual_nonempty_auto_count": 0,
                            "empty_auto_nonempty_manual_count": 0,
                            "total_comparisons": 0,
                        }
                    )

                # Add field values to field statistics
                for metric_config in STATS_CONFIG.metrics:
                    field_stats_dict[base_name][f"{metric_config.name}_values"] = (
                        pd.concat(
                            [
                                field_stats_dict[base_name][
                                    f"{metric_config.name}_values"
                                ],
                                field_scenario_metrics[metric_config.name],
                            ]
                        )
                    )
                    field_stats_dict[base_name][
                        f"{metric_config.name}_non_empty_values"
                    ] = pd.concat(
                        [
                            field_stats_dict[base_name][
                                f"{metric_config.name}_non_empty_values"
                            ],
                            field_scenario_non_empty_metrics[metric_config.name],
                        ]
                    )

                field_stats_dict[base_name]["empty_manual_nonempty_auto_count"] += int(
                    empty_manual_pct * total_comparisons / 100
                )
                field_stats_dict[base_name]["empty_auto_nonempty_manual_count"] += int(
                    empty_auto_pct * total_comparisons / 100
                )
                field_stats_dict[base_name]["total_comparisons"] += total_comparisons

            # Generate scenario-level statistics
            if scenario_total_comparisons > 0:
                scenario_stats = {
                    PID_COL: pid,
                    f"{SCENARIO_NAME_COL}_auto": scenario_auto,
                    f"{SCENARIO_NAME_COL}_manual": scenario_manual,
                    "Empty_Manual_NonEmpty_Auto_Pct": (
                        scenario_empty_manual_nonempty_auto_count
                        / scenario_total_comparisons
                        * 100
                    ),
                    "Empty_Auto_NonEmpty_Manual_Pct": (
                        scenario_empty_auto_nonempty_manual_count
                        / scenario_total_comparisons
                        * 100
                    ),
                }
                if "_scenario_name_similarity_score" in scenario_group.columns:
                    scenario_stats["_scenario_name_similarity_score"] = scenario_group[
                        "_scenario_name_similarity_score"
                    ].iloc[0]

                # Add metric statistics
                for metric_config in STATS_CONFIG.metrics:
                    all_values = scenario_metrics[metric_config.name]
                    non_empty_values = scenario_non_empty_metrics[metric_config.name]

                    metric_stats = _generate_metric_stats_dict(
                        metric_config,
                        all_values,
                        non_empty_values,
                        include_non_empty=True,
                    )
                    scenario_stats.update(metric_stats)

                stats_by_scenario.append(scenario_stats)

                # Add to paper-level percentages
                if scenario_total_comparisons > 0:
                    paper_empty_manual_nonempty_auto_pct.append(
                        scenario_empty_manual_nonempty_auto_count
                        / scenario_total_comparisons
                        * 100
                    )
                    paper_empty_auto_nonempty_manual_pct.append(
                        scenario_empty_auto_nonempty_manual_count
                        / scenario_total_comparisons
                        * 100
                    )

        # Generate paper-level statistics
        paper_stats = {PID_COL: pid}

        # Add metric statistics for paper
        for metric_config in STATS_CONFIG.metrics:
            all_values = paper_metrics[metric_config.name]
            non_empty_values = paper_non_empty_metrics[metric_config.name]

            metric_stats = _generate_metric_stats_dict(
                metric_config, all_values, non_empty_values, include_non_empty=True
            )
            paper_stats.update(metric_stats)

        # Add percentage statistics for paper
        if paper_empty_manual_nonempty_auto_pct:
            paper_stats.update(
                {
                    "Empty_Manual_NonEmpty_Auto_Pct_Mean": float(
                        np.mean(paper_empty_manual_nonempty_auto_pct)
                    ),
                    "Empty_Manual_NonEmpty_Auto_Pct_Min": float(
                        np.min(paper_empty_manual_nonempty_auto_pct)
                    ),
                    "Empty_Manual_NonEmpty_Auto_Pct_Max": float(
                        np.max(paper_empty_manual_nonempty_auto_pct)
                    ),
                }
            )
        else:
            paper_stats.update(
                {
                    "Empty_Manual_NonEmpty_Auto_Pct_Mean": 0.0,
                    "Empty_Manual_NonEmpty_Auto_Pct_Min": 0.0,
                    "Empty_Manual_NonEmpty_Auto_Pct_Max": 0.0,
                }
            )

        if paper_empty_auto_nonempty_manual_pct:
            paper_stats.update(
                {
                    "Empty_Auto_NonEmpty_Manual_Pct_Mean": float(
                        np.mean(paper_empty_auto_nonempty_manual_pct)
                    ),
                    "Empty_Auto_NonEmpty_Manual_Pct_Min": float(
                        np.min(paper_empty_auto_nonempty_manual_pct)
                    ),
                    "Empty_Auto_NonEmpty_Manual_Pct_Max": float(
                        np.max(paper_empty_auto_nonempty_manual_pct)
                    ),
                }
            )
        else:
            paper_stats.update(
                {
                    "Empty_Auto_NonEmpty_Manual_Pct_Mean": 0.0,
                    "Empty_Auto_NonEmpty_Manual_Pct_Min": 0.0,
                    "Empty_Auto_NonEmpty_Manual_Pct_Max": 0.0,
                }
            )

        stats_by_paper.append(paper_stats)

        # Add to overall percentages
        overall_empty_manual_nonempty_auto_pct.extend(
            paper_empty_manual_nonempty_auto_pct
        )
        overall_empty_auto_nonempty_manual_pct.extend(
            paper_empty_auto_nonempty_manual_pct
        )

    # Generate field-paper statistics by aggregating field-scenario stats
    field_scenario_df = pd.DataFrame(stats_by_field_scenario)
    if not field_scenario_df.empty:
        for pid, paper_field_group in field_scenario_df.groupby(PID_COL):
            for field_name, field_group in paper_field_group.groupby("Field"):
                field_paper_stats = {
                    PID_COL: pid,
                    "Field": field_name,
                }

                # Aggregate metric statistics across scenarios for this field and paper
                for metric_config in STATS_CONFIG.metrics:
                    mean_col = f"{metric_config.display_name.replace(' ', '_')}_mean"
                    min_col = f"{metric_config.display_name.replace(' ', '_')}_min"
                    max_col = f"{metric_config.display_name.replace(' ', '_')}_max"

                    if mean_col in field_group.columns:
                        field_paper_stats.update(
                            {
                                f"{metric_config.display_name.replace(' ', '_')}_mean": (
                                    float(field_group[mean_col].mean())
                                    if not field_group[mean_col].isna().all()
                                    else None
                                ),
                                f"{metric_config.display_name.replace(' ', '_')}_min": (
                                    float(field_group[min_col].min())
                                    if not field_group[min_col].isna().all()
                                    else None
                                ),
                                f"{metric_config.display_name.replace(' ', '_')}_max": (
                                    float(field_group[max_col].max())
                                    if not field_group[max_col].isna().all()
                                    else None
                                ),
                            }
                        )

                # Add percentage statistics
                if "Empty_Manual_NonEmpty_Auto_Pct" in field_group.columns:
                    field_paper_stats.update(
                        {
                            "Empty_Manual_NonEmpty_Auto_Pct_Mean": float(
                                field_group["Empty_Manual_NonEmpty_Auto_Pct"].mean()
                            ),
                            "Empty_Manual_NonEmpty_Auto_Pct_Min": float(
                                field_group["Empty_Manual_NonEmpty_Auto_Pct"].min()
                            ),
                            "Empty_Manual_NonEmpty_Auto_Pct_Max": float(
                                field_group["Empty_Manual_NonEmpty_Auto_Pct"].max()
                            ),
                        }
                    )

                if "Empty_Auto_NonEmpty_Manual_Pct" in field_group.columns:
                    field_paper_stats.update(
                        {
                            "Empty_Auto_NonEmpty_Manual_Pct_Mean": float(
                                field_group["Empty_Auto_NonEmpty_Manual_Pct"].mean()
                            ),
                            "Empty_Auto_NonEmpty_Manual_Pct_Min": float(
                                field_group["Empty_Auto_NonEmpty_Manual_Pct"].min()
                            ),
                            "Empty_Auto_NonEmpty_Manual_Pct_Max": float(
                                field_group["Empty_Auto_NonEmpty_Manual_Pct"].max()
                            ),
                        }
                    )

                stats_by_field_paper.append(field_paper_stats)

    # Generate field-level statistics
    for field_name, field_data in field_stats_dict.items():
        if field_data["total_comparisons"] > 0:
            field_stats = {"Field": field_name}

            # Add metric statistics for field
            for metric_config in STATS_CONFIG.metrics:
                all_values = field_data[f"{metric_config.name}_values"]
                non_empty_values = field_data[f"{metric_config.name}_non_empty_values"]

                metric_stats = _generate_metric_stats_dict(
                    metric_config, all_values, non_empty_values, include_non_empty=True
                )
                field_stats.update(metric_stats)

            # Add percentage statistics
            field_stats.update(
                {
                    "Empty_Manual_NonEmpty_Auto_Pct": (
                        field_data["empty_manual_nonempty_auto_count"]
                        / field_data["total_comparisons"]
                        * 100
                        if field_data["total_comparisons"] > 0
                        else 0.0
                    ),
                    "Empty_Auto_NonEmpty_Manual_Pct": (
                        field_data["empty_auto_nonempty_manual_count"]
                        / field_data["total_comparisons"]
                        * 100
                        if field_data["total_comparisons"] > 0
                        else 0.0
                    ),
                }
            )

            stats_by_field.append(field_stats)

    # Generate summary statistics
    matched_count = len(df_matched)
    unmatched_count = len(df_unmatched)
    total_generated_scenarios = matched_count + unmatched_count

    summary_stats = [
        {"Statistic": "Matched_Scenarios", "Value": matched_count},
        {"Statistic": "Unmatched_Scenarios", "Value": unmatched_count},
        {"Statistic": "Total_Generated_Scenarios", "Value": total_generated_scenarios},
    ]

    # Add overall metric statistics
    for metric_config in STATS_CONFIG.metrics:
        all_values = overall_metrics[metric_config.name]
        non_empty_values = overall_non_empty_metrics[metric_config.name]

        if not all_values.empty:
            summary_stats.extend(
                [
                    {
                        "Statistic": f"{metric_config.display_name.replace(' ', '_')}_Mean",
                        "Value": float(all_values.mean()),
                    },
                    {
                        "Statistic": f"{metric_config.display_name.replace(' ', '_')}_Min",
                        "Value": float(all_values.min()),
                    },
                    {
                        "Statistic": f"{metric_config.display_name.replace(' ', '_')}_Max",
                        "Value": float(all_values.max()),
                    },
                ]
            )
        else:
            summary_stats.extend(
                [
                    {
                        "Statistic": f"{metric_config.display_name.replace(' ', '_')}_Mean",
                        "Value": None,
                    },
                    {
                        "Statistic": f"{metric_config.display_name.replace(' ', '_')}_Min",
                        "Value": None,
                    },
                    {
                        "Statistic": f"{metric_config.display_name.replace(' ', '_')}_Max",
                        "Value": None,
                    },
                ]
            )

        if not non_empty_values.empty:
            summary_stats.extend(
                [
                    {
                        "Statistic": f"{metric_config.display_name.replace(' ', '_')}_Mean_NonEmpty",
                        "Value": float(non_empty_values.mean()),
                    },
                    {
                        "Statistic": f"{metric_config.display_name.replace(' ', '_')}_Min_NonEmpty",
                        "Value": float(non_empty_values.min()),
                    },
                    {
                        "Statistic": f"{metric_config.display_name.replace(' ', '_')}_Max_NonEmpty",
                        "Value": float(non_empty_values.max()),
                    },
                ]
            )
        else:
            summary_stats.extend(
                [
                    {
                        "Statistic": f"{metric_config.display_name.replace(' ', '_')}_Mean_NonEmpty",
                        "Value": None,
                    },
                    {
                        "Statistic": f"{metric_config.display_name.replace(' ', '_')}_Min_NonEmpty",
                        "Value": None,
                    },
                    {
                        "Statistic": f"{metric_config.display_name.replace(' ', '_')}_Max_NonEmpty",
                        "Value": None,
                    },
                ]
            )

    # Add overall percentage statistics
    if overall_empty_manual_nonempty_auto_pct:
        summary_stats.extend(
            [
                {
                    "Statistic": "Empty_Manual_NonEmpty_Auto_Pct_Mean",
                    "Value": float(np.mean(overall_empty_manual_nonempty_auto_pct)),
                },
                {
                    "Statistic": "Empty_Manual_NonEmpty_Auto_Pct_Min",
                    "Value": float(np.min(overall_empty_manual_nonempty_auto_pct)),
                },
                {
                    "Statistic": "Empty_Manual_NonEmpty_Auto_Pct_Max",
                    "Value": float(np.max(overall_empty_manual_nonempty_auto_pct)),
                },
            ]
        )
    else:
        summary_stats.extend(
            [
                {"Statistic": "Empty_Manual_NonEmpty_Auto_Pct_Mean", "Value": 0.0},
                {"Statistic": "Empty_Manual_NonEmpty_Auto_Pct_Min", "Value": 0.0},
                {"Statistic": "Empty_Manual_NonEmpty_Auto_Pct_Max", "Value": 0.0},
            ]
        )

    if overall_empty_auto_nonempty_manual_pct:
        summary_stats.extend(
            [
                {
                    "Statistic": "Empty_Auto_NonEmpty_Manual_Pct_Mean",
                    "Value": float(np.mean(overall_empty_auto_nonempty_manual_pct)),
                },
                {
                    "Statistic": "Empty_Auto_NonEmpty_Manual_Pct_Min",
                    "Value": float(np.min(overall_empty_auto_nonempty_manual_pct)),
                },
                {
                    "Statistic": "Empty_Auto_NonEmpty_Manual_Pct_Max",
                    "Value": float(np.max(overall_empty_auto_nonempty_manual_pct)),
                },
            ]
        )
    else:
        summary_stats.extend(
            [
                {"Statistic": "Empty_Auto_NonEmpty_Manual_Pct_Mean", "Value": 0.0},
                {"Statistic": "Empty_Auto_NonEmpty_Manual_Pct_Min", "Value": 0.0},
                {"Statistic": "Empty_Auto_NonEmpty_Manual_Pct_Max", "Value": 0.0},
            ]
        )

    # Calculate sensitivity analysis
    df_sensitivity = _calculate_sensitivity_analysis(df_joined)

    # Calculate BERTScore (Precision, Recall, F1) for auto vs manual text pairs
    df_f1 = _calculate_bert_score(df_matched, lang="en")

    return (
        pd.DataFrame(stats_by_scenario),
        pd.DataFrame(stats_by_paper),
        pd.DataFrame(stats_by_field_scenario),
        pd.DataFrame(stats_by_field_paper),
        pd.DataFrame(stats_by_field),
        pd.DataFrame(summary_stats),
        df_sensitivity,
        df_f1,
    )


def _calculate_overall_statistics(
    df_joined: pd.DataFrame,
    df_stats: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Build overall statistics across all papers and per data field.
    Only calculates stats for matched scenarios.
    Returns (overall_df, per_field_df)
    """
    if df_joined.empty:
        empty_stats = [
            {"Statistic": "Matched_Scenarios", "Value": 0},
            {"Statistic": "Unmatched_Scenarios", "Value": 0},
        ]
        return pd.DataFrame(empty_stats), pd.DataFrame()

    # Field metrics only for matched scenarios that have extracted fields.
    matched_mask = _field_score_mask(df_joined)

    unmatched_count = df_joined.loc[~matched_mask].shape[0]
    matched_count = df_joined.loc[matched_mask].shape[0]

    # Only calculate stats for matched scenarios
    df_matched = df_joined.loc[matched_mask].copy()

    if df_matched.empty:
        # Transpose empty stats too
        empty_stats = [
            {"Statistic": "Matched_Scenarios", "Value": matched_count},
            {"Statistic": "Unmatched_Scenarios", "Value": unmatched_count},
        ]
        return pd.DataFrame(empty_stats), pd.DataFrame()

    # Calculate statistics for each metric using configuration
    metric_values = {}
    for metric_config in STATS_CONFIG.metrics:
        # Get all columns for this metric
        metric_cols = metric_config.get_columns_from_dataframe(df_matched)

        # Collect all values for this metric
        values = pd.Series(dtype=float)
        for col in metric_cols:
            col_values = df_matched[col].dropna().astype(float)
            col_values = col_values[(col_values >= 0) & (col_values <= 1)]
            values = pd.concat([values, col_values])

        metric_values[metric_config.name] = values

    # Calculate empty manual, non-empty auto percentages from statistics
    # If df_stats is provided, use it; otherwise calculate from joined data
    if df_stats is not None and not df_stats.empty:
        # Filter to only matched scenarios (where similarity score is not 0 or None)
        if "Scenario_Name_Similarity_Score" in df_stats.columns:
            matched_stats = df_stats[
                (df_stats["Scenario_Name_Similarity_Score"].notna())
                & (df_stats["Scenario_Name_Similarity_Score"] != 0.0)
            ]
        else:
            matched_stats = df_stats

        # Get percentages from statistics dataframe (only matched scenarios)
        if "Empty_Manual_NonEmpty_Auto_Pct" in matched_stats.columns:
            empty_manual_nonempty_auto_pcts = (
                matched_stats["Empty_Manual_NonEmpty_Auto_Pct"]
                .dropna()
                .astype(float)
                .tolist()
            )
        else:
            empty_manual_nonempty_auto_pcts = []

        if "Empty_Auto_NonEmpty_Manual_Pct" in matched_stats.columns:
            empty_auto_nonempty_manual_pcts = (
                matched_stats["Empty_Auto_NonEmpty_Manual_Pct"]
                .dropna()
                .astype(float)
                .tolist()
            )
        else:
            empty_auto_nonempty_manual_pcts = []
    else:
        # Fallback: calculate from joined data (shouldn't happen in normal flow)
        empty_manual_nonempty_auto_pcts = []
        empty_auto_nonempty_manual_pcts = []

    # Build overall statistics using configuration
    overall_stats = {
        "Matched_Scenarios": matched_count,
        "Unmatched_Scenarios": unmatched_count,
    }

    # Add metric statistics
    for metric_config in STATS_CONFIG.metrics:
        values = metric_values[metric_config.name]
        if not values.empty:
            overall_stats.update(
                {
                    f"{metric_config.display_name.replace(' ', '_')}_Mean": float(
                        values.mean()
                    ),
                    f"{metric_config.display_name.replace(' ', '_')}_Min": float(
                        values.min()
                    ),
                    f"{metric_config.display_name.replace(' ', '_')}_Max": float(
                        values.max()
                    ),
                }
            )
        else:
            overall_stats.update(
                {
                    f"{metric_config.display_name.replace(' ', '_')}_Mean": None,
                    f"{metric_config.display_name.replace(' ', '_')}_Min": None,
                    f"{metric_config.display_name.replace(' ', '_')}_Max": None,
                }
            )

    # Add percentage statistics
    overall_stats.update(
        {
            "Empty_Manual_NonEmpty_Auto_Pct_Mean": (
                float(np.mean(empty_manual_nonempty_auto_pcts))
                if empty_manual_nonempty_auto_pcts
                else None
            ),
            "Empty_Manual_NonEmpty_Auto_Pct_Min": (
                float(np.min(empty_manual_nonempty_auto_pcts))
                if empty_manual_nonempty_auto_pcts
                else None
            ),
            "Empty_Manual_NonEmpty_Auto_Pct_Max": (
                float(np.max(empty_manual_nonempty_auto_pcts))
                if empty_manual_nonempty_auto_pcts
                else None
            ),
            "Empty_Auto_NonEmpty_Manual_Pct_Mean": (
                float(np.mean(empty_auto_nonempty_manual_pcts))
                if empty_auto_nonempty_manual_pcts
                else None
            ),
            "Empty_Auto_NonEmpty_Manual_Pct_Min": (
                float(np.min(empty_auto_nonempty_manual_pcts))
                if empty_auto_nonempty_manual_pcts
                else None
            ),
            "Empty_Auto_NonEmpty_Manual_Pct_Max": (
                float(np.max(empty_auto_nonempty_manual_pcts))
                if empty_auto_nonempty_manual_pcts
                else None
            ),
        }
    )

    # Per field stats - use configuration-driven approach
    field_stats = {}

    # Process each metric type
    for metric_config in STATS_CONFIG.metrics:
        metric_cols = metric_config.get_columns_from_dataframe(df_matched)

        for col in metric_cols:
            # Extract field name: remove metric suffix
            field_name = col.replace(
                f"_{MATCH_SUFFIX}_{metric_config.column_suffix}", ""
            )
            if field_name not in field_stats:
                field_stats[field_name] = {}

            series = df_matched[col].dropna().astype(float)
            series = series[(series >= 0) & (series <= 1)]

            if not series.empty:
                field_stats[field_name][f"{metric_config.name} mean"] = float(
                    series.mean()
                )
                field_stats[field_name][f"{metric_config.name} min"] = float(
                    series.min()
                )
                field_stats[field_name][f"{metric_config.name} max"] = float(
                    series.max()
                )
            else:
                field_stats[field_name][f"{metric_config.name} mean"] = None
                field_stats[field_name][f"{metric_config.name} min"] = None
                field_stats[field_name][f"{metric_config.name} max"] = None

    # Convert to list of rows
    per_field_rows = []
    for field_name, stats in sorted(field_stats.items()):
        row = {"Field": field_name}
        row.update(stats)
        per_field_rows.append(row)

    # Transpose Summary Statistics: convert to Statistic, Value format
    overall_stats_transposed = []
    for stat_name, stat_value in overall_stats.items():
        overall_stats_transposed.append({"Statistic": stat_name, "Value": stat_value})

    return pd.DataFrame(overall_stats_transposed), pd.DataFrame(per_field_rows)


def _null_string_to_null(v):
    if isinstance(v, str):
        if any([kw.lower() == v.lower() for kw in NULL_KEYWORDS]):
            return None
    return v


def _normalize_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    def normalize_value(v):
        v = _null_string_to_null(v)
        if not isinstance(v, str):
            return v
        # remove thousand separators like "2,092,755" → "2092755"
        return _number_commas.sub("", v)

    # apply elementwise, preserve non-strings as-is
    out = df.map(normalize_value)
    # System names are join keys. The matcher strips them, so a reference name
    # like "Vacas Felizes - Average " must be stripped too, or its pair is lost.
    if SCENARIO_NAME_COL in out.columns:
        out[SCENARIO_NAME_COL] = out[SCENARIO_NAME_COL].map(
            lambda v: v.strip() if isinstance(v, str) else v
        )
    return out


def compare_data(
    df_auto: pd.DataFrame,
    df_manual: pd.DataFrame,
    ranker: DocumentRanker,
    ignore_columns: list[str] | None = None,
    max_workers: int | None = None,
    llm_judge: Callable[[str, str], float | bool] | None = None,
    scenario_match_mode: str = SCENARIO_MATCH_MODE_AI,
    scenario_mapper=None,
    mapping_only: bool = False,
    include_literature_comparators: bool = False,
) -> pd.DataFrame:
    """
    Compare auto-extracted data with manual data using fuzzy scenario matching.

    Args:
        df_auto: DataFrame with auto-extracted data
        df_manual: DataFrame with manual data
        ranker: DocumentRanker instance for fuzzy matching
        scenario_match_mode: "ai" (default LLM 1-to-1 join), "name", or
            "name_description".
        mapping_only: If True, return the join without field-level metrics.
        include_literature_comparators: If True, match own_system_mentioned /
            external_reference / definition-only autos (test runs).

    Returns:
        DataFrame with comparison results
    """
    # Remove duplicate columns before processing
    df_auto = df_auto.loc[:, ~df_auto.columns.duplicated()]

    # Identify columns that exist in manual but not in auto
    lost_columns = [x for x in df_manual.columns if x not in df_auto.columns]
    if lost_columns:
        print(f"Columns in manual data but not in auto-extracted: {lost_columns}")

    lit_note = (
        ", include literature comparators"
        if include_literature_comparators
        else ""
    )
    print(
        "Merging dataframes using fuzzy scenario matching "
        f"(mode={scenario_match_mode}{lit_note})..."
    )
    # Merge using fuzzy scenario matching
    df_auto = _normalize_dataframe(df_auto)
    df_manual = _normalize_dataframe(df_manual)

    if scenario_match_mode == SCENARIO_MATCH_MODE_AI:
        max_workers = min(4, max_workers or 4)
        if scenario_mapper is None:
            from auto_lca.statistics.scenario_mapper import (
                DEFAULT_MAPPER_MODEL,
                ScenarioMapper,
            )

            print(f"Initializing AI scenario mapper ({DEFAULT_MAPPER_MODEL})...")
            scenario_mapper = ScenarioMapper()

    joined = _merge_dataframes_fuzzy(
        df_auto,
        df_manual,
        ranker,
        max_workers=max_workers,
        match_mode=scenario_match_mode,
        scenario_mapper=scenario_mapper,
        include_literature_comparators=include_literature_comparators,
    )

    if joined.empty:
        print("Warning: No matching rows found after merge.")
        return joined

    # Remove any duplicate columns that might have been created
    joined = joined.loc[:, ~joined.columns.duplicated()]

    if mapping_only:
        return joined

    print("Creating comparison columns...")
    # Create comparison columns
    sorted_cols, joined = _create_comparison_columns(
        df_auto,
        joined,
        [PID_COL, SCENARIO_NAME_COL],
        ignore_columns=ignore_columns,
        max_workers=max_workers,
        llm_judge=llm_judge,
    )

    # Reorder columns and remove duplicates from sorted_cols
    unique_sorted_cols = []
    seen = set()
    for col in sorted_cols:
        if col not in seen:
            unique_sorted_cols.append(col)
            seen.add(col)

    return joined[unique_sorted_cols]


def _calculate_execution_stats(json_paths: list[str]) -> pd.DataFrame:
    """
    Calculate execution time and token statistics from extracted JSON files.

    Args:
        json_paths: List of paths to extracted JSON files

    Returns:
        DataFrame with execution statistics (Statistic, Value format)
    """
    exec_times = []
    total_tokens = []

    for json_path in json_paths:
        try:
            with open(json_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            # Extract execution time
            metadata = data.get("metadata", {})
            exec_time = metadata.get("exec_time")
            if exec_time is not None:
                exec_times.append(float(exec_time))

            # Extract total tokens
            tokens = data.get("tokens", {})
            total = tokens.get("total")
            if not total:
                field_tokens = tokens.get("field_tokens", tokens.get("aspect_tokens", {}))
                total = field_tokens.get("total_tokens")
            if total is not None:
                total_tokens.append(int(total))
        except (FileNotFoundError, json.JSONDecodeError, KeyError, ValueError) as e:
            print(f"Warning: Could not extract stats from {json_path}: {e}")
            continue

    # Calculate statistics
    stats = []

    if exec_times:
        stats.append(
            {"Statistic": "Execution_Time_Mean", "Value": float(np.mean(exec_times))}
        )
        stats.append(
            {"Statistic": "Execution_Time_Min", "Value": float(np.min(exec_times))}
        )
        stats.append(
            {"Statistic": "Execution_Time_Max", "Value": float(np.max(exec_times))}
        )
    else:
        stats.append({"Statistic": "Execution_Time_Mean", "Value": None})
        stats.append({"Statistic": "Execution_Time_Min", "Value": None})
        stats.append({"Statistic": "Execution_Time_Max", "Value": None})

    if total_tokens:
        stats.append(
            {"Statistic": "Total_Tokens_Mean", "Value": float(np.mean(total_tokens))}
        )
        stats.append(
            {"Statistic": "Total_Tokens_Min", "Value": float(np.min(total_tokens))}
        )
        stats.append(
            {"Statistic": "Total_Tokens_Max", "Value": float(np.max(total_tokens))}
        )
    else:
        stats.append({"Statistic": "Total_Tokens_Mean", "Value": None})
        stats.append({"Statistic": "Total_Tokens_Min", "Value": None})
        stats.append({"Statistic": "Total_Tokens_Max", "Value": None})

    return pd.DataFrame(stats)


# Mapping for prettified execution/extraction statistics
EXEC_STATS_PRETTY_CONFIG = {
    "Execution_Time_Mean": (
        "Execution Time",
        "Execution time per extraction (seconds)",
        "Mean",
        "s",
    ),
    "Execution_Time_Min": ("", "Execution time per extraction (seconds)", "Min", "s"),
    "Execution_Time_Max": ("", "Execution time per extraction (seconds)", "Max", "s"),
    "Total_Tokens_Mean": (
        "Total Tokens",
        "Total tokens used per extraction",
        "Mean",
        "#",
    ),
    "Total_Tokens_Min": ("", "Total tokens used per extraction", "Min", "#"),
    "Total_Tokens_Max": ("", "Total tokens used per extraction", "Max", "#"),
}


def _human_readable_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Rename columns to human-readable format (spaces instead of underscores, title case)."""
    if df.empty:
        return df
    rename = {c: c.replace("_", " ").title() for c in df.columns}
    return df.rename(columns=rename)


def _build_prettified_execution_stats(df_execution_stats: pd.DataFrame) -> pd.DataFrame:
    """
    Build a human-readable execution/extraction statistics table.

    Returns DataFrame with columns: Statistic, Description, Metric, Value, Unit.
    """
    if df_execution_stats is None or df_execution_stats.empty:
        return pd.DataFrame()
    stats_dict = dict(zip(df_execution_stats["Statistic"], df_execution_stats["Value"]))
    rows = []
    for stat_name, (disp, desc, metric, unit) in EXEC_STATS_PRETTY_CONFIG.items():
        val = stats_dict.get(stat_name)
        if val is not None and not (isinstance(val, float) and np.isnan(val)):
            # Statistics: .3 precision; tokens as integers when whole
            fmt_val = (
                int(val)
                if isinstance(val, (int, float)) and val == int(val)
                else round(float(val), 3)
            )
        else:
            fmt_val = val
        rows.append(
            {
                "Statistic": disp,
                "Description": desc,
                "Metric": metric,
                "Value": fmt_val,
                "Unit": unit,
            }
        )
    return _human_readable_columns(pd.DataFrame(rows))


# Mapping for prettified summary statistics: raw Statistic name -> (display_name, description, metric, unit)
# Used when building "Summary Statistics - prettified" sheet. Empty display_name for continuation rows.
STATS_PRETTY_CONFIG = {
    "Matched_Scenarios": (
        "Matched Scenarios",
        "Matched scenarios between automated system and manually created dataset.",
        "Count",
        "#",
    ),
    "Unmatched_Scenarios": (
        "Unmatched Scenarios",
        "Scenarios extracted by automated system but not present in manually created dataset.",
        "Count",
        "#",
    ),
    "Total_Generated_Scenarios": (
        "Total Generated Scenarios",
        "Total number of scenarios generated by the automated system.",
        "Count",
        "#",
    ),
    "Matched_Scenarios_Pct": (
        "Matched Scenarios / Total Scenarios Generated",
        "Percentage of total matched scenarios between automated and manually extracted data.",
        "",
        "%",
    ),
    "Cosine_Similarity_Mean": (
        "Cosine Similarity",
        "Strict cosine similarity compared between SOG outputs and manually created dataset (per datapoint, per scenario).",
        "Mean",
        "Cosine ratio, 0-1",
    ),
    "Cosine_Similarity_Min": (
        "",
        "Strict cosine similarity compared between SOG outputs and manually created dataset (per datapoint, per scenario).",
        "Min",
        "",
    ),
    "Cosine_Similarity_Max": (
        "",
        "Strict cosine similarity compared between SOG outputs and manually created dataset (per datapoint, per scenario).",
        "Max",
        "",
    ),
    "Cosine_Similarity_Mean_NonEmpty": (
        "Cosine Similarity (Non-Empty)",
        "Strict cosine similarity calculated only for non-empty extracted fields between SOG outputs and manually created dataset (per datapoint, per scenario).",
        "Mean (Non-Empty)",
        "Cosine ratio, 0-1",
    ),
    "Cosine_Similarity_Min_NonEmpty": (
        "",
        "Strict cosine similarity calculated only for non-empty extracted fields between SOG outputs and manually created dataset (per datapoint, per scenario).",
        "Min (Non-Empty)",
        "",
    ),
    "Cosine_Similarity_Max_NonEmpty": (
        "",
        "Strict cosine similarity calculated only for non-empty extracted fields between SOG outputs and manually created dataset (per datapoint, per scenario).",
        "Max (Non-Empty)",
        "",
    ),
    "Fuzzy_Cosine_Similarity_Mean": (
        "Fuzzy Cosine Similarity",
        "Fuzzy cosine similarity compared between SOG outputs and manually created dataset (per datapoint, per scenario).",
        "Mean",
        "Cosine ratio, 0-1",
    ),
    "Fuzzy_Cosine_Similarity_Min": (
        "",
        "Fuzzy cosine similarity compared between SOG outputs and manually created dataset (per datapoint, per scenario).",
        "Min",
        "",
    ),
    "Fuzzy_Cosine_Similarity_Max": (
        "",
        "Fuzzy cosine similarity compared between SOG outputs and manually created dataset (per datapoint, per scenario).",
        "Max",
        "",
    ),
    "Fuzzy_Cosine_Similarity_Mean_NonEmpty": (
        "Fuzzy Cosine Similarity (Non-Empty)",
        "Fuzzy cosine similarity calculated only for non-empty extracted fields between SOG outputs and manually created dataset (per datapoint, per scenario).",
        "Mean (Non-Empty)",
        "Cosine ratio, 0-1",
    ),
    "Fuzzy_Cosine_Similarity_Min_NonEmpty": (
        "",
        "Fuzzy cosine similarity calculated only for non-empty extracted fields between SOG outputs and manually created dataset (per datapoint, per scenario).",
        "Min (Non-Empty)",
        "",
    ),
    "Fuzzy_Cosine_Similarity_Max_NonEmpty": (
        "",
        "Fuzzy cosine similarity calculated only for non-empty extracted fields between SOG outputs and manually created dataset (per datapoint, per scenario).",
        "Max (Non-Empty)",
        "",
    ),
    "Fuzzy_Cosine_Threshold_Equal_Mean": (
        "Fuzzy Cosine Similarity Threshold",
        "Fuzzy cosine similarity compared between SOG outputs and manually created dataset (per datapoint, per scenario) using a threshold for equality.",
        "Mean",
        "Ratio True / Total",
    ),
    "Fuzzy_Cosine_Threshold_Equal_Min": (
        "",
        "Fuzzy cosine similarity compared between SOG outputs and manually created dataset (per datapoint, per scenario) using a threshold for equality.",
        "Min",
        "",
    ),
    "Fuzzy_Cosine_Threshold_Equal_Max": (
        "",
        "Fuzzy cosine similarity compared between SOG outputs and manually created dataset (per datapoint, per scenario) using a threshold for equality.",
        "Max",
        "",
    ),
    "Fuzzy_Cosine_Threshold_Equal_Mean_NonEmpty": (
        "Fuzzy Cosine Similarity Threshold (Non-Empty)",
        "Fuzzy cosine similarity (with threshold) calculated only for non-empty extracted fields between SOG outputs and manually created dataset (per datapoint, per scenario).",
        "Mean (Non-Empty)",
        "Ratio True / Total",
    ),
    "Fuzzy_Cosine_Threshold_Equal_Min_NonEmpty": (
        "",
        "Fuzzy cosine similarity (with threshold) calculated only for non-empty extracted fields between SOG outputs and manually created dataset (per datapoint, per scenario).",
        "Min (Non-Empty)",
        "",
    ),
    "Fuzzy_Cosine_Threshold_Equal_Max_NonEmpty": (
        "",
        "Fuzzy cosine similarity (with threshold) calculated only for non-empty extracted fields between SOG outputs and manually created dataset (per datapoint, per scenario).",
        "Max (Non-Empty)",
        "",
    ),
    "LLM_Meaning_Equal_Mean": (
        "LLM-as-a-judge Meaning Equality",
        "True / False (1/0) output of LLM-as-a-judge for semantic equality (per datapoint, per scenario).",
        "Mean",
        "Ratio True / Total",
    ),
    "LLM_Meaning_Equal_Min": (
        "",
        "True / False (1/0) output of LLM-as-a-judge for semantic equality (per datapoint, per scenario).",
        "Min",
        "",
    ),
    "LLM_Meaning_Equal_Max": (
        "",
        "True / False (1/0) output of LLM-as-a-judge for semantic equality (per datapoint, per scenario).",
        "Max",
        "",
    ),
    "LLM_Meaning_Equal_Mean_NonEmpty": (
        "LLM-as-a-judge Meaning Equality (Non-Empty)",
        "True / False (1/0) output of LLM-as-a-judge for semantic equality, calculated only for non-empty extracted fields (per datapoint, per scenario).",
        "Mean (Non-Empty)",
        "Ratio True / Total",
    ),
    "LLM_Meaning_Equal_Min_NonEmpty": (
        "",
        "True / False (1/0) output of LLM-as-a-judge for semantic equality, calculated only for non-empty extracted fields (per datapoint, per scenario).",
        "Min (Non-Empty)",
        "",
    ),
    "LLM_Meaning_Equal_Max_NonEmpty": (
        "",
        "True / False (1/0) output of LLM-as-a-judge for semantic equality, calculated only for non-empty extracted fields (per datapoint, per scenario).",
        "Max (Non-Empty)",
        "",
    ),
    "BERTScore_Placeholder_Precision": (
        "BERTScore",
        "BERTScore on every matched field cell. Empty cells are filled with a dummy token, so this is pulled up by shared empties.",
        "Precision",
        "0-1",
    ),
    "BERTScore_Placeholder_Recall": (
        "",
        "BERTScore on every matched field cell. Empty cells are filled with a dummy token, so this is pulled up by shared empties.",
        "Recall",
        "0-1",
    ),
    "BERTScore_Placeholder_F1": (
        "",
        "BERTScore on every matched field cell. Empty cells are filled with a dummy token, so this is pulled up by shared empties.",
        "F1",
        "0-1",
    ),
    "BERTScore_Placeholder_N_pairs": (
        "",
        "Number of matched field cells included in the all-pairs BERTScore (dummy-filled empties included).",
        "N pairs",
        "#",
    ),
    "BERTScore_NonEmpty_Precision": (
        "BERTScore (Non-Empty)",
        "BERTScore after dropping field cells that are empty on both sides. One-sided empties stay in and count as mismatches.",
        "Precision",
        "0-1",
    ),
    "BERTScore_NonEmpty_Recall": (
        "",
        "BERTScore after dropping field cells that are empty on both sides. One-sided empties stay in and count as mismatches.",
        "Recall",
        "0-1",
    ),
    "BERTScore_NonEmpty_F1": (
        "",
        "BERTScore after dropping field cells that are empty on both sides. One-sided empties stay in and count as mismatches.",
        "F1",
        "0-1",
    ),
    "BERTScore_NonEmpty_N_pairs": (
        "",
        "Number of field cells that are non-empty on at least one side.",
        "N pairs",
        "#",
    ),
    "Empty_Manual_NonEmpty_Auto_Pct_Mean": (
        "Empty Manual, Non-Empty Auto",
        "Per scenario percentage of empty manually extracted fields with non-empty automatically extracted fields.",
        "Mean",
        "%",
    ),
    "Empty_Manual_NonEmpty_Auto_Pct_Min": (
        "",
        "Per scenario percentage of empty manually extracted fields with non-empty automatically extracted fields.",
        "Min",
        "%",
    ),
    "Empty_Manual_NonEmpty_Auto_Pct_Max": (
        "",
        "Per scenario percentage of empty manually extracted fields with non-empty automatically extracted fields.",
        "Max",
        "%",
    ),
    "Empty_Auto_NonEmpty_Manual_Pct_Mean": (
        "Empty Auto, Non-Empty Manual",
        "Per scenario percentage of empty automatically extracted fields with non-empty manually extracted fields.",
        "Mean",
        "%",
    ),
    "Empty_Auto_NonEmpty_Manual_Pct_Min": (
        "",
        "Per scenario percentage of empty automatically extracted fields with non-empty manually extracted fields.",
        "Min",
        "%",
    ),
    "Empty_Auto_NonEmpty_Manual_Pct_Max": (
        "",
        "Per scenario percentage of empty automatically extracted fields with non-empty manually extracted fields.",
        "Max",
        "%",
    ),
}


def _build_prettified_summary_stats(
    df_summary_stats: pd.DataFrame,
    df_f1: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """
    Build a human-readable summary statistics table from raw summary stats.

    Returns DataFrame with columns: Statistic, Description, Metric, Value, Unit.
    """
    rows = []
    stats_dict = dict(zip(df_summary_stats["Statistic"], df_summary_stats["Value"]))

    def add_row(stat_name: str, value: Any) -> None:
        if stat_name not in STATS_PRETTY_CONFIG:
            return
        val = stats_dict.get(stat_name, value)
        disp, desc, metric, unit = STATS_PRETTY_CONFIG[stat_name]
        if val is not None and not (isinstance(val, float) and np.isnan(val)):
            # Percentages: .2 precision; other statistics: .3 precision
            if unit == "%" or "Pct" in stat_name:
                fmt_val = round(val, 2) if isinstance(val, (int, float)) else val
            else:
                fmt_val = round(val, 3) if isinstance(val, float) else val
        else:
            fmt_val = val
        rows.append(
            {
                "Statistic": disp,
                "Description": desc,
                "Metric": metric,
                "Value": fmt_val,
                "Unit": unit,
            }
        )

    # Scenario counts
    matched = stats_dict.get("Matched_Scenarios")
    total = stats_dict.get("Total_Generated_Scenarios")
    pct = (
        (100.0 * matched / total)
        if (matched is not None and total and total > 0)
        else None
    )

    add_row("Matched_Scenarios", None)
    add_row("Unmatched_Scenarios", None)
    add_row("Total_Generated_Scenarios", None)
    add_row("Matched_Scenarios_Pct", round(pct, 2) if pct is not None else None)

    # Metrics (from STATS_CONFIG order)
    for metric_config in STATS_CONFIG.metrics:
        base = metric_config.display_name.replace(" ", "_")
        for suffix in ["_Mean", "_Min", "_Max"]:
            add_row(f"{base}{suffix}", None)
        for suffix in ["_Mean_NonEmpty", "_Min_NonEmpty", "_Max_NonEmpty"]:
            add_row(f"{base}{suffix}", None)

    # BERTScore from df_f1
    if df_f1 is not None and not df_f1.empty:
        f1_dict = dict(zip(df_f1["Statistic"], df_f1["Value"]))
        for stat in [
            "BERTScore_Placeholder_Precision",
            "BERTScore_Placeholder_Recall",
            "BERTScore_Placeholder_F1",
            "BERTScore_Placeholder_N_pairs",
            "BERTScore_NonEmpty_Precision",
            "BERTScore_NonEmpty_Recall",
            "BERTScore_NonEmpty_F1",
            "BERTScore_NonEmpty_N_pairs",
        ]:
            val = f1_dict.get(stat)
            if stat in STATS_PRETTY_CONFIG:
                disp, desc, metric, unit = STATS_PRETTY_CONFIG[stat]
                if isinstance(val, float) and val is not None and not np.isnan(val):
                    fmt_val = round(val, 3) if unit != "#" else int(val)
                elif isinstance(val, (int, float)) and val is not None:
                    fmt_val = val
                else:
                    fmt_val = val
                rows.append(
                    {
                        "Statistic": disp,
                        "Description": desc,
                        "Metric": metric,
                        "Value": fmt_val,
                        "Unit": unit,
                    }
                )

    # Empty/Non-empty percentages
    add_row("Empty_Manual_NonEmpty_Auto_Pct_Mean", None)
    add_row("Empty_Manual_NonEmpty_Auto_Pct_Min", None)
    add_row("Empty_Manual_NonEmpty_Auto_Pct_Max", None)
    add_row("Empty_Auto_NonEmpty_Manual_Pct_Mean", None)
    add_row("Empty_Auto_NonEmpty_Manual_Pct_Min", None)
    add_row("Empty_Auto_NonEmpty_Manual_Pct_Max", None)

    return _human_readable_columns(pd.DataFrame(rows))


COSINE_MEAN_COL = "Cosine_Similarity_mean"


def _build_top_bottom_fields_by_cosine(
    df_stats_by_field: pd.DataFrame,
    top_n: int = 5,
    bottom_n: int = 5,
) -> pd.DataFrame:
    """
    Build a table of top and bottom N performing fields by cosine similarity.

    From top fields, excludes those with cosine=1 (no data in manual nor auto).
    Values formatted to .3 precision.

    Returns DataFrame with columns: Rank, Field, Cosine_Similarity, Empty_Manual_NonEmpty_Auto_Pct, Empty_Auto_NonEmpty_Manual_Pct, Section.
    """
    if df_stats_by_field is None or df_stats_by_field.empty:
        return pd.DataFrame()
    if COSINE_MEAN_COL not in df_stats_by_field.columns:
        return pd.DataFrame()

    df = df_stats_by_field.copy()
    df[COSINE_MEAN_COL] = pd.to_numeric(df[COSINE_MEAN_COL], errors="coerce")
    df = df[df[COSINE_MEAN_COL].notna()].copy()
    if df.empty:
        return pd.DataFrame()

    # Top: exclude cosine=1 (no data), then take top N by cosine descending
    df_excl_perfect = df[df[COSINE_MEAN_COL] < 1.0].copy()
    top = df_excl_perfect.nlargest(top_n, COSINE_MEAN_COL)
    bottom = df.nsmallest(bottom_n, COSINE_MEAN_COL)

    def _format_row(r, rank_label: str, rank_val: int) -> dict:
        row = {
            "Section": rank_label,
            "Rank": rank_val,
            "Field": r["Field"],
            "Cosine Similarity": round(float(r[COSINE_MEAN_COL]), 3),
        }
        if "Empty_Manual_NonEmpty_Auto_Pct" in r.index and pd.notna(
            r.get("Empty_Manual_NonEmpty_Auto_Pct")
        ):
            row["Empty Manual, Non-Empty Auto (%)"] = round(
                float(r["Empty_Manual_NonEmpty_Auto_Pct"]), 2
            )
        else:
            row["Empty Manual, Non-Empty Auto (%)"] = None
        if "Empty_Auto_NonEmpty_Manual_Pct" in r.index and pd.notna(
            r.get("Empty_Auto_NonEmpty_Manual_Pct")
        ):
            row["Empty Auto, Non-Empty Manual (%)"] = round(
                float(r["Empty_Auto_NonEmpty_Manual_Pct"]), 2
            )
        else:
            row["Empty Auto, Non-Empty Manual (%)"] = None
        return row

    rows = []
    for i, (_, r) in enumerate(top.iterrows(), 1):
        rows.append(_format_row(r, "Top", i))
    for i, (_, r) in enumerate(bottom.iterrows(), 1):
        rows.append(_format_row(r, "Bottom", i))

    return _human_readable_columns(pd.DataFrame(rows))


SCENARIO_MATCH_THRESHOLD = 0.8


def build_scenario_matching(
    df_joined: pd.DataFrame,
    match_threshold: float = SCENARIO_MATCH_THRESHOLD,
    category_field: str | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Scenario-level view of the joined data: one row per extracted
    scenario, showing which manual scenario it matched and how closely.

    Returns (per-scenario rows, per-paper summary).

    The sheet keeps every row (extracted literature comparators,
    review-created GT, not-downloaded papers). Match rates and deltas use
    only rows with EXCLUDE_FROM_ACCURACY false.

    `category_field` (optional) names a domain content-category concept
    field already present in df_joined (e.g. "SCENARIO CHARACTERIZATION_
    TECHNOLOGY CATEGORY") whose auto/manual values get carried through as
    "Auto Category" / "Manual Category" columns, for grouping by
    build_scenario_category_stats. Domains with no comparable field just
    omit this argument.
    """
    if (
        df_joined is None
        or df_joined.empty
        or SCENARIO_NAME_COL not in df_joined.columns
    ):
        return pd.DataFrame(), pd.DataFrame()

    manual_scenario_col = f"{SCENARIO_NAME_COL}_manual"
    manual_pid_col = f"{PID_COL}_manual"

    def col(name, default=None):
        return df_joined[name] if name in df_joined.columns else default

    # Unmatched manual scenarios arrive with every auto column blanked,
    # PID included, so the paper they belong to is only in PID_manual.
    pid_series = col(PID_COL)
    if manual_pid_col in df_joined.columns:
        pid_series = (
            pid_series.fillna(df_joined[manual_pid_col])
            if pid_series is not None
            else df_joined[manual_pid_col]
        )

    work = pd.DataFrame(
        {
            PID_COL: pid_series,
            SCENARIO_NAME_COL: df_joined[SCENARIO_NAME_COL],
            "Manual Scenario": col(manual_scenario_col),
            "Similarity": col(SCENARIO_SIMILARITY_COL),
            "Scenario Category": col(SCENARIO_CATEGORY_COL),
            "Definition Only": col(SCENARIO_DEFINITION_ONLY_COL),
            "Exclude from accuracy": col(EXCLUDE_FROM_ACCURACY_COL),
            "Exclusion reason": col(EXCLUSION_REASON_COL),
            "Auto match text": col(SCENARIO_MATCH_TEXT_AUTO_COL),
            "Manual match text": col(SCENARIO_MATCH_TEXT_MANUAL_COL),
            "AI confidence": col(SCENARIO_AI_CONFIDENCE_COL),
            "AI reason": col(SCENARIO_AI_REASON_COL),
            "Cosine greedy partner": col(SCENARIO_COSINE_PARTNER_COL),
            "Collapsed into": col(SCENARIO_COLLAPSED_INTO_COL),
            "Review collapsed": col(SCENARIO_REVIEW_COLLAPSED_COL),
            "Paper scenario count": col(SCENARIO_PAPER_SCENARIO_COUNT_COL),
            "Text mentioned": col(SCENARIO_TEXT_MENTIONED_COL),
            "Text mention snippet": col(SCENARIO_TEXT_MENTION_SNIPPET_COL),
        }
    )
    if category_field:
        auto_cat_col = (
            category_field
            if category_field in df_joined.columns
            else f"{category_field}_auto"
        )
        manual_cat_col = f"{category_field}_manual"
        if auto_cat_col in df_joined.columns:
            work["Auto Category"] = df_joined[auto_cat_col]
        if manual_cat_col in df_joined.columns:
            work["Manual Category"] = df_joined[manual_cat_col]
    work["Similarity"] = pd.to_numeric(work["Similarity"], errors="coerce").fillna(0.0)
    if "AI confidence" in work.columns:
        work["AI confidence"] = pd.to_numeric(work["AI confidence"], errors="coerce")
    work["Definition Only"] = work["Definition Only"].map(
        lambda v: _truthy(v, default=False)
    )
    work["Exclude from accuracy"] = work["Exclude from accuracy"].map(
        lambda v: _truthy(v, default=False)
    )
    # Definition-only extracted rows must stay excluded even if the join
    # flag was missing (older reports / dairy).
    work.loc[work["Definition Only"], "Exclude from accuracy"] = True
    # Reason follows the class; only external systems are literature comparators.
    no_reason = work["Definition Only"] & work["Exclusion reason"].isna()
    cat = work["Scenario Category"].fillna("").astype(str).str.strip().str.lower()
    work.loc[no_reason, "Exclusion reason"] = REASON_LITERATURE_COMPARATOR
    work.loc[no_reason & (cat == "own_system_mentioned"), "Exclusion reason"] = (
        REASON_OWN_SYSTEM_MENTIONED
    )
    work.loc[no_reason & (cat == "subcategory_breakdown"), "Exclusion reason"] = (
        REASON_SUBCATEGORY_BREAKDOWN
    )

    ai_pairs = (
        "AI confidence" in work.columns and work["AI confidence"].fillna(0).gt(0).any()
    )
    if ai_pairs:
        work["Matched"] = (
            ~work["Exclude from accuracy"]
            & work["Manual Scenario"].notna()
            & work["AI confidence"].fillna(0).gt(0)
        )
    else:
        work["Matched"] = (
            ~work["Exclude from accuracy"]
            & work["Manual Scenario"].notna()
            & (work["Similarity"] >= match_threshold)
        )

    # Keep extracted rows and manual-only excluded rows (not downloaded /
    # review-created) so they remain visible on the sheet.
    extracted = work[work[SCENARIO_NAME_COL].notna()].copy()
    manual_only = work[
        work[SCENARIO_NAME_COL].isna() & work["Manual Scenario"].notna()
    ].copy()
    sheet = pd.concat([extracted, manual_only], ignore_index=True)
    sheet = sheet.drop_duplicates(
        subset=[PID_COL, SCENARIO_NAME_COL, "Manual Scenario"]
    ).sort_values([PID_COL, SCENARIO_NAME_COL], kind="stable")

    counted = sheet[~sheet["Exclude from accuracy"]]
    counted_extracted = counted[counted[SCENARIO_NAME_COL].notna()]

    summary_rows = []
    for pid, group in sheet.groupby(PID_COL, dropna=True):
        counted_group = group[~group["Exclude from accuracy"]]
        counted_ext = counted_group[counted_group[SCENARIO_NAME_COL].notna()]
        n_manual_counted = counted_group[counted_group["Manual Scenario"].notna()][
            "Manual Scenario"
        ].nunique()
        n_manual_all = group[group["Manual Scenario"].notna()][
            "Manual Scenario"
        ].nunique()
        paper_not_downloaded = bool(
            group["Exclusion reason"].fillna("").str.contains(REASON_NOT_DOWNLOADED).any()
            and counted_ext.empty
        )
        match_rate = (
            round(counted_ext["Matched"].mean(), 4) if len(counted_ext) else None
        )
        n_gt_paired = counted_ext.loc[
            counted_ext["Matched"], "Manual Scenario"
        ].nunique()
        gt_pair_rate = (
            round(n_gt_paired / n_manual_counted, 4) if n_manual_counted else None
        )
        n_collapsed = int(
            group["Exclusion reason"]
            .fillna("")
            .astype(str)
            .str.contains(REASON_REVIEW_COLLAPSED, na=False)
            .sum()
        )
        if "Review collapsed" in group.columns:
            n_collapsed_gt = int(
                group.loc[
                    group["Review collapsed"].map(lambda v: _truthy(v, default=False))
                    & group["Manual Scenario"].notna(),
                    "Manual Scenario",
                ].nunique()
            )
        else:
            n_collapsed_gt = 0
        summary_rows.append(
            {
                PID_COL: pid,
                "Extracted Scenarios": int(group[SCENARIO_NAME_COL].notna().sum()),
                "Extracted (counted)": len(counted_ext),
                "Definition-Only Scenarios": int(group["Definition Only"].sum()),
                "Excluded from accuracy": int(group["Exclude from accuracy"].sum()),
                "Manual Scenarios": n_manual_all,
                "Manual Scenarios (counted)": n_manual_counted,
                "Matched": int(counted_ext["Matched"].sum()),
                "Match Rate": match_rate,
                "GT paired (counted)": n_gt_paired,
                "GT pair rate": gt_pair_rate,
                "Collapsed paper variants": n_collapsed,
                "Review collapsed GT rows": n_collapsed_gt,
                "Paper not downloaded": paper_not_downloaded,
                "Scenario Count Delta (counted extracted - counted manual)": (
                    len(counted_ext) - n_manual_counted
                ),
            }
        )

    df_summary = pd.DataFrame(summary_rows)
    if not df_summary.empty:
        counted_papers = df_summary[~df_summary["Paper not downloaded"]]
        totals = {
            PID_COL: "ALL PAPERS (counted)",
            "Extracted Scenarios": int(df_summary["Extracted Scenarios"].sum()),
            "Extracted (counted)": int(counted_papers["Extracted (counted)"].sum()),
            "Definition-Only Scenarios": int(
                df_summary["Definition-Only Scenarios"].sum()
            ),
            "Excluded from accuracy": int(df_summary["Excluded from accuracy"].sum()),
            "Manual Scenarios": int(df_summary["Manual Scenarios"].sum()),
            "Manual Scenarios (counted)": int(
                counted_papers["Manual Scenarios (counted)"].sum()
            ),
            "Matched": int(counted_papers["Matched"].sum()),
            "Match Rate": (
                round(counted_extracted["Matched"].mean(), 4)
                if len(counted_extracted)
                else None
            ),
            "GT paired (counted)": int(counted_papers["GT paired (counted)"].sum()),
            "GT pair rate": (
                round(
                    counted_papers["GT paired (counted)"].sum()
                    / counted_papers["Manual Scenarios (counted)"].sum(),
                    4,
                )
                if counted_papers["Manual Scenarios (counted)"].sum()
                else None
            ),
            "Collapsed paper variants": int(
                counted_papers["Collapsed paper variants"].sum()
            ),
            "Review collapsed GT rows": int(
                counted_papers["Review collapsed GT rows"].sum()
            ),
            "Paper not downloaded": int(df_summary["Paper not downloaded"].sum()),
            "Scenario Count Delta (counted extracted - counted manual)": int(
                counted_papers["Extracted (counted)"].sum()
                - counted_papers["Manual Scenarios (counted)"].sum()
            ),
        }
        df_summary = pd.concat([df_summary, pd.DataFrame([totals])], ignore_index=True)

    return sheet, df_summary


def _canonical_category(value: Any, mapping: dict[str, str] | None) -> str | None:
    """Look up a raw category label (case/whitespace-insensitive) in a
    crosswalk mapping. Falls back to the raw label itself when there's no
    mapping entry, so unmapped labels still surface as their own bucket
    instead of silently disappearing. Returns None for empty/missing values.
    """
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    text = str(value).strip()
    if not text:
        return None
    if mapping:
        canon = mapping.get(text.lower())
        if canon:
            return canon
    return text


def build_scenario_category_stats(
    df_scenario_matching: pd.DataFrame,
    crosswalk: dict[str, dict[str, str]] | None = None,
) -> pd.DataFrame:
    """
    Group scenario-matching results by a domain content category (e.g.
    BBP "Technology category", CCU "Carbonation route") instead of one
    flat aggregate match rate -- so match/no-match can be read off per
    scenario group.

    Requires build_scenario_matching(..., category_field=...) to have
    been called first so "Auto Category" (and optionally "Manual
    Category") columns are present.

    `crosswalk` maps raw auto/manual category label text (case-insensitive)
    onto a shared canonical bucket name:
        {"auto": {"raw auto label": "canonical", ...},
         "manual": {"raw manual label": "canonical", ...}}
    Either side may be omitted, and unmapped raw labels pass through as
    their own bucket rather than being dropped.

    Generic fallback: when the ground truth has no comparable category
    field for this domain (no "Manual Category" column, or no non-null
    values, or no "manual" crosswalk entries at all), the table groups by
    the auto-side canonical category only (Extracted / Matched / Unmatched
    / Match rate) -- no GT-side columns are added, since there is nothing
    on the manual side to group by.
    """
    if (
        df_scenario_matching is None
        or df_scenario_matching.empty
        or "Auto Category" not in df_scenario_matching.columns
    ):
        return pd.DataFrame()

    crosswalk = crosswalk or {}
    auto_map = {
        str(k).strip().lower(): v for k, v in (crosswalk.get("auto") or {}).items()
    }
    manual_map = {
        str(k).strip().lower(): v for k, v in (crosswalk.get("manual") or {}).items()
    }

    sheet = df_scenario_matching.copy()
    sheet["_auto_canon"] = sheet["Auto Category"].map(
        lambda v: _canonical_category(v, auto_map)
    )

    # The GT-side axis only makes sense when the crosswalk actually defines
    # manual categories -- i.e. the ground truth has a comparable, mapped
    # category scheme for this domain. A "Manual Category" column merely
    # existing in the joined data (raw, unmapped GT text) is not enough:
    # without a crosswalk it would compare canonicalized auto labels
    # against un-canonicalized manual text and never line up, silently
    # producing all-orphan noise instead of the intended auto-only view.
    has_manual_axis = (
        bool(manual_map)
        and "Manual Category" in sheet.columns
        and sheet["Manual Category"].notna().any()
    )
    if has_manual_axis:
        sheet["_manual_canon"] = sheet["Manual Category"].map(
            lambda v: _canonical_category(v, manual_map)
        )

    counted = sheet[~sheet["Exclude from accuracy"]]
    counted_extracted = counted[
        counted[SCENARIO_NAME_COL].notna() & counted["_auto_canon"].notna()
    ]

    rows = []
    for cat, group in counted_extracted.groupby("_auto_canon", dropna=False):
        row = {
            "Category (auto)": cat,
            "Extracted (counted)": len(group),
            "Matched": int(group["Matched"].sum()),
            "Unmatched (extracted)": int((~group["Matched"]).sum()),
            "Match rate": (
                round(float(group["Matched"].mean()), 4) if len(group) else None
            ),
        }
        if has_manual_axis:
            in_bucket = sheet["_manual_canon"] == cat
            gt_in_bucket = sheet.loc[in_bucket, "Manual Scenario"].dropna().nunique()
            gt_matched = (
                sheet.loc[in_bucket & sheet["Matched"], "Manual Scenario"]
                .dropna()
                .nunique()
            )
            row["GT rows (this category)"] = gt_in_bucket
            row["GT rows matched"] = gt_matched
            row["GT rows unmatched"] = gt_in_bucket - gt_matched
        rows.append(row)

    df_stats = pd.DataFrame(rows)
    if not df_stats.empty:
        df_stats = df_stats.sort_values(
            "Extracted (counted)", ascending=False
        ).reset_index(drop=True)

    if has_manual_axis:
        # GT categories that no extracted scenario's canonical bucket ever
        # lands on -- e.g. a GT bucket the schema/crosswalk has no auto
        # equivalent for at all (structurally unmatchable, not a scoring
        # bug). Surfaced as their own rows with zero extracted/matched.
        auto_cats = set(df_stats["Category (auto)"]) if not df_stats.empty else set()
        orphan_mask = sheet["_manual_canon"].notna() & ~sheet["_manual_canon"].isin(
            auto_cats
        )
        if orphan_mask.any():
            orphan_rows = []
            for cat, group in sheet[orphan_mask].groupby("_manual_canon"):
                gt_count = group["Manual Scenario"].dropna().nunique()
                if gt_count:
                    orphan_rows.append(
                        {
                            "Category (auto)": f"(no auto category maps here) -- GT: {cat}",
                            "Extracted (counted)": 0,
                            "Matched": 0,
                            "Unmatched (extracted)": 0,
                            "Match rate": None,
                            "GT rows (this category)": gt_count,
                            "GT rows matched": 0,
                            "GT rows unmatched": gt_count,
                        }
                    )
            if orphan_rows:
                df_stats = pd.concat(
                    [df_stats, pd.DataFrame(orphan_rows)], ignore_index=True
                )

    return df_stats


def build_excluded_fields_sheet(
    excluded_fields: list[dict[str, str]] | None,
) -> pd.DataFrame:
    """
    Turn a domain's excluded-fields declaration (fields intentionally kept
    null / dropped from scoring because they're reviewer-computed rather
    than paper-derivable) into a documentation sheet: which field, from
    what source, and why. Keeps that reasoning visible in the report
    instead of the field just silently vanishing from "Joined Data".
    """
    if not excluded_fields:
        return pd.DataFrame()
    rows = [
        {
            "Field": item.get("field", ""),
            "Source": item.get("source", ""),
            "Reason excluded from scoring": item.get("reason", ""),
        }
        for item in excluded_fields
        if item.get("field")
    ]
    return pd.DataFrame(rows)


def save_results_to_excel(
    output_excel_path,
    df_auto,
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
    excluded_fields: list[dict[str, str]] | None = None,
    category_field: str | None = None,
    category_crosswalk: dict[str, dict[str, str]] | None = None,
    field_decomposition_cosine_threshold: float = 0.15,
    check_provenance: bool = False,
    audited_pids: set | None = None,
):
    # Sanitize dataframes before writing to Excel
    _ = df_auto  # currently unused but kept for potential future export
    print(f"Preparing file {output_excel_path}")
    df_joined_sanitized = _sanitize_dataframe_for_excel(df_joined)

    with pd.ExcelWriter(output_excel_path, engine="openpyxl") as writer:
        # Joined Data sheet
        df_joined_sanitized.to_excel(writer, sheet_name="Joined Data", index=False)

        # Scenario matching (one row per extracted scenario) + per-paper
        # match rates, derived from df_joined so every caller gets them.
        df_scenario_matching, df_scenario_match_stats = build_scenario_matching(
            df_joined, category_field=category_field
        )
        if not df_scenario_matching.empty:
            _sanitize_dataframe_for_excel(df_scenario_matching).to_excel(
                writer, sheet_name="Scenario Matching", index=False
            )
        if not df_scenario_match_stats.empty:
            _sanitize_dataframe_for_excel(df_scenario_match_stats).to_excel(
                writer, sheet_name="Scenario Matching Stats", index=False
            )

        # Scenario matching grouped by domain content category (e.g. BBP
        # technology category, CCU carbonation route). Falls back to an
        # auto-only grouping when the domain has no comparable GT field --
        # see build_scenario_category_stats.
        df_category_stats = build_scenario_category_stats(
            df_scenario_matching, crosswalk=category_crosswalk
        )
        if not df_category_stats.empty:
            _sanitize_dataframe_for_excel(df_category_stats).to_excel(
                writer, sheet_name="Scenario Matching by Category", index=False
            )

        # Documents which fields were excluded from scoring and why (e.g.
        # confirmed reviewer-computed values with no basis in the source
        # paper), so the exclusion is visible in the report, not just
        # silently absent from Joined Data.
        df_excluded_fields = build_excluded_fields_sheet(excluded_fields)
        if not df_excluded_fields.empty:
            _sanitize_dataframe_for_excel(df_excluded_fields).to_excel(
                writer, sheet_name="Excluded Fields", index=False
            )

        # Per-field decomposition of WHY a field's score is low: presence
        # mismatch (null on the wrong side) vs content mismatch (both sides
        # filled but disagree) -- computed for free from Joined Data, no
        # extraction or API calls needed. See field_decomposition.py.
        excluded_field_names = {
            item["field"] for item in (excluded_fields or []) if item.get("field")
        }
        df_field_decomposition = build_field_error_decomposition(
            df_joined,
            cosine_threshold=field_decomposition_cosine_threshold,
            excluded_fields_always=excluded_field_names,
        )
        if not df_field_decomposition.empty:
            _sanitize_dataframe_for_excel(df_field_decomposition).to_excel(
                writer, sheet_name="Field Error Decomposition", index=False
            )

        # Deterministic, no-LLM-cost check: does every number in an
        # extracted value actually appear in that paper's source PDF text?
        # Opt-in (reads every referenced PDF) -- see provenance_checker.py.
        if check_provenance:
            df_provenance = compute_field_provenance(
                df_joined, excluded_fields_always=excluded_field_names
            )
            flagged = (
                df_provenance[df_provenance["Status"] != "verified"]
                if not df_provenance.empty
                else df_provenance
            )
            if not flagged.empty:
                _sanitize_dataframe_for_excel(flagged).to_excel(
                    writer, sheet_name="Provenance Check", index=False
                )
            df_paper_status = compute_paper_status_table(
                df_joined, df_provenance, audited_pids=audited_pids
            )
            if not df_paper_status.empty:
                _sanitize_dataframe_for_excel(df_paper_status).to_excel(
                    writer, sheet_name="Paper Status", index=False
                )

        # Statistics by scenario
        if df_stats_by_scenario is not None and not df_stats_by_scenario.empty:
            _sanitize_dataframe_for_excel(df_stats_by_scenario).to_excel(
                writer, sheet_name="Statistics by Scenario", index=False
            )

        # Statistics by paper
        if df_stats_by_paper is not None and not df_stats_by_paper.empty:
            _sanitize_dataframe_for_excel(df_stats_by_paper).to_excel(
                writer, sheet_name="Statistics by Paper", index=False
            )

        # Statistics by field and scenario
        if (
            df_stats_by_field_scenario is not None
            and not df_stats_by_field_scenario.empty
        ):
            _sanitize_dataframe_for_excel(df_stats_by_field_scenario).to_excel(
                writer, sheet_name="Statistics by Field and Scenario", index=False
            )

        # Statistics by field and paper
        if df_stats_by_field_paper is not None and not df_stats_by_field_paper.empty:
            _sanitize_dataframe_for_excel(df_stats_by_field_paper).to_excel(
                writer, sheet_name="Statistics by Field and Paper", index=False
            )

        # Statistics by field
        if df_stats_by_field is not None and not df_stats_by_field.empty:
            _sanitize_dataframe_for_excel(df_stats_by_field).to_excel(
                writer, sheet_name="Statistics by Field", index=False
            )

        # Summary Statistics
        if df_summary_stats is not None and not df_summary_stats.empty:
            _sanitize_dataframe_for_excel(df_summary_stats).to_excel(
                writer, sheet_name="Summary Statistics", index=False
            )
            df_prettified = _build_prettified_summary_stats(df_summary_stats, df_f1)
            if not df_prettified.empty:
                _sanitize_dataframe_for_excel(df_prettified).to_excel(
                    writer, sheet_name="Summary Statistics - prettified", index=False
                )

        # Sensitivity Analysis
        if df_sensitivity is not None and not df_sensitivity.empty:
            _sanitize_dataframe_for_excel(df_sensitivity).to_excel(
                writer, sheet_name="Sensitivity", index=False
            )

        # BERTScore (Precision, Recall, F1 for text similarity)
        if df_f1 is not None and not df_f1.empty:
            _sanitize_dataframe_for_excel(df_f1).to_excel(
                writer, sheet_name="BERTScore", index=False
            )

        # Execution Statistics
        if df_execution_stats is not None and not df_execution_stats.empty:
            _sanitize_dataframe_for_excel(df_execution_stats).to_excel(
                writer, sheet_name="Execution Statistics", index=False
            )
            df_exec_prettified = _build_prettified_execution_stats(df_execution_stats)
            if not df_exec_prettified.empty:
                _sanitize_dataframe_for_excel(df_exec_prettified).to_excel(
                    writer, sheet_name="Execution Statistics - prettified", index=False
                )

        # Top and Bottom 5 Fields by Cosine Similarity
        df_top_bottom = _build_top_bottom_fields_by_cosine(df_stats_by_field)
        if not df_top_bottom.empty:
            _sanitize_dataframe_for_excel(df_top_bottom).to_excel(
                writer, sheet_name="Top and Bottom Fields by Cosine", index=False
            )
    print(f"Saved results to: {output_excel_path}")


def compare_files(
    extracted_json_path: str,
    manual_xlsx_path: str,
    manual_sheet_name: str,
    output_excel_path: str | None = None,
    csv_header_row: int | list[int] = None,
    ignore_columns: list[str] | None = None,
    max_workers: int | None = None,
    llm_judge: Callable[[str, str], float | bool] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Compare extracted JSON data with manual CSV data and save to Excel.

    Args:
        extracted_json_path: Path to JSON file with auto-extracted data
        manual_xlsx_path: Path to CSV file with manual data
        output_excel_path: Optional path to save Excel file with results
        csv_header_row: Row number to use as header in CSV (0-indexed)

    Returns:
        Tuple of (extracted_df, joined_df, stats_df)
    """
    if not csv_header_row:
        raise ValueError("No header int provided!")
    # Initialize DocumentRanker for fuzzy matching
    ranker = DocumentRanker()

    # Load extracted data
    print(f"Loading extracted data from: {extracted_json_path}")
    metadata, results = load_extracted_json(extracted_json_path)

    pid = metadata.get("pid")

    # Process each result (scenario) separately
    df_auto_list = []
    for result in results:
        concepts = result.get("concepts", [])
        if not concepts:
            continue

        # Extract scenario name
        scenario_name = extract_scenario_name_from_concepts(concepts)

        # Process extracted data for this scenario
        df_scenario = process_extracted_data(concepts, pid, scenario_name)
        df_auto_list.append(df_scenario)

    if not df_auto_list:
        print("Warning: No concepts found in extracted data.")
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    # Combine all scenarios into one dataframe
    df_auto = pd.concat(df_auto_list, ignore_index=True)

    # Load manual data
    print(f"Loading manual data from: {manual_xlsx_path}")
    df_manual, _manual_nested = load_manual_data(
        manual_xlsx_path, manual_sheet_name, csv_header_row, as_nested=True
    )

    # Compare data with fuzzy matching
    print("Comparing data with fuzzy scenario matching...")

    df_joined = compare_data(
        df_auto,
        df_manual,
        ranker,
        ignore_columns=ignore_columns,
        max_workers=max_workers,
        llm_judge=llm_judge,
    )

    # Calculate statistics
    print("Calculating statistics...")
    df_stats = _calculate_statistics(df_joined)
    df_overall, df_per_field = _calculate_overall_statistics(df_joined, df_stats)

    # Save to Excel if requested
    if output_excel_path:
        save_results_to_excel(
            output_excel_path, df_auto, df_joined, df_stats, df_overall, df_per_field
        )

    return df_auto, df_joined, df_stats
