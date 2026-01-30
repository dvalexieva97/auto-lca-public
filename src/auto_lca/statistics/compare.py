"""
Module for comparing auto-extracted data with manual data.

This module provides functionality to:
- Load extracted data from JSON files
- Compare with manual CSV data using fuzzy scenario name matching
- Generate comparison reports in Excel format with statistics
"""

import json
from concurrent.futures import ThreadPoolExecutor
from typing import Callable, Iterable, Optional

import numpy as np
import pandas as pd

from auto_lca.process.nlp.document_rank import DocumentRanker
from auto_lca.shared.util import flatten_extracted
from auto_lca.statistics.similarity import DataSimilarity

try:
    from litellm import completion

    LITELLM_AVAILABLE = True
except ImportError:
    LITELLM_AVAILABLE = False
    completion = None

# Constants
SCENARIO_NAME_COL = "SCENARIO NAME"
PID_COL = "PID"
MATCH_SUFFIX = "_match"
FUZZY_FLAG = True

col_rename_dict = {
    "STUDY DETAILS_SCENARIO NAME": "SCENARIO NAME",
    "METADATA_PID": "PID",
}
ds = DataSimilarity()


def create_llm_meaning_judge(
    model: str = "gemini/gemini-2.0-flash-exp",
    api_key: Optional[str] = None,
    temperature: float = 0.0,
) -> Callable[[str, str], Optional[float]]:
    """
    Create an LLM judge function that evaluates semantic meaning equality between two strings.

    The judge determines if two strings have the same meaning, even if they differ in specificity
    or wording. For example, "Simapro software v8.0.5" and "SimaPro" would be considered equal
    in meaning.

    Args:
        model: LLM model identifier (default: "gemini/gemini-2.0-flash-exp")
        api_key: API key for the LLM provider (if None, uses environment variables)
        temperature: Temperature for LLM generation (default: 0.0 for deterministic output)

    Returns:
        A callable function that takes two strings and returns 1.0 if they have the same meaning,
        0.0 otherwise. Returns float in [0,1] format.
    """
    if not LITELLM_AVAILABLE:
        raise ImportError(
            "litellm is required for LLM judge functionality. "
            "Install it with: pip install litellm"
        )

    def judge(val1: str, val2: str) -> Optional[float]:
        """
        Judge if two values have the same meaning.

        Args:
            val1: First value to compare
            val2: Second value to compare

        Returns:
            1.0 if values have the same meaning, 0.0 if different, None on error
        """
        prompt = f"""You are a semantic equivalence judge. Determine if two values have the same meaning, even if they differ in specificity or wording.

Examples:
- "Simapro software v8.0.5" and "SimaPro" -> SAME MEANING (one is more specific)
- "CO2" and "carbon dioxide" -> SAME MEANING (abbreviation vs full form)
- "kg" and "kilogram" -> SAME MEANING (abbreviation vs full form)
- "Apple" and "Orange" -> DIFFERENT MEANING (completely different things)
- "1.5" and "2.0" -> DIFFERENT MEANING (different numeric values)

Value 1: "{val1}"
Value 2: "{val2}"

Do these two values have the same meaning? Respond with only "YES" or "NO"."""

        try:
            response = completion(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
                api_key=api_key,
            )

            result_text = response.choices[0].message.content.strip().upper()

            # Check if response indicates same meaning
            if "YES" in result_text or "SAME" in result_text:
                return 1.0
            else:
                return 0.0
        except Exception as e:
            # On error, return None (will be handled by calling code)
            print(f"Warning: LLM judge error: {e}")
            return None

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


def extract_scenario_name_from_concepts(concepts: list) -> Optional[str]:
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
    concepts: list, pid: Optional[str] = None, scenario_name: Optional[str] = None
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
        pid = metadata.get("pid")
        pdf_path = metadata.get("pdf_path")

        # Process each result (scenario) separately
        for result in results:
            concepts = result.get("concepts", [])
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
    output_folder: str, output_csv_path: Optional[str] = None
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


def load_manual_data(
    xlsx_path: str,
    sheet_name=str,
    header_row: int | list[int] = 2,
    as_nested: bool = False,
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
    df = pd.read_excel(xlsx_path, sheet_name, header=header_row)
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


def _build_scenario_mapping(
    auto_scenarios: list,
    manual_scenarios: list,
    ranker: DocumentRanker,
) -> dict:
    """
    Build a 1-to-1 mapping of auto scenarios to manual scenarios based on similarity.

    Args:
        auto_scenarios: List of auto scenario names
        manual_scenarios: List of manual scenario names
        ranker: DocumentRanker instance for similarity computation

    Returns:
        Dictionary mapping auto_scenario -> (manual_scenario, similarity_score)
    """
    mapping = {}
    used_manual = set()

    # Compute similarity matrix
    similarity_matrix = {}
    for auto_scenario in auto_scenarios:
        for manual_scenario in manual_scenarios:
            try:
                similarity = ranker.compute_fuzzy_cosine_similarity(
                    str(auto_scenario), str(manual_scenario)
                )
                if similarity is None:
                    similarity = ranker.compute_cosine_similarity(
                        str(auto_scenario), str(manual_scenario)
                    )
            except Exception:
                similarity = ranker.compute_cosine_similarity(
                    str(auto_scenario), str(manual_scenario)
                )
            similarity_matrix[(auto_scenario, manual_scenario)] = similarity

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


def _fuzzy_match_scenarios(
    df_auto: pd.DataFrame,
    df_manual: pd.DataFrame,
    ranker: DocumentRanker,
    pid: str,
) -> pd.DataFrame:
    """
    Match scenarios using fuzzy cosine similarity with outer join.

    Args:
        df_auto: DataFrame with auto-extracted data
        df_manual: DataFrame with manual data
        ranker: DocumentRanker instance for similarity computation
        pid: Paper ID to filter on

    Returns:
        DataFrame with matched rows (outer join - includes unmatched scenarios)
    """
    # Filter by PID
    df_auto_filtered = df_auto[df_auto[PID_COL] == pid].copy()
    df_manual_filtered = df_manual[df_manual[PID_COL] == pid].copy()

    if df_auto_filtered.empty and df_manual_filtered.empty:
        return pd.DataFrame()

    # Get scenario names
    auto_scenarios = (
        df_auto_filtered[SCENARIO_NAME_COL].dropna().unique()
        if not df_auto_filtered.empty
        else []
    )
    manual_scenarios = (
        df_manual_filtered[SCENARIO_NAME_COL].dropna().unique()
        if not df_manual_filtered.empty
        else []
    )

    # Build scenario mapping
    if len(auto_scenarios) > 0 and len(manual_scenarios) > 0:
        scenario_mapping = _build_scenario_mapping(
            auto_scenarios, manual_scenarios, ranker
        )
    else:
        scenario_mapping = {}
        # If one side is empty, map all scenarios to None
        for auto_scenario in auto_scenarios:
            scenario_mapping[auto_scenario] = (None, 0.0)

    # Build joined rows
    joined_rows = []

    # Process matched auto scenarios
    for auto_scenario, (manual_scenario, similarity) in scenario_mapping.items():
        auto_rows = df_auto_filtered[
            df_auto_filtered[SCENARIO_NAME_COL] == auto_scenario
        ]

        if manual_scenario is not None:
            manual_rows = df_manual_filtered[
                df_manual_filtered[SCENARIO_NAME_COL] == manual_scenario
            ]
        else:
            manual_rows = pd.DataFrame()

        # Join each auto row with its matched manual row(s)
        for _, auto_row in auto_rows.iterrows():
            if not manual_rows.empty:
                # If multiple manual rows, take the first one (or could merge)
                for _, manual_row in manual_rows.iterrows():
                    combined = auto_row.to_dict()
                    # Add manual columns with _manual suffix
                    for col in manual_row.index:
                        combined[f"{col}_manual"] = manual_row[col]
                    combined["_scenario_name_similarity_score"] = similarity
                    joined_rows.append(combined)
                    break  # Take first manual row only
            else:
                # Unmatched auto scenario - fill with empty manual columns
                combined = auto_row.to_dict()
                for col in df_manual_filtered.columns:
                    combined[f"{col}_manual"] = None
                combined["_scenario_name_similarity_score"] = 0.0
                joined_rows.append(combined)

    # Process unmatched manual scenarios (scenarios not in mapping)
    matched_manual_scenarios = {
        mapping[0] for mapping in scenario_mapping.values() if mapping[0] is not None
    }
    unmatched_manual_scenarios = [
        s for s in manual_scenarios if s not in matched_manual_scenarios
    ]

    for manual_scenario in unmatched_manual_scenarios:
        manual_rows = df_manual_filtered[
            df_manual_filtered[SCENARIO_NAME_COL] == manual_scenario
        ]

        for _, manual_row in manual_rows.iterrows():
            # Unmatched manual scenario - fill with empty auto columns
            combined = {}
            # Add auto columns as empty
            for col in df_auto_filtered.columns:
                combined[col] = None
            # Add manual columns with _manual suffix
            for col in manual_row.index:
                combined[f"{col}_manual"] = manual_row[col]
            combined["_scenario_name_similarity_score"] = 0.0
            joined_rows.append(combined)

    if not joined_rows:
        return pd.DataFrame()

    return pd.DataFrame(joined_rows)


def _merge_dataframes_fuzzy(
    df_auto: pd.DataFrame,
    df_manual: pd.DataFrame,
    ranker: DocumentRanker,
    max_workers: Optional[int] = None,
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

    # Group by PID and match scenarios
    all_joined = []

    def _match_for_pid(pid_val):
        return _fuzzy_match_scenarios(df_auto, df_manual, ranker, pid_val)

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        for matched in executor.map(_match_for_pid, df_auto[PID_COL].unique()):
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
    """
    Parse a value to float, handling European number format (12,5 -> 12.5).
    Returns None if parsing fails.
    """
    # Handle pandas Series or numpy arrays - extract first element if it's an array-like
    if isinstance(val, (pd.Series, np.ndarray)):
        if len(val) == 0:
            return None
        val = val.iloc[0] if isinstance(val, pd.Series) else val.flat[0]

    # Use existing _is_null method to check for null/empty values
    if ds._is_null(val):
        return None

    # If already a number, return it
    if isinstance(val, (int, float)):
        return float(val)

    # Convert to string and strip
    str_val = str(val).strip()
    if not str_val:
        return None

    # Remove any whitespace
    str_val = str_val.replace(" ", "")

    # Try direct conversion first (handles standard format like "12.5")
    try:
        return float(str_val)
    except ValueError:
        pass

    # Try European format: comma as decimal separator
    # Pattern: digits, comma, digits (e.g., "12,5")
    if "," in str_val and "." not in str_val:
        try:
            # Simple case: just replace comma with dot
            return float(str_val.replace(",", "."))
        except ValueError:
            pass

    # Handle cases with both comma and dot
    # Determine which is the decimal separator based on position
    if "," in str_val and "." in str_val:
        comma_idx = str_val.rfind(",")
        dot_idx = str_val.rfind(".")

        # The one that appears last is likely the decimal separator
        if comma_idx > dot_idx:
            # Comma is decimal separator, dot is thousands separator
            # Remove dot (thousands), replace comma with dot (decimal)
            try:
                return float(str_val.replace(".", "").replace(",", "."))
            except ValueError:
                pass
        else:
            # Dot is decimal separator, comma is thousands separator
            # Remove comma (thousands)
            try:
                return float(str_val.replace(",", ""))
            except ValueError:
                pass

    return None


def _compute_metrics_fast(
    val1,
    val2,
    embeddings_dict: dict,
    llm_judge: Optional[Callable[[str, str], float | bool]] = None,
) -> dict:
    """Fast metric computation using pre-computed embeddings."""
    metrics: dict[str, float | None] = {}

    # Fast path: both null
    if ds._is_null(val1) and ds._is_null(val2):
        return {
            "cosine": 1.0,
            "fuzzy_cosine": 1.0,
            "clean_string_equal": 1.0,
            "numeric_exact": 1.0,
            "numeric_distance": None,
            "llm_meaning_equal": 1.0 if llm_judge else None,
        }

    # Fast path: one null, one not null - they are different
    if ds._is_null(val1) or ds._is_null(val2):
        return {
            "cosine": 0.0,
            "fuzzy_cosine": 0.0,
            "clean_string_equal": 0.0,
            "numeric_exact": 0.0,
            "numeric_distance": None,
            "llm_meaning_equal": 0.0 if llm_judge else None,
        }

    str1 = ds._safe_str(val1)
    str2 = ds._safe_str(val2)

    # Try to parse both values as numeric
    num1 = _parse_numeric_value(val1)
    num2 = _parse_numeric_value(val2)

    # If both are numeric, use numeric similarity
    if num1 is not None and num2 is not None:
        # Calculate numeric similarity: 1 - abs(difference / average)
        avg_val = (num1 + num2) / 2.0
        if avg_val == 0:
            # Both are zero, perfect match
            cosine_sim = 1.0
        else:
            diff = abs(num1 - num2)
            ratio = diff / avg_val
            cosine_sim = max(0.0, 1.0 - ratio)  # Clamp to [0, 1]
    else:
        # Not both numeric, use string similarity
        # Fast path: exact match (after stripping and lowercasing)
        clean_str1 = str1.strip().lower()
        clean_str2 = str2.strip().lower()
        if clean_str1 == clean_str2 and clean_str1:
            cosine_sim = 1.0
        else:
            # Use cached embeddings if available
            # Try both original and cleaned strings for lookup
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
                # Cosine similarity using numpy (already normalized)
                cosine_sim = float(np.dot(emb1, emb2))
                # If embeddings are identical (same key), cosine should be 1.0
                if lookup_str1 == lookup_str2:
                    cosine_sim = 1.0
            else:
                # Fallback to regular computation if embeddings not cached
                cosine_sim = ds.ranker.compute_cosine_similarity(str1, str2)

    metrics["cosine"] = cosine_sim

    # Always calculate fuzzy cosine similarity
    # If both are numeric, use the same numeric similarity
    if num1 is not None and num2 is not None:
        fuzzy_sim = cosine_sim  # Use the same numeric similarity
    elif cosine_sim == 1.0 or str1.lower() == str2.lower():
        fuzzy_sim = 1.0
    else:
        fuzzy_sim = ds.ranker.compute_fuzzy_cosine_similarity(str1, str2)
        # If fuzzy returns None (both words length 1), use normal cosine
        if fuzzy_sim is None:
            fuzzy_sim = cosine_sim

    metrics["fuzzy_cosine"] = fuzzy_sim

    # Clean string equality check (lowercase, stripped)
    clean_str1 = str1.lower().strip()
    clean_str2 = str2.lower().strip()
    metrics["clean_string_equal"] = (
        1.0 if clean_str1 == clean_str2 and clean_str1 else 0.0
    )

    # Numeric handling (keep numeric_exact and numeric_distance)
    # num1 and num2 are already parsed above using _parse_numeric_value
    # This handles European format (12,5 -> 12.5) and regular format
    metrics["numeric_exact"] = (
        1.0
        if num1 is not None and num2 is not None and num1 == num2
        else 0.0 if num1 is not None and num2 is not None else None
    )

    if num1 is not None and num2 is not None:
        metrics["numeric_distance"] = abs(num1 - num2)
    else:
        metrics["numeric_distance"] = None

    # Optional LLM judge for semantic equivalence (meaning equality)
    # Only call LLM if both values are non-null and cosine similarity is below threshold
    if llm_judge:
        # Check if both values are non-null
        both_non_null = not ds._is_null(val1) and not ds._is_null(val2)
        # Check if cosine similarity is below threshold (0.95)
        cosine_below_threshold = cosine_sim is not None and cosine_sim < 0.95

        if both_non_null and cosine_below_threshold:
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
            # Automatically assign True for null values or high cosine similarity
            metrics["llm_meaning_equal"] = 1.0
    else:
        metrics["llm_meaning_equal"] = None

    return metrics


def _create_comparison_columns(
    df_auto: pd.DataFrame,
    joined: pd.DataFrame,
    join_keys: list[str],
    ignore_columns: Optional[list[str]] = None,
    max_workers: Optional[int] = None,  # pylint: disable=unused-argument
    llm_judge: Optional[Callable[[str, str], float | bool]] = None,
) -> tuple[list[str], pd.DataFrame]:
    """Create comparison columns and return sorted column list."""
    ignore_set = {c.upper() for c in (ignore_columns or [])}
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

    metric_cols: list[str] = []
    column_pairs = []

    for col in df_auto.columns:
        # Skip join keys (we already added them)
        if col in join_keys:
            continue
        if col.upper() in ignore_set:
            continue

        auto_col = f"{col}_auto" if f"{col}_auto" in joined.columns else col
        manual_col = f"{col}_manual" if f"{col}_manual" in joined.columns else None

        if auto_col not in joined.columns:
            continue

        sorted_cols.append(auto_col)
        if manual_col and manual_col in joined.columns:
            sorted_cols.append(manual_col)
            column_pairs.append((col, auto_col, manual_col))

    # Optimized: Batch process by column pairs instead of row-by-row
    # This allows us to encode all unique values once per column pair
    print(
        f"Computing metrics for {len(column_pairs)} column pairs across {len(joined)} rows..."
    )

    row_metric_dicts: list[dict] = [{} for _ in range(len(joined))]
    c = 0
    # Process each column pair separately to batch encode values
    for base_col, a_col, m_col in column_pairs:
        c += 1
        print(f"Processing column pair {c} of {len(column_pairs)}")
        if a_col not in joined.columns or m_col not in joined.columns:
            continue

        # Extract all unique value pairs for this column
        auto_vals = joined[a_col].tolist()
        manual_vals = joined[m_col].tolist()

        # Get unique non-null values for batch encoding
        unique_auto = {ds._safe_str(v) for v in auto_vals if not ds._is_null(v)}
        unique_manual = {ds._safe_str(v) for v in manual_vals if not ds._is_null(v)}
        unique_values = list(unique_auto | unique_manual)

        # Batch encode all unique values at once (much faster) - in parallel
        embeddings_dict = {}
        if unique_values:
            try:
                # Encode in parallel batches to speed up
                batch_size = 100
                ranker = ds.ranker

                # Create batches
                batches = [
                    unique_values[i : i + batch_size]
                    for i in range(0, len(unique_values), batch_size)
                ]

                # Encode batches in parallel (but limit to 4 workers to avoid overwhelming model)
                def encode_batch(batch):
                    return ranker.encode_documents(batch, normalize_embeddings=True)

                with ThreadPoolExecutor(max_workers=min(4, len(batches))) as executor:
                    batch_results = list(executor.map(encode_batch, batches))

                # Combine results
                for batch, batch_embeddings in zip(batches, batch_results):
                    for val, emb in zip(batch, batch_embeddings):
                        embeddings_dict[val] = emb
            except Exception as e:
                print(
                    f"Warning: Batch encoding failed for {base_col}, falling back to row-by-row: {e}"
                )
                embeddings_dict = {}

        # Compute metrics for each row using cached embeddings
        # Get row indices to check similarity scores
        row_indices = list(joined.index)

        for list_idx, (val_a, val_m) in enumerate(zip(auto_vals, manual_vals)):
            # Get the actual dataframe row index
            row_idx = row_indices[list_idx] if list_idx < len(row_indices) else None

            # Check if scenario is unmatched (similarity score == 0 or None)
            is_unmatched = False
            if (
                "_scenario_name_similarity_score" in joined.columns
                and row_idx is not None
            ):
                sim_score = joined.loc[row_idx, "_scenario_name_similarity_score"]
                is_unmatched = pd.isna(sim_score) or sim_score == 0.0
            elif row_idx is not None:
                # If no similarity score column, check if both scenario names exist
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

            if is_unmatched:
                # Set all metrics to None for unmatched scenarios (skip computation)
                for metric_name in [
                    "cosine",
                    "fuzzy_cosine",
                    "clean_string_equal",
                    "numeric_exact",
                    "numeric_distance",
                    "llm_meaning_equal",
                ]:
                    col_name = f"{base_col}_{MATCH_SUFFIX}_{metric_name}"
                    row_metric_dicts[list_idx][col_name] = None
                    if col_name not in metric_cols:
                        metric_cols.append(col_name)
            else:
                # Compute metrics only for matched scenarios
                metrics = _compute_metrics_fast(
                    val_a,
                    val_m,
                    embeddings_dict,
                    llm_judge=llm_judge,
                )
                for metric_name, metric_val in metrics.items():
                    col_name = f"{base_col}_{MATCH_SUFFIX}_{metric_name}"
                    row_metric_dicts[list_idx][col_name] = metric_val
                    if col_name not in metric_cols:
                        metric_cols.append(col_name)

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

    # For each column pair, add: auto_col, manual_col, cosine, fuzzy_cosine, llm_meaning_equal
    for base_col, a_col, m_col in column_pairs:
        if a_col in joined.columns:
            reordered_cols.append(a_col)
        if m_col in joined.columns:
            reordered_cols.append(m_col)

        # Add metrics in order: cosine, fuzzy_cosine, llm_meaning_equal
        cosine_col = f"{base_col}_{MATCH_SUFFIX}_cosine"
        fuzzy_cosine_col = f"{base_col}_{MATCH_SUFFIX}_fuzzy_cosine"
        llm_col = f"{base_col}_{MATCH_SUFFIX}_llm_meaning_equal"

        if cosine_col in joined.columns:
            reordered_cols.append(cosine_col)
        if fuzzy_cosine_col in joined.columns:
            reordered_cols.append(fuzzy_cosine_col)
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


def _calculate_overall_statistics(
    df_joined: pd.DataFrame,
    df_stats: Optional[pd.DataFrame] = None,
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

    # Filter to only matched scenarios: where both scenario names are present
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

    # Get cosine similarity columns
    cosine_cols = [
        c
        for c in df_matched.columns
        if MATCH_SUFFIX in c and "cosine" in c.lower() and "fuzzy" not in c.lower()
    ]

    # Get fuzzy cosine similarity columns
    fuzzy_cosine_cols = [
        c
        for c in df_matched.columns
        if MATCH_SUFFIX in c and "fuzzy_cosine" in c.lower()
    ]

    # Get LLM meaning equal columns
    llm_cols = [
        c
        for c in df_matched.columns
        if MATCH_SUFFIX in c and "llm_meaning_equal" in c.lower()
    ]

    # Calculate cosine similarity stats
    cosine_values = pd.Series(dtype=float)
    for col in cosine_cols:
        values = df_matched[col].dropna().astype(float)
        values = values[(values >= 0) & (values <= 1)]
        cosine_values = pd.concat([cosine_values, values])

    # Calculate fuzzy cosine similarity stats
    fuzzy_cosine_values = pd.Series(dtype=float)
    for col in fuzzy_cosine_cols:
        values = df_matched[col].dropna().astype(float)
        values = values[(values >= 0) & (values <= 1)]
        fuzzy_cosine_values = pd.concat([fuzzy_cosine_values, values])

    # Calculate LLM meaning equal stats (True=1, False=0)
    llm_values = pd.Series(dtype=float)
    for col in llm_cols:
        values = df_matched[col].dropna().astype(float)
        values = values[(values >= 0) & (values <= 1)]
        llm_values = pd.concat([llm_values, values])

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

    overall_stats = {
        "Matched_Scenarios": matched_count,
        "Unmatched_Scenarios": unmatched_count,
        "Cosine_Similarity_Mean": (
            float(cosine_values.mean()) if not cosine_values.empty else None
        ),
        "Cosine_Similarity_Min": (
            float(cosine_values.min()) if not cosine_values.empty else None
        ),
        "Cosine_Similarity_Max": (
            float(cosine_values.max()) if not cosine_values.empty else None
        ),
        "Fuzzy_Cosine_Similarity_Mean": (
            float(fuzzy_cosine_values.mean()) if not fuzzy_cosine_values.empty else None
        ),
        "Fuzzy_Cosine_Similarity_Min": (
            float(fuzzy_cosine_values.min()) if not fuzzy_cosine_values.empty else None
        ),
        "Fuzzy_Cosine_Similarity_Max": (
            float(fuzzy_cosine_values.max()) if not fuzzy_cosine_values.empty else None
        ),
        "LLM_Meaning_Equal_Mean": (
            float(llm_values.mean()) if not llm_values.empty else None
        ),
        "LLM_Meaning_Equal_Min": (
            float(llm_values.min()) if not llm_values.empty else None
        ),
        "LLM_Meaning_Equal_Max": (
            float(llm_values.max()) if not llm_values.empty else None
        ),
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

    # Per field stats - group by field and show mean, min, max for each metric
    # Group columns by their base field name (everything before _match_)
    field_stats = {}

    # Process cosine columns
    for col in cosine_cols:
        # Extract field name: remove "_match_cosine" suffix
        field_name = col.replace(f"_{MATCH_SUFFIX}_cosine", "")
        if field_name not in field_stats:
            field_stats[field_name] = {}
        series = df_matched[col].dropna().astype(float)
        series = series[(series >= 0) & (series <= 1)]
        field_stats[field_name]["cosine mean"] = (
            float(series.mean()) if not series.empty else None
        )
        field_stats[field_name]["cosine min"] = (
            float(series.min()) if not series.empty else None
        )
        field_stats[field_name]["cosine max"] = (
            float(series.max()) if not series.empty else None
        )

    # Process fuzzy cosine columns
    for col in fuzzy_cosine_cols:
        # Extract field name: remove "_match_fuzzy_cosine" suffix
        field_name = col.replace(f"_{MATCH_SUFFIX}_fuzzy_cosine", "")
        if field_name not in field_stats:
            field_stats[field_name] = {}
        series = df_matched[col].dropna().astype(float)
        series = series[(series >= 0) & (series <= 1)]
        field_stats[field_name]["fuzzy mean"] = (
            float(series.mean()) if not series.empty else None
        )
        field_stats[field_name]["fuzzy min"] = (
            float(series.min()) if not series.empty else None
        )
        field_stats[field_name]["fuzzy max"] = (
            float(series.max()) if not series.empty else None
        )

    # Process LLM columns
    for col in llm_cols:
        # Extract field name: remove "_match_llm_meaning_equal" suffix
        field_name = col.replace(f"_{MATCH_SUFFIX}_llm_meaning_equal", "")
        if field_name not in field_stats:
            field_stats[field_name] = {}
        series = df_matched[col].dropna().astype(float)
        series = series[(series >= 0) & (series <= 1)]
        field_stats[field_name]["llm mean"] = (
            float(series.mean()) if not series.empty else None
        )
        field_stats[field_name]["llm min"] = (
            float(series.min()) if not series.empty else None
        )
        field_stats[field_name]["llm max"] = (
            float(series.max()) if not series.empty else None
        )

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


def compare_data(
    df_auto: pd.DataFrame,
    df_manual: pd.DataFrame,
    ranker: DocumentRanker,
    ignore_columns: Optional[list[str]] = None,
    max_workers: Optional[int] = None,
    llm_judge: Optional[Callable[[str, str], float | bool]] = None,
) -> pd.DataFrame:
    """
    Compare auto-extracted data with manual data using fuzzy scenario matching.

    Args:
        df_auto: DataFrame with auto-extracted data
        df_manual: DataFrame with manual data
        ranker: DocumentRanker instance for fuzzy matching
        join_keys: List of column names to join on (should include PID)

    Returns:
        DataFrame with comparison results
    """
    # Remove duplicate columns before processing
    df_auto = df_auto.loc[:, ~df_auto.columns.duplicated()]

    # Identify columns that exist in manual but not in auto
    lost_columns = [x for x in df_manual.columns if x not in df_auto.columns]
    if lost_columns:
        print(f"Columns in manual data but not in auto-extracted: {lost_columns}")

    print("Merging dataframes using fuzzy scenario matching...")
    # Merge using fuzzy scenario matching
    joined = _merge_dataframes_fuzzy(
        df_auto, df_manual, ranker, max_workers=max_workers
    )

    if joined.empty:
        print("Warning: No matching rows found after merge.")
        return joined

    # Remove any duplicate columns that might have been created
    joined = joined.loc[:, ~joined.columns.duplicated()]

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


def save_results_to_excel(
    output_excel_path,
    df_auto,
    df_joined,
    df_stats,
    df_overall=None,
    df_per_field=None,
    df_execution_stats=None,
):
    # Sanitize dataframes before writing to Excel
    _ = df_auto  # currently unused but kept for potential future export
    print(f"Preparing file {output_excel_path}")
    df_joined_sanitized = _sanitize_dataframe_for_excel(df_joined)
    df_stats_sanitized = _sanitize_dataframe_for_excel(df_stats)
    with pd.ExcelWriter(output_excel_path, engine="openpyxl") as writer:
        df_joined_sanitized.to_excel(writer, sheet_name="Joined Data", index=False)
        df_stats_sanitized.to_excel(writer, sheet_name="Statistics", index=False)
        if df_overall is not None and not df_overall.empty:
            _sanitize_dataframe_for_excel(df_overall).to_excel(
                writer, sheet_name="Summary Statistics", index=False
            )
        if df_per_field is not None and not df_per_field.empty:
            _sanitize_dataframe_for_excel(df_per_field).to_excel(
                writer, sheet_name="Field Stats", index=False
            )
        if df_execution_stats is not None and not df_execution_stats.empty:
            _sanitize_dataframe_for_excel(df_execution_stats).to_excel(
                writer, sheet_name="Execution Statistics", index=False
            )
    print(f"Saved results to: {output_excel_path}")

    return None


def compare_files(
    extracted_json_path: str,
    manual_xlsx_path: str,
    manual_sheet_name: str,
    output_excel_path: Optional[str] = None,
    csv_header_row: int | list[int] = None,
    ignore_columns: Optional[list[str]] = None,
    max_workers: Optional[int] = None,
    llm_judge: Optional[Callable[[str, str], float | bool]] = None,
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
