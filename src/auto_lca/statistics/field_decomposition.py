"""
Per-field error decomposition: splits a field's aggregate cosine score
into WHY it's low, using data already computed in a joined comparison
report ("Joined Data" sheet) -- no extraction, no API calls, pure
aggregation. Complements mismatch_sampler.py, which does the expensive
part (verifying a bounded sample against source papers).

For each field, over matched scenario rows only:
  - both_empty:            neither side has a value
  - gt_empty_auto_filled:  GT is null, auto filled something in (presence)
  - auto_empty_gt_filled:  auto is null, GT has a value (presence)
  - both_filled:           both sides have a value -- the only rows where
                            a content comparison is even meaningful
  - content_mismatch:      among both_filled, per-cell cosine < threshold

A field with a low aggregate cosine but a high both_filled rate and low
content_mismatch rate is failing on PRESENCE (nulls in the wrong place);
a field with a high content_mismatch rate among both_filled rows is
failing on CONTENT (values disagree even when both sides tried) -- very
different fixes.
"""

from __future__ import annotations

from typing import Optional

import pandas as pd

METADATA_COLS = {"PID", "SCENARIO NAME", "EXCLUDE_FROM_ACCURACY", "EXCLUSION_REASON", "PDF_PATH"}


def _is_data_field(col: str, excluded_always: set) -> bool:
    if col.startswith("_") or col.startswith("SCENARIO_"):
        return False
    if col.endswith("_manual") or "__match_" in col:
        return False
    if col.upper() in METADATA_COLS or col.upper().startswith("METADATA"):
        return False
    if col in excluded_always:
        return False
    return True


def _is_empty(v) -> bool:
    if v is None:
        return True
    if isinstance(v, float) and pd.isna(v):
        return True
    return str(v).strip().lower() in {"", "none", "nan", "null"}


def build_field_error_decomposition(
    df_joined: pd.DataFrame,
    cosine_threshold: float = 0.15,
    excluded_fields_always: Optional[set] = None,
) -> pd.DataFrame:
    """
    One row per field, decomposing where its score is coming from.
    Only considers matched scenario rows (both an auto and a manual
    scenario present, not flagged EXCLUDE_FROM_ACCURACY).
    """
    excluded_fields_always = excluded_fields_always or set()

    exclude_col = (
        df_joined["EXCLUDE_FROM_ACCURACY"]
        if "EXCLUDE_FROM_ACCURACY" in df_joined.columns
        else pd.Series(False, index=df_joined.index)
    )
    matched = df_joined[
        df_joined["SCENARIO NAME"].notna()
        & df_joined["SCENARIO NAME_manual"].notna()
        & ~exclude_col.fillna(False).astype(bool)
    ]
    n_matched = len(matched)
    if n_matched == 0:
        return pd.DataFrame()

    data_fields = [c for c in df_joined.columns if _is_data_field(c, excluded_fields_always)]

    rows = []
    for f in data_fields:
        auto_col = matched[f] if f in matched.columns else pd.Series([None] * n_matched)
        manual_col = matched.get(f"{f}_manual", pd.Series([None] * n_matched))
        cosine_col = matched.get(f"{f}__match_cosine")

        auto_empty = auto_col.map(_is_empty)
        manual_empty = manual_col.map(_is_empty)

        both_empty = int((auto_empty & manual_empty).sum())
        gt_empty_auto_filled = int((manual_empty & ~auto_empty).sum())
        auto_empty_gt_filled = int((auto_empty & ~manual_empty).sum())
        both_filled_mask = ~auto_empty & ~manual_empty
        both_filled = int(both_filled_mask.sum())

        content_mismatch = None
        mean_cosine_both_filled = None
        if both_filled and cosine_col is not None:
            filled_cosine = pd.to_numeric(cosine_col[both_filled_mask], errors="coerce")
            content_mismatch = int((filled_cosine < cosine_threshold).sum())
            mean_cosine_both_filled = float(filled_cosine.mean()) if filled_cosine.notna().any() else None

        rows.append(
            {
                "Field": f,
                "N matched rows": n_matched,
                "Both empty": both_empty,
                "Both empty %": round(100 * both_empty / n_matched, 2),
                "GT empty, auto filled (presence)": gt_empty_auto_filled,
                "GT empty, auto filled %": round(100 * gt_empty_auto_filled / n_matched, 2),
                "Auto empty, GT filled (presence)": auto_empty_gt_filled,
                "Auto empty, GT filled %": round(100 * auto_empty_gt_filled / n_matched, 2),
                "Both filled (content-comparable)": both_filled,
                "Both filled %": round(100 * both_filled / n_matched, 2),
                "Content mismatch (cosine < threshold, of both-filled)": content_mismatch,
                "Content mismatch % (of both-filled)": (
                    round(100 * content_mismatch / both_filled, 2)
                    if both_filled and content_mismatch is not None
                    else None
                ),
                "Mean cosine (both-filled only)": (
                    round(mean_cosine_both_filled, 4) if mean_cosine_both_filled is not None else None
                ),
                "Presence mismatch % (of all matched)": round(
                    100 * (gt_empty_auto_filled + auto_empty_gt_filled) / n_matched, 2
                ),
            }
        )

    df = pd.DataFrame(rows)
    return df.sort_values("Mean cosine (both-filled only)", na_position="first").reset_index(drop=True)


def worst_fields(
    df_decomposition: pd.DataFrame,
    n: int = 10,
    min_both_filled: int = 3,
) -> list:
    """
    Fields worth spending a verification budget on: enough both-filled
    (content-comparable) rows to matter, ranked by lowest mean cosine
    among those rows -- i.e. worst CONTENT agreement, not just low fill
    rate (a field that's simply rarely filled by both sides isn't
    necessarily a content-quality problem).
    """
    candidates = df_decomposition[
        df_decomposition["Both filled (content-comparable)"] >= min_both_filled
    ].copy()
    candidates = candidates.sort_values("Mean cosine (both-filled only)")
    return candidates["Field"].head(n).tolist()
