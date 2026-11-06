"""
Deterministic sampler for qualitative mismatch review.

Pulls a bounded, reproducible sample of (paper, scenario, field) cells from
a joined comparison report ("Joined Data" sheet), for handing to a human or
LLM reviewer to verify against the original source papers -- because
aggregate cosine/BERTScore numbers conflate real extraction errors with
ground-truth errors, reviewer-computed values, and pure formatting/
paraphrase differences that score low but aren't actually wrong.

Only one scenario per paper is sampled from, so a single paper's
systematically-repeating error can't dominate the sample and skew the
apparent error-class distribution.

Three mismatch shapes are pooled separately, then sampled evenly:
  - type1_gt_empty_auto_filled: ground truth is null, auto extracted a value
  - type2_auto_empty_gt_filled: auto is null, ground truth has a value
  - type3_low_cosine: both sides have a value, but the per-cell cosine
    (the "<field>__match_cosine" column already present in Joined Data)
    is below `cosine_threshold`
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field as dataclass_field
from typing import Optional

import pandas as pd

METADATA_COLS = {"PID", "SCENARIO NAME", "EXCLUDE_FROM_ACCURACY", "EXCLUSION_REASON", "PDF_PATH"}

TYPE1_GT_EMPTY_AUTO_FILLED = "type1_gt_empty_auto_filled"
TYPE2_AUTO_EMPTY_GT_FILLED = "type2_auto_empty_gt_filled"
TYPE3_LOW_COSINE = "type3_low_cosine"
MISMATCH_TYPES = (TYPE1_GT_EMPTY_AUTO_FILLED, TYPE2_AUTO_EMPTY_GT_FILLED, TYPE3_LOW_COSINE)


@dataclass
class MismatchSamplerConfig:
    n_per_type: int = 3
    seed: int = 20260917
    cosine_threshold: float = 0.15
    # Fields to skip entirely (e.g. confirmed reviewer-computed, not
    # paper-derivable -- already documented elsewhere, no need to re-sample).
    excluded_fields_always: set = dataclass_field(default_factory=set)
    # Fields to skip only for the type3 (low-cosine) pool -- e.g. a
    # bibliographic reference field known to score low purely from
    # citation-style formatting differences, not content disagreement.
    excluded_fields_type3_only: set = dataclass_field(default_factory=set)


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


def find_mismatch_candidates(
    df_joined: pd.DataFrame,
    domain: str,
    pdf_dir: str,
    config: Optional[MismatchSamplerConfig] = None,
) -> dict:
    """
    Pick one matched scenario per paper (seeded random choice) and scan its
    fields for the three mismatch shapes. Returns the full candidate pools
    per shape -- call `sample_from_candidates` to draw the bounded sample.
    """
    config = config or MismatchSamplerConfig()
    rng = random.Random(config.seed)

    exclude_col = (
        df_joined["EXCLUDE_FROM_ACCURACY"]
        if "EXCLUDE_FROM_ACCURACY" in df_joined.columns
        else pd.Series(False, index=df_joined.index)
    )
    matched = df_joined[
        df_joined["SCENARIO NAME"].notna()
        & df_joined["SCENARIO NAME_manual"].notna()
        & ~exclude_col.fillna(False).astype(bool)
    ].copy()

    chosen_rows = []
    for _pid, group in matched.groupby("PID"):
        idx = rng.randrange(len(group))
        chosen_rows.append(group.iloc[idx])

    data_fields = [c for c in df_joined.columns if _is_data_field(c, config.excluded_fields_always)]

    candidates = {t: [] for t in MISMATCH_TYPES}
    for row in chosen_rows:
        pid = row["PID"]
        for f in data_fields:
            auto_val = row.get(f)
            manual_val = row.get(f"{f}_manual")
            cosine_val = row.get(f"{f}__match_cosine")
            auto_empty, manual_empty = _is_empty(auto_val), _is_empty(manual_val)
            rec = dict(
                domain=domain,
                pid=pid,
                scenario=row.get("SCENARIO NAME"),
                scenario_manual=row.get("SCENARIO NAME_manual"),
                field=f,
                auto_value=auto_val,
                manual_value=manual_val,
                cosine=cosine_val,
                pdf=f"{pdf_dir}/{pid}.pdf",
            )
            if manual_empty and not auto_empty:
                candidates[TYPE1_GT_EMPTY_AUTO_FILLED].append(rec)
            elif auto_empty and not manual_empty:
                candidates[TYPE2_AUTO_EMPTY_GT_FILLED].append(rec)
            elif (
                not auto_empty
                and not manual_empty
                and f not in config.excluded_fields_type3_only
                and cosine_val is not None
                and not pd.isna(cosine_val)
                and cosine_val < config.cosine_threshold
            ):
                candidates[TYPE3_LOW_COSINE].append(rec)
    return candidates


def sample_from_candidates(
    candidates: dict,
    config: Optional[MismatchSamplerConfig] = None,
) -> list:
    """
    Draw up to `n_per_type` examples from each mismatch-type pool, capped
    at one field per paper per type (so one paper's repeated bug doesn't
    fill the whole sample for that type).
    """
    config = config or MismatchSamplerConfig()
    rng = random.Random(config.seed)
    sample = []
    for type_key in MISMATCH_TYPES:
        rows = candidates.get(type_key, [])
        seen_pid, deduped = set(), []
        shuffled = list(rows)
        rng.shuffle(shuffled)
        for r in shuffled:
            if r["pid"] not in seen_pid:
                deduped.append(r)
                seen_pid.add(r["pid"])
        picked = rng.sample(deduped, min(config.n_per_type, len(deduped)))
        for r in picked:
            r = dict(r)
            r["error_type"] = type_key
            sample.append(r)
    return sample


def sample_for_target_fields(
    df_joined: pd.DataFrame,
    domain: str,
    pdf_dir: str,
    target_fields: list,
    n_per_field: int = 3,
    seed: int = 20260917,
    cosine_threshold: float = 0.15,
) -> list:
    """
    Targeted variant of the sampler: instead of a broad random sweep across
    all fields, draw examples for a specific list of fields (typically the
    worst performers from field_decomposition.worst_fields), so a bounded
    PDF-verification budget goes to the fields actually dragging the
    aggregate down. Still capped at one example per (field, paper) so a
    single paper doesn't dominate a field's sample.

    Each field's pool mixes whichever mismatch shapes it actually has
    (GT-empty/auto-filled, auto-empty/GT-filled, or both-filled-low-cosine)
    rather than forcing an even split across shapes -- some fields simply
    don't have all three shapes present.
    """
    rng = random.Random(seed)
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

    sample = []
    for f in target_fields:
        if f not in matched.columns:
            continue
        manual_col = f"{f}_manual"
        cosine_col = f"{f}__match_cosine"
        pool = []
        for _, row in matched.iterrows():
            auto_val = row.get(f)
            manual_val = row.get(manual_col)
            cosine_val = row.get(cosine_col)
            auto_empty, manual_empty = _is_empty(auto_val), _is_empty(manual_val)
            rec = dict(
                domain=domain,
                pid=row["PID"],
                scenario=row.get("SCENARIO NAME"),
                scenario_manual=row.get("SCENARIO NAME_manual"),
                field=f,
                auto_value=auto_val,
                manual_value=manual_val,
                cosine=cosine_val,
            )
            if manual_empty and not auto_empty:
                rec["error_type"] = TYPE1_GT_EMPTY_AUTO_FILLED
            elif auto_empty and not manual_empty:
                rec["error_type"] = TYPE2_AUTO_EMPTY_GT_FILLED
            elif (
                not auto_empty
                and not manual_empty
                and cosine_val is not None
                and not pd.isna(cosine_val)
                and cosine_val < cosine_threshold
            ):
                rec["error_type"] = TYPE3_LOW_COSINE
            else:
                continue
            rec["pdf"] = f"{pdf_dir}/{rec['pid']}.pdf"
            pool.append(rec)

        seen_pid, deduped = set(), []
        rng.shuffle(pool)
        for r in pool:
            if r["pid"] not in seen_pid:
                deduped.append(r)
                seen_pid.add(r["pid"])
        picked = rng.sample(deduped, min(n_per_field, len(deduped)))
        sample.extend(picked)
    return sample


def sample_mismatches(
    df_joined: pd.DataFrame,
    domain: str,
    pdf_dir: str,
    config: Optional[MismatchSamplerConfig] = None,
) -> tuple:
    """Convenience wrapper: find candidates then sample. Returns (sample, candidate_pools)."""
    candidates = find_mismatch_candidates(df_joined, domain, pdf_dir, config)
    sample = sample_from_candidates(candidates, config)
    return sample, candidates
