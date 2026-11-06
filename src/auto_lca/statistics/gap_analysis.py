"""
Formalizes the ad-hoc "diff system output vs. manual ground truth" analysis
used repeatedly this session (dairy 40-paper quality audit, chemicals/
plastics ground-truth enrichment) into a reusable function, instead of a
one-off inline script every time.

Given a comparison pipeline's "Joined Data" sheet (produced by
sog_runner.run_comparison / compare.compare_data), for every MATCHED
scenario and every field pair (SYSTEM value / MANUAL value), classifies
each pair into one of:
  - "system_only": system has a value, manual is null -- candidate gaps
    to backfill into the ground truth (this is what agent_sog's enrich
    mode consumes).
  - "manual_only": manual has a value, system is null -- a real
    extraction miss (useful for auditing the pipeline, not for
    enrichment).
  - "mismatch": both have a value but disagree -- needs a human/agent
    judgment call, not a blind fill.
Null pairs (both empty) are skipped entirely.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

NULL_KEYWORDS = {
    "null",
    "none",
    "missing",
    "missing value",
    "nan",
    "",
    "nr in accessible text",
    "units not stated",  # SYSTEM_EXTRACT_CONCEPT's explicit fallback text for an unknown unit
    "not stated",
    "not explicitly stated",
    "not applicable",
    "n/a",
}


def is_null(value) -> bool:
    if value is None:
        return True
    if isinstance(value, float) and np.isnan(value):
        return True
    return str(value).strip().lower() in NULL_KEYWORDS


@dataclass
class FieldGap:
    pid: str
    scenario_system: str
    scenario_manual: str
    concept: str
    field: str
    system_value: object
    manual_value: object
    category: str  # "system_only" | "manual_only" | "mismatch"


@dataclass
class GapReport:
    gaps: list[FieldGap] = field(default_factory=list)

    def by_category(self, category: str) -> list[FieldGap]:
        return [g for g in self.gaps if g.category == category]

    def by_pid(self) -> dict[str, list[FieldGap]]:
        out: dict[str, list[FieldGap]] = {}
        for g in self.gaps:
            out.setdefault(g.pid, []).append(g)
        return out

    def summary(self) -> dict[str, int]:
        counts = {"system_only": 0, "manual_only": 0, "mismatch": 0}
        for g in self.gaps:
            counts[g.category] += 1
        return counts


def _split_field_col(col: str) -> tuple[str, str]:
    """'CONCEPT NAME_FIELD NAME' -> ('CONCEPT NAME', 'FIELD NAME')."""
    concept, _, fname = col.partition("_")
    return concept, fname


def find_gaps(
    df_joined: pd.DataFrame,
    matched_only: bool = True,
    ignore_columns: set[str] | None = None,
) -> GapReport:
    """
    df_joined: the "Joined Data" sheet from a run_comparison() output
               (or any dataframe with the same PID / SCENARIO NAME /
               SCENARIO NAME_manual / <FIELD> / <FIELD>_manual shape).
    matched_only: only consider rows with a non-empty SCENARIO NAME_manual
               (i.e. rows the fuzzy scenario matcher actually paired up).
               Gaps on unmatched rows aren't meaningful -- there's no
               manual counterpart to be missing a value in the first place.
    """
    ignore_columns = ignore_columns or set()
    df = df_joined
    if matched_only:
        df = df[
            df["SCENARIO NAME_manual"].notna()
            & (df["SCENARIO NAME_manual"].astype(str).str.strip() != "")
        ]

    manual_cols = [
        c
        for c in df.columns
        if c.endswith("_manual") and c != "SCENARIO NAME_manual" and c not in ignore_columns
    ]
    base_fields = [c[: -len("_manual")] for c in manual_cols]

    report = GapReport()
    for _, row in df.iterrows():
        pid = row.get("PID")
        scenario_system = row.get("SCENARIO NAME")
        scenario_manual = row.get("SCENARIO NAME_manual")
        for f in base_fields:
            sysval = row.get(f)
            manval = row.get(f + "_manual")
            sys_null = is_null(sysval)
            man_null = is_null(manval)
            if sys_null and man_null:
                continue
            concept, fname = _split_field_col(f)
            if not sys_null and man_null:
                category = "system_only"
            elif sys_null and not man_null:
                category = "manual_only"
            else:
                match_flag = row.get(f + "__match_fuzzy_threshold_equal")
                if pd.notna(match_flag) and float(match_flag) == 1.0:
                    continue  # already agree, not a gap
                category = "mismatch"
            report.gaps.append(
                FieldGap(
                    pid=pid,
                    scenario_system=scenario_system,
                    scenario_manual=scenario_manual,
                    concept=concept,
                    field=fname,
                    system_value=sysval,
                    manual_value=manval,
                    category=category,
                )
            )
    return report
