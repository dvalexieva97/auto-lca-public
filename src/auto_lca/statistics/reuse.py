"""Reuse matching and LLM-judge verdicts from an earlier comparison run.

Re-running a comparison calls two LLMs again: the scenario matcher and the
meaning judge. Both are slightly random even with pinned models, so papers
whose inputs did not change would still shift. With ``reuse_from`` set to an
earlier joined workbook:

- a paper whose extracted system names and reference system names equal those
  in the earlier run gets that run's pairings, collapses and reasons back;
  only papers whose names changed are matched live;
- every (extracted value, reference value) pair judged in the earlier run
  keeps its verdict; only new pairs go to the judge.
"""

from __future__ import annotations

from collections import defaultdict

import pandas as pd

from auto_lca.statistics.scenario_mapper import PaperMapping
from auto_lca.statistics.similarity import DataSimilarity

LLM_SUFFIX = "__match_llm_meaning_equal"


def _text(value) -> str:
    return "" if DataSimilarity._is_null(value) else str(value).strip()


def _truthy(value) -> bool:
    return bool(value) and str(value).strip().lower() not in {"false", "0", "nan", "none"}


class ReplayScenarioMapper:
    """Wraps the live matcher; replays papers whose names are unchanged."""

    def __init__(self, live_mapper, scenario_matching: pd.DataFrame):
        self.live = live_mapper
        self.recorded: dict[tuple[frozenset, frozenset], PaperMapping] = {}
        self.replayed: list[str] = []
        self.matched_live: list[str] = []
        self._pid_by_key: dict[tuple[frozenset, frozenset], str] = {}
        self.not_replayable: list[str] = []
        for pid, rows in scenario_matching.groupby("PID"):
            if self._lost_pairs(rows):
                # The earlier run paired a system but its join dropped the pair
                # (reference name with stray spaces); re-match this paper live.
                self.not_replayable.append(str(pid))
                continue
            key, mapping = self._record(rows)
            self.recorded[key] = mapping
            self._pid_by_key[key] = str(pid)

    @staticmethod
    def _lost_pairs(rows: pd.DataFrame) -> bool:
        auto = rows[rows["SCENARIO NAME"].notna()]
        conf = pd.to_numeric(auto["AI confidence"], errors="coerce").fillna(0)
        return bool(((conf > 0) & auto["Manual Scenario"].isna()
                     & auto["Collapsed into"].isna()).any())

    @staticmethod
    def _record(rows: pd.DataFrame) -> tuple[tuple[frozenset, frozenset], PaperMapping]:
        auto_rows = rows[rows["SCENARIO NAME"].notna()]
        gt_only = rows[rows["SCENARIO NAME"].isna() & rows["Manual Scenario"].notna()]
        gt_only = gt_only[~gt_only["Exclude from accuracy"].map(_truthy)]
        auto_names = frozenset(_text(n) for n in auto_rows["SCENARIO NAME"])
        gt_names = frozenset(
            _text(n) for n in pd.concat([auto_rows["Manual Scenario"], gt_only["Manual Scenario"]])
            if _text(n)
        )
        mapping = PaperMapping()
        for _, r in auto_rows.iterrows():
            name, manual = _text(r["SCENARIO NAME"]), _text(r["Manual Scenario"])
            reason = _text(r.get("AI reason"))
            collapsed_into = _text(r.get("Collapsed into"))
            confidence = pd.to_numeric(r.get("AI confidence"), errors="coerce")
            if collapsed_into:
                mapping.collapsed[name] = (collapsed_into, reason)
            elif manual and pd.notna(confidence) and confidence > 0:
                mapping.pairs[name] = (manual, float(confidence), reason)
            else:
                mapping.unmatched_extracted[name] = reason or "unmatched"
        return (auto_names, gt_names), mapping

    def map_paper(self, extracted, gt_rows, **kwargs) -> PaperMapping:
        key = (frozenset(_text(c["name"]) for c in extracted),
               frozenset(_text(c["name"]) for c in gt_rows))
        if key in self.recorded:
            self.replayed.append(self._pid_by_key[key])
            return self.recorded[key]
        self.matched_live.append(", ".join(sorted(key[0]))[:80])
        return self.live.map_paper(extracted, gt_rows, **kwargs)


def judge_verdicts(joined: pd.DataFrame) -> dict[tuple[str, str], float]:
    """(extracted value, reference value) -> verdict from an earlier Joined Data sheet."""
    verdicts: dict[tuple[str, str], float] = {}
    for col in joined.columns:
        if not col.endswith(LLM_SUFFIX):
            continue
        field = col[: -len(LLM_SUFFIX)]
        auto = field if field in joined.columns else f"{field}_auto"
        manual = f"{field}_manual"
        if auto not in joined.columns or manual not in joined.columns:
            continue
        for a, m, v in joined[[auto, manual, col]].itertuples(index=False):
            if pd.isna(v) or DataSimilarity._is_null(a) or DataSimilarity._is_null(m):
                continue
            verdicts[(DataSimilarity._safe_str(a), DataSimilarity._safe_str(m))] = float(v)
    return verdicts


def load_previous_run(path: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    return (pd.read_excel(path, sheet_name="Scenario Matching"),
            pd.read_excel(path, sheet_name="Joined Data"))


def summary(mapper: ReplayScenarioMapper | None, judge) -> dict:
    out = defaultdict(int)
    if mapper is not None:
        out["papers_replayed"] = len(mapper.replayed)
        out["papers_matched_live"] = len(mapper.matched_live)
        out["papers_not_replayable"] = len(mapper.not_replayable)
    stats = getattr(judge, "cache_stats", None)
    if stats:
        out.update({f"judge_{k}": v for k, v in stats.items()})
    return dict(out)
