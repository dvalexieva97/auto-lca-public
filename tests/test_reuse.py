"""Reusing an earlier comparison run: matching replay and judge verdicts."""

import pandas as pd

from auto_lca.statistics.compare import _normalize_dataframe
from auto_lca.statistics.reuse import ReplayScenarioMapper, judge_verdicts
from auto_lca.statistics.scenario_mapper import PaperMapping

MATCHING = pd.DataFrame([
    # paper p1: Farm A paired, Farm B collapsed into Average, Farm C unmatched
    {"PID": "p1", "SCENARIO NAME": "Farm A", "Manual Scenario": "Farm A ref",
     "AI confidence": 0.9, "AI reason": "same farm", "Collapsed into": None,
     "Exclude from accuracy": False},
    {"PID": "p1", "SCENARIO NAME": "Farm B", "Manual Scenario": "Average",
     "AI confidence": 0.0, "AI reason": "variant", "Collapsed into": "Average",
     "Exclude from accuracy": True},
    {"PID": "p1", "SCENARIO NAME": "Farm C", "Manual Scenario": None,
     "AI confidence": 0.0, "AI reason": "no row", "Collapsed into": None,
     "Exclude from accuracy": False},
    {"PID": "p1", "SCENARIO NAME": None, "Manual Scenario": "Average",
     "AI confidence": None, "AI reason": None, "Collapsed into": None,
     "Exclude from accuracy": False},
    # paper p2: the earlier join dropped a pair the matcher made
    {"PID": "p2", "SCENARIO NAME": "Site 1", "Manual Scenario": None,
     "AI confidence": 1.0, "AI reason": "exact title", "Collapsed into": None,
     "Exclude from accuracy": False},
])


class LiveMapper:
    def __init__(self):
        self.calls = 0

    def map_paper(self, extracted, gt_rows, **kwargs):
        self.calls += 1
        return PaperMapping(pairs={"live": ("live", 1.0, "live")})


def _cards(names):
    return [{"id": f"E{i}", "name": n} for i, n in enumerate(names)]


def test_unchanged_paper_is_replayed():
    live = LiveMapper()
    mapper = ReplayScenarioMapper(live, MATCHING)
    result = mapper.map_paper(_cards(["Farm A", "Farm B", "Farm C"]),
                              _cards(["Farm A ref", "Average"]))
    assert live.calls == 0
    assert result.pairs == {"Farm A": ("Farm A ref", 0.9, "same farm")}
    assert result.collapsed == {"Farm B": ("Average", "variant")}
    assert result.unmatched_extracted == {"Farm C": "no row"}
    assert mapper.replayed == ["p1"]


def test_changed_reference_names_are_matched_live():
    live = LiveMapper()
    mapper = ReplayScenarioMapper(live, MATCHING)
    mapper.map_paper(_cards(["Farm A", "Farm B", "Farm C"]),
                     _cards(["Farm A ref", "Average", "Farm D ref"]))
    assert live.calls == 1


def test_record_with_a_lost_pair_is_not_replayed():
    live = LiveMapper()
    mapper = ReplayScenarioMapper(live, MATCHING)
    assert mapper.not_replayable == ["p2"]
    mapper.map_paper(_cards(["Site 1"]), _cards(["Site 1 ref"]))
    assert live.calls == 1


def test_judge_verdicts_from_joined_data():
    joined = pd.DataFrame({
        "F_X": ["IPCC Tier 2", "GaBi", None],
        "F_X_manual": ["IPCC 2006 Tier 2", "SimaPro", "x"],
        "F_X__match_llm_meaning_equal": [1.0, 0.0, None],
    })
    assert judge_verdicts(joined) == {("IPCC Tier 2", "IPCC 2006 Tier 2"): 1.0,
                                      ("GaBi", "SimaPro"): 0.0}


def test_system_names_are_stripped_before_joining():
    df = pd.DataFrame({"SCENARIO NAME": ["Vacas Felizes - Average ", " (case 3) Fermoy"]})
    assert _normalize_dataframe(df)["SCENARIO NAME"].tolist() == [
        "Vacas Felizes - Average", "(case 3) Fermoy"]
