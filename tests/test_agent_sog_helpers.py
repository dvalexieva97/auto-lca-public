from types import SimpleNamespace

import pytest

from auto_lca.process.sog.agent_sog import DEFAULT_MODEL, AgentExtractor, _to_litellm_model
from auto_lca.process.sog.concept import JsonObjectConcept

A = AgentExtractor
MODELLED = A.SCENARIO_CATEGORY_OWN_SYSTEM_MODELLED
MENTIONED = A.SCENARIO_CATEGORY_OWN_SYSTEM_MENTIONED
EXTERNAL = A.SCENARIO_CATEGORY_EXTERNAL_REFERENCE
BREAKDOWN = A.SCENARIO_CATEGORY_SUBCATEGORY_BREAKDOWN
ALTERNATE = A.SCENARIO_CATEGORY_ALTERNATE_DIMENSION
UNCLASSIFIED = A.SCENARIO_CATEGORY_UNCLASSIFIED


def _concept(name, n_fields, by_scenario=True):
    return JsonObjectConcept(
        name=name, description=name, by_scenario=by_scenario,
        structure={f"{name}_f{i}": "str" for i in range(n_fields)},
    )


# --- models -------------------------------------------------------------------


def test_default_model_is_pinned():
    assert DEFAULT_MODEL == "mistral/mistral-large-2512"


@pytest.mark.parametrize("name,expected", [
    ("mistral-large-2512", "mistral/mistral-large-2512"),
    ("mistral/mistral-small-2603", "mistral/mistral-small-2603"),
    ("openai/gpt-5.4-mini", "openai/gpt-5.4-mini"),
])
def test_to_litellm_model(name, expected):
    assert _to_litellm_model(name) == expected


def test_models_come_from_config(make_extractor):
    ex = make_extractor()
    assert ex.model == "mistral/mistral-small-test"
    assert ex.scenario_model == "mistral/mistral-large-test"


def test_legacy_scenario_model_key(make_extractor):
    ex = make_extractor({"production_system_model": None, "scenario_model": "legacy-large"})
    assert ex.scenario_model == "mistral/legacy-large"


def test_scenario_model_falls_back_to_model(make_extractor):
    ex = make_extractor({"production_system_model": None})
    assert ex.scenario_model == ex.model


def test_constructor_overrides_config_models(make_extractor):
    ex = make_extractor(model="openai/a", scenario_model="openai/b")
    assert (ex.model, ex.scenario_model) == ("openai/a", "openai/b")


def test_accepts_csv_config(tmp_path):
    csv = tmp_path / "c.csv"
    csv.write_text("field_set,description,by_production_system,field,data_type\n"
                   "Goal,Setup,FALSE,FU,str\n")
    ex = AgentExtractor(config_path=str(csv))
    assert ex.config_path.endswith("c.json")
    assert [c["name"] for c in ex.config["field_sets"]] == ["Goal"]


# --- config flags -------------------------------------------------------------


@pytest.mark.parametrize("defaults,expected", [
    ({}, False),
    ({"extract_non_primary": True}, True),
    ({"extract_non_primary_production_systems": True}, True),
    ({"extract_non_primary_production_systems": False, "extract_external_scenarios": True}, False),
    ({"extract_external_production_systems": True}, True),
    ({"extract_external_scenarios": True}, True),
    ({"include_aggregated_production_systems": True}, True),
    ({"include_aggregated_scenarios": True}, True),
])
def test_resolve_extract_non_primary(defaults, expected):
    assert A._resolve_extract_non_primary(defaults) is expected


def test_extract_production_systems_flag(make_extractor):
    assert make_extractor().extract_scenarios is True
    assert make_extractor({"extract_production_systems": False}).extract_scenarios is False
    assert make_extractor({"extract_scenarios": False}).extract_scenarios is False


# --- tool schema -------------------------------------------------------------


@pytest.mark.parametrize("type_str,expected_type,has_max_len", [
    ("str", "string", True),
    ("str | None", ["string", "null"], True),
    ("int", "integer", False),
    ("float | None", ["number", "null"], False),
    ("unknown type", ["string", "null"], True),
])
def test_field_schema(type_str, expected_type, has_max_len):
    schema = A._field_schema(type_str)
    assert schema["type"] == expected_type
    assert ("maxLength" in schema) is has_max_len
    if has_max_len:
        assert schema["maxLength"] == A.MAX_STR_FIELD_LENGTH


def test_field_schema_does_not_mutate_shared_table():
    A._field_schema("str")
    assert "maxLength" not in A._TYPE_JSON_SCHEMA["str"]


def test_concepts_tool_required_follows_config_types():
    concept = JsonObjectConcept(
        name="Inventory", description="d", by_scenario=True,
        structure={"Feed": "str", "Milk": "float | None"},
    )
    params = A._concepts_tool([concept])["function"]["parameters"]
    assert params["required"] == ["Inventory"]
    inv = params["properties"]["Inventory"]
    assert inv["required"] == ["Feed"]
    assert set(inv["properties"]) == {"Feed", "Milk"}


def test_concepts_tool_all_optional_fields_has_no_required_list():
    concept = JsonObjectConcept(name="C", description="d", by_scenario=True,
                                structure={"x": "str | None"})
    props = A._concepts_tool([concept])["function"]["parameters"]["properties"]
    assert "required" not in props["C"]


# --- chunking -----------------------------------------------------------------


def test_chunks_respect_field_budget():
    budget = A.CONCEPT_CHUNK_FIELD_BUDGET
    concepts = [_concept("a", budget - 5), _concept("b", 5), _concept("c", 1)]
    chunks = A._chunk_concepts(concepts)
    assert [[c.name for c in ch] for ch in chunks] == [["a", "b"], ["c"]]


def test_oversized_concept_gets_its_own_chunk():
    budget = A.CONCEPT_CHUNK_FIELD_BUDGET
    chunks = A._chunk_concepts([_concept("small", 1), _concept("huge", budget + 10)])
    assert [[c.name for c in ch] for ch in chunks] == [["small"], ["huge"]]


def test_no_concepts_no_chunks():
    assert A._chunk_concepts([]) == []


# --- categories --------------------------------------------------------------


def _names(scenarios):
    return [s["name"] for s in scenarios]


def test_partition_default_only_modelled_systems_get_fields(make_extractor):
    ex = make_extractor()
    cats = [MODELLED, MENTIONED, EXTERNAL, BREAKDOWN, ALTERNATE, UNCLASSIFIED]
    scenarios = [{"name": c, "scenario_category": c} for c in cats]
    extractable, shells = ex._partition_scenarios(scenarios)
    assert _names(extractable) == [MODELLED]
    assert _names(shells) == [MENTIONED, EXTERNAL, BREAKDOWN, ALTERNATE, UNCLASSIFIED]


def test_partition_with_extract_non_primary(make_extractor):
    ex = make_extractor({"extract_non_primary": True})
    cats = [MODELLED, MENTIONED, EXTERNAL, BREAKDOWN, ALTERNATE, UNCLASSIFIED]
    extractable, shells = ex._partition_scenarios(
        [{"name": c, "scenario_category": c} for c in cats])
    assert _names(extractable) == [MODELLED, EXTERNAL, ALTERNATE, UNCLASSIFIED]
    assert _names(shells) == [MENTIONED, BREAKDOWN]


def test_definition_only_categories(make_extractor):
    assert make_extractor()._definition_only_categories() == {MENTIONED, BREAKDOWN, EXTERNAL}
    # Alternate-dimension shells stay match-eligible.
    assert ALTERNATE not in make_extractor()._definition_only_categories()
    assert make_extractor({"extract_non_primary": True})._definition_only_categories() == {
        MENTIONED, BREAKDOWN}


def test_scenario_without_category_is_extracted(make_extractor):
    extractable, shells = make_extractor()._partition_scenarios([{"name": "x"}])
    assert _names(extractable) == ["x"] and shells == []


# --- grouping dimensions -----------------------------------------------------


def test_merge_dimension_aliases_by_word_subset():
    # Real case from a dairy paper: the same regional axis tagged two ways.
    alias = A._merge_dimension_aliases(
        {"region + soil drainage variant", "soil drainage variant", "feed type"})
    assert alias["region + soil drainage variant"] == "soil drainage variant"
    assert alias["soil drainage variant"] == "soil drainage variant"
    assert alias["feed type"] == "feed type"


def test_merged_aliases_count_as_one_dimension(make_extractor):
    kept = [_cand("A", "soil drainage variant"), _cand("B", "region + soil drainage variant")]
    result = make_extractor()._retag_minority_dimensions(kept)
    assert [r["scenario_category"] for r in result] == [MODELLED, MODELLED]


def _cand(name, dim, category=MODELLED):
    return {"name": name, "scenario_category": category, "_grouping_dimension": dim}


def test_minority_dimension_is_retagged_not_dropped(make_extractor):
    kept = [_cand("Farm A", "farm"), _cand("Farm B", "farm"), _cand("Farm C", "farm"),
            _cand("Low 10%", "percentile distribution")]
    result = make_extractor()._retag_minority_dimensions(kept)
    assert _names(result) == ["Farm A", "Farm B", "Farm C", "Low 10%"]
    assert [r["scenario_category"] for r in result] == [MODELLED, MODELLED, MODELLED, ALTERNATE]
    assert all("_grouping_dimension" not in r for r in result)


def test_single_dimension_is_left_alone(make_extractor):
    kept = [_cand("A", "farm"), _cand("B", "farm"), _cand("Lit", "", EXTERNAL)]
    result = make_extractor()._retag_minority_dimensions(kept)
    assert [r["scenario_category"] for r in result] == [MODELLED, MODELLED, EXTERNAL]


def test_dimension_tie_keeps_the_first_seen(make_extractor):
    kept = [_cand("A", "management type"), _cand("B", "region")]
    result = make_extractor()._retag_minority_dimensions(kept)
    assert [r["scenario_category"] for r in result] == [MODELLED, ALTERNATE]


# --- small plumbing ----------------------------------------------------------


def test_flatten_usage_pulls_out_cached_tokens():
    class Usage(dict):
        prompt_tokens_details = SimpleNamespace(cached_tokens=7)

    flat = A._flatten_usage(Usage(prompt_tokens=10, completion_tokens=2))
    assert flat == {"prompt_tokens": 10, "completion_tokens": 2, "cached_tokens": 7}


def test_flatten_usage_without_cache_details():
    assert A._flatten_usage({"prompt_tokens": 3}) == {"prompt_tokens": 3}


def test_prepare_paper_text_strips_bibliography(make_extractor):
    text = "Body text of the study.\n" * 50 + "References\nSmith, J. 2019. Paper.\n"
    stripped, meta = make_extractor()._prepare_paper_text(text)
    assert "Smith, J." not in stripped
    assert meta["bibliography_stripped"] is True
    assert meta["bibliography_chars_removed"] > 0


def test_prepare_paper_text_when_disabled(make_extractor):
    text = "Body.\n" * 50 + "References\nSmith, J. 2019.\n"
    assert make_extractor({"strip_bibliography": False})._prepare_paper_text(text) == (text, None)


def test_prepare_paper_text_nothing_to_strip(make_extractor):
    text, meta = make_extractor()._prepare_paper_text("Short paper.")
    assert text == "Short paper."
    assert meta == {"strip_bibliography": True, "bibliography_stripped": False}
