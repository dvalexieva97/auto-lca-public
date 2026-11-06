import json

from auto_lca.process.sog.concept import SAMPLE_CONFIG_PATH, build_concepts_from_config


def _concepts(write_config, concepts, defaults=None):
    path = write_config({"defaults": defaults or {}, "field_sets": concepts})
    study_wide, by_system = build_concepts_from_config(path)
    return [c.name for c in study_wide], [c.name for c in by_system]


def _c(name, **kw):
    return {"name": name, "description": name, "fields": {"x": "str"}, **kw}


def test_splits_by_grain_and_keeps_config_order(write_config):
    study_wide, by_system = _concepts(
        write_config,
        [
            _c("A", by_production_system=True),
            _c("B", by_production_system=False),
            _c("C", by_production_system=True),
            _c("D", by_production_system=False),
        ],
    )
    assert study_wide == ["B", "D"]
    assert by_system == ["A", "C"]


def test_legacy_by_scenario_key(write_config):
    study_wide, by_system = _concepts(
        write_config, [_c("A", by_scenario=False), _c("B", by_scenario=True)]
    )
    assert study_wide == ["A"]
    assert by_system == ["B"]


def test_by_production_system_wins_over_legacy_key(write_config):
    study_wide, _ = _concepts(
        write_config, [_c("A", by_production_system=False, by_scenario=True)]
    )
    assert study_wide == ["A"]


def test_falls_back_to_config_defaults(write_config):
    study_wide, by_system = _concepts(
        write_config, [_c("A"), _c("B", by_production_system=True)],
        defaults={"by_production_system": False},
    )
    assert study_wide == ["A"]
    assert by_system == ["B"]


def test_legacy_default_key(write_config):
    study_wide, _ = _concepts(write_config, [_c("A")], defaults={"by_scenario": False})
    assert study_wide == ["A"]


def test_defaults_to_per_system_when_unset(write_config):
    _, by_system = _concepts(write_config, [_c("A")])
    assert by_system == ["A"]


def test_aspect_label_does_not_change_grain(write_config):
    # Regression: the old aspect-grouping loader sent every concept without an
    # `aspect` label to study-wide, even when marked per-system.
    study_wide, by_system = _concepts(
        write_config,
        [_c("No label", by_production_system=True),
         _c("Labelled", aspect="Tables", by_production_system=True)],
    )
    assert study_wide == []
    assert by_system == ["No label", "Labelled"]


def test_fields_and_description_are_kept(write_config):
    path = write_config({"field_sets": [{"name": "A", "description": "desc",
                                         "fields": {"f1": "str", "f2": "float | None"}}]})
    _, (concept,) = build_concepts_from_config(path)
    assert concept.description == "desc"
    assert concept.structure == {"f1": "str", "f2": "float | None"}


def test_legacy_concepts_and_structure_keys(write_config):
    path = write_config({"concepts": [{"name": "A", "description": "desc",
                                       "by_production_system": False,
                                       "structure": {"f1": "str"}}]})
    (concept,), by_system = build_concepts_from_config(path)
    assert by_system == []
    assert concept.structure == {"f1": "str"}


def test_default_path_is_the_bundled_sample_config():
    with open(SAMPLE_CONFIG_PATH) as f:
        assert json.load(f)["field_sets"]
    study_wide, by_system = build_concepts_from_config()
    assert study_wide or by_system
