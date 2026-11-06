"""
Lightweight concept class and config loader for LCA extraction.

Configs describe the schema as ``field_sets``, each with a ``fields`` map of
field name to data type. Internally a field set is a "concept" (the LLM
prompts use that word). Older configs that use ``concepts`` / ``structure``
still load.
"""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

# Bundled with the package, so this works from any working directory.
SAMPLE_CONFIG_PATH = str(Path(__file__).with_name("sample_config.json"))


@dataclass
class JsonObjectConcept:
    """A field set from the domain config: a named set of typed fields."""

    name: str
    description: str
    by_scenario: bool
    structure: dict[str, Any]


def config_field_sets(config: dict) -> list[dict]:
    """The config's field sets (``field_sets``, or legacy ``concepts``)."""
    return config.get("field_sets", config.get("concepts", []))


def field_set_fields(field_set: dict) -> dict[str, Any]:
    """A field set's fields (``fields``, or legacy ``structure``)."""
    return field_set.get("fields", field_set.get("structure", {}))


def build_concepts_from_config(
    config_path: str = SAMPLE_CONFIG_PATH,
) -> tuple[list[JsonObjectConcept], list[JsonObjectConcept]]:
    """
    Load field sets from a domain config and split them by grain.

    A field set's grain comes from ``by_production_system`` (or legacy
    ``by_scenario``), falling back to the config defaults, then True.

    Returns:
        tuple: (study_wide_concepts, by_scenario_concepts), in config order.
    """
    with open(config_path) as f:
        config = json.load(f)

    defaults = config.get("defaults", {})
    default_by_ps = defaults.get(
        "by_production_system", defaults.get("by_scenario", True)
    )

    study_wide: list[JsonObjectConcept] = []
    by_scenario: list[JsonObjectConcept] = []
    for concept_def in config_field_sets(config):
        by_ps = concept_def.get("by_production_system")
        if by_ps is None:
            by_ps = concept_def.get("by_scenario")
        if by_ps is None:
            by_ps = default_by_ps
        concept = JsonObjectConcept(
            name=concept_def["name"],
            description=concept_def["description"],
            by_scenario=bool(by_ps),
            structure=field_set_fields(concept_def),
        )
        (by_scenario if concept.by_scenario else study_wide).append(concept)

    return study_wide, by_scenario
