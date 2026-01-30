import json
from copy import deepcopy
from typing import List, Literal, Optional

from contextgem import (
    Aspect,
    JsonObjectConcept,  # assuming this is your class
)
from pydantic import Field, create_model


def build_aspect(aspect_def: dict) -> Aspect:
    return Aspect(
        name=aspect_def["name"],
        description=aspect_def["description"],
        add_justifications=False,
    )


config_path = "src/auto_lca/process/sog/config.json"


def build_concepts_from_config_scenario_constraint(
    scenarios: List[str], config_path: str
):

    with open(config_path) as f:
        config = json.load(f)

    defaults = config.get("defaults", {})
    scenarios = tuple(scenarios)
    n_scenarios = len(scenarios)

    TYPE_MAP = {
        "str": str,
        "float": float,
        "int": int,
        "bool": bool,
        "dict": dict,
        "list": list,
    }

    def parse_type(type_str: str):
        """Parse strings like 'Optional[List[float]]', 'Literal:scenarios', 'dict[str, str]'."""
        type_str = type_str.strip()

        # Literal from scenarios
        if type_str.startswith("Literal:"):
            _, source = type_str.split(":", 1)
            if source == "scenarios":
                return Literal[scenarios]
            raise ValueError(f"Unknown literal source: {source}")

        # Handle Optional or "| None"
        if "| None" in type_str or "Optional[" in type_str:
            base = (
                type_str.replace("| None", "")
                .replace("Optional[", "")
                .rstrip("]")
                .strip()
            )
            inner = parse_type(base)

            # Only allow Optional for primitives or Literals
            if inner in (str, int, float, bool) or str(inner).startswith(
                "typing.Literal"
            ):
                return Optional[inner]
            # Otherwise, just return the non-optional version
            return inner

        # Handle List[...] generics
        if type_str.startswith("List[") and type_str.endswith("]"):
            inner = type_str[5:-1].strip()
            return List[parse_type(inner)]

        # Handle dict[...] generics
        if type_str.startswith("dict[") and type_str.endswith("]"):
            key_t, val_t = [x.strip() for x in type_str[5:-1].split(",", 1)]
            return dict[parse_type(key_t), parse_type(val_t)]

        # Basic types
        if type_str in TYPE_MAP:
            return TYPE_MAP[type_str]

        raise ValueError(f"Unsupported type: {type_str}")

    concepts = []
    aspect_map = {}

    for concept_config in config["concepts"]:
        structure = concept_config["structure"]
        by_scenario = concept_config.get(
            "by_scenario", defaults.get("by_scenario", False)
        )

        # List-based structure
        if "items" in structure:
            fields = {"scenario": "Literal:scenarios"} if by_scenario else {}
            for field, type_str in structure["items"][0].items():
                fields[field] = (parse_type(type_str), None)

            ConceptItem = create_model(f"{concept_config['name']}Item", **fields)
            ConceptList = create_model(
                f"{concept_config['name']}List",
                items=(
                    List[ConceptItem],
                    Field(..., min_length=n_scenarios, max_length=n_scenarios),
                ),
            )
            model_structure = {"data": ConceptList}

        # Single-object structure
        else:
            structure["scenario"] = "Literal:scenarios"
            fields = {f: (parse_type(t), None) for f, t in structure.items()}
            ConceptItem = create_model(f"{concept_config['name']}Item", **fields)
            model_structure = {"data": ConceptItem}

        concept = JsonObjectConcept(
            name=concept_config["name"],
            description=concept_config["description"],
            structure=model_structure,
            add_references=concept_config.get(
                "add_references", defaults.get("add_references")
            ),
            llm_role=concept_config.get("llm_role", defaults.get("llm_role")),
            add_justifications=concept_config.get(
                "add_justifications", defaults.get("add_justifications")
            ),
        )

        concepts.append(concept)

        aspect_name = concept_config.get("aspect", defaults["aspect"])  # TODO Enum
        if not aspect_map.get(aspect_name):
            aspect_map[aspect_name] = [concept]
        else:
            aspect_map[aspect_name].append(concept)

    return concepts, aspect_map


def build_concepts_and_aspects_from_config(config_path: str):
    """
    Builds list of concepts and aspects witht their concepts assigned.
    Filtered by flag 'by scenario' - meaning weather the concept
    needs to be extracted for each production scenario, or once per document
    """

    with open(config_path) as f:
        config = json.load(f)

    defaults = config.get("defaults", {})

    TYPE_MAP = {
        "str": str,
        "float": float,
        "int": int,
        "bool": bool,
        "dict": dict,
        "list": list,
    }

    def parse_type(type_str: str):
        """Parse strings like 'Optional[List[float]]', 'Literal:scenarios', 'dict[str, str]'."""
        type_str = type_str.strip()

        # Literal from scenarios
        if type_str.startswith("Literal:"):
            _, source = type_str.split(":", 1)
            raise ValueError(f"Unknown literal source: {source}")

        # Handle Optional or "| None"
        if "| None" in type_str or "Optional[" in type_str:
            base = (
                type_str.replace("| None", "")
                .replace("Optional[", "")
                .rstrip("]")
                .strip()
            )
            inner = parse_type(base)

            # Only allow Optional for primitives or Literals
            if inner in (str, int, float, bool) or str(inner).startswith(
                "typing.Literal"
            ):
                return Optional[inner]
            # Otherwise, just return the non-optional version
            return inner

        # Handle List[...] generics
        if type_str.startswith("List[") and type_str.endswith("]"):
            inner = type_str[5:-1].strip()
            return List[parse_type(inner)]

        # Handle dict[...] generics
        if type_str.startswith("dict[") and type_str.endswith("]"):
            key_t, val_t = [x.strip() for x in type_str[5:-1].split(",", 1)]
            return dict[parse_type(key_t), parse_type(val_t)]

        # Basic types
        if type_str in TYPE_MAP:
            return TYPE_MAP[type_str]

        raise ValueError(f"Unsupported type: {type_str}")

    concepts = []
    aspects = {conf["name"]: build_aspect(conf) for conf in config["aspects"]}
    aspect_name_map = {aspect_name: [] for aspect_name in aspects.keys()}
    aspect_map = {
        "by_scenario": deepcopy(aspect_name_map),
        "not_by_scenario": deepcopy(aspect_name_map),
    }

    for concept_config in config["concepts"]:
        structure = concept_config["structure"]
        by_scenario = concept_config.get("by_scenario", defaults.get("by_scenario"))
        if by_scenario is None:
            raise ValueError("Default 'by_scenario' flag not defined")

        # List-based structure
        if "items" in structure:
            fields = {}
            for field, type_str in structure["items"][0].items():
                fields[field] = parse_type(type_str)
        # Single-object structure
        else:
            fields = {f: parse_type(t) for f, t in structure.items()}

        concept = JsonObjectConcept(
            name=concept_config["name"],
            description=concept_config["description"],
            structure=fields,
            add_references=concept_config.get(
                "add_references", defaults.get("add_references")
            ),
            llm_role=concept_config.get("llm_role", defaults.get("llm_role")),
            add_justifications=concept_config.get(
                "add_justifications", defaults.get("add_justifications")
            ),
            singular_occurrence=concept_config.get(
                "singular_occurrence", defaults.get("singular_occurrence")
            ),
        )

        aspect_name = concept_config.get("aspect", defaults["aspect"])  # TODO Enum

        if not aspect_name:  # TODO See how to handle non-aspect data
            concepts.append(concept)
            continue

        if by_scenario:
            base_key = "by_scenario"  # TODO Enum
        else:
            base_key = "not_by_scenario"

        # Append concepts
        aspect_map[base_key][aspect_name].append(concept)

    return aspects, concepts, aspect_map


# concepts = build_concepts_from_config_scenario_constraint(
#     scenarios=["S1", "S2"], config_path="src/auto_lca/process/sog/config.json"
# )

# # Example: Use the first dynamically built concept
# test_concept = concepts[1]
# print(test_concept.name)  # "Test Concept"
# print(test_concept.structure)

# test_concept.model_post_init


# from pydantic import BaseModel


# class ImpactItem(BaseModel):
#     scenario: Literal[
#         (
#             "Baseline Scenario (BS)",
#             "Third Daily Milking Strategy (3MS)",
#             "Anaerobic Digestion Strategy (ADS)",
#         )
#     ]
#     CO2: Optional[float] = None
#     N20: Optional[str] = None


# class ImpactList(BaseModel):
#     items: List[ImpactItem] = Field(..., min_length=3, max_length=3)


# test_concept = JsonObjectConcept(
#     name="Test Concept",
#     description="Test Concept Midpoint Indicators",
#     structure={"data": ImpactList},
#     add_references=False,
#     llm_role="reasoner_text",
#     add_justifications=False,
# )
