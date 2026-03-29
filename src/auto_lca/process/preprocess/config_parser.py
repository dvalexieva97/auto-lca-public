import json
import os
from typing import Optional

import pandas as pd


def parse_csv_config(csv_path: str, output_json_path: Optional[str] = None) -> str:
    """
    Parse a CSV config file and convert it to JSON format.

    Expected CSV format:
    - Columns: name, description, aspect, by_scenario, structure_title, structure_data_type
    - Each row represents a field in a concept's structure
    - Multiple rows with the same 'name' are grouped into one concept
    - The 'description', 'aspect', and 'by_scenario' must be consistent for all rows with the same name
    - The structure is built from structure_title (field name) and structure_data_type (field type) pairs

    Example:
        name,description,aspect,by_scenario,structure_title,structure_data_type
        Study Details,Specifics of the study,Methods,FALSE,Country,str
        Study Details,Specifics of the study,Methods,FALSE,Specific location,str | None

    Args:
        csv_path: Path to the CSV config file
        output_json_path: Optional path to save the JSON. If None, creates a temp file.

    Returns:
        Path to the generated JSON config file
    """
    df = pd.read_csv(csv_path)

    # Initialize config structure
    config = {
        "defaults": {
            "add_references": False,
            "add_justifications": False,
            "llm_role": "reasoner_text",
            "singular_occurrence": False,
            "by_scenario": True,
            "aspect": "Methods_Results",
        },
        "aspects": [],
        "concepts": [],
    }

    # Normalize column names (case-insensitive, strip whitespace)
    df.columns = [col.strip() for col in df.columns]
    col_mapping = {col.lower(): col for col in df.columns}

    # Required columns
    required_cols = [
        "name",
        "description",
        "aspect",
        "by_scenario",
        "structure_title",
        "structure_data_type",
    ]
    missing_cols = [col for col in required_cols if col not in col_mapping]
    if missing_cols:
        raise ValueError(
            f"Missing required columns: {missing_cols}. Found columns: {list(df.columns)}"
        )

    # Remove empty rows
    df = df.dropna(subset=[col_mapping["name"]])
    df = df[df[col_mapping["name"]].astype(str).str.strip() != ""]

    # Extract aspects (unique from Aspect column)
    aspects_seen = set()
    for _, row in df.iterrows():
        aspect_name = str(row[col_mapping["aspect"]]).strip()
        if aspect_name and aspect_name.lower() not in ["nan", "none", ""]:
            if aspect_name not in aspects_seen:
                config["aspects"].append(
                    {
                        "name": aspect_name,
                        "description": aspect_name,  # Default description
                    }
                )
                aspects_seen.add(aspect_name)

    # If no aspects found, add default
    if not config["aspects"]:
        config["aspects"] = [
            {"name": "Methods", "description": "Methods of the study"},
            {"name": "Results", "description": "Results of the study"},
            {
                "name": "Methods_Results",
                "description": "Joined Methods and Results of the study",
            },
        ]

    # Group rows by concept name
    concepts_dict = {}
    for _, row in df.iterrows():
        concept_name = str(row[col_mapping["name"]]).strip()

        # Skip empty rows or special marker rows
        if not concept_name or concept_name.lower() in ["nan", "none", ""]:
            continue

        # Skip if it's a special marker (for future use)
        if concept_name.startswith("DEFAULT:") or concept_name.startswith("ASPECT:"):
            continue

        description = str(row[col_mapping["description"]]).strip()
        aspect = str(row[col_mapping["aspect"]]).strip()
        by_scenario_str = str(row[col_mapping["by_scenario"]]).strip().lower()
        structure_title = str(row[col_mapping["structure_title"]]).strip()
        structure_data_type = str(row[col_mapping["structure_data_type"]]).strip()

        # Parse by_scenario (handle FALSE, false, False, etc.)
        by_scenario = by_scenario_str in ["true", "1", "yes", "y"]

        # Initialize concept if not seen before
        if concept_name not in concepts_dict:
            concepts_dict[concept_name] = {
                "name": concept_name,
                "description": description,
                "aspect": aspect,
                "by_scenario": by_scenario,
                "structure": {},
            }
        else:
            # Verify consistency of description, aspect, and by_scenario
            existing = concepts_dict[concept_name]
            if existing["description"] != description:
                raise ValueError(
                    f"Inconsistent description for concept '{concept_name}': "
                    f"found '{description}' but expected '{existing['description']}'"
                )
            if existing["aspect"] != aspect:
                raise ValueError(
                    f"Inconsistent aspect for concept '{concept_name}': "
                    f"found '{aspect}' but expected '{existing['aspect']}'"
                )
            if existing["by_scenario"] != by_scenario:
                raise ValueError(
                    f"Inconsistent by_scenario for concept '{concept_name}': "
                    f"found '{by_scenario}' but expected '{existing['by_scenario']}'"
                )

        # Add structure field (skip if title is empty)
        if structure_title and structure_title.lower() not in ["nan", "none", ""]:
            concepts_dict[concept_name]["structure"][
                structure_title
            ] = structure_data_type

    # Convert to list and validate
    for concept_name, concept in concepts_dict.items():
        if not concept["structure"]:
            raise ValueError(
                f"Concept '{concept_name}' has no structure fields defined"
            )
        config["concepts"].append(concept)

    # Save to JSON
    if output_json_path is None:
        output_json_path = csv_path.replace(".csv", ".json")

    with open(output_json_path, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)

    print(
        f"Converted original .csv config from '{csv_path}' to .json config: '{output_json_path}'"
    )
    return output_json_path


def ensure_json_config(config_path: str) -> str:
    """
    Ensure config is in JSON format. If CSV, convert to JSON (temp file).
    If already JSON, return the path as-is.

    Args:
        config_path: Path to config file (CSV or JSON)

    Returns:
        Path to JSON config file (original if JSON, converted temp file if CSV)
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")

    _, ext = os.path.splitext(config_path.lower())

    if ext == ".json":
        return config_path
    elif ext == ".csv":
        # Convert CSV to JSON (temp file)
        return parse_csv_config(config_path)
    else:
        raise ValueError(
            f"Unsupported config file format: {ext}. Expected .csv or .json"
        )
