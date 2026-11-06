import json
import os

import pandas as pd

# sample_config.json is used as the base when a simpler format (CSV or
# minimal XLSX) is missing keys. defaults/optimization are merged key-by-key;
# field sets are only substituted when completely absent.
_DEFAULT_CONFIG_PATH = os.path.join(
    os.path.dirname(__file__), "..", "sog", "sample_config.json"
)


def _load_base_config() -> dict:
    """Load the full config.json as the base default config."""
    path = os.path.normpath(_DEFAULT_CONFIG_PATH)
    if not os.path.exists(path):
        return {}
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _coerce_value(raw_val):
    """Coerce an Excel/CSV cell value to the appropriate Python scalar type."""
    if isinstance(raw_val, bool):
        return raw_val
    if isinstance(raw_val, (int, float)):
        return raw_val
    if isinstance(raw_val, str):
        low = raw_val.strip().lower()
        if low == "true":
            return True
        if low == "false":
            return False
        try:
            return int(raw_val)
        except ValueError:
            pass
        try:
            return float(raw_val)
        except ValueError:
            pass
        return raw_val.strip()
    return raw_val


def _parse_bool(raw) -> bool:
    return str(raw).strip().lower() in ("true", "1", "yes", "y")


def _is_empty(val: str) -> bool:
    return not val or val.lower() in ("nan", "none", "")


def _parse_field_sets_df(df: pd.DataFrame) -> list:
    """
    Parse a DataFrame of field rows (one row per field) into field set dicts.

    Columns: field_set, description, by_production_system, field, data_type.
    Older column names (name / concept_name, structure_title / structure_field,
    structure_data_type / structure_type, by_scenario) are still accepted.
    """
    df = df.copy()
    df.columns = [c.strip() for c in df.columns]

    aliases = {
        "name": "field_set",
        "concept_name": "field_set",
        "structure_title": "field",
        "structure_field": "field",
        "structure_data_type": "data_type",
        "structure_type": "data_type",
        "by_scenario": "by_production_system",
    }
    df = df.rename(columns={c: aliases.get(c.lower(), c) for c in df.columns})
    col_map = {c.lower(): c for c in df.columns}

    required = ["field_set", "description", "by_production_system", "field", "data_type"]
    missing = [c for c in required if c not in col_map]
    if missing:
        raise ValueError(
            f"Missing required columns: {missing}. Found: {list(df.columns)}"
        )

    name_col = col_map["field_set"]
    df = df.dropna(subset=[name_col])
    df = df[df[name_col].astype(str).str.strip() != ""]

    field_sets: dict = {}
    for _, row in df.iterrows():
        set_name = str(row[name_col]).strip()
        if _is_empty(set_name):
            continue

        description = str(row[col_map["description"]]).strip()
        by_ps = _parse_bool(row[col_map["by_production_system"]])
        field = str(row[col_map["field"]]).strip()
        dtype = str(row[col_map["data_type"]]).strip()

        if set_name not in field_sets:
            field_sets[set_name] = {
                "name": set_name,
                "description": description,
                "by_production_system": by_ps,
                "fields": {},
            }
        elif field_sets[set_name]["description"] != description:
            raise ValueError(f"Inconsistent description for field set '{set_name}'")

        if not _is_empty(field):
            field_sets[set_name]["fields"][field] = dtype

    for field_set in field_sets.values():
        if not field_set["fields"]:
            raise ValueError(f"Field set '{field_set['name']}' has no fields defined")

    return list(field_sets.values())


def _parse_settings_df(df: pd.DataFrame) -> tuple[dict, dict]:
    """
    Parse a DataFrame of settings rows into (defaults, optimization) dicts.

    Columns: section, key, value  (section is either "defaults" or "optimization")
    """
    df = df.copy()
    df.columns = [c.strip().lower() for c in df.columns]
    defaults: dict = {}
    optimization: dict = {}
    for _, row in df.iterrows():
        section = str(row.get("section", "")).strip().lower()
        key = str(row.get("key", "")).strip()
        if _is_empty(key):
            continue
        value = _coerce_value(row.get("value"))
        if section == "defaults":
            defaults[key] = value
        elif section == "optimization":
            optimization[key] = value
    return defaults, optimization


def _enrich_config(config: dict, base: dict) -> dict:
    """
    Enrich a parsed config with values from the base (sample_config.json):
    - defaults:      merged key-by-key (base fills in missing keys)
    - optimization:  merged key-by-key (base fills in missing keys)
    - field_sets:    substituted from base only when completely absent
    - aspects:       no longer used; any legacy ``aspects`` array is dropped
    """
    result = dict(config)

    # key-by-key merge for scalar sections
    for section in ("defaults", "optimization"):
        base_section = base.get(section, {})
        merged = dict(base_section)
        merged.update(result.get(section, {}))  # provided values win
        if merged:
            result[section] = merged

    # field sets: all-or-nothing from base when absent
    if not result.get("field_sets"):
        result["field_sets"] = base.get("field_sets", [])

    result.pop("aspects", None)

    return result


def _save_json(config: dict, path: str) -> str:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)
    return path


# ══════════════════════════════════════════════════════════════════════════════
# Public parsers
# ══════════════════════════════════════════════════════════════════════════════


def parse_csv_config(csv_path: str, output_json_path: str | None = None) -> str:
    """
    Parse a CSV field-sets-only config and convert it to JSON format.

    The CSV only defines fields (field_set, description, by_production_system,
    field, data_type). Defaults are taken from
    sample_config.json (the base default config) so that the full pipeline works even
    when the user provides a minimal CSV.

    Args:
        csv_path: Path to the CSV config file
        output_json_path: Optional path to save the JSON. Defaults to <csv>.json.

    Returns:
        Path to the generated JSON config file
    """
    base = _load_base_config()

    df = pd.read_csv(csv_path)
    field_sets = _parse_field_sets_df(df)

    config = _enrich_config({"field_sets": field_sets}, base)

    if output_json_path is None:
        output_json_path = csv_path.replace(".csv", ".json")

    _save_json(config, output_json_path)
    print(
        f"Converted .csv config '{csv_path}' → '{output_json_path}' "
        f"(missing keys enriched from sample_config.json)"
    )
    return output_json_path


def parse_xlsx_config(xlsx_path: str, output_json_path: str | None = None) -> str:
    """
    Parse an Excel config file and convert it to JSON format.

    Expected sheets:
    - "Field sets":  field_set, description, by_production_system, field,
                     data_type (older sheet name "Concepts" also accepted)
    - "Settings":  section, key, value   (sections: "defaults", "optimization")

    Missing Settings keys are filled from sample_config.json defaults.

    Args:
        xlsx_path: Path to the .xlsx config file
        output_json_path: Optional path to save the JSON. Defaults to <xlsx>.json.

    Returns:
        Path to the generated JSON config file
    """
    base = _load_base_config()
    xl = pd.ExcelFile(xlsx_path)

    # ── Field sets (required) ─────────────────────────────────────────────────
    sheet = next((n for n in ("Field sets", "Concepts") if n in xl.sheet_names), None)
    if sheet is None:
        raise ValueError(
            f"Excel config is missing required sheet 'Field sets'. Found: {xl.sheet_names}"
        )
    field_sets = _parse_field_sets_df(xl.parse(sheet))

    # ── Settings (optional) ───────────────────────────────────────────────────
    if "Settings" in xl.sheet_names:
        defaults, optimization = _parse_settings_df(xl.parse("Settings"))
    else:
        defaults, optimization = {}, {}

    config = _enrich_config(
        {
            "defaults": defaults,
            "optimization": optimization,
            "field_sets": field_sets,
        },
        base,
    )

    if output_json_path is None:
        output_json_path = xlsx_path.replace(".xlsx", ".json")

    _save_json(config, output_json_path)
    print(
        f"Converted .xlsx config '{xlsx_path}' → '{output_json_path}' "
        f"(missing keys enriched from sample_config.json)"
    )
    return output_json_path


def ensure_json_config(config_path: str) -> str:
    """
    Ensure a config is in JSON format, converting CSV or XLSX if necessary.

    Args:
        config_path: Path to config file (.json, .csv, .xlsx, or .xls)

    Returns:
        Path to the JSON config file
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")

    _, ext = os.path.splitext(config_path.lower())

    if ext == ".json":
        return config_path
    elif ext == ".csv":
        return parse_csv_config(config_path)
    elif ext in (".xlsx", ".xls"):
        return parse_xlsx_config(config_path)
    else:
        raise ValueError(
            f"Unsupported config format: {ext}. Expected .json, .csv, or .xlsx"
        )
