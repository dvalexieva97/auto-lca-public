"""Parse LCA field cells into (value, unit) and classify numeric pair outcomes.

Binary ``numeric_exact`` is True only when both sides parse to the same scalar
(after optional compatible unit conversion). Outcome labels explain *why*
pairs agree or fail — they do not soften equality into a graded score.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Outcome vocabulary (stable strings for Joined Data / SI tables)
# ---------------------------------------------------------------------------
OUTCOME_EQUAL = "equal"
OUTCOME_UNIT_FORMAT = "unit_format_only"  # same value; unit present on one side only
OUTCOME_CONVERTIBLE = "unit_convertible"  # equal after t↔kg / MJ↔GJ / …
OUTCOME_VALUE_MISMATCH = "value_mismatch"
OUTCOME_REVIEW_HARMONIZED = "review_harmonized_diff"  # GT bare + auto unit + ≠
OUTCOME_RANGE_VS_POINT = "range_vs_point"
OUTCOME_UNPARSEABLE = "unparseable"
OUTCOME_NA = "not_numeric"  # at least one side empty / non-applicable

# Leading quantity: optional approx marker, optional sign, digits with . or ,
# or European decimal, optional scientific exponent.
#
# Thousands grouping must NOT apply when the integer part is 0, otherwise
# ``0.000000188`` / ``0,000000188`` match only ``0.000`` / ``0,000`` and the
# remainder is rejected. Leading-zero decimals are matched explicitly.
_LEADING_NUMBER_RE = re.compile(
    r"^\s*[≈~≃∼]?\s*"
    r"(?P<num>"
    r"[+-]?"
    r"(?:"
    r"\d+[.,]\d+[eE][+-]?\d+"  # 1.88e-07 / 1,88e-07
    r"|\d+[eE][+-]?\d+"  # 1e-07
    # thousands (+ optional decimal with the other separator): 1,234.56 / 1.234,56
    # integer part must be 1–9… so 0.000188 never takes this branch
    r"|[1-9]\d{0,2}(?:([ ,.])\d{3})+(?:[.,]\d+)?"
    r"|0[.,]\d+"  # leading-zero decimals: 0.000000188 / 0,000000188
    r"|[1-9]\d*[.,]\d+"  # other decimals: 12.76 / 1,5
    r"|\d+"  # integer
    r")"
    r")"
)

# Inline scientific suffix after mantissa: "8.9 × 10^-11", "8.9 x 10^−11", "8.9*10^-11"
_SCI_SUFFIX_RE = re.compile(
    r"^\s*[×xX∗*·⋅]\s*10\s*[\^∗*]?\s*([−\-]?\d+)\b"
)

_RANGE_RE = re.compile(
    r"^\s*[≈~≃∼]?\s*"
    r"[+-]?\d+(?:[.,]\d+)?(?:[eE][+-]?\d+)?"
    r"\s*(?:–|—|-|−|to|To|TO)\s*"
    r"[+-]?\d+(?:[.,]\d+)?(?:[eE][+-]?\d+)?"
    r"(?:\s.*)?$"
)

_NON_NUMERIC_START = re.compile(
    r"^\s*(null|n/?a|none|not\b|no\b|unknown|undisclosed|not\s+explicit|n\.\s*\.?s)",
    re.IGNORECASE,
)

# Strip parenthetical notes / trailing prose after the unit token.
_NOTE_CUT_RE = re.compile(r"\s*[\(\[;].*$")

# Unit synonym → canonical token (dimension family encoded in name).
_UNIT_ALIASES: dict[str, str] = {
    "%": "%",
    "pct": "%",
    "percent": "%",
    "vol%": "%",
    "wt%": "%",
    "kg": "kg",
    "g": "g",
    "mg": "mg",
    "t": "t",
    "ton": "t",
    "tonne": "t",
    "tonnes": "t",
    "kt": "kt",
    "mt": "Mt",
    "mj": "MJ",
    "gj": "GJ",
    "kj": "kJ",
    "kwh": "kWh",
    "mwh": "MWh",
    "l": "L",
    "liter": "L",
    "litre": "L",
    "liters": "L",
    "litres": "L",
    "ml": "mL",
    "m3": "m3",
    "m³": "m3",
    "m2": "m2",
    "m²": "m2",
    "ha": "ha",
    "bar": "bar",
    "pa": "Pa",
    "mpa": "MPa",
    "year": "year",
    "years": "year",
    "a": "year",
    "lu": "LU",
    # climate / impact
    "kgco2eq": "kg_CO2eq",
    "kgco2e": "kg_CO2eq",
    "kgco2-eq": "kg_CO2eq",
    "tco2eq": "t_CO2eq",
    "tco2e": "t_CO2eq",
    "gco2eq": "g_CO2eq",
}

# Compatible conversions to a common base (factor multiplies value → base).
# Keys are canonical unit tokens from _UNIT_ALIASES values.
_TO_BASE: dict[str, tuple[str, float]] = {
    "kg": ("g", 1000.0),
    "g": ("g", 1.0),
    "mg": ("g", 0.001),
    "t": ("g", 1_000_000.0),
    "kt": ("g", 1_000_000_000.0),
    "Mt": ("g", 1_000_000_000_000.0),
    "kg_CO2eq": ("g_CO2eq", 1000.0),
    "g_CO2eq": ("g_CO2eq", 1.0),
    "t_CO2eq": ("g_CO2eq", 1_000_000.0),
    "kJ": ("J", 1000.0),
    "MJ": ("J", 1_000_000.0),
    "GJ": ("J", 1_000_000_000.0),
    "kWh": ("J", 3_600_000.0),
    "MWh": ("J", 3_600_000_000.0),
    "mL": ("L", 0.001),
    "L": ("L", 1.0),
    "m3": ("L", 1000.0),
    "%": ("%", 1.0),
    "bar": ("Pa", 100_000.0),
    "MPa": ("Pa", 1_000_000.0),
    "Pa": ("Pa", 1.0),
    "year": ("year", 1.0),
    "LU": ("LU", 1.0),
    "m2": ("m2", 1.0),
    "ha": ("m2", 10_000.0),
}


@dataclass(frozen=True)
class Quantity:
    value: float | None
    unit: str | None  # canonical, or None if bare / unknown
    raw_unit: str | None
    is_range: bool = False
    parseable: bool = False


def _is_null(val: Any) -> bool:
    if isinstance(val, (pd.Series, np.ndarray)):
        if len(val) == 0:
            return True
        val = val.iloc[0] if isinstance(val, pd.Series) else val.flat[0]
    if val is None:
        return True
    try:
        if pd.isna(val):
            return True
    except (TypeError, ValueError):
        pass
    if isinstance(val, (list, dict)):
        return not val
    return str(val).strip() == ""


def _canonicalize_number_token(tok: str) -> float | None:
    s = tok.strip().replace(" ", "").replace("\u00a0", "")
    if not s:
        return None
    if re.search(r"[eE]", s):
        mant, _, exp = s.lower().partition("e")
        mant = (
            mant.replace(",", ".")
            if mant.count(",") == 1 and "." not in mant
            else mant
        )
        try:
            return float(f"{mant}e{exp}")
        except ValueError:
            return None
    if "," in s and "." in s:
        if s.rfind(",") > s.rfind("."):
            try:
                return float(s.replace(".", "").replace(",", "."))
            except ValueError:
                return None
        try:
            return float(s.replace(",", ""))
        except ValueError:
            return None
    if "," in s and "." not in s:
        parts = s.split(",")
        # Thousands grouping: every group after the first is exactly 3 digits,
        # and the integer part is not a lone 0 (0,000188 is a European decimal).
        int_part = parts[0].lstrip("+-")
        if (
            len(parts) >= 2
            and int_part.isdigit()
            and int_part != "0"
            and all(p.isdigit() and len(p) == 3 for p in parts[1:])
        ):
            try:
                return float(s.replace(",", ""))
            except ValueError:
                return None
        # Otherwise treat comma as decimal separator (incl. 0,000000188).
        try:
            return float(s.replace(",", "."))
        except ValueError:
            return None
    try:
        return float(s)
    except ValueError:
        return None


def normalize_unit(raw: str | None) -> str | None:
    """Map a raw unit/note fragment to a canonical unit token, if possible."""
    if not raw:
        return None
    s = raw.strip()
    s = _NOTE_CUT_RE.sub("", s).strip()
    if not s:
        return None
    # Collapse whitespace and common separators for alias lookup.
    key = (
        s.lower()
        .replace("₂", "2")
        .replace("³", "3")
        .replace("²", "2")
        .replace("·", "")
        .replace("⋅", "")
    )
    key = re.sub(r"\s+", "", key)
    key = key.replace("carbon dioxide", "co2")
    # Prefer longer impact-style aliases first via explicit patterns.
    if re.search(r"kg.*co2", key):
        return "kg_CO2eq"
    if re.search(r"(^|/)t.*co2", key) or key.startswith("tco2"):
        return "t_CO2eq"
    if re.search(r"g.*co2", key):
        return "g_CO2eq"
    # Take first path segment before / for mass-like "kg/kg FPCM"
    head = key.split("/")[0]
    if head in _UNIT_ALIASES:
        return _UNIT_ALIASES[head]
    if key in _UNIT_ALIASES:
        return _UNIT_ALIASES[key]
    # Leading unit token (kg, MJ, …)
    m = re.match(r"(?i)^(kg|mg|g|t|kt|mt|mj|gj|kj|kwh|mwh|l|ml|m3|m2|ha|bar|mpa|pa|lu|%|pct)", key)
    if m:
        return _UNIT_ALIASES.get(m.group(1).lower(), m.group(1))
    return None


def _extract_raw_unit(text: str, num_end: int) -> str | None:
    rest = text[num_end:].strip()
    if not rest:
        return None
    rest = _NOTE_CUT_RE.sub("", rest).strip()
    if not rest:
        return None
    # Drop leading punctuation
    rest = rest.lstrip(":=").strip()
    if rest in {"%", "°", "°C"}:
        return rest
    return rest or None


def parse_quantity(val: Any) -> Quantity:
    """Parse a cell into value + unit (canonical when recognised)."""
    if isinstance(val, (pd.Series, np.ndarray)):
        if len(val) == 0:
            return Quantity(None, None, None, parseable=False)
        val = val.iloc[0] if isinstance(val, pd.Series) else val.flat[0]

    if _is_null(val):
        return Quantity(None, None, None, parseable=False)

    if isinstance(val, bool):
        return Quantity(None, None, None, parseable=False)

    if isinstance(val, (int, float, np.integer, np.floating)):
        f = float(val)
        if not math.isfinite(f):
            return Quantity(None, None, None, parseable=False)
        return Quantity(f, None, None, parseable=True)

    text = str(val).strip().replace("\u2212", "-").replace("\u00a0", " ")
    if not text:
        return Quantity(None, None, None, parseable=False)
    if _NON_NUMERIC_START.match(text):
        return Quantity(None, None, None, parseable=False)

    if _RANGE_RE.match(text):
        return Quantity(None, None, None, is_range=True, parseable=False)

    compact = re.sub(r"\s+", "", text)
    if re.fullmatch(r"[0-9a-fA-F]{16,}", compact):
        return Quantity(None, None, None, parseable=False)

    m = _LEADING_NUMBER_RE.match(text)
    if not m:
        return Quantity(None, None, None, parseable=False)

    num = _canonicalize_number_token(m.group("num"))
    if num is None or not math.isfinite(num):
        return Quantity(None, None, None, parseable=False)

    rest = text[m.end() :]
    # Fold "× 10^-11" / "x 10^−11" into the mantissa before unit extraction.
    sci = _SCI_SUFFIX_RE.match(rest)
    num_end = m.end()
    if sci:
        try:
            exp = int(sci.group(1).replace("−", "-"))
            num = num * (10**exp)
            num_end = m.end() + sci.end()
            rest = text[num_end:]
        except (TypeError, ValueError, OverflowError):
            pass

    if rest:
        if rest[0].isspace() or rest[0] in "(%°µ%":
            pass
        elif re.match(
            r"(?i)^(kg|mg|g|t|kt|mt|mj|gj|kj|kwh|mwh|l|ml|m3|m²|m2|ha|co2|n2o|ch4|eq|bar|%|lu)",
            rest,
        ):
            pass
        elif rest[0].isalnum():
            return Quantity(None, None, None, parseable=False)

    raw_unit = _extract_raw_unit(text, num_end)
    canon = normalize_unit(raw_unit) if raw_unit else None
    # Bare percent glued: "15%"
    if raw_unit is None and text[num_end : num_end + 1] == "%":
        raw_unit = "%"
        canon = "%"
    return Quantity(num, canon, raw_unit, parseable=True)


def parse_numeric_value(val: Any) -> float | None:
    """Back-compat: scalar only."""
    q = parse_quantity(val)
    return q.value if q.parseable else None


def _to_base(value: float, unit: str | None) -> tuple[str, float] | None:
    if unit is None:
        return None
    if unit not in _TO_BASE:
        return None
    base, factor = _TO_BASE[unit]
    return base, value * factor


def quantities_equal(
    a: Quantity,
    b: Quantity,
    *,
    rel_tol: float = 1e-9,
    abs_tol: float = 1e-12,
) -> tuple[bool | None, str | None]:
    """
    Compare two quantities.

    Returns (equal?, match_mode) where match_mode is
    ``exact``, ``unit_format``, ``convertible``, or None if N/A / unequal.
    """
    if not a.parseable or not b.parseable or a.value is None or b.value is None:
        return None, None

    # Same bare values
    if math.isclose(a.value, b.value, rel_tol=rel_tol, abs_tol=abs_tol):
        if bool(a.unit) != bool(b.unit):
            # One bare, one with unit — treat as format-only equal
            return True, "unit_format"
        if a.unit and b.unit and a.unit != b.unit:
            # Same number, different unit labels that didn't convert — still
            # count as value-equal only if aliases collapsed; else mismatch.
            return False, None
        return True, "exact"

    # Compatible unit conversion
    ba = _to_base(a.value, a.unit)
    bb = _to_base(b.value, b.unit)
    if ba and bb and ba[0] == bb[0]:
        if math.isclose(ba[1], bb[1], rel_tol=1e-6, abs_tol=1e-9):
            return True, "convertible"

    # One bare + one united: cannot convert; values already differed
    return False, None


def numeric_values_equal(
    a: Any,
    b: Any,
    *,
    rel_tol: float = 1e-9,
    abs_tol: float = 1e-12,
) -> bool | None:
    qa, qb = parse_quantity(a), parse_quantity(b)
    eq, _ = quantities_equal(qa, qb, rel_tol=rel_tol, abs_tol=abs_tol)
    return eq


def classify_numeric_pair(
    auto: Any,
    manual: Any,
    *,
    field_class: str | None = None,
) -> dict[str, Any]:
    """
    Full pair assessment for Joined Data / SI.

    Returns dict with keys:
      numeric_exact (0/1/None), numeric_distance, outcome,
      auto_value, manual_value, auto_unit, manual_unit
    """
    qa, qb = parse_quantity(auto), parse_quantity(manual)

    # Empties → N/A (not counted in numeric-equality denominator)
    if _is_null(auto) or _is_null(manual):
        return {
            "numeric_exact": None,
            "numeric_distance": None,
            "outcome": OUTCOME_NA,
            "auto_value": qa.value,
            "manual_value": qb.value,
            "auto_unit": qa.unit,
            "manual_unit": qb.unit,
        }

    if qa.is_range or qb.is_range:
        # Range on one side, point on the other (or both ranges)
        outcome = OUTCOME_RANGE_VS_POINT if (qa.is_range != qb.is_range) or qa.is_range else OUTCOME_UNPARSEABLE
        return {
            "numeric_exact": None,
            "numeric_distance": None,
            "outcome": outcome,
            "auto_value": None,
            "manual_value": qb.value if qb.parseable else None,
            "auto_unit": qa.unit,
            "manual_unit": qb.unit,
        }

    if not qa.parseable or not qb.parseable:
        return {
            "numeric_exact": None,
            "numeric_distance": None,
            "outcome": OUTCOME_UNPARSEABLE,
            "auto_value": qa.value,
            "manual_value": qb.value,
            "auto_unit": qa.unit,
            "manual_unit": qb.unit,
        }

    eq, mode = quantities_equal(qa, qb)
    dist = abs(qa.value - qb.value) if qa.value is not None and qb.value is not None else None

    if eq is True:
        if mode == "unit_format":
            outcome = OUTCOME_UNIT_FORMAT
        elif mode == "convertible":
            outcome = OUTCOME_CONVERTIBLE
        else:
            outcome = OUTCOME_EQUAL
        return {
            "numeric_exact": 1.0,
            "numeric_distance": 0.0 if mode != "convertible" else dist,
            "outcome": outcome,
            "auto_value": qa.value,
            "manual_value": qb.value,
            "auto_unit": qa.unit,
            "manual_unit": qb.unit,
        }

    # Unequal parseable values
    auto_has_u = qa.unit is not None or bool(qa.raw_unit)
    man_has_u = qb.unit is not None or bool(qb.raw_unit)
    # Heuristic: meta-analysis often stores bare numbers while extract keeps units.
    reviewish = (
        field_class in {"numeric_units", "numeric_reviewer_harmonized", None}
        and auto_has_u
        and not man_has_u
    )
    # Stronger if field explicitly tagged
    if field_class == "numeric_reviewer_harmonized" or reviewish:
        # Only call it review-harmonized when GT is bare and values disagree
        if auto_has_u and not man_has_u:
            outcome = OUTCOME_REVIEW_HARMONIZED
        else:
            outcome = OUTCOME_VALUE_MISMATCH
    else:
        outcome = OUTCOME_VALUE_MISMATCH

    return {
        "numeric_exact": 0.0,
        "numeric_distance": dist,
        "outcome": outcome,
        "auto_value": qa.value,
        "manual_value": qb.value,
        "auto_unit": qa.unit,
        "manual_unit": qb.unit,
    }
