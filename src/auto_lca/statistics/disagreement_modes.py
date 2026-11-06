"""Automatic disagreement-mode labels for both-populated field cells.

Primary mode (mutually exclusive, apply in order, stop at first match):

1. format_only
2. granularity  (+ direction: auto_finer | auto_coarser)
3. code_vs_prose (+ code_side: ref|auto; code_inferable: yes|no|unclear)
4. different_value_selected
5. content_conflict
6. wrong_field
7. unclear

Also label each side independently as ``auto_form`` / ``manual_form``:
``numeric`` | ``code`` | ``short_label`` | ``prose``.

Use the per-side forms to *explain* the primary mode (who is prose, who is
more detailed). Do not treat them as a second competing taxonomy.

The heuristic cascade above (``assign_disagreement_mode``) is a cheap
fallback for pool-level counts. For reportable labels use the LLM judge
(``create_disagreement_mode_judge``), which classifies every cell — numeric
ones included — into the finer ``DISAGREEMENT_CLASSES`` with structured
outputs.
"""

from __future__ import annotations

import json
import math
import os
import re
import threading
import time
from collections import Counter
from typing import Any, Callable

import pandas as pd

from auto_lca.statistics.numeric_parse import classify_numeric_pair

PRIMARY_MODES = (
    "format_only",
    "granularity",
    "code_vs_prose",
    "different_value_selected",
    "content_conflict",
    "wrong_field",
    "unclear",
)

_YES_NO = re.compile(r"^(yes|no|y|n|true|false)$", re.I)
_DIGIT = re.compile(r"^\d+(\.\d+)?$")
_TRL_BIN = re.compile(r"^(1 to 3|4 and 5|6 to 9)$", re.I)
_UNITISH = re.compile(
    r"\b(kg|g|mg|t|mj|gj|kwh|l|m3|m²|m2|ha|yr|year|co2|co₂|eq\.?|%|percent)\b",
    re.I,
)
_NUM_TOKEN = re.compile(
    r"[-+]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?(?:[eE][-+]?\d+)?"
)


def _is_empty(v: Any) -> bool:
    if v is None:
        return True
    if isinstance(v, float) and pd.isna(v):
        return True
    return str(v).strip().lower() in {"", "none", "nan", "null"}


def _norm_text(v: Any) -> str:
    s = str(v).strip()
    s = re.sub(r"\s+", " ", s)
    return s


def _casefold_alnum(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", s.casefold())


def _extract_numbers(s: str) -> list[float]:
    out: list[float] = []
    for tok in _NUM_TOKEN.findall(s.replace(" ", "")):
        try:
            out.append(float(tok.replace(",", "")))
        except ValueError:
            continue
    return out


def _is_code(s: str) -> bool:
    t = s.strip()
    if not t:
        return False
    if _YES_NO.match(t):
        return True
    if _DIGIT.match(t):
        return True
    if _TRL_BIN.match(t):
        return True
    # short closed-ish tokens: few words, no sentence punctuation
    if len(t) <= 48 and t.count(" ") <= 5 and not re.search(r"[.?!;:]", t):
        # known review-ish patterns
        if re.search(
            r"\b(accounted for|not accounted|pure feedstock|dLUC|iLUC|"
            r"lab-scale|pilot-scale|attributional|consequential|"
            r"biochemical|thermochemical|mechanical|conventional|"
            r"bioplast|biopolymer|biorefinery|biocomposite|biofiber|"
            r"waste stream|residues|virgin|residual)\b",
            t,
            re.I,
        ):
            return True
        if len(t) <= 24 and t.count(" ") <= 3:
            return True
    return False


def _is_numeric_cell(s: str) -> bool:
    t = s.strip()
    if _DIGIT.match(t):
        return True
    nums = _extract_numbers(t)
    if not nums:
        return False
    # mostly a number + optional unit
    alnum = _casefold_alnum(t)
    digits = re.sub(r"[^0-9]", "", alnum)
    return len(digits) >= 1 and len(digits) / max(len(alnum), 1) >= 0.35


def _is_prose(s: str) -> bool:
    t = s.strip()
    if len(t) >= 80:
        return True
    if t.count(" ") >= 10:
        return True
    if re.search(r"[.?!].+\w", t) and len(t) > 40:
        return True
    return False


def classify_form(value: Any) -> str:
    """Per-side form: numeric | code | short_label | prose."""
    if _is_empty(value):
        return "empty"
    s = _norm_text(value)
    if _is_numeric_cell(s):
        return "numeric"
    if _is_prose(s):
        return "prose"
    if _is_code(s):
        return "code"
    return "short_label"


def _format_only(a: str, m: str) -> bool:
    if _casefold_alnum(a) == _casefold_alnum(m) and a.casefold() != m.casefold():
        return True
    na, nm = _extract_numbers(a), _extract_numbers(m)
    if len(na) == 1 and len(nm) == 1:
        if math.isclose(na[0], nm[0], rel_tol=1e-3, abs_tol=1e-6):
            # same number, possibly different unit/punctuation
            if _casefold_alnum(a) != _casefold_alnum(m):
                return True
            return a.strip() != m.strip()
        # 2092755 vs 2,092,755 already equal after parse
    # strip common unit words and compare
    a2 = _UNITISH.sub("", a)
    m2 = _UNITISH.sub("", m)
    na2, nm2 = _extract_numbers(a2), _extract_numbers(m2)
    if len(na2) == 1 and len(nm2) == 1 and math.isclose(
        na2[0], nm2[0], rel_tol=1e-3, abs_tol=1e-6
    ):
        return True
    return False


def _contains_soft(longer: str, shorter: str) -> bool:
    L, S = longer.casefold(), shorter.casefold()
    if S in L and len(S) >= 3 and len(L) > len(S) + 2:
        return True
    # token containment
    st = set(re.findall(r"[a-z0-9]+", S))
    lt = set(re.findall(r"[a-z0-9]+", L))
    if st and st <= lt and len(L) > len(S) + 5:
        return True
    return False


def _granularity(a: str, m: str) -> str | None:
    """Return auto_finer | auto_coarser | None."""
    if _contains_soft(a, m):
        return "auto_finer"
    if _contains_soft(m, a):
        return "auto_coarser"
    # number + extra prose on one side
    na, nm = _extract_numbers(a), _extract_numbers(m)
    if len(na) == 1 and len(nm) == 1 and math.isclose(
        na[0], nm[0], rel_tol=1e-3, abs_tol=1e-6
    ):
        if len(a) > len(m) + 10:
            return "auto_finer"
        if len(m) > len(a) + 10:
            return "auto_coarser"
    return None


def _code_inferable(code: str, prose: str) -> str:
    c = code.strip().casefold()
    p = prose.casefold()
    if c in p:
        return "yes"
    # digit mentioned in prose
    if _DIGIT.match(code.strip()) and re.search(
        rf"\btrl\b.*\b{re.escape(code.strip().rstrip('.0'))}\b", p
    ):
        return "yes"
    if _YES_NO.match(code.strip()):
        # Yes/No vs descriptive inclusion — usually not safely inferable
        if code.strip().lower() == "yes" and re.search(
            r"\b(yes|performed|included|accounted|considered)\b", p
        ):
            return "unclear"
        if code.strip().lower() == "no" and re.search(
            r"\b(no|not |excluded|without)\b", p
        ):
            return "unclear"
        return "no"
    return "unclear"


def _wrong_field(field: str, auto: str, manual: str) -> bool:
    """Heuristic: auto looks like a DB/source while field is not LCI source, etc."""
    f = field.casefold()
    a = auto.casefold()
    # Background DB dumped into non-source fields
    dbish = bool(
        re.search(r"\b(ecoinvent|plasticseurope|greet|gaibi|agribalyse)\b", a)
    )
    if dbish and "lci source" not in f and "database" not in f and "software" not in f:
        return True
    # Long assumption list into Uncertainty analysis / Yes-No fields
    if re.search(r"uncertainty|heat recovery|solvent|recycling|simulation", f):
        if _is_prose(auto) and _YES_NO.match(manual.strip()):
            return False  # handled as code_vs_prose
    return False


def assign_disagreement_mode(
    field: str,
    auto_value: Any,
    manual_value: Any,
) -> dict[str, Any]:
    """Return primary mode + per-side forms + optional direction flags.

    Heuristic path (used as fallback when no LLM judge is supplied). Numeric
    equality goes through ``classify_numeric_pair`` so unit/format-only pairs
    become ``format_only`` instead of ``different_value_selected``.
    """
    auto_form = classify_form(auto_value)
    manual_form = classify_form(manual_value)
    base = {
        "auto_form": auto_form,
        "manual_form": manual_form,
        "granularity_direction": "",
        "code_side": "",
        "code_inferable": "",
        "primary_mode": "unclear",
        "mode_rank": 7,
        "numeric_outcome": "",
        "numeric_exact": None,
        "mode_source": "heuristic",
    }
    if _is_empty(auto_value) or _is_empty(manual_value):
        base["primary_mode"] = "unclear"
        base["note"] = "not_both_populated"
        return base

    a, m = _norm_text(auto_value), _norm_text(manual_value)

    # 0) Shared numeric equality checker (unit / decimal / sci notation)
    num = classify_numeric_pair(auto_value, manual_value)
    base["numeric_outcome"] = num.get("outcome") or ""
    base["numeric_exact"] = num.get("numeric_exact")
    if num.get("numeric_exact") == 1.0:
        # equal / unit_format_only / unit_convertible → format_only disagreement
        base.update(primary_mode="format_only", mode_rank=1)
        return base
    if (
        num.get("numeric_exact") == 0.0
        and num.get("outcome")
        in {"value_mismatch", "review_harmonized_diff"}
    ):
        base.update(primary_mode="different_value_selected", mode_rank=4)
        return base

    # 1 format only (non-numeric string/format heuristics)
    if _format_only(a, m):
        base.update(primary_mode="format_only", mode_rank=1)
        return base

    # 2 granularity
    gdir = _granularity(a, m)
    if gdir:
        base.update(
            primary_mode="granularity",
            mode_rank=2,
            granularity_direction=gdir,
        )
        return base

    # 3 code vs prose
    a_code, m_code = _is_code(a), _is_code(m)
    a_prose, m_prose = _is_prose(a), _is_prose(m)
    if (m_code and a_prose and not a_code) or (a_code and m_prose and not m_code):
        if m_code and a_prose:
            code, prose, side = m, a, "ref"
        else:
            code, prose, side = a, m, "auto"
        base.update(
            primary_mode="code_vs_prose",
            mode_rank=3,
            code_side=side,
            code_inferable=_code_inferable(code, prose),
        )
        return base
    # short code vs longer descriptive (even if not full prose)
    if m_code and not a_code and len(a) > len(m) + 15:
        base.update(
            primary_mode="code_vs_prose",
            mode_rank=3,
            code_side="ref",
            code_inferable=_code_inferable(m, a),
        )
        return base
    if a_code and not m_code and len(m) > len(a) + 15:
        base.update(
            primary_mode="code_vs_prose",
            mode_rank=3,
            code_side="auto",
            code_inferable=_code_inferable(a, m),
        )
        return base

    # 4 different value — only if numeric checker did not already decide
    # (legacy crude digit extract removed; it misfired on European decimals)

    # 5 content conflict (contradictory short answers / different codes)
    if a_code and m_code and _casefold_alnum(a) != _casefold_alnum(m):
        base.update(primary_mode="content_conflict", mode_rank=5)
        return base
    if _YES_NO.match(m.strip()) and a_prose:
        # e.g. No vs "Sensitivity analysis performed…"
        base.update(
            primary_mode="content_conflict",
            mode_rank=5,
            code_side="ref",
            code_inferable="no",
        )
        return base

    # 6 wrong field
    if _wrong_field(field, a, m):
        base.update(primary_mode="wrong_field", mode_rank=6)
        return base

    # 7 unclear
    base.update(primary_mode="unclear", mode_rank=7)
    return base


# --- LLM disagreement classes (supersede PRIMARY_MODES for LLM labeling) ---
# Ordered decision list: the judge applies these in order and stops at the
# first that fits. ``false_empty`` is a pre-filter (one side only states that
# no value is available, so the cell is really one-side-empty, not
# both-populated); ``empty_side`` records which side that text comes from.
DISAGREEMENT_CLASSES: tuple[tuple[str, str, str], ...] = (
    (
        "false_empty",
        "One side holds text but no value: it only states that the datum is "
        "missing or not applicable ('Not mentioned', 'not stated', 'not "
        "reported', 'unavailable', 'n/a', 'null', '-', a list whose entries "
        "are all null), optionally with a short reason ('N/A: \"upscaled\"'). "
        "The cell is effectively empty on that side. Not if the side also "
        "gives a value or partial content.",
        "GaBi vs 'Not mentioned'; 0.42 kg CO2 eq vs 'not stated in the paper'",
    ),
    (
        "format_only",
        "Same content in the same unit; differs only in delimiters, notation, "
        "rounding or case.",
        "2092755 L/year vs 2,092,755; 1.2E-3 vs 0,0012",
    ),
    (
        "different_unit",
        "Same data expressed in a different unit; both are correct and agree "
        "once converted (within rounding). Write the conversion.",
        "0.18 (water/slag ratio) vs 18 wt% water; 278 t CO2-eq vs 278000 kg CO2e",
    ),
    (
        "unit_not_converted",
        "One side reports the paper's number in a unit other than the "
        "field's target unit without converting it, so the values disagree "
        "(typically same significant digits, off by a power of ten). Record "
        "which side is in the target unit.",
        "1.88e-07 vs 1.88e-10 when the paper reports 1.88e-7 g and the field "
        "wants kg",
    ),
    (
        "near_match",
        "Same quantity on the same basis, values within ~5% of each other "
        "(rounding, value read off a figure).",
        "8.9e-11 vs 9.2e-11 kg CFC-11 eq",
    ),
    (
        "value_subset",
        "One side lists several separate values (per sub-process, variant or "
        "metric) and the other side gives one of them, matching. Applies in "
        "either direction; record it (auto_finer = auto has the list, "
        "auto_coarser = the reference has the list).",
        "480 (extraction), 550 (carbonation) vs 550; 20 bar vs "
        "1 bar (extraction), 20 bar (carbonation)",
    ),
    (
        "granularity",
        "The same single value described at a different level of detail (a "
        "specific item vs its broader category). Not for lists of separate "
        "values (use value_subset). Record the direction.",
        "Northeast US softwood biomass vs Wood",
    ),
    (
        "code_vs_prose",
        "One side is a code, flag or digit and the other describes it in prose. "
        "Record whether the prose supports the code; if the prose contradicts "
        "the code, use content_conflict instead.",
        "Yes vs 'World ReCiPe H/H Weighting'",
    ),
    (
        "different_reference_basis",
        "Same quantity but a different functional unit, normalisation, or "
        "life-cycle stage / scope (e.g. per farm vs per ha; stage credit vs "
        "total).",
        "7788 kg CO2e vs 5.50 t CO2-eq/ha; EoL credit vs cradle-to-grave total",
    ),
    (
        "different_attribute",
        "Both values plausibly fit the field name but describe different "
        "properties of it (signals an ambiguous field definition, not an "
        "extraction error).",
        "Concentrate composition (%) vs 47 t DM/year",
    ),
    (
        "different_value_selected",
        "Same attribute and same basis, but a different value was chosen: two "
        "numbers, or two alternatives of the same kind (two database names, "
        "two categories from one label set). Includes zero vs non-zero.",
        "1117 vs 1700 kWh/m2a; cell vs module; ecoinvent vs GaBi; 0 vs 0.214",
    ),
    (
        "complementary",
        "Both are partially correct and cover different parts of a "
        "multi-part answer.",
        "GaBi + CROPWAT + company data vs ISO 14040/44-based literature",
    ),
    (
        "content_conflict",
        "The values are logically incompatible: one asserts what the other "
        "negates (Yes/No, present vs absent, a claim vs its opposite). Two "
        "different alternatives of the same kind are different_value_selected, "
        "not content_conflict.",
        "No vs 'excess electricity sold back to the grid'",
    ),
    (
        "wrong_field",
        "The auto value is correct content for a different field.",
        "'baking soda' for Product use/application",
    ),
    (
        "unclear",
        "Cannot be decided from the two values, the field description and the "
        "production-system names.",
        "",
    ),
)
DISAGREEMENT_CLASS_NAMES: tuple[str, ...] = tuple(c[0] for c in DISAGREEMENT_CLASSES)
DISAGREEMENT_CLASS_DEFINITIONS: dict[str, str] = {
    c[0]: c[1] for c in DISAGREEMENT_CLASSES
}

# Broader category per class (a pure mapping; the judge only picks the
# fine class). "Same value, different representation" = the right datum was
# found and only its representation differs (notation, unit, rounding, code
# vs prose); unit_not_converted sits here too although the stored number is
# wrong in the field's target unit.
DISAGREEMENT_BROAD_CATEGORY: dict[str, str] = {
    "false_empty": "False empty (one side states no value)",
    "format_only": "Same value, different representation",
    "different_unit": "Same value, different representation",
    "unit_not_converted": "Same value, different representation",
    "near_match": "Same value, different representation",
    "code_vs_prose": "Same value, different representation",
    "value_subset": "Partial overlap",
    "granularity": "Partial overlap",
    "complementary": "Partial overlap",
    "different_reference_basis": "Different basis or attribute",
    "different_attribute": "Different basis or attribute",
    "different_value_selected": "Different value",
    "content_conflict": "Different value",
    "wrong_field": "Different value",
    "unclear": "Unclear",
}
BROAD_CATEGORY_ORDER: tuple[str, ...] = tuple(
    dict.fromkeys(DISAGREEMENT_BROAD_CATEGORY[c] for c in DISAGREEMENT_CLASS_NAMES)
)


def _class_table_text() -> str:
    lines = []
    for i, (name, definition, example) in enumerate(DISAGREEMENT_CLASSES):
        ex = f" Example: {example}." if example else ""
        lines.append(f"{i}. {name} — {definition}{ex}")
    return "\n".join(lines)


def create_disagreement_mode_judge(
    model: str = "openai/gpt-5.4-mini-2026-03-17",
    api_key: str | None = None,
    temperature: float | None = None,
    max_workers: int | None = None,
    n_runs: int = 3,
    reasoning_effort: str | None = "low",
) -> Callable[..., dict[str, Any]]:
    """LLM judge that assigns one ``DISAGREEMENT_CLASSES`` label per cell.

    Each cell is judged up to ``n_runs`` times (GPT-5 takes no temperature,
    so single runs flip between neighbouring classes) and the majority class
    wins. Runs stop early once a majority is certain (with 3 runs: 2 agreeing
    votes skip the third call); details (rationale, side flags) come from the first run with that
    class. No majority → ``unclear``. ``run_modes`` / ``run_agreement``
    record the individual votes.

    Every cell goes to the LLM (no numeric short-circuit: the numeric checker
    mislabels unit conversions, near matches and multi-value cells). Output
    is constrained by a strict JSON schema (structured outputs).

    Returns ``judge(field, auto, manual, *, field_description="",
    scenario_auto="", scenario_manual="") -> dict``; ``judge.judge_many(rows)``
    annotates row dicts in parallel. Mistral models are rejected because the
    extraction pipeline itself runs on Mistral (self-preference bias).
    """
    if "mistral" in model.lower():
        raise ValueError(
            f"Refusing judge model {model!r}: the extractor is Mistral, so a "
            "Mistral judge would be biased. Use another provider."
        )
    try:
        from litellm import completion
    except ImportError as e:
        raise ImportError(
            "litellm is required for LLM disagreement-mode labeling. "
            "Install it with: pip install litellm"
        ) from e

    # rationale comes first so the model reasons before committing to a class.
    MODE_SCHEMA = {
        "type": "json_schema",
        "json_schema": {
            "name": "disagreement_class",
            "strict": True,
            "schema": {
                "type": "object",
                "properties": {
                    "rationale": {"type": "string"},
                    "unit_conversion": {"type": "string"},
                    "primary_mode": {
                        "type": "string",
                        "enum": list(DISAGREEMENT_CLASS_NAMES),
                    },
                    "granularity_direction": {
                        "type": "string",
                        "enum": ["", "auto_finer", "auto_coarser"],
                    },
                    "code_side": {"type": "string", "enum": ["", "ref", "auto"]},
                    "code_supports": {
                        "type": "string",
                        "enum": ["", "yes", "no", "unclear"],
                    },
                    "empty_side": {
                        "type": "string",
                        "enum": ["", "auto", "ref", "both"],
                    },
                    "unit_correct_side": {
                        "type": "string",
                        "enum": ["", "auto", "ref", "unclear"],
                    },
                },
                "required": [
                    "rationale",
                    "unit_conversion",
                    "primary_mode",
                    "granularity_direction",
                    "code_side",
                    "code_supports",
                    "empty_side",
                    "unit_correct_side",
                ],
                "additionalProperties": False,
            },
        },
    }

    system_prompt = f"""You classify WHY an automatic LCA data extraction (auto) disagrees with a human reference value (ref) for the same field, paper and production system.

Classes — apply in order and stop at the FIRST that fits:
{_class_table_text()}

Rules:
- Compare significant digits: if both sides have the same digits but differ by a power of ten, it is unit_not_converted, even when neither side states a unit.
- If the units differ, convert one side to the other's units and write the conversion in unit_conversion (e.g. "14.7 psi = 1.01 bar"); otherwise leave unit_conversion empty. If the converted values agree, it is different_unit; otherwise use the converted values when deciding between near_match and different_value_selected.
- Use the field description and the production-system names to judge whether both sides describe the same attribute and basis.
- granularity_direction: set only for granularity and value_subset (auto_finer = auto more detailed / auto has the list). Otherwise "".
- code_side and code_supports: set only for code_vs_prose (code_side = which side is the code; code_supports = whether the prose supports the code). Otherwise "".
- empty_side: set only for false_empty (which side only states that no value is available; "both" if both do). Otherwise "".
- unit_correct_side: set only for unit_not_converted (which side is in the field's target unit, from the field description). Otherwise "".
- rationale: 1-3 sentences, written before choosing the class."""

    workers = max_workers
    if workers is None:
        env_w = os.environ.get("LLM_JUDGE_WORKERS")
        workers = int(env_w) if env_w and env_w.isdigit() else 16
    workers = max(1, int(workers))

    cache: dict[tuple[str, ...], dict[str, Any]] = {}
    cache_lock = threading.Lock()
    stats = {"calls": 0, "hits": 0, "errors": 0}
    stats_lock = threading.Lock()
    rank = {m: i for i, m in enumerate(DISAGREEMENT_CLASS_NAMES)}

    def _call_api(
        field: str,
        auto: str,
        manual: str,
        field_description: str,
        scenario_auto: str,
        scenario_manual: str,
    ) -> dict[str, Any]:
        user_prompt = (
            f"Field: {field!r}\n"
            f"Field description (from the extraction schema): {field_description or '(none)'}\n"
            f"Auto production system: {scenario_auto or '(none)'}\n"
            f"Reference production system: {scenario_manual or '(none)'}\n"
            f"Auto value: {auto!r}\n"
            f"Reference value: {manual!r}"
        )
        kwargs: dict[str, Any] = {}
        if temperature is not None:
            kwargs["temperature"] = temperature
        if reasoning_effort:
            kwargs["reasoning_effort"] = reasoning_effort
        max_attempts = int(os.environ.get("LLM_JUDGE_RETRIES", "6"))
        last_err: Exception | None = None
        for attempt in range(max_attempts):
            try:
                response = completion(
                    model=model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                    api_key=api_key,
                    response_format=MODE_SCHEMA,
                    **kwargs,
                )
                data = json.loads(response.choices[0].message.content)
                mode = data["primary_mode"]
                return {
                    "primary_mode": mode,
                    "mode_rank": rank[mode],
                    "broad_category": DISAGREEMENT_BROAD_CATEGORY[mode],
                    "class_definition": DISAGREEMENT_CLASS_DEFINITIONS[mode],
                    "granularity_direction": data["granularity_direction"],
                    "code_side": data["code_side"],
                    "code_supports": data["code_supports"],
                    "empty_side": data["empty_side"],
                    "unit_correct_side": data["unit_correct_side"],
                    "unit_conversion": data["unit_conversion"],
                    "rationale": data["rationale"],
                    "mode_source": "llm",
                    "judge_model": getattr(response, "model", model),
                }
            except Exception as e:
                # Billing / quota failures will not recover with retries.
                if any(
                    t in str(e).lower()
                    for t in ("no credits", "insufficient_quota", "billing")
                ):
                    raise RuntimeError(f"LLM judge unusable: {e}") from e
                last_err = e
                time.sleep(min(30.0, 0.5 * (2**attempt)))
        with stats_lock:
            stats["errors"] += 1
        print(f"Warning: disagreement-class LLM error: {last_err}")
        # Not a label: an error must never count as an ``unclear`` vote.
        return {
            "primary_mode": "",
            "mode_rank": "",
            "broad_category": "",
            "class_definition": "",
            "rationale": f"LLM error: {last_err}",
            "mode_source": "llm_error",
        }

    def _majority(runs: list[dict[str, Any]]) -> dict[str, Any]:
        ok = [r for r in runs if r.get("mode_source") != "llm_error"]
        if not ok:
            out = dict(runs[0])
            out["run_modes"] = ""
            out["run_agreement"] = f"0/{len(runs)}"
            return out
        n_err = len(runs) - len(ok)
        runs = ok
        modes = [r["primary_mode"] for r in runs]
        top, top_n = Counter(modes).most_common(1)[0]
        if top_n * 2 <= len(modes):
            out = {
                "primary_mode": "unclear",
                "mode_rank": rank["unclear"],
                "broad_category": DISAGREEMENT_BROAD_CATEGORY["unclear"],
                "class_definition": DISAGREEMENT_CLASS_DEFINITIONS["unclear"],
                "rationale": "No majority across runs: "
                + " | ".join(f"{r['primary_mode']}: {r.get('rationale', '')}" for r in runs),
                "mode_source": "llm_no_majority",
                "judge_model": runs[0].get("judge_model", model),
            }
        else:
            out = dict(next(r for r in runs if r["primary_mode"] == top))
            if len(runs) > 1:
                out["mode_source"] = "llm_majority"
        out["run_modes"] = "|".join(modes)
        out["run_agreement"] = f"{top_n}/{len(modes)}"
        if n_err:
            out["mode_source"] += f"_{n_err}err"
        return out

    def judge(
        field: str,
        auto: Any,
        manual: Any,
        *,
        field_description: str = "",
        scenario_auto: Any = "",
        scenario_manual: Any = "",
    ) -> dict[str, Any]:
        a, m = _norm_text(auto), _norm_text(manual)
        sa, sm = _norm_text(scenario_auto), _norm_text(scenario_manual)
        key = (str(field), a, m, sa, sm)
        with cache_lock:
            if key in cache:
                with stats_lock:
                    stats["hits"] += 1
                return dict(cache[key])
        runs: list[dict[str, Any]] = []
        need = n_runs // 2 + 1  # votes that settle the majority
        while len(runs) < n_runs:
            runs.append(_call_api(str(field), a, m, field_description or "", sa, sm))
            with stats_lock:
                stats["calls"] += 1
            votes = Counter(
                r["primary_mode"] for r in runs if r.get("mode_source") != "llm_error"
            )
            if votes and votes.most_common(1)[0][1] >= need:
                break
        result = {
            "auto_form": classify_form(auto),
            "manual_form": classify_form(manual),
            **_majority(runs),
        }
        if result.get("mode_source") != "llm_error":  # retry failures next time
            with cache_lock:
                cache[key] = result
        return dict(result)

    def judge_many(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Annotate row dicts (field/auto_value/manual_value + context)."""
        from concurrent.futures import ThreadPoolExecutor

        def _one(r: dict[str, Any]) -> dict[str, Any]:
            mode = judge(
                r.get("field", ""),
                r.get("auto_value"),
                r.get("manual_value"),
                field_description=r.get("field_description", ""),
                scenario_auto=r.get("scenario_auto", ""),
                scenario_manual=r.get("scenario_manual", ""),
            )
            # Drop stale heuristic attributes so they can't be misread.
            base = {
                k: v
                for k, v in r.items()
                if k not in ("code_inferable", "numeric_outcome", "numeric_exact")
            }
            return {**base, **mode}

        with ThreadPoolExecutor(max_workers=workers) as ex:
            return list(ex.map(_one, rows))

    judge.judge_many = judge_many  # type: ignore[attr-defined]
    judge.stats = stats  # type: ignore[attr-defined]
    return judge
