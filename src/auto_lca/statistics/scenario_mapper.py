"""Align extracted paper scenarios to literature-review GT rows.

This is a join-key mapper, not a field-level LLM judge. The review often
rewrites titles as product/feedstock/technology (or route/feedstock) while
the extractor keeps the paper's own names. One LLM call per paper returns
a 1-to-1 correspondence; leftover extracted rows are paper-only variants.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from typing import Callable

import pandas as pd

try:
    from litellm import completion

    LITELLM_AVAILABLE = True
except (ImportError, AttributeError):
    LITELLM_AVAILABLE = False
    completion = None

# Paid-profile mapper default from model_profiles.json (not a runner hardcode).
from auto_lca.process.sog.model_profiles import get_profile, to_litellm_model

DEFAULT_MAPPER_MODEL = to_litellm_model(get_profile("paid")["scenario_mapper_model"])

# Review identity slots (present on some GT sheets, absent on others).
IDENTITY_COLS = (
    "SCENARIO CHARACTERIZATION_PRODUCT",
    "SCENARIO CHARACTERIZATION_EMERGING TECHNOLOGY",
    "SCENARIO CHARACTERIZATION_TECHNOLOGY CATEGORY",
    "FEEDSTOCK_FEEDSTOCK",
    "SCENARIO AND PRODUCT IDENTITY_CARBONATION ROUTE",
    "SCENARIO AND PRODUCT IDENTITY_SPECIFIC ROUTE NAME",
    "SCENARIO AND PRODUCT IDENTITY_MAIN PRODUCT",
    "ALKALINE FEEDSTOCK AND MATERIALS_ALKALINE FEEDSTOCK",
)

ALIGN_TOOL = [
    {
        "type": "function",
        "function": {
            "name": "align_scenarios",
            "description": (
                "Primary 1-to-1 alignment of extracted paper scenarios to "
                "literature-review rows, plus paper variants the review collapsed."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "pairs": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "extracted_id": {"type": "string"},
                                "gt_id": {"type": "string"},
                                "confidence": {"type": "number"},
                                "reason": {"type": "string"},
                            },
                            "required": [
                                "extracted_id",
                                "gt_id",
                                "confidence",
                                "reason",
                            ],
                        },
                    },
                    "collapsed": {
                        "type": "array",
                        "description": (
                            "Extracted scenarios that are variants of a paired "
                            "review row (same pathway; review kept one row)."
                        ),
                        "items": {
                            "type": "object",
                            "properties": {
                                "extracted_id": {"type": "string"},
                                "gt_id": {"type": "string"},
                                "reason": {"type": "string"},
                            },
                            "required": ["extracted_id", "gt_id", "reason"],
                        },
                    },
                    "unmatched_extracted": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "id": {"type": "string"},
                                "reason": {"type": "string"},
                            },
                            "required": ["id", "reason"],
                        },
                    },
                },
                "required": ["pairs", "collapsed", "unmatched_extracted"],
            },
        },
    }
]

SYSTEM_PROMPT = """You align LCA scenarios extracted from a source paper with the rows a literature review coded for that same paper.

The review does NOT keep the paper's scenario titles. It recodes each case as an identity sentence (product + feedstock + conversion, or route + feedstock). Extracted scenarios use the paper's own names and descriptions. A review row often COLLAPSES several paper scenarios of the same pathway into one coded row.

Your job is a join, not a quality score:
- pairs: 1-to-1. Each review id and each extracted id is used at most once here. Map the review row to the paper's base/primary case for that product / feedstock / conversion pathway.
- collapsed: other extracted scenarios that are variants of a paired review row (electricity mix, end-of-life, location, temperature/pressure, allocation, 50% vs 100% of the same product). Use the SAME gt_id as that row's primary pair. Do not also list them in pairs or unmatched_extracted.
- unmatched_extracted: a different pathway with no review row. Not a variant of a paired row.
- If nothing extracted corresponds to a review row, do not force a pair.
- Cosine hints are title-overlap only. Use them when they point at the same pathway; ignore them when the product or feedstock differs.
- Do not refuse a pair only because the titles look different.
- Exact or near-exact titles (e.g. extracted "Farm 3" and review "Farm 3") MUST go in pairs, not unmatched_extracted.
- When the review has both an Average/mean aggregate row AND the individual units (Farm 1..N), pair each unit to its unit row; do not dump units into unmatched_extracted. Pair or leave the Average row separately.
- When the review has ONLY an Average/mean (or other aggregate) and the paper extracted Farm/Site/Case 1..N of that same system, put those units in collapsed under the Average gt_id — not unmatched_extracted.

Call align_scenarios with the mapping."""


def _cell_text(val) -> str:
    if val is None:
        return ""
    try:
        if pd.isna(val):
            return ""
    except (TypeError, ValueError):
        pass
    s = str(val).strip()
    if not s or s.lower() in {"nan", "none", "null"}:
        return ""
    return s


def _identity_lines(row: pd.Series) -> str:
    lines = []
    for col in IDENTITY_COLS:
        val = _cell_text(row.get(col))
        if not val:
            continue
        label = col.rsplit("_", 1)[-1].title()
        lines.append(f"{label}: {val}")
    return "\n".join(lines)


def card_from_auto_row(row: pd.Series, sid: str) -> dict:
    desc = _cell_text(row.get("SCENARIO_SCENARIO_DESCRIPTION"))
    if len(desc) > 600:
        desc = desc[:597] + "..."
    return {
        "id": sid,
        "name": _cell_text(row.get("SCENARIO NAME")),
        "description": desc,
        "category": _cell_text(row.get("SCENARIO_SCENARIO_CATEGORY")),
    }


def card_from_gt_row(row: pd.Series, sid: str) -> dict:
    return {
        "id": sid,
        "name": _cell_text(row.get("SCENARIO NAME")),
        "identity": _identity_lines(row),
    }


@dataclass
class PaperMapping:
    """auto_name -> (manual_name or None, confidence, reason)."""

    pairs: dict = field(default_factory=dict)
    collapsed: dict = field(default_factory=dict)
    unmatched_extracted: dict = field(default_factory=dict)
    raw: dict | None = None
    error: str | None = None
    fallback: bool = False


class ScenarioMapper:
    def __init__(
        self,
        model: str = DEFAULT_MAPPER_MODEL,
        api_key: str | None = None,
        temperature: float = 0.0,
        max_retries: int = 3,
    ):
        if not LITELLM_AVAILABLE:
            raise ImportError("litellm is required for the scenario mapper")
        self.model = model
        self.api_key = api_key or os.environ.get("MISTRAL_API_KEY")
        self.temperature = temperature
        self.max_retries = max_retries

    def _call(self, user_prompt: str) -> dict:
        last_err: Exception | None = None
        for attempt in range(self.max_retries):
            try:
                resp = completion(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": user_prompt},
                    ],
                    tools=ALIGN_TOOL,
                    tool_choice="required",
                    temperature=self.temperature,
                    top_p=1.0 if self.temperature == 0.0 else None,
                    api_key=self.api_key,
                    max_tokens=6000,
                )
                message = resp.choices[0].message
                tool_calls = getattr(message, "tool_calls", None)
                if not tool_calls:
                    raise ValueError("Model did not return a tool call")
                raw_args = tool_calls[0].function.arguments
                return json.loads(raw_args) if isinstance(raw_args, str) else raw_args
            except Exception as e:  # noqa: BLE001
                last_err = e
                print(
                    f"⚠️ scenario mapper failed (attempt {attempt + 1}/{self.max_retries}): {e}"
                )
                time.sleep(2.0 * (attempt + 1))
        raise RuntimeError(
            f"scenario mapper failed after {self.max_retries} attempts"
        ) from last_err

    def map_paper(
        self,
        extracted: list[dict],
        gt_rows: list[dict],
        cosine_hints: dict[str, list[tuple[str, float]]] | None = None,
        cosine_fallback: dict[str, tuple[str | None, float]] | None = None,
    ) -> PaperMapping:
        """Return mapping keyed by extracted scenario name."""
        if not extracted or not gt_rows:
            return PaperMapping()

        ext_by_id = {c["id"]: c for c in extracted}
        gt_by_id = {c["id"]: c for c in gt_rows}
        name_by_ext_id = {c["id"]: c["name"] for c in extracted}
        name_by_gt_id = {c["id"]: c["name"] for c in gt_rows}

        blocks = ["EXTRACTED SCENARIOS (paper language):"]
        for c in extracted:
            blocks.append(
                f"[{c['id']}] {c['name']}\n"
                f"  category: {c.get('category') or 'unspecified'}\n"
                f"  description: {c.get('description') or '(none)'}"
            )
        blocks.append("\nREVIEW ROWS (coded identity, not paper titles):")
        for c in gt_rows:
            hint_txt = ""
            if cosine_hints and c["name"] in cosine_hints:
                bits = [
                    f"{eid} ({sim:.2f})"
                    for eid, sim in cosine_hints[c["name"]][:3]
                    if eid in ext_by_id
                ]
                if bits:
                    hint_txt = f"\n  cosine_hint: {', '.join(bits)}"
            ident = c.get("identity") or "(no extra identity slots)"
            blocks.append(f"[{c['id']}] {c['name']}\n  {ident}{hint_txt}")

        try:
            raw = self._call("\n".join(blocks))
        except Exception as e:  # noqa: BLE001
            mapping = _fallback_cosine(extracted, cosine_fallback, str(e))
            mapping.error = str(e)
            mapping.fallback = True
            return mapping

        try:
            return _enforce_one_to_one(
                raw,
                name_by_ext_id,
                name_by_gt_id,
                extracted,
                cosine_fallback,
            )
        except Exception as e:  # noqa: BLE001
            mapping = _fallback_cosine(extracted, cosine_fallback, str(e))
            mapping.error = str(e)
            mapping.fallback = True
            mapping.raw = raw if isinstance(raw, dict) else {"raw": raw}
            return mapping


def _fallback_cosine(
    extracted: list[dict],
    cosine_fallback: dict[str, tuple[str | None, float]] | None,
    error: str,
) -> PaperMapping:
    pairs = {}
    unmatched = {}
    cosine_fallback = cosine_fallback or {}
    for c in extracted:
        partner, sim = cosine_fallback.get(c["name"], (None, 0.0))
        if partner:
            pairs[c["name"]] = (partner, float(sim or 0.0), f"fallback_cosine ({error})")
        else:
            unmatched[c["name"]] = f"fallback_unmatched ({error})"
    return PaperMapping(pairs=pairs, unmatched_extracted=unmatched)


def _as_pair_dict(pair) -> dict | None:
    if isinstance(pair, dict):
        return pair
    if isinstance(pair, (list, tuple)) and len(pair) >= 2:
        conf = pair[2] if len(pair) > 2 else 0.0
        reason = pair[3] if len(pair) > 3 else "mapped"
        return {
            "extracted_id": pair[0],
            "gt_id": pair[1],
            "confidence": conf,
            "reason": reason,
        }
    return None


def _enforce_one_to_one(
    raw: dict,
    name_by_ext_id: dict[str, str],
    name_by_gt_id: dict[str, str],
    extracted: list[dict],
    cosine_fallback: dict[str, tuple[str | None, float]] | None,
) -> PaperMapping:
    if isinstance(raw, list):
        raw = {"pairs": raw, "collapsed": [], "unmatched_extracted": []}
    raw = raw or {}
    if not isinstance(raw, dict):
        raw = {}
    candidates = []
    for pair in raw.get("pairs") or []:
        pair = _as_pair_dict(pair)
        if not pair:
            continue
        eid = str(pair.get("extracted_id") or "").strip()
        gid = str(pair.get("gt_id") or "").strip()
        if eid not in name_by_ext_id or gid not in name_by_gt_id:
            continue
        try:
            conf = float(pair.get("confidence") or 0.0)
        except (TypeError, ValueError):
            conf = 0.0
        reason = str(pair.get("reason") or "").strip() or "mapped"
        candidates.append((conf, eid, gid, reason))
    candidates.sort(key=lambda x: x[0], reverse=True)

    used_ext = set()
    used_gt = set()
    pairs: dict[str, tuple[str, float, str]] = {}
    for conf, eid, gid, reason in candidates:
        if eid in used_ext or gid in used_gt:
            continue
        used_ext.add(eid)
        used_gt.add(gid)
        pairs[name_by_ext_id[eid]] = (name_by_gt_id[gid], conf, reason)

    collapsed: dict[str, tuple[str, str]] = {}
    for item in raw.get("collapsed") or []:
        item = _as_pair_dict(item)
        if not item:
            continue
        eid = str(item.get("extracted_id") or "").strip()
        gid = str(item.get("gt_id") or "").strip()
        if eid not in name_by_ext_id or gid not in name_by_gt_id:
            continue
        if eid in used_ext:
            continue
        if gid not in used_gt:
            continue
        used_ext.add(eid)
        reason = str(item.get("reason") or "").strip() or "paper_variant"
        collapsed[name_by_ext_id[eid]] = (name_by_gt_id[gid], reason)

    unmatched_reasons = {}
    for item in raw.get("unmatched_extracted") or []:
        if isinstance(item, dict):
            eid = str(item.get("id") or "").strip()
            reason = str(item.get("reason") or "unmatched")
        elif isinstance(item, (list, tuple)) and item:
            eid = str(item[0]).strip()
            reason = str(item[1]) if len(item) > 1 else "unmatched"
        else:
            eid = str(item).strip()
            reason = "unmatched"
        if eid in name_by_ext_id and eid not in used_ext:
            unmatched_reasons[name_by_ext_id[eid]] = reason

    for c in extracted:
        if c["name"] in pairs or c["name"] in collapsed or c["name"] in unmatched_reasons:
            continue
        unmatched_reasons[c["name"]] = "unmatched"

    return PaperMapping(
        pairs=pairs,
        collapsed=collapsed,
        unmatched_extracted=unmatched_reasons,
        raw=raw,
    )


def cosine_hints_from_mapping(
    extracted: list[dict],
    gt_rows: list[dict],
    cosine_matrix: dict[tuple[str, str], float],
    top_k: int = 3,
) -> dict[str, list[tuple[str, float]]]:
    """Per GT name, top-k extracted ids by cosine."""
    ext_id_by_name = {c["name"]: c["id"] for c in extracted}
    hints: dict[str, list[tuple[str, float]]] = {}
    for g in gt_rows:
        scored = []
        for e in extracted:
            sim = float(cosine_matrix.get((e["name"], g["name"]), 0.0) or 0.0)
            scored.append((ext_id_by_name[e["name"]], sim))
        scored.sort(key=lambda x: x[1], reverse=True)
        hints[g["name"]] = scored[:top_k]
    return hints


MapperFn = Callable[..., PaperMapping]
