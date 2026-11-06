"""
Agent-based SOG (Structured Output Generation) extractor. Replaced the
original batched Mistral-SDK extractor (mistral_sog.MistralExtractor,
removed; see git tag ``legacy-mistral-sog``).

Why this exists: this session's manual, one-off "read the whole paper,
then verify/fill specific fields" subagent passes consistently
out-performed the batched multi-pass Mistral pipeline on ground-truth
enrichment tasks -- fewer fabricated citations, no field-masking-by-retry
issues, no section-detection dependency, one comprehensive pass instead of
many fragmented ones. This module formalizes that pattern into reusable,
rerunnable code instead of hand-written one-off scripts each time.

Model-agnostic via litellm (https://docs.litellm.ai/) -- works with any
provider litellm supports (Mistral, Gemini, Together, Fireworks, OpenAI,
Anthropic once a key is available, etc.) through one interface, selected
via a plain `provider/model` string. Defaults to Mistral Large since it's
already a proven, working key in this project with no new setup, but
nothing here is Mistral-specific.

Two modes:
  - enrich_gaps(): given a PDF and a list of specific (concept, field)
    gaps (from gap_analysis.find_gaps), verifies each against the real
    paper text in ONE call and returns only confirmed values -- this is
    the mode exercised and validated this session.
  - full_extract(): full-parity replacement for the legacy extractor's
    sog_pipeline -- identifies every scenario in the paper, then extracts
    every concept in the domain config (study-wide concepts once,
    per-scenario concepts once per identified scenario), producing the
    exact same {metadata, tokens, results, extracted_from} output shape
    sog_pipeline produces, so it plugs into the exact same
    join_all_json_results / run_comparison / aggregate_domains tooling
    unchanged. Required/optional fields follow the config's own
    "str" vs "str | None" type strings (never forced non-nullable).
    Runs concept-chunk extraction calls concurrently (real asyncio
    concurrency, not the blocking-call-under-asyncio.gather bug the
    Mistral pipeline had) -- this is what makes many-scenario papers
    (one dairy paper has 35) tractable: sequential per-scenario calls at
    Mistral's per-paper wall time would be prohibitive.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import random
import time
from dataclasses import dataclass
from typing import Any, ClassVar

import litellm

from auto_lca.process.nlp.extract import TextExtractor
from auto_lca.process.preprocess.config_parser import ensure_json_config
from auto_lca.process.sog.bibliography import strip_bibliography
from auto_lca.process.sog.concept import SAMPLE_CONFIG_PATH, build_concepts_from_config
from auto_lca.process.sog.section_context import (
    build_sections_dict,
    validator_context_from_sections,
)

# Imported verbatim from prompt_templates.py rather than re-written. A hand-abbreviated
# copy of this prompt was tried first and caused real, measured harm: it
# dropped the concrete worked example for the exact mixed-dimension case
# (mean/lowest-10%/highest-10% of a farm survey vs. with/without land-use-
# change emissions) and the explicit include/exclude lists with examples,
# and a from-scratch "evidence" instruction added to compensate then
# over-corrected into enumerating individual units (e.g. "Cow 1".."Cow 30")
# whenever each had its own number, which this verbatim prompt's own
# "statistical breakdowns... of one population" exclusion already guards
# against more precisely. Confirmed via a live re-check after switching to
# this verbatim prompt (see full_extract_async) before trusting it.
from auto_lca.process.sog.prompt_templates import (
    PROMPT_EXTRACT_SCENARIOS,
    SYSTEM_EXTRACT_SCENARIOS,
)
from auto_lca.statistics.gap_analysis import FieldGap

from auto_lca.process.sog.model_profiles import get_profile, to_litellm_model

# Legacy alias (enrich-excel CLI); paid scenario model from model_profiles.json.
DEFAULT_MODEL = to_litellm_model(get_profile("paid")["production_system_model"])


def _to_litellm_model(name: str) -> str:
    """config.json holds bare Mistral model ids (e.g. "mistral-large-2512");
    litellm needs a provider prefix."""
    return to_litellm_model(name)


@dataclass
class EnrichmentResult:
    pid: str
    additions: dict[
        str, dict[str, dict[str, Any]]
    ]  # scenario -> concept -> field -> value
    rejected: list[dict[str, str]]  # [{scenario, concept, field, reason}]
    tokens: dict[str, int]
    raw_model: str


class AgentExtractor(TextExtractor):
    """Litellm-backed SOG extractor. See module docstring."""

    def __init__(
        self,
        config_path: str = SAMPLE_CONFIG_PATH,
        model: str | None = None,
        scenario_model: str | None = None,
        api_key: str | None = None,
        max_retries: int = 3,
        cache_namespace: str | None = None,
        model_profile: str | None = None,
    ):
        super().__init__()
        self.config_path = ensure_json_config(config_path)
        self.api_key = (
            api_key  # None -> litellm falls back to provider's default env var
        )
        self.max_retries = max_retries
        # Salts every prompt_cache_key this instance builds, so a fresh
        # AgentExtractor() can never accidentally hit a cache entry warmed
        # by an earlier process (e.g. this session's own live caching
        # tests, which touched a few real paper PIDs directly). Mistral's
        # prompt cache has no manual clear/purge API -- it's TTL-based --
        # so a run-specific namespace is the only way to guarantee a
        # genuinely cold measurement rather than trusting the TTL already
        # expired.
        self.cache_namespace = cache_namespace or f"run{int(time.time())}"

        with open(self.config_path) as f:
            self.config = json.load(f)
        defaults = self.config.get("defaults", {}) or {}

        # Model split: scenario ID on the stronger model; field extraction on
        # the cheaper one. Defaults come from the Mistral plan; config/CLI
        # overrides are preflight-checked against that plan.
        from auto_lca.process.sog.model_profiles import resolve_profile_models
        from auto_lca.shared.model_preflight import preflight_user_models

        resolved = resolve_profile_models(
            model_profile,
            model=model,
            scenario_model=scenario_model,
            config_defaults=defaults,
            api_key=self.api_key,
        )
        self.model = resolved["model"]
        self.scenario_model = resolved["scenario_model"]
        self.model_profile = resolved["profile"]
        preflight_user_models(
            model=self.model,
            scenario_model=self.scenario_model,
            api_key=self.api_key,
            plan_profile=resolved.get("plan_profile"),
            plan_defaults=resolved.get("plan_defaults"),
            has_user_override=bool(resolved.get("has_user_override")),
        )

        # Scenario / production-system flags (two primary knobs):
        #   extract_scenarios: run identification+classification at all.
        #     False → general literature extractor (one synthetic "Paper" row).
        #   extract_non_primary: fan out concept/LCI extraction for non-primary
        #     categories (alternate_dimension, external_reference). When False
        #     (default), every detected system is still kept and labeled; only
        #     own_system_modelled gets field extraction.
        # Detected candidates are always kept and labeled (never dropped for
        # being external / secondary-axis / subcategory_breakdown). Only
        # category ``other`` is rejected.
        self.extract_scenarios = bool(
            defaults.get(
                "extract_production_systems",
                defaults.get("extract_scenarios", True),
            )
        )
        self.extract_non_primary = self._resolve_extract_non_primary(defaults)

        # Back-compat aliases for older call sites / metadata / prints.
        self.extract_external_scenarios = self.extract_non_primary
        self.exclude_external_scenarios = False
        self.include_statistical_scenarios = True
        self.include_aggregated_scenarios = True

        if defaults.get("exclude_external_scenarios") or defaults.get(
            "exclude_external_production_systems"
        ):
            print(
                "⚠️ exclude_external_* is deprecated: all detected systems are "
                "kept and labeled. Use extract_non_primary to control field "
                "extraction on non-primary categories."
            )

        # Drop bibliography / references citation list from PDF text while
        # keeping any post-refs appendices/tables.
        self.strip_bibliography = bool(defaults.get("strip_bibliography", True))

    @staticmethod
    def _resolve_extract_non_primary(defaults: dict) -> bool:
        """Whether to extract fields for non-primary scenario categories.

        Prefers ``extract_non_primary`` / ``extract_non_primary_production_systems``.
        Legacy: ``extract_external_*=true`` or ``include_aggregated_*=true``
        previously caused non-primary rows to carry field payloads → True.
        """
        if any(
            k in defaults
            for k in (
                "extract_non_primary",
                "extract_non_primary_production_systems",
            )
        ):
            return bool(
                defaults.get(
                    "extract_non_primary_production_systems",
                    defaults.get("extract_non_primary", False),
                )
            )
        if defaults.get("extract_external_production_systems") or defaults.get(
            "extract_external_scenarios"
        ):
            return True
        if defaults.get("include_aggregated_production_systems") or defaults.get(
            "include_aggregated_scenarios"
        ):
            return True
        return False

    # ------------------------------------------------------------------
    # Shared plumbing
    # ------------------------------------------------------------------

    def _prepare_paper_text(
        self, text: str
    ) -> tuple[str, dict[str, Any] | None]:
        """Optionally strip bibliography; return (text, strip_meta_or_None)."""
        if not self.strip_bibliography:
            return text, None
        result = strip_bibliography(text)
        if not result.stripped:
            return text, {
                "strip_bibliography": True,
                "bibliography_stripped": False,
            }
        meta = {
            "strip_bibliography": True,
            "bibliography_stripped": True,
            "bibliography_header": result.header,
            "bibliography_method": result.method,
            "bibliography_chars_removed": result.bib_chars,
            "bibliography_post_refs_chars_kept": result.post_refs_chars,
        }
        return result.text, meta

    @staticmethod
    def _flatten_usage(usage) -> dict:
        """dict(usage) only captures top-level fields -- cached_tokens
        lives nested under prompt_tokens_details, so pull it out
        explicitly to make aggregate cache-hit-rate reporting possible
        (see run_extract's summary print)."""
        flat = dict(usage)
        details = getattr(usage, "prompt_tokens_details", None)
        cached = getattr(details, "cached_tokens", None) if details else None
        if cached is not None:
            flat["cached_tokens"] = cached
        return flat

    def _call(
        self,
        messages: list[dict],
        tools: list[dict],
        max_tokens: int = 16000,
        prompt_cache_key: str | None = None,
        model: str | None = None,
    ) -> tuple[dict, dict]:
        """One litellm call with tool-calling, retried on transient errors.
        Returns (parsed_tool_arguments, usage_dict).

        `prompt_cache_key`: passed through as extra_body (litellm's mistral
        transformation layer doesn't map prompt_cache_key as a known
        OpenAI-style param, so it's dropped if passed directly -- but
        Mistral's actual REST API does accept it as a top-level field, and
        litellm's openai-compatible base transformation merges extra_body
        straight into the request. Confirmed live: a repeat call sharing
        this key AND a shared prompt PREFIX (see callers -- text must come
        first) returned cached_tokens equal to ~99.7% of prompt_tokens,
        vs. 0 on the first call. Does not change prompt_tokens/total_tokens
        in the usage report -- the benefit is lower billed cost and
        latency, not a smaller reported token count."""
        last_err = None
        for attempt in range(self.max_retries):
            try:
                resp = litellm.completion(
                    model=model or self.model,
                    messages=messages,
                    tools=tools,
                    tool_choice="required",
                    api_key=self.api_key,
                    max_tokens=max_tokens,
                    temperature=0.0,
                    top_p=1.0,
                    extra_body=(
                        {"prompt_cache_key": prompt_cache_key}
                        if prompt_cache_key
                        else None
                    ),
                )
                message = resp.choices[0].message
                tool_calls = getattr(message, "tool_calls", None)
                if not tool_calls:
                    raise ValueError("Model did not return a tool call")
                raw_args = tool_calls[0].function.arguments
                args = json.loads(raw_args) if isinstance(raw_args, str) else raw_args
                usage = (
                    self._flatten_usage(resp.usage)
                    if getattr(resp, "usage", None)
                    else {}
                )
                return args, usage
            except Exception as e:
                last_err = e
                print(
                    f"⚠️ agent_sog call failed (attempt {attempt + 1}/{self.max_retries}): {e}"
                )
                time.sleep(2.0 * (attempt + 1))
        raise RuntimeError(
            f"agent_sog call failed after {self.max_retries} attempts"
        ) from last_err

    async def _acall(
        self,
        messages: list[dict],
        tools: list[dict],
        max_tokens: int = 16000,
        prompt_cache_key: str | None = None,
        model: str | None = None,
    ) -> tuple[dict, dict]:
        """Async twin of _call, used by full_extract_async so many concept
        chunks/scenarios can be in flight at once -- genuine concurrency via
        litellm.acompletion, not a blocking call wrapped in asyncio.gather
        (the bug found and fixed in the Mistral pipeline earlier this
        session). See _call's docstring for prompt_cache_key details --
        same mechanism, confirmed live to produce real cache hits."""
        last_err = None
        for attempt in range(self.max_retries):
            try:
                resp = await litellm.acompletion(
                    model=model or self.model,
                    messages=messages,
                    tools=tools,
                    tool_choice="required",
                    api_key=self.api_key,
                    max_tokens=max_tokens,
                    # Large-4 rejects temperature=0 unless top_p=1
                    # (invalid_request_greedy_sampling).
                    temperature=0.0,
                    top_p=1.0,
                    extra_body=(
                        {"prompt_cache_key": prompt_cache_key}
                        if prompt_cache_key
                        else None
                    ),
                )
                message = resp.choices[0].message
                tool_calls = getattr(message, "tool_calls", None)
                if not tool_calls:
                    raise ValueError("Model did not return a tool call")
                raw_args = tool_calls[0].function.arguments
                args = json.loads(raw_args) if isinstance(raw_args, str) else raw_args
                usage = (
                    self._flatten_usage(resp.usage)
                    if getattr(resp, "usage", None)
                    else {}
                )
                return args, usage
            except Exception as e:
                last_err = e
                print(
                    f"⚠️ agent_sog async call failed (attempt {attempt + 1}/{self.max_retries}): {e}"
                )
                # Rate limits (confirmed live, recurring across full_extract
                # runs at max_concurrency=6) need a real cooldown, not the
                # same few-second backoff used for one-off transient
                # glitches -- a 429 right now means the same window is very
                # likely still exceeded 2-6s later.
                if isinstance(e, litellm.exceptions.RateLimitError):
                    await asyncio.sleep(15.0 * (attempt + 1) + random.uniform(0, 5.0))
                    continue
                # Jittered backoff: full_extract_async fires many concurrent
                # chunk calls at once (bounded by max_concurrency), and a
                # fixed backoff means every failed chunk from the same
                # burst retries at the exact same moment, reproducing
                # whatever load-related condition caused the empty
                # completion in the first place (confirmed live: a chunk
                # failed 3/3 retries during a concurrent run, then
                # succeeded 3/3 when replayed one at a time with no
                # concurrent load).
                await asyncio.sleep(2.0 * (attempt + 1) + random.uniform(0, 2.0))
        raise RuntimeError(
            f"agent_sog call failed after {self.max_retries} attempts"
        ) from last_err

    # ------------------------------------------------------------------
    # Enrichment mode (validated this session)
    # ------------------------------------------------------------------

    # Chunk size confirmed live: a 74-gap batch completed cleanly in one
    # call (68 confirmed, 6 rejected, no truncation). Larger batches (232,
    # 312 gaps) truncated mid-JSON even with a raised max_tokens ceiling
    # -- chunking keeps each call's completion length in a range that's
    # actually been observed to complete, rather than gambling on an
    # ever-higher token ceiling for arbitrarily large gap lists.
    GAP_CHUNK_SIZE = 60

    def enrich_gaps(
        self, pdf_path: str, pid: str, gaps: list[FieldGap]
    ) -> EnrichmentResult:
        """
        For every (scenario, concept, field) gap where the extraction
        system found a value the ground truth doesn't have, verify it
        against the real PDF text and return only confirmed additions --
        never fabricate, never blindly copy the system's value.

        Chunks large gap lists (see GAP_CHUNK_SIZE) into multiple calls
        rather than one giant call -- confirmed via a real run that a
        single call covering 232-312 gaps at once truncates mid-JSON
        regardless of max_tokens, the same failure mode found earlier in
        the legacy batched extractor. The paper text is re-sent
        per chunk, cached via prompt_cache_key across chunks of the same
        paper (see _enrich_gap_chunk) -- confirmed live this actually
        produces cache hits, unlike the "prompt-cacheable on the provider
        side" this docstring used to just assert without ever wiring up
        the cache key or the text-first prompt order caching requires.
        """
        text = self.extract_text_from_pdf(pdf_path)
        text, _bib_meta = self._prepare_paper_text(text)
        cache_key = f"{self.cache_namespace}:{pid}:full_text"

        all_additions: dict[str, dict[str, dict[str, Any]]] = {}
        all_rejected: list[dict[str, str]] = []
        total_usage: dict[str, int] = {}
        for start in range(0, len(gaps), self.GAP_CHUNK_SIZE):
            chunk = gaps[start : start + self.GAP_CHUNK_SIZE]
            chunk_additions, chunk_rejected, usage = self._enrich_gap_chunk(
                text, chunk, cache_key=cache_key
            )
            for scenario, concepts in chunk_additions.items():
                for concept, fields in concepts.items():
                    all_additions.setdefault(scenario, {}).setdefault(
                        concept, {}
                    ).update(fields)
            all_rejected.extend(chunk_rejected)
            for k, v in (usage or {}).items():
                if isinstance(v, (int, float)):
                    total_usage[k] = total_usage.get(k, 0) + v

        return EnrichmentResult(
            pid=pid,
            additions=all_additions,
            rejected=all_rejected,
            tokens=total_usage,
            raw_model=self.model,
        )

    def _enrich_gap_chunk(
        self, text: str, gaps: list[FieldGap], cache_key: str | None = None
    ) -> tuple[dict[str, dict[str, dict[str, Any]]], list[dict[str, str]], dict]:
        gap_lines = []
        for i, g in enumerate(gaps):
            gap_lines.append(
                f'{i}. Scenario "{g.scenario_system}" / Concept "{g.concept}" / '
                f'Field "{g.field}": system found -> {g.system_value!r}'
            )

        tool = {
            "type": "function",
            "function": {
                "name": "record_gap_verifications",
                "description": "Report, for each numbered gap, whether the system's value is confirmed by the paper text.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "verdicts": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "gap_id": {"type": "integer"},
                                    "confirmed": {"type": "boolean"},
                                    "value": {
                                        "type": "string",
                                        "description": "The verified value to add (may lightly clean up the system's wording, but preserve numbers/units exactly; only set if confirmed=true). A better/more precise value found directly in the text is preferred over the system's if both exist.",
                                    },
                                    "reason": {
                                        "type": "string",
                                        "description": "One-sentence reason, always required -- especially important when confirmed=false (why it's wrong/unverifiable) so nothing is silently dropped.",
                                    },
                                },
                                "required": ["gap_id", "confirmed", "reason"],
                            },
                        }
                    },
                    "required": ["verdicts"],
                },
            },
        }

        # Text FIRST -- every chunk of the same paper's gap list shares
        # this exact prefix, which is what lets prompt_cache_key actually
        # get cache hits (see agent_sog.py's _call/_acall docstrings;
        # confirmed live).
        prompt = f"""Paper text:
{text}

Verify a list of extracted datapoints against the source paper text above, one at a time. For each, decide whether the system's extracted value is actually supported by the text.

Rules:
- Only mark confirmed=true if you can point to the specific supporting text/table/number.
- Do NOT fabricate citations, sources, or values not actually present.
- If you find a more precise or corrected version of the fact in the text, use that instead of the system's exact wording (but never change the underlying number/unit).
- If the field genuinely isn't answerable from this text, confirmed=false with a reason.

Gaps to verify:
{chr(10).join(gap_lines)}"""

        # Scaled per-gap, not a flat ceiling -- each verdict needs a
        # gap_id, confirmed bool, and a reason (always) plus a value
        # (often), so budget generously per gap rather than guessing one
        # ceiling for every chunk size.
        max_tokens = min(16000, max(2000, 220 * len(gaps)))

        args, usage = self._call(
            messages=[
                {
                    "role": "system",
                    "content": "You are a meticulous research-data verification analyst. You never fabricate facts and always cite what you actually found.",
                },
                {"role": "user", "content": prompt},
            ],
            tools=[tool],
            max_tokens=max_tokens,
            prompt_cache_key=cache_key,
        )

        additions: dict[str, dict[str, dict[str, Any]]] = {}
        rejected: list[dict[str, str]] = []
        verdicts = args.get("verdicts", [])
        by_id = {v.get("gap_id"): v for v in verdicts}
        for i, g in enumerate(gaps):
            v = by_id.get(i)
            if v is None:
                rejected.append(
                    {
                        "scenario": g.scenario_system,
                        "concept": g.concept,
                        "field": g.field,
                        "reason": "model returned no verdict for this gap",
                    }
                )
                continue
            if v.get("confirmed") and v.get("value"):
                additions.setdefault(
                    g.scenario_manual or g.scenario_system, {}
                ).setdefault(g.concept, {})[g.field] = v["value"]
            else:
                rejected.append(
                    {
                        "scenario": g.scenario_system,
                        "concept": g.concept,
                        "field": g.field,
                        "reason": v.get("reason", "not confirmed"),
                    }
                )

        return additions, rejected, usage

    # ------------------------------------------------------------------
    # Full-extraction mode: full parity with the legacy sog_pipeline
    # ------------------------------------------------------------------

    # Fields-per-call budget for concept-extraction chunks. Same reasoning
    # as GAP_CHUNK_SIZE above: a single call trying to extract too many
    # fields at once truncates mid-JSON regardless of max_tokens (confirmed
    # live for gap verification; extraction output is the same shape of
    # problem -- one JSON blob whose size scales with field count). 40
    # fields/chunk keeps every call's completion in the range that's
    # actually been observed to complete.
    CONCEPT_CHUNK_FIELD_BUDGET = 40

    # Concept structure type strings this project's configs actually use
    # (dairy/plastics/fashion configs checked directly), mapped to JSON
    # Schema. "X | None" -> nullable, optional (not in "required"); a bare
    # type -> required, non-nullable -- mirrors the legacy extractor's
    # Optional[...] vs plain-type handling, so a field's required/optional
    # status always matches what the config actually declares, never
    # forced to non-nullable.
    _TYPE_JSON_SCHEMA: ClassVar[dict[str, dict]] = {
        "str": {"type": "string"},
        "str | None": {"type": ["string", "null"]},
        "int": {"type": "integer"},
        "int | None": {"type": ["integer", "null"]},
        "float": {"type": "number"},
        "float | None": {"type": ["number", "null"]},
    }

    # The legacy extractor capped every string field at this length via
    # pydantic's Field(max_length=800), added there because uncapped
    # fields let the model "dump an entire narrative into one field
    # while leaving sibling fields null" -- ported here for the same
    # reason (tool-calling JSON Schema honors maxLength on string
    # properties the same way).
    MAX_STR_FIELD_LENGTH = 800

    @classmethod
    def _field_schema(cls, type_str: str) -> dict:
        schema = dict(
            cls._TYPE_JSON_SCHEMA.get(type_str.strip(), {"type": ["string", "null"]})
        )
        if schema.get("type") == "string" or (
            isinstance(schema.get("type"), list) and "string" in schema["type"]
        ):
            schema["maxLength"] = cls.MAX_STR_FIELD_LENGTH
        return schema

    @classmethod
    def _concepts_tool(cls, concepts: list) -> dict:
        """Build a tool spec whose top-level properties are the given
        concepts' names, each an object of their own fields -- concept
        presence is required (the model must attempt every concept, even
        if every field inside comes back null), but individual fields
        follow the config's own required/optional typing."""
        concept_props = {}
        for c in concepts:
            field_props = {}
            field_required = []
            for field_name, type_str in c.structure.items():
                field_props[field_name] = cls._field_schema(type_str)
                if "None" not in type_str:
                    field_required.append(field_name)
            concept_schema = {"type": "object", "properties": field_props}
            if field_required:
                concept_schema["required"] = field_required
            concept_props[c.name] = concept_schema

        return {
            "type": "function",
            "function": {
                "name": "record_extraction",
                "description": (
                    "Fill in EVERY field individually where the text supports it -- "
                    "do not skip a specific field just because related context "
                    "appears elsewhere, and do not dump everything into one field "
                    "while leaving its siblings null. Use null for anything not "
                    "explicitly stated in the text -- never guess or infer. Always "
                    "include every concept listed, even if every field inside it "
                    "is null."
                ),
                "parameters": {
                    "type": "object",
                    "properties": concept_props,
                    "required": [c.name for c in concepts],
                },
            },
        }

    @classmethod
    def _chunk_concepts(cls, concepts: list) -> list[list]:
        """Split concepts into chunks whose total field count stays under
        CONCEPT_CHUNK_FIELD_BUDGET."""
        chunks: list[list] = []
        current: list = []
        current_fields = 0
        for c in concepts:
            n = len(c.structure)
            if current and current_fields + n > cls.CONCEPT_CHUNK_FIELD_BUDGET:
                chunks.append(current)
                current, current_fields = [], 0
            current.append(c)
            current_fields += n
        if current:
            chunks.append(current)
        return chunks

    async def _identify_scenarios_async(self, text: str) -> tuple[list[dict], dict]:
        tool = {
            "type": "function",
            "function": {
                "name": "record_scenarios",
                "description": "List every distinct scenario in this paper, per the rules given.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "scenarios": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "name": {"type": "string"},
                                    "description": {"type": "string"},
                                },
                                "required": ["name"],
                            },
                        }
                    },
                    "required": ["scenarios"],
                },
            },
        }
        # Verbatim PROMPT_EXTRACT_SCENARIOS/SYSTEM_EXTRACT_SCENARIOS (see
        # import comment above) -- no extra prose bolted on here, since
        # that's exactly what caused the over-enumeration regression this
        # rewrite fixes. Granularity discipline now comes from this proven
        # prompt plus the _filter_valid_scenarios_async validation pass
        # below, matching the legacy extractor's two-stage architecture.
        prompt = PROMPT_EXTRACT_SCENARIOS.format(text=text)
        args, usage = await self._acall(
            messages=[
                {"role": "system", "content": SYSTEM_EXTRACT_SCENARIOS},
                {"role": "user", "content": prompt},
            ],
            tools=[tool],
            max_tokens=8000,
            model=self.scenario_model,
        )
        return args.get("scenarios", []), usage

    # Scenario category taxonomy + cross-candidate dimension filter, ported
    # from the legacy extractor's _filter_valid_scenarios/_drop_minority_dimensions
    # -- the established pipeline's own proven fix for exactly the
    # over-enumeration failure this extractor hit on real papers (e.g. one
    # dairy paper producing 88 "scenarios" where ground truth has 1; the
    # evidence-gating instruction in _identify_scenarios_async's prompt
    # alone did not reliably catch this). Per-candidate classification
    # catches subcategory_breakdown (a table-row category masquerading as
    # a scenario, e.g. "Farm 12" when the results are only ever reported
    # in aggregate); the cross-candidate dimension pass catches the paper
    # slicing the SAME systems along two different axes at once (e.g.
    # mean/Min10/Max10 by percentile AND without-LUC/with-sequestration by
    # accounting variant) -- something no per-candidate check alone can see.
    SCENARIO_CATEGORY_OWN_SYSTEM_MODELLED = "own_system_modelled"
    SCENARIO_CATEGORY_OWN_SYSTEM = SCENARIO_CATEGORY_OWN_SYSTEM_MODELLED  # legacy alias
    SCENARIO_CATEGORY_OWN_SYSTEM_MENTIONED = "own_system_mentioned"
    SCENARIO_CATEGORY_EXTERNAL_REFERENCE = "external_reference"
    SCENARIO_CATEGORY_SUBCATEGORY_BREAKDOWN = "subcategory_breakdown"
    SCENARIO_CATEGORY_OTHER = "other"
    SCENARIO_CATEGORY_UNCLASSIFIED = "unclassified"
    SCENARIO_CATEGORY_ALTERNATE_DIMENSION = "alternate_dimension"

    # How much front-matter to keep for study-scope (own vs external).
    # Modelled-vs-mentioned uses Results/Tables via section_context instead.
    PAPER_CONTEXT_CHARS = 3000
    VALIDATOR_CONTEXT_CHARS = 12000

    async def _filter_valid_scenarios_async(
        self,
        candidates: list[dict],
        paper_context: str = "",
        *,
        preserve_all: bool = False,
    ) -> tuple[list[dict], dict]:
        """Classify candidates into scenario_category tags.

        By default labels every candidate (only ``other`` is dropped). Secondary
        grouping axes are retagged ``alternate_dimension`` but kept. Pass
        ``preserve_all=True`` to also keep ``other`` retagged in place — used
        when refreshing categories on an existing extract.

        The LLM sees candidate name+description plus optional
        ``paper_context`` (Results/Tables/Methods when available, else a
        short front-matter prefix), not concept field payloads.
        """
        if not candidates:
            return candidates, {}

        categories = (
            self.SCENARIO_CATEGORY_OWN_SYSTEM,
            self.SCENARIO_CATEGORY_OWN_SYSTEM_MENTIONED,
            self.SCENARIO_CATEGORY_EXTERNAL_REFERENCE,
            self.SCENARIO_CATEGORY_SUBCATEGORY_BREAKDOWN,
            self.SCENARIO_CATEGORY_OTHER,
        )
        tool = {
            "type": "function",
            "function": {
                "name": "record_scenario_verdicts",
                "description": "Classify each candidate scenario using the category definitions provided.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "verdicts": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "scenario_id": {"type": "integer"},
                                    "category": {
                                        "type": "string",
                                        "enum": list(categories),
                                    },
                                    "grouping_dimension": {
                                        "type": "string",
                                        "description": (
                                            "Only for own_system_modelled / own_system_mentioned: "
                                            "a short (2-4 word) tag naming the axis this "
                                            "scenario varies along relative to the other "
                                            "own-system candidates (e.g. 'percentile "
                                            "distribution', 'management type', 'region', "
                                            "'LUC accounting variant', 'sensitivity bound'). "
                                            "Candidates that are really the same systems "
                                            "sliced a different way must get DIFFERENT tags. "
                                            "Leave empty otherwise."
                                        ),
                                    },
                                },
                                "required": ["scenario_id", "category"],
                            },
                        }
                    },
                    "required": ["verdicts"],
                },
            },
        }

        candidates_str = "\n".join(
            f"{i}. {c.get('name')}: {c.get('description', '')}"
            for i, c in enumerate(candidates)
        )
        context_block = (
            f"Paper context (title/abstract, for scope only):\n{paper_context}\n\n"
            if paper_context
            else ""
        )
        prompt = (
            f"{context_block}"
            "For each candidate below, classify it:\n"
            f"- '{self.SCENARIO_CATEGORY_OWN_SYSTEM}': a distinct system/process/product "
            "THIS paper actually modeled with disclosed inventory and/or quantitative "
            "results (e.g. a row in an LCI/LCIA results table).\n"
            f"- '{self.SCENARIO_CATEGORY_OWN_SYSTEM_MENTIONED}': named or hinted by THIS "
            "paper (often best/worst/base/sensitivity) but WITHOUT a distinct inventory "
            "or results presentation of its own.\n"
            f"- '{self.SCENARIO_CATEGORY_EXTERNAL_REFERENCE}': a system from another cited "
            "study, used only for comparison, not modeled by this paper.\n"
            f"- '{self.SCENARIO_CATEGORY_SUBCATEGORY_BREAKDOWN}': an emission source, "
            "material/feed input, impact category, or statistical bucket describing "
            "something within one system's results rather than a separate system.\n"
            f"- '{self.SCENARIO_CATEGORY_OTHER}': anything else.\n\n"
            "Use the paper context above (if given) to tell whether a site/region/facility "
            "is part of THIS study's own survey (own_system_modelled / own_system_mentioned) rather "
            "than an externally cited study (external_reference) -- e.g. if the abstract "
            "says the study surveyed multiple countries or regions, those regions are "
            "this paper's systems, not external_reference.\n\n"
            "When unsure whether quantitative data exist for the case, prefer "
            f"'{self.SCENARIO_CATEGORY_OWN_SYSTEM_MENTIONED}'.\n\n"
            "For every own_system_modelled or own_system_mentioned candidate, also fill in "
            "grouping_dimension -- this catches cases where the candidate list mixes two "
            "different ways of slicing the SAME underlying systems (e.g. one paper's farms "
            "grouped both by result percentile AND by an accounting variant). If you see "
            "more than one such axis among those candidates, tag each candidate with which "
            "axis it belongs to so the mismatched ones can be identified.\n\n"
            f"Candidates (id. name: description):\n{candidates_str}\n\n"
            "Return exactly one verdict per candidate id listed above."
        )

        try:
            args, usage = await self._acall(
                messages=[
                    {
                        "role": "system",
                        "content": "You are an expert LCA analyst reviewing candidate scenarios for correctness.",
                    },
                    {"role": "user", "content": prompt},
                ],
                tools=[tool],
                max_tokens=4096,
                model=self.scenario_model,
            )
        except Exception as e:
            print(
                f"⚠️ Scenario validation failed ({e}), keeping all candidates unfiltered"
            )
            for c in candidates:
                c.setdefault("scenario_category", self.SCENARIO_CATEGORY_UNCLASSIFIED)
            return candidates, {}

        def _normalize_category(raw: str | None) -> str:
            if raw == "own_system":
                return self.SCENARIO_CATEGORY_OWN_SYSTEM_MODELLED
            if raw in categories:
                return raw
            return self.SCENARIO_CATEGORY_OTHER

        verdicts = args.get("verdicts", [])
        verdict_by_id = {
            v.get("scenario_id"): (
                _normalize_category(v.get("category")),
                (v.get("grouping_dimension") or "").strip().lower(),
            )
            for v in verdicts
        }

        # Always keep labeled systems. Only ``other`` is rejected as noise.
        # Secondary axes are retagged alternate_dimension in
        # _retag_minority_dimensions (never removed). Field extraction for
        # non-primary categories is gated by extract_non_primary later.
        dropped_categories = {self.SCENARIO_CATEGORY_OTHER}

        kept, rejected = [], []
        for i, c in enumerate(candidates):
            verdict = verdict_by_id.get(i)
            if verdict is None:
                c["scenario_category"] = self.SCENARIO_CATEGORY_UNCLASSIFIED
                kept.append(c)
                continue
            category, grouping_dimension = verdict
            c["scenario_category"] = category
            if category in (
                self.SCENARIO_CATEGORY_OWN_SYSTEM,
                self.SCENARIO_CATEGORY_OWN_SYSTEM_MENTIONED,
            ):
                c["_grouping_dimension"] = grouping_dimension
            if category in dropped_categories and not preserve_all:
                rejected.append((c.get("name"), category))
            else:
                if category in dropped_categories and preserve_all:
                    rejected.append((c.get("name"), category))
                kept.append(c)

        if rejected:
            verb = "would drop" if preserve_all else "rejected"
            print(
                f"🧹 Scenario validator {verb} {len(rejected)} candidate(s): {rejected}"
            )

        kept = self._retag_minority_dimensions(kept)
        return (kept if kept else candidates), usage

    @staticmethod
    def _merge_dimension_aliases(dimensions: set[str]) -> dict[str, str]:
        """Map each raw grouping_dimension tag to a canonical one, merging
        tags that name the same axis in different words.

        The classifier tagged one paper's regions 'region + soil drainage
        variant' but one region's rows just 'soil drainage variant'.
        Compared literally those look like rival axes, and the minority
        pass discarded six scenarios the ground truth actually has. Two
        tags are treated as one axis when one's words are a subset of the
        other's.
        """
        alias: dict[str, str] = {}
        groups: list[tuple[set[str], str]] = []
        # Shortest first, so the most general wording becomes canonical.
        for dim in sorted(dimensions, key=lambda d: (len(d.split()), d)):
            words = set(dim.split())
            for group_words, canonical in groups:
                if words <= group_words or group_words <= words:
                    group_words |= words
                    alias[dim] = canonical
                    break
            else:
                groups.append((set(words), dim))
                alias[dim] = dim
        return alias

    def _retag_minority_dimensions(self, kept: list[dict]) -> list[dict]:
        """Retag non-primary grouping axes to ``alternate_dimension``.

        Always keeps every candidate: secondary-axis rows stay in the extract
        as labeled shells (or fully extracted when ``extract_non_primary``).
        """
        own_system = [
            c
            for c in kept
            if c.get("scenario_category") == self.SCENARIO_CATEGORY_OWN_SYSTEM
        ]
        dimensions = {c.get("_grouping_dimension", "") for c in own_system}
        dimensions.discard("")
        alias = self._merge_dimension_aliases(dimensions)
        if len(set(alias.values())) <= 1:
            for c in kept:
                c.pop("_grouping_dimension", None)
            return kept

        counts: dict[str, int] = {}
        first_seen: dict[str, int] = {}
        for idx, c in enumerate(own_system):
            dim = alias.get(c.get("_grouping_dimension", ""), "")
            if not dim:
                continue
            counts[dim] = counts.get(dim, 0) + 1
            first_seen.setdefault(dim, idx)
        primary_dimension = max(
            set(alias.values()), key=lambda d: (counts.get(d, 0), -first_seen.get(d, 0))
        )

        result, retagged = [], []
        for c in kept:
            dim = alias.get(c.get("_grouping_dimension", ""), "")
            if (
                c.get("scenario_category") == self.SCENARIO_CATEGORY_OWN_SYSTEM
                and dim
                and dim != primary_dimension
            ):
                c["scenario_category"] = self.SCENARIO_CATEGORY_ALTERNATE_DIMENSION
                retagged.append((c.get("name"), dim))
            result.append(c)
            c.pop("_grouping_dimension", None)

        if retagged:
            print(
                f"🧹 Scenario validator kept {len(retagged)} candidate(s) as "
                f"alternate_dimension (primary dimension: {primary_dimension!r}): "
                f"{retagged}"
            )
        return result

    # Back-compat alias (older call sites / tests).
    def _drop_minority_dimensions(
        self, kept: list[dict], *, preserve_all: bool = False
    ) -> list[dict]:
        return self._retag_minority_dimensions(kept)

    # Categories that may be recorded without concept fan-out.
    # own_system_mentioned / subcategory_breakdown: always shells.
    # alternate_dimension / external_reference: shells unless extract_non_primary.
    def _categories_skip_extraction(self) -> frozenset[str]:
        """Categories that do not receive concept/LCI field extraction."""
        cats = {
            self.SCENARIO_CATEGORY_OWN_SYSTEM_MENTIONED,
            self.SCENARIO_CATEGORY_SUBCATEGORY_BREAKDOWN,
        }
        if not self.extract_non_primary:
            cats.add(self.SCENARIO_CATEGORY_EXTERNAL_REFERENCE)
            cats.add(self.SCENARIO_CATEGORY_ALTERNATE_DIMENSION)
            cats.add(self.SCENARIO_CATEGORY_UNCLASSIFIED)
        return frozenset(cats)

    def _config_sha256(self) -> str:
        with open(self.config_path, "rb") as f:
            return hashlib.sha256(f.read()).hexdigest()

    def _definition_only_categories(self) -> frozenset[str]:
        """Categories written with ``definition_only=true`` (match-excluded).

        Alternate-dimension shells stay ``definition_only=false`` so they
        remain eligible for scenario matching without field extraction.
        """
        cats = {
            self.SCENARIO_CATEGORY_OWN_SYSTEM_MENTIONED,
            self.SCENARIO_CATEGORY_SUBCATEGORY_BREAKDOWN,
        }
        if not self.extract_non_primary:
            cats.add(self.SCENARIO_CATEGORY_EXTERNAL_REFERENCE)
        return frozenset(cats)

    def _partition_scenarios(
        self, scenarios: list[dict]
    ) -> tuple[list[dict], list[dict]]:
        """Split kept scenarios into (fully extracted, shell-only)."""
        skip_cats = self._categories_skip_extraction()
        extractable, shells = [], []
        for sc in scenarios:
            if sc.get("scenario_category") in skip_cats:
                shells.append(sc)
            else:
                extractable.append(sc)
        return extractable, shells

    async def _extract_concepts_async(
        self,
        text: str,
        concepts: list,
        scenario: dict | None,
        all_scenarios: list[dict],
        cache_key: str | None = None,
    ) -> tuple[dict, dict]:
        """One chunk call: extract the given concepts' fields, either
        study-wide (scenario=None) or scoped to one specific scenario."""
        n_fields = sum(len(c.structure) for c in concepts)
        max_tokens = min(16000, max(3000, 260 * n_fields))

        concept_lines = [
            f"- {c.name}: {c.description}\n  Fields: {', '.join(c.structure.keys())}"
            for c in concepts
        ]

        if scenario is None:
            scope_txt = "These concepts are STUDY-WIDE: extract them once for the paper as a whole, not per scenario."
        else:
            other_names = [
                s.get("name")
                for s in all_scenarios
                if s.get("name") != scenario.get("name")
            ]
            scope_txt = (
                f'These concepts must be extracted SPECIFICALLY for the scenario "{scenario.get("name")}" '
                f'({scenario.get("description", "")}). This paper has other distinct scenarios too '
                f"({', '.join(other_names) if other_names else 'none'}) -- only report data belonging to "
                f'"{scenario.get("name")}", not to those others.'
            )

        # Text FIRST, varying instructions last -- this is what actually
        # lets prompt_cache_key get cache hits (confirmed live: providers
        # cache on shared PREFIXES only). Every chunk call for one paper
        # sends this exact same paper text; putting the per-call-varying
        # scope/concepts/instructions after it means every call after the
        # first one hits cache for the (large, expensive) text portion.
        # Mistral caches on shared prefixes only, so text-first layout is
        # what makes these repeated per-paper calls cache hits.
        prompt = f"""Paper text:
{text}

Extract structured LCA data from the paper above for the following concepts.

{scope_txt}

Concepts to extract:
{chr(10).join(concept_lines)}

Instructions:
- Look into tables for data as a first priority.
- When a numeric value is present, include its unit if the paper states one; if the unit is missing, write the number alone -- do not invent a unit disclaimer.
- Use null for any field not explicitly stated -- never guess or infer, and never fill a missing field with placeholder text such as "units not stated", "not stated", or "no data".
- Report every concept listed, even if every field inside it is null."""

        args, usage = await self._acall(
            messages=[
                {
                    "role": "system",
                    # Keep aligned with prompt_templates.SYSTEM_EXTRACT_CONCEPT
                    # (null for missing fields; no "units not stated" placeholder).
                    "content": (
                        "You are an expert in lifecycle assessment (LCA) and sustainability reporting. "
                        "Extract structured data from research papers. Include units with numeric "
                        "values when the paper states them; if the unit is missing, write the number "
                        "alone. Use null for any field not explicitly stated -- never guess, and never "
                        "fill missing fields with placeholder text such as 'units not stated' or "
                        "'not stated'. Look into tables for data as a first priority. Do not "
                        "hallucinate."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
            tools=[self._concepts_tool(concepts)],
            max_tokens=max_tokens,
            prompt_cache_key=cache_key,
        )
        return args, usage

    async def full_extract_async(
        self,
        pdf_path: str,
        max_concurrency: int = 6,
        scenarios_only: bool = False,
        extra_text: str | None = None,
    ) -> dict:
        """
        Full-parity extraction: identifies every scenario, then extracts
        every concept in the domain config (study-wide concepts once,
        per-scenario concepts once per scenario), with concept-chunk calls
        run concurrently (bounded by max_concurrency) rather than
        sequentially -- the piece that makes many-scenario papers
        (dairy has one with 35) tractable in reasonable wall time.
        A chunk that fails after all retries doesn't lose the rest of the
        paper: its fields are just left absent, and a warning is printed.

        `scenarios_only` stops after identification + validation and emits
        the same output shape with every scenario as a stub. Useful for
        checking scenario accuracy against a new domain's ground truth
        before investing in a field schema, and cheap: it is two calls per
        paper on scenario_model instead of dozens on model.

        `extra_text` is appended to the PDF text (e.g. supplementary
        materials rendered as tables/text) before scenario ID + field extract.
        """
        text = self.extract_text_from_pdf(pdf_path)
        if extra_text and str(extra_text).strip():
            text = f"{text}\n\n{extra_text}"
        text, bib_meta = self._prepare_paper_text(text)
        pid = os.path.splitext(os.path.basename(pdf_path))[0]
        # Cheap non-AI section split (Methods/Results/…) for validator context
        # and for extracted_sections/{pid}.json — same artifact the old
        # Mistral pipeline used to write beside each extract.
        sections_dict = build_sections_dict(text)
        paper_context = validator_context_from_sections(
            sections_dict,
            max_chars=self.VALIDATOR_CONTEXT_CHARS,
            intro_chars=self.PAPER_CONTEXT_CHARS,
        )
        # One cache key per paper -- every concept-chunk call below sends
        # this same paper text as its prompt's shared prefix (see
        # _extract_concepts_async), so they all warm/reuse the same cache
        # entry regardless of which scenario or concept chunk they're for.
        cache_key = f"{self.cache_namespace}:{pid}:full_text"
        study_wide_concepts, by_scenario_concepts = build_concepts_from_config(
            config_path=self.config_path
        )

        total_usage: dict[str, int] = {}
        # Scenario identification + validation run on scenario_model, the
        # rest on model. Kept in a separate bucket so cost reports price
        # each at its own rate -- outputs-40papers lumped them together
        # and understated its own baseline by ~14%.
        scenario_usage: dict[str, int] = {}

        def _accumulate(into: dict, usage: dict) -> None:
            for k, v in (usage or {}).items():
                if isinstance(v, (int, float)):
                    into[k] = into.get(k, 0) + v

        def add_usage(usage: dict) -> None:
            _accumulate(total_usage, usage)

        if not self.extract_scenarios:
            # Paper-level extraction: one synthetic row, study-wide concepts only.
            scenarios = [
                {
                    "name": "Paper",
                    "description": "Paper-level extraction (scenario identification disabled).",
                    "scenario_category": "own_system_modelled",
                }
            ]
            print("  extract_scenarios=false: skipping scenario identification")
        else:
            scenarios, sc_usage = await self._identify_scenarios_async(text)
            _accumulate(scenario_usage, sc_usage)

            # Second-opinion validation: label categories, retag secondary
            # grouping axes to alternate_dimension, drop only ``other``.
            # Non-primary field extraction is gated later by extract_non_primary.
            if scenarios:
                n_before = len(scenarios)
                scenarios, val_usage = await self._filter_valid_scenarios_async(
                    scenarios, paper_context=paper_context
                )
                _accumulate(scenario_usage, val_usage)
                if len(scenarios) != n_before:
                    print(
                        f"  scenario validator: {n_before} candidates -> "
                        f"{len(scenarios)} kept"
                    )

            if not scenarios:
                # No scenario ever identified -- still extract study-wide data
                # and per-scenario concepts under one fallback scenario rather
                # than silently dropping everything.
                scenarios = [
                    {
                        "name": "Own system",
                        "description": "Single system described in the paper.",
                    }
                ]

        extractable, shells = self._partition_scenarios(scenarios)
        # scenarios_only suppresses the fan-out without touching the
        # definition_only flag, which stays a statement about the
        # classifier's verdict rather than about this run's mode.
        if scenarios_only:
            extractable = []
        if shells:
            print(
                f"  {len(shells)} scenario(s) kept as labeled shells "
                f"(name/description/category, no data extracted): "
                f"{[s.get('name') for s in shells]}"
            )

        sem = asyncio.Semaphore(max_concurrency)

        async def run_chunk(concepts_chunk: list, scenario: dict | None):
            async with sem:
                try:
                    return await self._extract_concepts_async(
                        text, concepts_chunk, scenario, scenarios, cache_key=cache_key
                    )
                except Exception as e:
                    names = [c.name for c in concepts_chunk]
                    scen_txt = f' for scenario "{scenario["name"]}"' if scenario else ""
                    print(
                        f"⚠️  chunk {names}{scen_txt} failed after retries, skipping: {e}"
                    )
                    return {}, {}

        study_wide_chunks = (
            [] if scenarios_only else self._chunk_concepts(study_wide_concepts)
        )
        by_scenario_chunks = self._chunk_concepts(by_scenario_concepts)

        # (chunk_index_key, concepts_chunk, scenario_or_None)
        specs: list[tuple[tuple[str, str | None], list, dict | None]] = []
        for chunk in study_wide_chunks:
            specs.append((("study_wide", None), chunk, None))
        for scenario in extractable:
            for chunk in by_scenario_chunks:
                specs.append((("scenario", scenario.get("name")), chunk, scenario))

        task_keys = [key for key, _, _ in specs]

        # The first call is awaited alone before the rest fan out. Every
        # chunk shares one cache_key and one paper-text prefix, but a
        # prefix is only cached once a call returns -- gathering all of
        # them at once puts the whole first concurrency wave in flight
        # against a cold cache (measured: 63% hit rate, i.e. roughly one
        # wave's worth of misses). Paying one serialized call up front
        # means every remaining chunk reads the text from cache at 10% of
        # the input rate.
        if specs:
            _, first_chunk, first_scenario = specs[0]
            first_result = await run_chunk(first_chunk, first_scenario)
            rest_results = await asyncio.gather(
                *(run_chunk(chunk, scenario) for _, chunk, scenario in specs[1:])
            )
            chunk_results = [first_result, *rest_results]
        else:
            chunk_results = []

        study_wide_data: dict[str, dict] = {}
        by_scenario_data: dict[str, dict[str, dict]] = {
            s.get("name"): {} for s in scenarios
        }
        for (kind, scenario_name), (args, usage) in zip(task_keys, chunk_results):
            add_usage(usage)
            if kind == "study_wide":
                study_wide_data.update(args or {})
            else:
                by_scenario_data.setdefault(scenario_name, {}).update(args or {})

        definition_only_cats = self._definition_only_categories()
        shell_names = {s.get("name") for s in shells}
        results = []
        for sc in scenarios:
            name = sc.get("name")
            cat = sc.get("scenario_category", self.SCENARIO_CATEGORY_OWN_SYSTEM)
            fields_extracted = name not in shell_names
            group = [
                {
                    "Scenario": {
                        "Scenario Name": name,
                        "scenario_description": sc.get("description", ""),
                        "scenario_category": cat,
                        "definition_only": cat in definition_only_cats,
                        "fields_extracted": fields_extracted,
                    }
                }
            ]
            # Shells (no fan-out) get no per-scenario or study-wide fields —
            # record that the paper named them without attributing this
            # study's inventory numbers to a non-extracted system.
            if fields_extracted:
                for concept_name, fields in by_scenario_data.get(name, {}).items():
                    group.append({concept_name: fields})
                for concept_name, fields in study_wide_data.items():
                    group.append({concept_name: fields})
            results.append(group)

        meta = {
            "pid": pid,
            "pdf_path": pdf_path,
            "model": self.model,
            "scenario_model": self.scenario_model,
            "extractor": "agent_sog.AgentExtractor",
            # Which config produced this file, so results stay traceable.
            "config_path": self.config_path,
            "config_sha256": self._config_sha256(),
            "extract_scenarios": self.extract_scenarios,
            "extract_non_primary": self.extract_non_primary,
        }
        if bib_meta:
            meta.update(bib_meta)
        else:
            meta["strip_bibliography"] = False
        if extra_text and str(extra_text).strip():
            meta["supplementary_text_chars"] = len(str(extra_text))
            meta["included_supplementary"] = True
        return {
            "metadata": meta,
            # Field-extraction usage. Extracts written before 2026-10-24 call
            # this "aspect_tokens"; readers accept both.
            "tokens": {
                "field_tokens": {**total_usage, "model": self.model},
                "scenario_tokens": {**scenario_usage, "model": self.scenario_model},
            },
            "results": results,
            "sections_dict": sections_dict,
            "extracted_from": (
                "full_text+SI (agent, scenarios only)"
                if scenarios_only and extra_text
                else (
                    "full_text (agent, scenarios only)"
                    if scenarios_only
                    else (
                        "full_text+SI (agent, per-scenario concurrent chunks)"
                        if extra_text
                        else "full_text (agent, per-scenario concurrent chunks)"
                    )
                )
            ),
        }

    def full_extract(
        self,
        pdf_path: str,
        max_concurrency: int = 6,
        scenarios_only: bool = False,
        extra_text: str | None = None,
    ) -> dict:
        """Sync wrapper around full_extract_async (see there for details)."""
        return asyncio.run(
            self.full_extract_async(
                pdf_path,
                max_concurrency=max_concurrency,
                scenarios_only=scenarios_only,
                extra_text=extra_text,
            )
        )
