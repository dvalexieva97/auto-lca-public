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

# Quiet the litellm "Give Feedback / Get Help" banner on every error.
litellm.suppress_debug_info = True

from auto_lca.process.nlp.extract import TextExtractor
from auto_lca.process.preprocess.config_parser import ensure_json_config
from auto_lca.process.sog.bibliography import strip_bibliography
from auto_lca.process.sog.concept import SAMPLE_CONFIG_PATH, build_concepts_from_config
from auto_lca.process.sog.section_context import (
    build_sections_dict,
    validator_context_from_sections,
)

# Scenario prompts come from the domain pack in prompt_templates.py (via
# domains.prompts_for_domain). Keep those packs verbatim -- a hand-abbreviated
# LCA scenario prompt previously caused over-enumeration regressions.
from auto_lca.statistics.gap_analysis import FieldGap

from auto_lca.process.sog.domains import (
    DEFAULT_DOMAIN,
    extract_behavior_defaults,
    prompts_for_domain,
    resolve_domain,
)
from auto_lca.process.sog.model_profiles import (
    get_profile,
    rate_limit_backoff_seconds,
    rate_limit_policy_for,
    to_litellm_model,
)

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
        domain: str | None = None,
        entry_default_domain: str = DEFAULT_DOMAIN,
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

        # Domain selects the fixed Python prompt pack (lca | generic).
        # Precedence: --domain > defaults.domain > entry-point default.
        self.domain = resolve_domain(
            cli_domain=domain,
            config_defaults=defaults,
            entry_default=entry_default_domain,
        )
        self.prompts = prompts_for_domain(self.domain)

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
        self.plan_profile = resolved.get("plan_profile") or "free"
        self.rate_limit_policy = rate_limit_policy_for(self.plan_profile)
        # Shared request pacing (free plan ~1 RPS). Sync + async paths share
        # monotonic next-slot; async also uses a lock created lazily in-loop.
        self._next_request_at = 0.0
        self._async_throttle_lock: asyncio.Lock | None = None
        # Historical fallback: if the user overrode only the field model
        # (CLI/config) and left scenario unspecified, reuse the field model
        # rather than the plan's scenario default.
        field_overridden = bool(model or defaults.get("model"))
        scenario_overridden = bool(
            scenario_model
            or defaults.get("production_system_model")
            or defaults.get("scenario_model")
        )
        if field_overridden and not scenario_overridden:
            self.scenario_model = self.model
        preflight_user_models(
            model=self.model,
            scenario_model=self.scenario_model,
            api_key=self.api_key,
            plan_profile=resolved.get("plan_profile"),
            plan_defaults=resolved.get("plan_defaults"),
            has_user_override=bool(resolved.get("has_user_override")),
        )
        pol = self.rate_limit_policy
        print(
            f"rate-limit policy={self.plan_profile}: "
            f"concurrency≤{pol['max_concurrency']}, "
            f"min_interval={pol['min_interval_s']}s, "
            f"429 retries≤{pol['rate_limit_retries']} "
            f"(backoff≥{pol['base_backoff_s']}s, cap {pol['backoff_cap_s']}s; "
            f"honors Retry-After when present)"
        )

        # Scenario / production-system flags (two primary knobs):
        #   extract_scenarios: run identification+classification at all.
        #     False → paper-level extraction (LCA: legacy synthetic "Paper"
        #     Scenario row; generic: field sets only, no Scenario stub).
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

        # Domain-aware extract extras when config omits the key (sections off
        # for both; generic enables PDF-table inject). Config ``defaults.*``
        # always wins.
        behavior = extract_behavior_defaults(self.domain)
        self.extract_sections = bool(
            defaults["extract_sections"]
            if "extract_sections" in defaults
            else behavior["extract_sections"]
        )
        self.allow_additional_fields = bool(
            defaults["allow_additional_fields"]
            if "allow_additional_fields" in defaults
            else behavior["allow_additional_fields"]
        )
        self.include_pdf_tables = bool(
            defaults["include_pdf_tables"]
            if "include_pdf_tables" in defaults
            else behavior["include_pdf_tables"]
        )

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

    def _throttle_sync(self) -> None:
        """Pace sync calls to stay under plan RPS (no-op when interval is 0)."""
        interval = float(self.rate_limit_policy.get("min_interval_s", 0) or 0)
        if interval <= 0:
            return
        now = time.monotonic()
        wait = self._next_request_at - now
        if wait > 0:
            time.sleep(wait)
        self._next_request_at = time.monotonic() + interval

    async def _throttle_async(self) -> None:
        """Pace async calls; free plan defaults to ~1 RPS."""
        interval = float(self.rate_limit_policy.get("min_interval_s", 0) or 0)
        if interval <= 0:
            return
        if self._async_throttle_lock is None:
            self._async_throttle_lock = asyncio.Lock()
        async with self._async_throttle_lock:
            now = time.monotonic()
            wait = self._next_request_at - now
            if wait > 0:
                await asyncio.sleep(wait)
            self._next_request_at = time.monotonic() + interval

    @staticmethod
    def _parse_tool_response(resp) -> tuple[dict, dict]:
        message = resp.choices[0].message
        tool_calls = getattr(message, "tool_calls", None)
        if not tool_calls:
            raise ValueError("Model did not return a tool call")
        raw_args = tool_calls[0].function.arguments
        args = json.loads(raw_args) if isinstance(raw_args, str) else raw_args
        usage = (
            AgentExtractor._flatten_usage(resp.usage)
            if getattr(resp, "usage", None)
            else {}
        )
        return args, usage

    def _raise_call_exhausted(
        self, last_err: BaseException | None, *, rate_hits: int
    ) -> None:
        cause = f"{type(last_err).__name__}: {last_err}" if last_err else "unknown"
        if isinstance(last_err, litellm.exceptions.RateLimitError):
            raise RuntimeError(
                f"agent_sog call failed after {rate_hits} rate-limit retries "
                f"({self.plan_profile} policy exhausted): {cause}"
            ) from last_err
        raise RuntimeError(
            f"agent_sog call failed after {self.max_retries} attempts: {cause}"
        ) from last_err

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
        last_err: BaseException | None = None
        transient_attempts = 0
        rate_hits = 0
        max_429 = int(self.rate_limit_policy["rate_limit_retries"])
        while True:
            try:
                self._throttle_sync()
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
                return self._parse_tool_response(resp)
            except litellm.exceptions.RateLimitError as e:
                last_err = e
                rate_hits += 1
                if rate_hits > max_429:
                    self._raise_call_exhausted(last_err, rate_hits=rate_hits)
                wait = rate_limit_backoff_seconds(
                    plan_profile=self.plan_profile, hit=rate_hits, exc=e
                )
                print(
                    f"⚠️ rate limited ({self.plan_profile} "
                    f"{rate_hits}/{max_429}), sleeping {wait:.0f}s: {e}"
                )
                time.sleep(wait)
            except Exception as e:
                last_err = e
                transient_attempts += 1
                print(
                    f"⚠️ agent_sog call failed "
                    f"(attempt {transient_attempts}/{self.max_retries}): {e}"
                )
                if transient_attempts >= self.max_retries:
                    self._raise_call_exhausted(last_err, rate_hits=rate_hits)
                time.sleep(2.0 * transient_attempts)

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
        same mechanism, confirmed live to produce real cache hits.

        429 handling follows the plan rate-limit policy (free waits longer /
        retries more; paid is snappier) and prefers Retry-After when Mistral
        sends it."""
        last_err: BaseException | None = None
        transient_attempts = 0
        rate_hits = 0
        max_429 = int(self.rate_limit_policy["rate_limit_retries"])
        while True:
            try:
                await self._throttle_async()
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
                return self._parse_tool_response(resp)
            except litellm.exceptions.RateLimitError as e:
                last_err = e
                rate_hits += 1
                if rate_hits > max_429:
                    self._raise_call_exhausted(last_err, rate_hits=rate_hits)
                wait = rate_limit_backoff_seconds(
                    plan_profile=self.plan_profile, hit=rate_hits, exc=e
                )
                # Small jitter so concurrent free waiters don't resume in lockstep.
                wait += random.uniform(0, 3.0)
                print(
                    f"⚠️ rate limited ({self.plan_profile} "
                    f"{rate_hits}/{max_429}), sleeping {wait:.0f}s: {e}"
                )
                await asyncio.sleep(wait)
            except Exception as e:
                last_err = e
                transient_attempts += 1
                print(
                    f"⚠️ agent_sog async call failed "
                    f"(attempt {transient_attempts}/{self.max_retries}): {e}"
                )
                if transient_attempts >= self.max_retries:
                    self._raise_call_exhausted(last_err, rate_hits=rate_hits)
                # Jittered backoff so concurrent chunks don't retry in lockstep.
                await asyncio.sleep(
                    2.0 * transient_attempts + random.uniform(0, 2.0)
                )

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
    #
    # Nested values are also supported in JSON configs:
    #   {"unit": "str | None", "amount": "float | None"}  → object
    #   [{"name": "str", "amount": "float | None"}]       → list of objects
    #   {"type": "list", "items": {...}}                  → same list form
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
    def _type_is_required(cls, type_spec: Any) -> bool:
        """True when a scalar type string is non-nullable (no ``| None``)."""
        return isinstance(type_spec, str) and "None" not in type_spec

    @classmethod
    def _count_fields(cls, type_spec: Any) -> int:
        """Leaf-field budget for chunking (nested objects/lists count leaves)."""
        if isinstance(type_spec, dict):
            if str(type_spec.get("type", "")).lower() in ("list", "array"):
                return cls._count_fields(type_spec.get("items"))
            return sum(cls._count_fields(v) for v in type_spec.values()) or 1
        if isinstance(type_spec, list):
            return cls._count_fields(type_spec[0]) if type_spec else 1
        return 1

    @classmethod
    def _spec_is_list(cls, type_spec: Any) -> bool:
        if isinstance(type_spec, list):
            return True
        return isinstance(type_spec, dict) and str(
            type_spec.get("type", "")
        ).lower() in ("list", "array")

    @classmethod
    def _field_schema(
        cls,
        type_spec: Any,
        *,
        as_list_item: bool = False,
        allow_additional: bool = False,
    ) -> dict:
        """Map a config field type (scalar string or nested JSON) to JSON Schema.

        Array *items* are plain objects (not ``object|null``): nullable list
        elements confuse several providers' tool-calling validators and can
        cause the whole list field to be dropped.
        """
        if isinstance(type_spec, dict):
            kind = str(type_spec.get("type", "")).lower()
            if kind in ("list", "array") and "items" in type_spec:
                return {
                    "type": ["array", "null"],
                    "items": cls._field_schema(
                        type_spec["items"],
                        as_list_item=True,
                        allow_additional=allow_additional,
                    ),
                }
            props: dict[str, Any] = {}
            required: list[str] = []
            for key, child in type_spec.items():
                props[key] = cls._field_schema(
                    child, allow_additional=allow_additional
                )
                if cls._type_is_required(child):
                    required.append(key)
            schema: dict[str, Any] = {
                "type": (
                    "object" if as_list_item else ["object", "null"]
                ),
                "properties": props,
                "additionalProperties": bool(allow_additional),
            }
            if required:
                schema["required"] = required
            return schema

        if isinstance(type_spec, list):
            if len(type_spec) != 1:
                raise ValueError(
                    "List field types must be a one-element list of an object "
                    "schema, e.g. [{\"name\": \"str\", \"amount\": \"float | None\"}]"
                )
            return {
                "type": ["array", "null"],
                "items": cls._field_schema(
                    type_spec[0],
                    as_list_item=True,
                    allow_additional=allow_additional,
                ),
            }

        if not isinstance(type_spec, str):
            raise TypeError(
                f"Unsupported field type spec {type_spec!r}; "
                "use a type string, nested object, or one-element list of objects"
            )

        schema = dict(
            cls._TYPE_JSON_SCHEMA.get(type_spec.strip(), {"type": ["string", "null"]})
        )
        if schema.get("type") == "string" or (
            isinstance(schema.get("type"), list) and "string" in schema["type"]
        ):
            schema["maxLength"] = cls.MAX_STR_FIELD_LENGTH
        return schema

    @classmethod
    def _concepts_tool(
        cls, concepts: list, *, allow_additional: bool = False
    ) -> dict:
        """Build a tool spec whose top-level properties are the given
        concepts' names, each an object of their own fields -- concept
        presence is required (the model must attempt every concept, even
        if every field inside comes back null), but individual fields
        follow the config's own required/optional typing."""
        concept_props = {}
        for c in concepts:
            field_props = {}
            field_required = []
            for field_name, type_spec in c.structure.items():
                field_props[field_name] = cls._field_schema(
                    type_spec, allow_additional=allow_additional
                )
                # Nested objects/lists are optional as a whole; only scalar
                # non-nullable strings are required at the concept root.
                if cls._type_is_required(type_spec):
                    field_required.append(field_name)
            concept_schema = {
                "type": "object",
                "properties": field_props,
                "additionalProperties": bool(allow_additional),
            }
            if field_required:
                concept_schema["required"] = field_required
            concept_props[c.name] = concept_schema

        invent_clause = (
            "Extra keys beyond the listed fields are allowed when useful "
            "(e.g. table columns not named in the config)."
            if allow_additional
            else "Never invent keys beyond the schema."
        )
        return {
            "type": "function",
            "function": {
                "name": "record_extraction",
                "description": (
                    "Fill in EVERY field individually where the text supports it -- "
                    "do not skip a specific field just because related context "
                    "appears elsewhere, and do not dump everything into one field "
                    "while leaving its siblings null. Use null for anything not "
                    f"explicitly stated in the text -- never guess or infer. {invent_clause} "
                    "Always include every concept listed, even if every field "
                    "inside it is null."
                ),
                "parameters": {
                    "type": "object",
                    "properties": concept_props,
                    "required": [c.name for c in concepts],
                    "additionalProperties": bool(allow_additional),
                },
            },
        }

    @classmethod
    def _scalar_type_kind(cls, type_spec: Any) -> str | None:
        if not isinstance(type_spec, str):
            return None
        t = type_spec.strip()
        if t in ("float", "float | None", "int", "int | None"):
            return "number"
        if t in ("str", "str | None"):
            return "string"
        return None

    @classmethod
    def _coerce_scalar_string(cls, value: Any) -> str | None:
        if value is None:
            return None
        if isinstance(value, str):
            s = value.strip()
            return s if s else None
        if isinstance(value, bool):
            return None
        if isinstance(value, (int, float)):
            return str(value)
        if isinstance(value, dict):
            amount = value.get("amount")
            if amount is None:
                amount = value.get("value")
            unit = value.get("unit")
            if amount is None:
                return None
            if unit is not None and str(unit).strip():
                return f"{amount} {unit}".strip()
            return str(amount).strip()
        return None

    @classmethod
    def _coerce_scalar_float(cls, value: Any) -> float | None:
        if value is None:
            return None
        if isinstance(value, bool):
            return None
        if isinstance(value, (int, float)):
            return float(value)
        if isinstance(value, dict):
            for key in ("amount", "value", "number"):
                if key in value and value[key] is not None:
                    return cls._coerce_scalar_float(value[key])
            if len(value) == 1:
                return cls._coerce_scalar_float(next(iter(value.values())))
            return None
        if isinstance(value, str):
            from auto_lca.statistics.numeric_parse import parse_numeric_value

            parsed = parse_numeric_value(value)
            if parsed is not None:
                return parsed
            return value
        return None

    @classmethod
    def _raw_field_value(cls, raw: dict, field_name: str) -> Any:
        """Resolve a field from model output, including minor key variants."""
        if field_name in raw:
            return raw[field_name]
        base = field_name.split("(", 1)[0].strip().lower()
        for key, val in raw.items():
            if not isinstance(key, str):
                continue
            key_base = key.split("(", 1)[0].strip().lower()
            if key_base == base:
                return val
        return None

    @classmethod
    def _analyze_concept_schema(cls, concepts: list) -> dict[str, bool]:
        """Leaf-level schema flags for the concepts in one extraction chunk."""
        flags = {
            "nested_object": False,
            "string_scalar": False,
            "number_scalar": False,
            "list_field": False,
        }

        def walk(spec: Any, *, ctx: str = "toplevel") -> None:
            if isinstance(spec, dict):
                kind = str(spec.get("type", "")).lower()
                if kind in ("list", "array") and "items" in spec:
                    flags["list_field"] = True
                    walk(spec["items"], ctx="list_row")
                    return
                if kind in ("list", "array"):
                    return
                if ctx == "toplevel":
                    flags["nested_object"] = True
                    child_ctx = "nested_leaf"
                else:
                    child_ctx = ctx
                for child in spec.values():
                    walk(child, ctx=child_ctx)
                return
            if isinstance(spec, list):
                flags["list_field"] = True
                if spec:
                    walk(spec[0], ctx="list_row")
                return
            if ctx in ("toplevel", "list_row"):
                sk = cls._scalar_type_kind(spec)
                if sk == "string":
                    flags["string_scalar"] = True
                elif sk == "number":
                    flags["number_scalar"] = True

        for concept in concepts:
            for spec in concept.structure.values():
                walk(spec)
        return flags

    @classmethod
    def _extraction_instruction_lines(cls, concepts: list) -> list[str]:
        """User-prompt bullets matched to this chunk's tool schema (flat vs nested)."""
        flags = cls._analyze_concept_schema(concepts)
        lines = [
            "Prefer the \"EXTRACTED TABLES\" section (structured CSV grids) over prose "
            "when both exist; blank CSV cells mean the value was not stated — use null, "
            "do not copy another row.",
        ]
        if flags["nested_object"]:
            lines.append(
                "Object fields: fill each sub-key from the schema (e.g. put the numeric "
                "magnitude in amount and the unit string in unit). Do not merge into one "
                "string unless that field's type is string."
            )
        if flags["string_scalar"]:
            if flags["nested_object"]:
                lines.append(
                    "String-typed fields: write the value as the paper states it."
                )
            else:
                lines.append(
                    "String-typed fields: write the value as the paper states it "
                    "(digits and unit in the same field when both are given)."
                )
        if flags["number_scalar"] and not flags["nested_object"]:
            lines.append(
                "Number-typed fields: store only the numeric magnitude; do not use null "
                "just because a unit appears nearby in the text."
            )
        lines.extend(
            [
                "Use null for any field not explicitly stated — never guess or infer, and "
                "never fill a missing field with placeholder text such as \"units not stated\", "
                "\"not stated\", or \"no data\".",
                "Report every concept listed, even if every field inside it is null.",
            ]
        )
        if flags["list_field"]:
            lines.append(
                "When a field is a list of objects, include every distinct row the paper "
                "reports (models, experiments, impacts, ablations); do not collapse them "
                "into one entry or leave the list null if values are present."
            )
        return lines

    @classmethod
    def _coerce_value_to_spec(
        cls, value: Any, type_spec: Any, *, allow_additional: bool = False
    ) -> Any:
        """Align a value to the config field schema (optionally keeping extras)."""
        if value is None:
            return None
        if isinstance(type_spec, dict):
            kind = str(type_spec.get("type", "")).lower()
            if kind in ("list", "array") and "items" in type_spec:
                if not isinstance(value, list):
                    return None
                return [
                    cls._coerce_value_to_spec(
                        item,
                        type_spec["items"],
                        allow_additional=allow_additional,
                    )
                    for item in value
                ]
            if not isinstance(value, dict):
                return None
            out = {
                key: cls._coerce_value_to_spec(
                    value.get(key), child, allow_additional=allow_additional
                )
                for key, child in type_spec.items()
            }
            if allow_additional:
                for key, extra in value.items():
                    if key not in out:
                        out[key] = extra
            return out
        if isinstance(type_spec, list):
            if not isinstance(value, list) or not type_spec:
                return None if not isinstance(value, list) else []
            item_spec = type_spec[0]
            return [
                cls._coerce_value_to_spec(
                    item, item_spec, allow_additional=allow_additional
                )
                for item in value
            ]
        scalar = cls._scalar_type_kind(type_spec)
        if scalar == "string":
            return cls._coerce_scalar_string(value)
        if scalar == "number":
            return cls._coerce_scalar_float(value)
        return value

    @classmethod
    def _coerce_extraction_to_concepts(
        cls,
        args: dict | None,
        concepts: list,
        *,
        allow_additional: bool = False,
    ) -> dict:
        """Align model output to the config; optionally keep extra keys."""
        args = args or {}
        out: dict[str, Any] = {}
        for concept in concepts:
            raw = args.get(concept.name)
            if raw is None:
                out[concept.name] = {
                    field: None for field in concept.structure
                }
                continue
            if not isinstance(raw, dict):
                out[concept.name] = {
                    field: None for field in concept.structure
                }
                continue
            coerced = {
                field: cls._coerce_value_to_spec(
                    cls._raw_field_value(raw, field),
                    spec,
                    allow_additional=allow_additional,
                )
                for field, spec in concept.structure.items()
            }
            if allow_additional:
                for key, extra in raw.items():
                    if key not in coerced:
                        coerced[key] = extra
            out[concept.name] = coerced
        if allow_additional:
            known = {c.name for c in concepts}
            for key, extra in args.items():
                if key not in known:
                    out[key] = extra
        return out

    @classmethod
    def _chunk_concepts(cls, concepts: list) -> list[list]:
        """Split concepts into chunks whose total field count stays under
        CONCEPT_CHUNK_FIELD_BUDGET.

        Any field set that contains a list-of-objects field is extracted
        alone: those arrays are long-output and otherwise get truncated or
        dropped when packed with sibling concepts (seen live on free-tier
        Ministral with Models[] next to Study Design).
        """
        chunks: list[list] = []
        current: list = []
        current_fields = 0
        for c in concepts:
            n = sum(cls._count_fields(spec) for spec in c.structure.values())
            has_list = any(cls._spec_is_list(spec) for spec in c.structure.values())
            if has_list:
                if current:
                    chunks.append(current)
                    current, current_fields = [], 0
                chunks.append([c])
                continue
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
        # Domain prompt pack (lca / generic) -- no freeform config prompts.
        # Granularity discipline comes from this prompt plus the validation
        # pass below, matching the legacy extractor's two-stage architecture.
        prompt = self.prompts["prompt_extract_scenarios"].format(text=text)
        args, usage = await self._acall(
            messages=[
                {
                    "role": "system",
                    "content": self.prompts["system_extract_scenarios"],
                },
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
                        "content": self.prompts["system_validate_scenarios"],
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
        # Budget from leaf fields, not top-level keys: a single list field
        # like Models[] used to count as 1 → max_tokens=3000, which truncated
        # the array mid-generation (out6 vaswani: Study Design kept, Models gone).
        n_fields = sum(
            self._count_fields(spec)
            for c in concepts
            for spec in c.structure.values()
        )
        has_list = any(
            self._spec_is_list(spec)
            for c in concepts
            for spec in c.structure.values()
        )
        max_tokens = min(16000, max(8000 if has_list else 3000, 260 * max(n_fields, 1)))

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
        instruction_lines = self._extraction_instruction_lines(concepts)
        instructions_block = "\n".join(f"- {line}" for line in instruction_lines)

        prompt = f"""Paper text:
{text}

{self.prompts["user_extract_concepts_lead"]}

{scope_txt}

Concepts to extract:
{chr(10).join(concept_lines)}

Instructions:
{instructions_block}"""

        args, usage = await self._acall(
            messages=[
                {
                    "role": "system",
                    "content": self.prompts["system_extract_concept"],
                },
                {"role": "user", "content": prompt},
            ],
            tools=[
                self._concepts_tool(
                    concepts, allow_additional=self.allow_additional_fields
                )
            ],
            max_tokens=max_tokens,
            prompt_cache_key=cache_key,
        )
        return (
            self._coerce_extraction_to_concepts(
                args,
                concepts,
                allow_additional=self.allow_additional_fields,
            ),
            usage,
        )

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
        A field-extraction chunk that fails after all retries aborts the
        whole paper (so a hard rate-limit does not write a silently
        incomplete JSON). Scenario ID failures already abort the same way.

        `scenarios_only` stops after identification + validation and emits
        the same output shape with every scenario as a stub. Useful for
        checking scenario accuracy against a new domain's ground truth
        before investing in a field schema, and cheap: it is two calls per
        paper on scenario_model instead of dozens on model.

        `extra_text` is appended to the PDF text (e.g. supplementary
        materials rendered as tables/text) before scenario ID + field extract.
        """
        text = self.extract_text_from_pdf(pdf_path)
        # Strip bibliography on body text first, then append structured tables
        # (and optional SI). Tables are usually before References in the PDF,
        # but our CSV block is appended at the end — stripping after append
        # would delete it along with the reference list.
        text, bib_meta = self._prepare_paper_text(text)
        tables_meta: dict[str, Any] = {}
        if self.include_pdf_tables:
            tables_block, n_tables = TextExtractor.format_pdf_tables_as_text(
                pdf_path
            )
            tables_meta["include_pdf_tables"] = True
            tables_meta["pdf_tables_count"] = n_tables
            if tables_block:
                text = f"{text}\n\n{tables_block}"
                print(f"  attached {n_tables} structured PDF table(s) to prompt")
            else:
                print("  no structured PDF tables detected")
        else:
            tables_meta["include_pdf_tables"] = False
        if extra_text and str(extra_text).strip():
            text = f"{text}\n\n{extra_text}"
        pid = os.path.splitext(os.path.basename(pdf_path))[0]
        # Optional MiniLM Methods/Results/… split (defaults.extract_sections).
        # When off, keep a full_text-only stub so the scenario validator still
        # gets a front-matter + truncated body context — no extracted_sections
        # files and no sections_dict in the result JSON.
        if self.extract_sections:
            sections_dict = build_sections_dict(text)
        else:
            sections_dict = {
                "full_text": {
                    "section_title": "full_text",
                    "section_text": text,
                    "cosine_similarity": None,
                    "fuzzy_cosine_similarity": None,
                }
            }
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
        if not self.extract_scenarios and by_scenario_concepts:
            # No Scenario fan-out: still extract per-system field sets once,
            # as study-wide (otherwise they would be silently dropped).
            study_wide_concepts = [*study_wide_concepts, *by_scenario_concepts]
            by_scenario_concepts = []

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
            # Paper-level extraction: study-wide concepts only, no Scenario stub.
            scenarios = []
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

        # Cap fan-out to the plan policy (free ≈ 1 RPS → concurrency 1).
        plan_cap = int(self.rate_limit_policy.get("max_concurrency", max_concurrency))
        if max_concurrency > plan_cap:
            max_concurrency = plan_cap

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
                    # Do not soft-skip: exhausted retries (incl. hard 429 windows)
                    # must fail the paper instead of writing incomplete fields.
                    raise RuntimeError(
                        f"field extraction chunk {names}{scen_txt} "
                        f"failed after retries: {e}"
                    ) from e

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

        results = []
        if not self.extract_scenarios and self.domain == "generic":
            # Generic paper-level mode: no synthetic Scenario stub.
            group = [
                {concept_name: fields}
                for concept_name, fields in study_wide_data.items()
            ]
            if group:
                results.append(group)
        elif not self.extract_scenarios:
            # LCA paper-level parity: keep the legacy synthetic "Paper" row.
            group = [
                {
                    "Scenario": {
                        "Scenario Name": "Paper",
                        "scenario_description": (
                            "Paper-level extraction (scenario identification disabled)."
                        ),
                        "scenario_category": self.SCENARIO_CATEGORY_OWN_SYSTEM,
                        "definition_only": False,
                        "fields_extracted": True,
                    }
                }
            ]
            for concept_name, fields in study_wide_data.items():
                group.append({concept_name: fields})
            results.append(group)
        else:
            definition_only_cats = self._definition_only_categories()
            shell_names = {s.get("name") for s in shells}
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
            "domain": self.domain,
            "extractor": "agent_sog.AgentExtractor",
            # Which config produced this file, so results stay traceable.
            "config_path": self.config_path,
            "config_sha256": self._config_sha256(),
            "extract_scenarios": self.extract_scenarios,
            "extract_non_primary": self.extract_non_primary,
            "extract_sections": self.extract_sections,
            "allow_additional_fields": self.allow_additional_fields,
        }
        if bib_meta:
            meta.update(bib_meta)
        else:
            meta["strip_bibliography"] = False
        meta.update(tables_meta)
        if extra_text and str(extra_text).strip():
            meta["supplementary_text_chars"] = len(str(extra_text))
            meta["included_supplementary"] = True
        out = {
            "metadata": meta,
            # Field-extraction usage. Extracts written before 2026-10-24 call
            # this "aspect_tokens"; readers accept both.
            "tokens": {
                "field_tokens": {**total_usage, "model": self.model},
                "scenario_tokens": {**scenario_usage, "model": self.scenario_model},
            },
            "results": results,
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
        if self.extract_sections:
            out["sections_dict"] = sections_dict
        return out

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
