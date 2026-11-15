"""Extraction domain (``lca`` | ``generic``): resolve + load Python prompt packs.

Domain is never freeform prompt text. Precedence:
  1. explicit CLI ``--domain``
  2. ``defaults.domain`` in the domain config
  3. entry-point default (``auto-lca`` → lca, ``auto-extract`` → generic)
"""

from __future__ import annotations

from auto_lca.process.sog import prompt_templates as prompts

DOMAINS = ("lca", "generic")
DEFAULT_DOMAIN = "lca"

# Domain-specific extract defaults when a config omits the key.
_LCA_EXTRACT_DEFAULTS = {
    "extract_sections": False,
    "include_pdf_tables": False,
    "allow_additional_fields": False,
}
_GENERIC_EXTRACT_DEFAULTS = {
    "extract_sections": False,
    "include_pdf_tables": True,
    "allow_additional_fields": False,
}


def extract_behavior_defaults(domain: str) -> dict[str, bool]:
    """Default extract knobs for ``domain`` (overridable via config ``defaults``)."""
    d = normalize_domain(domain) or DEFAULT_DOMAIN
    base = (
        _GENERIC_EXTRACT_DEFAULTS if d == "generic" else _LCA_EXTRACT_DEFAULTS
    )
    return dict(base)


def normalize_domain(name: str | None) -> str | None:
    if name is None:
        return None
    raw = str(name).strip().lower()
    if not raw:
        return None
    if raw not in DOMAINS:
        known = ", ".join(DOMAINS)
        raise ValueError(f"Unknown domain {name!r}. Expected one of: {known}")
    return raw


def resolve_domain(
    *,
    cli_domain: str | None = None,
    config_defaults: dict | None = None,
    entry_default: str = DEFAULT_DOMAIN,
) -> str:
    """Resolve the active domain for a run."""
    defaults = config_defaults or {}
    for candidate in (
        cli_domain,
        defaults.get("domain"),
        entry_default,
        DEFAULT_DOMAIN,
    ):
        resolved = normalize_domain(candidate) if candidate is not None else None
        if resolved:
            return resolved
    return DEFAULT_DOMAIN


def prompts_for_domain(domain: str) -> dict[str, str]:
    """Return the fixed Python prompt pack for ``domain``."""
    d = normalize_domain(domain) or DEFAULT_DOMAIN
    if d == "generic":
        return {
            "system_extract_concept": prompts.GENERIC_SYSTEM_EXTRACT_CONCEPT,
            "system_extract_scenarios": prompts.GENERIC_SYSTEM_EXTRACT_SCENARIOS,
            "system_validate_scenarios": prompts.GENERIC_SYSTEM_VALIDATE_SCENARIOS,
            "prompt_extract_scenarios": prompts.GENERIC_PROMPT_EXTRACT_SCENARIOS,
            "user_extract_concepts_lead": prompts.GENERIC_USER_EXTRACT_CONCEPTS_LEAD,
        }
    return {
        "system_extract_concept": prompts.LCA_SYSTEM_EXTRACT_CONCEPT,
        "system_extract_scenarios": prompts.LCA_SYSTEM_EXTRACT_SCENARIOS,
        "system_validate_scenarios": prompts.LCA_SYSTEM_VALIDATE_SCENARIOS,
        "prompt_extract_scenarios": prompts.LCA_PROMPT_EXTRACT_SCENARIOS,
        "user_extract_concepts_lead": prompts.LCA_USER_EXTRACT_CONCEPTS_LEAD,
    }
