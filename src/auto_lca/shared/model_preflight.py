"""Validate user-overridden LLM models against the caller's Mistral plan.

Plan-default models (from ``model_profiles.json`` via subscription detection)
are trusted and not probed. When the user sets models in config or CLI, we
send a 1-token completion per unique model before a long extract/compare;
on tier rejection we raise with the plan's default pair as the proposed fix.
"""

from __future__ import annotations

import os
import re


_TIER_UNAVAILABLE_MARKERS = (
    re.compile(r"rpm[\"'\s:=]+0\b", re.I),
    re.compile(r"rate[_ ]?limit", re.I),
    re.compile(r"rate_limited", re.I),
    re.compile(r"\b429\b"),
    re.compile(r"\b403\b"),
    re.compile(r"forbidden", re.I),
    re.compile(r"not[_ ]?(allowed|available|authorized|permitted)", re.I),
    re.compile(r"service[_ ]?tier", re.I),
    re.compile(r"tier_not_allowed", re.I),
    re.compile(r"insufficient[_ ]?(quota|funds|balance)", re.I),
    re.compile(r"does not have access", re.I),
    re.compile(r"model[_ ]?(is )?(not|unavailable)", re.I),
    re.compile(r"invalid[_ ]?model", re.I),
    re.compile(r"unknown[_ ]?model", re.I),
)


class ModelUnavailableError(RuntimeError):
    """User-chosen model(s) rejected by the provider for this API key/tier."""

    def __init__(
        self,
        message: str,
        *,
        failures: dict[str, str] | None = None,
        suggestions: dict[str, str] | None = None,
        suggestion_profile: str | None = None,
    ):
        super().__init__(message)
        self.failures = failures or {}
        self.suggestions = suggestions or {}
        self.suggestion_profile = suggestion_profile


def looks_like_tier_block(exc: BaseException) -> bool:
    text = f"{type(exc).__name__}: {exc}"
    return any(p.search(text) for p in _TIER_UNAVAILABLE_MARKERS)


def probe_model(model: str, *, api_key: str | None = None) -> None:
    """One cheap completion; raises on any failure."""
    import litellm

    kwargs = {
        "model": model,
        "messages": [{"role": "user", "content": "ping"}],
        "max_tokens": 1,
        "temperature": 0.0,
        "top_p": 1.0,
    }
    key = api_key or os.environ.get("MISTRAL_API_KEY")
    if key:
        kwargs["api_key"] = key
    litellm.completion(**kwargs)


def ensure_models_available(
    models: dict[str, str],
    *,
    api_key: str | None = None,
    suggestions: dict[str, str] | None = None,
    suggestion_profile: str | None = None,
) -> None:
    """Probe each unique model once; raise ``ModelUnavailableError`` on tier blocks."""
    by_model: dict[str, list[str]] = {}
    for role, model in models.items():
        if not model:
            continue
        by_model.setdefault(model, []).append(role)

    failures: dict[str, str] = {}
    for model, roles in by_model.items():
        try:
            probe_model(model, api_key=api_key)
        except Exception as exc:  # noqa: BLE001
            if looks_like_tier_block(exc):
                role_txt = ", ".join(roles)
                failures[model] = f"{role_txt}: {type(exc).__name__}: {exc}"
            else:
                raise RuntimeError(
                    f"Model check failed for {model!r} ({', '.join(roles)}): {exc}"
                ) from exc

    if not failures:
        return

    lines = [
        "These model(s) are not usable with this API key / Mistral plan:",
    ]
    for model, detail in failures.items():
        lines.append(f"  - {model}: {detail}")

    if suggestions:
        profile_note = (
            f" (profile {suggestion_profile!r})" if suggestion_profile else ""
        )
        lines.append(
            f"Remove the custom model overrides (or fix them) and use the "
            f"plan defaults{profile_note}:"
        )
        for role, model in suggestions.items():
            lines.append(f"  - {role}: {model}")
        lines.append(
            "Unset defaults.model / defaults.production_system_model / "
            "defaults.scenario_mapper_model in the config, and drop "
            "--model / --scenario-model / --mapper-model."
        )
    else:
        lines.append(
            "Unset custom model overrides so plan defaults from "
            "model_profiles.json are used."
        )

    raise ModelUnavailableError(
        "\n".join(lines),
        failures=failures,
        suggestions=suggestions,
        suggestion_profile=suggestion_profile,
    )


def preflight_user_models(
    *,
    model: str | None = None,
    scenario_model: str | None = None,
    mapper_model: str | None = None,
    api_key: str | None = None,
    plan_profile: str | None = None,
    plan_defaults: dict[str, str] | None = None,
    has_user_override: bool = False,
) -> None:
    """Probe only when the user overrode plan defaults (config or CLI)."""
    if not has_user_override:
        return

    from auto_lca.process.sog.model_profiles import plan_default_role_suggestions

    roles: dict[str, str] = {}
    if model:
        roles["field extraction"] = model
    if scenario_model:
        roles["scenario identification"] = scenario_model
    if mapper_model:
        roles["scenario mapper"] = mapper_model

    suggestions = plan_default_role_suggestions(plan_defaults or {})
    ensure_models_available(
        roles,
        api_key=api_key,
        suggestions=suggestions,
        suggestion_profile=plan_profile,
    )
