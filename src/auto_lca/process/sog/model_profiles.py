"""Mistral model profiles (paid vs free) and resolution helpers.

Default models come from the API key's Mistral plan
(``GET /v1/billing/subscription`` → ``plan_name``). Optional
``defaults.model`` / ``defaults.production_system_model`` /
``defaults.scenario_mapper_model`` in a domain config (or CLI flags) override
those defaults and are validated with a cheap preflight probe.
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from pathlib import Path

_PROFILES_PATH = Path(__file__).with_name("model_profiles.json")
_MISTRAL_SUBSCRIPTION_URL = "https://api.mistral.ai/v1/billing/subscription"

def to_litellm_model(name: str) -> str:
    """Bare Mistral ids (e.g. ``mistral-small-2603``) need a provider prefix."""
    return name if "/" in name else f"mistral/{name}"


def load_model_profiles() -> dict[str, dict[str, str]]:
    with open(_PROFILES_PATH, encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict) or not data:
        raise ValueError(f"Invalid model profiles file: {_PROFILES_PATH}")
    return data


def get_profile(name: str) -> dict[str, str]:
    profiles = load_model_profiles()
    if name not in profiles:
        known = ", ".join(sorted(profiles))
        raise KeyError(f"Unknown model profile {name!r}. Known: {known}")
    return dict(profiles[name])


def fetch_mistral_subscription(api_key: str | None = None) -> dict | None:
    """Return the billing subscription payload, or None if unavailable."""
    key = api_key or os.environ.get("MISTRAL_API_KEY")
    if not key:
        return None
    req = urllib.request.Request(
        _MISTRAL_SUBSCRIPTION_URL,
        headers={"Authorization": f"Bearer {key}"},
        method="GET",
    )
    try:
        with urllib.request.urlopen(req, timeout=20) as resp:
            return json.loads(resp.read().decode())
    except (
        urllib.error.URLError,
        urllib.error.HTTPError,
        TimeoutError,
        json.JSONDecodeError,
        OSError,
    ):
        return None


def profile_from_subscription(
    subscription: dict | None,
) -> tuple[str | None, str | None]:
    """Map a subscription payload to ``(profile_name, plan_name)``.

    ``FREE`` → free profile; any other plan name → paid.
    """
    if not subscription:
        return None, None
    plan = subscription.get("plan_name")
    if plan is None:
        return None, None
    plan_s = str(plan).strip()
    if not plan_s:
        return None, None
    if plan_s.upper() == "FREE":
        return "free", plan_s
    return "paid", plan_s


def detect_model_profile(api_key: str | None = None) -> tuple[str | None, str | None]:
    """Detect ``free``/``paid`` from the API key's Mistral plan."""
    return profile_from_subscription(fetch_mistral_subscription(api_key))


def resolve_profile_models(
    profile: str | None = None,
    *,
    model: str | None = None,
    scenario_model: str | None = None,
    mapper_model: str | None = None,
    config_defaults: dict | None = None,
    api_key: str | None = None,
) -> dict:
    """Resolve extraction / scenario / mapper models.

    Precedence per role:
      1. explicit CLI arg
      2. domain config ``defaults.*`` (user override)
      3. plan profile from Mistral ``plan_name`` (or forced ``--model-profile``)
      4. paid profile as last resort

    Returns models plus metadata: ``plan_profile``, ``plan_name``,
    ``plan_defaults``, ``has_user_override`` (True when any role came from
    CLI or config — those need a preflight check against the plan).
    """
    defaults = config_defaults or {}
    paid = get_profile("paid")
    plan_name: str | None = None
    forced = profile not in (None, "auto")

    if forced:
        if profile not in load_model_profiles():
            known = ", ".join(sorted(load_model_profiles()))
            raise KeyError(f"Unknown model profile {profile!r}. Known: {known}")
        plan_profile = profile
    else:
        detected, plan_name = detect_model_profile(api_key)
        plan_profile = detected or "paid"
        if detected:
            print(
                f"Mistral plan={plan_name!r} → default models from "
                f"profile {plan_profile!r}"
            )
        else:
            print(
                "Could not detect Mistral plan from /v1/billing/subscription; "
                f"using profile {plan_profile!r} defaults "
                "(pass --model-profile paid|free to force)."
            )

    plan_defaults = {
        k: to_litellm_model(v) for k, v in get_profile(plan_profile).items()
    }

    sources: dict[str, str] = {}

    def _pick(
        role: str,
        cli: str | None,
        config_keys: tuple[str, ...],
        plan_keys: tuple[str, ...],
    ) -> str:
        if cli:
            sources[role] = "cli"
            return to_litellm_model(cli)
        for key in config_keys:
            if defaults.get(key):
                sources[role] = "config"
                return to_litellm_model(str(defaults[key]))
        for key in plan_keys:
            if plan_defaults.get(key):
                sources[role] = "plan"
                return plan_defaults[key]
        for key in plan_keys:
            if paid.get(key):
                sources[role] = "paid_fallback"
                return to_litellm_model(paid[key])
        raise ValueError(f"No model configured for role {role!r}")

    field_model = _pick("model", model, ("model",), ("model",))
    scenario = _pick(
        "scenario_model",
        scenario_model,
        ("production_system_model", "scenario_model"),
        ("production_system_model", "scenario_model"),
    )
    mapper = _pick(
        "mapper_model",
        mapper_model,
        ("scenario_mapper_model",),
        ("scenario_mapper_model", "model"),
    )

    has_user_override = any(src in ("cli", "config") for src in sources.values())

    label = plan_profile
    if plan_name:
        label = f"{plan_profile} (plan={plan_name})"
    if has_user_override:
        label = f"{label}+overrides"

    return {
        "model": field_model,
        "scenario_model": scenario,
        "mapper_model": mapper,
        "profile": label,
        "plan_profile": plan_profile,
        "plan_name": plan_name or "",
        "plan_defaults": plan_defaults,
        "has_user_override": has_user_override,
        "sources": sources,
    }


def plan_default_role_suggestions(plan_defaults: dict[str, str]) -> dict[str, str]:
    """Role labels → plan default model ids, for preflight error messages."""
    out = {
        "field extraction": plan_defaults.get("model", ""),
        "scenario identification": plan_defaults.get(
            "production_system_model", plan_defaults.get("scenario_model", "")
        ),
        "scenario mapper": plan_defaults.get(
            "scenario_mapper_model", plan_defaults.get("model", "")
        ),
    }
    return {k: v for k, v in out.items() if v}
