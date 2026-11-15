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
from typing import Any

_PROFILES_PATH = Path(__file__).with_name("model_profiles.json")
_MISTRAL_SUBSCRIPTION_URL = "https://api.mistral.ai/v1/billing/subscription"

# Mistral publishes exact RPS/TPM per model only in the Admin → API → Limits
# UI (not via subscription). Free mode is ~1 RPS / tight TPM; paid tiers are
# higher. These policies are conservative defaults keyed off plan_name →
# free|paid; live 429s still prefer Retry-After when present.
RATE_LIMIT_POLICIES: dict[str, dict[str, float | int]] = {
    "free": {
        # Stay near 1 RPS and avoid bursting TPM windows.
        "max_concurrency": 1,
        "min_interval_s": 1.25,
        "rate_limit_retries": 8,
        "base_backoff_s": 60.0,
        "backoff_cap_s": 180.0,
    },
    "paid": {
        "max_concurrency": 6,
        "min_interval_s": 0.0,
        "rate_limit_retries": 4,
        "base_backoff_s": 15.0,
        "backoff_cap_s": 90.0,
    },
}


def rate_limit_policy_for(plan_profile: str | None) -> dict[str, float | int]:
    """Return the free/paid rate-limit policy (defaults to free)."""
    key = (plan_profile or "free").strip().lower()
    if key not in RATE_LIMIT_POLICIES:
        key = "free"
    return dict(RATE_LIMIT_POLICIES[key])


def retry_after_seconds(exc: Any | None) -> float | None:
    """Parse Retry-After from a litellm/httpx rate-limit error, if any."""
    if exc is None:
        return None
    headers = None
    h = getattr(exc, "headers", None)
    if isinstance(h, dict) and h:
        headers = h
    if headers is None:
        resp = getattr(exc, "response", None)
        h = getattr(resp, "headers", None) if resp is not None else None
        if h is not None:
            try:
                headers = dict(h)
            except Exception:
                headers = None
    if not headers:
        return None
    lower = {str(k).lower(): v for k, v in headers.items()}
    raw = lower.get("retry-after")
    if raw is None:
        return None
    try:
        return max(0.0, float(raw))
    except (TypeError, ValueError):
        return None


def rate_limit_backoff_seconds(
    *,
    plan_profile: str | None,
    hit: int,
    exc: Any | None = None,
) -> float:
    """Seconds to sleep after a 429: Retry-After if present, else plan policy."""
    policy = rate_limit_policy_for(plan_profile)
    hinted = retry_after_seconds(exc)
    if hinted is not None and hinted > 0:
        # Add a little jitter so concurrent waiters don't stampede.
        return min(float(policy["backoff_cap_s"]), hinted + 0.5)
    base = float(policy["base_backoff_s"])
    cap = float(policy["backoff_cap_s"])
    # hit is 1-based
    return min(cap, base * max(1, hit))


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


def _resolve_api_key(api_key: str | None = None) -> str | None:
    key = (api_key if api_key is not None else os.environ.get("MISTRAL_API_KEY")) or ""
    key = str(key).strip().strip('"').strip("'")
    return key or None


def _ssl_verify_path() -> str | bool:
    """CA bundle for subscription HTTPS (macOS Python.org builds often lack one).

    litellm/httpx chat calls already use certifi; urllib's default context on
    some Mac installs raises CERTIFICATE_VERIFY_FAILED without it.
    """
    try:
        import certifi

        return certifi.where()
    except Exception:
        return True


def _ssl_context():
    import ssl

    verify = _ssl_verify_path()
    if isinstance(verify, str):
        return ssl.create_default_context(cafile=verify)
    return ssl.create_default_context()


def _subscription_get_json(headers: dict[str, str]) -> tuple[dict | None, str | None]:
    """GET subscription JSON. Prefer httpx (same stack as litellm chat).

    Returns ``(payload, None)`` on success, or ``(None, error_detail)``.

    Cursor (and some corporate setups) inject ``HTTP(S)_PROXY`` that 403s
    Mistral. Retry once with ``trust_env=False`` so we dial api.mistral.ai
    directly — the same path chat often takes after litellm's own fallbacks.
    """
    verify = _ssl_verify_path()
    try:
        import httpx

        def _once(trust_env: bool) -> tuple[dict | None, str | None]:
            with httpx.Client(
                timeout=20.0, trust_env=trust_env, verify=verify
            ) as client:
                resp = client.get(_MISTRAL_SUBSCRIPTION_URL, headers=headers)
            if resp.status_code == 200:
                data = resp.json()
                if isinstance(data, dict):
                    return data, None
                return None, f"unexpected JSON type {type(data).__name__}"
            return None, f"HTTP {resp.status_code}: {resp.text[:200]}"

        try:
            return _once(trust_env=True)
        except httpx.ProxyError as e:
            try:
                return _once(trust_env=False)
            except httpx.HTTPError as e2:
                return None, (
                    f"httpx ProxyError via env proxy ({e}); "
                    f"direct retry {type(e2).__name__}: {e2}"
                )
        except httpx.HTTPError as e:
            # Connect through a broken proxy sometimes surfaces as ConnectError
            # rather than ProxyError — retry direct once.
            msg = str(e).lower()
            if "proxy" in msg or "tunnel" in msg or "403" in msg:
                try:
                    return _once(trust_env=False)
                except httpx.HTTPError as e2:
                    return None, (
                        f"httpx {type(e).__name__}: {e}; "
                        f"direct retry {type(e2).__name__}: {e2}"
                    )
            return None, f"httpx {type(e).__name__}: {e}"
    except ImportError:
        pass

    req = urllib.request.Request(
        _MISTRAL_SUBSCRIPTION_URL, headers=headers, method="GET"
    )
    ctx = _ssl_context()

    def _urllib_once():
        with urllib.request.urlopen(req, timeout=20, context=ctx) as resp:
            data = json.loads(resp.read().decode())
            if isinstance(data, dict):
                return data, None
            return None, f"unexpected JSON type {type(data).__name__}"

    try:
        return _urllib_once()
    except urllib.error.HTTPError as e:
        detail = ""
        try:
            detail = e.read().decode()[:200]
        except Exception:
            pass
        return None, f"HTTP {e.code}" + (f": {detail}" if detail else "")
    except urllib.error.URLError as e:
        # urllib honors HTTP_PROXY too; retry without env proxies.
        reason = e.reason
        reason_s = repr(reason)
        if "403" in reason_s or "Tunnel" in reason_s or "proxy" in reason_s.lower():
            saved = {
                k: os.environ.pop(k)
                for k in list(os.environ)
                if k.lower() in {
                    "http_proxy",
                    "https_proxy",
                    "all_proxy",
                    "no_proxy",
                }
            }
            try:
                return _urllib_once()
            except Exception as e2:
                return None, f"network error ({reason!r}); direct retry: {e2}"
            finally:
                os.environ.update(saved)
        return None, f"network error ({reason!r})"
    except (TimeoutError, json.JSONDecodeError, OSError) as e:
        return None, f"{type(e).__name__}: {e}"


def fetch_mistral_subscription(
    api_key: str | None = None,
    *,
    quiet: bool = False,
) -> dict | None:
    """Return the billing subscription payload, or None if unavailable.

    Tries auth schemes one at a time (never both on the same request — some
    gateways reject dual auth). Order: Bearer (what litellm/chat uses and what
    historically returned ``plan_name`` for us), then ``x-api-key`` (OpenAPI
    ``DashboardContextAuth``).
    """
    key = _resolve_api_key(api_key)
    if not key:
        if not quiet:
            print(
                "Mistral subscription check skipped: no API key "
                "(set MISTRAL_API_KEY or pass api_key)."
            )
        return None

    attempts: list[tuple[str, dict[str, str]]] = [
        ("Bearer", {"Authorization": f"Bearer {key}", "Accept": "application/json"}),
        ("x-api-key", {"x-api-key": key, "Accept": "application/json"}),
    ]
    errors: list[str] = []
    for label, headers in attempts:
        data, err = _subscription_get_json(headers)
        if data is not None:
            return data
        errors.append(f"{label}: {err}")

    if not quiet:
        print(
            "Mistral subscription check failed for /v1/billing/subscription: "
            + " | ".join(errors)
        )
    return None


def _is_free_plan_name(plan_name: str) -> bool:
    """True for Mistral free / Free-mode plan labels."""
    s = plan_name.strip().upper().replace("-", "_").replace(" ", "_")
    if s in {"FREE", "FREE_MODE", "FREEMIUM"}:
        return True
    # e.g. "Free mode", "FREE_TIER"
    return s.startswith("FREE")


def profile_from_subscription(
    subscription: dict | None,
) -> tuple[str | None, str | None]:
    """Map a subscription payload to ``(profile_name, plan_name)``.

    Free / Free-mode plan names → free profile unless Pay-as-you-go or enterprise
    is enabled (Mistral still reports ``plan_name`` ``FREE`` for many PAYG keys).
    Other non-empty plan names → paid (OpenAPI: FREE | INDIVIDUAL | EDU | TEAM).
    """
    if not subscription:
        return None, None
    plan = subscription.get("plan_name")
    if plan is None:
        return None, None
    plan_s = str(plan).strip()
    if not plan_s:
        return None, None
    if subscription.get("payg_enabled") or subscription.get("enterprise"):
        return "paid", plan_s
    if _is_free_plan_name(plan_s):
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
      4. free profile as last resort (safer when detection fails: Ministral +
         free rate-limit pacing; paid Small/Large often 429 on free keys)

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
        plan_profile = detected or "free"
        if detected:
            print(
                f"Mistral plan={plan_name!r} → default models from "
                f"profile {plan_profile!r}"
            )
        else:
            print(
                "Could not detect Mistral plan from /v1/billing/subscription "
                f"(see error above); using profile {plan_profile!r} defaults "
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
