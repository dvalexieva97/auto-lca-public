"""Plan-aware Mistral rate-limit policy helpers."""

from types import SimpleNamespace

from auto_lca.process.sog.model_profiles import (
    rate_limit_backoff_seconds,
    rate_limit_policy_for,
    retry_after_seconds,
)


def test_free_policy_is_stricter_than_paid():
    free = rate_limit_policy_for("free")
    paid = rate_limit_policy_for("paid")
    assert free["max_concurrency"] < paid["max_concurrency"]
    assert free["min_interval_s"] > paid["min_interval_s"]
    assert free["base_backoff_s"] > paid["base_backoff_s"]
    assert free["rate_limit_retries"] > paid["rate_limit_retries"]


def test_unknown_profile_defaults_to_free():
    assert rate_limit_policy_for("mystery") == rate_limit_policy_for("free")
    assert rate_limit_policy_for(None) == rate_limit_policy_for("free")


def test_retry_after_from_error_headers():
    exc = SimpleNamespace(headers={"Retry-After": "42"}, response=None)
    assert retry_after_seconds(exc) == 42.0


def test_retry_after_from_response_headers():
    exc = SimpleNamespace(
        headers=None,
        response=SimpleNamespace(headers={"retry-after": "7.5"}),
    )
    assert retry_after_seconds(exc) == 7.5


def test_backoff_prefers_retry_after():
    exc = SimpleNamespace(headers={"retry-after": "12"}, response=None)
    wait = rate_limit_backoff_seconds(plan_profile="free", hit=1, exc=exc)
    assert 12.0 <= wait <= 13.0


def test_backoff_scales_with_plan_and_hit():
    free1 = rate_limit_backoff_seconds(plan_profile="free", hit=1)
    free2 = rate_limit_backoff_seconds(plan_profile="free", hit=2)
    paid1 = rate_limit_backoff_seconds(plan_profile="paid", hit=1)
    assert free1 == 60.0
    assert free2 == 120.0
    assert paid1 == 15.0
    assert free1 > paid1
