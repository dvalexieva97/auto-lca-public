"""Subscription plan detection and model profile resolution."""

from __future__ import annotations

import io
import json
from urllib.error import HTTPError, URLError

import pytest

from auto_lca.process.sog import model_profiles as mp


class _FakeResp:
    def __init__(self, payload: dict, status: int = 200):
        self.status = status
        self._body = json.dumps(payload).encode()

    def read(self):
        return self._body

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False


def test_is_free_plan_name_variants():
    assert mp._is_free_plan_name("FREE")
    assert mp._is_free_plan_name("free")
    assert mp._is_free_plan_name("Free mode")
    assert mp._is_free_plan_name("FREE_MODE")
    assert not mp._is_free_plan_name("INDIVIDUAL")
    assert not mp._is_free_plan_name("EDU")
    assert not mp._is_free_plan_name("TEAM")


def test_profile_from_subscription_free_and_paid():
    assert mp.profile_from_subscription({"plan_name": "FREE"}) == ("free", "FREE")
    assert mp.profile_from_subscription(
        {"plan_name": "FREE", "payg_enabled": True}
    ) == ("paid", "FREE")
    assert mp.profile_from_subscription(
        {"plan_name": "FREE", "enterprise": True}
    ) == ("paid", "FREE")
    assert mp.profile_from_subscription({"plan_name": "EDU"}) == ("paid", "EDU")
    assert mp.profile_from_subscription({"plan_name": "INDIVIDUAL"}) == (
        "paid",
        "INDIVIDUAL",
    )
    assert mp.profile_from_subscription(None) == (None, None)
    assert mp.profile_from_subscription({}) == (None, None)


def test_fetch_tries_bearer_then_x_api_key(monkeypatch):
    """Dual-auth on one request is avoided; Bearer is tried first."""
    seen: list[dict[str, str]] = []

    def fake_get(headers):
        seen.append({k.lower(): v for k, v in headers.items()})
        # Fail Bearer, succeed x-api-key
        if "authorization" in {k.lower() for k in headers}:
            return None, "HTTP 401"
        return {"plan_name": "FREE"}, None

    monkeypatch.setattr(mp, "_subscription_get_json", fake_get)
    sub = mp.fetch_mistral_subscription("sk-test-key", quiet=True)
    assert sub == {"plan_name": "FREE"}
    assert len(seen) == 2
    assert "authorization" in seen[0]
    assert "x-api-key" in seen[1]
    assert "authorization" not in seen[1]


def test_fetch_succeeds_on_bearer_without_second_attempt(monkeypatch):
    seen: list[dict[str, str]] = []

    def fake_get(headers):
        seen.append(headers)
        return {"plan_name": "FREE", "payg_enabled": False}, None

    monkeypatch.setattr(mp, "_subscription_get_json", fake_get)
    sub = mp.fetch_mistral_subscription("sk-test-key", quiet=True)
    assert sub["plan_name"] == "FREE"
    assert len(seen) == 1
    assert seen[0]["Authorization"].startswith("Bearer ")


def test_fetch_strips_key_whitespace(monkeypatch):
    seen = {}

    def fake_get(headers):
        seen["headers"] = headers
        return {"plan_name": "FREE"}, None

    monkeypatch.setattr(mp, "_subscription_get_json", fake_get)
    mp.fetch_mistral_subscription('  "sk-quoted"  ', quiet=True)
    assert seen["headers"]["Authorization"] == "Bearer sk-quoted"


def test_urllib_http_error_detail(monkeypatch):
    """When httpx is missing, urllib path reports HTTP status."""
    import builtins

    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "httpx" or (isinstance(name, str) and name.startswith("httpx.")):
            raise ImportError("no httpx")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)

    def fake_urlopen(req, timeout=20, context=None):
        raise HTTPError(
            url=mp._MISTRAL_SUBSCRIPTION_URL,
            code=401,
            msg="Unauthorized",
            hdrs=None,
            fp=io.BytesIO(b'{"message":"Unauthorized"}'),
        )

    monkeypatch.setattr(mp.urllib.request, "urlopen", fake_urlopen)
    data, err = mp._subscription_get_json(
        {"Authorization": "Bearer x", "Accept": "application/json"}
    )
    assert data is None
    assert "HTTP 401" in (err or "")


def test_fetch_reports_all_attempt_errors(monkeypatch, capsys):
    monkeypatch.setattr(
        mp,
        "_subscription_get_json",
        lambda headers: (None, "HTTP 403"),
    )
    assert mp.fetch_mistral_subscription("sk-any") is None
    out = capsys.readouterr().out
    assert "Bearer: HTTP 403" in out
    assert "x-api-key: HTTP 403" in out


def test_httpx_retries_direct_after_proxy_error(monkeypatch):
    """Env HTTP_PROXY 403 → retry with trust_env=False."""
    httpx = pytest.importorskip("httpx")

    class FakeResp:
        status_code = 200

        @staticmethod
        def json():
            return {"plan_name": "FREE"}

    calls: list[bool] = []

    class FakeClient:
        def __init__(self, timeout=20.0, trust_env=True, verify=True):
            self.trust_env = trust_env
            self.verify = verify

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def get(self, url, headers=None):
            calls.append(self.trust_env)
            if self.trust_env:
                raise httpx.ProxyError("403 Forbidden")
            return FakeResp()

    monkeypatch.setattr(httpx, "Client", FakeClient)
    data, err = mp._subscription_get_json(
        {"Authorization": "Bearer x", "Accept": "application/json"}
    )
    assert err is None
    assert data == {"plan_name": "FREE"}
    assert calls == [True, False]


def test_ssl_verify_path_prefers_certifi(monkeypatch):
    class FakeCertifi:
        @staticmethod
        def where():
            return "/tmp/fake-cacert.pem"

    monkeypatch.setitem(__import__("sys").modules, "certifi", FakeCertifi)
    assert mp._ssl_verify_path() == "/tmp/fake-cacert.pem"


def test_resolve_falls_back_to_free_when_detection_fails(monkeypatch, capsys):
    monkeypatch.setattr(mp, "detect_model_profile", lambda api_key=None: (None, None))
    resolved = mp.resolve_profile_models(api_key="sk-x")
    assert resolved["plan_profile"] == "free"
    assert "ministral" in resolved["model"]
    assert "using profile 'free'" in capsys.readouterr().out


def test_resolve_uses_detected_paid(monkeypatch, capsys):
    monkeypatch.setattr(
        mp, "detect_model_profile", lambda api_key=None: ("paid", "EDU")
    )
    resolved = mp.resolve_profile_models(api_key="sk-x")
    assert resolved["plan_profile"] == "paid"
    assert resolved["plan_name"] == "EDU"
    assert "mistral-small" in resolved["model"]
    assert "EDU" in capsys.readouterr().out
