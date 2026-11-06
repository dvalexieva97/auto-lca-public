"""Shared fixtures. Every test runs offline: no API keys, no model downloads."""

import json
from pathlib import Path

import pymupdf
import pytest

import auto_lca.process.sog as sog_package
from auto_lca.process.sog.agent_sog import AgentExtractor

# Two field sets: one study-wide, one per production system.
TEST_CONFIG = {
    "defaults": {
        "model": "mistral-small-test",
        "production_system_model": "mistral-large-test",
        "by_production_system": True,
        "strip_bibliography": True,
    },
    "field_sets": [
        {
            "name": "Goal and Scope",
            "description": "Study-wide LCA setup.",
            "by_production_system": False,
            "fields": {"Functional Unit": "str", "System Boundary": "str | None"},
        },
        {
            "name": "Inventory",
            "description": "Per-system inputs.",
            "fields": {"Milk yield": "float | None", "Feed": "str | None"},
        },
    ],
}


@pytest.fixture
def write_config(tmp_path):
    """Write a config dict to a JSON file; returns its path as str."""

    def _write(config: dict = None, name: str = "config.json") -> str:
        path = tmp_path / name
        path.write_text(json.dumps(config or TEST_CONFIG))
        return str(path)

    return _write


@pytest.fixture
def make_pdf(tmp_path):
    """Create a one-page PDF containing the given text; returns its path as str."""

    def _make(text: str = "A dairy farm LCA.", name: str = "paper.pdf") -> str:
        path = tmp_path / name
        doc = pymupdf.open()
        page = doc.new_page()
        page.insert_text((72, 72), text, fontsize=9)
        doc.save(path)
        doc.close()
        return str(path)

    return _make


@pytest.fixture
def make_extractor(write_config):
    """AgentExtractor on a test config. Pass a dict to override `defaults` keys."""

    def _make(defaults: dict = None, **kwargs) -> AgentExtractor:
        config = json.loads(json.dumps(TEST_CONFIG))
        config["defaults"].update(defaults or {})
        return AgentExtractor(
            config_path=write_config(config), cache_namespace="test", **kwargs
        )

    return _make


@pytest.fixture
def sample_config_dir() -> Path:
    """The bundled sample configs, wherever the package is installed."""
    return Path(sog_package.__file__).resolve().parent
