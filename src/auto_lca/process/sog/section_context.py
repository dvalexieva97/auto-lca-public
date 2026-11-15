"""
Non-AI paper sectioning for SOG extractors.

Reinstates the cheap header/embedding section split that used to land in
``extracted_sections/{pid}.json``. No LLM calls — PyMuPDF text is assumed
already extracted; headers are matched with local MiniLM via TextExtractor.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from auto_lca.process.nlp.extract import TextExtractor

# Prefer evidence that a case was actually modeled (results / tables) over
# title/abstract alone when classifying own_system_modelled vs _mentioned.
_VALIDATOR_SECTION_KEYS = (
    "Tables",
    "Results",
    "Methods_Results",
    "Methods",
    "Discussion",
)


def build_sections_dict(full_text: str) -> dict[str, dict[str, Any]]:
    """Heuristic Methods/Results/… split; always includes full_text."""
    extractor = TextExtractor()
    sections = extractor.extract_sections_from_text(full_text) or {}
    # Drop empty bodies so callers can fall through cleanly.
    cleaned: dict[str, dict[str, Any]] = {}
    for name, meta in sections.items():
        if not isinstance(meta, dict):
            continue
        text = meta.get("section_text") or ""
        if str(text).strip():
            cleaned[name] = dict(meta)
    cleaned["full_text"] = {
        "section_title": "full_text",
        "section_text": full_text,
        "cosine_similarity": None,
        "fuzzy_cosine_similarity": None,
    }
    return cleaned


def validator_context_from_sections(
    sections: dict[str, dict[str, Any]] | None,
    *,
    max_chars: int = 12000,
    intro_chars: int = 2000,
) -> str:
    """Scoped paper text for scenario category validation.

    Uses Results/Tables/Methods (when detected) plus a short front-matter
    snippet for study scope — not a blind full-PDF dump and not only the
    first 3k chars of the file.
    """
    sections = sections or {}
    chunks: list[str] = []
    full = (sections.get("full_text") or {}).get("section_text") or ""
    if full and intro_chars > 0:
        chunks.append(full[:intro_chars])

    used = 0
    for key in _VALIDATOR_SECTION_KEYS:
        meta = sections.get(key) or {}
        text = (meta.get("section_text") or "").strip()
        if not text:
            continue
        header = f"\n\n===== {key} =====\n"
        room = max_chars - used - len(header)
        if room <= 200:
            break
        piece = text[:room]
        chunks.append(header + piece)
        used += len(header) + len(piece)

    if not chunks and full:
        return full[:max_chars]
    return "".join(chunks)[:max_chars]


def save_extracted_sections(
    output_folder: str | Path,
    pid: str,
    sections: dict[str, dict[str, Any]],
) -> Path:
    """Write ``{output_folder}/extracted_sections/{pid}.json`` (text only)."""
    out_dir = Path(output_folder) / "extracted_sections"
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{pid}.json"
    # Persist a lean copy (drop huge unused keys if any).
    lean = {
        k: {
            "section_title": (v or {}).get("section_title"),
            "cosine_similarity": (v or {}).get("cosine_similarity"),
            "fuzzy_cosine_similarity": (v or {}).get("fuzzy_cosine_similarity"),
            "section_text": (v or {}).get("section_text") or "",
        }
        for k, v in sections.items()
        if isinstance(v, dict)
    }
    path.write_text(json.dumps(lean, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return path


def load_extracted_sections(
    extracts_dir: str | Path, pid: str
) -> dict[str, dict[str, Any]] | None:
    path = Path(extracts_dir) / "extracted_sections" / f"{pid}.json"
    if not path.is_file():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return data if isinstance(data, dict) else None
