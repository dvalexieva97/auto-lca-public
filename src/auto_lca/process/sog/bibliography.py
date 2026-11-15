"""Strip bibliography / references from paper text while keeping post-refs content.

Used as optional AgentExtractor preprocessing (``defaults.strip_bibliography``).
Keeps appendices/tables that appear after the citation list.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

_HEADER_LINE = re.compile(
    r"(?m)^\s*(?:\d+[\.\)]\s*)?"
    r"(references|bibliography|literature cited|works cited|reference list)\s*\.?\s*$",
    re.IGNORECASE,
)
_HEADER_WORD = re.compile(r"\b(references|bibliography)\b", re.IGNORECASE)
_YEAR = re.compile(r"\b(19|20)\d{2}\b")
_DOI_URL = re.compile(r"(doi|https?://|www\.)", re.IGNORECASE)
_AUTHOR = re.compile(r"[A-Z][a-z]+,\s*[A-Z]\.")


@dataclass(frozen=True)
class BibliographyStripResult:
    text: str
    stripped: bool
    refs_start: int | None = None
    bib_end: int | None = None
    bib_chars: int = 0
    post_refs_chars: int = 0
    header: str | None = None
    method: str | None = None


def _candidates(text: str) -> list[tuple[int, int, str, str]]:
    found: list[tuple[int, int, str, str]] = []
    for m in _HEADER_LINE.finditer(text):
        found.append((m.start(), m.end(), "line", m.group(1)))
    # Fallback: word match near start-of-line, only in the late paper body.
    cutoff = int(len(text) * 0.55)
    for m in _HEADER_WORD.finditer(text):
        if m.start() < cutoff:
            continue
        line_start = text.rfind("\n", 0, m.start()) + 1
        prefix = text[line_start : m.start()]
        if len(prefix.strip()) <= 6:
            found.append((m.start(), m.end(), "word_late", m.group(1)))
    return found


def strip_bibliography(
    text: str,
    *,
    min_header_pos_frac: float = 0.40,
) -> BibliographyStripResult:
    """Remove the bibliography citation block; keep text after it.

    If no safe late-paper header is found, returns the original text unchanged.
    """
    if not text or not str(text).strip():
        return BibliographyStripResult(text=text or "", stripped=False)

    cands = _candidates(text)
    if not cands:
        return BibliographyStripResult(text=text, stripped=False)

    half = len(text) * 0.5
    late = [c for c in cands if c[0] >= half]
    pick = (late or cands)[-1]
    refs_start, _header_end, method, header = pick
    pos_frac = refs_start / max(len(text), 1)
    # Refuse early false positives (e.g. a mid-paper "REFERENCES" part label).
    if pos_frac < min_header_pos_frac and not late:
        return BibliographyStripResult(text=text, stripped=False)

    after = text[refs_start:]
    lines = after.split("\n")
    last_ref_idx = 0
    for i, line in enumerate(lines):
        s = line.strip()
        if not s or i == 0:
            continue
        if _YEAR.search(s) or _DOI_URL.search(s) or _AUTHOR.search(s):
            last_ref_idx = i

    if last_ref_idx == 0:
        bib_end = len(text)
        post = ""
    else:
        before = "\n".join(lines[: last_ref_idx + 1])
        bib_end = refs_start + len(before)
        post = text[bib_end:]

    kept = text[:refs_start] + post
    return BibliographyStripResult(
        text=kept,
        stripped=True,
        refs_start=refs_start,
        bib_end=bib_end,
        bib_chars=bib_end - refs_start,
        post_refs_chars=len(post.strip()),
        header=header,
        method=method,
    )
