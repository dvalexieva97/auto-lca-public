"""
Deterministic, no-LLM-cost provenance check: for every extracted field value
that contains a number, verify the number actually appears somewhere in the
source PDF's full text. Catches a meaningful fraction of hallucinated or
invented values for near-zero cost, on the whole corpus rather than a
hand-picked sample -- inspired by Paper2Agent's pdf_to_skill.py number-token
diff (https://github.com/jmiao24/Paper2Agent, MIT licensed), adapted here to
per-field structured extraction rather than full-page markdown reproduction.

This is NOT a semantic-correctness check: a hallucinated number that happens
to equal a different real number elsewhere in the paper still passes, and a
correct value using a differently-formatted-but-equivalent number may need
normalization to match. Treat it as a cheap triage signal that narrows where
to spend expensive PDF-audit review, not a verdict on its own.

Also defines the per-paper completion status ladder (adapted from
Paper2Agent's phase/status model): mechanical_failure > unreviewed >
unresolved_discrepancies > reviewed_with_limitations > reviewed. Right up to
now this pipeline had no equivalent of "was this actually checked" per paper.
"""

from __future__ import annotations

import re
from collections import Counter
from typing import Optional

import pandas as pd

METADATA_COLS = {"PID", "SCENARIO NAME", "EXCLUDE_FROM_ACCURACY", "EXCLUSION_REASON", "PDF_PATH"}

# Same shape as Paper2Agent's number_tokens(): thousands-separator groups of
# exactly 3 digits, decimals, signs, percents, and scientific notation,
# without conflating a short citation list like "[2,3]" with "2,300". The
# leading sign requires no preceding digit (`(?<!\d)`) so a hyphenated range
# like "TRL 4-5" or "10-15 bar" tokenizes as [4, 5] / [10, 15] -- a range
# dash, not a minus sign -- rather than [4, -5], which was a real false-
# positive generator (range expressions are common in LCA papers: TRL
# brackets, temperature/pressure ranges, percentage ranges).
_NUMBER_TOKEN_RE = re.compile(
    r"(?<!\d)[+−-]?(?:\d{1,3}(?:,\d{3})+(?!\d)|\d+)(?:\.\d+)*(?:[eE][+−-]?\d+)?%?"
)

STATUS_MECHANICAL_FAILURE = "mechanical_failure"
STATUS_UNREVIEWED = "unreviewed"
STATUS_UNRESOLVED_DISCREPANCIES = "unresolved_discrepancies"
STATUS_REVIEWED_WITH_LIMITATIONS = "reviewed_with_limitations"
STATUS_REVIEWED = "reviewed"
STATUS_ORDER = [
    STATUS_MECHANICAL_FAILURE,
    STATUS_UNREVIEWED,
    STATUS_UNRESOLVED_DISCREPANCIES,
    STATUS_REVIEWED_WITH_LIMITATIONS,
    STATUS_REVIEWED,
]


def _is_data_field(col: str, excluded_always: set) -> bool:
    if col.startswith("_") or col.startswith("SCENARIO_"):
        return False
    if col.endswith("_manual") or "__match_" in col:
        return False
    if col.upper() in METADATA_COLS or col.upper().startswith("METADATA"):
        return False
    if col in excluded_always:
        return False
    return True


def _is_empty(v) -> bool:
    if v is None:
        return True
    if isinstance(v, float) and pd.isna(v):
        return True
    return str(v).strip().lower() in {"", "none", "nan", "null"}


def extract_numeric_tokens(text: str) -> Counter:
    """Numeric tokens in `text`, as a Counter (raw, unnormalized)."""
    if not text:
        return Counter()
    return Counter(_NUMBER_TOKEN_RE.findall(text))


def normalize_number(tok: str) -> str:
    """
    Normalize a numeric token so equivalent formattings match: strips
    thousands commas, unifies the unicode minus, drops a leading '+', and
    canonicalizes trailing zeros (1,500 == 1500; 1.50 == 1.5). A trailing
    '%' is kept since 50 and 50% are not interchangeable claims.
    """
    t = tok.strip().replace("−", "-").lstrip("+")
    has_pct = t.endswith("%")
    if has_pct:
        t = t[:-1]
    t = t.replace(",", "")
    if re.fullmatch(r"-?\d+\.0+", t):
        t = t.split(".")[0]
    elif "." in t and "e" not in t.lower() and "E" not in t:
        try:
            t = "%g" % float(t)
        except ValueError:
            pass
    return t + ("%" if has_pct else "")


def check_value_provenance(value, source_tokens_normalized: set) -> Optional[dict]:
    """
    Check the numeric tokens inside `value` against a source document's
    normalized numeric-token set. Returns None if `value` has no numbers
    to check at all -- that's "not applicable", not a pass or a fail.
    """
    if _is_empty(value):
        return None
    raw_tokens = list(extract_numeric_tokens(str(value)).keys())
    if not raw_tokens:
        return None
    normalized = [normalize_number(t) for t in raw_tokens]
    verified = [t for t, n in zip(raw_tokens, normalized) if n in source_tokens_normalized]
    unverified = [t for t, n in zip(raw_tokens, normalized) if n not in source_tokens_normalized]
    if not unverified:
        status = "verified"
    elif verified:
        status = "partially_verified"
    else:
        status = "unverified"
    return {
        "numbers_found": raw_tokens,
        "numbers_verified": verified,
        "numbers_unverified": unverified,
        "status": status,
    }


def _load_pdf_number_tokens(pdf_path: str) -> Optional[set]:
    from auto_lca.process.nlp.extract import TextExtractor

    text = TextExtractor.extract_text_from_pdf(pdf_path)
    if not text:
        return None
    return {normalize_number(t) for t in extract_numeric_tokens(text).keys()}


def compute_field_provenance(
    df_joined: pd.DataFrame,
    excluded_fields_always: Optional[set] = None,
) -> pd.DataFrame:
    """
    One row per (PID, scenario, field) where the auto-extracted value has at
    least one number, with its provenance check against that paper's full
    PDF text. Each paper's PDF is read and tokenized once (cached), no
    matter how many scenario/field rows reference it, using `PDF_PATH`
    already carried in df_joined.
    """
    excluded_fields_always = excluded_fields_always or set()
    data_fields = [c for c in df_joined.columns if _is_data_field(c, excluded_fields_always)]

    token_cache: dict = {}
    rows = []
    for _, row in df_joined.iterrows():
        pdf_path = row.get("PDF_PATH")
        if _is_empty(pdf_path):
            continue
        if pdf_path not in token_cache:
            try:
                token_cache[pdf_path] = _load_pdf_number_tokens(pdf_path)
            except Exception:
                token_cache[pdf_path] = None
        source_tokens = token_cache[pdf_path]
        if source_tokens is None:
            continue
        for f in data_fields:
            result = check_value_provenance(row.get(f), source_tokens)
            if result is None:
                continue
            rows.append(
                {
                    "PID": row.get("PID"),
                    "Scenario": row.get("SCENARIO NAME"),
                    "Field": f,
                    "Value": row.get(f),
                    "Numbers found": ", ".join(result["numbers_found"]),
                    "Numbers unverified": ", ".join(result["numbers_unverified"]),
                    "Status": result["status"],
                }
            )
    return pd.DataFrame(rows)


def compute_paper_status(
    has_data: bool,
    provenance_available: bool,
    has_unverified: bool,
    has_partial: bool,
    is_audited: bool,
) -> str:
    """
    Per-paper completion status, adapted from Paper2Agent's phase/status
    ladder. Evaluated top-down, first match wins -- mirrors the precedence
    style of the source (mechanical problems first, then "nobody's looked",
    then "looks wrong and nobody's looked", then "looked, with caveats",
    then clean).

    `is_audited` means this paper appears in our own qualitative PDF-audit
    trail (mismatch_sampler's random or worst-field samples) -- i.e. a
    human/agent actually read the source PDF and judged specific cells,
    not just the deterministic numeric check.
    """
    if not has_data:
        return STATUS_MECHANICAL_FAILURE
    if not provenance_available and not is_audited:
        return STATUS_UNREVIEWED
    if has_unverified and not is_audited:
        return STATUS_UNRESOLVED_DISCREPANCIES
    if has_unverified or has_partial or is_audited:
        return STATUS_REVIEWED_WITH_LIMITATIONS
    return STATUS_REVIEWED


def compute_paper_status_table(
    df_joined: pd.DataFrame,
    provenance_df: pd.DataFrame,
    audited_pids: Optional[set] = None,
) -> pd.DataFrame:
    """One row per PID with its completion status and supporting counts."""
    audited_pids = audited_pids or set()
    rows = []
    for pid, group in df_joined.groupby("PID", dropna=True):
        # Manual-only rows (e.g. GT pseudo-papers like "unscraped:19" that
        # were never part of our extraction input at all) have no PDF_PATH
        # anywhere in their group -- a completion status doesn't apply to a
        # paper we never attempted, so it's excluded rather than mislabeled
        # mechanical_failure.
        if not group["PDF_PATH"].notna().any():
            continue
        has_data = bool(group["SCENARIO NAME"].notna().any())
        pid_prov = provenance_df[provenance_df["PID"] == pid] if not provenance_df.empty else provenance_df
        provenance_available = bool(len(pid_prov)) or bool(group["PDF_PATH"].notna().any())
        n_unverified = int((pid_prov["Status"] == "unverified").sum()) if len(pid_prov) else 0
        n_partial = int((pid_prov["Status"] == "partially_verified").sum()) if len(pid_prov) else 0
        status = compute_paper_status(
            has_data=has_data,
            provenance_available=provenance_available,
            has_unverified=n_unverified > 0,
            has_partial=n_partial > 0,
            is_audited=pid in audited_pids,
        )
        rows.append(
            {
                "PID": pid,
                "Status": status,
                "N scenarios": int(group["SCENARIO NAME"].notna().sum()),
                "N unverified-number fields": n_unverified,
                "N partially-verified-number fields": n_partial,
                "In PDF audit trail": pid in audited_pids,
            }
        )
    df = pd.DataFrame(rows)
    if not df.empty:
        order = {s: i for i, s in enumerate(STATUS_ORDER)}
        df = df.sort_values("Status", key=lambda s: s.map(order)).reset_index(drop=True)
    return df


def load_audited_pids(audit_sample_paths: list) -> set:
    """Union of every 'pid' appearing across a list of mismatch_sampler
    JSON sample files (random and/or worst-field samples)."""
    import json

    pids: set = set()
    for path in audit_sample_paths:
        try:
            with open(path) as f:
                records = json.load(f)
        except (OSError, ValueError):
            continue
        for r in records:
            pid = r.get("pid")
            if pid:
                pids.add(pid)
    return pids
