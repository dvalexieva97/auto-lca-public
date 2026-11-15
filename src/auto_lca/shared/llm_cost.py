"""
EUR cost estimates for Mistral API usage, plus the Aug 2026 40-paper baseline.

Baseline numbers are measured from `outputs-40papers` (mistral-small-latest,
2026-08-28): 7,850,213 total tokens across 40 papers ≈ €1.51 at current
Small list prices. Use this as the cost yardstick when iterating on SOG —
compare €/paper and tokens/paper against these averages, not vibes.
"""

from __future__ import annotations

from typing import Any

# Official Mistral API list prices (EUR / million tokens), regional inference.
# Update if https://docs.mistral.ai/inference/pricing changes.
MISTRAL_EUR_PER_MTOK: dict[str, tuple[float, float]] = {
    # (input, output)
    "mistral-small-latest": (0.12, 0.50),
    "mistral-small-2506": (0.12, 0.50),
    "mistral-small-2603": (0.12, 0.50),
    "mistral/mistral-small-2603": (0.12, 0.50),
    "mistral-large-latest": (0.44, 1.30),
    "mistral-large-2512": (0.44, 1.30),
    "mistral-medium-latest": (1.25, 6.40),
    "mistral/mistral-large-latest": (0.44, 1.30),
    "mistral/mistral-large-2512": (0.44, 1.30),
    "mistral/mistral-small-latest": (0.12, 0.50),
}

# Measured from outputs-40papers (see module docstring).
BASELINE_40PAPERS: dict[str, Any] = {
    "label": "outputs-40papers",
    "n_papers": 40,
    "model": "mistral-small-latest",
    "avg_prompt_tokens": 158_981,
    "avg_completion_tokens": 37_274,
    "avg_total_tokens": 196_255,
    "total_prompt_tokens": 6_359_251,
    "total_completion_tokens": 1_490_962,
    "total_tokens": 7_850_213,
    # €1.51 / 40 at Small list prices on the measured token mix
    "avg_cost_eur": 0.03775,
    "total_cost_eur": 1.51,
}


def normalize_model_name(model: str | None) -> str:
    if not model:
        return "mistral-small-latest"
    return model.strip()


def estimate_cost_eur(
    prompt_tokens: int | float,
    completion_tokens: int | float,
    model: str | None = "mistral-small-latest",
) -> float:
    """Estimate EUR cost from token counts for a known Mistral model alias."""
    key = normalize_model_name(model)
    rates = MISTRAL_EUR_PER_MTOK.get(key)
    if rates is None:
        # Fall back to Small — better a slightly-low estimate than crashing.
        rates = MISTRAL_EUR_PER_MTOK["mistral-small-latest"]
    pin, pout = rates
    return (float(prompt_tokens) / 1e6) * pin + (float(completion_tokens) / 1e6) * pout


def tokens_from_usage_blob(tokens: dict[str, Any] | None) -> tuple[int, int, int]:
    """
    Flatten the nested token dict written into result JSON.

    Supports shapes like:
      {"field_tokens": {"prompt_tokens": …, …}, "scenario_tokens": {…}}
      {"prompt_tokens": …, "completion_tokens": …, "total_tokens": …}
    """
    if not tokens or not isinstance(tokens, dict):
        return 0, 0, 0

    if "prompt_tokens" in tokens or "completion_tokens" in tokens:
        pt = int(tokens.get("prompt_tokens") or 0)
        ct = int(tokens.get("completion_tokens") or 0)
        tt = int(tokens.get("total_tokens") or (pt + ct))
        return pt, ct, tt

    pt = ct = tt = 0
    for value in tokens.values():
        if not isinstance(value, dict):
            continue
        pt += int(value.get("prompt_tokens") or 0)
        ct += int(value.get("completion_tokens") or 0)
        tt += int(value.get("total_tokens") or 0)
    if tt == 0:
        tt = pt + ct
    return pt, ct, tt


def estimate_usage_cost_eur(
    tokens: dict[str, Any] | None,
    default_model: str | None = "mistral-small-latest",
) -> tuple[float, int, int, int]:
    """
    Estimate EUR for a saved token blob.

    If the blob has nested buckets (field_tokens / scenario_tokens), each
    bucket is priced with its own `model` field when present; otherwise
    `default_model` is used. Returns (cost_eur, prompt, completion, total).
    """
    if not tokens or not isinstance(tokens, dict):
        return 0.0, 0, 0, 0

    # Flat shape
    if "prompt_tokens" in tokens or "completion_tokens" in tokens:
        pt, ct, tt = tokens_from_usage_blob(tokens)
        model = tokens.get("model") or default_model
        return estimate_cost_eur(pt, ct, model), pt, ct, tt

    cost = 0.0
    pt = ct = tt = 0
    for value in tokens.values():
        if not isinstance(value, dict):
            continue
        b_pt = int(value.get("prompt_tokens") or 0)
        b_ct = int(value.get("completion_tokens") or 0)
        b_tt = int(value.get("total_tokens") or (b_pt + b_ct))
        model = value.get("model") or default_model
        cost += estimate_cost_eur(b_pt, b_ct, model)
        pt += b_pt
        ct += b_ct
        tt += b_tt
    return cost, pt, ct, tt


def format_paper_cost_line(
    *,
    pid: str,
    prompt_tokens: int,
    completion_tokens: int,
    total_tokens: int,
    model: str | None,
    exec_time_s: float | None = None,
    cost_eur: float | None = None,
) -> str:
    """One-line cost report vs the 40-paper baseline average."""
    cost = (
        cost_eur
        if cost_eur is not None
        else estimate_cost_eur(prompt_tokens, completion_tokens, model)
    )
    base = BASELINE_40PAPERS
    tok_ratio = total_tokens / base["avg_total_tokens"] if base["avg_total_tokens"] else 0
    cost_ratio = cost / base["avg_cost_eur"] if base["avg_cost_eur"] else 0
    time_part = f", {exec_time_s:.0f}s" if exec_time_s is not None else ""
    return (
        f"💶 {pid[:12]}…  {total_tokens:,} tok "
        f"(p={prompt_tokens:,} c={completion_tokens:,})  "
        f"≈€{cost:.3f}  "
        f"vs baseline avg {tok_ratio:.2f}× tokens / {cost_ratio:.2f}× €"
        f"{time_part}"
    )


def format_batch_cost_summary(
    papers: list[dict[str, Any]],
    model: str | None,
) -> str:
    """
    Summarize a canary/batch run against the 40-paper baseline.

    Each item in `papers` should have prompt_tokens, completion_tokens,
    total_tokens, and optionally cost_eur (preferred when mixed models).
    """
    if not papers:
        return "💶 No papers to summarize."

    n = len(papers)
    pt = sum(int(p.get("prompt_tokens") or 0) for p in papers)
    ct = sum(int(p.get("completion_tokens") or 0) for p in papers)
    tt = sum(int(p.get("total_tokens") or 0) for p in papers)
    if any("cost_eur" in p for p in papers):
        cost = sum(float(p.get("cost_eur") or 0) for p in papers)
    else:
        cost = estimate_cost_eur(pt, ct, model)
    base = BASELINE_40PAPERS
    avg_tt = tt / n
    avg_cost = cost / n
    lines = [
        "💶 Cost summary vs baseline (outputs-40papers):",
        f"   this run:  {n} papers, {tt:,} tok, ≈€{cost:.2f}  "
        f"(avg {avg_tt:,.0f} tok / €{avg_cost:.3f} per paper)",
        f"   baseline:  avg {base['avg_total_tokens']:,} tok / "
        f"€{base['avg_cost_eur']:.3f} per paper "
        f"(€{base['total_cost_eur']:.2f} for {base['n_papers']})",
        f"   ratio:     {avg_tt / base['avg_total_tokens']:.2f}× tokens, "
        f"{avg_cost / base['avg_cost_eur']:.2f}× € per paper",
        f"   extrapolated 40-paper cost at this rate: ≈€{avg_cost * 40:.2f}",
    ]
    return "\n".join(lines)
