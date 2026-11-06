"""Stratified random sample of both-populated disagreement cells + auto modes.

Design
------
* **primary_mode** — one label per cell (priority order 1–7).
* **auto_form / manual_form** — independent side labels (numeric | code |
  short_label | prose). Use these to see *who* is prose / more detailed;
  they are attributes, not a second competing mode.

Sampling
--------
Both-populated cells with fuzzy cosine below a threshold (default 0.5).
Pairs that the numeric equality checker marks exact (incl. unit/format-only)
are dropped — they are not content disagreements. Draw ``n_per_field`` cells
per field, at most one cell per (field, paper), seeded.

Mode labels: heuristic by default; pass an LLM judge
(``create_disagreement_mode_judge``) to classify the sample with AI into
``DISAGREEMENT_CLASSES`` (every cell goes to the LLM, numeric ones included).
Pass ``field_descriptions`` (raw field → schema description) so the judge
sees what each field is meant to hold.
"""

from __future__ import annotations

import argparse
import json
import random
import re
from pathlib import Path
from typing import Any, Callable

import pandas as pd

from auto_lca.statistics.disagreement_modes import (
    BROAD_CATEGORY_ORDER,
    DISAGREEMENT_CLASS_NAMES,
    PRIMARY_MODES,
    assign_disagreement_mode,
    classify_form,
)
from auto_lca.statistics.mismatch_sampler import _is_data_field, _is_empty
from auto_lca.statistics.numeric_parse import classify_numeric_pair


def _matched_rows(df: pd.DataFrame) -> pd.DataFrame:
    exclude_col = (
        df["EXCLUDE_FROM_ACCURACY"]
        if "EXCLUDE_FROM_ACCURACY" in df.columns
        else pd.Series(False, index=df.index)
    )
    return df[
        df["SCENARIO NAME"].notna()
        & df["SCENARIO NAME_manual"].notna()
        & ~exclude_col.fillna(False).astype(bool)
    ].copy()


def _field_group(field: str) -> str:
    """BBP-style ``SECTION_FIELD`` prefix; else whole field name."""
    if "_" in field:
        return field.rsplit("_", 1)[0]
    return field


# Text that only states no value is available ("not mentioned", "N/A: ...").
_STATED_ABSENT = re.compile(
    r"^\W*(not\s+(mentioned|stated|reported|specified|available|given|provided"
    r"|applicable|disclosed|found|known)|n/?a|unavailable|unknown|none|no data|nd)\b",
    re.I,
)


def states_no_value(v: Any) -> bool:
    """True for values that only say the datum is missing (effectively empty)."""
    return bool(_STATED_ABSENT.search(str(v)))


def iter_both_populated_cells(
    df_joined: pd.DataFrame,
    *,
    fuzzy_threshold: float = 0.5,
    excluded_fields: set[str] | None = None,
    metric: str = "fuzzy",
    domain: str = "",
    mode_assigner: Callable[[str, Any, Any], dict[str, Any]] | None = None,
    drop_numeric_exact: bool = True,
    field_descriptions: dict[str, str] | None = None,
    field_roles: dict[str, str] | None = None,
    llm_unequal: bool = False,
    drop_stated_absent: bool = False,
) -> list[dict[str, Any]]:
    """Both-populated cells below fuzzy threshold (candidate disagreement pool).

    ``field_roles`` maps a joined column name to its LCA role label (the
    Table S4 role axis); stored as ``lca_role`` on each row.

    ``llm_unequal`` also admits cells at or above the threshold that the
    comparison's LLM judge marks as different in meaning
    (``__match_llm_meaning_equal`` == 0), e.g. "Yes" vs "No", which embeds
    at cosine 0.73. ``pool_reason`` records "fuzzy" (below threshold) or
    "llm_only". ``drop_stated_absent`` treats values that only say no value
    is available ("not mentioned", "N/A: ...") as empty.

    Requires **both** auto and manual non-empty. If ``drop_numeric_exact``,
    pairs the shared numeric checker marks equal (incl. unit/format-only) are
    excluded even when fuzzy cosine is low (embedding ≠ numeric equality).
    """
    excluded_fields = excluded_fields or set()
    field_descriptions = field_descriptions or {}
    field_roles = field_roles or {}
    assign = mode_assigner or assign_disagreement_mode
    matched = _matched_rows(df_joined)
    fields = [
        c
        for c in df_joined.columns
        if _is_data_field(c, excluded_fields)
    ]
    rows: list[dict[str, Any]] = []
    for _, row in matched.iterrows():
        pid = row["PID"]
        for f in fields:
            auto_v = row.get(f)
            man_v = row.get(f"{f}_manual")
            # both non-empty only
            if _is_empty(auto_v) or _is_empty(man_v):
                continue
            if drop_stated_absent and (
                states_no_value(auto_v) or states_no_value(man_v)
            ):
                continue
            if metric == "fuzzy":
                score = row.get(f"{f}__match_fuzzy_cosine")
            else:
                score = row.get(f"{f}__match_cosine")
            if score is None or (isinstance(score, float) and pd.isna(score)):
                continue
            llm_eq = row.get(f"{f}__match_llm_meaning_equal")
            if float(score) < fuzzy_threshold:
                pool_reason = "fuzzy"
            elif llm_unequal and llm_eq is not None and not pd.isna(llm_eq) and float(llm_eq) == 0:
                pool_reason = "llm_only"
            else:
                continue
            if drop_numeric_exact:
                num = classify_numeric_pair(auto_v, man_v)
                if num.get("numeric_exact") == 1.0:
                    continue
            mode = assign(f, auto_v, man_v)
            rows.append(
                {
                    "domain": domain,
                    "field_group": _field_group(f),
                    "pid": pid,
                    "scenario_auto": row.get("SCENARIO NAME"),
                    "scenario_manual": row.get("SCENARIO NAME_manual"),
                    "field": f,
                    "lca_role": field_roles.get(f, ""),
                    "field_description": field_descriptions.get(f, ""),
                    "auto_value": auto_v,
                    "manual_value": man_v,
                    "cosine": row.get(f"{f}__match_cosine"),
                    "fuzzy_cosine": row.get(f"{f}__match_fuzzy_cosine"),
                    "score_used": float(score),
                    "metric": metric,
                    "threshold": fuzzy_threshold,
                    "pool_reason": pool_reason,
                    "llm_meaning_equal": llm_eq,
                    **mode,
                }
            )
    return rows


def stratified_sample(
    pool: list[dict[str, Any]],
    *,
    n_per_field: int = 3,
    seed: int = 20261008,
    max_total: int | None = 120,
) -> list[dict[str, Any]]:
    """Fixed n per field, ≤1 per (field, paper), then optional global cap."""
    rng = random.Random(seed)
    by_field: dict[str, list[dict[str, Any]]] = {}
    for r in pool:
        by_field.setdefault(r["field"], []).append(r)

    sample: list[dict[str, Any]] = []
    for field in sorted(by_field):
        rows = list(by_field[field])
        rng.shuffle(rows)
        seen_pid: set[str] = set()
        picked: list[dict[str, Any]] = []
        for r in rows:
            if r["pid"] in seen_pid:
                continue
            picked.append(r)
            seen_pid.add(str(r["pid"]))
            if len(picked) >= n_per_field:
                break
        sample.extend(picked)

    rng.shuffle(sample)
    if max_total is not None and len(sample) > max_total:
        # keep stratification as much as possible: trim round-robin by field
        by_f: dict[str, list] = {}
        for r in sample:
            by_f.setdefault(r["field"], []).append(r)
        trimmed: list[dict[str, Any]] = []
        # take 1 from each field cycling until max_total
        while len(trimmed) < max_total and any(by_f.values()):
            for f in list(by_f.keys()):
                if not by_f[f]:
                    continue
                trimmed.append(by_f[f].pop(0))
                if len(trimmed) >= max_total:
                    break
        sample = trimmed
    return sample


def stratified_sample_by_role(
    pool: list[dict[str, Any]],
    *,
    n_per_role: int | dict[str, int] = 30,
    seed: int = 20261008,
    exclude: set[tuple] | None = None,
) -> list[dict[str, Any]]:
    """Fixed n per ``lca_role``, spread across domains and fields.

    ``n_per_role`` may be a dict (role label → n; roles missing → 0).
    ``exclude`` holds (domain, field, pid) keys already sampled, e.g. by an
    earlier draw that this one tops up.

    Within a role, picks greedily the cell whose field, then domain, has
    been drawn least so far (ties broken at random), with ≤1 cell per
    (field, paper). Roles with fewer candidates contribute all of them.
    """
    rng = random.Random(seed)
    by_role: dict[str, list[dict[str, Any]]] = {}
    for r in pool:
        if r.get("lca_role"):  # unmapped / excluded-from-role fields skipped
            by_role.setdefault(r["lca_role"], []).append(r)

    sample: list[dict[str, Any]] = []
    for role in sorted(by_role):
        cands = list(by_role[role])
        rng.shuffle(cands)
        tie = {id(r): i for i, r in enumerate(cands)}
        n_field: dict[tuple, int] = {}
        n_dom: dict[str, int] = {}
        used: set[tuple] = set(exclude or ())
        picked: list[dict[str, Any]] = []
        n_role = n_per_role.get(role, 0) if isinstance(n_per_role, dict) else n_per_role
        while len(picked) < n_role:
            avail = [
                r
                for r in cands
                if (r["domain"], r["field"], str(r["pid"])) not in used
            ]
            if not avail:
                break
            best = min(
                avail,
                key=lambda r: (
                    n_field.get((r["domain"], r["field"]), 0),
                    n_dom.get(r["domain"], 0),
                    tie[id(r)],
                ),
            )
            picked.append(best)
            used.add((best["domain"], best["field"], str(best["pid"])))
            fk = (best["domain"], best["field"])
            n_field[fk] = n_field.get(fk, 0) + 1
            n_dom[best["domain"]] = n_dom.get(best["domain"], 0) + 1
        sample.extend(picked)
    rng.shuffle(sample)
    return sample


def summarize_by_role(
    rows: list[dict[str, Any]], col: str = "broad_category"
) -> pd.DataFrame:
    """LCA role × ``col`` counts with a row total and row % (long format)."""
    df = pd.DataFrame([r for r in rows if r.get(col)])
    if df.empty or "lca_role" not in df.columns:
        return pd.DataFrame(columns=["lca_role", col, "n", "pct_of_role"])
    out = df.groupby(["lca_role", col]).size().reset_index(name="n")
    out["n_role"] = out.groupby("lca_role")["n"].transform("sum")
    out["pct_of_role"] = (out["n"] / out["n_role"] * 100).round(1)
    order = (
        {c: i for i, c in enumerate(BROAD_CATEGORY_ORDER)}
        if col == "broad_category"
        else {c: i for i, c in enumerate(DISAGREEMENT_CLASS_NAMES)}
    )
    out["_ord"] = out[col].map(lambda x: order.get(x, 99))
    return (
        out.sort_values(["lca_role", "_ord"]).drop(columns="_ord").reset_index(drop=True)
    )


def role_crosstab(rows: list[dict[str, Any]], col: str = "broad_category") -> pd.DataFrame:
    """Wide LCA role × ``col`` table of 'n (pct%)' cells, for reading."""
    long = summarize_by_role(rows, col)
    if long.empty:
        return long
    long["cell"] = long.apply(lambda r: f"{r['n']} ({r['pct_of_role']:.0f}%)", axis=1)
    wide = long.pivot(index="lca_role", columns=col, values="cell").fillna("")
    order = BROAD_CATEGORY_ORDER if col == "broad_category" else DISAGREEMENT_CLASS_NAMES
    cols = [c for c in order if c in wide.columns]
    wide = wide[cols + [c for c in wide.columns if c not in cols]]
    wide.insert(0, "n", long.groupby("lca_role")["n"].sum())
    return wide


def summarize_modes(rows: list[dict[str, Any]]) -> pd.DataFrame:
    if not rows:
        return pd.DataFrame(columns=["primary_mode", "n", "pct"])
    s = pd.Series([r["primary_mode"] for r in rows])
    vc = s.value_counts()
    out = pd.DataFrame({"primary_mode": vc.index, "n": vc.values})
    out["pct"] = (out["n"] / out["n"].sum() * 100).round(1)
    # stable order
    order = {
        m: i
        for i, m in enumerate(
            DISAGREEMENT_CLASS_NAMES
            + tuple(m for m in PRIMARY_MODES if m not in DISAGREEMENT_CLASS_NAMES)
        )
    }
    out["_ord"] = out["primary_mode"].map(lambda x: order.get(x, 99))
    return out.sort_values("_ord").drop(columns="_ord").reset_index(drop=True)


def summarize_broad_categories(rows: list[dict[str, Any]]) -> pd.DataFrame:
    """Counts by ``broad_category`` (LLM-labeled rows only)."""
    cats = [r["broad_category"] for r in rows if r.get("broad_category")]
    if not cats:
        return pd.DataFrame(columns=["broad_category", "n", "pct"])
    vc = pd.Series(cats).value_counts()
    out = pd.DataFrame({"broad_category": vc.index, "n": vc.values})
    out["pct"] = (out["n"] / out["n"].sum() * 100).round(1)
    order = {c: i for i, c in enumerate(BROAD_CATEGORY_ORDER)}
    out["_ord"] = out["broad_category"].map(lambda x: order.get(x, 99))
    return out.sort_values("_ord").drop(columns="_ord").reset_index(drop=True)


def form_crosstab(rows: list[dict[str, Any]]) -> pd.DataFrame:
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    return pd.crosstab(df["manual_form"], df["auto_form"], margins=True)


_CSV_COL_ORDER = [
    "domain",
    "field_group",
    "field",
    "lca_role",
    "pid",
    "scenario_auto",
    "scenario_manual",
    "auto_value",
    "manual_value",
    "primary_mode",
    "run_agreement",
    "run_modes",
    "broad_category",
    "class_definition",
    "rationale",
    "unit_conversion",
    "mode_rank",
    "auto_form",
    "manual_form",
    "granularity_direction",
    "code_side",
    "code_supports",
    "empty_side",
    "unit_correct_side",
    "code_inferable",
    "cosine",
    "fuzzy_cosine",
    "score_used",
    "metric",
    "threshold",
    "pool_reason",
    "llm_meaning_equal",
]


def _ordered_df(rows: list[dict[str, Any]]) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    front = [c for c in _CSV_COL_ORDER if c in df.columns]
    rest = [c for c in df.columns if c not in front]
    return df[front + rest]


def _write_outputs(
    out_dir: Path,
    *,
    sample: list[dict[str, Any]],
    pool: list[dict[str, Any]],
    summary: dict[str, Any],
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    _ordered_df(sample).to_csv(out_dir / "stratified_sample.csv", index=False)
    _ordered_df(pool).to_csv(out_dir / "mismatch_pool.csv", index=False)
    summarize_modes(pool).to_csv(out_dir / "mode_counts_pool.csv", index=False)
    summarize_modes(sample).to_csv(out_dir / "mode_counts_sample.csv", index=False)
    form_crosstab(sample).to_csv(out_dir / "form_crosstab_sample.csv")
    broad = summarize_broad_categories(sample)
    if not broad.empty:
        broad.to_csv(out_dir / "broad_category_counts_sample.csv", index=False)
    if any(r.get("lca_role") for r in sample):
        for col, name in (("broad_category", "broad"), ("primary_mode", "class")):
            if any(r.get(col) for r in sample):
                summarize_by_role(sample, col).to_csv(
                    out_dir / f"role_x_{name}_long.csv", index=False
                )
                role_crosstab(sample, col).to_csv(out_dir / f"role_x_{name}.csv")
    # per-domain mode tallies (useful for multi-domain packs)
    if sample and "domain" in sample[0]:
        by_dom = (
            pd.DataFrame(sample)
            .groupby(["domain", "primary_mode"], dropna=False)
            .size()
            .reset_index(name="n")
        )
        by_dom.to_csv(out_dir / "mode_counts_sample_by_domain.csv", index=False)
    (out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, default=str) + "\n"
    )


def load_excluded_field_names(path: Path | None) -> set[str]:
    if path is None or not path.is_file():
        return set()
    raw = json.loads(path.read_text())
    if isinstance(raw, list):
        out: set[str] = set()
        for item in raw:
            if isinstance(item, str):
                out.add(item)
            elif isinstance(item, dict) and item.get("field"):
                out.add(str(item["field"]))
        return out
    return set()


def _apply_modes(
    rows: list[dict[str, Any]],
    mode_assigner: Callable[..., Any] | None,
) -> list[dict[str, Any]]:
    """Label ``rows`` with LLM (sample only) or leave heuristic labels."""
    if mode_assigner is None:
        return rows
    judge_many = getattr(mode_assigner, "judge_many", None)
    if callable(judge_many):
        return judge_many(rows)
    out = []
    for r in rows:
        mode = mode_assigner(r.get("field", ""), r.get("auto_value"), r.get("manual_value"))
        out.append({**r, **mode})
    return out


def build_from_joined(
    joined_path: Path,
    *,
    fuzzy_threshold: float = 0.5,
    n_per_field: int = 3,
    seed: int = 20261008,
    max_total: int = 120,
    excluded_fields: set[str] | None = None,
    out_dir: Path | None = None,
    domain: str = "bbp",
    mode_assigner: Callable[[str, Any, Any], dict[str, Any]] | None = None,
    drop_numeric_exact: bool = True,
) -> dict[str, Any]:
    df = pd.read_excel(joined_path, sheet_name="Joined Data")
    # Pool always uses cheap heuristic; LLM (if any) runs on the sample only.
    pool = iter_both_populated_cells(
        df,
        fuzzy_threshold=fuzzy_threshold,
        excluded_fields=excluded_fields,
        metric="fuzzy",
        domain=domain,
        mode_assigner=None,
        drop_numeric_exact=drop_numeric_exact,
    )
    sample = stratified_sample(
        pool, n_per_field=n_per_field, seed=seed, max_total=max_total
    )
    sample = _apply_modes(sample, mode_assigner)
    summary = {
        "domain": domain,
        "joined": str(joined_path),
        "fuzzy_threshold": fuzzy_threshold,
        "n_per_field": n_per_field,
        "seed": seed,
        "pool_n": len(pool),
        "sample_n": len(sample),
        "n_fields_in_pool": len({r["field"] for r in pool}),
        "n_fields_in_sample": len({r["field"] for r in sample}),
        "mode_counts_pool": summarize_modes(pool).to_dict(orient="records"),
        "mode_counts_sample": summarize_modes(sample).to_dict(orient="records"),
        "form_crosstab_sample": form_crosstab(sample).to_dict(),
        "design_note": (
            "primary_mode is mutually exclusive (priority 1–7). "
            "auto_form/manual_form are per-side attributes so you can see "
            "who is prose / numeric / coded; granularity_direction and "
            "code_side refine modes 2–3. "
            "LLM mode labels (if enabled) apply to the sample only; "
            "pool stays heuristic."
        ),
    }
    if out_dir is not None:
        _write_outputs(out_dir, sample=sample, pool=pool, summary=summary)
    return {"summary": summary, "sample": sample, "pool": pool}


def build_cross_domain(
    domains: list[dict[str, Any]],
    *,
    fuzzy_threshold: float = 0.5,
    n_per_field: int = 1,
    max_per_domain: int = 5,
    seed: int = 20261008,
    out_dir: Path | None = None,
    mode_assigner: Callable[[str, Any, Any], dict[str, Any]] | None = None,
    drop_numeric_exact: bool = True,
    stratify: str = "field",
    n_per_role: int = 30,
) -> dict[str, Any]:
    """Stratify within each domain, then concatenate (``stratify="field"``),
    or pool all domains and draw ``n_per_role`` cells per LCA role
    (``stratify="role"``; needs ``field_roles`` in the specs).

    ``domains`` entries: ``{key, joined, excluded_fields?, field_descriptions?,
    field_roles?}``.
    """
    pool_all: list[dict[str, Any]] = []
    sample_all: list[dict[str, Any]] = []
    per_domain: list[dict[str, Any]] = []
    for i, spec in enumerate(domains):
        key = str(spec["key"])
        joined = Path(spec["joined"])
        excl = set(spec.get("excluded_fields") or set())
        descs = dict(spec.get("field_descriptions") or {})
        df = pd.read_excel(joined, sheet_name="Joined Data")
        # Heuristic pool only — LLM labels applied after sampling.
        pool = iter_both_populated_cells(
            df,
            fuzzy_threshold=fuzzy_threshold,
            excluded_fields=excl,
            metric="fuzzy",
            domain=key,
            mode_assigner=None,
            drop_numeric_exact=drop_numeric_exact,
            field_descriptions=descs,
            field_roles=dict(spec.get("field_roles") or {}),
        )
        if stratify == "role":
            pool_all.extend(pool)
            per_domain.append(
                {
                    "domain": key,
                    "joined": str(joined),
                    "pool_n": len(pool),
                    "n_fields_in_pool": len({r["field"] for r in pool}),
                }
            )
            continue
        # independent shuffle per domain (seed+offset) so domains don't share order
        sample = stratified_sample(
            pool,
            n_per_field=n_per_field,
            seed=seed + i * 17,
            max_total=max_per_domain,
        )
        sample = _apply_modes(sample, mode_assigner)
        pool_all.extend(pool)
        sample_all.extend(sample)
        per_domain.append(
            {
                "domain": key,
                "joined": str(joined),
                "pool_n": len(pool),
                "sample_n": len(sample),
                "n_fields_in_pool": len({r["field"] for r in pool}),
                "n_fields_in_sample": len({r["field"] for r in sample}),
                "mode_counts_sample": summarize_modes(sample).to_dict(
                    orient="records"
                ),
            }
        )
    if stratify == "role":
        sample_all = _apply_modes(
            stratified_sample_by_role(pool_all, n_per_role=n_per_role, seed=seed),
            mode_assigner,
        )
        for d in per_domain:
            d["sample_n"] = sum(r["domain"] == d["domain"] for r in sample_all)
    agreements = [r["run_agreement"] for r in sample_all if r.get("run_agreement")]
    summary = {
        "domains": [d["domain"] for d in per_domain],
        "judge_run_agreement": {
            k: int(v) for k, v in pd.Series(agreements).value_counts().items()
        }
        if agreements
        else {},
        "fuzzy_threshold": fuzzy_threshold,
        "stratify": stratify,
        "n_per_field": n_per_field,
        "max_per_domain": max_per_domain,
        "n_per_role": n_per_role,
        "seed": seed,
        "pool_n_by_role": {
            k: int(v)
            for k, v in pd.Series([r.get("lca_role", "") for r in pool_all])
            .value_counts()
            .items()
        },
        "pool_n": len(pool_all),
        "sample_n": len(sample_all),
        "per_domain": per_domain,
        "mode_counts_pool": summarize_modes(pool_all).to_dict(orient="records"),
        "mode_counts_sample": summarize_modes(sample_all).to_dict(orient="records"),
        "form_crosstab_sample": form_crosstab(sample_all).to_dict(),
        "design_note": (
            (
                f"Role-stratified sample: {n_per_role} cells per LCA role "
                "(all if fewer), pooled across domains, spread over fields "
                "and domains, ≤1 per (field, paper)."
                if stratify == "role"
                else "Cross-domain tiny sample: stratified within each domain "
                f"(n_per_field={n_per_field}, max_per_domain={max_per_domain}), "
                "then concatenated."
            )
            + " LLM labels (if enabled) apply to the sample "
            "only and use DISAGREEMENT_CLASSES (0–14, first that fits); the "
            "pool keeps the heuristic 1–7 modes. auto_form/manual_form are "
            "per-side attributes."
        ),
    }
    if out_dir is not None:
        _write_outputs(out_dir, sample=sample_all, pool=pool_all, summary=summary)
    return {"summary": summary, "sample": sample_all, "pool": pool_all}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--joined", required=True, type=Path)
    p.add_argument("--out-dir", required=True, type=Path)
    p.add_argument("--fuzzy-threshold", type=float, default=0.5)
    p.add_argument("--n-per-field", type=int, default=3)
    p.add_argument("--max-total", type=int, default=120)
    p.add_argument("--seed", type=int, default=20261008)
    p.add_argument("--domain", type=str, default="bbp")
    p.add_argument(
        "--exclude-field",
        action="append",
        default=[],
        help="Field raw name to exclude (repeatable)",
    )
    args = p.parse_args()
    result = build_from_joined(
        args.joined,
        fuzzy_threshold=args.fuzzy_threshold,
        n_per_field=args.n_per_field,
        seed=args.seed,
        max_total=args.max_total,
        excluded_fields=set(args.exclude_field or []),
        out_dir=args.out_dir,
        domain=args.domain,
    )
    s = result["summary"]
    print(
        f"pool={s['pool_n']} sample={s['sample_n']} "
        f"fields_pool={s['n_fields_in_pool']} → {args.out_dir}"
    )
    print(summarize_modes(result["sample"]).to_string(index=False))


if __name__ == "__main__":
    main()
