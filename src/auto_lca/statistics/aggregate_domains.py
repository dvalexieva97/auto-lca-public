"""
Aggregates comparison results (joined_*.xlsx from sog_runner.run_comparison)
across multiple domains -- one row per domain plus a pooled "ALL NON-AGRI"
row (every domain except dairy) and a pooled "ALL DOMAINS" row.

Reuses compare._calculate_comprehensive_statistics on the POOLED "Joined
Data" rows for each aggregate, rather than averaging each domain's
already-computed summary stats -- that would implicitly weight every
domain equally regardless of how many scenarios/fields it actually
contributes. A "Domain" column is tagged onto every row so the pooled
sheets stay traceable back to their source domain.

Usage:
    python -m auto_lca.statistics.aggregate_domains \\
        --domain dairy=outputs-40papers/stats/joined20260828_211450.xlsx \\
        --domain chemicals=___chemicals_test/output-7papers/stats/joined20260828_174053.xlsx \\
        --domain plastics=__other_domains_test/output-plastics/stats/joined20260831_201051.xlsx \\
        --agri-domain dairy \\
        --output outputs-domain-aggregate/domain_aggregate.xlsx
"""

import argparse
import os

import pandas as pd

from auto_lca.statistics.compare import _calculate_comprehensive_statistics


def load_joined(domain: str, path: str, sheet_name: str = "Joined Data") -> pd.DataFrame:
    df = pd.read_excel(path, sheet_name=sheet_name)
    df = df.copy()
    df.insert(0, "DOMAIN", domain)
    return df


def summarize(df_joined: pd.DataFrame) -> pd.DataFrame:
    """Run the same comprehensive-statistics computation the per-domain
    comparison reports use, but keep only the Summary Statistics table --
    that's the one that rolls up cleanly across a pooled multi-domain
    frame without needing every domain to share identical columns."""
    (
        _stats_by_scenario,
        _stats_by_paper,
        _stats_by_field_scenario,
        _stats_by_field_paper,
        _stats_by_field,
        df_summary_stats,
        _sensitivity,
        _f1,
    ) = _calculate_comprehensive_statistics(df_joined, None)
    return df_summary_stats


def build_domain_aggregate_report(
    domain_paths: dict[str, str],
    agri_domains: set[str],
    output_path: str,
    sheet_name: str = "Joined Data",
) -> str:
    """
    domain_paths: {domain_name: path_to_joined_xlsx}
    agri_domains: domain names to exclude from the "ALL NON-AGRI" pool
                  (e.g. {"dairy"})
    """
    per_domain_joined: dict[str, pd.DataFrame] = {}
    for domain, path in domain_paths.items():
        print(f"Loading {domain}: {path}")
        per_domain_joined[domain] = load_joined(domain, path, sheet_name=sheet_name)

    # Per-domain summary rows, each tagged with which domain they're for.
    per_domain_summaries = []
    for domain, df in per_domain_joined.items():
        summary = summarize(df)
        summary.insert(0, "Domain", domain)
        per_domain_summaries.append(summary)

    non_agri = [d for d in per_domain_joined if d not in agri_domains]
    aggregate_rows = []
    if non_agri:
        pooled_non_agri = pd.concat(
            [per_domain_joined[d] for d in non_agri], ignore_index=True
        )
        summary_non_agri = summarize(pooled_non_agri)
        summary_non_agri.insert(0, "Domain", f"ALL NON-AGRI ({', '.join(non_agri)})")
        aggregate_rows.append(summary_non_agri)

    if per_domain_joined:
        pooled_all = pd.concat(list(per_domain_joined.values()), ignore_index=True)
        summary_all = summarize(pooled_all)
        summary_all.insert(0, "Domain", f"ALL DOMAINS ({', '.join(per_domain_joined)})")
        aggregate_rows.append(summary_all)

    combined = pd.concat(per_domain_summaries + aggregate_rows, ignore_index=True)
    # Pivot so each domain/aggregate is a column and each statistic is a row
    # -- much easier to eyeball differences across domains side by side
    # than the long format _calculate_comprehensive_statistics returns.
    wide = combined.pivot(index="Statistic", columns="Domain", values="Value")
    # Keep a stable, readable column order: real domains first (in the
    # order they were passed), then the two pooled aggregates last.
    ordered_cols = list(domain_paths.keys()) + [
        c for c in wide.columns if c not in domain_paths
    ]
    wide = wide[[c for c in ordered_cols if c in wide.columns]]

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
        wide.to_excel(writer, sheet_name="Domain Comparison")
        combined.to_excel(writer, sheet_name="Domain Summary (long)", index=False)
        for domain, df in per_domain_joined.items():
            # Sheet names capped at 31 chars by Excel
            df.to_excel(writer, sheet_name=f"{domain[:25]} raw"[:31], index=False)

    print(f"\nWrote {output_path}")
    print(wide.to_string())
    return output_path


def _parse_domain_arg(value: str) -> tuple[str, str]:
    if "=" not in value:
        raise argparse.ArgumentTypeError(
            f"--domain expects NAME=PATH, got: {value!r}"
        )
    name, path = value.split("=", 1)
    return name.strip(), path.strip()


def main():
    parser = argparse.ArgumentParser(
        description="Aggregate comparison reports across domains, with a pooled non-agri view."
    )
    parser.add_argument(
        "--domain",
        action="append",
        type=_parse_domain_arg,
        required=True,
        help="NAME=PATH to a joined_*.xlsx comparison report. Repeatable.",
    )
    parser.add_argument(
        "--agri-domain",
        action="append",
        default=[],
        help="Domain name to exclude from the ALL NON-AGRI pool (repeatable). Default: 'dairy'.",
    )
    parser.add_argument(
        "--output",
        default="outputs-domain-aggregate/domain_aggregate.xlsx",
        help="Path to write the aggregate report to.",
    )
    args = parser.parse_args()

    domain_paths = dict(args.domain)
    agri_domains = set(args.agri_domain) or {"dairy"}
    build_domain_aggregate_report(domain_paths, agri_domains, args.output)


if __name__ == "__main__":
    main()
