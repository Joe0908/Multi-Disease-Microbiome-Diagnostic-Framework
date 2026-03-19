#!/usr/bin/env python3
"""
Build a first-pass disease panel from:
1) run_level_abundance.tsv
2) marker_taxa_master.tsv (optional)

Outputs:
- disease_panel_v1.tsv
- included_phenotypes_v1.txt

Usage example:
python build_disease_panel_v1.py \
    --run-level run_level_abundance.tsv \
    --marker-master marker_taxa_master.tsv \
    --outdir outputs_disease_panel \
    --rank genus \
    --min-runs 150 \
    --min-marker-projects 1
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

import pandas as pd


VAGUE_LABELS = {
    "",
    "unknown",
    "unspecified",
    "other",
    "others",
    "mixed",
    "unclear",
    "na",
    "n/a",
    "nan",
    "none",
    "null",
    "not provided",
    "not available",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build disease panel v1")
    parser.add_argument("--run-level", required=True, help="Path to run_level_abundance.tsv")
    parser.add_argument("--marker-master", default=None, help="Path to marker_taxa_master.tsv")
    parser.add_argument("--outdir", default="outputs_disease_panel", help="Output directory")
    parser.add_argument("--rank", default="genus", help="Taxonomic rank to keep, e.g. genus or species")
    parser.add_argument("--min-runs", type=int, default=150, help="Minimum number of unique runs for inclusion")
    parser.add_argument(
        "--min-marker-projects",
        type=int,
        default=1,
        help="Minimum number of marker-supporting projects for inclusion when marker file is provided",
    )
    parser.add_argument(
        "--healthy-label",
        default="Health",
        help="Healthy reference label used in marker_taxa_master.tsv",
    )
    return parser.parse_args()


def clean_text(x: object) -> str:
    if pd.isna(x):
        return "unknown"
    s = str(x).strip()
    return s if s else "unknown"


def canonicalize_run_phenotype(label: object) -> str:
    """
    Conservative canonicalization for run_level_abundance phenotype column.
    Keep labels mostly unchanged to avoid accidental over-merging.
    """
    s = clean_text(label)
    low = s.lower()

    if low in VAGUE_LABELS:
        return "unknown"

    # Healthy aliases
    healthy_aliases = {
        "healthy",
        "health",
        "control",
        "controls",
        "normal",
        "normals",
        "healthy control",
        "healthy controls",
    }
    if low in healthy_aliases:
        return "Health"

    return s


def load_run_level_table(path: str, rank: str) -> pd.DataFrame:
    df = pd.read_csv(path, sep="\t", low_memory=False)

    required_cols = {
        "run_id",
        "phenotype",
        "project_id",
        "scientific_name",
        "final_rank",
        "relative_abundance",
    }
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"run_level_abundance.tsv is missing required columns: {sorted(missing)}")

    rank_low = rank.strip().lower()
    df["final_rank"] = df["final_rank"].astype(str).str.lower()
    df = df.loc[df["final_rank"] == rank_low].copy()

    df["phenotype_canonical"] = df["phenotype"].apply(canonicalize_run_phenotype)
    return df


def build_run_counts(run_df: pd.DataFrame) -> pd.DataFrame:
    counts = (
        run_df[["run_id", "project_id", "phenotype", "phenotype_canonical"]]
        .drop_duplicates()
        .groupby("phenotype_canonical", dropna=False)
        .agg(
            n_runs=("run_id", "nunique"),
            n_projects=("project_id", "nunique"),
            phenotype_raw_examples=("phenotype", lambda x: " | ".join(sorted(pd.Series(x).dropna().astype(str).unique())[:10])),
        )
        .reset_index()
    )
    return counts


def summarize_marker_support(marker_path: Optional[str], healthy_label: str) -> Optional[pd.DataFrame]:
    if marker_path is None:
        return None

    marker_df = pd.read_csv(marker_path, sep="\t", low_memory=False)

    required_cols = {
        "scientific_name",
        "project_id",
        "health",
        "disease",
        "phenotype_comparison",
        "direction",
        "nr_projects_marker",
    }
    missing = required_cols - set(marker_df.columns)
    if missing:
        raise ValueError(f"marker_taxa_master.tsv is missing required columns: {sorted(missing)}")

    marker_df["health"] = marker_df["health"].apply(clean_text)
    marker_df["disease"] = marker_df["disease"].apply(clean_text)

    # Keep only Health vs Disease comparisons for v1 disease panel construction
    hvd = marker_df.loc[marker_df["health"].eq(healthy_label)].copy()

    if hvd.empty:
        return pd.DataFrame(
            columns=[
                "phenotype_canonical",
                "marker_rows",
                "marker_taxa_n",
                "marker_projects_n",
                "marker_disease_enriched_rows",
                "marker_health_enriched_rows",
            ]
        )

    summary = (
        hvd.groupby("disease", dropna=False)
        .agg(
            marker_rows=("scientific_name", "size"),
            marker_taxa_n=("scientific_name", "nunique"),
            marker_projects_n=("project_id", "nunique"),
            marker_disease_enriched_rows=("direction", lambda x: int((pd.Series(x) == "Disease-enriched").sum())),
            marker_health_enriched_rows=("direction", lambda x: int((pd.Series(x) == "Health-enriched").sum())),
        )
        .reset_index()
        .rename(columns={"disease": "phenotype_canonical"})
    )

    return summary


def decide_inclusion(
    panel_df: pd.DataFrame,
    min_runs: int,
    min_marker_projects: int,
    use_marker_support: bool,
) -> pd.DataFrame:
    df = panel_df.copy()
    df["included"] = False
    df["exclusion_reason"] = ""

    for idx, row in df.iterrows():
        phenotype = row["phenotype_canonical"]
        n_runs = int(row["n_runs"])

        if phenotype == "unknown":
            df.at[idx, "included"] = False
            df.at[idx, "exclusion_reason"] = "ambiguous_or_missing_label"
            continue

        if phenotype == "Health":
            df.at[idx, "included"] = True
            df.at[idx, "exclusion_reason"] = ""
            continue

        if n_runs < min_runs:
            df.at[idx, "included"] = False
            df.at[idx, "exclusion_reason"] = f"too_few_runs(<{min_runs})"
            continue

        if use_marker_support:
            marker_projects = int(row.get("marker_projects_n", 0) if pd.notna(row.get("marker_projects_n", 0)) else 0)
            if marker_projects < min_marker_projects:
                df.at[idx, "included"] = False
                df.at[idx, "exclusion_reason"] = f"insufficient_marker_support(<{min_marker_projects}_projects)"
                continue

        df.at[idx, "included"] = True
        df.at[idx, "exclusion_reason"] = ""

    return df


def save_outputs(panel_df: pd.DataFrame, outdir: str) -> None:
    out_path = Path(outdir)
    out_path.mkdir(parents=True, exist_ok=True)

    panel_file = out_path / "disease_panel_v1.tsv"
    included_file = out_path / "included_phenotypes_v1.txt"

    ordered_cols = [
        "phenotype_canonical",
        "phenotype_raw_examples",
        "n_runs",
        "n_projects",
        "marker_rows",
        "marker_taxa_n",
        "marker_projects_n",
        "marker_disease_enriched_rows",
        "marker_health_enriched_rows",
        "included",
        "exclusion_reason",
    ]

    existing_cols = [c for c in ordered_cols if c in panel_df.columns]
    panel_df = panel_df[existing_cols].sort_values(
        by=["included", "n_runs", "marker_projects_n"] if "marker_projects_n" in panel_df.columns else ["included", "n_runs"],
        ascending=[False, False, False] if "marker_projects_n" in panel_df.columns else [False, False],
    )

    panel_df.to_csv(panel_file, sep="\t", index=False)

    included = panel_df.loc[panel_df["included"], "phenotype_canonical"].drop_duplicates().tolist()
    with open(included_file, "w", encoding="utf-8") as f:
        for item in included:
            f.write(f"{item}\n")

    print(f"Saved: {panel_file}")
    print(f"Saved: {included_file}")
    print("\nTop included phenotypes:")
    print(panel_df.loc[panel_df["included"]].head(20).to_string(index=False))


def main() -> None:
    args = parse_args()

    run_df = load_run_level_table(args.run_level, rank=args.rank)
    run_counts = build_run_counts(run_df)

    marker_summary = summarize_marker_support(args.marker_master, healthy_label=args.healthy_label)

    if marker_summary is not None:
        panel = run_counts.merge(marker_summary, on="phenotype_canonical", how="left")
        numeric_cols = [
            "marker_rows",
            "marker_taxa_n",
            "marker_projects_n",
            "marker_disease_enriched_rows",
            "marker_health_enriched_rows",
        ]
        for col in numeric_cols:
            if col in panel.columns:
                panel[col] = panel[col].fillna(0).astype(int)

        panel = decide_inclusion(
            panel_df=panel,
            min_runs=args.min_runs,
            min_marker_projects=args.min_marker_projects,
            use_marker_support=True,
        )
    else:
        panel = decide_inclusion(
            panel_df=run_counts,
            min_runs=args.min_runs,
            min_marker_projects=args.min_marker_projects,
            use_marker_support=False,
        )

    save_outputs(panel, args.outdir)


if __name__ == "__main__":
    main()
