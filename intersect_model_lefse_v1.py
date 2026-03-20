#!/usr/bin/env python3

from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser(description="Intersect model markers with LEfSe markers")

    parser.add_argument("--model-markers", required=True, help="Path to disease_markers_v1.tsv")
    parser.add_argument("--lefse", required=True, help="Path to marker_taxa_master.tsv")
    parser.add_argument("--outdir", default="outputs_biomarker_intersection_v1")

    parser.add_argument("--taxonomic-rank", default="genus", help="genus or species")
    parser.add_argument("--top-k-per-disease", type=int, default=20, help="Use top K model markers per disease")
    parser.add_argument("--min-lefse-projects", type=int, default=1, help="Minimum supporting LEfSe projects")
    parser.add_argument("--require-same-direction", action="store_true", help="Only keep overlaps with same direction")

    return parser.parse_args()


def normalize_text(x) -> str:
    if pd.isna(x):
        return ""
    return str(x).strip()


def model_direction_from_logfc(logfc: float) -> str:
    return "Disease-enriched" if logfc > 0 else "Health-enriched"


def aggregate_lefse(lefse_df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate LEfSe rows to disease + taxon level.
    """
    rows = []

    grouped = lefse_df.groupby(["disease", "scientific_name"], dropna=False)

    for (disease, taxon), sub in grouped:
        direction_counts = sub["direction"].value_counts(dropna=False).to_dict()

        disease_enriched_n = int((sub["direction"] == "Disease-enriched").sum())
        health_enriched_n = int((sub["direction"] == "Health-enriched").sum())

        if disease_enriched_n > health_enriched_n:
            dominant_direction = "Disease-enriched"
        elif health_enriched_n > disease_enriched_n:
            dominant_direction = "Health-enriched"
        else:
            dominant_direction = "Ambiguous"

        rows.append({
            "disease": disease,
            "taxon": taxon,
            "lefse_rows": len(sub),
            "lefse_projects_n": sub["project_id"].nunique(),
            "lefse_taxonomic_rank": sub["taxonomic_rank"].iloc[0],
            "lefse_disease_enriched_n": disease_enriched_n,
            "lefse_health_enriched_n": health_enriched_n,
            "lefse_dominant_direction": dominant_direction,
            "lefse_mean_abs_lda": sub["lda_score"].abs().mean(),
            "lefse_max_abs_lda": sub["lda_score"].abs().max(),
            "lefse_source_files_n": sub["source_file"].nunique(),
        })

    return pd.DataFrame(rows)


def main():
    args = parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # =========================
    # Load model markers
    # =========================
    model_df = pd.read_csv(args.model_markers, sep="\t", low_memory=False)

    required_model_cols = {"disease", "taxon", "logFC"}
    missing_model = required_model_cols - set(model_df.columns)
    if missing_model:
        raise ValueError(f"Model marker file is missing columns: {sorted(missing_model)}")

    model_df["disease"] = model_df["disease"].apply(normalize_text)
    model_df["taxon"] = model_df["taxon"].apply(normalize_text)

    if "abs_logFC" not in model_df.columns:
        model_df["abs_logFC"] = model_df["logFC"].abs()

    model_df["model_direction"] = model_df["logFC"].apply(model_direction_from_logfc)

    model_df = (
        model_df.sort_values(["disease", "abs_logFC"], ascending=[True, False])
        .groupby("disease", group_keys=False)
        .head(args.top_k_per_disease)
        .copy()
    )

    # =========================
    # Load LEfSe markers
    # =========================
    lefse_df = pd.read_csv(args.lefse, sep="\t", low_memory=False)

    required_lefse_cols = {
        "scientific_name",
        "taxonomic_rank",
        "project_id",
        "health",
        "disease",
        "lda_score",
        "direction",
        "source_file",
    }
    missing_lefse = required_lefse_cols - set(lefse_df.columns)
    if missing_lefse:
        raise ValueError(f"LEfSe file is missing columns: {sorted(missing_lefse)}")

    lefse_df["scientific_name"] = lefse_df["scientific_name"].apply(normalize_text)
    lefse_df["health"] = lefse_df["health"].apply(normalize_text)
    lefse_df["disease"] = lefse_df["disease"].apply(normalize_text)
    lefse_df["taxonomic_rank"] = lefse_df["taxonomic_rank"].astype(str).str.lower()

    # Only use Health vs Disease and requested rank
    lefse_df = lefse_df[
        (lefse_df["health"] == "Health") &
        (lefse_df["taxonomic_rank"] == args.taxonomic_rank.lower())
    ].copy()

    lefse_agg = aggregate_lefse(lefse_df)

    if lefse_agg.empty:
        raise ValueError("No usable LEfSe rows remained after filtering to Health vs Disease and selected rank.")

    lefse_agg = lefse_agg[lefse_agg["lefse_projects_n"] >= args.min_lefse_projects].copy()

    # =========================
    # Intersect
    # =========================
    overlap = model_df.merge(
        lefse_agg,
        on=["disease", "taxon"],
        how="inner"
    )

    if overlap.empty:
        print("No overlap found between model markers and LEfSe markers.")
        overlap.to_csv(outdir / "overlap_biomarkers_all_v1.tsv", sep="\t", index=False)
        return

    overlap["same_direction"] = overlap["model_direction"] == overlap["lefse_dominant_direction"]

    overlap["confidence_score"] = (
        overlap["abs_logFC"] *
        np.log1p(overlap["lefse_projects_n"]) *
        np.where(overlap["same_direction"], 1.5, 0.75)
    )

    overlap = overlap.sort_values(
        ["disease", "same_direction", "confidence_score"],
        ascending=[True, False, False]
    ).copy()

    high_conf = overlap[
        (overlap["same_direction"]) &
        (overlap["lefse_dominant_direction"] != "Ambiguous")
    ].copy()

    if args.require_same_direction:
        final_out = high_conf.copy()
    else:
        final_out = overlap.copy()

    # =========================
    # Save detailed tables
    # =========================
    overlap_path = outdir / "overlap_biomarkers_all_v1.tsv"
    high_conf_path = outdir / "high_confidence_biomarkers_v1.tsv"
    final_out_path = outdir / "final_biomarkers_for_use_v1.tsv"

    overlap.to_csv(overlap_path, sep="\t", index=False)
    high_conf.to_csv(high_conf_path, sep="\t", index=False)
    final_out.to_csv(final_out_path, sep="\t", index=False)

    # =========================
    # Save compact top table
    # =========================
    compact = high_conf.copy()
    compact = compact.sort_values(["disease", "confidence_score"], ascending=[True, False])
    compact = compact.groupby("disease", group_keys=False).head(10)

    compact_cols = [
        "disease",
        "taxon",
        "model_direction",
        "logFC",
        "abs_logFC",
        "lefse_dominant_direction",
        "lefse_projects_n",
        "lefse_rows",
        "lefse_mean_abs_lda",
        "lefse_max_abs_lda",
        "confidence_score",
    ]
    compact = compact[compact_cols]

    compact_path = outdir / "top_high_confidence_biomarkers_per_disease_v1.tsv"
    compact.to_csv(compact_path, sep="\t", index=False)

    print(f"Saved: {overlap_path}")
    print(f"Saved: {high_conf_path}")
    print(f"Saved: {final_out_path}")
    print(f"Saved: {compact_path}")

    print("\nTop high-confidence biomarkers:")
    print(compact.head(40).to_string(index=False))


if __name__ == "__main__":
    main()