#!/usr/bin/env python3

from __future__ import annotations

import argparse
from pathlib import Path
import pandas as pd
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(description="Build multi-disease microbiome index v1")

    parser.add_argument("--biomarkers", required=True, help="top_high_confidence_biomarkers_per_disease_v1.tsv")
    parser.add_argument("--X", required=True, help="X_filtered_v1.tsv")
    parser.add_argument("--y", required=True, help="y_v1.tsv")
    parser.add_argument("--outdir", default="outputs_multidisease_index_v1")

    parser.add_argument("--top-enriched", type=int, default=10)
    parser.add_argument("--top-depleted", type=int, default=10)
    parser.add_argument("--min-prev-disease", type=float, default=0.10)
    parser.add_argument("--use-weights", action="store_true")

    return parser.parse_args()


def compute_weighted_mean(df: pd.DataFrame, weights: pd.Series) -> pd.Series:
    if df.shape[1] == 0:
        return pd.Series(0.0, index=df.index)

    w = weights.reindex(df.columns).fillna(1.0).astype(float)
    numerator = df.mul(w, axis=1).sum(axis=1)
    denominator = w.sum()

    if denominator == 0:
        return pd.Series(0.0, index=df.index)

    return numerator / denominator


def select_index_panel(
    biomarkers: pd.DataFrame,
    X: pd.DataFrame,
    y: pd.Series,
    top_enriched: int,
    top_depleted: int,
    min_prev_disease: float,
) -> pd.DataFrame:
    selected = []

    for disease in sorted(biomarkers["disease"].unique()):
        sub = biomarkers[biomarkers["disease"] == disease].copy()

        disease_run_ids = y[y == disease].index
        X_d = X.loc[disease_run_ids]

        taxa_in_X = [t for t in sub["taxon"].unique() if t in X.columns]
        if not taxa_in_X:
            continue

        prev_d = (X_d[taxa_in_X] > 0).sum(axis=0) / len(X_d)
        prev_df = prev_d.rename("prev_disease").reset_index().rename(columns={"index": "taxon"})

        sub = sub.merge(prev_df, on="taxon", how="left")
        sub["prev_disease"] = sub["prev_disease"].fillna(0.0)

        sub = sub[sub["prev_disease"] >= min_prev_disease].copy()

        if sub.empty:
            continue

        enriched = (
            sub[sub["model_direction"] == "Disease-enriched"]
            .sort_values(["confidence_score", "prev_disease"], ascending=[False, False])
            .head(top_enriched)
            .copy()
        )

        depleted = (
            sub[sub["model_direction"] == "Health-enriched"]
            .sort_values(["confidence_score", "prev_disease"], ascending=[False, False])
            .head(top_depleted)
            .copy()
        )

        selected.append(enriched)
        selected.append(depleted)

    if not selected:
        return pd.DataFrame()

    panel = pd.concat(selected, axis=0, ignore_index=True)
    return panel


def build_scores(X: pd.DataFrame, panel_df: pd.DataFrame, use_weights: bool) -> pd.DataFrame:
    all_scores = []

    for disease in sorted(panel_df["disease"].unique()):
        sub = panel_df[panel_df["disease"] == disease].copy()

        enriched_taxa = sub.loc[sub["model_direction"] == "Disease-enriched", "taxon"].tolist()
        depleted_taxa = sub.loc[sub["model_direction"] == "Health-enriched", "taxon"].tolist()

        enriched_taxa = [t for t in enriched_taxa if t in X.columns]
        depleted_taxa = [t for t in depleted_taxa if t in X.columns]

        if use_weights:
            weight_map = sub.set_index("taxon")["confidence_score"]
            enriched_score = compute_weighted_mean(X[enriched_taxa], weight_map) if enriched_taxa else pd.Series(0.0, index=X.index)
            depleted_score = compute_weighted_mean(X[depleted_taxa], weight_map) if depleted_taxa else pd.Series(0.0, index=X.index)
        else:
            enriched_score = X[enriched_taxa].mean(axis=1) if enriched_taxa else pd.Series(0.0, index=X.index)
            depleted_score = X[depleted_taxa].mean(axis=1) if depleted_taxa else pd.Series(0.0, index=X.index)

        raw_score = enriched_score - depleted_score

        score_df = pd.DataFrame({
            "run_id": X.index,
            "disease": disease,
            "raw_score": raw_score.values,
            "enriched_component": enriched_score.values,
            "depleted_component": depleted_score.values,
            "n_enriched_taxa": len(enriched_taxa),
            "n_depleted_taxa": len(depleted_taxa),
        })

        all_scores.append(score_df)

    score_long = pd.concat(all_scores, axis=0, ignore_index=True)

    # z-score normalize within each disease score
    score_long["score"] = (
        score_long.groupby("disease")["raw_score"]
        .transform(lambda s: (s - s.mean()) / (s.std(ddof=0) + 1e-8))
    )

    return score_long


def make_wide_score_table(score_long: pd.DataFrame) -> pd.DataFrame:
    score_wide = score_long.pivot(index="run_id", columns="disease", values="score")
    score_wide = score_wide.reset_index()
    return score_wide


def summarize_index_performance(score_long: pd.DataFrame, y_df: pd.DataFrame) -> pd.DataFrame:
    # Reset index so 'run_id' is just a column and not an index level
    truth = y_df.rename(columns={"phenotype": "true_disease"}).reset_index()
    
    # Ensure the column name matches what score_long uses
    if "run_id" not in truth.columns:
        # If your index was unnamed, it's usually called 'index' after reset_index()
        truth = truth.rename(columns={truth.columns[0]: "run_id"})

    merged = score_long.merge(truth[["run_id", "true_disease"]], on="run_id", how="left")
    # ... rest of the function ...

    rows = []
    for disease in sorted(merged["disease"].unique()):
        sub = merged[merged["disease"] == disease].copy()

        in_class = sub[sub["true_disease"] == disease]["score"]
        out_class = sub[sub["true_disease"] != disease]["score"]

        rows.append({
            "disease": disease,
            "mean_score_in_class": in_class.mean(),
            "mean_score_out_class": out_class.mean(),
            "delta_mean": in_class.mean() - out_class.mean(),
            "median_score_in_class": in_class.median(),
            "median_score_out_class": out_class.median(),
            "n_in_class": len(in_class),
            "n_out_class": len(out_class),
        })

    return pd.DataFrame(rows).sort_values("delta_mean", ascending=False)


def main():
    args = parse_args()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    biomarkers = pd.read_csv(args.biomarkers, sep="\t", low_memory=False)
    X = pd.read_csv(args.X, sep="\t", index_col=0)
    y_df = pd.read_csv(args.y, sep="\t", index_col=0)
    y = y_df["phenotype"]

    panel = select_index_panel(
        biomarkers=biomarkers,
        X=X,
        y=y,
        top_enriched=args.top_enriched,
        top_depleted=args.top_depleted,
        min_prev_disease=args.min_prev_disease,
    )

    if panel.empty:
        raise ValueError("No taxa passed the panel selection criteria.")

    score_long = build_scores(X=X, panel_df=panel, use_weights=args.use_weights)
    score_wide = make_wide_score_table(score_long)
    perf = summarize_index_performance(score_long, y_df)

    panel_path = outdir / "index_taxa_panel_v1.tsv"
    long_path = outdir / "multidisease_index_scores_long_v1.tsv"
    wide_path = outdir / "multidisease_index_scores_wide_v1.tsv"
    perf_path = outdir / "multidisease_index_performance_summary_v1.tsv"

    panel.to_csv(panel_path, sep="\t", index=False)
    score_long.to_csv(long_path, sep="\t", index=False)
    score_wide.to_csv(wide_path, sep="\t", index=False)
    perf.to_csv(perf_path, sep="\t", index=False)

    print(f"Saved: {panel_path}")
    print(f"Saved: {long_path}")
    print(f"Saved: {wide_path}")
    print(f"Saved: {perf_path}")

    print("\nTop disease index separations (v1):")
    print(perf.head(20).to_string(index=False))


if __name__ == "__main__":
    main()
