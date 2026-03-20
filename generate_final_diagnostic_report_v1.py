#!/usr/bin/env python3

from __future__ import annotations

import argparse
from pathlib import Path
import pandas as pd
import numpy as np
import joblib


def parse_args():
    parser = argparse.ArgumentParser(description="Generate final diagnostic report")

    parser.add_argument("--model", required=True, help="Path to trained LightGBM model .pkl")
    parser.add_argument("--label-encoder", required=True, help="Path to label encoder .pkl")
    parser.add_argument("--X", required=True, help="Path to X_filtered_v1.tsv")
    parser.add_argument("--index-scores-wide", required=True, help="Path to multidisease_index_scores_wide_v1.tsv")
    parser.add_argument("--index-panel", required=True, help="Path to index_taxa_panel_v1.tsv")
    parser.add_argument("--outdir", default="outputs_final_diagnostic_report_v1")

    parser.add_argument("--top-n-supporting-taxa", type=int, default=5)

    return parser.parse_args()


def get_top_supporting_taxa_for_sample(
    sample_vector: pd.Series,
    predicted_disease: str,
    index_panel: pd.DataFrame,
    top_n: int = 5,
) -> str:
    sub = index_panel[index_panel["disease"] == predicted_disease].copy()

    if sub.empty:
        return ""

    taxa_in_sample = [t for t in sub["taxon"].tolist() if t in sample_vector.index]
    if not taxa_in_sample:
        return ""

    sub = sub[sub["taxon"].isin(taxa_in_sample)].copy()
    sub["sample_value"] = sub["taxon"].map(sample_vector.to_dict())

    # For disease-enriched taxa, higher abundance supports disease
    # For health-enriched taxa, lower abundance supports disease
    def compute_support(row):
        value = float(row["sample_value"])
        weight = float(row.get("confidence_score", 1.0))
        if row["model_direction"] == "Disease-enriched":
            return value * weight
        else:
            return (-value) * weight

    sub["support_score"] = sub.apply(compute_support, axis=1)
    sub = sub.sort_values("support_score", ascending=False).head(top_n)

    formatted = []
    for _, row in sub.iterrows():
        arrow = "↑" if row["model_direction"] == "Disease-enriched" else "↓"
        formatted.append(f"{row['taxon']} {arrow}")

    return " | ".join(formatted)


def main():
    args = parse_args()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    print("Loading model and data...")
    model = joblib.load(args.model)
    label_encoder = joblib.load(args.label_encoder)

    X = pd.read_csv(args.X, sep="\t", index_col=0)
    index_scores = pd.read_csv(args.index_scores_wide, sep="\t")
    index_panel = pd.read_csv(args.index_panel, sep="\t")

    # Align index scores
    index_scores = index_scores.set_index("run_id")
    index_scores = index_scores.loc[X.index].copy()

    print("Predicting model probabilities...")
    probs = model.predict_proba(X)
    pred_idx = model.predict(X)
    pred_labels = label_encoder.inverse_transform(pred_idx)

    top_model_prob = probs.max(axis=1)

    print("Extracting top index scores...")
    top_index_disease = index_scores.idxmax(axis=1)
    top_index_score = index_scores.max(axis=1)

    print("Building final summary...")
    summary_df = pd.DataFrame({
        "run_id": X.index,
        "predicted_disease": pred_labels,
        "top_model_probability": top_model_prob,
        "top_index_disease": top_index_disease.values,
        "top_index_score": top_index_score.values,
    })

    summary_df["model_index_agree"] = summary_df["predicted_disease"] == summary_df["top_index_disease"]

    print("Generating supporting taxa...")
    supporting_taxa = []
    for run_id, pred_disease in zip(summary_df["run_id"], summary_df["predicted_disease"]):
        sample_vector = X.loc[run_id]
        taxa_text = get_top_supporting_taxa_for_sample(
            sample_vector=sample_vector,
            predicted_disease=pred_disease,
            index_panel=index_panel,
            top_n=args.top_n_supporting_taxa,
        )
        supporting_taxa.append(taxa_text)

    summary_df["top_supporting_taxa"] = supporting_taxa

    # Build long-form disease probability table
    prob_df = pd.DataFrame(
        probs,
        index=X.index,
        columns=label_encoder.classes_,
    ).reset_index().rename(columns={"index": "run_id"})

    prob_long = prob_df.melt(
        id_vars="run_id",
        var_name="disease",
        value_name="model_probability"
    )

    index_long = index_scores.reset_index().melt(
        id_vars="run_id",
        var_name="disease",
        value_name="index_score"
    )

    detailed_long = prob_long.merge(index_long, on=["run_id", "disease"], how="left")

    summary_path = outdir / "final_diagnostic_summary_v1.tsv"
    detailed_path = outdir / "final_diagnostic_detailed_scores_v1.tsv"

    summary_df.to_csv(summary_path, sep="\t", index=False)
    detailed_long.to_csv(detailed_path, sep="\t", index=False)

    print(f"Saved: {summary_path}")
    print(f"Saved: {detailed_path}")

    print("\nPreview of final summary:")
    print(summary_df.head(10).to_string(index=False))

    print("\nAgreement rate:")
    print(summary_df["model_index_agree"].mean())


if __name__ == "__main__":
    main()