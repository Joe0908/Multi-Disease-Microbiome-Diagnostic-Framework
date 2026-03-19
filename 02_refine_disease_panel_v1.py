#!/usr/bin/env python3

from __future__ import annotations

import argparse
from pathlib import Path
import pandas as pd


KEEP = {
    "Health",
    "Colorectal Neoplasms",
    "COVID-19",
    "Crohn Disease",
    "Diarrhea",
    "Parkinson Disease",
    "Colitis, Ulcerative",
    "Diabetes Mellitus, Type 2",
    "Non-alcoholic Fatty Liver Disease",
    "Irritable Bowel Syndrome",
    "HIV Infections",
    "Multiple Sclerosis",
    "Alzheimer Disease",
}

REVIEW = {
    "Renal Insufficiency, Chronic",
    "Kidney Failure, Chronic",
    "Inflammatory Bowel Diseases",
    "Diabetes Mellitus, Type 1",
    "Diabetes, Gestational",
    "Breast Neoplasms",
    "Cystic Fibrosis",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Refine disease panel v1")
    parser.add_argument(
        "--input",
        required=True,
        help="Path to disease_panel_v1.tsv",
    )
    parser.add_argument(
        "--outdir",
        default="outputs_disease_panel_refined",
        help="Output directory",
    )
    return parser.parse_args()


def assign_panel_decision(phenotype: str) -> tuple[str, str]:
    if phenotype in KEEP:
        return "keep", ""
    if phenotype in REVIEW:
        return "review", "needs_manual_review"
    return "exclude", "not_selected_for_v1"


def main() -> None:
    args = parse_args()

    df = pd.read_csv(args.input, sep="\t", low_memory=False)

    if "phenotype_canonical" not in df.columns:
        raise ValueError("Input file must contain 'phenotype_canonical' column")

    decisions = df["phenotype_canonical"].apply(assign_panel_decision)
    df["panel_decision"] = decisions.apply(lambda x: x[0])
    df["panel_note"] = decisions.apply(lambda x: x[1])

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    final_panel_path = outdir / "final_disease_panel_v1.tsv"
    keep_list_path = outdir / "keep_phenotypes_v1.txt"
    review_list_path = outdir / "review_phenotypes_v1.txt"

    df = df.sort_values(
        by=["panel_decision", "included", "n_runs"],
        ascending=[True, False, False],
    )

    df.to_csv(final_panel_path, sep="\t", index=False)

    keep_df = df.loc[df["panel_decision"] == "keep", "phenotype_canonical"].drop_duplicates()
    review_df = df.loc[df["panel_decision"] == "review", "phenotype_canonical"].drop_duplicates()

    with open(keep_list_path, "w", encoding="utf-8") as f:
        for item in keep_df:
            f.write(f"{item}\n")

    with open(review_list_path, "w", encoding="utf-8") as f:
        for item in review_df:
            f.write(f"{item}\n")

    print(f"Saved: {final_panel_path}")
    print(f"Saved: {keep_list_path}")
    print(f"Saved: {review_list_path}")

    print("\nKeep phenotypes:")
    print(keep_df.to_string(index=False))

    print("\nReview phenotypes:")
    print(review_df.to_string(index=False))


if __name__ == "__main__":
    main()
