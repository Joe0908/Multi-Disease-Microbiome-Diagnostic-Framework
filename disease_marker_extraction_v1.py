#!/usr/bin/env python3

import argparse
import pandas as pd
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(description="Disease-specific marker extraction")

    parser.add_argument("--X", required=True)
    parser.add_argument("--y", required=True)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--out", default="disease_markers_v1.tsv")

    return parser.parse_args()


def main():
    args = parse_args()

    X = pd.read_csv(args.X, sep="\t", index_col=0)
    y = pd.read_csv(args.y, sep="\t", index_col=0)

    y = y["phenotype"]

    diseases = sorted(y.unique())

    results = []

    for disease in diseases:
        print(f"Processing: {disease}")

        mask_d = y == disease
        mask_other = y != disease

        X_d = X[mask_d]
        X_o = X[mask_other]

        # ===== mean abundance =====
        mean_d = X_d.mean(axis=0)
        mean_o = X_o.mean(axis=0)

        # ===== log fold change =====
        logfc = np.log2((mean_d + 1e-6) / (mean_o + 1e-6))

        df = pd.DataFrame({
            "taxon": X.columns,
            "logFC": logfc,
            "mean_disease": mean_d,
            "mean_other": mean_o,
        })

        df["abs_logFC"] = df["logFC"].abs()

        df = df.sort_values("abs_logFC", ascending=False)

        top_df = df.head(args.top_k).copy()
        top_df["disease"] = disease

        results.append(top_df)

    final_df = pd.concat(results)

    final_df.to_csv(args.out, sep="\t", index=False)

    print("\nTop markers saved.")
    print(final_df.head(30))


if __name__ == "__main__":
    main()