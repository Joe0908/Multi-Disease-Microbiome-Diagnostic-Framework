#!/usr/bin/env python3

from __future__ import annotations

import argparse
from pathlib import Path
import pandas as pd
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(description="Feature engineering v1")

    parser.add_argument("--X", required=True)
    parser.add_argument("--y", required=True)
    parser.add_argument("--outdir", default="outputs_feature_v1")

    parser.add_argument("--min-prevalence", type=float, default=0.05)
    parser.add_argument("--log-transform", action="store_true")

    return parser.parse_args()


def main():
    args = parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    print("Loading data...")

    X = pd.read_csv(args.X, sep="\t", index_col=0)
    y = pd.read_csv(args.y, sep="\t", index_col=0)

    print(f"Original features: {X.shape[1]}")

    # ===== prevalence filtering =====
    print("Applying prevalence filter...")

    prevalence = (X > 0).sum(axis=0) / len(X)

    keep_features = prevalence[prevalence >= args.min_prevalence].index

    X = X[keep_features]

    print(f"Features after prevalence filtering: {X.shape[1]}")

    # ===== log transform =====
    if args.log_transform:
        print("Applying log transform...")
        X = np.log1p(X)

    # ===== save =====
    X_path = outdir / "X_filtered_v1.tsv"
    y_path = outdir / "y_v1.tsv"

    X.to_csv(X_path, sep="\t")
    y.to_csv(y_path, sep="\t")

    print(f"Saved: {X_path}")
    print(f"Saved: {y_path}")


if __name__ == "__main__":
    main()