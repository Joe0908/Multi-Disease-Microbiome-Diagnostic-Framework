#!/usr/bin/env python3

from __future__ import annotations

import argparse
from pathlib import Path
import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser(description="Build modeling dataset v1")

    parser.add_argument("--run-level", required=True)
    parser.add_argument("--keep-list", required=True)
    parser.add_argument("--outdir", default="outputs_modeling_v1")

    parser.add_argument("--rank", default="genus")
    parser.add_argument("--abundance-threshold", type=float, default=0.0)

    return parser.parse_args()


def load_keep_list(path: str) -> set[str]:
    with open(path, "r") as f:
        return set([line.strip() for line in f if line.strip()])


def main():
    args = parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    print("Loading data...")

    df = pd.read_csv(args.run_level, sep="\t", low_memory=False)

    # ===== basic cleaning =====
    df["final_rank"] = df["final_rank"].astype(str).str.lower()
    df = df[df["final_rank"] == args.rank]

    # ===== load keep phenotypes =====
    keep_set = load_keep_list(args.keep_list)

    df["phenotype"] = df["phenotype"].astype(str).str.strip()

    df = df[df["phenotype"].isin(keep_set)]

    print(f"Remaining rows after phenotype filter: {len(df)}")

    # ===== optional abundance filter =====
    if args.abundance_threshold > 0:
        df = df[df["relative_abundance"] >= args.abundance_threshold]

    # ===== build feature matrix =====
    print("Building feature matrix...")

    X = (
        df.pivot_table(
            index="run_id",
            columns="scientific_name",
            values="relative_abundance",
            aggfunc="mean",
            fill_value=0,
        )
    )

    # ===== build labels =====
    print("Building labels...")

    y = (
        df[["run_id", "phenotype"]]
        .drop_duplicates()
        .set_index("run_id")
        .loc[X.index]
    )

    # ===== sanity check =====
    assert len(X) == len(y)

    print(f"Samples: {len(X)}")
    print(f"Features (taxa): {X.shape[1]}")

    # ===== save =====
    X_path = outdir / "X_v1.tsv"
    y_path = outdir / "y_v1.tsv"

    X.to_csv(X_path, sep="\t")
    y.to_csv(y_path, sep="\t")

    print(f"Saved: {X_path}")
    print(f"Saved: {y_path}")


if __name__ == "__main__":
    main()