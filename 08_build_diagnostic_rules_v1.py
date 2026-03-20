#!/usr/bin/env python3

import argparse
import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser(description="Build diagnostic rules")

    parser.add_argument("--markers", required=True)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--out", default="diagnostic_rules_v1.tsv")

    return parser.parse_args()


def main():
    args = parse_args()

    df = pd.read_csv(args.markers, sep="\t")

    results = []

    for disease in sorted(df["disease"].unique()):
        sub = df[df["disease"] == disease].copy()

        sub = sub.sort_values("abs_logFC", ascending=False).head(args.top_k)

        for _, row in sub.iterrows():
            direction = "enriched" if row["logFC"] > 0 else "depleted"

            results.append({
                "disease": disease,
                "taxon": row["taxon"],
                "direction": direction,
                "logFC": row["logFC"]
            })

    final_df = pd.DataFrame(results)

    final_df.to_csv(args.out, sep="\t", index=False)

    print(final_df.head(30))


if __name__ == "__main__":
    main()
