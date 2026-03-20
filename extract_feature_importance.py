#!/usr/bin/env python3

import argparse
import pandas as pd
import joblib


def parse_args():
    parser = argparse.ArgumentParser(description="Extract feature importance")

    parser.add_argument("--model", required=True)
    parser.add_argument("--X", required=True)
    parser.add_argument("--out", default="feature_importance.tsv")

    return parser.parse_args()


def main():
    args = parse_args()

    model = joblib.load(args.model)
    X = pd.read_csv(args.X, sep="\t", index_col=0)

    feature_names = X.columns
    importances = model.feature_importances_

    df = pd.DataFrame({
        "taxon": feature_names,
        "importance": importances
    })

    df = df.sort_values("importance", ascending=False)

    df.to_csv(args.out, sep="\t", index=False)

    print(df.head(30))


if __name__ == "__main__":
    main()