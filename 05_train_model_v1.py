#!/usr/bin/env python3

from __future__ import annotations

import argparse
from pathlib import Path
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report

import lightgbm as lgb


def parse_args():
    parser = argparse.ArgumentParser(description="Train model v1")

    parser.add_argument("--X", required=True)
    parser.add_argument("--y", required=True)
    parser.add_argument("--outdir", default="outputs_model_v1")

    return parser.parse_args()


def main():
    args = parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    print("Loading data...")

    X = pd.read_csv(args.X, sep="\t", index_col=0)
    y = pd.read_csv(args.y, sep="\t", index_col=0)

    y = y["phenotype"]

    # ===== encode labels =====
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)

    print(f"Classes: {list(le.classes_)}")

    # ===== split =====
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
    )

    print(f"Train samples: {len(X_train)}")
    print(f"Test samples: {len(X_test)}")

    # ===== model =====
    model = lgb.LGBMClassifier(
        objective="multiclass",
        num_class=len(le.classes_),
        n_estimators=200,
        learning_rate=0.05,
        max_depth=-1,
        random_state=42,
    )

    print("Training model...")
    model.fit(X_train, y_train)

    # ===== evaluation =====
    print("Evaluating...")

    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)

    print(f"\nAccuracy: {acc:.4f}\n")

    print("Classification report:")
    print(classification_report(y_test, y_pred, target_names=le.classes_))

    # ===== save model =====
    import joblib

    model_path = outdir / "model_v1.pkl"
    encoder_path = outdir / "label_encoder.pkl"

    joblib.dump(model, model_path)
    joblib.dump(le, encoder_path)

    print(f"Saved: {model_path}")
    print(f"Saved: {encoder_path}")


if __name__ == "__main__":
    main()
