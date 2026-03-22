#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, label_binarize
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    precision_recall_fscore_support,
    roc_auc_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
    roc_curve,
)
from sklearn.calibration import calibration_curve
from sklearn.utils import resample

import lightgbm as lgb


def parse_args():
    parser = argparse.ArgumentParser(description="Train model v1 with full evaluation")

    parser.add_argument("--X", required=True, help="TSV file of features")
    parser.add_argument("--y", required=True, help="TSV file of labels")
    parser.add_argument("--outdir", default="outputs_model_v1")
    parser.add_argument("--bootstrap", type=int, default=1000, help="Number of bootstrap iterations")

    return parser.parse_args()


def bootstrap_metrics(y_true, y_pred, y_prob, n_classes, n_bootstrap=1000, random_state=42):
    rng = np.random.default_rng(random_state)

    accs = []
    macro_precisions = []
    macro_recalls = []
    macro_f1s = []
    macro_aucs = []

    n = len(y_true)

    for _ in range(n_bootstrap):
        idx = rng.choice(n, size=n, replace=True)

        y_t = y_true[idx]
        y_p = y_pred[idx]
        y_pr = y_prob[idx]

        accs.append(accuracy_score(y_t, y_p))

        p, r, f1, _ = precision_recall_fscore_support(
            y_t, y_p, average="macro", zero_division=0
        )
        macro_precisions.append(p)
        macro_recalls.append(r)
        macro_f1s.append(f1)

        # AUC can fail if a class is absent in a bootstrap sample
        try:
            y_t_bin = label_binarize(y_t, classes=np.arange(n_classes))
            auc = roc_auc_score(
                y_t_bin,
                y_pr,
                multi_class="ovr",
                average="macro",
            )
            macro_aucs.append(auc)
        except Exception:
            pass

    def ci(arr):
        arr = np.array(arr, dtype=float)
        return {
            "mean": float(np.mean(arr)),
            "lower": float(np.percentile(arr, 2.5)),
            "upper": float(np.percentile(arr, 97.5)),
        }

    return {
        "accuracy": ci(accs),
        "macro_precision": ci(macro_precisions),
        "macro_recall": ci(macro_recalls),
        "macro_f1": ci(macro_f1s),
        "macro_roc_auc": ci(macro_aucs) if len(macro_aucs) > 0 else None,
    }


def plot_confusion_matrix(y_true, y_pred, class_names, outpath):
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(10, 8))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    disp.plot(ax=ax, xticks_rotation=90, colorbar=False)
    plt.tight_layout()
    plt.savefig(outpath, dpi=300, bbox_inches="tight")
    plt.close()


def plot_roc_curves(y_true, y_prob, class_names, outpath):
    n_classes = len(class_names)
    y_true_bin = label_binarize(y_true, classes=np.arange(n_classes))

    fig, ax = plt.subplots(figsize=(8, 6))

    per_class_auc = []

    for i in range(n_classes):
        # Skip classes absent from test set
        if y_true_bin[:, i].sum() == 0:
            per_class_auc.append({"class": class_names[i], "roc_auc_ovr": None})
            continue

        fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_prob[:, i])
        auc_i = roc_auc_score(y_true_bin[:, i], y_prob[:, i])
        per_class_auc.append({"class": class_names[i], "roc_auc_ovr": float(auc_i)})

        ax.plot(fpr, tpr, label=f"{class_names[i]} (AUC={auc_i:.3f})")

    ax.plot([0, 1], [0, 1], linestyle="--")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("Per-class ROC curves (one-vs-rest)")
    ax.legend(fontsize=8, loc="lower right")
    plt.tight_layout()
    plt.savefig(outpath, dpi=300, bbox_inches="tight")
    plt.close()

    return per_class_auc


def plot_calibration_curves(y_true, y_prob, class_names, outpath, n_bins=10):
    n_classes = len(class_names)
    y_true_bin = label_binarize(y_true, classes=np.arange(n_classes))

    fig, ax = plt.subplots(figsize=(8, 6))

    for i in range(n_classes):
        # Calibration is only meaningful if both positive and negative samples exist
        positives = y_true_bin[:, i].sum()
        negatives = len(y_true_bin[:, i]) - positives

        if positives == 0 or negatives == 0:
            continue

        prob_true, prob_pred = calibration_curve(
            y_true_bin[:, i],
            y_prob[:, i],
            n_bins=n_bins,
            strategy="uniform",
        )
        ax.plot(prob_pred, prob_true, marker="o", label=class_names[i])

    ax.plot([0, 1], [0, 1], linestyle="--")
    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Observed frequency")
    ax.set_title("Calibration curves (one-vs-rest)")
    ax.legend(fontsize=8, loc="upper left")
    plt.tight_layout()
    plt.savefig(outpath, dpi=300, bbox_inches="tight")
    plt.close()


def main():
    args = parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    print("Loading data...")
    X = pd.read_csv(args.X, sep="\t", index_col=0)
    y = pd.read_csv(args.y, sep="\t", index_col=0)

    y = y["phenotype"]

    print(f"X shape: {X.shape}")
    print(f"y shape: {y.shape}")

    # Encode labels
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    class_names = le.classes_
    n_classes = len(class_names)

    print(f"Classes: {list(class_names)}")

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y_encoded,
        test_size=0.2,
        random_state=42,
        stratify=y_encoded,
    )

    print(f"Train samples: {len(X_train)}")
    print(f"Test samples: {len(X_test)}")

    # Model
    model = lgb.LGBMClassifier(
        objective="multiclass",
        num_class=n_classes,
        n_estimators=200,
        learning_rate=0.05,
        max_depth=-1,
        random_state=42,
    )

    print("Training model...")
    model.fit(X_train, y_train)

    print("Evaluating model...")
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)

    # Basic metrics
    acc = accuracy_score(y_test, y_pred)

    macro_p, macro_r, macro_f1, _ = precision_recall_fscore_support(
        y_test, y_pred, average="macro", zero_division=0
    )
    weighted_p, weighted_r, weighted_f1, _ = precision_recall_fscore_support(
        y_test, y_pred, average="weighted", zero_division=0
    )

    # Multiclass macro AUC
    y_test_bin = label_binarize(y_test, classes=np.arange(n_classes))
    macro_auc = roc_auc_score(
        y_test_bin,
        y_prob,
        multi_class="ovr",
        average="macro",
    )
    weighted_auc = roc_auc_score(
        y_test_bin,
        y_prob,
        multi_class="ovr",
        average="weighted",
    )

    print(f"\nAccuracy: {acc:.4f}")
    print(f"Macro precision: {macro_p:.4f}")
    print(f"Macro recall: {macro_r:.4f}")
    print(f"Macro F1: {macro_f1:.4f}")
    print(f"Weighted precision: {weighted_p:.4f}")
    print(f"Weighted recall: {weighted_r:.4f}")
    print(f"Weighted F1: {weighted_f1:.4f}")
    print(f"Macro ROC-AUC (OVR): {macro_auc:.4f}")
    print(f"Weighted ROC-AUC (OVR): {weighted_auc:.4f}\n")

    print("Classification report:")
    class_report_dict = classification_report(
        y_test,
        y_pred,
        target_names=class_names,
        output_dict=True,
        zero_division=0,
    )
    print(classification_report(
        y_test,
        y_pred,
        target_names=class_names,
        zero_division=0,
    ))

    # Bootstrap CI
    print(f"Running bootstrap ({args.bootstrap} iterations)...")
    ci_results = bootstrap_metrics(
        y_true=y_test,
        y_pred=y_pred,
        y_prob=y_prob,
        n_classes=n_classes,
        n_bootstrap=args.bootstrap,
        random_state=42,
    )

    print("\nBootstrap 95% CI:")
    for metric_name, vals in ci_results.items():
        if vals is None:
            print(f"{metric_name}: NA")
            continue
        print(
            f"{metric_name}: {vals['mean']:.4f} "
            f"[{vals['lower']:.4f}, {vals['upper']:.4f}]"
        )

    # Save metrics summary
    metrics_summary = {
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
        "classes": list(map(str, class_names)),
        "accuracy": float(acc),
        "macro_precision": float(macro_p),
        "macro_recall": float(macro_r),
        "macro_f1": float(macro_f1),
        "weighted_precision": float(weighted_p),
        "weighted_recall": float(weighted_r),
        "weighted_f1": float(weighted_f1),
        "macro_roc_auc_ovr": float(macro_auc),
        "weighted_roc_auc_ovr": float(weighted_auc),
        "bootstrap_95ci": ci_results,
    }

    with open(outdir / "metrics_summary.json", "w") as f:
        json.dump(metrics_summary, f, indent=2)

    # Save classification report
    report_df = pd.DataFrame(class_report_dict).transpose()
    report_df.to_csv(outdir / "classification_report.tsv", sep="\t")

    # Save test predictions
    pred_df = pd.DataFrame(
        {
            "true_label_encoded": y_test,
            "pred_label_encoded": y_pred,
            "true_label": le.inverse_transform(y_test),
            "pred_label": le.inverse_transform(y_pred),
        },
        index=X_test.index,
    )

    prob_df = pd.DataFrame(
        y_prob,
        index=X_test.index,
        columns=[f"prob_{c}" for c in class_names],
    )

    pred_full_df = pd.concat([pred_df, prob_df], axis=1)
    pred_full_df.to_csv(outdir / "test_predictions.tsv", sep="\t")

    # Plots
    print("Saving plots...")
    plot_confusion_matrix(
        y_true=y_test,
        y_pred=y_pred,
        class_names=class_names,
        outpath=outdir / "confusion_matrix.png",
    )

    per_class_auc = plot_roc_curves(
        y_true=y_test,
        y_prob=y_prob,
        class_names=class_names,
        outpath=outdir / "per_class_roc_curves.png",
    )
    pd.DataFrame(per_class_auc).to_csv(outdir / "per_class_auc.tsv", sep="\t", index=False)

    plot_calibration_curves(
        y_true=y_test,
        y_prob=y_prob,
        class_names=class_names,
        outpath=outdir / "calibration_curves.png",
        n_bins=10,
    )

    # Save model
    model_path = outdir / "model_v1.pkl"
    encoder_path = outdir / "label_encoder.pkl"

    joblib.dump(model, model_path)
    joblib.dump(le, encoder_path)

    print(f"Saved: {model_path}")
    print(f"Saved: {encoder_path}")
    print(f"Saved: {outdir / 'metrics_summary.json'}")
    print(f"Saved: {outdir / 'classification_report.tsv'}")
    print(f"Saved: {outdir / 'test_predictions.tsv'}")
    print(f"Saved: {outdir / 'per_class_auc.tsv'}")
    print(f"Saved: {outdir / 'confusion_matrix.png'}")
    print(f"Saved: {outdir / 'per_class_roc_curves.png'}")
    print(f"Saved: {outdir / 'calibration_curves.png'}")


if __name__ == "__main__":
    main()
