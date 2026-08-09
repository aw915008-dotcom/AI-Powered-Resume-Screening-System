"""
evaluate_role_classifier.py
-----------------------------
Module 6 — evaluation of the tuned RoleClassifier on the held-out test
split saved by training/train_role_classifier.py (never re-splitting,
to avoid any train/test leakage).

Metrics: accuracy, precision/recall/f1 (macro AND weighted — macro
treats every one of the 42 classes equally so it isn't dominated by the
large "Technology" class; weighted reflects real-world class frequency),
and a confusion matrix for the top-10 most frequent categories (the full
42x42 matrix is too dense to read).
"""

import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import joblib
import pandas as pd
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support,
    classification_report, confusion_matrix,
)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = ROOT / "saved_models" / "role_classifier_tuned.joblib"
TEST_PATH = ROOT / "data" / "processed" / "role_classifier_test_split.parquet"
OUT_DIR = ROOT / "evaluation" / "reports"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def main():
    bundle = joblib.load(MODEL_PATH)
    vectorizer, classifier = bundle["vectorizer"], bundle["classifier"]

    test_df = pd.read_parquet(TEST_PATH)
    X_test = vectorizer.transform(test_df["resume_text_clean"])
    y_test = test_df["category"]
    y_pred = classifier.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    p_macro, r_macro, f1_macro, _ = precision_recall_fscore_support(y_test, y_pred, average="macro", zero_division=0)
    p_weighted, r_weighted, f1_weighted, _ = precision_recall_fscore_support(y_test, y_pred, average="weighted", zero_division=0)

    print(f"Accuracy:            {acc:.4f}")
    print(f"Precision (macro):   {p_macro:.4f}   | (weighted): {p_weighted:.4f}")
    print(f"Recall    (macro):   {r_macro:.4f}   | (weighted): {r_weighted:.4f}")
    print(f"F1-score  (macro):   {f1_macro:.4f}   | (weighted): {f1_weighted:.4f}")

    report = classification_report(y_test, y_pred, zero_division=0)
    (OUT_DIR / "classification_report.txt").write_text(report)
    print("\nFull classification report saved to evaluation/reports/classification_report.txt")

    # Confusion matrix for the 10 most frequent categories only (readable size)
    top_classes = y_test.value_counts().head(10).index.tolist()
    mask = y_test.isin(top_classes)
    cm = confusion_matrix(y_test[mask], pd.Series(y_pred, index=y_test.index)[mask], labels=top_classes)

    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(top_classes)))
    ax.set_yticks(range(len(top_classes)))
    ax.set_xticklabels(top_classes, rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(top_classes, fontsize=8)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title("Confusion matrix — top 10 categories by frequency")
    for i in range(len(top_classes)):
        for j in range(len(top_classes)):
            ax.text(j, i, cm[i, j], ha="center", va="center",
                     color="white" if cm[i, j] > cm.max() / 2 else "black", fontsize=7)
    fig.colorbar(im)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "confusion_matrix_top10.png", dpi=150)
    print("Confusion matrix chart saved to evaluation/reports/confusion_matrix_top10.png")

    return {"accuracy": acc, "f1_macro": f1_macro, "f1_weighted": f1_weighted}


if __name__ == "__main__":
    main()
