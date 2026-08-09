"""
train_role_classifier.py
--------------------------
Module 5 — formal training procedure for RoleClassifier.

Steps:
  1. Train/test split (80/20, stratified — required because Category is
     imbalanced; a random split could leave a rare class entirely out of
     the test set).
  2. Hyperparameter tuning of the LinearSVC regularization strength C via
     GridSearchCV, scored on f1_macro (not accuracy — accuracy would be
     dominated by the large "Technology" class, see Module 1).
  3. 5-fold cross-validation on the training split to pick the best C.
  4. Refit best model on the FULL training split, evaluate once on the
     held-out test split (Module 6 does the detailed evaluation report).
  5. Save the tuned vectorizer + classifier to saved_models/.
"""

import sys
import time
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.text import TfidfVectorizer
import joblib

DATA = Path(__file__).resolve().parent.parent / "data" / "processed" / "resumes_clean.parquet"
OUT = Path(__file__).resolve().parent.parent / "saved_models"
OUT.mkdir(parents=True, exist_ok=True)


def main():
    df = pd.read_parquet(DATA)
    X_text, y = df["resume_text_clean"], df["category"]

    # Stratified split so every one of the 42 categories keeps its
    # proportional share in both train and test (critical given the
    # class imbalance documented in Module 1: some categories have <70 rows).
    X_train_text, X_test_text, y_train, y_test = train_test_split(
        X_text, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"Train: {len(X_train_text)} rows | Test: {len(X_test_text)} rows")

    vectorizer = TfidfVectorizer(max_features=10000, ngram_range=(1, 2), min_df=2)
    X_train = vectorizer.fit_transform(X_train_text)
    X_test = vectorizer.transform(X_test_text)

    # Hyperparameter search over C (regularization strength). A small
    # grid is enough here: LinearSVC on this data is already near-ceiling
    # (Module 4 CV showed f1_macro=0.996 at default C=1), so the search
    # mainly confirms we're not over/under-regularizing.
    param_grid = {"C": [0.1, 0.5, 1.0, 2.0, 5.0]}
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    t0 = time.time()
    search = GridSearchCV(
        LinearSVC(class_weight="balanced"),
        param_grid,
        scoring="f1_macro",
        cv=cv,
        n_jobs=-1,
    )
    search.fit(X_train, y_train)
    print(f"Grid search done in {time.time()-t0:.1f}s")
    print("Best C:", search.best_params_["C"], "| CV f1_macro:", round(search.best_score_, 4))
    for c, mean in zip(param_grid["C"], search.cv_results_["mean_test_score"]):
        print(f"  C={c:<5} f1_macro={mean:.4f}")

    # Refit with calibration (for predict_proba) using the best C, on the
    # full training split.
    best_c = search.best_params_["C"]
    calibrated = CalibratedClassifierCV(LinearSVC(C=best_c, class_weight="balanced"), cv=5)
    calibrated.fit(X_train, y_train)

    test_f1 = None
    from sklearn.metrics import f1_score
    test_preds = calibrated.predict(X_test)
    test_f1 = f1_score(y_test, test_preds, average="macro")
    print(f"Held-out test f1_macro: {test_f1:.4f}")

    joblib.dump({"vectorizer": vectorizer, "classifier": calibrated}, OUT / "role_classifier_tuned.joblib")
    # Persist the split so evaluation.py (Module 6) scores on the EXACT
    # same held-out rows rather than re-splitting (which could leak).
    X_test_text.to_frame(name="resume_text_clean").assign(category=y_test.values).to_parquet(
        Path(__file__).resolve().parent.parent / "data" / "processed" / "role_classifier_test_split.parquet",
        index=False,
    )
    print("Saved tuned model + held-out test split.")


if __name__ == "__main__":
    main()
