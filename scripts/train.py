"""Train and compare fraud-detection models on data/creditcard.csv.

Usage: python scripts/train.py [--data data/creditcard.csv] [--out models]
"""
import argparse
from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, classification_report,
                             roc_auc_score)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parent.parent


def get_models():
    return {
        "logreg": make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=1000, class_weight="balanced"),
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=200, class_weight="balanced_subsample",
            n_jobs=-1, random_state=42,
        ),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default=str(ROOT / "data" / "creditcard.csv"))
    parser.add_argument("--out", default=str(ROOT / "models"))
    args = parser.parse_args()

    if not Path(args.data).exists():
        raise SystemExit(f"Data file not found: {args.data}")
    df = pd.read_csv(args.data)
    df["small_spend"] = (df["Amount"] <= 10).astype(int)
    df["large_spend"] = (df["Amount"] >= 100).astype(int)
    df["hour"] = ((df["Time"] // 3600) % 24).astype(int)
    X, y = df.drop(columns="Class"), df["Class"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    out = Path(args.out)
    out.mkdir(exist_ok=True)

    for name, model in get_models().items():
        print(f"\n=== {name} ===")
        model.fit(X_train, y_train)
        proba = model.predict_proba(X_test)[:, 1]
        print(classification_report(y_test, proba > 0.5, digits=4))
        print("ROC AUC:", round(roc_auc_score(y_test, proba), 4))
        print("PR AUC :", round(average_precision_score(y_test, proba), 4))
        joblib.dump(model, out / f"{name}.joblib")


if __name__ == "__main__":
    main()
