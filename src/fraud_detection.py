"""Credit card fraud detection on the Kaggle mlg-ulb/creditcardfraud dataset.

Download creditcard.csv from
https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud/data into data/.
"""
import argparse

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.model_selection import train_test_split

TARGET = "Class"


def load_data(path):
    df = pd.read_csv(path)
    if TARGET not in df.columns:
        raise ValueError(f"Expected a '{TARGET}' column in {path}")
    return df


def summarize(df):
    """Return basic counts that show how imbalanced the data is."""
    frauds = int((df[TARGET] == 1).sum())
    return {"transactions": len(df), "frauds": frauds, "fraud_rate": frauds / len(df)}


def train_and_evaluate(df, seed=42):
    X, y = df.drop(columns=[TARGET]), df[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, stratify=y, random_state=seed
    )
    model = RandomForestClassifier(
        n_estimators=100, class_weight="balanced", n_jobs=-1, random_state=seed
    )
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    proba = model.predict_proba(X_test)[:, 1]
    return model, classification_report(y_test, pred), roc_auc_score(y_test, proba)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default="data/creditcard.csv")
    args = parser.parse_args()
    df = load_data(args.data)
    print(summarize(df))
    _, report, auc = train_and_evaluate(df)
    print(report)
    print(f"ROC AUC: {auc:.4f}")


if __name__ == "__main__":
    main()
