"""Compare traditional fake-review classifiers on the labeled demo dataset."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

try:
    from models.train_model import DATA_PATH, load_dataset
except ModuleNotFoundError:  # Support the documented direct script command.
    from train_model import DATA_PATH, load_dataset


OUTPUT_PATH = Path("data/model_comparison.csv")
METRIC_COLUMNS = ["model", "accuracy", "precision", "recall", "f1_score", "roc_auc"]


def build_models() -> dict[str, Pipeline]:
    """Return the three traditional text-classification pipelines."""
    return {
        "Logistic Regression": Pipeline(
            [("tfidf", TfidfVectorizer()), ("classifier", LogisticRegression(max_iter=1000, random_state=42))]
        ),
        "Multinomial Naive Bayes": Pipeline(
            [("tfidf", TfidfVectorizer()), ("classifier", MultinomialNB())]
        ),
        "Random Forest": Pipeline(
            [
                ("tfidf", TfidfVectorizer()),
                ("classifier", RandomForestClassifier(n_estimators=100, random_state=42)),
            ]
        ),
    }


def compare_models(
    data_path: str | Path = DATA_PATH,
    output_path: str | Path = OUTPUT_PATH,
) -> tuple[pd.DataFrame, dict[str, list[list[int]]], str]:
    """Train, evaluate, rank, and save all comparison models."""
    texts, labels = load_dataset(data_path)
    x_train, x_test, y_train, y_test = train_test_split(
        texts, labels, test_size=0.20, random_state=42, stratify=labels
    )

    rows: list[dict[str, Any]] = []
    confusion_matrices: dict[str, list[list[int]]] = {}
    for model_name, model in build_models().items():
        model.fit(x_train, y_train)
        predictions = model.predict(x_test)
        probabilities = model.predict_proba(x_test)[:, list(model.classes_).index(1)]
        rows.append(
            {
                "model": model_name,
                "accuracy": float(accuracy_score(y_test, predictions)),
                "precision": float(precision_score(y_test, predictions, zero_division=0)),
                "recall": float(recall_score(y_test, predictions, zero_division=0)),
                "f1_score": float(f1_score(y_test, predictions, zero_division=0)),
                "roc_auc": float(roc_auc_score(y_test, probabilities)),
            }
        )
        confusion_matrices[model_name] = confusion_matrix(y_test, predictions, labels=[0, 1]).tolist()

    results = pd.DataFrame(rows, columns=METRIC_COLUMNS)
    results = results.sort_values(["f1_score", "roc_auc"], ascending=False, kind="stable").reset_index(drop=True)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(output_path, index=False)
    best_model = str(results.iloc[0]["model"])
    return results, confusion_matrices, best_model


def main() -> None:
    results, confusion_matrices, best_model = compare_models()
    print("Model comparison on the 24-row demo dataset")
    print(results.to_string(index=False, float_format=lambda value: f"{value:.4f}"))
    for model_name, matrix in confusion_matrices.items():
        print(f"\nConfusion matrix - {model_name}")
        print(matrix)
    print(f"\nBest model by F1-score, then ROC-AUC: {best_model}")
    print(f"Saved comparison results: {OUTPUT_PATH}")
    print("Note: these results validate the pipeline only and cannot establish real-world model superiority.")


if __name__ == "__main__":
    main()