"""Train the baseline TF-IDF and Logistic Regression fake-review model."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline


DATA_PATH = Path("data/labeled_reviews.csv")
MODEL_PATH = Path("models/fake_review_model.pkl")
REQUIRED_COLUMNS = {"processed_text", "label"}


class ModelDataError(ValueError):
    """Raised when the labeled dataset cannot support baseline training."""


def load_dataset(data_path: str | Path = DATA_PATH) -> tuple[pd.Series, pd.Series]:
    """Load and validate the processed text and binary labels."""
    data_path = Path(data_path)
    if not data_path.is_file():
        raise FileNotFoundError(f"Dataset not found: {data_path}")

    data = pd.read_csv(data_path)
    missing_columns = REQUIRED_COLUMNS - set(data.columns)
    if missing_columns:
        raise ModelDataError("Dataset is missing column(s): " + ", ".join(sorted(missing_columns)))

    data = data[["processed_text", "label"]].copy()
    data["processed_text"] = data["processed_text"].fillna("").astype(str).str.strip()
    data["label"] = pd.to_numeric(data["label"], errors="coerce")
    data = data[(data["processed_text"] != "") & data["label"].notna()]
    if data.empty:
        raise ModelDataError("Dataset contains no usable processed text and labels.")
    if not set(data["label"].unique()).issubset({0, 1}):
        raise ModelDataError("Labels must use only 0 for genuine and 1 for fake.")
    if data["label"].nunique() != 2:
        raise ModelDataError("Dataset must contain both label classes 0 and 1.")
    return data["processed_text"], data["label"].astype(int)


def build_pipeline() -> Pipeline:
    """Create the baseline TF-IDF plus Logistic Regression pipeline."""
    return Pipeline(
        [
            ("tfidf", TfidfVectorizer()),
            ("classifier", LogisticRegression(max_iter=1000, random_state=42)),
        ]
    )


def train_model(
    data_path: str | Path = DATA_PATH,
    model_path: str | Path = MODEL_PATH,
) -> dict[str, Any]:
    """Train, evaluate, save, and return baseline model metrics."""
    texts, labels = load_dataset(data_path)
    x_train, x_test, y_train, y_test = train_test_split(
        texts, labels, test_size=0.2, random_state=42, stratify=labels
    )

    model = build_pipeline()
    model.fit(x_train, y_train)
    predictions = model.predict(x_test)
    probabilities = model.predict_proba(x_test)[:, list(model.classes_).index(1)]
    metrics: dict[str, Any] = {
        "dataset_size": int(len(labels)),
        "training_size": int(len(y_train)),
        "testing_size": int(len(y_test)),
        "class_distribution": {str(label): int(count) for label, count in labels.value_counts().sort_index().items()},
        "accuracy": float(accuracy_score(y_test, predictions)),
        "precision": float(precision_score(y_test, predictions, zero_division=0)),
        "recall": float(recall_score(y_test, predictions, zero_division=0)),
        "f1_score": float(f1_score(y_test, predictions, zero_division=0)),
        "confusion_matrix": confusion_matrix(y_test, predictions, labels=[0, 1]).tolist(),
        "roc_auc": float(roc_auc_score(y_test, probabilities)),
    }

    model_path = Path(model_path)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, model_path)
    return metrics


def main() -> None:
    metrics = train_model()
    print("Baseline fake-review model training")
    print("Dataset size:", metrics["dataset_size"])
    print("Training size:", metrics["training_size"])
    print("Testing size:", metrics["testing_size"])
    print("Class distribution:", metrics["class_distribution"])
    print("Accuracy:", f"{metrics['accuracy']:.4f}")
    print("Precision:", f"{metrics['precision']:.4f}")
    print("Recall:", f"{metrics['recall']:.4f}")
    print("F1-score:", f"{metrics['f1_score']:.4f}")
    print("Confusion matrix:")
    print(json.dumps(metrics["confusion_matrix"]))
    print("ROC-AUC:", f"{metrics['roc_auc']:.4f}")
    print("Saved model:", MODEL_PATH)
    print("Note: these metrics are for the 24-row demo dataset and are not representative of real-world performance.")


if __name__ == "__main__":
    main()