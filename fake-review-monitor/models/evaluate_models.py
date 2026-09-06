"""Evaluate traditional fake-review models on the prepared real dataset."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline


DATA_PATH = Path("data/real_labeled_reviews.csv")
RESULTS_PATH = Path("data/phase6_model_results.csv")
FINAL_RESULTS_PATH = Path("data/final_test_results.csv")
FINAL_MODEL_PATH = Path("models/final_fake_review_model.pkl")
FINAL_MODEL_NAME_PATH = Path("models/final_model_name.txt")
REQUIRED_COLUMNS = {"processed_text", "label"}
RESULT_COLUMNS = [
    "model", "validation_accuracy", "validation_precision", "validation_recall",
    "validation_f1", "validation_roc_auc", "cv_mean_f1", "cv_std_f1",
]


def build_models() -> dict[str, Pipeline]:
    return {
        "Logistic Regression": Pipeline([
            ("tfidf", TfidfVectorizer()),
            ("classifier", LogisticRegression(max_iter=1000, random_state=42)),
        ]),
        "Multinomial Naive Bayes": Pipeline([
            ("tfidf", TfidfVectorizer()), ("classifier", MultinomialNB()),
        ]),
        "Random Forest": Pipeline([
            ("tfidf", TfidfVectorizer()),
            ("classifier", RandomForestClassifier(n_estimators=100, random_state=42)),
        ]),
    }


def _metrics(model: Pipeline, texts: pd.Series, labels: pd.Series) -> dict[str, float]:
    predictions = model.predict(texts)
    probabilities = model.predict_proba(texts)[:, list(model.classes_).index(1)]
    return {
        "accuracy": float(accuracy_score(labels, predictions)),
        "precision": float(precision_score(labels, predictions, zero_division=0)),
        "recall": float(recall_score(labels, predictions, zero_division=0)),
        "f1": float(f1_score(labels, predictions, zero_division=0)),
        "roc_auc": float(roc_auc_score(labels, probabilities)),
    }


def evaluate_models(
    data_path: str | Path = DATA_PATH,
    results_path: str | Path = RESULTS_PATH,
    final_results_path: str | Path = FINAL_RESULTS_PATH,
    final_model_path: str | Path = FINAL_MODEL_PATH,
    final_model_name_path: str | Path = FINAL_MODEL_NAME_PATH,
) -> dict[str, Any]:
    """Select on validation data, then evaluate exactly once on untouched test data."""
    data = pd.read_csv(data_path)
    missing_columns = REQUIRED_COLUMNS - set(data.columns)
    if missing_columns:
        raise ValueError("Prepared dataset is missing column(s): " + ", ".join(sorted(missing_columns)))
    texts = data["processed_text"].fillna("").astype(str)
    labels = pd.to_numeric(data["label"], errors="raise").astype(int)
    if texts.str.strip().eq("").any() or not set(labels.unique()).issubset({0, 1}) or labels.nunique() != 2:
        raise ValueError("Prepared dataset must contain non-empty text and both labels 0 and 1")

    class_counts = labels.value_counts().sort_index().to_dict()
    print(f"Total samples: {len(data)}")
    print(f"Class counts: {class_counts}")
    minimum_class_count = min(class_counts.values())
    if minimum_class_count < 2:
        raise ValueError(
            f"Current class counts are {class_counts}; at least 2 samples per class are required "
            "for a stratified train/test split. A larger properly labeled dataset is needed."
        )
    if minimum_class_count < 5:
        raise ValueError(
            f"Current class counts are {class_counts}; at least 5 samples per class are required "
            "for 5-fold StratifiedKFold cross-validation. A larger properly labeled dataset is needed."
        )

    train_val_texts, test_texts, train_val_labels, test_labels = train_test_split(
        texts, labels, test_size=0.15, random_state=42, stratify=labels
    )
    train_texts, validation_texts, train_labels, validation_labels = train_test_split(
        train_val_texts, train_val_labels, test_size=0.15 / 0.85, random_state=42, stratify=train_val_labels
    )
    rows: list[dict[str, float | str]] = []
    models = build_models()
    for model_name, model in models.items():
        model.fit(train_texts, train_labels)
        validation = _metrics(model, validation_texts, validation_labels)
        cv_scores = cross_val_score(
            model, train_texts, train_labels,
            cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42), scoring="f1",
        )
        rows.append({
            "model": model_name,
            "validation_accuracy": validation["accuracy"],
            "validation_precision": validation["precision"],
            "validation_recall": validation["recall"],
            "validation_f1": validation["f1"],
            "validation_roc_auc": validation["roc_auc"],
            "cv_mean_f1": float(cv_scores.mean()),
            "cv_std_f1": float(cv_scores.std()),
        })

    results = pd.DataFrame(rows, columns=RESULT_COLUMNS)
    results = results.sort_values(["validation_f1", "validation_roc_auc"], ascending=False, kind="stable").reset_index(drop=True)
    selected_name = str(results.iloc[0]["model"])
    selected_model = build_models()[selected_name]
    combined_texts = pd.concat([train_texts, validation_texts])
    combined_labels = pd.concat([train_labels, validation_labels])
    selected_model.fit(combined_texts, combined_labels)

    # The test set remains untouched until model selection and final retraining are complete.
    final_metrics = _metrics(selected_model, test_texts, test_labels)
    results_path = Path(results_path)
    final_results_path = Path(final_results_path)
    final_model_path = Path(final_model_path)
    final_model_name_path = Path(final_model_name_path)
    results_path.parent.mkdir(parents=True, exist_ok=True)
    final_results_path.parent.mkdir(parents=True, exist_ok=True)
    final_model_path.parent.mkdir(parents=True, exist_ok=True)
    final_model_name_path.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(results_path, index=False)
    pd.DataFrame([{
        "accuracy": final_metrics["accuracy"], "precision": final_metrics["precision"],
        "recall": final_metrics["recall"], "f1_score": final_metrics["f1"], "roc_auc": final_metrics["roc_auc"],
    }]).to_csv(final_results_path, index=False)
    joblib.dump(selected_model, final_model_path)
    final_model_name_path.write_text(selected_name, encoding="utf-8")
    return {
        "dataset_size": len(data), "class_distribution": labels.value_counts().sort_index().to_dict(),
        "validation_results": results, "selected_model": selected_name,
        "final_metrics": final_metrics, "confusion_matrix": confusion_matrix(test_labels, selected_model.predict(test_texts), labels=[0, 1]).tolist(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate Phase 6 fake-review models.")
    parser.add_argument("--input", default=str(DATA_PATH))
    args = parser.parse_args()
    report = evaluate_models(args.input)
    print(f"Dataset size: {report['dataset_size']}")
    print(f"Class distribution: {report['class_distribution']}")
    print("Validation and cross-validation results:")
    print(report["validation_results"].to_string(index=False, float_format=lambda value: f"{value:.4f}"))
    print(f"Selected final model: {report['selected_model']}")
    print("\nFINAL TEST RESULTS")
    for name, value in report["final_metrics"].items():
        print(f"{name.replace('_', ' ').title()}: {value:.4f}")
    print("Confusion matrix:")
    print(report["confusion_matrix"])


if __name__ == "__main__":
    main()