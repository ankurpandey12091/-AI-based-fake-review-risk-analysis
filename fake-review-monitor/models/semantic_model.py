"""Phase 7: Semantic NLP model using Sentence Transformers for fake-review detection."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sentence_transformers import SentenceTransformer


DATA_PATH = Path("data/real_labeled_reviews.csv")
RESULTS_PATH = Path("data/phase7_results.csv")
SEMANTIC_MODEL_PATH = Path("models/semantic_embedder.pkl")
SEMANTIC_CLASSIFIER_PATH = Path("models/semantic_classifier.pkl")
SEMANTIC_CONFIG_PATH = Path("models/semantic_config.json")

EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
REQUIRED_COLUMNS = {"review_text", "label"}


def load_dataset(data_path: str | Path = DATA_PATH) -> tuple[pd.Series, pd.Series]:
    """Load and validate the review text and binary labels."""
    data_path = Path(data_path)
    if not data_path.is_file():
        raise FileNotFoundError(f"Dataset not found: {data_path}")

    data = pd.read_csv(data_path)
    missing_columns = REQUIRED_COLUMNS - set(data.columns)
    if missing_columns:
        raise ValueError("Dataset is missing column(s): " + ", ".join(sorted(missing_columns)))

    data = data[["review_text", "label"]].copy()
    data["review_text"] = data["review_text"].fillna("").astype(str).str.strip()
    data["label"] = pd.to_numeric(data["label"], errors="coerce")
    data = data[(data["review_text"] != "") & data["label"].notna()]
    if data.empty:
        raise ValueError("Dataset contains no usable review text and labels.")
    if not set(data["label"].unique()).issubset({0, 1}):
        raise ValueError("Labels must use only 0 for genuine and 1 for fake.")
    if data["label"].nunique() != 2:
        raise ValueError("Dataset must contain both label classes 0 and 1.")
    return data["review_text"], data["label"].astype(int)


def generate_embeddings(
    texts: pd.Series,
    model_name: str = EMBEDDING_MODEL_NAME,
) -> np.ndarray:
    """Generate sentence embeddings using the Sentence Transformer model."""
    model = SentenceTransformer(model_name)
    embeddings = model.encode(texts.tolist(), show_progress_bar=True)
    return embeddings


def train_semantic_model(
    data_path: str | Path = DATA_PATH,
    results_path: str | Path = RESULTS_PATH,
    semantic_model_path: str | Path = SEMANTIC_MODEL_PATH,
    semantic_classifier_path: str | Path = SEMANTIC_CLASSIFIER_PATH,
    semantic_config_path: str | Path = SEMANTIC_CONFIG_PATH,
) -> dict[str, Any]:
    """
    Train a semantic model with Sentence Transformers and Logistic Regression.
    Follows the same 70/15/15 split as Phase 6 for fair comparison.
    """
    print("=" * 80)
    print("PHASE 7: Semantic NLP Model with Sentence Transformers")
    print("=" * 80)
    print(f"Embedding model: {EMBEDDING_MODEL_NAME}")

    # Load and validate dataset
    texts, labels = load_dataset(data_path)
    print(f"Dataset size: {len(texts)}")
    print(f"Class distribution: {labels.value_counts().sort_index().to_dict()}")

    # Split: 70% train+val, 15% test
    train_val_texts, test_texts, train_val_labels, test_labels = train_test_split(
        texts, labels, test_size=0.15, random_state=42, stratify=labels
    )

    # Split train+val into 70% train, 15% val (from original)
    train_texts, validation_texts, train_labels, validation_labels = train_test_split(
        train_val_texts, train_val_labels, test_size=0.15 / 0.85, random_state=42, stratify=train_val_labels
    )

    print(f"\nTrain set size: {len(train_texts)}")
    print(f"Validation set size: {len(validation_texts)}")
    print(f"Test set size: {len(test_texts)}")

    # Generate embeddings for all splits
    print("\nGenerating embeddings for training data...")
    train_embeddings = generate_embeddings(train_texts)
    print("Generating embeddings for validation data...")
    val_embeddings = generate_embeddings(validation_texts)
    print("Generating embeddings for test data...")
    test_embeddings = generate_embeddings(test_texts)

    # Train classifier on training embeddings
    print("\nTraining Logistic Regression classifier on training embeddings...")
    classifier = LogisticRegression(max_iter=1000, random_state=42)
    classifier.fit(train_embeddings, train_labels)

    # Validate on validation set
    val_predictions = classifier.predict(val_embeddings)
    val_probabilities = classifier.predict_proba(val_embeddings)[:, 1]
    validation_metrics = {
        "accuracy": float(accuracy_score(validation_labels, val_predictions)),
        "precision": float(precision_score(validation_labels, val_predictions, zero_division=0)),
        "recall": float(recall_score(validation_labels, val_predictions, zero_division=0)),
        "f1": float(f1_score(validation_labels, val_predictions, zero_division=0)),
        "roc_auc": float(roc_auc_score(validation_labels, val_probabilities)),
    }

    print("\nValidation Metrics:")
    print(f"  Accuracy: {validation_metrics['accuracy']:.4f}")
    print(f"  Precision: {validation_metrics['precision']:.4f}")
    print(f"  Recall: {validation_metrics['recall']:.4f}")
    print(f"  F1-score: {validation_metrics['f1']:.4f}")
    print(f"  ROC-AUC: {validation_metrics['roc_auc']:.4f}")

    # Retrain on combined train+val for final evaluation
    print("\nRetraining on combined train+validation data...")
    combined_embeddings = np.vstack([train_embeddings, val_embeddings])
    combined_labels = pd.concat([train_labels, validation_labels])
    final_classifier = LogisticRegression(max_iter=1000, random_state=42)
    final_classifier.fit(combined_embeddings, combined_labels)

    # Evaluate on untouched test set (only once)
    test_predictions = final_classifier.predict(test_embeddings)
    test_probabilities = final_classifier.predict_proba(test_embeddings)[:, 1]
    test_metrics = {
        "accuracy": float(accuracy_score(test_labels, test_predictions)),
        "precision": float(precision_score(test_labels, test_predictions, zero_division=0)),
        "recall": float(recall_score(test_labels, test_predictions, zero_division=0)),
        "f1": float(f1_score(test_labels, test_predictions, zero_division=0)),
        "roc_auc": float(roc_auc_score(test_labels, test_probabilities)),
    }

    print("\n" + "=" * 80)
    print("FINAL TEST RESULTS (Phase 7 - Semantic Model)")
    print("=" * 80)
    print(f"Accuracy:  {test_metrics['accuracy']:.4f}")
    print(f"Precision: {test_metrics['precision']:.4f}")
    print(f"Recall:    {test_metrics['recall']:.4f}")
    print(f"F1-score:  {test_metrics['f1']:.4f}")
    print(f"ROC-AUC:   {test_metrics['roc_auc']:.4f}")
    print(f"Confusion matrix:\n{confusion_matrix(test_labels, test_predictions, labels=[0, 1]).tolist()}")

    # Phase 6 baseline for comparison
    phase6_metrics = {
        "accuracy": 0.8667,
        "precision": 0.8548,
        "recall": 0.8833,
        "f1": 0.8689,
        "roc_auc": 0.9391,
    }

    print("\n" + "=" * 80)
    print("PHASE 6 vs PHASE 7 COMPARISON")
    print("=" * 80)
    print(f"{'Metric':<15} {'Phase 6 (TF-IDF)':<20} {'Phase 7 (Semantic)':<20} {'Difference':<15}")
    print("-" * 70)
    for metric in ["accuracy", "precision", "recall", "f1", "roc_auc"]:
        p6_val = phase6_metrics[metric]
        p7_val = test_metrics[metric]
        diff = p7_val - p6_val
        print(f"{metric:<15} {p6_val:<20.4f} {p7_val:<20.4f} {diff:+.4f}")

    # Save results
    results_path = Path(results_path)
    results_path.parent.mkdir(parents=True, exist_ok=True)

    results_df = pd.DataFrame([{
        "model": "Sentence Transformers + Logistic Regression",
        "embedding_model": EMBEDDING_MODEL_NAME,
        "accuracy": test_metrics["accuracy"],
        "precision": test_metrics["precision"],
        "recall": test_metrics["recall"],
        "f1_score": test_metrics["f1"],
        "roc_auc": test_metrics["roc_auc"],
        "dataset_size": len(texts),
        "train_size": len(train_texts),
        "validation_size": len(validation_texts),
        "test_size": len(test_texts),
    }])
    results_df.to_csv(results_path, index=False)
    print(f"\nPhase 7 results saved to: {results_path}")

    # Save classifier and configuration
    semantic_model_path = Path(semantic_model_path)
    semantic_classifier_path = Path(semantic_classifier_path)
    semantic_config_path = Path(semantic_config_path)

    semantic_model_path.parent.mkdir(parents=True, exist_ok=True)
    semantic_classifier_path.parent.mkdir(parents=True, exist_ok=True)
    semantic_config_path.parent.mkdir(parents=True, exist_ok=True)

    # Load the embedding model and save it
    embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    joblib.dump(embedding_model, semantic_model_path)

    # Save the classifier
    joblib.dump(final_classifier, semantic_classifier_path)

    # Save configuration
    config = {
        "embedding_model_name": EMBEDDING_MODEL_NAME,
        "classifier_type": "LogisticRegression",
        "classifier_params": {
            "max_iter": 1000,
            "random_state": 42,
        },
        "random_state": 42,
        "stratified": True,
    }
    with open(semantic_config_path, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)

    print(f"Semantic model saved to: {semantic_model_path}")
    print(f"Classifier saved to: {semantic_classifier_path}")
    print(f"Configuration saved to: {semantic_config_path}")

    return {
        "embedding_model": EMBEDDING_MODEL_NAME,
        "dataset_size": len(texts),
        "class_distribution": labels.value_counts().sort_index().to_dict(),
        "validation_metrics": validation_metrics,
        "test_metrics": test_metrics,
        "confusion_matrix": confusion_matrix(test_labels, test_predictions, labels=[0, 1]).tolist(),
    }


def main() -> None:
    report = train_semantic_model()
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"Embedding model: {report['embedding_model']}")
    print(f"Dataset size: {report['dataset_size']}")
    print(f"Class distribution: {report['class_distribution']}")
    print("\nValidation metrics:")
    for metric, value in report["validation_metrics"].items():
        print(f"  {metric}: {value:.4f}")
    print("\nTest metrics:")
    for metric, value in report["test_metrics"].items():
        print(f"  {metric}: {value:.4f}")


if __name__ == "__main__":
    main()
