"""Tests for Phase 7 semantic model with Sentence Transformers."""

from pathlib import Path

import joblib
import pandas as pd

from models.semantic_model import train_semantic_model
from models.semantic_predict import predict_semantic_review


DEMO_DATASET = Path(__file__).parents[1] / "data" / "labeled_reviews.csv"


def test_semantic_model_loads_and_trains(tmp_path: Path):
    """Test that semantic model can be trained on test data."""
    prepared_path = tmp_path / "real_labeled_reviews.csv"
    demo = pd.read_csv(DEMO_DATASET)
    # Ensure required columns are present
    if "review_text" not in demo.columns:
        raise ValueError("Demo dataset must have review_text column")
    demo.to_csv(prepared_path, index=False)

    results_path = tmp_path / "phase7_results.csv"
    semantic_model_path = tmp_path / "semantic_embedder.pkl"
    semantic_classifier_path = tmp_path / "semantic_classifier.pkl"
    semantic_config_path = tmp_path / "semantic_config.json"

    report = train_semantic_model(
        prepared_path,
        results_path,
        semantic_model_path,
        semantic_classifier_path,
        semantic_config_path,
    )

    assert report["dataset_size"] == len(demo)
    assert report["class_distribution"] is not None
    assert len(report["validation_metrics"]) > 0
    assert len(report["test_metrics"]) > 0


def test_semantic_model_files_are_created(tmp_path: Path):
    """Test that all required model files are created."""
    prepared_path = tmp_path / "real_labeled_reviews.csv"
    demo = pd.read_csv(DEMO_DATASET)
    demo.to_csv(prepared_path, index=False)

    results_path = tmp_path / "phase7_results.csv"
    semantic_model_path = tmp_path / "semantic_embedder.pkl"
    semantic_classifier_path = tmp_path / "semantic_classifier.pkl"
    semantic_config_path = tmp_path / "semantic_config.json"

    train_semantic_model(
        prepared_path,
        results_path,
        semantic_model_path,
        semantic_classifier_path,
        semantic_config_path,
    )

    assert results_path.is_file(), f"Results file not created: {results_path}"
    assert semantic_model_path.is_file(), f"Semantic model file not created: {semantic_model_path}"
    assert semantic_classifier_path.is_file(), f"Classifier file not created: {semantic_classifier_path}"
    assert semantic_config_path.is_file(), f"Config file not created: {semantic_config_path}"


def test_semantic_embedder_can_be_loaded(tmp_path: Path):
    """Test that the saved semantic embedder can be loaded."""
    prepared_path = tmp_path / "real_labeled_reviews.csv"
    demo = pd.read_csv(DEMO_DATASET)
    demo.to_csv(prepared_path, index=False)

    semantic_model_path = tmp_path / "semantic_embedder.pkl"

    train_semantic_model(
        prepared_path,
        tmp_path / "phase7_results.csv",
        semantic_model_path,
        tmp_path / "semantic_classifier.pkl",
        tmp_path / "semantic_config.json",
    )

    embedder = joblib.load(semantic_model_path)
    assert embedder is not None

    # Test that embedder can generate embeddings
    test_texts = ["Great hotel experience!", "Terrible stay."]
    embeddings = embedder.encode(test_texts)
    assert embeddings.shape[0] == 2
    assert embeddings.shape[1] > 0  # Embedding dimension


def test_semantic_classifier_can_be_loaded(tmp_path: Path):
    """Test that the saved classifier can be loaded."""
    prepared_path = tmp_path / "real_labeled_reviews.csv"
    demo = pd.read_csv(DEMO_DATASET)
    demo.to_csv(prepared_path, index=False)

    semantic_classifier_path = tmp_path / "semantic_classifier.pkl"

    train_semantic_model(
        prepared_path,
        tmp_path / "phase7_results.csv",
        tmp_path / "semantic_embedder.pkl",
        semantic_classifier_path,
        tmp_path / "semantic_config.json",
    )

    classifier = joblib.load(semantic_classifier_path)
    assert classifier is not None


def test_semantic_prediction_works(tmp_path: Path):
    """Test that semantic prediction works correctly."""
    prepared_path = tmp_path / "real_labeled_reviews.csv"
    demo = pd.read_csv(DEMO_DATASET)
    demo.to_csv(prepared_path, index=False)

    semantic_model_path = tmp_path / "semantic_embedder.pkl"
    semantic_classifier_path = tmp_path / "semantic_classifier.pkl"
    semantic_config_path = tmp_path / "semantic_config.json"

    # Copy paths to expected locations for predict function
    import sys
    sys.path.insert(0, str(tmp_path.parent.parent))

    train_semantic_model(
        prepared_path,
        tmp_path / "phase7_results.csv",
        semantic_model_path,
        semantic_classifier_path,
        semantic_config_path,
    )

    # Monkey-patch the paths in the predict module
    import models.semantic_predict as sp
    sp.SEMANTIC_MODEL_PATH = semantic_model_path
    sp.SEMANTIC_CLASSIFIER_PATH = semantic_classifier_path
    sp.SEMANTIC_CONFIG_PATH = semantic_config_path

    result = predict_semantic_review("This hotel was fantastic and wonderful!")
    assert result["prediction"] in {"genuine", "fake"}
    assert 0.0 <= result["fake_probability"] <= 1.0


def test_semantic_prediction_probability_range(tmp_path: Path):
    """Test that prediction probability is always between 0 and 1."""
    prepared_path = tmp_path / "real_labeled_reviews.csv"
    demo = pd.read_csv(DEMO_DATASET)
    demo.to_csv(prepared_path, index=False)

    semantic_model_path = tmp_path / "semantic_embedder.pkl"
    semantic_classifier_path = tmp_path / "semantic_classifier.pkl"
    semantic_config_path = tmp_path / "semantic_config.json"

    import sys
    sys.path.insert(0, str(tmp_path.parent.parent))

    train_semantic_model(
        prepared_path,
        tmp_path / "phase7_results.csv",
        semantic_model_path,
        semantic_classifier_path,
        semantic_config_path,
    )

    import models.semantic_predict as sp
    sp.SEMANTIC_MODEL_PATH = semantic_model_path
    sp.SEMANTIC_CLASSIFIER_PATH = semantic_classifier_path
    sp.SEMANTIC_CONFIG_PATH = semantic_config_path

    test_reviews = [
        "Amazing experience!",
        "Terrible place.",
        "It was okay.",
        "Best hotel ever in the world!",
        "Worst place I have ever been.",
    ]

    for review in test_reviews:
        result = predict_semantic_review(review)
        assert isinstance(result, dict)
        assert "prediction" in result
        assert "fake_probability" in result
        assert 0.0 <= result["fake_probability"] <= 1.0, \
            f"Probability out of range: {result['fake_probability']}"


def test_semantic_results_csv_is_valid(tmp_path: Path):
    """Test that phase7_results.csv is created with valid structure."""
    prepared_path = tmp_path / "real_labeled_reviews.csv"
    demo = pd.read_csv(DEMO_DATASET)
    demo.to_csv(prepared_path, index=False)

    results_path = tmp_path / "phase7_results.csv"

    train_semantic_model(
        prepared_path,
        results_path,
        tmp_path / "semantic_embedder.pkl",
        tmp_path / "semantic_classifier.pkl",
        tmp_path / "semantic_config.json",
    )

    assert results_path.is_file()
    results = pd.read_csv(results_path)

    required_columns = [
        "model", "embedding_model", "accuracy", "precision",
        "recall", "f1_score", "roc_auc", "dataset_size",
        "train_size", "validation_size", "test_size",
    ]
    for col in required_columns:
        assert col in results.columns, f"Missing column: {col}"

    # Verify metric values are in valid ranges
    assert 0.0 <= results["accuracy"].iloc[0] <= 1.0
    assert 0.0 <= results["precision"].iloc[0] <= 1.0
    assert 0.0 <= results["recall"].iloc[0] <= 1.0
    assert 0.0 <= results["f1_score"].iloc[0] <= 1.0
    assert 0.0 <= results["roc_auc"].iloc[0] <= 1.0
