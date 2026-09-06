"""Prediction helper for semantic fake-review model using Sentence Transformers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib


SEMANTIC_MODEL_PATH = Path("models/semantic_embedder.pkl")
SEMANTIC_CLASSIFIER_PATH = Path("models/semantic_classifier.pkl")
SEMANTIC_CONFIG_PATH = Path("models/semantic_config.json")


def predict_semantic_review(review_text: str) -> dict[str, Any]:
    """
    Return a genuine/fake prediction and probability for one new review.

    Args:
        review_text: The review text to classify.

    Returns:
        A dictionary with:
        - "prediction": "genuine" or "fake"
        - "fake_probability": float between 0 and 1

    Raises:
        ValueError: If review_text is empty or not a string.
        FileNotFoundError: If required model files are not found.
    """
    if not isinstance(review_text, str) or not review_text.strip():
        raise ValueError("review_text must be a non-empty string")

    # Load configuration
    if not SEMANTIC_CONFIG_PATH.is_file():
        raise FileNotFoundError(f"Semantic configuration not found: {SEMANTIC_CONFIG_PATH}")

    with open(SEMANTIC_CONFIG_PATH, "r", encoding="utf-8") as f:
        config = json.load(f)

    # Load embedding model
    if not SEMANTIC_MODEL_PATH.is_file():
        raise FileNotFoundError(f"Semantic embedder model not found: {SEMANTIC_MODEL_PATH}")

    embedding_model = joblib.load(SEMANTIC_MODEL_PATH)

    # Load classifier
    if not SEMANTIC_CLASSIFIER_PATH.is_file():
        raise FileNotFoundError(f"Semantic classifier model not found: {SEMANTIC_CLASSIFIER_PATH}")

    classifier = joblib.load(SEMANTIC_CLASSIFIER_PATH)

    # Generate embedding for the review
    embedding = embedding_model.encode([review_text.strip()])

    # Make prediction
    prediction_value = int(classifier.predict(embedding)[0])
    probabilities = classifier.predict_proba(embedding)[0]

    # Get probability for fake class (class 1)
    fake_probability = float(probabilities[1])

    return {
        "prediction": "fake" if prediction_value == 1 else "genuine",
        "fake_probability": fake_probability,
    }


if __name__ == "__main__":
    # Simple test
    test_reviews = [
        "This hotel was absolutely amazing! Best experience ever!",
        "Terrible service, dirty rooms, would never return.",
    ]

    for review in test_reviews:
        result = predict_semantic_review(review)
        print(f"Review: {review[:50]}...")
        print(f"Prediction: {result['prediction']}")
        print(f"Fake probability: {result['fake_probability']:.4f}")
        print()
