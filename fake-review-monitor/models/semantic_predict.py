"""Prediction helper for semantic fake-review model using Sentence Transformers."""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any

import joblib

logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parent
SEMANTIC_MODEL_PATH = BASE_DIR / "semantic_embedder.pkl"
SEMANTIC_CLASSIFIER_PATH = BASE_DIR / "semantic_classifier.pkl"
SEMANTIC_CONFIG_PATH = BASE_DIR / "semantic_config.json"

_cached_embedder: Any = None
_cached_classifier: Any = None


def _get_models() -> tuple[Any, Any]:
    """Return cached embedding model and classifier, loading once into memory."""
    global _cached_embedder, _cached_classifier

    # Check if semantic model is explicitly disabled (e.g. for low-memory Render Free Tier)
    if os.environ.get("DISABLE_SEMANTIC_MODEL", "").lower() in ("1", "true", "yes"):
        raise RuntimeError("Semantic model disabled via DISABLE_SEMANTIC_MODEL environment variable")

    if not SEMANTIC_MODEL_PATH.is_file():
        raise FileNotFoundError(f"Semantic embedder model not found: {SEMANTIC_MODEL_PATH}")

    if not SEMANTIC_CLASSIFIER_PATH.is_file():
        raise FileNotFoundError(f"Semantic classifier model not found: {SEMANTIC_CLASSIFIER_PATH}")

    if _cached_embedder is None:
        logger.info("Loading semantic embedder model into memory from %s...", SEMANTIC_MODEL_PATH)
        _cached_embedder = joblib.load(SEMANTIC_MODEL_PATH)

    if _cached_classifier is None:
        logger.info("Loading semantic classifier into memory from %s...", SEMANTIC_CLASSIFIER_PATH)
        _cached_classifier = joblib.load(SEMANTIC_CLASSIFIER_PATH)

    return _cached_embedder, _cached_classifier


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

    embedding_model, classifier = _get_models()

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
