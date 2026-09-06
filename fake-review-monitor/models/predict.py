"""Prediction helper for the saved baseline fake-review model."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib

from analyzer.text_cleaner import preprocess_text
from models.evaluate_models import FINAL_MODEL_PATH
from models.train_model import MODEL_PATH


def predict_review(review_text: str, model_path: str | Path | None = None) -> dict[str, Any]:
    """Return a genuine/fake prediction and probability for one new review."""
    if not isinstance(review_text, str) or not review_text.strip():
        raise ValueError("review_text must be a non-empty string")

    model_path = Path(model_path) if model_path is not None else (FINAL_MODEL_PATH if FINAL_MODEL_PATH.is_file() else MODEL_PATH)
    if not model_path.is_file():
        raise FileNotFoundError(f"Saved model not found: {model_path}")

    model = joblib.load(model_path)
    processed_text = " ".join(preprocess_text(review_text))
    prediction_value = int(model.predict([processed_text])[0])
    probabilities = model.predict_proba([processed_text])[0]
    fake_index = list(model.classes_).index(1)
    return {
        "prediction": "fake" if prediction_value == 1 else "genuine",
        "fake_probability": float(probabilities[fake_index]),
    }