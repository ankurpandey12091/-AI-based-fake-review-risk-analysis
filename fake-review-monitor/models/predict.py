"""Prediction helper for the saved baseline fake-review model."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib

from analyzer.text_cleaner import preprocess_text

# Anchor paths relative to this file's directory
BASE_DIR = Path(__file__).resolve().parent
FINAL_MODEL_PATH = BASE_DIR / "final_fake_review_model.pkl"
MODEL_PATH = BASE_DIR / "fake_review_model.pkl"

_cached_model: Any = None
_cached_path: str | None = None


def _get_model(model_path: Path) -> Any:
    """Return loaded model, caching it in memory to avoid repeated disk reads."""
    global _cached_model, _cached_path
    path_str = str(model_path.resolve())
    if _cached_model is None or _cached_path != path_str:
        _cached_model = joblib.load(model_path)
        _cached_path = path_str
    return _cached_model


def predict_review(review_text: str, model_path: str | Path | None = None) -> dict[str, Any]:
    """Return a genuine/fake prediction and probability for one new review."""
    if not isinstance(review_text, str) or not review_text.strip():
        raise ValueError("review_text must be a non-empty string")

    if model_path is not None:
        target_path = Path(model_path)
        if not target_path.is_file():
            # Check relative to BASE_DIR as well
            alt_path = BASE_DIR / target_path.name
            if alt_path.is_file():
                target_path = alt_path
    else:
        target_path = FINAL_MODEL_PATH if FINAL_MODEL_PATH.is_file() else MODEL_PATH

    if not target_path.is_file():
        raise FileNotFoundError(f"Saved model not found: {target_path}")

    model = _get_model(target_path)
    processed_text = " ".join(preprocess_text(review_text))
    prediction_value = int(model.predict([processed_text])[0])
    probabilities = model.predict_proba([processed_text])[0]
    fake_index = list(model.classes_).index(1)
    return {
        "prediction": "fake" if prediction_value == 1 else "genuine",
        "fake_probability": float(probabilities[fake_index]),
    }