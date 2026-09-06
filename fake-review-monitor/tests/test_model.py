from pathlib import Path

import joblib

from models.predict import predict_review
from models.train_model import DATA_PATH, MODEL_PATH, train_model


def test_training_creates_loadable_pipeline():
    metrics = train_model(DATA_PATH, MODEL_PATH)

    assert MODEL_PATH.is_file()
    loaded_model = joblib.load(MODEL_PATH)
    assert loaded_model.named_steps["tfidf"] is not None
    assert loaded_model.named_steps["classifier"] is not None
    assert metrics["dataset_size"] == 24


def test_prediction_returns_label_and_probability():
    train_model(DATA_PATH, MODEL_PATH)

    result = predict_review("Amazing perfect product buy this now", MODEL_PATH)

    assert result["prediction"] in {"genuine", "fake"}
    assert 0.0 <= result["fake_probability"] <= 1.0


def test_prediction_requires_saved_model(tmp_path: Path):
    missing_model = tmp_path / "missing.pkl"

    try:
        predict_review("A useful product", missing_model)
    except FileNotFoundError:
        pass
    else:
        raise AssertionError("Expected missing model to raise FileNotFoundError")