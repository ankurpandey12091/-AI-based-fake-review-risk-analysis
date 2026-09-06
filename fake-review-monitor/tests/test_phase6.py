from pathlib import Path

import joblib
import pandas as pd

from models.evaluate_models import evaluate_models
from models.predict import predict_review
from models.prepare_real_dataset import prepare_real_dataset


DEMO_DATASET = Path(__file__).parents[1] / "data" / "labeled_reviews.csv"
SOURCE_DATASET = Path(__file__).parents[1] / "data" / "raw_source.csv"


def test_real_dataset_is_validated_and_duplicates_are_removed(tmp_path: Path):
    source = tmp_path / "raw_reviews.csv"
    output = tmp_path / "real_labeled_reviews.csv"
    demo = pd.read_csv(DEMO_DATASET)
    demo["label"] = demo["label"].map({0: "truthful", 1: "deceptive"})
    pd.concat([demo[["review_text", "label"]], demo.iloc[[0]][["review_text", "label"]]]).to_csv(source, index=False)

    prepared = prepare_real_dataset(source, output, {"review_text": "review_text", "label": "label"})

    assert output.is_file()
    assert list(prepared.columns) == ["review_text", "processed_text", "label"]
    assert len(prepared) == len(demo)
    assert set(prepared["label"]) == {0, 1}
    assert prepared["review_text"].is_unique


def test_raw_source_dataset_is_converted_for_phase6(tmp_path: Path):
    source = pd.read_csv(SOURCE_DATASET)
    output = tmp_path / "real_labeled_reviews.csv"

    prepared = prepare_real_dataset(SOURCE_DATASET, output)

    assert len(source) == 1600
    assert output.is_file()
    assert {"review_text", "label", "processed_text"}.issubset(prepared.columns)
    assert set(prepared["label"]) == {0, 1}
    assert prepared["label"].nunique() == 2
    assert prepared["review_text"].notna().all()
    assert prepared["review_text"].astype(str).str.strip().ne("").all()


def test_phase6_evaluation_writes_results_model_and_prediction(tmp_path: Path):
    prepared_path = tmp_path / "real_labeled_reviews.csv"
    demo = pd.read_csv(DEMO_DATASET)
    demo.to_csv(prepared_path, index=False)
    report = evaluate_models(
        prepared_path,
        tmp_path / "phase6_model_results.csv",
        tmp_path / "final_test_results.csv",
        tmp_path / "final_fake_review_model.pkl",
        tmp_path / "final_model_name.txt",
    )

    assert report["dataset_size"] == 24
    assert set(report["validation_results"]["model"]) == {
        "Logistic Regression", "Multinomial Naive Bayes", "Random Forest"
    }
    assert (tmp_path / "phase6_model_results.csv").is_file()
    assert (tmp_path / "final_test_results.csv").is_file()
    model_path = tmp_path / "final_fake_review_model.pkl"
    assert model_path.is_file()
    assert joblib.load(model_path) is not None

    result = predict_review("This product is perfect and everyone must buy it", model_path)
    assert result["prediction"] in {"genuine", "fake"}
    assert 0.0 <= result["fake_probability"] <= 1.0