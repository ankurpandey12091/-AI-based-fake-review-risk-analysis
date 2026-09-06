from pathlib import Path

import pandas as pd
import pytest

from models.prepare_dataset import prepare_demo_dataset


def test_prepare_demo_dataset_creates_expected_labeled_schema(tmp_path: Path):
    source_path = tmp_path / "cleaned_reviews.csv"
    output_path = tmp_path / "labeled_reviews.csv"
    original = pd.DataFrame(
        {
            "review_text": ["Original review"],
            "processed_text": ["original review"],
            "rating": [5],
        }
    )
    original.to_csv(source_path, index=False)
    original_contents = source_path.read_bytes()

    labeled = prepare_demo_dataset(source_path, output_path)

    assert len(labeled) >= 20
    assert list(labeled.columns) == ["review_text", "processed_text", "label"]
    assert set(labeled["label"]) == {0, 1}
    assert labeled["label"].value_counts().to_dict() == {0: 12, 1: 12}
    assert source_path.read_bytes() == original_contents
    assert output_path.is_file()


def test_prepare_demo_dataset_requires_phase_three_columns(tmp_path: Path):
    source_path = tmp_path / "invalid.csv"
    pd.DataFrame({"review_text": ["Missing processed text"]}).to_csv(source_path, index=False)

    with pytest.raises(ValueError, match="processed_text"):
        prepare_demo_dataset(source_path, tmp_path / "labeled.csv")


def test_prepare_demo_dataset_requires_source_file(tmp_path: Path):
    with pytest.raises(FileNotFoundError):
        prepare_demo_dataset(tmp_path / "missing.csv", tmp_path / "labeled.csv")