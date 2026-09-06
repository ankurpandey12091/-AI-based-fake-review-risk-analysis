from pathlib import Path

import pandas as pd

from models.compare_models import METRIC_COLUMNS, compare_models


def test_comparison_creates_csv_with_all_models(tmp_path: Path):
    output_path = tmp_path / "model_comparison.csv"

    results, confusion_matrices, best_model = compare_models(output_path=output_path)

    assert output_path.is_file()
    assert set(results["model"]) == {"Logistic Regression", "Multinomial Naive Bayes", "Random Forest"}
    assert list(results.columns) == METRIC_COLUMNS
    assert len(confusion_matrices) == 3
    assert best_model in set(results["model"])

    saved = pd.read_csv(output_path)
    assert list(saved.columns) == METRIC_COLUMNS
    for metric in METRIC_COLUMNS[1:]:
        assert saved[metric].between(0, 1).all()