from pathlib import Path

import pandas as pd
import pytest

from analyzer.text_cleaner import clean_text, preprocess_reviews, preprocess_text


def test_clean_text_removes_html_urls_punctuation_and_extra_spaces():
    text = " <p>Great PRODUCT!</p> Visit https://example.com now.  "

    assert clean_text(text) == "great product visit now"


def test_preprocess_text_removes_stopwords_and_lemmatizes():
    assert preprocess_text("The cats are running") == ["cat", "running"]


def test_preprocess_reviews_preserves_original_and_saves_cleaned_data(tmp_path: Path):
    input_path = tmp_path / "reviews.csv"
    output_path = tmp_path / "cleaned_reviews.csv"
    pd.DataFrame({"review_text": ["<b>Great!</b>", None]}).to_csv(input_path, index=False)

    processed = preprocess_reviews(input_path, output_path)

    assert processed.loc[0, "review_text"] == "<b>Great!</b>"
    assert processed.loc[0, "cleaned_review_text"] == "great"
    assert processed.loc[1, "cleaned_review_text"] == ""
    assert processed.loc[0, "processed_text"] == "great"
    assert output_path.is_file()


def test_preprocess_reviews_reports_missing_input_and_column(tmp_path: Path):
    with pytest.raises(FileNotFoundError):
        preprocess_reviews(tmp_path / "missing.csv")

    input_path = tmp_path / "invalid.csv"
    pd.DataFrame({"title": ["No review text"]}).to_csv(input_path, index=False)
    with pytest.raises(ValueError, match="review_text"):
        preprocess_reviews(input_path)