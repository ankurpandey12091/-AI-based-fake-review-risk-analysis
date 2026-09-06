from pathlib import Path

import pandas as pd

from scraper.review_scraper import SAMPLE_HTML, parse_reviews_html, run_demo


def test_parse_reviews_html_extracts_reviews_and_keeps_missing_fields_empty():
    reviews = parse_reviews_html(SAMPLE_HTML)

    assert len(reviews) == 2
    assert reviews[0] == {
        "product_name": "Example Coffee Maker",
        "reviewer": "Asha",
        "rating": "5",
        "review_title": "Great coffee every morning",
        "review_text": "Simple to use and easy to clean.",
        "review_date": "2026-01-10",
    }
    assert reviews[1]["review_title"] == ""
    assert reviews[1]["review_date"] == ""


def test_run_demo_saves_csv(tmp_path: Path):
    output_path = tmp_path / "reviews.csv"

    reviews = run_demo(output_path)

    saved = pd.read_csv(output_path).fillna("")
    assert len(reviews) == len(saved) == 2
    assert list(saved.columns) == [
        "product_name",
        "reviewer",
        "rating",
        "review_title",
        "review_text",
        "review_date",
    ]