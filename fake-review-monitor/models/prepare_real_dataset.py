"""Convert and preprocess the deceptive opinion dataset for Phase 6."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Mapping

import pandas as pd

try:
    from analyzer.text_cleaner import preprocess_text
except ModuleNotFoundError:  # Support the documented direct script command.
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from analyzer.text_cleaner import preprocess_text


DEFAULT_INPUT_PATH = Path("data/raw_source.csv")
DEFAULT_OUTPUT_PATH = Path("data/real_labeled_reviews.csv")
DEFAULT_COLUMN_MAPPING = {"review_text": "text", "label": "deceptive"}
LABEL_MAPPING = {"truthful": 0, "deceptive": 1}
OPTIONAL_COLUMNS = ["hotel", "polarity", "source"]


def prepare_real_dataset(
    input_path: str | Path = DEFAULT_INPUT_PATH,
    output_path: str | Path = DEFAULT_OUTPUT_PATH,
    column_mapping: Mapping[str, str] | None = None,
) -> pd.DataFrame:
    """Validate, deduplicate, preprocess, and save a labeled dataset.

    ``column_mapping`` maps normalized names (``review_text`` and ``label``)
    to the exact source column names. No source-column guessing is performed.
    """
    input_path = Path(input_path)
    output_path = Path(output_path)
    if not input_path.is_file():
        raise FileNotFoundError(f"Raw labeled dataset not found: {input_path}")

    mapping = dict(DEFAULT_COLUMN_MAPPING if column_mapping is None else column_mapping)
    required_normalized = set(DEFAULT_COLUMN_MAPPING)
    if set(mapping) != required_normalized or any(not str(value).strip() for value in mapping.values()):
        raise ValueError("column_mapping must map review_text and label to non-empty source column names")

    source = pd.read_csv(input_path)
    missing_source_columns = set(mapping.values()) - set(source.columns)
    if missing_source_columns:
        missing = ", ".join(sorted(missing_source_columns))
        raise ValueError(
            f"Raw dataset is missing mapped column(s): {missing}. "
            "Set --review-column and --label-column for different source names."
        )

    selected_columns = [mapping["review_text"], mapping["label"]]
    selected_columns.extend(column for column in OPTIONAL_COLUMNS if column in source.columns)
    data = source[selected_columns].rename(
        columns={mapping["review_text"]: "review_text", mapping["label"]: "label"}
    )
    missing_rows = data["review_text"].isna() | data["review_text"].astype(str).str.strip().eq("")
    missing_row_count = int(missing_rows.sum())
    data = data.loc[~missing_rows].copy()
    if data.empty:
        raise ValueError("review_text is completely empty")

    source_labels = data["label"].astype(str).str.strip().str.lower()
    if not set(source_labels).issubset(LABEL_MAPPING):
        raise ValueError("Labels must contain only truthful or deceptive")
    data["label"] = source_labels.map(LABEL_MAPPING).astype(int)

    duplicate_count = int(data.duplicated(subset=["review_text"], keep="first").sum())
    data = data.drop_duplicates(subset=["review_text"], keep="first").copy()
    if data.empty:
        raise ValueError("No reviews remain after duplicate removal")
    if data["label"].nunique() != 2:
        raise ValueError("Dataset must contain both label classes 0 and 1")

    data["processed_text"] = data["review_text"].apply(lambda text: " ".join(preprocess_text(text)))
    if (data["processed_text"].str.strip() == "").any():
        raise ValueError("At least one review becomes empty after preprocessing")
    output_columns = ["review_text", "processed_text", "label"]
    output_columns.extend(column for column in OPTIONAL_COLUMNS if column in data.columns)
    data = data[output_columns]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    data.to_csv(output_path, index=False)
    counts = data["label"].value_counts().to_dict()
    truthful_count = int(counts.get(0, 0))
    deceptive_count = int(counts.get(1, 0))
    print(f"Total rows: {len(data)}")
    print(f"Truthful count: {truthful_count}")
    print(f"Deceptive count: {deceptive_count}")
    print(f"Duplicate rows removed: {duplicate_count}")
    print(f"Missing rows removed: {missing_row_count}")
    print(f"Class percentages: truthful {truthful_count / len(data) * 100:.2f}%, deceptive {deceptive_count / len(data) * 100:.2f}%")
    print(f"Saved validated dataset: {output_path}")
    return data


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare a manually labeled real-review dataset.")
    parser.add_argument("--input", default=str(DEFAULT_INPUT_PATH))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT_PATH))
    parser.add_argument("--review-column", default="text", help="Exact source column containing review text")
    parser.add_argument("--label-column", default="deceptive", help="Exact source column containing truthful/deceptive labels")
    args = parser.parse_args()
    prepare_real_dataset(
        args.input,
        args.output,
        {"review_text": args.review_column, "label": args.label_column},
    )


if __name__ == "__main__":
    main()