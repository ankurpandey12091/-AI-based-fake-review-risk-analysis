"""Clean and preprocess review text for later analysis."""

from __future__ import annotations

import argparse
from functools import lru_cache
import re
import string
from pathlib import Path
from typing import Any

import nltk
import pandas as pd
from bs4 import BeautifulSoup
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize, wordpunct_tokenize


DEFAULT_INPUT_PATH = Path("data/reviews.csv")
DEFAULT_OUTPUT_PATH = Path("data/cleaned_reviews.csv")
REQUIRED_COLUMNS = {"review_text"}
URL_PATTERN = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)
FALLBACK_STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "for",
    "from",
    "has",
    "he",
    "i",
    "in",
    "is",
    "it",
    "its",
    "me",
    "my",
    "of",
    "on",
    "or",
    "that",
    "the",
    "this",
    "to",
    "was",
    "we",
    "were",
    "with",
    "you",
}


def clean_text(text: Any) -> str:
    """Return normalized review text, safely handling null or non-string input."""
    if pd.isna(text):
        return ""

    text = BeautifulSoup(str(text), "html.parser").get_text(" ", strip=True)
    text = URL_PATTERN.sub(" ", text.lower())
    text = text.translate(str.maketrans({character: " " for character in string.punctuation}))
    return re.sub(r"\s+", " ", text).strip()


@lru_cache(maxsize=1)
def _stopword_set() -> set[str]:
    """Load NLTK stopwords, using a small fallback when data is unavailable."""
    try:
        return set(stopwords.words("english"))
    except LookupError:
        return FALLBACK_STOPWORDS


def _tokenize(text: str) -> list[str]:
    try:
        return word_tokenize(text)
    except LookupError:
        return wordpunct_tokenize(text)


def _fallback_lemma(token: str) -> str:
    """Apply a small fallback normalization when WordNet data is unavailable."""
    if len(token) > 3 and token.endswith("ies"):
        return f"{token[:-3]}y"
    if len(token) > 3 and token.endswith("s") and not token.endswith("ss"):
        return token[:-1]
    return token


def preprocess_text(text: Any) -> list[str]:
    """Tokenize cleaned text, remove stopwords, and lemmatize each token."""
    cleaned = clean_text(text)
    if not cleaned:
        return []

    lemmatizer = WordNetLemmatizer()
    words = []
    for token in _tokenize(cleaned):
        if not token.isalpha() or token in _stopword_set():
            continue
        try:
            token = lemmatizer.lemmatize(token)
        except LookupError:
            token = _fallback_lemma(token)
        words.append(token)
    return words


def preprocess_reviews(
    input_path: str | Path = DEFAULT_INPUT_PATH,
    output_path: str | Path = DEFAULT_OUTPUT_PATH,
) -> pd.DataFrame:
    """Load, clean, preprocess, and save reviews as a new CSV dataset."""
    input_path = Path(input_path)
    output_path = Path(output_path)
    if not input_path.is_file():
        raise FileNotFoundError(f"Input review CSV not found: {input_path}")

    reviews = pd.read_csv(input_path)
    missing_columns = REQUIRED_COLUMNS - set(reviews.columns)
    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise ValueError(f"Input CSV is missing required column(s): {missing}")

    reviews["cleaned_review_text"] = reviews["review_text"].apply(clean_text)
    reviews["processed_tokens"] = reviews["cleaned_review_text"].apply(preprocess_text)
    reviews["processed_text"] = reviews["processed_tokens"].apply(" ".join)
    reviews["processed_tokens"] = reviews["processed_tokens"].apply(" ".join)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    reviews.to_csv(output_path, index=False)
    return reviews


def main() -> None:
    parser = argparse.ArgumentParser(description="Clean and preprocess review text.")
    parser.add_argument("--input", default=str(DEFAULT_INPUT_PATH), help="Input reviews CSV path")
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT_PATH), help="Output cleaned CSV path")
    args = parser.parse_args()

    reviews = preprocess_reviews(args.input, args.output)
    print(f"Saved {len(reviews)} cleaned reviews to {args.output}")


if __name__ == "__main__":
    main()