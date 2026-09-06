"""Collect reviews from permitted public HTML pages.

This module only requests pages that are already publicly accessible. It does
not log in, bypass CAPTCHAs, or attempt to work around anti-bot controls.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import pandas as pd
import requests
from bs4 import BeautifulSoup


REVIEW_COLUMNS = [
    "product_name",
    "reviewer",
    "rating",
    "review_title",
    "review_text",
    "review_date",
]

SAMPLE_HTML = """
<html>
  <head><title>Example Coffee Maker</title></head>
  <body>
    <main data-product-name="Example Coffee Maker">
      <article class="review" data-review-id="review-1">
        <span class="reviewer">Asha</span>
        <span class="rating">5</span>
        <h3 class="review-title">Great coffee every morning</h3>
        <p class="review-text">Simple to use and easy to clean.</p>
        <time class="review-date">2026-01-10</time>
      </article>
      <article class="review" data-review-id="review-2">
        <span class="reviewer">Guest Reviewer</span>
        <span class="rating">4</span>
        <p class="review-text">Works well for a small kitchen.</p>
      </article>
    </main>
  </body>
</html>
"""


class ReviewScraperError(RuntimeError):
    """Raised when a permitted public page cannot be downloaded."""


def _text(element: Any) -> str:
    """Return clean text for a BeautifulSoup element, or an empty string."""
    return element.get_text(" ", strip=True) if element else ""


def _first_text(review: Any, selectors: tuple[str, ...]) -> str:
    for selector in selectors:
        value = _text(review.select_one(selector))
        if value:
            return value
    return ""


def parse_reviews_html(html: str, source_url: str = "") -> list[dict[str, str]]:
    """Parse review records from common, class-based public HTML markup.

    Pages should be adapted with their permitted selectors when their markup
    differs. Missing fields are represented by empty strings.
    """
    soup = BeautifulSoup(html, "html.parser")
    product = soup.select_one("[data-product-name]")
    product_name = product.get("data-product-name", "") if product else ""
    product_name = product_name or _text(soup.select_one("h1, meta[property='og:title'], title"))

    reviews = soup.select("[data-review-id], article.review, .review")
    records: list[dict[str, str]] = []
    for review in reviews:
        rating_element = review.select_one("[data-rating], .rating, [class*='rating']")
        rating = ""
        if rating_element:
            rating = rating_element.get("data-rating", "") or _text(rating_element)

        records.append(
            {
                "product_name": product_name,
                "reviewer": _first_text(review, ("[data-reviewer]", ".reviewer", ".author", ".user")),
                "rating": rating,
                "review_title": _first_text(review, ("[data-review-title]", ".review-title", "h3", "h4")),
                "review_text": _first_text(review, ("[data-review-text]", ".review-text", ".content", "p")),
                "review_date": _first_text(review, ("[data-review-date]", ".review-date", "time", "[datetime]")),
            }
        )
    return records


def save_reviews(reviews: list[dict[str, str]], output_path: str | Path = "data/reviews.csv") -> Path:
    """Write review records to CSV and return the output path."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(reviews, columns=REVIEW_COLUMNS).to_csv(path, index=False)
    return path


def scrape_reviews(
    url: str,
    output_path: str | Path = "data/reviews.csv",
    timeout: int = 10,
) -> list[dict[str, str]]:
    """Download and parse a permitted public review page, then save its reviews."""
    parsed_url = urlparse(url)
    if parsed_url.scheme not in {"http", "https"} or not parsed_url.netloc:
        raise ValueError("url must be a permitted public http(s) review page")

    try:
        response = requests.get(
            url,
            timeout=timeout,
            headers={"User-Agent": "fake-review-monitor/1.0 (public page parser)"},
        )
        response.raise_for_status()
    except requests.RequestException as exc:
        raise ReviewScraperError(f"Could not fetch review page: {exc}") from exc

    reviews = parse_reviews_html(response.text, source_url=url)
    save_reviews(reviews, output_path)
    return reviews


def run_demo(output_path: str | Path = "data/reviews.csv") -> list[dict[str, str]]:
    """Parse the built-in sample page without making a network request."""
    reviews = parse_reviews_html(SAMPLE_HTML)
    save_reviews(reviews, output_path)
    return reviews


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect reviews from a permitted public page.")
    parser.add_argument("url", nargs="?", help="Public http(s) review page URL")
    parser.add_argument("--output", default="data/reviews.csv", help="CSV output path")
    parser.add_argument("--demo", action="store_true", help="Parse built-in sample HTML")
    args = parser.parse_args()

    if args.demo:
        reviews = run_demo(args.output)
    elif args.url:
        reviews = scrape_reviews(args.url, args.output)
    else:
        parser.error("provide a public review page URL or use --demo")
    print(f"Saved {len(reviews)} reviews to {args.output}")


if __name__ == "__main__":
    main()