from __future__ import annotations

import logging
import os
import sys
from pathlib import Path

# Ensure the directory containing app.py is in sys.path
BASE_DIR = Path(__file__).resolve().parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from flask import Flask, jsonify, render_template, request

from analyzer.hybrid_detector import analyze_review


# ------------------------------------------------------------
# Application setup
# ------------------------------------------------------------

app = Flask(__name__)

# Limit incoming request size to avoid unnecessarily large payloads.
app.config["MAX_CONTENT_LENGTH"] = 64 * 1024


# ------------------------------------------------------------
# Logging
# ------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)

logger = logging.getLogger(__name__)


# ------------------------------------------------------------
# Frontend
# ------------------------------------------------------------

@app.get("/")
def home():
    """Render the web application."""
    return render_template("index.html")


# ------------------------------------------------------------
# Health check
# ------------------------------------------------------------

@app.get("/api/health")
def health():
    """Return service health information."""
    return jsonify(
        {
            "success": True,
            "status": "ok",
            "service": "AI Fake Review Monitor",
        }
    )


# ------------------------------------------------------------
# Review analysis API
# ------------------------------------------------------------

@app.post("/api/analyze")
def analyze():
    """Analyze one review and return a JSON response."""

    # Ensure the request body is JSON.
    if not request.is_json:
        return (
            jsonify(
                {
                    "success": False,
                    "error": "Request must contain JSON",
                }
            ),
            400,
        )

    # Parse JSON safely.
    data = request.get_json(silent=True)

    if not isinstance(data, dict):
        return (
            jsonify(
                {
                    "success": False,
                    "error": "Invalid JSON body",
                }
            ),
            400,
        )

    # Get review text.
    review = data.get("review")

    if not isinstance(review, str):
        return (
            jsonify(
                {
                    "success": False,
                    "error": "review must be a string",
                }
            ),
            400,
        )

    review = review.strip()

    if not review:
        return (
            jsonify(
                {
                    "success": False,
                    "error": "review cannot be empty",
                }
            ),
            400,
        )

    if len(review) > 5000:
        return (
            jsonify(
                {
                    "success": False,
                    "error": "review is too long (maximum 5000 characters)",
                }
            ),
            400,
        )

    # Run the actual hybrid detector.
    try:
        logger.info(
            "Analyzing review (%d characters)",
            len(review),
        )

        result = analyze_review(review)

        return jsonify(
            {
                "success": True,
                "result": result,
            }
        )

    except Exception:
        logger.exception("Review analysis failed")

        # IMPORTANT:
        # Never return an HTML error page from the API.
        return (
            jsonify(
                {
                    "success": False,
                    "error": "Unable to analyze review. Check Render logs for details.",
                }
            ),
            500,
        )


# ------------------------------------------------------------
# API error handlers
# ------------------------------------------------------------

@app.errorhandler(400)
def bad_request(_error):
    """Return JSON for bad API requests."""
    return (
        jsonify(
            {
                "success": False,
                "error": "Bad request",
            }
        ),
        400,
    )


@app.errorhandler(404)
def not_found(_error):
    """Return JSON for missing API routes."""
    return (
        jsonify(
            {
                "success": False,
                "error": "Endpoint not found",
            }
        ),
        404,
    )


@app.errorhandler(413)
def request_too_large(_error):
    """Return JSON when request payload is too large."""
    return (
        jsonify(
            {
                "success": False,
                "error": "Request payload is too large",
            }
        ),
        413,
    )


@app.errorhandler(500)
def server_error(_error):
    """Return JSON for unexpected application errors."""
    return (
        jsonify(
            {
                "success": False,
                "error": "Internal server error",
            }
        ),
        500,
    )


# ------------------------------------------------------------
# Local development
# ------------------------------------------------------------

if __name__ == "__main__":
    port = int(os.environ.get("PORT", "10000"))

    app.run(
        host="0.0.0.0",
        port=port,
        debug=False,
    )