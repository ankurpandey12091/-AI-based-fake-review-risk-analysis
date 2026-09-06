"""
Hybrid fake-review detector combining multiple signals:

1. Traditional ML (Phase 6: TF-IDF + Logistic Regression)
2. Semantic NLP (Phase 7: Sentence Transformers)
3. Heuristic rules (suspicious language patterns)
"""

from __future__ import annotations

import logging
import re
from typing import Any

import joblib

from analyzer.config import (
    CONTAINS_URL,
    PROMOTIONAL_KEYWORDS,
    PREDICTION_THRESHOLD,
)
from models.predict import predict_review
from models.semantic_predict import predict_semantic_review


# Configure logging
logger = logging.getLogger(__name__)


# ============================================================
# HYBRID MODEL WEIGHTS
# ============================================================
#
# Phase 6 performed better than Phase 7, so ML gets the
# highest weight.
#
TRADITIONAL_ML_WEIGHT = 0.50
SEMANTIC_WEIGHT = 0.20
HEURISTIC_WEIGHT = 0.30


# Additional score added when multiple strong suspicious
# signals are detected.
STRONG_SIGNAL_BONUS = 0.15


# ============================================================
# MAIN ANALYSIS FUNCTION
# ============================================================

def analyze_review(review_text: str) -> dict[str, Any]:
    """
    Analyze a review using:
        - Traditional ML
        - Semantic NLP
        - Heuristic rules

    Returns:
        {
            "prediction": "fake" or "genuine",
            "fake_probability": 0.0-1.0,
            "ml_probability": 0.0-1.0,
            "semantic_probability": 0.0-1.0,
            "heuristic_score": 0.0-1.0,
            "risk_level": "low"/"medium"/"high",
            "reasons": [...]
        }
    """

    if not isinstance(review_text, str) or not review_text.strip():
        raise ValueError("review_text must be a non-empty string")

    review_text = review_text.strip()

    try:
        # ====================================================
        # 1. PHASE 6 - TRADITIONAL ML
        # ====================================================

        try:
            ml_result = predict_review(review_text)

            ml_probability = float(
                ml_result.get("fake_probability", 0.5)
            )

        except Exception as exc:
            logger.warning("ML prediction failed: %s", exc)
            ml_probability = 0.5

        # Ensure valid probability
        ml_probability = _clamp(ml_probability)


        # ====================================================
        # 2. PHASE 7 - SEMANTIC MODEL
        # ====================================================

        try:
            semantic_result = predict_semantic_review(review_text)

            semantic_probability = float(
                semantic_result.get("fake_probability", 0.5)
            )

        except Exception as exc:
            logger.warning("Semantic prediction failed: %s", exc)
            semantic_probability = 0.5

        # Ensure valid probability
        semantic_probability = _clamp(semantic_probability)


        # ====================================================
        # 3. HEURISTIC ANALYSIS
        # ====================================================

        heuristic_score, heuristic_reasons = (
            _calculate_heuristic_score(review_text)
        )


        # ====================================================
        # 4. INITIAL HYBRID SCORE
        # ====================================================

        hybrid_probability = (
            TRADITIONAL_ML_WEIGHT * ml_probability
            + SEMANTIC_WEIGHT * semantic_probability
            + HEURISTIC_WEIGHT * heuristic_score
        )

        hybrid_probability = _clamp(hybrid_probability)


        # ====================================================
        # 5. STRONG SUSPICIOUS SIGNAL ADJUSTMENT
        # ====================================================

        strong_signal_count = _count_strong_signals(
            heuristic_reasons
        )

        if (
            heuristic_score >= 0.65
            and strong_signal_count >= 2
        ):
            hybrid_probability += STRONG_SIGNAL_BONUS

            heuristic_reasons.append(
                "Multiple strong suspicious signals detected"
            )

        hybrid_probability = _clamp(hybrid_probability)


        # ====================================================
        # 6. FINAL PREDICTION
        # ====================================================

        prediction = (
            "fake"
            if hybrid_probability >= PREDICTION_THRESHOLD
            else "genuine"
        )


        # ====================================================
        # 7. RISK LEVEL
        # ====================================================

        risk_level = _get_risk_level(
            hybrid_probability
        )


        # ====================================================
        # 8. HUMAN-READABLE REASONS
        # ====================================================

        reasons = _generate_reasons(
            review_text=review_text,
            ml_probability=ml_probability,
            semantic_probability=semantic_probability,
            heuristic_score=heuristic_score,
            hybrid_probability=hybrid_probability,
            heuristic_reasons=heuristic_reasons,
        )


        # ====================================================
        # 9. RETURN RESULT
        # ====================================================

        return {
            "prediction": prediction,
            "fake_probability": round(
                hybrid_probability, 4
            ),
            "ml_probability": round(
                ml_probability, 4
            ),
            "semantic_probability": round(
                semantic_probability, 4
            ),
            "heuristic_score": round(
                heuristic_score, 4
            ),
            "risk_level": risk_level,
            "reasons": reasons,
        }


    except Exception as exc:
        logger.exception(
            "Hybrid analysis error"
        )
        raise


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def _clamp(value: float) -> float:
    """Keep a value between 0 and 1."""

    return max(
        0.0,
        min(1.0, float(value))
    )


def _calculate_heuristic_score(
    review_text: str,
) -> tuple[float, list[str]]:
    """
    Calculate a suspicious-language score.

    Returns:
        (score, reasons)
    """

    if not review_text.strip():
        return 0.0, []

    reasons: list[str] = []
    score_components: list[float] = []


    # ========================================================
    # EXCESSIVE EXCLAMATION MARKS
    # ========================================================

    exclamation_ratio = (
        review_text.count("!")
        / max(len(review_text), 1)
    )

    if exclamation_ratio > 0.15:
        score_components.append(
            min(
                1.0,
                exclamation_ratio / 0.20
            )
        )

        reasons.append(
            "Excessive exclamation marks"
        )


    # ========================================================
    # EXCESSIVE QUESTION MARKS
    # ========================================================

    question_ratio = (
        review_text.count("?")
        / max(len(review_text), 1)
    )

    if question_ratio > 0.10:
        score_components.append(
            min(
                1.0,
                question_ratio / 0.15
            )
        )

        reasons.append(
            "Excessive question marks"
        )


    # ========================================================
    # EXCESSIVE CAPITAL LETTERS
    # ========================================================

    alpha_chars = [
        char
        for char in review_text
        if char.isalpha()
    ]

    if alpha_chars:

        capital_ratio = (
            sum(
                1
                for char in alpha_chars
                if char.isupper()
            )
            / len(alpha_chars)
        )

        if capital_ratio > 0.30:

            score_components.append(
                min(
                    1.0,
                    (
                        capital_ratio - 0.30
                    ) / 0.40,
                )
            )

            reasons.append(
                "Excessive capital letters"
            )


    # ========================================================
    # VERY SHORT REVIEW
    # ========================================================

    if len(review_text.split()) < 5:

        score_components.append(0.30)

        reasons.append(
            "Very short review"
        )


    # ========================================================
    # PROMOTIONAL LANGUAGE
    # ========================================================

    text_lower = review_text.lower()

    matched_promotional = [
        phrase
        for phrase in PROMOTIONAL_KEYWORDS
        if phrase.lower() in text_lower
    ]

    if matched_promotional:

        promotional_score = min(
            1.0,
            len(matched_promotional) * 0.20
        )

        score_components.append(
            promotional_score
        )

        if len(matched_promotional) >= 2:

            reasons.append(
                "Excessive promotional language"
            )

        else:

            reasons.append(
                "Contains promotional language"
            )


    # ========================================================
    # URL DETECTION
    # ========================================================

    if re.search(
        CONTAINS_URL,
        review_text,
        re.IGNORECASE,
    ):

        score_components.append(0.40)

        reasons.append(
            "Contains URL or suspicious link"
        )


    # ========================================================
    # REPEATED WORD DETECTION
    # ========================================================

    words = re.findall(
        r"\b[a-zA-Z0-9]+\b",
        text_lower,
    )

    if words:

        word_frequency: dict[str, int] = {}

        for word in words:

            if len(word) <= 2:
                continue

            word_frequency[word] = (
                word_frequency.get(word, 0) + 1
            )

        if word_frequency:

            max_frequency = max(
                word_frequency.values()
            )

            if max_frequency >= 3:

                repeated_score = min(
                    0.50,
                    (
                        max_frequency - 1
                    ) / max(len(words), 1),
                )

                score_components.append(
                    repeated_score
                )

                reasons.append(
                    "Repeated words detected"
                )


    # ========================================================
    # REPEATED PUNCTUATION
    # ========================================================

    punctuation_score = 0.0

    punctuation_score += (
        review_text.count("!!!") * 0.20
    )

    punctuation_score += (
        review_text.count("???") * 0.20
    )

    punctuation_score = min(
        0.50,
        punctuation_score
    )

    if punctuation_score > 0:

        score_components.append(
            punctuation_score
        )

        reasons.append(
            "Repeated punctuation patterns detected"
        )


    # ========================================================
    # FINAL HEURISTIC SCORE
    # ========================================================

    if score_components:

        heuristic_score = (
            sum(score_components)
            / len(score_components)
        )

    else:

        heuristic_score = 0.0


    return (
        _clamp(heuristic_score),
        reasons,
    )


def _count_strong_signals(
    heuristic_reasons: list[str],
) -> int:
    """
    Count strong suspicious heuristic signals.
    """

    strong_signals = {
        "Excessive exclamation marks",
        "Excessive question marks",
        "Excessive capital letters",
        "Excessive promotional language",
        "Repeated punctuation patterns detected",
        "Repeated words detected",
        "Contains URL or suspicious link",
    }

    return sum(
        1
        for reason in heuristic_reasons
        if reason in strong_signals
    )


def _get_risk_level(
    probability: float,
) -> str:
    """
    Convert probability to:
        low
        medium
        high
    """

    if probability >= 0.70:
        return "high"

    if probability >= 0.40:
        return "medium"

    return "low"


def _generate_reasons(
    review_text: str,
    ml_probability: float,
    semantic_probability: float,
    heuristic_score: float,
    hybrid_probability: float,
    heuristic_reasons: list[str],
) -> list[str]:
    """
    Generate final human-readable explanations.
    """

    reasons: list[str] = []


    # ========================================================
    # MODEL SIGNAL REASONS
    # ========================================================

    if ml_probability >= 0.70:

        reasons.append(
            "High traditional ML fake probability"
        )

    elif ml_probability >= 0.55:

        reasons.append(
            "Moderate traditional ML fake probability"
        )


    if semantic_probability >= 0.70:

        reasons.append(
            "High semantic fake probability"
        )

    elif semantic_probability >= 0.55:

        reasons.append(
            "Moderate semantic fake probability"
        )


    # ========================================================
    # HEURISTIC REASONS
    # ========================================================

    for reason in heuristic_reasons:

        if reason not in reasons:

            reasons.append(reason)


    # ========================================================
    # GENERAL RESULT
    # ========================================================

    if not reasons:

        if hybrid_probability >= 0.70:

            reasons.append(
                "Multiple suspicious signals combined"
            )

        elif hybrid_probability >= 0.50:

            reasons.append(
                "Mixed signals suggest possible fake review"
            )

        else:

            reasons.append(
                "Review appears genuine based on analysis"
            )


    # ========================================================
    # REMOVE DUPLICATES
    # ========================================================

    unique_reasons = []

    for reason in reasons:

        if reason not in unique_reasons:

            unique_reasons.append(reason)


    return unique_reasons[:5]