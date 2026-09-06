"""
Configuration for the hybrid fake-review detector.

Weights control how much each signal contributes to the final hybrid score:
- traditional_ml_weight: TF-IDF + Logistic Regression from Phase 6
- semantic_weight: Sentence Transformer embeddings from Phase 7
- heuristic_weight: Rule-based suspicious language patterns

Total weights should sum to 1.0 for normalized probability.
"""

# Hybrid scoring weights
TRADITIONAL_ML_WEIGHT = 0.50  # 50% - Phase 6 TF-IDF model
SEMANTIC_WEIGHT = 0.25  # 25% - Phase 7 semantic model
HEURISTIC_WEIGHT = 0.25  # 25% - Heuristic rules

# Prediction threshold: if hybrid score >= threshold, predict "fake"
PREDICTION_THRESHOLD = 0.50

# Risk level boundaries
RISK_LEVEL_LOW = 0.39
RISK_LEVEL_MEDIUM = 0.69
RISK_LEVEL_HIGH = 1.00

# Heuristic thresholds
MAX_EXCLAMATION_RATIO = 0.15  # Ratio of ! to total length
MAX_QUESTION_RATIO = 0.10  # Ratio of ? to total length
MAX_CAPITAL_RATIO = 0.30  # Ratio of capital letters to total length
MIN_REVIEW_LENGTH = 20  # Minimum acceptable review length
MAX_DUPLICATED_WORDS = 0.20  # Ratio of repeated words

# Promotional keywords and phrases
PROMOTIONAL_KEYWORDS = [
    "buy now",
    "must buy",
    "best product ever",
    "five stars",
    "amazing",
    "perfect",
    "greatest",
    "incredible",
    "unbelievable",
    "highly recommend",
    "purchase",
    "order now",
    "limited time",
    "hurry",
    "don't miss",
]

# URLs pattern
CONTAINS_URL = r"https?://|www\.|\.com|\.org|\.net"
