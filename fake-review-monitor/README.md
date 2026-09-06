# Fake Review Monitor

This project collects public review data, cleans review text, and provides a
machine-learning pipeline for experimenting with fake-review detection.

## Dataset labeling

`models/prepare_dataset.py` creates `data/labeled_reviews.csv`, a separate
hand-labeled demo dataset for testing the ML pipeline. It contains:

- `review_text`
- `processed_text`
- `label`, where genuine reviews are `0` and fake reviews are `1`

The demo labels are manually authored examples and are not labels inferred from
the original scraped data. The original `data/cleaned_reviews.csv` remains
unchanged.

Real-world training requires manually verified labels from domain reviewers or
a properly labeled public dataset. Demo labels must not be treated as a
validated measure of production model quality.

Run the dataset preparation script with:

```powershell
python -m models.prepare_dataset
```

Run the tests with:

```powershell
python -m pytest
```

## Phase 4B baseline model

The baseline uses `processed_text` from `data/labeled_reviews.csv` with an
80/20 stratified split (`random_state=42`). An sklearn Pipeline combines
`TfidfVectorizer` and `LogisticRegression`. Run it with:

```powershell
python models/train_model.py
```

The pipeline is saved to `models/fake_review_model.pkl`. Evaluation results are
for the 24 manually authored demo samples only and are not representative of
real-world performance. Production evaluation requires a substantially larger,
manually verified dataset.

## Phase 5 model comparison

`models/compare_models.py` compares Logistic Regression, Multinomial Naive
Bayes, and Random Forest using the same TF-IDF features and the same 80/20
stratified split. Comparing several traditional algorithms helps identify a
reasonable baseline for this text classification task before selecting one for
future work. Results are saved to `data/model_comparison.csv`.

The comparison uses only 24 manually authored demo reviews, so it validates the
pipeline but cannot establish real-world model superiority. A larger,
independently verified dataset is required for that conclusion.

## Phase 6 real-world evaluation

The 24-row demo dataset is intentionally small and manually authored. It is
useful for demonstrating the pipeline, but it cannot estimate how the system
will perform on real e-commerce reviews. Real evaluation requires a larger,
manually verified or properly labeled public dataset; this project never
invents or randomly assigns labels.

Place the labeled source file at `data/raw_reviews.csv` with `review_text` and
`label` columns, where `0` means genuine and `1` means fake or suspicious. If
the source uses different names, pass the exact names explicitly:

```powershell
python models/prepare_real_dataset.py --review-column review --label-column is_fake
python models/evaluate_models.py
```

Preparation removes exact duplicate reviews, reuses the shared text cleaner,
and writes `data/real_labeled_reviews.csv`. Evaluation uses a 70% training,
15% validation, and 15% test split. TF-IDF is fitted inside each model
pipeline on training data only. Stratified five-fold cross-validation is run
only on the training partition. Models are selected by validation F1-score,
with ROC-AUC as the tie-breaker, then the selected model is retrained on the
training plus validation partitions. The test partition remains untouched
until the single final evaluation.

F1-score is important because fake-review detection must balance finding
suspicious reviews (recall) against avoiding unjustified flags (precision).
Dataset size, sampling bias, domain differences, label disagreement, and
label quality can all make the reported metrics misleading. Model performance
depends heavily on the quality, size, representativeness, and reliability of
the labels.

Run Phase 6 with:

```powershell
python models/prepare_real_dataset.py
python models/evaluate_models.py
```

The final model is saved to `models/final_fake_review_model.pkl`; prediction
uses it automatically when present and otherwise retains the demo-model
fallback. A fake-review prediction is a risk signal, not proof that a reviewer
committed fraud.

## Phase 7 semantic NLP model

Phase 7 introduces an advanced semantic model using **Sentence Transformers**
to create rich semantic representations of reviews. Instead of sparse TF-IDF
features, the model generates dense embeddings using a pretrained transformer
model (`sentence-transformers/all-MiniLM-L6-v2`), which captures deeper
semantic meaning and contextual nuance.

**Key features:**

- **Embedding model:** `sentence-transformers/all-MiniLM-L6-v2` generates 384-dimensional embeddings
- **Classifier:** Logistic Regression trained on semantic embeddings
- **Fair comparison:** Uses identical 70/15/15 train/validation/test split with `random_state=42` and stratification
- **Evaluation:** Model selection on validation data, single final evaluation on untouched test set
- **Reproducibility:** Embedding model, classifier, and configuration are saved for consistent inference

**Important limitations:**

- Sentence Transformers provide semantic representations but are **NOT** proof that a review is fake
- Embeddings capture linguistic patterns, linguistic sophistication, and semantic coherence—factors that may correlate with authenticity but do not determine it
- Domain transfer: A model trained on hotel reviews may perform differently on product reviews
- Label quality: Model performance depends entirely on the quality and representativeness of training labels
- Adversarial reviews: Sophisticated fake reviews designed to mimic genuine reviews may evade both TF-IDF and semantic models
- Context limitations: Embeddings cannot detect sophisticated fraud schemes that require external knowledge (timing, reviewer history, purchase verification)

**Running Phase 7:**

First, prepare the real dataset (if not already done):

```powershell
python models/prepare_real_dataset.py
```

Then train and evaluate the semantic model:

```powershell
python models/semantic_model.py
```

This creates:

- `data/phase7_results.csv` – test metrics and model details
- `models/semantic_embedder.pkl` – saved Sentence Transformer embedding model
- `models/semantic_classifier.pkl` – trained Logistic Regression classifier
- `models/semantic_config.json` – configuration for reproducible inference

**Making predictions with Phase 7:**

```python
from models.semantic_predict import predict_semantic_review

result = predict_semantic_review("The hotel was fantastic and wonderful!")
print(result)
# Output: {"prediction": "genuine", "fake_probability": 0.23}
```

**Phase 6 vs Phase 7 comparison:**

Run both Phase 6 and Phase 7 and compare results:

```powershell
python models/evaluate_models.py
python models/semantic_model.py
```

Phase 6 uses TF-IDF + Logistic Regression (sparse, interpretable features).
Phase 7 uses Sentence Transformers + Logistic Regression (dense, semantic embeddings).

Different model families may excel on different characteristics:
- TF-IDF captures explicit keywords and writing patterns
- Sentence Transformers capture semantic coherence and linguistic sophistication

Neither is universally superior; performance depends on the dataset, label quality, and domain.