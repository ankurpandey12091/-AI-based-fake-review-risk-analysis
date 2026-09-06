# Fake Review Monitor

This project collects public review data, cleans review text, and provides a machine-learning pipeline for detecting fake reviews using a hybrid multi-signal approach (Phase 6 TF-IDF + Logistic Regression, Phase 7 Semantic Embeddings, and Linguistic Heuristics).

---

## Deploying to Render (Manual Setup Guide)

Follow this step-by-step checklist to manually configure and deploy this application on [Render](https://dashboard.render.com).

### 1. Create Web Service
1. Log in to your [Render Dashboard](https://dashboard.render.com/).
2. Click **New +** in the top navigation bar and select **Web Service**.
3. Under **Connect a repository**, select:
   `ankurpandey12091/-AI-based-fake-review-risk-analysis`
4. Choose **Public Git repository** if needed, or connect via your GitHub account.

---

### 2. Configure Build & Deploy Settings

In the service setup page, fill in the following exact settings:

| Setting | Value | Why it is needed |
| :--- | :--- | :--- |
| **Name** | `fake-review-monitor` | Name of your web service on Render |
| **Region** | *Any* (e.g. `Oregon (US West)` or `Frankfurt (EU)`) | Server hosting location |
| **Branch** | `main` | Production branch to deploy from |
| **Root Directory** | *(Leave blank OR set to `.`) | The repository includes root `app.py` and `wsgi.py` wrappers |
| **Runtime** | `Python 3` | Native Python environment |
| **Build Command** | `pip install -r requirements.txt && python -m nltk.downloader punkt punkt_tab stopwords wordnet` | Installs dependencies and downloads required NLTK tokenizers |
| **Start Command** | `gunicorn --bind 0.0.0.0:$PORT --workers 1 --threads 2 app:app` | Binds to Render's dynamic `$PORT` using 1 worker to save RAM |
| **Instance Type** | `Free` (or `Starter`) | 512 MB RAM Free tier |

> **Free Tier Tip (Fastest & Lightest):**
> If you want builds under 45 seconds that consume < 50 MB RAM, use this Build Command instead:
> ```bash
> pip install -r requirements-light.txt && python -m nltk.downloader punkt punkt_tab stopwords wordnet
> ```

---

### 3. Configure Environment Variables (`.env`)

In the **Environment** section of your Web Service, configure these variables:

#### Option A: Quick Import (Recommended)
Click **"Add from .env"** and paste the following block directly:
```env
PYTHON_VERSION=3.11.9
PORT=10000
PYTHONUNBUFFERED=1
DISABLE_SEMANTIC_MODEL=true
FLASK_ENV=production
WEB_CONCURRENCY=1
```

#### Option B: Manual Key-Value Input
Click **"Add Environment Variable"** and enter each row individually:

| Key | Value | Description |
| :--- | :--- | :--- |
| `PYTHON_VERSION` | `3.11.9` | Locks the build environment to Python 3.11 for wheel compatibility |
| `PORT` | `10000` | Fallback port (Render automatically provides its own `$PORT`) |
| `PYTHONUNBUFFERED` | `1` | Forces real-time log output in Render's log console |
| `DISABLE_SEMANTIC_MODEL` | `true` | **Critical for Free Tier:** Disables loading heavy 92MB PyTorch weights, preventing Status 137 OOM crashes. (Set to `false` only if on a 1GB+ paid tier) |
| `FLASK_ENV` | `production` | Sets Flask to production mode |
| `WEB_CONCURRENCY` | `1` | Limits Gunicorn to 1 worker to conserve memory |

---

### 4. Health Check Path (Optional)

In **Settings** -> **Health Check Path**, set:
```text
/api/health
```
Render will periodically check this endpoint to ensure the service is running and healthy.

---

### 5. Deploy & Verify

1. Click **Create Web Service** (or **Manual Deploy** -> **Deploy latest commit**).
2. Once the build finishes and the logs display `Listening at: http://0.0.0.0:10000`, click your service URL (e.g. `https://fake-review-monitor.onrender.com`).
3. Test a review in the input box to verify the analyzer returns results.

---

## Local Development & ML Pipeline

For details on dataset labeling, model training, evaluation, Phase 6 (TF-IDF), and Phase 7 (Sentence Transformers), see [`fake-review-monitor/README.md`](fake-review-monitor/README.md).
