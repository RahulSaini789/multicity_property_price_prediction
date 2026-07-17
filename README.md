# PropML — Multi-City Property Price Prediction

**Production-grade MLOps pipeline for real estate price prediction across 13 Indian cities.** From raw scraping to live API serving — 7 layers, end-to-end.

![Python](https://img.shields.io/badge/Python-3.11-3b82f6?style=for-the-badge)
![XGBoost](https://img.shields.io/badge/XGBoost-2.0-10b981?style=flat-square)
![FastAPI](https://img.shields.io/badge/FastAPI-0.110-009688?style=flat-square&logo=fastapi&logoColor=white)
![MLflow](https://img.shields.io/badge/MLflow-2.11-0194E2?style=flat-square)
![Docker](https://img.shields.io/badge/Docker-Multi--Stage-2496ED?style=flat-square&logo=docker&logoColor=white)
![CI/CD](https://img.shields.io/badge/CI%2FCD-GitHub%20Actions-2088FF?style=flat-square&logo=githubactions&logoColor=white)

[Live Demo](#-live-demo) · [API Docs](#-api-reference) · [Architecture](#-architecture) · [Quick Start](#-quick-start) 
---

## What Is This?

PropML is a **full production MLOps project**, not a Jupyter notebook. It predicts property prices across 13 Indian cities (Gurgaon, Noida, Mumbai, Bangalore, Hyderabad, Pune, Jaipur, Kota, Chandigarh and more) using an XGBoost + LightGBM + CatBoost ensemble trained on 29,821 real listings from MagicBricks.

**What makes it production-grade:**

- City-aware data cleaning pipeline (grouped IQR per city × property type × tier)
- 53 engineered features, including K-Fold target-encoded locality and city-tier signals
- Bayesian hyperparameter tuning with Optuna (100 trials), plus a secondary Optuna search to tune ensemble weights
- SHAP explainability on every prediction (top-3 drivers returned in the API response)
- HuggingFace Hub model registry (`Dumdigi/multicity`), pulled at startup on Render
- Multi-stage Docker build, deployed on Render
- 6-stage GitHub Actions CI/CD with a model quality gate

---

## Live Demo

| Service | URL | Description |
|---|---|---|
| **Website** | [rahulsaini789.github.io/multicity_property_price_prediction](https://rahulsaini789.github.io/multicity_property_price_prediction/) | Live price estimator, market intelligence dashboard, MLOps architecture walkthrough |
| **API** | [multicity-property-price-prediction-2.onrender.com](https://multicity-property-price-prediction-2.onrender.com) | REST API (may sleep on free tier) |
| **API Docs** | [/docs](https://multicity-property-price-prediction-2.onrender.com/docs) | Swagger UI — try it live |
| **Model Hub** | [huggingface.co/Dumdigi/multicity](https://huggingface.co/Dumdigi/multicity) | Trained model artifacts |

> **Note:** Free tier on Render sleeps after inactivity. First request after a cold start can take 30-60s to wake up.

---

## Model Performance

| Metric | Value |
|---|---|
| CV MAPE | **20.31%** |
| CV R² | **0.9321** |
| Training rows | 29,821 |
| Cities | 13 |
| Optuna trials | 100 |
| API latency (p50) | 123ms |
| Production gate | MAPE ≤ 22%, R² ≥ 0.82 |

**Best/worst city MAPE:**

| City | MAPE | Notes |
|---|---|---|
| Pune | 15.51% | Best-performing city |
| Hyderabad | 17.14% | |
| Mumbai | 17.33% | |
| Bangalore | 18.96% | |
| Noida | 20.72% | 1,500 listings |
| Gurgaon | 21.13% | 3,417 listings |
| Jaipur | 28.88% | 35% plots (higher variance) |
| Kota | 24.26% | Coaching-hub micro-market |

**Top SHAP Features (Global Importance):**

```
city_tier_num   ████████████████████ 48.7%
is_plot         █████████            23.2%
bhk_x_city      ███                   6.4%
locality_enc    ██                    5.8%
city_encoded    █                     3.6%
```

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                        DATA LAYER                                    │
│  MagicBricks (Playwright scraper, 13 cities) ──► DVC pipeline        │
└─────────────────────────────────────────────────────────────────────┘
                                 │
                                 ▼
┌─────────────────────────────────────────────────────────────────────┐
│                     VALIDATION + CLEANING                            │
│  35 schema checks ──► Grouped IQR (city × type × tier) ──► master    │
└─────────────────────────────────────────────────────────────────────┘
                                 │
                                 ▼
┌─────────────────────────────────────────────────────────────────────┐
│                    FEATURE ENGINEERING (53 features)                 │
│  city_tier_num · is_plot · bhk_x_city · locality_enc (K-Fold target  │
│  encoding) · city_encoded · amenity_score · log1p(price/area) ...    │
└─────────────────────────────────────────────────────────────────────┘
                                 │
                                 ▼
┌─────────────────────────────────────────────────────────────────────┐
│                          TRAINING LAYER                               │
│  XGBoost + LightGBM + CatBoost ──► Optuna (100 trials, 5-fold CV)    │
│  Ensemble weights (0.483 / 0.159 / 0.358) tuned via secondary Optuna │
│  MLflow tracking ──► Quality gate: MAPE ≤ 22% AND R² ≥ 0.82          │
└─────────────────────────────────────────────────────────────────────┘
                                 │
                                 ▼
┌─────────────────────────────────────────────────────────────────────┐
│                          SERVING LAYER                                │
│  FastAPI + Docker ──► model pulled from HuggingFace Hub at startup   │
│  POST /predict → validation → feature vector → ensemble predict      │
│  → SHAP KernelExplainer (top-3 drivers) → confidence interval        │
│  GET /health · GET /model-info                                       │
└─────────────────────────────────────────────────────────────────────┘
                                 │
                                 ▼
┌─────────────────────────────────────────────────────────────────────┐
│                          CI/CD LAYER                                  │
│  git push ──► lint ──► test (40+ cases) ──► model-gate ──► docker    │
│                        build ──► Render deploy ──► smoke-test        │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Quick Start

### Option 1 — Docker (Recommended)

```bash
git clone https://github.com/RahulSaini789/propml.git
cd propml

echo "DB_PASSWORD=propml_secure_123" > .env

docker-compose up --build

curl http://localhost:8000/health
```

### Option 2 — Local Development

```bash
git clone https://github.com/RahulSaini789/propml.git
cd propml
conda create -n propml python=3.11 -y
conda activate propml
pip install -r requirements.txt

dvc pull
dvc repro

uvicorn src.serving.main:app --host 0.0.0.0 --port 8000 --reload
```

---

## API Reference

### POST /predict

**Request:**

```json
{
  "city": "gurgaon",
  "property_type": "flat",
  "bedRoom": 3,
  "bathroom": 3,
  "balcony": 2,
  "area_sqft": 1800,
  "floor_pos": 10,
  "total_floors": 15,
  "amenity_score": 7.0
}
```

**Response:**

```json
{
  "prediction_cr": 1.82,
  "confidence_interval": {
    "low": 1.55,
    "high": 2.09
  },
  "price_per_sqft": 10111,
  "model_version": "propml-multicity/Production",
  "shap_top_features": [
    {"feature": "city_tier_num", "impact": 0.487, "direction": "positive"},
    {"feature": "is_plot", "impact": 0.232, "direction": "negative"},
    {"feature": "bhk_x_city", "impact": 0.064, "direction": "positive"}
  ],
  "latency_ms": 123
}
```

### GET /health

```json
{
  "status": "healthy",
  "model_version": "v4.0",
  "uptime_seconds": 3600
}
```

Full interactive docs: [/docs](https://multicity-property-price-prediction-2.onrender.com/docs)

---

## Tech Stack

| Layer | Technology | Why |
|---|---|---|
| Data Versioning | DVC | Git tracks code, DVC tracks data |
| Scraping | Playwright | Handles `__SERVER_PRELOADED_STATE__` JS-rendered pages |
| Validation | Great Expectations | 35 schema checks, city-specific rules |
| Cleaning | Pandas + Regex | Grouped IQR, city-aware pipeline |
| Features | scikit-learn | K-Fold target encoding, 53 features |
| Model | XGBoost + LightGBM + CatBoost | Ensemble beats any single learner across heterogeneous city markets |
| Tuning | Optuna (TPE) | Bayesian search for hyperparameters and ensemble weights |
| Explainability | SHAP | Per-prediction top-3 drivers |
| Model Registry | HuggingFace Hub | `Dumdigi/multicity`, pulled at Render startup |
| Serving | FastAPI + Uvicorn | Async, auto-docs, Pydantic validation |
| Containerization | Docker (multi-stage) | Lean production image |
| CI/CD | GitHub Actions | Lint → Test → Gate → Docker → Deploy → Smoke-test |

---

## Key Engineering Decisions

**Why an ensemble instead of a single model?** Property markets vary sharply by city tier (Mumbai luxury flats vs Jaipur plots vs Kota micro-market). XGBoost, LightGBM and CatBoost each capture slightly different structure; blending them (weights tuned via a secondary Optuna search) outperforms any single model across all 13 cities.

**Why Grouped IQR instead of Global IQR?** Global IQR on merged houses + flats + plots data artificially pulled the upper fence down, deleting a large share of valid luxury listings. Grouped IQR applies separate statistical bounds per city × property type × tier.

**Why K-Fold Target Encoding for locality?** Naive target encoding leaks the row's own price into its locality's mean. K-Fold encoding uses out-of-fold means, so no row sees its own target during encoding.

**Why MAPE over MSE as the primary metric?** MSE penalizes a large error on a luxury Mumbai flat far more than a proportionally similar error on a budget Kota flat. MAPE is scale-invariant: a stakeholder can read "within 20% of actual price" and understand it regardless of price level.

---

## Author

**Rahul Saini**

- B.Sc. Mathematics, University of Kota
- DSMP 2.0 Certification — CampusX
- Target: Data Scientist / MLOps Engineer

[LinkedIn](https://www.linkedin.com/in/rahul-saini-122321229/) · [GitHub](https://github.com/RahulSaini789)

---

## License

MIT License — see [LICENSE](https://github.com/RahulSaini789/propml/blob/main/LICENSE) for details.

---

**PropML** — Built with production engineering, not just model training.
*From raw scraped data across 13 cities to a live, explainable API — 7 layers, fully documented.*