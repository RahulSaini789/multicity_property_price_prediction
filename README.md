# PropML — Multi-City Property Price Prediction

Production-grade MLOps system for residential property price prediction
across 4 Indian cities (Gurgaon, Noida, Chandigarh, Kota).

## Live
- API: https://propml.onrender.com
- Frontend: https://YOUR_USERNAME.github.io/propml
- Model: https://huggingface.co/YOUR_USERNAME/propml-models

## Architecture
MagicBricks scraping → DVC pipeline → XGBoost+LGB+CatBoost ensemble
→ FastAPI on Render → GitHub Pages frontend

## Pipeline
| Layer | Description | Status |
|-------|-------------|--------|
| L1 | Git + Project Setup | ✅ |
| L2 | DVC Data Versioning | ✅ |
| L3 | MagicBricks Scraping | 🔲 |
| L4 | Data Validation | 🔲 |
| L5 | Data Cleaning | 🔲 |
| L6 | Feature Engineering | 🔲 |
| L7 | Model Training | 🔲 |
| L8 | FastAPI Serving | 🔲 |
| L9 | CI/CD + Testing | 🔲 |
| L10 | Monitoring + Drift | 🔲 |

## Local Setup
```bash
conda create -n propml python=3.11 -y
conda activate propml
pip install -r requirements.txt
playwright install chromium
dvc pull
```

## Metrics
| Metric | Value |
|--------|-------|
| CV MAPE | TBD |
| CV R² | TBD |
| API Latency | TBD |

## Tech Stack
Python 3.11 · XGBoost · LightGBM · CatBoost · Optuna · SHAP · MLflow ·
FastAPI · Docker · DVC · GitHub Actions · Render · HuggingFace Hub