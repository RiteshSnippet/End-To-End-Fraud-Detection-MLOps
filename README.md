# End-To-End-Fraud-Detection-MLOps

A production-oriented machine learning system for detecting fraudulent financial transactions, built with a full MLOps stack: data versioning, experiment tracking, a training pipeline, a Flask inference service, and Docker packaging.

## Overview

Fraudulent transactions are rare relative to legitimate ones, which makes them hard to catch and easy to overfit to. This project treats fraud detection as a full lifecycle problem rather than a single notebook: it covers data preparation, model training and evaluation, experiment tracking, versioning, and a served prediction endpoint, with the aim of making the model reproducible and deployable rather than just accurate on a held-out set.

## Key Features

- **Reproducible pipeline** for data ingestion, preprocessing, model training and model evaluation
- **Data version control with DVC** to keep datasets and pipeline stages in sync with code
- **Experiment tracking via MLflow and DagsHub** for comparing model runs and metrics
- **Multiple model backends** (scikit-learn, XGBoost, CatBoost) for benchmarking against the same feature set
- **Flask web application** with a simple form-based UI for single-transaction predictions
- **Containerized deployment** with a lightweight Docker image served by Gunicorn
- **Automated tests** (pytest) covering the trained model and prediction path

## Tech Stack

| Layer | Tools |
|---|---|
| Language | Python 3.12 |
| Data & ML | pandas, NumPy, scikit-learn, XGBoost, CatBoost |
| Experiment tracking | MLflow, DagsHub |
| Data versioning | DVC |
| Serving | Flask, Gunicorn |
| Visualization (EDA) | Matplotlib, Seaborn |
| Testing | pytest |
| Containerization | Docker |

## Project Structure

```
End-To-End-Fraud-Detection-MLOps/
├── .github/
│   └── workflows/
│       └── cicd.yml          # CI/CD pipeline (test, build, push, deploy)
├── .dvc/                      # DVC configuration for data/pipeline versioning
├── artifacts/                 # Generated pipeline outputs (data splits, models)
├── catboost_info/             # CatBoost training logs
├── notebooks/                 # Exploratory data analysis and experimentation
├── src/
│   └── fraud_detection/       # Core package: ingestion, transformation, training, prediction pipeline
├── static/                    # Static assets for the Flask app
├── templates/                 # HTML templates (index, result, error pages)
├── tests/                     # Test suite
├── app.py                     # Flask application entry point
├── model_test.py              # Model validation script
├── Dockerfile                 # Container build definition
├── requirements.txt           # Python dependencies
├── setup.py                   # Package metadata and installation
└── template.py                # Project scaffolding script
```

## Model Input Features

The prediction pipeline accepts the following transaction attributes:

| Feature | Description |
|---|---|
| `type` | Transaction type (e.g., transfer, cash-out) |
| `amount` | Transaction amount |
| `oldbalanceOrg` / `newbalanceOrig` | Sender balance before and after the transaction |
| `oldbalanceDest` / `newbalanceDest` | Recipient balance before and after the transaction |
| `errorBalanceOrig` / `errorBalanceDest` | Derived balance discrepancy features |
| `hour` / `day` | Time of transaction |

The service returns a classification (fraudulent or legitimate) along with a confidence score.

## Getting Started

### Prerequisites

- Python 3.12 or later
- Git
- Docker (optional, for containerized deployment)

### Installation

```bash
git clone https://github.com/RiteshSnippet/End-To-End-Fraud-Detection-MLOps.git
cd End-To-End-Fraud-Detection-MLOps

python -m venv venv
source venv/bin/activate      # On Windows: venv\Scripts\activate

pip install -r requirements.txt
```

### Running the Training Pipeline

Pipeline stages (ingestion, transformation, training) are defined under `src/fraud_detection`. Data and outputs are tracked with DVC:

```bash
dvc pull        # Fetch versioned data/artifacts, if a remote is configured
dvc repro        # Run the pipeline stages defined in dvc.yaml
```

Track experiments in MLflow:

```bash
mlflow ui
```

### Running the Web Application

```bash
python app.py
```

The app runs at `http://localhost:5000` with a form to submit transaction details and receive a fraud prediction.

### Running Tests

```bash
pytest
```

### Docker Deployment

Build and run the containerized service:

```bash
docker build -t fraud-detection-mlops .
docker run -p 5000:5000 fraud-detection-mlops
```

The container serves the Flask app through Gunicorn on port 5000.

## CI/CD

The pipeline in `.github/workflows/cicd.yml` runs on every push and pull request to `main`:

1. **Test** — installs dependencies and runs the `pytest` suite.
2. **Build & Push** (on `main` only) — builds the Docker image and pushes it to GitHub Container Registry (`ghcr.io`), tagged with both `latest` and the commit SHA.
3. **Deploy** (on `main` only) — placeholder stage to trigger your hosting provider's deployment (server SSH, cloud run update, etc.); add your target-specific step here.

No additional secrets are required for the build/push stage since it authenticates with the built-in `GITHUB_TOKEN`. Add secrets under **Settings → Secrets and variables → Actions** if the deploy stage needs credentials for your hosting target.

## Roadmap

- CI/CD pipeline for automated testing and deployment
- Model monitoring and drift detection in production
- REST API endpoint alongside the existing web UI
- Expanded model explainability (e.g., SHAP) for individual predictions

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.

## Author

**Ritesh Kumar Behera**
GitHub: [@RiteshSnippet](https://github.com/RiteshSnippet)

## Contributing

Issues and pull requests are welcome. For significant changes, open an issue first to discuss what you would like to change.