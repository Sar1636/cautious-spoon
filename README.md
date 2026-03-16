# 🍕 EDT Model – Deployment Guide

**Estimated Delivery Time (EDT) Prediction System**  
Japan Pizza Market | Azure ML + FastAPI + GitHub Actions

---

## Table of Contents

1. [Architecture Overview](#architecture-overview)
2. [Prerequisites](#prerequisites)
3. [Local Setup](#local-setup)
4. [Dataset](#dataset)
5. [Training the Model Locally](#training-the-model-locally)
6. [Azure ML Setup](#azure-ml-setup)
7. [GitHub Actions CI/CD Setup](#github-actions-cicd-setup)
8. [Running the API Locally](#running-the-api-locally)
9. [Deploying the API to Azure](#deploying-the-api-to-azure)
10. [Making Prediction Requests](#making-prediction-requests)
11. [Scaling to New Markets](#scaling-to-new-markets)
12. [Troubleshooting](#troubleshooting)

---

## Architecture Overview

```
┌─────────────┐     push/PR      ┌──────────────────────────────────┐
│  Developer  │ ───────────────► │        GitHub Actions            │
└─────────────┘                  │  ① Lint & Tests                  │
                                 │  ② Submit Azure ML Pipeline      │
                                 │  ③ Register Model                │
                                 │  ④ Build & Push Docker image     │
                                 │  ⑤ Deploy to Azure ACI           │
                                 └──────────────┬───────────────────┘
                                                │
                         ┌──────────────────────▼──────────────────┐
                         │           Azure ML Workspace            │
                         │  ┌────────────┐   ┌──────────────────┐  │
                         │  │ Training   │   │  Model Registry  │  │
                         │  │ Compute    │──►│  edt-model-japan │  │
                         │  │ Cluster   │   │  version: 1,2,3… │  │
                         │  └────────────┘   └──────────────────┘  │
                         └─────────────────────────────────────────┘
                                                │
                         ┌──────────────────────▼──────────────────┐
                         │      Azure Container Instance           │
                         │         FastAPI  :8000                  │
                         │   POST /predict                         │
                         │   POST /predict/batch                   │
                         └─────────────────────────────────────────┘
```

---

## Prerequisites

| Tool               | Version | Purpose          |
| ------------------ | ------- | ---------------- |
| Python             | 3.10+   | Runtime          |
| Docker             | 24+     | Container builds |
| Azure CLI          | 2.50+   | Azure resources  |
| Git                | 2.40+   | Version control  |
| Azure Subscription | —       | Cloud resources  |

---

## Local Setup

```bash
# 1. Clone the repository
git clone https://github.com/<your-org>/edt-model.git
cd edt-model

# 2. Create virtual environment
python3 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Verify setup
python -c "import sklearn, pandas, fastapi; print('✅ All packages OK')"
```

---

## Dataset

The dataset `D:/Project_ML/japan_pizza_delivery.csv` contains **5,000 delivery orders** with these columns:

| Column              | Type    | Description                               |
| ------------------- | ------- | ----------------------------------------- |
| `order_id`          | str     | Unique order identifier (JP######)        |
| `store_id`          | str     | Store name/location                       |
| `order_hour`        | int     | Hour of order placement (9–22)            |
| `day_of_week`       | str     | Day name (Monday–Sunday)                  |
| `is_weekend`        | int     | 1 if Saturday/Sunday                      |
| `is_peak_hour`      | int     | 1 if 12–13 or 18–20                       |
| `weather_condition` | str     | clear / cloudy / rain / heavy_rain / snow |
| `traffic_level`     | str     | low / medium / high / very_high           |
| `vehicle_type`      | str     | bicycle / scooter / car                   |
| `distance_km`       | float   | Delivery distance in kilometres           |
| `num_items`         | int     | Number of pizzas ordered                  |
| `prep_time_seconds` | int     | Kitchen preparation time                  |
| **`edt_seconds`**   | **int** | **Target: total delivery time (seconds)** |

To regenerate the dataset:

```bash
cd data && python3 generate_dataset.py
```

---

## Training the Model Locally

```bash
# Train with defaults (saves to outputs/)
python src/train.py

# Train with custom paths
python src/train.py \
  --data  data/japan_pizza_delivery.csv \
  --output outputs/ \
  --test_size 0.2 \
  --random_seed 42
```

**Expected output:**

```
✅ Best model: gradient_boosting  RMSE=187s
   Model saved → outputs/edt_model.pkl
   Metadata saved → outputs/train_metadata.json
```

**Model Performance (on test set):**
| Metric | Value | Meaning |
|--------|-------|---------|
| RMSE | ~187s (~3.1 min) | Avg prediction error |
| MAE | ~113s (~1.9 min) | Median prediction error |
| R² | 0.987 | 98.7% variance explained |
| MAPE | ~4.9% | Relative % error |

---

## Azure ML Setup

### Step 1 – Create Azure Resources

```bash
# Login
az login

# Create resource group
az group create --name rg-edt-japan --location japaneast

# Create Azure ML workspace
az ml workspace create \
  --name edt-ml-workspace \
  --resource-group rg-edt-japan \
  --location japaneast

# Create service principal for CI/CD
az ad sp create-for-rbac \
  --name "edt-model-sp" \
  --role Contributor \
  --scopes /subscriptions/<SUBSCRIPTION_ID>/resourceGroups/rg-edt-japan \
  --sdk-auth
# ⚠️  Save this JSON output – it becomes AZURE_CREDENTIALS in GitHub Secrets
```

### Step 2 – Upload Dataset to Azure ML

```bash
az ml data create \
  --name japan-pizza-delivery-data \
  --version 1 \
  --path data/japan_pizza_delivery.csv \
  --type uri_file \
  --workspace-name edt-ml-workspace \
  --resource-group rg-edt-japan
```

### Step 3 – Submit Pipeline Manually

```bash
export AZURE_SUBSCRIPTION_ID=<your-sub-id>
export AZURE_RESOURCE_GROUP=rg-edt-japan
export AZURE_ML_WORKSPACE=edt-ml-workspace
export AZURE_TENANT_ID=<from-sp-output>
export AZURE_CLIENT_ID=<from-sp-output>
export AZURE_CLIENT_SECRET=<from-sp-output>

python pipelines/submit_pipeline.py
```

---

## GitHub Actions CI/CD Setup

### Step 1 – Add GitHub Secrets

Go to **Settings → Secrets and variables → Actions** and add:

| Secret                  | Value                                     |
| ----------------------- | ----------------------------------------- |
| `AZURE_CREDENTIALS`     | Full JSON from `az ad sp create-for-rbac` |
| `AZURE_SUBSCRIPTION_ID` | Your subscription ID                      |
| `AZURE_RESOURCE_GROUP`  | `rg-edt-japan`                            |
| `AZURE_ML_WORKSPACE`    | `edt-ml-workspace`                        |
| `AZURE_TENANT_ID`       | From service principal                    |
| `AZURE_CLIENT_ID`       | From service principal                    |
| `AZURE_CLIENT_SECRET`   | From service principal                    |
| `ACR_LOGIN_SERVER`      | `<yourregistry>.azurecr.io`               |
| `ACR_USERNAME`          | ACR username                              |
| `ACR_PASSWORD`          | ACR password                              |

### Step 2 – Trigger the Pipeline

**Automatic triggers:**

- Push to `main` branch (any changes in `src/`, `data/`, `pipelines/`)
- Every Monday at 2 AM UTC (scheduled retraining)

**Manual trigger:**

```
GitHub → Actions → "EDT Model Training Pipeline" → Run workflow
```

---

## Running the API Locally

```bash
# Ensure model exists
ls outputs/edt_model.pkl

# Start the server
uvicorn api.main:app --reload --port 8000

# Test it
curl http://localhost:8000/health

# Run demo script
python api/sample_request.py
```

API docs available at: `http://localhost:8000/docs`

---

## Deploying the API to Azure

### Option A – Azure Container Instances (quick)

```bash
# Build image
docker build -t edt-api:latest .

# Tag & push to ACR
az acr login --name <your-acr-name>
docker tag edt-api:latest <your-acr>.azurecr.io/edt-api:latest
docker push <your-acr>.azurecr.io/edt-api:latest

# Deploy
az container create \
  --resource-group rg-edt-japan \
  --name edt-api \
  --image <your-acr>.azurecr.io/edt-api:latest \
  --ports 8000 \
  --dns-name-label edt-prediction-api \
  --location japaneast \
  --environment-variables MODEL_PATH=outputs/edt_model.pkl \
  --registry-login-server <your-acr>.azurecr.io \
  --registry-username <acr-user> \
  --registry-password <acr-pass>
```

### Option B – Azure App Service (production)

```bash
az webapp create \
  --resource-group rg-edt-japan \
  --plan edt-app-plan \
  --name edt-prediction-api \
  --deployment-container-image-name <your-acr>.azurecr.io/edt-api:latest
```

---

## Making Prediction Requests

### Single Order

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "store_id":          "Tokyo-Shibuya",
    "order_hour":        19,
    "day_of_week":       "Friday",
    "is_weekend":        0,
    "is_peak_hour":      1,
    "weather_condition": "rain",
    "traffic_level":     "high",
    "vehicle_type":      "scooter",
    "distance_km":       4.2,
    "num_items":         3,
    "prep_time_seconds": 720
  }'
```

**Response:**

```json
{
  "edt_seconds": 2340,
  "edt_minutes": 39.0,
  "model_version": "1.0.0"
}
```

### Batch Orders

```bash
curl -X POST http://localhost:8000/predict/batch \
  -H "Content-Type: application/json" \
  -d '{"orders": [<order1>, <order2>, ...]}'
```

---

## Scaling to New Markets

To deploy in a new market (e.g., South Korea, Singapore):

1. **Collect local data** matching the same CSV schema
2. **Retrain** with `python src/train.py --data data/<new_market>.csv`
3. **Register** under a new model name: `edt-model-korea`
4. **Deploy** a new ACI container with `MODEL_PATH` pointing to the new model
5. **Update** GitHub Actions to add the new market pipeline

---

## Troubleshooting

| Problem                    | Fix                                                          |
| -------------------------- | ------------------------------------------------------------ |
| `Model not found`          | Run `python src/train.py` first                              |
| `422 Unprocessable Entity` | Check field values match allowed enums (see dataset section) |
| `Azure auth failure`       | Verify service principal secrets in GitHub                   |
| `Port 8000 in use`         | Change port: `uvicorn api.main:app --port 8001`              |
| `Docker build fails`       | Ensure `outputs/edt_model.pkl` exists before building        |

---

## Project Structure

```
edt-model/
├── data/
│   ├── generate_dataset.py      # Synthetic dataset generator
│   └── japan_pizza_delivery.csv # Training dataset (5,000 rows)
├── src/
│   ├── preprocess.py            # Shared feature engineering
│   ├── train.py                 # Model training & evaluation
│   └── score.py                 # Inference logic
├── api/
│   ├── main.py                  # FastAPI application
│   └── sample_request.py        # Demo prediction script
├── pipelines/
│   ├── submit_pipeline.py       # Azure ML pipeline submission
│   └── register_model.py        # Azure Model Registry
├── tests/
│   ├── test_model.py            # ML unit tests
│   └── test_api.py              # API integration tests
├── outputs/                     # Generated: model + metadata
├── .github/workflows/
│   └── train.yml                # CI/CD pipeline
├── Dockerfile                   # Container definition
├── requirements.txt             # Python dependencies
├── environment.yml              # Conda env for Azure ML
└── README.md                    # This file
```
#   t r i g g e r e d   w o r k f l o w  
 