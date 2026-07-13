# Biomarker Analysis Platform

A desktop application for prostate cancer risk classification using **DPV voltammetry** data. The platform trains a 5-model ensemble (Logistic Regression, Random Forest, SVM, XGBoost, Graph Neural Network) on 200-point current features extracted from electrochemical assays, and exposes results through a FastAPI server with an Electron + React frontend.

## Architecture Overview

```
┌─────────────────────────────────────────────────────┐
│                  Electron Desktop App                │
│  ┌─────────────┐  IPC   ┌────────────────────────┐ │
│  │  React UI   │◄──────►│   main/index.js        │ │
│  │  (Renderer) │  HTTP  │   - spawns Python      │ │
│  │             │◄──────►│   - file sync handlers  │ │
│  └─────────────┘  :8001 └────────────────────────┘ │
└─────────────────────────────────────────────────────┘
                           │
                    HTTP :8001
                           ▼
┌─────────────────────────────────────────────────────┐
│              FastAPI Server (main.py)                │
│  ┌──────────┐ ┌──────────┐ ┌────────────────────┐ │
│  │ /predict │ │ /stats   │ │ /shap, /trajectory │ │
│  │ /metrics │ │/ensemble │ │ /counterfactual    │ │
│  └──────────┘ └──────────┘ └────────────────────┘ │
│              │                                     │
│         loads .pkl files                           │
└─────────────────────────────────────────────────────┘
                           │
              reads Raw_data_dpv.xlsx
                           ▼
┌─────────────────────────────────────────────────────┐
│            Training Pipeline (analyse_data.py)       │
│  DPV Excel → 200 features → 5 models → .pkl files  │
└─────────────────────────────────────────────────────┘
```

## Features

| Document | Description |
|----------|-------------|
| [DPV Data Pipeline](features/dpv_pipeline.md) | Excel loading, feature extraction, preprocessing |
| [Model Ensemble](features/model_ensemble.md) | 5-model training, evaluation, voting |
| [GNN Model](features/gnn_model.md) | Graph Neural Network architecture & training |
| [API Endpoints](features/api_endpoints.md) | FastAPI REST API reference |
| [Electron App](features/electron_app.md) | Desktop shell, process management, IPC |
| [Frontend Components](features/frontend_components.md) | React UI, state management, component tree |
| [Risk Analytics](features/risk_analytics.md) | Risk stratification & population-level analytics |

## Quick Start

```bash
# Train models
cd server/analysis/src
python analyse_data.py

# Start server
cd server
uvicorn main:app --reload --port 8000

# Start Electron app (from project root)
npm run dev
```

## Data Source

- **File**: `server/analysis/data/Raw_data_dpv.xlsx`
- **Sheets**: `Target_Concentrations` (metadata) + `Sample_0001` through `Sample_1000`
- **Features**: Voltammetry currents at 200 potential steps (columns `V0`–`V199`)
- **Target**: Binary risk label derived from PSA concentration (>4000 pg/mL = high risk)

## Port Convention

| Port | Usage |
|------|-------|
| 8001 | Default server port (Electron spawns Python on this) |
| 8000 | Hot-reload `uvicorn` for development |
