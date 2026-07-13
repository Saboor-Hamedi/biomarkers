# API Endpoints

FastAPI server at `server/main.py` — 17 endpoints serving predictions, analytics, and model management.

## Server Startup

```bash
# Production (port 8001)
python main.py

# Development hot-reload (port 8000)
uvicorn main:app --reload --port 8000
```

## Endpoint Reference

### `GET /`

Health check.

**Response**: `{"message": "Prostate Cancer Risk Prediction API", "status": "healthy"}`

---

### `POST /predict`

Single-patient risk prediction using all 5 models.

**Request**:
```json
{ "sample_id": "Sample_0001" }
```

**Response**:
```json
{
  "sample_id": "Sample_0001",
  "risk_score": 0.87,
  "risk_level": "High",
  "prediction": 1,
  "consensus": 1,
  "model_predictions": {
    "Logistic Regression": 1,
    "Random Forest": 1,
    "SVM": 0,
    "XGBoost": 1,
    "GNN": 1
  },
  "probabilities": {
    "Logistic Regression": 0.92,
    "Random Forest": 0.88,
    "SVM": 0.45,
    "XGBoost": 0.91,
    "GNN": 0.85
  }
}
```

**Logic**:
- Loads patient's 200 DPV currents from Excel
- Scales with `RobustScaler`
- Runs all 5 models
- `risk_score` = mean probability across models
- `consensus` = majority vote (≥3 models predict 1)
- `risk_level` = "High" if risk_score > 0.5, else "Low"

---

### `POST /ensemble`

Ensemble prediction for multiple patients.

**Request**:
```json
{ "sample_ids": ["Sample_0001", "Sample_0002"] }
```

**Response**:
```json
{
  "results": [ /* array of per-patient objects, same shape as /predict */ ]
}
```

---

### `GET /stats`

Population-level aggregate statistics. Skips GNN for speed.

**Response**:
```json
{
  "total_patients": 1000,
  "high_risk_count": 184,
  "low_risk_count": 816,
  "mean_risk": 0.21,
  "mean_consensus": 0.18,
  "model_stats": {
    "Logistic Regression": { "high_risk_count": 175, "mean_risk": 0.19 },
    "Random Forest": { "high_risk_count": 190, "mean_risk": 0.22 },
    "SVM": { "high_risk_count": 165, "mean_risk": 0.18 },
    "XGBoost": { "high_risk_count": 195, "mean_risk": 0.23 }
  }
}
```

---

### `GET /metrics`

Performance metrics for all models on holdout test set.

**Response**:
```json
{
  "Logistic Regression": { "accuracy": 0.95, "precision": 0.92, "recall": 0.89, "f1_score": 0.90, "roc_auc": 0.987 },
  "Random Forest": { ... },
  "SVM": { ... },
  "XGBoost": { ... },
  "GNN": { "error": "GNN model not loaded for /metrics" }
}
```

---

### `GET /sample-ids`

Returns all available sample IDs.

**Response**:
```json
["Sample_0001", "Sample_0002", …, "Sample_1000"]
```

---

### `GET /concentrations/{sample_id}`

Returns biomarker concentrations for a patient.

**Response**:
```json
{
  "sample_id": "Sample_0001",
  "PSA": 4567.8,
  "AFP": 12.3,
  "CA125": 34.5,
  "CA19_9": 5.6,
  "Label": 1
}
```

---

### `POST /concentrations/batch`

Batch lookup of concentrations.

**Request**:
```json
{ "sample_ids": ["Sample_0001", "Sample_0002"] }
```

---

### `POST /trajectory`

Simulates risk trajectory by perturbing individual DPV feature values.

**Request**:
```json
{ "sample_id": "Sample_0001", "feature_index": 50, "range": [-0.5, 0.5] }
```

**Response**: Array of (perturbation_value, risk_score) pairs.

---

### `POST /shap`

Returns SHAP-like feature importance values.

**Request**:
```json
{ "sample_id": "Sample_0001" }
```

**Response**: `{ "shap_values": [ /* shap value per DPV feature */ ] }`

---

### `POST /counterfactual`

Finds minimal perturbation to flip the prediction.

**Request**:
```json
{ "sample_id": "Sample_0001", "target_class": 1 }
```

---

### `POST /reset`

Reloads models from disk (useful after retraining).

---

### `POST /analyze-batch`

Runs batch predictions on all patients.

---

### `GET /confusion-matrix`

Returns confusion matrices for all models.

---

### `GET /alerts`

Returns patients crossing the high-risk threshold.

---

### `GET /export`

Exports all predictions to JSON.

---

### `POST /load-pickle`

Manually loads a specific pickle file.

**Request**:
```json
{ "path": "models/rf_model.pkl" }
```

## Internal Functions

| Function | Purpose |
|----------|---------|
| `load_dpv_data()` | Loads all DPV currents + labels from Excel |
| `is_dpv_features(cols)` | Checks if columns match V0–V199 pattern |
| `preprocess_for_inference(sid)` | Single-patient load + scale |
| `preprocess_batch_for_inference()` | All-patient load + scale |

## Error Handling

- Unknown `sample_id` → `{"detail": "Sample ID not found"}`
- Missing Excel file → `{"detail": "DPV data file not found"}`
- GNN unavailable → endpoint gracefully skips or returns `null`
