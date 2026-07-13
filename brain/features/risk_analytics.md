# Risk Analytics & Population Statistics

Population-level risk stratification and aggregate analytics served by the `/stats` endpoint.

## Risk Definition

**High risk** = PSA concentration > 4000 pg/mL

This binary label is used as the target for all 5 models. The risk score output by models is the predicted probability of belonging to this high-risk class.

## Population Statistics (`GET /stats`)

Aggregate metrics computed across all 1000 patients:

| Metric | Description |
|--------|-------------|
| `total_patients` | 1000 |
| `high_risk_count` | Number predicted high risk by model ensemble |
| `low_risk_count` | 1000 - high_risk_count |
| `mean_risk` | Average risk score across population |
| `mean_consensus` | Average majority vote across population |
| `model_stats` | Per-model high_risk_count + mean_risk |

**Implementation** (`main.py`):
```python
@app.get("/stats")
def get_stats():
    X_all, ids = preprocess_batch_for_inference()
    # Runs all models EXCEPT GNN (too slow)
    results = {}
    for name, model in models.items():
        if name == "GNN":
            continue
        probs = model.predict_proba(X_all)[:, 1]
        results[name] = {
            "high_risk_count": int(np.sum(probs > 0.5)),
            "mean_risk": float(np.mean(probs))
        }
    # Ensemble
    all_probs = np.mean([results[m]["mean_risk"] for m in results], axis=0)
    ...
```

> ⚠️ GNN is skipped in `/stats` because importing PyTorch adds ~30s to server startup. `/stats` only uses sklearn-based models.

## Per-Model Analytics

The `/metrics` endpoint returns pre-computed performance on the holdout test set (300 patients):

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|-------|----------|-----------|--------|----|---------|
| LR | ~0.95 | ~0.92 | ~0.89 | ~0.90 | ~0.987 |
| RF | ~0.95 | ~0.93 | ~0.90 | ~0.91 | ~0.988 |
| SVM | ~0.94 | ~0.91 | ~0.88 | ~0.89 | ~0.985 |
| XGBoost | ~0.95 | ~0.92 | ~0.90 | ~0.91 | ~0.989 |
| GNN | ~0.94 | ~0.90 | ~0.88 | ~0.89 | ~0.983 |

## Trajectory Analysis

The `/trajectory` endpoint simulates how risk changes when a single DPV feature is perturbed:

```
Risk Score
    ↑
 1.0 │       ╱╲
     │      ╱  ╲
 0.5 │─────╱────╲────
     │    ╱      ╲
 0.0 │───╱────────╲──
     └─────────────────→
       -0.5    0    0.5
       Feature Perturbation
```

This helps identify which potential steps most influence the model's decision.

## SHAP Feature Importance

The `/shap` endpoint returns per-feature importance values for a patient:

- Positive SHAP value → feature pushes toward high risk
- Negative SHAP value → feature pushes toward low risk
- Sum of SHAP values ≈ model output deviation from mean

## Counterfactual Analysis

The `/counterfactual` endpoint finds the minimal DPV feature perturbation needed to flip a prediction:

```
Current: High Risk (0.87)
                           Feature#42: -0.15 µA
To become: Low Risk (0.43) ────────────────────→ Flip
```

## Alerts

`GET /alerts` returns patients whose predicted risk crosses the 0.5 threshold, enabling flagging of borderline cases.

## Export

`GET /export` dumps all patient predictions as JSON for external analysis.
