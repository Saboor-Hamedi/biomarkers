# Model Ensemble

A 5-model ensemble trained on DPV voltammetry features for binary prostate cancer risk classification.

## Models

| Model | File | Training Time | F1 Score | ROC-AUC |
|-------|------|---------------|----------|---------|
| Logistic Regression | `lr_model.pkl` | Fast | ~0.79 | ~0.99 |
| Random Forest | `rf_model.pkl` | Medium | ~0.79 | ~0.99 |
| SVM | `svm_model.pkl` | Medium | ~0.79 | ~0.99 |
| XGBoost | `xgb_model.pkl` | Medium | ~0.79 | ~0.99 |
| GNN (Graph) | `gnn_model.pkl` | Slow | ~0.79 | ~0.99 |

> GNN stores `{"model": <TorchGNN>, "feature_names": [...], "edge_index": tensor}`

## Training Pipeline

`server/analysis/src/analyse_data.py` → function `run_complete_pipeline()`:

1. **Data loading**: DPV currents + target labels from Excel
2. **Preprocessing**: Scaling via `RobustScaler`
3. **Train/Validation/Test split**: 560 / 140 / 300
4. **Class balancing**: SMOTE oversampling applied
5. **Training**: Each model fit on balanced data
6. **Evaluation**: Accuracy, Precision, Recall, F1, ROC-AUC, Confusion Matrix
7. **Persistence**: All 5 models + scaler + features saved to `models/`

## Ensemble Voting

In `main.py`, the `/ensemble` endpoint:

```python
# For each of the 5 models, get predict_proba
probabilities = []
for name, model in models:
    proba = model.predict_proba(X_scaled)[:, 1]
    probabilities.append(proba)

# Average probability across all models
ensemble_prob = np.mean(probabilities, axis=0)
ensemble_pred = (ensemble_prob >= 0.5).astype(int)
```

The frontend `/predict` endpoint also returns individual model scores and a consensus value (majority vote).

## Pickle Dictionary Format

### ML Models (LR, RF, SVM, XGBoost)
```python
{
    "model": <sklearn/xgb model object>,
    "feature_names": ["V0", …, "V199"]
}
```

### GNN Model
```python
{
    "model": <gnn_model.TorchGNN instance>,
    "feature_names": ["V0", …, "V199"],
    "edge_index": <torch.Tensor of shape (2, E)>
}
```

### Scaler
```python
{
    "scaler": <RobustScaler>,
    "feature_names": ["V0", …, "V199"]
}
```

## Class Imbalance Handling

- Approximately 20% positive class (high risk)
- SMOTE applied during training to equalize class distribution
- Ensemble voting reduces variance from any single classifier
