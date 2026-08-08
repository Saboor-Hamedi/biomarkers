"""
Console report generators for the DPV biomarker pipeline.

All values are computed from real data (out-of-fold predictions, raw
voltammetry, and trained models) — nothing is hardcoded.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, precision_score, recall_score


def print_fold_cv_metrics(fold_scores):
    """Per-fold CV metrics as mean ± std across the 5 folds."""
    print("\n" + "=" * 70)
    print("FOLD-WISE CROSS-VALIDATION METRICS (mean ± std over 5 folds)")
    print("=" * 70)
    metrics = ["accuracy", "precision", "recall", "f1_score", "roc_auc"]
    labels = ["Accuracy", "Precision", "Recall", "F1-Score", "ROC-AUC"]
    header = f"{'Model':<22}" + "".join(f"{lab:>14}" for lab in labels)
    print(header)
    print("-" * len(header))
    for name, scores in fold_scores.items():
        row = f"{name:<22}"
        for m in metrics:
            vals = np.asarray(scores[m])
            row += f"{vals.mean():.3f}±{vals.std():.3f}".rjust(14)
        print(row)
    print()


def print_biophysical_mapping(df, dpv_cols):
    """Correlation of DPV currents at the three redox peaks vs their biomarkers."""
    print("\n" + "=" * 70)
    print("BIOPHYSICAL PEAK MAPPING (DPV vs biomarker concentration)")
    print("=" * 70)

    def voltage_of(col):
        return float(col.replace("curr_", "").replace("mV", ""))

    def nearest_col(peak):
        cols = sorted(dpv_cols, key=lambda c: abs(voltage_of(c) - peak))
        return cols[0]

    targets = {
        "PSA": ("PSA_pg_per_ml", -468),
        "AFP": ("AFP_pg_per_ml", 365),
        "CA125": ("CA125_U_per_ml", 968),
    }
    print(f"{'Biomarker':<10}{'Peak (mV)':>12}{'Measured (mV)':>16}{'r (log-conc)':>16}")
    print("-" * 54)
    for label, (target_col, peak) in targets.items():
        col = nearest_col(peak)
        x = df[col].values
        yv = np.log1p(df[target_col].values)
        r = np.corrcoef(x, yv)[0, 1]
        print(f"{label:<10}{peak:>12}{voltage_of(col):>16.0f}{r:>16.4f}")
    print()


def print_risk_stratification(proba, y_true, model_name="Best model"):
    """Risk band counts using the given probability vector."""
    print("\n" + "=" * 70)
    print(f"RISK STRATIFICATION ({model_name})")
    print("=" * 70)
    safe = int((proba < 0.45).sum())
    moderate = int(((proba >= 0.45) & (proba < 0.60)).sum())
    high = int(((proba >= 0.60) & (proba <= 0.75)).sum())
    critical = int((proba > 0.75).sum())
    print(f"  Safe (<45%):        {safe}")
    print(f"  Moderate (45-60%):  {moderate}")
    print(f"  High (60-75%):      {high}")
    print(f"  Critical (>75%):    {critical}")
    print(f"  Total:              {safe + moderate + high + critical}")
    print()


def print_optimal_threshold(proba, y_true, model_name="Best model"):
    """Threshold (0-1) that maximizes F1 for the given model."""
    print("\n" + "=" * 70)
    print(f"OPTIMAL DECISION THRESHOLD ({model_name})")
    print("=" * 70)
    best_f1 = -1.0
    best_thr = 0.5
    thresholds = np.linspace(0, 1, 201)
    for thr in thresholds:
        y_pred = (proba >= thr).astype(int)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_thr = thr
    prec = precision_score(y_true, (proba >= best_thr).astype(int), zero_division=0)
    rec = recall_score(y_true, (proba >= best_thr).astype(int), zero_division=0)
    print(f"  Optimal threshold:  {best_thr:.2f}")
    print(f"  F1 at threshold:    {best_f1:.4f}")
    print(f"  Precision:          {prec:.4f}")
    print(f"  Recall:             {rec:.4f}")
    print()


def print_top_features(trained_models, feature_columns, dpv_cols, top_n=10):
    """Top-N most important voltage steps from the best available model."""
    print("\n" + "=" * 70)
    print(f"TOP-{top_n} DISCRIMINATIVE VOLTAGE STEPS (feature importance)")
    print("=" * 70)

    def voltage_of(col):
        return float(col.replace("curr_", "").replace("mV", ""))

    importances = None
    source = None
    for name, model in trained_models.items():
        if hasattr(model, "feature_importances_"):
            if len(model.feature_importances_) == len(feature_columns):
                importances = model.feature_importances_
                source = name
                break
        elif hasattr(model, "coef_"):
            if len(model.coef_[0]) == len(feature_columns):
                importances = np.abs(model.coef_[0])
                source = name
                break

    if importances is None:
        print("  ⚠️ No model with feature importances / coefficients found")
        return

    dpv_idx = [feature_columns.index(c) for c in dpv_cols]
    imp_dpv = importances[dpv_idx]
    order = np.argsort(imp_dpv)[::-1][:top_n]
    print(f"  Source model: {source}")
    print(f"{'#':>3}{'Voltage (mV)':>14}{'Importance':>14}")
    print("-" * 31)
    for rank, i in enumerate(order, 1):
        print(f"{rank:>3}{voltage_of(dpv_cols[i]):>14.0f}{imp_dpv[i]:>14.4f}")
    print()


def print_patient_prediction(trained_models, scaler, feature_columns, X_scaled_row):
    """Print the per-patient risk summary exactly like the server /predict endpoint."""
    print("\n" + "=" * 70)
    print("PATIENT RISK PREDICTION (ensemble, matches Electron /predict)")
    print("=" * 70)

    probs = {}
    for name, model in trained_models.items():
        actual = model["model"] if isinstance(model, dict) and "model" in model else model
        try:
            if hasattr(actual, "predict_proba"):
                probs[name] = float(actual.predict_proba(X_scaled_row)[0, 1])
            else:
                probs[name] = float(actual.predict(X_scaled_row)[0])
        except Exception as e:
            print(f"  ⚠️ {name} failed: {e}")

    if not probs:
        print("  No model predictions available")
        return

    for name, p in probs.items():
        print(f"  {name:<22} {p*100:6.1f}%")
    avg = float(np.mean(list(probs.values())))
    risk = "Positive" if avg > 0.5 else "Negative"
    consensus = (avg if avg > 0.5 else (1 - avg)) * 100
    print("-" * 40)
    print(f"  Risk Score:        {avg:.4f}")
    print(f"  Prediction:        {risk}")
    print(f"  Model Agreement:   {consensus:.1f}%")
    print()


def print_ensemble_agreement(results, n_samples, model_names=None):
    """Percentage of samples where all models agree on the same class."""
    print("\n" + "=" * 70)
    print("ENSEMBLE CONSENSUS AGREEMENT")
    print("=" * 70)
    names = model_names if model_names is not None else list(results.keys())
    preds = np.column_stack([results[n]["y_pred"] for n in names if results[n]["y_pred"] is not None])
    if preds.shape[1] == 0:
        print("  ⚠️ No predictions available for consensus analysis")
        return
    unanimous = np.all(preds == preds[:, [0]], axis=1)
    n_agree = int(unanimous.sum())
    n_split = int((~unanimous).sum())
    print(f"  Models compared:     {preds.shape[1]}")
    print(f"  All models agree:    {n_agree} ({n_agree / len(preds) * 100:.1f}%)")
    print(f"  Split decision:      {n_split} ({n_split / len(preds) * 100:.1f}%)")
    print()
