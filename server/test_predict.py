import pickle
import os
import sys
import pandas as pd
import numpy as np

MODELS_DIR = os.path.join(os.path.dirname(__file__), "analysis", "models")
DATA_PATH = os.path.join(os.path.dirname(__file__), "analysis", "data", "Raw_data_dpv.xlsx")

with open(os.path.join(MODELS_DIR, "feature_columns.pkl"), "rb") as f:
    feature_columns = pickle.load(f)
with open(os.path.join(MODELS_DIR, "scaler.pkl"), "rb") as f:
    scaler = pickle.load(f)

model_files = [f for f in os.listdir(MODELS_DIR) if f.endswith("_model.pkl")]
models = {}
for mf in model_files:
    name = mf.replace("_model.pkl", "").title()
    try:
        with open(os.path.join(MODELS_DIR, mf), "rb") as f:
            models[name] = pickle.load(f)
        print(f"  Loaded {name}")
    except Exception as e:
        print(f"  Failed {name}: {e}")

print(f"\nFeature columns ({len(feature_columns)}): {feature_columns[:5]} ... {feature_columns[-3:]}")
print(f"Scaler: {type(scaler).__name__}")

dpv_cols = [c for c in feature_columns if c.startswith('V') and c[1:].isdigit()]
print(f"DPV features: {len(dpv_cols)}")
target = pd.read_excel(DATA_PATH, sheet_name="Target_Concentrations")

sample_dfs = []
for i in range(1, 1001):
    sheet = f"Sample_{i:04d}"
    try:
        sdf = pd.read_excel(DATA_PATH, sheet_name=sheet)
        sdf['sample_id'] = sheet
        sample_dfs.append(sdf)
    except Exception:
        pass

df = pd.concat(sample_dfs, ignore_index=True)
print(f"Loaded {len(df)} DPV rows from {len(sample_dfs)} samples")
df = df.merge(target, on='sample_id', how='left')
print(f"Merged shape: {df.shape}")

row = df[df['sample_id'] == 'Sample_0001']
print(f"\nSample_0001 DPV shape: {row[dpv_cols].shape}")
print(f"Sample_0001 biomarkers: AFP={row['AFP_pg_per_ml'].values[0]:.1f}, CA125={row['CA125_U_per_ml'].values[0]:.1f}, PSA={row['PSA_pg_per_ml'].values[0]:.1f}")

X_raw = row[feature_columns].values
X_scaled = scaler.transform(X_raw)

print("\nPredictions:")
for name, model in models.items():
    actual = model["model"] if isinstance(model, dict) and "model" in model else model
    if hasattr(actual, 'predict_proba'):
        prob = actual.predict_proba(X_scaled)[0, 1]
        print(f"  {name}: {prob:.4f}")
    else:
        pred = actual.predict(X_scaled)[0]
        print(f"  {name}: {pred}")
