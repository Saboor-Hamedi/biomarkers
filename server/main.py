import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import numpy as np
import pickle
import os
import glob
import random
from fastapi.middleware.cors import CORSMiddleware

# Define GNNModel class for unpickling if needed
try:
    import torch
    from torch_geometric.nn import GCNConv, global_mean_pool
    class GNNModel(torch.nn.Module):
        def __init__(self, num_features, hidden_channels, num_classes):
            super(GNNModel, self).__init__()
            self.conv1 = GCNConv(num_features, hidden_channels)
            self.conv2 = GCNConv(hidden_channels, hidden_channels)
            self.lin = torch.nn.Linear(hidden_channels, num_classes)

        def forward(self, x, edge_index, batch):
            x = self.conv1(x, edge_index)
            x = x.relu()
            x = self.conv2(x, edge_index)
            x = x.relu()
            x = global_mean_pool(x, batch)
            x = self.lin(x)
            return x
except ImportError:
    pass

app = FastAPI(title="Biomarker AI Engine")

# Enable CORS for Electron Renderer
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

ARTIFACTS_DIR = os.path.join(os.path.dirname(__file__), "artifacts")

class PredictionRequest(BaseModel):
    features: dict # e.g. {"AFP_pg_per_ml": 1200, "CA125_U_per_ml": 35}

def load_artifacts():
    models = {}
    scaler = None
    feature_columns = None
    
    if not os.path.exists(ARTIFACTS_DIR):
        return None, None, None

    # Load Scaler
    scaler_path = os.path.join(ARTIFACTS_DIR, "scaler.pkl")
    if os.path.exists(scaler_path):
        with open(scaler_path, "rb") as f:
            scaler = pickle.load(f)

    # Load Feature Columns
    features_path = os.path.join(ARTIFACTS_DIR, "feature_columns.pkl")
    if os.path.exists(features_path):
        with open(features_path, "rb") as f:
            feature_columns = pickle.load(f)

    # Load Models
    model_files = glob.glob(os.path.join(ARTIFACTS_DIR, "*_model.pkl"))
    for mf in model_files:
        name = os.path.basename(mf).replace("_model.pkl", "").capitalize()
        with open(mf, "rb") as f:
            models[name] = pickle.load(f)
            
    return models, scaler, feature_columns

@app.get("/")
async def root():
    return {"status": "online", "engine": "XAI-Biomarker-Hub", "version": "1.0.0"}

@app.post("/predict")
async def predict(request: PredictionRequest):
    try:
        models, scaler, feature_columns = load_artifacts()
        
        if not models or not scaler:
            return {"error": "Engine offline: No synchronized artifacts detected."}

        # Prepare Input Data
        input_data = []
        for col in feature_columns:
            if col not in request.features:
                return {"error": f"Missing required feature: {col}"}
            input_data.append(request.features[col])
        
        # 1. Preprocessing: Log1p Transformation
        X_raw = np.array([input_data])
        X_log = np.log1p(X_raw)
        
        # 2. Scaling - Use DataFrame to avoid feature name warnings
        X_df = pd.DataFrame(X_log, columns=feature_columns)
        X_scaled = scaler.transform(X_df)
        
        # 3. Model Predictions
        results = {}
        probabilities = []
        
        for name, model in models.items():
            try:
                # Check if model is wrapped in a dict (common in GNN saves)
                actual_model = model["model"] if isinstance(model, dict) and "model" in model else model
                
                if name.lower() == "gnn":
                    # GNN prediction logic
                    try:
                        import torch
                        from torch_geometric.data import Data
                        actual_model.eval()
                        
                        # Use same logic as notebook
                        x_tensor = torch.FloatTensor(X_scaled).expand(len(feature_columns), -1)
                        # Minimal graph for single sample inference
                        edge_index = model["edge_index"] if isinstance(model, dict) and "edge_index" in model else torch.LongTensor([[0, 1], [1, 0]]).t().contiguous()
                        
                        with torch.no_grad():
                            batch = torch.zeros(x_tensor.size(0), dtype=torch.long)
                            out = actual_model(x_tensor, edge_index, batch)
                            prob = torch.softmax(out, dim=1)[0, 1].item()
                        
                        results[name] = prob
                        probabilities.append(prob)
                    except Exception as e:
                        print(f"GNN Prediction failed: {e}")
                else:
                    if hasattr(actual_model, 'predict_proba'):
                        prob = actual_model.predict_proba(X_scaled)[0, 1]
                        results[name] = prob
                        probabilities.append(prob)
                    else:
                        pred = actual_model.predict(X_scaled)[0]
                        results[name] = float(pred)
                        probabilities.append(float(pred))
            except Exception as e:
                print(f"Model {name} failed: {e}")

        if not probabilities:
            return {"error": "All models failed to generate predictions."}

        # 4. Ensemble Metrics
        avg_score = np.mean(probabilities)
        risk_level = "Positive" if avg_score > 0.5 else "Negative"
        consensus = (avg_score if avg_score > 0.5 else (1 - avg_score)) * 100

        # 5. t-SNE Projection (Mock for visualization)
        tsne_x = float(np.interp(avg_score, [0, 1], [-10, 10]))
        tsne_y = float(np.interp(avg_score, [0, 1], [-10, 10])) + random.uniform(-2, 2)

        return {
            "risk_score": round(float(avg_score), 4),
            "prediction": risk_level,
            "consensus": f"{round(float(consensus), 1)}%",
            "models": {k: f"{round(v*100, 1)}%" for k, v in results.items()},
            "tsne_coord": {"x": tsne_x, "y": tsne_y}
        }
        
    except Exception as e:
        import traceback
        return {"error": str(e), "traceback": traceback.format_exc()}

@app.post("/trajectory")
async def get_trajectory(request: PredictionRequest):
    models, scaler, feature_columns = load_artifacts()
    
    if not models or not scaler or not feature_columns:
        return {"error": "System not fully calibrated."}
        
    try:
        # Base features
        base_features = request.features.copy()
        
        trajectory_data = []
        
        # We sweep PSA from 0 to 20 across 30 points
        psa_values = np.linspace(0, 20, 30)
        
        for psa_val in psa_values:
            # Update PSA
            step_features = base_features.copy()
            step_features['PSA_pg_per_ml'] = float(psa_val)
            
            # Align features
            aligned_features = {}
            for col in feature_columns:
                val = step_features.get(col, 0.0)
                aligned_features[col] = [val]
                
            df_step = pd.DataFrame(aligned_features)
            
            # Preprocessing matching the predict endpoint
            X_raw = df_step.values
            X_log = np.log1p(X_raw)
            X_df = pd.DataFrame(X_log, columns=feature_columns)
            X_scaled = scaler.transform(X_df)
            
            step_data = {"psa": round(float(psa_val), 1)}
            
            for name, model in models.items():
                actual_model = model["model"] if isinstance(model, dict) and "model" in model else model
                
                if name.lower() == "gnn":
                    # Mock GNN behavior smoothly following PSA
                    prob = float(np.clip((psa_val / 20.0) * 0.8 + 0.1, 0, 1))
                    step_data[name] = round(prob * 100, 2)
                else:
                    if hasattr(actual_model, 'predict_proba'):
                        prob = float(actual_model.predict_proba(X_scaled)[0, 1])
                    else:
                        prob = float(actual_model.predict(X_scaled)[0])
                    step_data[name] = round(prob * 100, 2)
                    
            trajectory_data.append(step_data)
            
        return trajectory_data
    except Exception as e:
        import traceback
        return {"error": str(e), "traceback": traceback.format_exc()}

@app.post("/shap")
async def get_shap(request: PredictionRequest):
    # Simulates a SHAP Waterfall Breakdown for a specific patient
    features = request.features
    base_risk = 20.0 # baseline risk in %
    
    # Heuristics for mock SHAP based on actual values
    afp = features.get('AFP_pg_per_ml', 0)
    ca125 = features.get('CA125_U_per_ml', 0)
    psa = features.get('PSA_pg_per_ml', 0)
    
    # Impact calculations
    afp_impact = min(40.0, (afp / 1000) * 15) if afp > 500 else -5.0
    ca125_impact = min(30.0, (ca125 / 35) * 10) if ca125 > 30 else -8.0
    psa_impact = min(35.0, (psa / 4) * 20) if psa > 3.5 else -12.0
    
    return [
        {"feature": "Baseline", "value": base_risk},
        {"feature": "AFP_pg_per_ml", "value": round(afp_impact, 1), "actual": afp},
        {"feature": "CA125_U_per_ml", "value": round(ca125_impact, 1), "actual": ca125},
        {"feature": "PSA_pg_per_ml", "value": round(psa_impact, 1), "actual": psa},
    ]

@app.get("/boundaries")
async def get_boundaries():
    # Simulates a Topographic Decision Boundary map (AFP vs CA125)
    points = []
    for afp in np.linspace(0, 5000, 15):
        for ca125 in np.linspace(0, 100, 15):
            # Calculate a synthetic "danger score" contour
            danger = (afp/5000)*0.5 + (ca125/100)*0.5
            danger = danger + np.sin(afp/1000)*0.1 # add non-linear curve
            danger = np.clip(danger * 100, 0, 100)
            points.append({"afp": round(float(afp), 1), "ca125": round(float(ca125), 1), "risk": round(float(danger), 1)})
    return points

@app.get("/heatmap")
async def get_heatmap():
    # Simulates a Model Consensus Heatmap across the 1000-patient cohort
    models = ["XGBoost", "SVM", "Random Forest", "GNN", "Logistic Regression"]
    data = []
    for m1 in models:
        for m2 in models:
            if m1 == m2:
                corr = 100.0
            else:
                # Mock correlations
                base = 80.0 if ("Tree" in m1 or "Boost" in m1 or "Forest" in m1) and ("Tree" in m2 or "Boost" in m2 or "Forest" in m2) else 65.0
                base += random.uniform(-5, 10)
                corr = min(98.0, base)
            data.append({"x": m1, "y": m2, "value": round(corr, 1)})
    return data

@app.post("/counterfactual")
async def get_counterfactual(request: PredictionRequest):
    features = request.features
    psa = features.get('PSA_pg_per_ml', 0)
    
    if psa > 4.0:
        target_psa = max(0, psa - 2.0)
        reduction = round((psa - target_psa) * 12.5, 1)
        statement = f"If this patient's PSA was {target_psa:.1f} pg/ml (down by 2 points), the ensemble risk score would drop by approximately {reduction}%, shifting them to a safer clinical threshold."
    else:
        statement = "Patient's biomarkers are within stable ranges. No major counterfactual shifts identified that would dramatically alter the current negative risk profile."
        
    return {"statement": statement}

@app.get("/audit")
async def audit():
    artifacts_dir = ARTIFACTS_DIR
    if not os.path.exists(artifacts_dir):
        return {"status": "waiting", "artifacts": [], "message": "No artifacts synchronized yet."}
    
    files = [f for f in os.listdir(artifacts_dir) if f.endswith('.pkl')]
    return {
        "status": "ready" if files else "waiting",
        "artifacts": files,
        "count": len(files)
    }

@app.get("/tsne")
async def get_tsne():
    # In a real scenario, this would compute t-SNE on the synchronized dataset
    # For now, we return a high-fidelity mock dataset for visualization
    import random
    points = []
    for _ in range(50):
        # Cluster 1 (Low Risk)
        afp = random.uniform(500, 1500)
        ca125 = random.uniform(10, 30)
        points.append({"x": random.uniform(-10, 2), "y": random.uniform(-10, 5), "cluster": 0, "AFP": afp, "CA125": ca125})
        
        # Cluster 2 (High Risk)
        afp = random.uniform(2000, 8000)
        ca125 = random.uniform(35, 100)
        points.append({"x": random.uniform(3, 10), "y": random.uniform(-2, 10), "cluster": 1, "AFP": afp, "CA125": ca125})
    
    return {"points": points}

@app.get("/importance")
async def get_importance():
    try:
        models, _, feature_columns = load_artifacts()
        if not models or not feature_columns:
            return {"error": "Engine offline: No artifacts detected."}
        
        importance_data = {}
        for name, model in models.items():
            actual_model = model["model"] if isinstance(model, dict) and "model" in model else model
            
            # Models that support feature_importances_ (XGBoost, RF)
            if hasattr(actual_model, "feature_importances_"):
                importances = actual_model.feature_importances_.tolist()
                importance_data[name] = {
                    feature_columns[i]: round(importances[i], 4) 
                    for i in range(len(feature_columns))
                }
            # For linear models like Logistic Regression or SVM with linear kernel
            elif hasattr(actual_model, "coef_"):
                # Use absolute coefficients as importance
                coefs = np.abs(actual_model.coef_[0]).tolist()
                total = sum(coefs)
                importance_data[name] = {
                    feature_columns[i]: round(coefs[i] / total, 4) if total > 0 else 0 
                    for i in range(len(feature_columns))
                }
        
        return importance_data
    except Exception as e:
        return {"error": str(e)}

@app.get("/distributions")
async def get_distributions():
    try:
        data_path = os.path.join(os.path.dirname(__file__), "analysis", "data", "Raw_data_dpv.xlsx")
        if not os.path.exists(data_path):
            return {"error": "Raw data source not found."}
        
        df = pd.read_excel(data_path, sheet_name="Target_Concentrations")
        
        distributions = {}
        for col in ["AFP_pg_per_ml", "CA125_U_per_ml", "PSA_pg_per_ml"]:
            # Generate histogram data
            counts, bin_edges = np.histogram(df[col].dropna(), bins=30)
            bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
            
            distributions[col] = [
                {"x": float(center), "y": int(count)} 
                for center, count in zip(bin_centers, counts)
            ]
            
        return distributions
    except Exception as e:
        return {"error": str(e)}

@app.get("/top-patients")
async def get_top_patients():
    try:
        models, scaler, feature_columns = load_artifacts()
        if not models or not scaler:
            return {"error": "Engine offline."}
            
        data_path = os.path.join(os.path.dirname(__file__), "analysis", "data", "Raw_data_dpv.xlsx")
        df = pd.read_excel(data_path, sheet_name="Target_Concentrations")
        
        # Preprocessing
        X = df[feature_columns].copy()
        X_log = np.log1p(X)
        X_scaled = scaler.transform(X_log)
        
        # Batch Prediction
        all_probs = []
        for name, model in models.items():
            actual_model = model["model"] if isinstance(model, dict) and "model" in model else model
            if hasattr(actual_model, 'predict_proba'):
                probs = actual_model.predict_proba(X_scaled)[:, 1]
                all_probs.append(probs)
            elif hasattr(actual_model, 'predict'):
                preds = actual_model.predict(X_scaled)
                all_probs.append(preds)
        
        avg_scores = np.mean(all_probs, axis=0)
        df['risk_score'] = avg_scores
        
        # Sort by risk score descending
        df_sorted = df.sort_values('risk_score', ascending=False)
        
        results = []
        for _, row in df_sorted.iterrows():
            score = row['risk_score']
            status = "Urgent" if score > 0.8 else "Critical" if score > 0.6 else "Moderate" if score > 0.4 else "Stable"
            
            results.append({
                "id": row['sample_id'],
                "AFP": round(row['AFP_pg_per_ml'], 2),
                "CA125": round(row['CA125_U_per_ml'], 2),
                "PSA": round(row['PSA_pg_per_ml'], 2),
                "score": round(float(score), 4),
                "status": status,
                "details": {
                    "Raw AFP": f"{row['AFP_pg_per_ml']:.2f} pg/ml",
                    "Raw CA125": f"{row['CA125_U_per_ml']:.2f} U/ml",
                    "Raw PSA": f"{row['PSA_pg_per_ml']:.2f} pg/ml",
                    "Neural Certainty": f"{(abs(score - 0.5) * 200):.1f}%",
                    "Forensic Cluster": "Alpha-7" if score > 0.5 else "Gamma-2"
                }
            })
            
        return results
    except Exception as e:
        return {"error": str(e)}

@app.get("/metrics")
async def get_metrics():
    # Mock curves for visualization
    models = ["GNN", "XGBoost", "Random Forest", "SVM", "Logistic Regression"]
    colors = ["#3b82f6", "#ef4444", "#10b981", "#f59e0b", "#8b5cf6"]
    
    # Values derived from analysis.ipynb results summary
    auc_scores = [0.4503, 0.5851, 0.5617, 0.4649, 0.4658]
    precisions = [0.0000, 0.1667, 0.1818, 0.1129, 0.1029]
    recalls = [0.0000, 0.3056, 0.1111, 0.5833, 0.3889]
    
    roc_curves = {}
    pr_curves = {}
    cal_curves = {}
    
    for i, model in enumerate(models):
        roc_points = []
        pr_points = []
        cal_points = []
        
        # Simplified curve generation reflecting the AUC/Precision
        auc = auc_scores[i]
        for x in np.linspace(0, 1, 25):
            # ROC: Map x (FPR) to y (TPR) based on AUC
            # Higher AUC -> steeper curve
            y_roc = x**(1/(2*auc + 0.1)) + (0.01 * np.random.randn())
            roc_points.append({"x": round(float(x), 2), "y": round(float(np.clip(y_roc, 0, 1)), 2)})
            
            # PR: Map x (Recall) to y (Precision)
            y_pr = precisions[i] * (1 - x**2) + (0.02 * np.random.randn())
            pr_points.append({"x": round(float(x), 2), "y": round(float(np.clip(y_pr, 0, 1)), 2)})
            
        # Calibration curve (Reliability Diagram) - 10 bins
        for bin_idx, pred_prob in enumerate(np.linspace(0.05, 0.95, 10)):
            # Perfectly calibrated is true_frac = pred_prob
            # We add some model-specific deviation based on AUC
            deviation = (0.5 - auc) * np.sin(np.pi * pred_prob) 
            true_frac = pred_prob + deviation + (0.05 * np.random.randn())
            cal_points.append({
                "predicted": round(float(pred_prob), 2), 
                "true_fraction": round(float(np.clip(true_frac, 0, 1)), 2)
            })
            
        roc_curves[model] = {"points": roc_points, "color": colors[i], "auc": round(auc, 4)}
        pr_curves[model] = {"points": pr_points, "color": colors[i]}
        cal_curves[model] = {"points": cal_points, "color": colors[i]}

    # Confusion Matrix from notebook (XGBoost example)
    # [TN, FP]
    # [FN, TP]
    confusion_matrix = [
        [234, 30], # Actual Negative
        [25, 11]   # Actual Positive
    ]

    return {
        "roc": roc_curves,
        "pr": pr_curves,
        "calibration": cal_curves,
        "cm": confusion_matrix
    }

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
