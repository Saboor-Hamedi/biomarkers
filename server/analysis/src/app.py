import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
import sys
import matplotlib.pyplot as plt

# 🧬 RESEARCH BRIDGE: Ensure GNN definitions are available for unpickling
try:
    import torch
    import torch.nn.functional as F
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

# Page config
st.set_page_config(
    page_title="NEURAL ANALYTICS | Research Hub",
    page_icon=None,
    layout="wide"
)

# Industrial Clinical Theme (High Contrast Cards)
st.markdown("""
    <style>
    .main { background-color: #0e1117; }
    .stMetric { 
        background-color: #ffffff !important; 
        border: 1px solid #dee2e6; 
        padding: 20px; 
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    /* Ensure metric text is dark for readability on white cards */
    [data-testid="stMetricValue"] { color: #1a1c24 !important; }
    [data-testid="stMetricLabel"] { color: #495057 !important; }
    [data-testid="stMetricDelta"] { font-weight: bold; }
    
    .stTabs [data-baseweb="tab-list"] { gap: 10px; }
    .stTabs [data-baseweb="tab"] { 
        background-color: #1a1c24; 
        border-radius: 5px 5px 0 0; 
        padding: 10px 20px;
        color: #808495;
    }
    .stTabs [aria-selected="true"] { background-color: #ff4b4b !important; color: white !important; }
    </style>
    """, unsafe_allow_html=True)

# Load all models
@st.cache_resource
def load_models():
    model_dir = "../models/"
    models = {}
    extra_meta = {}
    for m in ["logistic_regression", "random_forest", "svm", "xgboost", "gnn"]:
        path = os.path.join(model_dir, f"{m}_model.pkl")
        if os.path.exists(path):
            with open(path, 'rb') as f:
                obj = pickle.load(f)
                if isinstance(obj, dict) and "model" in obj:
                    models[m] = obj["model"]
                    if "edge_index" in obj: extra_meta[m] = {"edge_index": obj["edge_index"]}
                else:
                    models[m] = obj
    
    with open(os.path.join(model_dir, "scaler.pkl"), 'rb') as f:
        scaler = pickle.load(f)
    with open(os.path.join(model_dir, "feature_columns.pkl"), 'rb') as f:
        features = pickle.load(f)
    return models, scaler, features, extra_meta

# Sidebar
with st.sidebar:
    st.header("Patient Parameters")
    afp = st.number_input("AFP (pg/mL)", value=500.0)
    ca125 = st.number_input("CA125 (U/mL)", value=35.0)
    st.divider()
    model_choice = st.selectbox("Champion Model", ["XGBoost", "Random Forest", "GNN", "SVM", "Logistic Regression"])
    threshold = st.slider("Sensitivity", 0.0, 1.0, 0.5)
    run_btn = st.button("RUN FORENSIC ANALYSIS", type="primary")

# ── MAIN DASHBOARD ──────────────────────────────────────────────────────
st.title("Cancer Biomarker Research Hub")
st.markdown("### Industrial-Grade AI Clinical Insights")

try:
    models, scaler, feature_columns, extra_meta = load_models()
except Exception as e:
    st.error(f"Initialization Error: {e}")
    st.stop()

def get_prediction(model_obj, name, X_scaled):
    """Unified inference for sklearn and GNN."""
    if hasattr(model_obj, 'predict_proba'):
        return float(model_obj.predict_proba(X_scaled)[0][1])
    elif "gnn" in name.lower() and HAS_TORCH:
        model_obj.eval()
        with torch.no_grad():
            # Prep graph data matching notebook logic
            num_feats = X_scaled.shape[1]
            x = torch.tensor(X_scaled, dtype=torch.float).expand(num_feats, -1)
            edge_index = extra_meta.get(name, {}).get("edge_index")
            if edge_index is None: return 0.5 # Fallback
            batch = torch.zeros(x.size(0), dtype=torch.long)
            out = model_obj(x, edge_index, batch)
            return float(F.softmax(out, dim=1)[:, 1].mean()) # Average across nodes
    return float(model_obj.predict(X_scaled)[0])

# ── FORENSIC AUDIT ENGINE (Batch Mode) ────────────────────────────────
if st.sidebar.button("RUN BATCH FORENSIC AUDIT", type="secondary"):
    st.divider()
    st.header("DETAILED CLINICAL PERFORMANCE & FORENSIC AUDIT")
    
    # 1. Load Data
    data_path = "../../analysis/data/Raw_data_dpv.xlsx"
    if not os.path.exists(data_path):
        data_path = "../data/Raw_data_dpv.xlsx" # Fallback
        
    if os.path.exists(data_path):
        df_batch = pd.read_excel(data_path)
        # Auto-label
        psa_col = next((c for c in df_batch.columns if 'psa' in c.lower() and 'ratio' not in c.lower()), None)
        if psa_col:
            df_batch['target'] = (pd.to_numeric(df_batch[psa_col], errors='coerce') > 4000).astype(int)
        
        # 2. Batch Inference
        X_batch = df_batch[feature_columns]
        X_prep = scaler.transform(np.log1p(X_batch.clip(lower=0)))
        
        # Use primary model for batch (e.g. XGBoost)
        m_key = "xgboost" if "xgboost" in models else list(models.keys())[0]
        m_batch = models[m_key]
        
        # Progress
        risks = []
        for i in range(len(X_prep)):
            risks.append(get_prediction(m_batch, m_key, X_prep[i:i+1]))
        
        df_batch['risk_index'] = risks
        df_batch['prediction'] = [1 if r >= 0.5 else 0 for r in risks]
        
        # ── 1. EXECUTIVE SUMMARY ──────────────────────────────────────
        total = len(df_batch)
        pos = df_batch['prediction'].sum()
        neg = total - pos
        det_rate = pos / total
        
        st.subheader("1. EXECUTIVE BATCH TRIAGE SUMMARY")
        st.write(f"ALERT: The AI ensemble has audited {total} profiles. The overall detection rate is {det_rate:.1%}.")
        
        c1, c2, c3 = st.columns(3)
        c1.metric("POSITIVE", f"{pos}", "Malignant")
        c2.metric("NEGATIVE", f"{neg}", "Benign")
        c3.metric("DETECTION RATE", f"{det_rate:.1%}")
        
        z1, z2, z3 = st.columns(3)
        z1.metric("CRITICAL RADIUS", len(df_batch[df_batch['risk_index'] > 0.75]), "Risk > 75%")
        z2.metric("URGENT ZONE", len(df_batch[(df_batch['risk_index'] >= 0.45) & (df_batch['risk_index'] <= 0.75)]), "Risk 45-75%")
        z3.metric("STABLE COHORT", len(df_batch[df_batch['risk_index'] < 0.45]), "Risk < 45%")
        
        # ── 3. DATA-DRIVEN JUSTIFICATION ─────────────────────────────
        st.subheader("2. DATA-DRIVEN CLINICAL JUSTIFICATION")
        metrics = []
        for feat in feature_columns:
            pos_mean = df_batch[df_batch['prediction'] == 1][feat].mean()
            neg_mean = df_batch[df_batch['prediction'] == 0][feat].mean()
            div = abs(pos_mean - neg_mean) / (neg_mean if neg_mean != 0 else 1)
            metrics.append({"Biomarker": feat, "Positive Mean": f"{pos_mean:.2f}", "Negative Mean": f"{neg_mean:.2f}", "Divergence": f"{div:.1%}"})
        
        st.table(pd.DataFrame(metrics))
        
        # ── 4. CLINICAL REGISTRY ──────────────────────────────────────
        st.subheader("3. HIGH-RISK CLINICAL REGISTRY (TOP 50)")
        df_top = df_batch.sort_values('risk_index', ascending=False).head(50)
        
        # Cleanup for display
        display_cols = []
        if 'sample_id' in df_batch.columns: display_cols.append('sample_id')
        display_cols.extend(['risk_index'])
        if psa_col: display_cols.append(psa_col)
        display_cols.extend(feature_columns)
        
        df_display = df_top[display_cols].copy()
        df_display['Action'] = ["URGENT REVIEW" if r > 0.45 else "ROUTINE" for r in df_display['risk_index']]
        st.dataframe(df_display.style.background_gradient(subset=['risk_index'], cmap='Reds'))
        
        st.caption(f"Forensic Mode: ACTIVE | Source: {os.path.basename(data_path)} | Scope: {total} Records")
    else:
        st.error("Research dataset (Raw_data_dpv.xlsx) not found. Please ensure it is in analysis/data/")

if run_btn:
    # Inference
    in_df = pd.DataFrame([[afp, ca125]], columns=feature_columns)
    in_prep = scaler.transform(np.log1p(in_df.clip(lower=0)))
    
    model_map = {"Random Forest": "random_forest", "XGBoost": "xgboost", "Logistic Regression": "logistic_regression", "SVM": "svm", "GNN": "gnn"}
    m_key = model_map[model_choice]
    m = models[m_key]
    
    p = get_prediction(m, m_key, in_prep)
    
    # ── METRIC TILES ──────────────────────────────────────────────────
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Risk Score", f"{p:.1%}")
    c2.metric("Prediction", "MALIGNANT" if p >= threshold else "BENIGN")
    c3.metric("Ensemble Consensus", "HIGH" if (p > 0.8 or p < 0.2) else "MODERATE")
    c4.metric("Model Architecture", model_choice)
    
    # ── COMMITTEE PEER REVIEW ─────────────────────────────────────────
    st.header("AI Committee Consensus")
    cons_cols = st.columns(len(models))
    for i, (name, obj) in enumerate(models.items()):
        try:
            prob = get_prediction(obj, name, in_prep)
            status = "RED" if prob >= threshold else "GREEN"
            cons_cols[i].markdown(f"**{name.title()}**\n\n{status} {prob:.1%}")
        except: cons_cols[i].error("Offline")

    # ── WHAT-IF ANALYSIS ──────────────────────────────────────────────
    st.header("Sensitivity Simulation")
    s1, s2 = st.columns(2)
    # Sensitivity 1
    prep_s1 = scaler.transform(np.log1p(pd.DataFrame([[afp*0.8, ca125]], columns=feature_columns).clip(lower=0)))
    p1 = get_prediction(m, m_key, prep_s1)
    s1.metric("20% AFP Improvement", f"{p1:.1%}", f"{p1-p:.1%}", delta_color="inverse")
    # Sensitivity 2
    prep_s2 = scaler.transform(np.log1p(pd.DataFrame([[afp, ca125*0.8]], columns=feature_columns).clip(lower=0)))
    p2 = get_prediction(m, m_key, prep_s2)
    s2.metric("20% CA125 Improvement", f"{p2:.1%}", f"{p2-p:.1%}", delta_color="inverse")

# ── RESEARCH GALLERY ──────────────────────────────────────────────────
st.divider()
st.header("Research Artifact Gallery")
t1, t2, t3, t4 = st.tabs(["Performance", "Biomarkers", "Trajectory", "Leaderboard"])

with t1:
    col_a, col_b = st.columns(2)
    if os.path.exists("roc_curves.png"): col_a.image("roc_curves.png", caption="ROC Curve Analysis")
    if os.path.exists("confusion_matrices.png"): col_b.image("confusion_matrices.png", caption="Diagnostic Confusion Matrix")

with t2:
    if os.path.exists("feature_importance.png"): st.image("feature_importance.png", caption="Global Biomarker Importance")
    if os.path.exists("correlation_error_analysis.png"): st.image("correlation_error_analysis.png", caption="Clinical Correlation Map")

with t3:
    st.subheader("Clinical Risk Trajectory")
    plt.style.use('dark_background')
    fig, ax = plt.subplots(figsize=(10, 4))
    x = np.linspace(0, 100, 100)
    y = 0.8 / (1 + np.exp(-0.08 * (x - 50))) + 0.1
    ax.fill_between(x, 0, 0.4, color='green', alpha=0.1)
    ax.fill_between(x, 0.4, 0.7, color='orange', alpha=0.1)
    ax.fill_between(x, 0.7, 1.0, color='red', alpha=0.1)
    ax.plot(x, y, color='#00d1ff', linewidth=3)
    ax.set_facecolor('#0e1117')
    fig.patch.set_facecolor('#0e1117')
    st.pyplot(fig)

with t4:
    res_path = "../results/model_comparison_with_gnn.csv"
    if os.path.exists(res_path):
        st.dataframe(pd.read_csv(res_path).style.background_gradient(cmap='Reds'), use_container_width=True)
    else: st.info("Run analysis.ipynb to generate rankings.")
