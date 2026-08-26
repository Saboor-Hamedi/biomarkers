"""
AI-Assisted Decoding of Multiplexed Electrochemical Fingerprints
================================================================
Regression pipeline for quantitative decoding of PSA, AFP and CA125
concentrations from raw 200-point DPV fingerprints.

Scientific question (per supervisor instruction):
    Can a single multiplexed DPV fingerprint be computationally decoded
    into simultaneous quantitative estimates of PSA, AFP and CA125?

NOT clinical diagnosis. These are experimental spiked serum samples.
The classification result (PSA-threshold) is treated as a SECONDARY
application only.

Phases implemented:
    P1  Data audit & preparation
    P2  DPV signal understanding (visualization)
    P3  Biomarker-specific correlation analysis
    P4  Single-biomarker regression (7 models)
    P5  Multi-output regression (central experiment)
    P6  Raw DPV vs engineered features vs biomarker regions
    P7  Ablation study
    P8  Cross-biomarker interference check
    P9  Strict train/validation/test + out-of-fold + bootstrap uncertainty
    P10 SHAP per biomarker
"""

import os
import pickle
import sys
import warnings
from math import pi

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.cross_decomposition import PLSRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import RobustScaler, StandardScaler
from sklearn.svm import SVR
from xgboost import XGBRegressor

# PyTorch 1D-CNN regression head (model #7 in the hierarchy)
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    TORCH_OK = True
except ImportError:
    TORCH_OK = False

warnings.filterwarnings("ignore")

BASE = os.path.join(os.path.dirname(__file__), "..")
DATA_PATH = os.path.join(BASE, "data", "Raw_DPV_Dataset_for_Cancer_biomarker.csv")
FEAT_PATH = os.path.join(BASE, "data", "data_with_features_engineered.csv")
RESULT_DIR = os.path.join(BASE, "results")
FIGURE_DIR = os.path.join(BASE, "figure")
MODEL_DIR = os.path.join(BASE, "models")

BIOMARKERS = ["PSA_pg_per_ml", "AFP_pg_per_ml", "CA125_U_per_ml"]
# Experimentally validated potential regions (mV) from the biosensor team
REGION_PSA = (-468, -448)
REGION_AFP = (365, 385)
REGION_CA125 = (958, 978)

SEED = 42


# ============================================================================
# P1 - DATA AUDIT & PREPARATION
# ============================================================================
def load_data():
    """Load raw DPV dataset and separate X (currents), Y (concentrations)."""
    df = pd.read_csv(DATA_PATH)
    dpv_cols = [c for c in df.columns if c.startswith("curr_")]
    voltages = np.array([float(c.replace("curr_", "").replace("mV", "")) for c in dpv_cols])
    X = df[dpv_cols].values
    Y = df[BIOMARKERS].values.astype(float)
    return df, X, Y, voltages, dpv_cols


def audit_data(df, X, Y, voltages):
    """Print a formal data audit."""
    print("\n" + "=" * 70)
    print("DATA AUDIT REPORT")
    print("=" * 70)
    print(f"  Samples:                {len(df)}")
    print(f"  Unique sample IDs:      {df['sample_id'].nunique()}")
    print(f"  Duplicate sample IDs:   {int(df['sample_id'].duplicated().sum())}")
    print(f"  DPV measurements/sample:{X.shape[1]}")
    print(f"  Missing values:         {int(df.isnull().sum().sum())}")
    print(f"  Potential range (mV):   {voltages[0]:.0f} to {voltages[-1]:.0f}")
    print(f"  Potential step (mV):    {voltages[1]-voltages[0]:.1f}")
    print("\n  Concentration ranges:")
    for b in BIOMARKERS:
        v = df[b].values
        print(f"    {b:<16} min={v.min():.4g}  max={v.max():.4g}  log10 range=[{np.log10(max(v.min(),1e-9)):.1f},{np.log10(v.max()):.1f}]")


# ============================================================================
# P2 - DPV VISUALIZATION
# ============================================================================
def plot_dpv_overview(df, X, voltages, dpv_cols):
    """Produce representative, overlay and mean±SD fingerprint figures."""
    os.makedirs(FIGURE_DIR, exist_ok=True)

    # A: 30 representative individual curves
    fig, ax = plt.subplots(figsize=(12, 6), dpi=200)
    for i in range(30):
        ax.plot(voltages, X[i], lw=0.8, alpha=0.6)
    ax.set_xlabel("Applied Potential E (mV)")
    ax.set_ylabel("Current (µA)")
    ax.set_title("Figure A - Representative Individual DPV Curves (n=30)")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGURE_DIR, "P2A_representative_curves.png"))
    plt.close(fig)

    # B: all 1000 curves transparent overlay
    fig, ax = plt.subplots(figsize=(12, 6), dpi=200)
    for i in range(len(X)):
        ax.plot(voltages, X[i], lw=0.2, alpha=0.05, color="#1f77b4")
    ax.set_xlabel("Applied Potential E (mV)")
    ax.set_ylabel("Current (µA)")
    ax.set_title("Figure B - All 1000 DPV Curves Overlay")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGURE_DIR, "P2B_all_curves_overlay.png"))
    plt.close(fig)

    # C: mean ± SD
    mean = X.mean(axis=0)
    sd = X.std(axis=0)
    fig, ax = plt.subplots(figsize=(12, 6), dpi=200)
    ax.plot(voltages, mean, color="#1f77b4", lw=2, label="Mean")
    ax.fill_between(voltages, mean - sd, mean + sd, color="#1f77b4", alpha=0.2, label="±1 SD")
    for r, name, c in [(REGION_PSA, "PSA", "#2ca02c"), (REGION_AFP, "AFP", "#ff7f0e"), (REGION_CA125, "CA125", "#9467bd")]:
        ax.axvspan(r[0], r[1], color=c, alpha=0.15)
        ax.axvline(r[0], color=c, ls="--", lw=1)
    ax.set_xlabel("Applied Potential E (mV)")
    ax.set_ylabel("Current (µA)")
    ax.set_title("Figure C - Mean ± SD DPV Fingerprint (n=1000)")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGURE_DIR, "P2C_mean_sd_fingerprint.png"))
    plt.close(fig)

    # D/E/F: concentration-group curves per biomarker
    for b, label in zip(BIOMARKERS, ["D - PSA Groups", "E - AFP Groups", "F - CA125 Groups"]):
        fig, ax = plt.subplots(figsize=(12, 6), dpi=200)
        q = pd.qcut(df[b], q=3, labels=["Low", "Mid", "High"])
        for g, color in zip(["Low", "Mid", "High"], ["#2ca02c", "#ff7f0e", "#d62728"]):
            idx = q[q == g].index
            ax.plot(voltages, X[idx].mean(axis=0), lw=2, label=f"{g} ({len(idx)})", color=color)
        ax.set_xlabel("Applied Potential E (mV)")
        ax.set_ylabel("Mean Current (µA)")
        ax.set_title(f"Figure {label}")
        ax.legend()
        ax.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(os.path.join(FIGURE_DIR, f"P2{label.replace(' - ', '_').replace(' ', '_')}.png"))
        plt.close(fig)


# ============================================================================
# P3 - BIOMARKER-SPECIFIC CORRELATION
# ============================================================================
def biomarker_correlation(df, X, voltages, dpv_cols):
    """Pearson and Spearman correlation of DPV current vs each biomarker (log-scale)."""
    print("\n" + "=" * 70)
    print("BIOMARKER-SPECIFIC CORRELATION (DPV vs log-concentration)")
    print("=" * 70)

    rows = []
    for b in BIOMARKERS:
        y = np.log10(df[b].values.clip(min=1e-9))
        pearson = np.array([stats.pearsonr(X[:, i], y)[0] for i in range(X.shape[1])])
        spearman = np.array([stats.spearmanr(X[:, i], y).correlation for i in range(X.shape[1])])

        # strongest correlation point
        i_best = int(np.argmax(np.abs(pearson)))
        rows.append({
            "biomarker": b,
            "peak_potential_mV": voltages[i_best],
            "pearson_r": pearson[i_best],
            "spearman_rho": spearman[i_best],
        })
        print(f"  {b:<16} strongest |r| = {pearson[i_best]:.4f} at {voltages[i_best]:.0f} mV"
              f" (Spearman {spearman[i_best]:.4f})")

    return pd.DataFrame(rows)


# ============================================================================
# MODEL DEFINITIONS
# ============================================================================
def get_models():
    """Return the model hierarchy: Linear, PLS, RF, SVR, XGB, MLP."""
    return {
        "Linear": LinearRegression(),
        "PLS": PLSRegression(n_components=20),
        "RandomForest": RandomForestRegressor(n_estimators=200, random_state=SEED, n_jobs=-1),
        "SVR": SVR(kernel="rbf", C=10.0, gamma="scale"),
        "XGBoost": XGBRegressor(n_estimators=300, learning_rate=0.05, max_depth=6, random_state=SEED, n_jobs=-1),
        "MLP": MLPRegressor(hidden_layer_sizes=(128, 64), max_iter=1000, random_state=SEED),
    }


def regression_metrics(y_true, y_pred):
    """Return R2, MAE, RMSE, Pearson r, Spearman rho."""
    r2 = r2_score(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    r_p = float(np.corrcoef(y_true, y_pred)[0, 1]) if np.std(y_pred) > 0 else 0.0
    rho = float(stats.spearmanr(y_true, y_pred).correlation) if np.std(y_pred) > 0 else 0.0
    return {"R2": r2, "MAE": mae, "RMSE": rmse, "Pearson_r": r_p, "Spearman": rho}


# ============================================================================
# 1D-CNN REGRESSION HEAD (model #7)
# ============================================================================
class CNN1DRegressor(nn.Module):
    """1D-CNN for quantitative regression from the 200-point DPV fingerprint."""

    def __init__(self, input_length=200, hidden_channels=64, n_outputs=1):
        super(CNN1DRegressor, self).__init__()
        self.input_length = input_length
        self.conv1 = nn.Conv1d(1, hidden_channels, kernel_size=7, padding=3)
        self.bn1 = nn.BatchNorm1d(hidden_channels)
        self.conv2 = nn.Conv1d(hidden_channels, hidden_channels * 2, kernel_size=5, padding=2)
        self.bn2 = nn.BatchNorm1d(hidden_channels * 2)
        self.conv3 = nn.Conv1d(hidden_channels * 2, hidden_channels * 2, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm1d(hidden_channels * 2)
        self.pool = nn.MaxPool1d(2)
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.dropout = nn.Dropout(0.2)
        self.fc = nn.Linear(hidden_channels * 2, n_outputs)

    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.global_pool(x).squeeze(-1)
        x = self.dropout(x)
        return self.fc(x)


def _train_cnn_reg(model, X_train, y_train, X_val, y_val=None, epochs=60, lr=0.001, batch_size=64, seed=42, patience=8):
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    X_tr = torch.FloatTensor(np.asarray(X_train, dtype=np.float32)).to(device)
    X_va = torch.FloatTensor(np.asarray(X_val, dtype=np.float32)).to(device)
    y_tr = torch.FloatTensor(np.asarray(y_train, dtype=np.float32)).view(-1, 1).to(device)
    y_va = (torch.FloatTensor(np.asarray(y_val, dtype=np.float32)).view(-1, 1).to(device)
            if y_val is not None else None)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    n = len(X_tr)
    best_val = float("inf")
    best_state = None
    no_impr = 0

    for epoch in range(epochs):
        model.train()
        perm = torch.randperm(n, device=device)
        for i in range(0, n, batch_size):
            idx = perm[i:i + batch_size]
            optimizer.zero_grad()
            out = model(X_tr[idx])
            loss = criterion(out, y_tr[idx])
            loss.backward()
            optimizer.step()
        if y_va is not None:
            model.eval()
            with torch.no_grad():
                val_loss = criterion(model(X_va), y_va).item()
            if val_loss < best_val:
                best_val = val_loss
                best_state = {k: v.clone() for k, v in model.state_dict().items()}
                no_impr = 0
            else:
                no_impr += 1
                if no_impr >= patience:
                    break
    if best_state is not None:
        model.load_state_dict(best_state)
    model.to(torch.device("cpu"))
    return model


def cnn_reg_predict(model, X):
    model.eval()
    Xt = torch.FloatTensor(np.asarray(X, dtype=np.float32))
    with torch.no_grad():
        return model(Xt).numpy().ravel()


def train_cnn_regression(X_train, y_train, X_val, n_outputs=1, epochs=60):
    """Train a 1D-CNN regressor and return out-of-sample predictions (internal 80/20 split)."""
    if not TORCH_OK:
        return None
    # internal validation split for early stopping
    from sklearn.model_selection import train_test_split as tts
    Xtr, Xva, ytr, yva = tts(X_train, y_train, test_size=0.2, random_state=SEED)
    model = CNN1DRegressor(input_length=X_train.shape[1], n_outputs=n_outputs)
    model = _train_cnn_reg(model, Xtr, ytr, Xva, epochs=epochs)
    return model


# ============================================================================
# P4 - SINGLE-BIOMARKER REGRESSION (with strict CV)
# ============================================================================
def run_single_regression(X, Y, df, cv=5):
    """Regress each biomarker (log10) from the full 200-point DPV fingerprint."""
    print("\n" + "=" * 70)
    print("P4 - SINGLE-BIOMARKER REGRESSION (log10 concentration)")
    print("=" * 70)

    models = get_models()
    # Log-transform targets
    Y_log = np.log10(Y.clip(min=1e-9))

    all_rows = []
    for bi, b in enumerate(BIOMARKERS):
        y = Y_log[:, bi]
        kf = KFold(n_splits=cv, shuffle=True, random_state=SEED)
        for name, model in models.items():
            oof = np.zeros(len(y))
            for tr, va in kf.split(X):
                # scale INSIDE fold to prevent leakage
                sc = StandardScaler().fit(X[tr])
                Xtr, Xva = sc.transform(X[tr]), sc.transform(X[va])
                m = model.__class__(**model.get_params()) if hasattr(model, "get_params") else model
                m.fit(Xtr, y[tr])
                oof[va] = m.predict(Xva)
            met = regression_metrics(y, oof)
            met.update({"biomarker": b, "model": name})
            all_rows.append(met)
            print(f"  {b:<16} {name:<12} R2={met['R2']:.4f} MAE={met['MAE']:.4f} "
                  f"RMSE={met['RMSE']:.4f} r={met['Pearson_r']:.4f} rho={met['Spearman']:.4f}")

        # 1D-CNN regression head
        if TORCH_OK:
            oof_cnn = np.zeros(len(y))
            for tr, va in kf.split(X):
                sc = StandardScaler().fit(X[tr])
                Xtr, Xva = sc.transform(X[tr]), sc.transform(X[va])
                model = CNN1DRegressor(input_length=X.shape[1], n_outputs=1)
                model = _train_cnn_reg(model, Xtr, y[tr], Xva, y_val=y[va], epochs=60)
                oof_cnn[va] = cnn_reg_predict(model, Xva)
            met = regression_metrics(y, oof_cnn)
            met.update({"biomarker": b, "model": "1D-CNN"})
            all_rows.append(met)
            print(f"  {b:<16} {'1D-CNN':<12} R2={met['R2']:.4f} MAE={met['MAE']:.4f} "
                  f"RMSE={met['RMSE']:.4f} r={met['Pearson_r']:.4f} rho={met['Spearman']:.4f}")

    return pd.DataFrame(all_rows)


# ============================================================================
# P5 - MULTI-OUTPUT REGRESSION (central experiment)
# ============================================================================
def run_multi_output(X, Y):
    """One fingerprint -> PSA + AFP + CA125 simultaneously (log10 targets)."""
    print("\n" + "=" * 70)
    print("P5 - MULTI-OUTPUT REGRESSION (single fingerprint -> PSA+AFP+CA125)")
    print("=" * 70)

    from sklearn.multioutput import MultiOutputRegressor

    Y_log = np.log10(Y.clip(min=1e-9))
    models = get_models()
    kf = KFold(n_splits=5, shuffle=True, random_state=SEED)

    rows = []
    for name, model in models.items():
        oof = np.zeros_like(Y_log)
        for tr, va in kf.split(X):
            sc = StandardScaler().fit(X[tr])
            Xtr, Xva = sc.transform(X[tr]), sc.transform(X[va])
            m = MultiOutputRegressor(model.__class__(**model.get_params()))
            m.fit(Xtr, Y_log[tr])
            oof[va] = m.predict(Xva)
        for bi, b in enumerate(BIOMARKERS):
            met = regression_metrics(Y_log[:, bi], oof[:, bi])
            met.update({"biomarker": b, "model": name, "output": "multi"})
            rows.append(met)
        print(f"  {name:<12} PSA R2={regression_metrics(Y_log[:,0], oof[:,0])['R2']:.4f} "
              f"AFP R2={regression_metrics(Y_log[:,1], oof[:,1])['R2']:.4f} "
              f"CA125 R2={regression_metrics(Y_log[:,2], oof[:,2])['R2']:.4f}")

    # Multi-output 1D-CNN (one fingerprint -> PSA+AFP+CA125 simultaneously)
    if TORCH_OK:
        oof = np.zeros_like(Y_log)
        for tr, va in kf.split(X):
            sc = StandardScaler().fit(X[tr])
            Xtr, Xva = sc.transform(X[tr]), sc.transform(X[va])
            model = CNN1DRegressor(input_length=X.shape[1], n_outputs=3)
            # train on all 3 outputs with MSE
            model = _train_multi_cnn(model, Xtr, Y_log[tr], Xva, Y_val=Y_log[va], epochs=60)
            oof[va] = model_predict_multi(model, Xva)
        for bi, b in enumerate(BIOMARKERS):
            met = regression_metrics(Y_log[:, bi], oof[:, bi])
            met.update({"biomarker": b, "model": "1D-CNN", "output": "multi"})
            rows.append(met)
        print(f"  {'1D-CNN':<12} PSA R2={regression_metrics(Y_log[:,0], oof[:,0])['R2']:.4f} "
              f"AFP R2={regression_metrics(Y_log[:,1], oof[:,1])['R2']:.4f} "
              f"CA125 R2={regression_metrics(Y_log[:,2], oof[:,2])['R2']:.4f}")

    return pd.DataFrame(rows)


def _train_multi_cnn(model, X_train, Y_train, X_val, Y_val=None, epochs=60, lr=0.001, batch_size=64, seed=42, patience=8):
    """Train the 1D-CNN regressor on multiple outputs simultaneously."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    X_tr = torch.FloatTensor(np.asarray(X_train, dtype=np.float32)).to(device)
    X_va = torch.FloatTensor(np.asarray(X_val, dtype=np.float32)).to(device)
    Y_tr = torch.FloatTensor(np.asarray(Y_train, dtype=np.float32)).to(device)
    Y_va = (torch.FloatTensor(np.asarray(Y_val, dtype=np.float32)).to(device)
            if Y_val is not None else None)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    n = len(X_tr)
    best_val = float("inf")
    best_state = None
    no_impr = 0

    for epoch in range(epochs):
        model.train()
        perm = torch.randperm(n, device=device)
        for i in range(0, n, batch_size):
            idx = perm[i:i + batch_size]
            optimizer.zero_grad()
            out = model(X_tr[idx])
            loss = criterion(out, Y_tr[idx])
            loss.backward()
            optimizer.step()
        if Y_va is not None:
            model.eval()
            with torch.no_grad():
                val_loss = criterion(model(X_va), Y_va).item()
            if val_loss < best_val:
                best_val = val_loss
                best_state = {k: v.clone() for k, v in model.state_dict().items()}
                no_impr = 0
            else:
                no_impr += 1
                if no_impr >= patience:
                    break
    if best_state is not None:
        model.load_state_dict(best_state)
    model.to(torch.device("cpu"))
    return model


def model_predict_multi(model, X):
    model.eval()
    Xt = torch.FloatTensor(np.asarray(X, dtype=np.float32))
    with torch.no_grad():
        return model(Xt).numpy()


# ============================================================================
# P6 - RAW DPV vs ENGINEERED FEATURES vs BIOMARKER REGIONS
# ============================================================================
def region_indices(voltages, region):
    return np.where((voltages >= region[0]) & (voltages <= region[1]))[0]


def run_representation_comparison(df, X, Y, voltages):
    """Compare prediction using: full DPV, engineered features, biomarker regions."""
    print("\n" + "=" * 70)
    print("P6 - RAW DPV vs ENGINEERED FEATURES vs BIOMARKER REGIONS")
    print("=" * 70)

    feats = pd.read_csv(FEAT_PATH)
    feat_cols = ["peak_anodic_current", "peak_anodic_potential", "peak_cathodic_current",
                 "peak_cathodic_potential", "area_under_curve", "peak_separation"]
    X_feat = feats[feat_cols].values

    # region-based inputs: concatenate the three biomarker region windows
    idx = np.concatenate([region_indices(voltages, REGION_PSA),
                          region_indices(voltages, REGION_AFP),
                          region_indices(voltages, REGION_CA125)])
    X_regions = X[:, idx]

    representations = {"full_dpv_200": X, "engineered_6": X_feat, "regions_only": X_regions}
    Y_log = np.log10(Y.clip(min=1e-9))

    rows = []
    for rep_name, Xr in representations.items():
        # handle NaN in engineered features (drop rows with NaN for this rep)
        nan_rows = np.isnan(Xr).any(axis=1)
        for bi, b in enumerate(BIOMARKERS):
            y = Y_log[:, bi]
            keep = ~nan_rows
            if keep.sum() < len(y):
                print(f"  Note: {rep_name} dropping {int((~keep).sum())} rows with missing features")
            kf = KFold(n_splits=5, shuffle=True, random_state=SEED)
            oof = np.zeros(len(y)) * np.nan
            for tr, va in kf.split(Xr[keep]):
                sc = StandardScaler().fit(Xr[keep][tr])
                m = LinearRegression()
                m.fit(sc.transform(Xr[keep][tr]), y[keep][tr])
                oof_idx = np.where(keep)[0][va]
                oof[oof_idx] = m.predict(sc.transform(Xr[keep][va]))
            met = regression_metrics(y[keep], oof[keep])
            met.update({"representation": rep_name, "biomarker": b, "model": "Linear"})
            rows.append(met)
            print(f"  {rep_name:<12} {b:<16} R2={met['R2']:.4f} r={met['Pearson_r']:.4f}")

    return pd.DataFrame(rows)


# ============================================================================
# P7 - ABLATION STUDY
# ============================================================================
def run_ablation(X, Y, voltages):
    """Ablation over biomarker regions and the full fingerprint."""
    print("\n" + "=" * 70)
    print("P7 - ABLATION STUDY")
    print("=" * 70)

    i_psa = region_indices(voltages, REGION_PSA)
    i_afp = region_indices(voltages, REGION_AFP)
    i_ca125 = region_indices(voltages, REGION_CA125)

    combos = {
        "PSA_region_only": i_psa,
        "AFP_region_only": i_afp,
        "CA125_region_only": i_ca125,
        "PSA+AFP": np.concatenate([i_psa, i_afp]),
        "PSA+CA125": np.concatenate([i_psa, i_ca125]),
        "AFP+CA125": np.concatenate([i_afp, i_ca125]),
        "All_three_regions": np.concatenate([i_psa, i_afp, i_ca125]),
        "Full_200_point": np.arange(X.shape[1]),
    }
    Y_log = np.log10(Y.clip(min=1e-9))
    kf = KFold(n_splits=5, shuffle=True, random_state=SEED)

    rows = []
    for combo, idx in combos.items():
        Xc = X[:, idx]
        for bi, b in enumerate(BIOMARKERS):
            y = Y_log[:, bi]
            oof = np.zeros(len(y))
            for tr, va in kf.split(Xc):
                sc = StandardScaler().fit(Xc[tr])
                m = LinearRegression()
                m.fit(sc.transform(Xc[tr]), y[tr])
                oof[va] = m.predict(sc.transform(Xc[va]))
            met = regression_metrics(y, oof)
            met.update({"combination": combo, "biomarker": b})
            rows.append(met)
        r2_avg = np.mean([regression_metrics(Y_log[:, bi], _ablation_oof(X, idx, bi, kf))["R2"] for bi in range(3)])
        print(f"  {combo:<16} avg R2={r2_avg:.4f}")

    return pd.DataFrame(rows)


def _ablation_oof(X, idx, bi, kf):
    Y_local = _GLOBAL_YLOG
    y = Y_local[:, bi]
    Xc = X[:, idx]
    oof = np.zeros(len(y))
    for tr, va in kf.split(Xc):
        sc = StandardScaler().fit(Xc[tr])
        m = LinearRegression()
        m.fit(sc.transform(Xc[tr]), y[tr])
        oof[va] = m.predict(sc.transform(Xc[va]))
    return oof


# ============================================================================
# P8 - CROSS-BIOMARKER INTERFERENCE
# ============================================================================
def run_interference(X, Y, voltages):
    """Check whether varying one biomarker alters prediction of another.

    Strategy: for a given target biomarker, train the model only on the
    OTHER two biomarker regions + the target region, then evaluate how
    removing / adding each region changes target R2. Because the samples
    independently vary all three biomarkers, region-specific ablations
    reveal cross-biomarker information coupling.
    """
    print("\n" + "=" * 70)
    print("P8 - CROSS-BIOMARKER INTERFERENCE")
    print("=" * 70)

    i_psa = region_indices(voltages, REGION_PSA)
    i_afp = region_indices(voltages, REGION_AFP)
    i_ca125 = region_indices(voltages, REGION_CA125)
    region_map = {"PSA": i_psa, "AFP": i_afp, "CA125": i_ca125}
    biomarker_idx = {"PSA": BIOMARKERS.index("PSA_pg_per_ml"),
                     "AFP": BIOMARKERS.index("AFP_pg_per_ml"),
                     "CA125": BIOMARKERS.index("CA125_U_per_ml")}
    Y_log = np.log10(Y.clip(min=1e-9))
    kf = KFold(n_splits=5, shuffle=True, random_state=SEED)

    rows = []
    for target, t_idx in region_map.items():
        others = {k: v for k, v in region_map.items() if k != target}
        for k, v in others.items():
            # target region + one other region
            Xc = X[:, np.concatenate([t_idx, v])]
            bi = biomarker_idx[target]
            y = Y_log[:, bi]
            oof = np.zeros(len(y))
            for tr, va in kf.split(Xc):
                sc = StandardScaler().fit(Xc[tr])
                m = LinearRegression()
                m.fit(sc.transform(Xc[tr]), y[tr])
                oof[va] = m.predict(sc.transform(Xc[va]))
            met = regression_metrics(y, oof)
            met.update({"target": target, "interfering_region": k, "R2_target_plus_interf": met["R2"]})
            rows.append(met)
            print(f"  {target:<6} with {k:<6} region: R2={met['R2']:.4f} r={met['Pearson_r']:.4f}")

    return pd.DataFrame(rows)


# ============================================================================
# P9 - STRICT SPLIT + BOOTSTRAP UNCERTAINTY
# ============================================================================
def strict_split_and_bootstrap(X, Y, test_size=0.20, n_boot=200):
    """Hold-out test set + bootstrap confidence intervals on the test R2."""
    print("\n" + "=" * 70)
    print("P9 - STRICT SPLIT + BOOTSTRAP UNCERTAINTY")
    print("=" * 70)

    Y_log = np.log10(Y.clip(min=1e-9))

    X_train, X_test, Ytr, Yte = train_test_split(X, Y_log, test_size=test_size, random_state=SEED)
    sc = StandardScaler().fit(X_train)
    Xtr_s, Xte_s = sc.transform(X_train), sc.transform(X_test)

    rows = []
    for bi, b in enumerate(BIOMARKERS):
        model = XGBRegressor(n_estimators=200, learning_rate=0.05, max_depth=6, random_state=SEED)
        model.fit(Xtr_s, Ytr[:, bi])
        pred = model.predict(Xte_s)
        met = regression_metrics(Yte[:, bi], pred)

        # bootstrap CIs on R2
        boot_r2 = []
        rng = np.random.default_rng(SEED)
        for _ in range(n_boot):
            idx = rng.integers(0, len(pred), len(pred))
            boot_r2.append(r2_score(Yte[:, bi][idx], pred[idx]))
        boot_r2 = np.array(boot_r2)
        ci_lo, ci_hi = np.percentile(boot_r2, [2.5, 97.5])
        met.update({"biomarker": b, "model": "XGBoost", "test_R2_95CI_lo": ci_lo, "test_R2_95CI_hi": ci_hi})
        rows.append(met)
        print(f"  {b:<16} TEST R2={met['R2']:.4f} (95% CI [{ci_lo:.4f},{ci_hi:.4f}]) "
              f"MAE={met['MAE']:.4f} RMSE={met['RMSE']:.4f} r={met['Pearson_r']:.4f}")

    return pd.DataFrame(rows)


# ============================================================================
# P10 - SHAP PER BIOMARKER
# ============================================================================
def run_shap(X, Y, voltages):
    """SHAP feature importance per biomarker mapped against potential."""
    print("\n" + "=" * 70)
    print("P10 - SHAP PER BIOMARKER")
    print("=" * 70)
    try:
        import shap
    except ImportError:
        print("  SHAP not installed - skipping")
        return None

    Y_log = np.log10(Y.clip(min=1e-9))
    sc = StandardScaler().fit(X)
    Xs = sc.transform(X)

    results = {}
    for bi, b in enumerate(BIOMARKERS):
        model = XGBRegressor(n_estimators=200, random_state=SEED, n_jobs=-1)
        model.fit(Xs, Y_log[:, bi])
        explainer = shap.TreeExplainer(model)
        sh = explainer.shap_values(Xs[:200])
        mean_abs = np.abs(sh).mean(axis=0)
        results[b] = mean_abs
        top = np.argsort(mean_abs)[::-1][:5]
        print(f"  {b:<16} top importance voltages: {[float(voltages[i]) for i in top]}")
    return results


# ============================================================================
# PREDICTED vs MEASURED + RESIDUAL PLOTS (Phase 15)
# ============================================================================
def plot_pred_vs_measured(X, Y, model_name="XGBoost"):
    """Out-of-fold predicted vs measured + residual plots per biomarker."""
    os.makedirs(FIGURE_DIR, exist_ok=True)
    Y_log = np.log10(Y.clip(min=1e-9))
    kf = KFold(n_splits=5, shuffle=True, random_state=SEED)
    sc = StandardScaler().fit(X)
    Xs = sc.transform(X)

    for bi, b in enumerate(BIOMARKERS):
        y = Y_log[:, bi]
        model = XGBRegressor(n_estimators=300, learning_rate=0.05, max_depth=6, random_state=SEED, n_jobs=-1)
        oof = np.zeros(len(y))
        for tr, va in kf.split(Xs):
            model.fit(Xs[tr], y[tr])
            oof[va] = model.predict(Xs[va])

        fig, axes = plt.subplots(1, 2, figsize=(14, 6), dpi=200)
        # Predicted vs measured
        axes[0].scatter(y, oof, alpha=0.4, s=18, color="#1f77b4")
        lims = [min(y.min(), oof.min()), max(y.max(), oof.max())]
        axes[0].plot(lims, lims, "k--", lw=1.5, label="Perfect")
        axes[0].set_xlabel(f"Measured log10({b})")
        axes[0].set_ylabel(f"Predicted log10({b})")
        axes[0].set_title(f"Predicted vs Measured - {b}")
        axes[0].legend()
        axes[0].grid(alpha=0.3)
        # Residuals
        resid = y - oof
        axes[1].scatter(oof, resid, alpha=0.4, s=18, color="#d62728")
        axes[1].axhline(0, color="k", ls="--", lw=1.5)
        axes[1].set_xlabel(f"Predicted log10({b})")
        axes[1].set_ylabel("Residual")
        axes[1].set_title(f"Residuals - {b}")
        axes[1].grid(alpha=0.3)

        fig.tight_layout()
        safe = b.replace("_pg_per_ml", "").replace("_U_per_ml", "")
        fig.savefig(os.path.join(FIGURE_DIR, f"P15_pred_meas_{safe}.png"))
        plt.close(fig)
        print(f"  ✓ Saved: P15_pred_meas_{safe}.png")


# ============================================================================
# PUBLICATION FIGURES (Phase 20) - wide format for journal/report
# ============================================================================
def plot_publication_figures(multi_df, rep_df, abla_df, shap_results, voltages):
    """Generate 4 publication-ready wide figures from computed results."""
    os.makedirs(FIGURE_DIR, exist_ok=True)
    labels = {"PSA_pg_per_ml": "PSA", "AFP_pg_per_ml": "AFP", "CA125_U_per_ml": "CA125"}
    colors = {"PSA_pg_per_ml": "#2ca02c", "AFP_pg_per_ml": "#ff7f0e", "CA125_U_per_ml": "#9467bd"}
    region_of = {"PSA_pg_per_ml": REGION_PSA, "AFP_pg_per_ml": REGION_AFP, "CA125_U_per_ml": REGION_CA125}

    # ---- (a) MULTI-OUTPUT BAR CHART ----
    piv = multi_df[multi_df["output"] == "multi"].pivot_table(index="model", columns="biomarker", values="R2")
    order = [m for m in ["Linear", "PLS", "RandomForest", "SVR", "XGBoost", "MLP", "1D-CNN"] if m in piv.index]
    piv = piv.loc[order]
    fig, ax = plt.subplots(figsize=(14, 6), dpi=200)
    x = np.arange(len(piv.index))
    width = 0.26
    for k, b in enumerate(BIOMARKERS):
        bars = ax.bar(x + (k - 1) * width, piv[b], width, label=labels[b], color=colors[b], alpha=0.85, edgecolor='black', linewidth=1.2)
        for bar in bars:
            h = bar.get_height()
            if h > 0:
                ax.annotate(f'{h*100:.1f}%', (bar.get_x() + bar.get_width() / 2, h), ha='center', va='bottom', fontsize=10, xytext=(0, 4), textcoords='offset points')
    ax.set_xticks(x)
    ax.set_xticklabels(piv.index, rotation=15)
    ax.set_ylabel("R² (log₁₀ concentration)")
    ax.set_title("Multi-Output Regression: one DPV fingerprint → PSA + AFP + CA125")
    ax.legend(ncol=3, loc="upper left")
    ax.set_ylim(0, 1)
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(os.path.join(FIGURE_DIR, "P20A_multi_output.png"))
    plt.close(fig)
    print("  ✓ Saved: P20A_multi_output.png")

    # ---- (b) REPRESENTATION COMPARISON ----
    fig, ax = plt.subplots(figsize=(14, 6), dpi=200)
    reps = sorted(rep_df["representation"].unique())
    x = np.arange(len(reps))
    width = 0.26
    for k, b in enumerate(BIOMARKERS):
        vals = [rep_df[(rep_df["representation"] == r) & (rep_df["biomarker"] == b)]["R2"].mean() for r in reps]
        bars = ax.bar(x + (k - 1) * width, vals, width, label=labels[b], color=colors[b], alpha=0.85, edgecolor='black', linewidth=1.2)
        for bar in bars:
            h = bar.get_height()
            if h > 0:
                ax.annotate(f'{h*100:.1f}%', (bar.get_x() + bar.get_width() / 2, h), ha='center', va='bottom', fontsize=10, xytext=(0, 4), textcoords='offset points')
    ax.set_xticks(x)
    ax.set_xticklabels(reps, rotation=0)
    ax.set_ylabel("R² (log₁₀ concentration)")
    ax.set_title("Input Representation Comparison (Linear)")
    ax.legend(ncol=3)
    ax.set_ylim(0, 1)
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(os.path.join(FIGURE_DIR, "P20B_representations.png"))
    plt.close(fig)
    print("  ✓ Saved: P20B_representations.png")

    # ---- (c) ABLATION BARS ----
    avg = abla_df.groupby("combination")["R2"].mean().reindex(
        ["PSA_region_only", "AFP_region_only", "CA125_region_only",
         "PSA+AFP", "PSA+CA125", "AFP+CA125", "All_three_regions", "Full_200_point"])
    fig, ax = plt.subplots(figsize=(14, 6), dpi=200)
    bars = ax.bar(np.arange(len(avg)), avg.values, color="#1f77b4", alpha=0.85, edgecolor='black', linewidth=1.2)
    bars[-1].set_color("#d62728")
    bars[-2].set_color("#ff7f0e")
    ax.set_xticks(np.arange(len(avg)))
    ax.set_xticklabels(avg.index, rotation=20, ha="right")
    ax.set_ylabel("Average R² across biomarkers")
    ax.set_title("Ablation: information content of potential regions vs full fingerprint")
    ax.set_ylim(0, 1)
    ax.grid(alpha=0.3, axis="y")
    for i, v in enumerate(avg.values):
        ax.text(i, v + 0.015, f"{v*100:.1f}%", ha="center", fontsize=10)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGURE_DIR, "P20C_ablation.png"))
    plt.close(fig)
    print("  ✓ Saved: P20C_ablation.png")

    # ---- (d) SHAP-vs-POTENTIAL OVERLAY ----
    if shap_results is not None:
        fig, axes = plt.subplots(3, 1, figsize=(14, 10), dpi=200, sharex=True)
        for bi, b in enumerate(BIOMARKERS):
            ax = axes[bi]
            mean_abs = shap_results[b]
            ax.plot(voltages, mean_abs, color=colors[b], lw=1.8)
            lo, hi = region_of[b]
            ax.axvspan(lo, hi, color=colors[b], alpha=0.15, label=f"exp. region {lo}–{hi} mV")
            top = np.argsort(mean_abs)[::-1][:3]
            ax.scatter(voltages[top], mean_abs[top], color=colors[b], s=30, zorder=5)
            ax.set_ylabel(f"|SHAP| ({labels[b]})")
            ax.legend(loc="upper right", fontsize=8)
            ax.grid(alpha=0.3)
        axes[2].set_xlabel("Potential (mV)")
        axes[0].set_title("SHAP importance vs potential with experimental biomarker regions")
        fig.tight_layout()
        fig.savefig(os.path.join(FIGURE_DIR, "P20D_shap_vs_potential.png"))
        plt.close(fig)
        print("  ✓ Saved: P20D_shap_vs_potential.png")


# ============================================================================
# MAIN
# ============================================================================
def main():
    os.makedirs(RESULT_DIR, exist_ok=True)
    os.makedirs(FIGURE_DIR, exist_ok=True)

    df, X, Y, voltages, dpv_cols = load_data()
    audit_data(df, X, Y, voltages)
    plot_dpv_overview(df, X, voltages, dpv_cols)

    corr = biomarker_correlation(df, X, voltages, dpv_cols)
    corr.to_csv(os.path.join(RESULT_DIR, "P3_biomarker_correlation.csv"), index=False)

    single = run_single_regression(X, Y, df)
    single.to_csv(os.path.join(RESULT_DIR, "P4_single_regression.csv"), index=False)

    multi = run_multi_output(X, Y)
    multi.to_csv(os.path.join(RESULT_DIR, "P5_multi_output.csv"), index=False)

    rep = run_representation_comparison(df, X, Y, voltages)
    rep.to_csv(os.path.join(RESULT_DIR, "P6_representations.csv"), index=False)

    global _GLOBAL_YLOG
    _GLOBAL_YLOG = np.log10(Y.clip(min=1e-9))
    abla = run_ablation(X, Y, voltages)
    abla.to_csv(os.path.join(RESULT_DIR, "P7_ablation.csv"), index=False)

    inter = run_interference(X, Y, voltages)
    inter.to_csv(os.path.join(RESULT_DIR, "P8_interference.csv"), index=False)

    strict = strict_split_and_bootstrap(X, Y)
    strict.to_csv(os.path.join(RESULT_DIR, "P9_strict_test.csv"), index=False)

    shap_res = run_shap(X, Y, voltages)

    plot_pred_vs_measured(X, Y)

    plot_publication_figures(multi, rep, abla, shap_res, voltages)

    print("\n" + "=" * 70)
    print("REGRESSION PIPELINE COMPLETE")
    print("=" * 70)
    print("Saved: P3/P4/P5/P6/P7/P8/P9 CSV reports + P2/P15/P20 figures")


if __name__ == "__main__":
    main()
