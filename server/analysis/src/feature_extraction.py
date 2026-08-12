"""
Electrochemical Feature Extraction from Raw DPV Fingerprints
=============================================================
Reproducible generation of engineered electrochemical features from the
raw 200-point differential pulse voltammetry (DPV) fingerprints.

Features produced (per sample):
  1. peak_anodic_current      - minimum current (most negative dip) in the negative region
  2. peak_anodic_potential    - potential at which that minimum occurs (mV)
  3. peak_cathodic_current    - maximum current (most positive rise) in the positive region
  4. peak_cathodic_potential  - potential at which that maximum occurs (mV)
  5. area_under_curve         - trapezoidal integral of the DPV curve over the full range
  6. peak_separation          - absolute difference between cathodic and anodic potentials

Method notes (for reproducibility):
  * No baseline correction is applied (the raw measured currents are used).
  * Peak detection is a simple global extrema search within the experimental
    potential window (no smoothing).
  * Missing-peak handling: if a region contains no data (should not happen),
    NaN is recorded; the reason is documented in the audit output.
"""

import os

import numpy as np
import pandas as pd

DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "Raw_DPV_Dataset_for_Cancer_biomarker.csv")
OUT_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "data_with_features_engineered.csv")

# Experimentally validated potential regions (mV) established by the biosensor team.
# These are used to locate the biomarker-associated peaks, NOT invented here.
REGIONS = {
    "anodic": (-750.0, -400.0),    # negative region (PSA-associated dip region)
    "cathodic": (300.0, 1100.0),   # positive region (AFP / CA125 associated rise region)
}


def load_raw(path=DATA_PATH):
    """Load the raw DPV dataset."""
    df = pd.read_csv(path)
    return df


def extract_features(df):
    """Compute engineered electrochemical features from raw DPV curves."""
    dpv_cols = [c for c in df.columns if c.startswith("curr_")]
    voltages = np.array([float(c.replace("curr_", "").replace("mV", "")) for c in dpv_cols])
    X = df[dpv_cols].values

    records = []
    for i in range(len(df)):
        currents = X[i]
        rec = {"sample_id": df["sample_id"].iloc[i]}

        # Anodic (negative region) peak = most negative current
        mask_a = (voltages >= REGIONS["anodic"][0]) & (voltages <= REGIONS["anodic"][1])
        if mask_a.sum() > 0:
            ia = np.argmin(currents[mask_a])
            rec["peak_anodic_current"] = currents[mask_a][ia]
            rec["peak_anodic_potential"] = voltages[mask_a][ia]
        else:
            rec["peak_anodic_current"] = np.nan
            rec["peak_anodic_potential"] = np.nan

        # Cathodic (positive region) peak = most positive current
        mask_c = (voltages >= REGIONS["cathodic"][0]) & (voltages <= REGIONS["cathodic"][1])
        if mask_c.sum() > 0:
            ic = np.argmax(currents[mask_c])
            rec["peak_cathodic_current"] = currents[mask_c][ic]
            rec["peak_cathodic_potential"] = voltages[mask_c][ic]
        else:
            rec["peak_cathodic_current"] = np.nan
            rec["peak_cathodic_potential"] = np.nan

        # Area under curve (trapezoidal) over full potential range
        rec["area_under_curve"] = np.trapezoid(currents, voltages)

        # Peak separation
        if not (np.isnan(rec["peak_anodic_potential"]) or np.isnan(rec["peak_cathodic_potential"])):
            rec["peak_separation"] = abs(rec["peak_cathodic_potential"] - rec["peak_anodic_potential"])
        else:
            rec["peak_separation"] = np.nan

        records.append(rec)

    feats = pd.DataFrame(records)

    # Merge with biomarker concentrations
    out = df[["sample_id", "PSA_pg_per_ml", "AFP_pg_per_ml", "CA125_U_per_ml"]].merge(
        feats, on="sample_id", how="left"
    )
    return out


def audit_features(feats):
    """Report missing values and reasons per feature."""
    print("\n" + "=" * 70)
    print("ENGINEERED FEATURE AUDIT")
    print("=" * 70)
    feature_cols = [
        "peak_anodic_current", "peak_anodic_potential",
        "peak_cathodic_current", "peak_cathodic_potential",
        "area_under_curve", "peak_separation",
    ]
    for col in feature_cols:
        n_missing = int(feats[col].isna().sum())
        status = "OK" if n_missing == 0 else f"MISSING {n_missing}"
        print(f"  {col:<26} {status}")
    if feats[feature_cols].isna().any().any():
        print("\n  Reason: region search produced no data for these samples.")
        print("  (Should not happen with a full 200-point sweep; investigate before filling.)")


def main():
    df = load_raw()
    feats = extract_features(df)
    audit_features(feats)
    feats.to_csv(OUT_PATH, index=False)
    print(f"\n✓ Saved engineered features to {OUT_PATH}")
    print(f"  Shape: {feats.shape}")


if __name__ == "__main__":
    main()
