"""
COMPLETE PROSTATE CANCER RISK CLASSIFICATION PIPELINE
with Advanced Visualizations
=====================================================
"""

import os
import pickle
import warnings
from math import pi

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import gaussian_kde
from sklearn.calibration import calibration_curve
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.preprocessing import RobustScaler
from sklearn.svm import SVC
from sklearn.utils.class_weight import compute_class_weight
from xgboost import XGBClassifier

from gnn_model import train_gnn

warnings.filterwarnings('ignore')

# Set global style for beautiful plots
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans']
plt.rcParams['axes.facecolor'] = '#f8f9fa'
plt.rcParams['figure.facecolor'] = 'white'
plt.rcParams['grid.alpha'] = 0.3
plt.rcParams['grid.linestyle'] = '--'

# Custom color palettes for each model
MODEL_COLORS = {
    'Logistic Regression': {
        'primary': '#FF6B6B',
        'secondary': '#4ECDC4',
        'gradient': ['#FF6B6B', '#FF8E8E', '#FFB2B2'],
        'heatmap': 'Reds'
    },
    'Random Forest': {
        'primary': '#95E77E',
        'secondary': '#3B82F6',
        'gradient': ['#95E77E', '#7BC5AE', '#5FA8D3'],
        'heatmap': 'Greens'
    },
    'SVM': {
        'primary': '#F59E0B',
        'secondary': '#8B5CF6',
        'gradient': ['#F59E0B', '#FBBF24', '#FCD34D'],
        'heatmap': 'Oranges'
    },
    'XGBoost': {
        'primary': '#EF4444',
        'secondary': '#06B6D4',
        'gradient': ['#EF4444', '#F97316', '#FBBF24'],
        'heatmap': 'Purples'
    }
}

# ============================================================================
# SECTION 1: DATA LOADING AND PREPROCESSING
# ============================================================================

def load_data(file_path, sheet_name="Target_Concentrations"):
    """Load the Excel data file."""
    print("\n" + "="*70)
    print("STEP 1: LOADING DATA")
    print("="*70)

    df = pd.read_excel(file_path, sheet_name=sheet_name)
    print(f"✓ Data loaded successfully")
    print(f"  Shape: {df.shape}")
    print(f"  Columns: {df.columns.tolist()}")
    return df


def load_dpv_features(file_path, target_df):
    """Load DPV voltammetry data from Sample_XXXX sheets and merge with target data."""
    print("\n" + "="*70)
    print("STEP: LOADING DPV VOLTAMMETRY FEATURES")
    print("="*70)

    all_sheets = pd.read_excel(file_path, sheet_name=None)
    sample_sheets = {k: v for k, v in all_sheets.items() if k.startswith('Sample_')}

    print(f"  Found {len(sample_sheets)} sample sheets with DPV data")

    records = []
    voltages = None
    for sid in sorted(sample_sheets.keys()):
        sdf = sample_sheets[sid]
        if voltages is None:
            voltages = sdf['voltage_mV'].values
        currents = sdf['current_uA'].values
        record = {'sample_id': sid}
        for i, val in enumerate(currents):
            record[f'V{i}'] = val
        records.append(record)

    dpv_df = pd.DataFrame(records)
    print(f"  DPV feature matrix: {dpv_df.shape} ({len(voltages)} current measurements per sample)")

    merged = target_df.merge(dpv_df, on='sample_id', how='left')
    n_missing = merged[merged['V0'].isna()].shape[0]
    if n_missing > 0:
        print(f"  {n_missing} samples missing DPV data")
    print(f"  Merged dataset: {merged.shape}")

    voltage_labels = {f'V{i}': f'{voltages[i]:.0f} mV' for i in range(len(voltages))}
    return merged, voltage_labels


def create_target_variable(df, psa_column="PSA_pg_per_ml", cutoff=4000):
    """Create binary target variable based on PSA clinical cutoff."""
    df["high_risk"] = (df[psa_column] > cutoff).astype(int)

    print("\n" + "="*70)
    print("STEP 2: TARGET VARIABLE CREATION")
    print("="*70)
    print(f"  Definition: {psa_column} > {cutoff} pg/mL indicates high risk")
    print(f"  Distribution:")
    print(f"    Low Risk (0): {(df['high_risk']==0).sum()} ({((df['high_risk']==0).sum()/len(df))*100:.1f}%)")
    print(f"    High Risk (1): {(df['high_risk']==1).sum()} ({((df['high_risk']==1).sum()/len(df))*100:.1f}%)")

    return df

def preprocess_features(df, feature_columns):
    """Preprocess features: handle missing values, log transform (when safe), and scale."""
    print("\n" + "="*70)
    print("STEP 3: FEATURE PREPROCESSING")
    print("="*70)

    # Remove missing values
    df_clean = df[feature_columns + ["high_risk"]].dropna()
    print(f"  Samples after removing missing values: {len(df_clean)}")

    # Separate features and target
    X = df_clean[feature_columns].copy()
    y = df_clean["high_risk"].copy()

    # Log transformation (skip if features contain negative values, e.g. DPV currents)
    X_log = X.copy()
    has_negative = (X < 0).any().any()
    if has_negative:
        print(f"\n  Skipping log transform — features contain negative values")
    else:
        print(f"\n  Log transformation applied:")
        for col in feature_columns:
            X_log[col] = np.log1p(X_log[col])
            print(f"    {col}: [{X[col].min():.2f}, {X[col].max():.2f}] → [{X_log[col].min():.2f}, {X_log[col].max():.2f}]")

    # Scale features
    scaler = RobustScaler()
    X_scaled = scaler.fit_transform(X_log)
    print(f"\n  Scaling: RobustScaler applied")
    print(f"  Final feature matrix shape: {X_scaled.shape}")

    return X_scaled, y, scaler

def split_data(X, y, test_size=0.30, random_state=42):
    """Split data into training and validation sets with stratification."""
    print("\n" + "="*70)
    print("STEP 4: DATA SPLITTING")
    print("="*70)

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    print(f"  Training set: {len(X_train)} samples ({len(X_train)/len(X)*100:.1f}%)")
    print(f"  Validation set: {len(X_val)} samples ({len(X_val)/len(X)*100:.1f}%)")
    print(f"\n  Training distribution:")
    print(f"    Low Risk: {(y_train==0).sum()} ({((y_train==0).sum()/len(y_train))*100:.1f}%)")
    print(f"    High Risk: {(y_train==1).sum()} ({((y_train==1).sum()/len(y_train))*100:.1f}%)")

    return X_train, X_val, y_train, y_val

# ============================================================================
# SECTION 2: MODEL DEFINITION AND TRAINING
# ============================================================================

def initialize_models(X_train, y_train):
    """Initialize all machine learning models with appropriate parameters."""
    print("\n" + "="*70)
    print("STEP 5: MODEL INITIALIZATION")
    print("="*70)

    # Calculate class weights for imbalance
    classes = np.unique(y_train)
    weights = compute_class_weight("balanced", classes=classes, y=y_train)
    class_weight_dict = dict(zip(classes, weights))

    models = {
        "Logistic Regression": LogisticRegression(
            random_state=42, max_iter=1000, class_weight="balanced", C=1.0
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=100, random_state=42, class_weight="balanced",
            max_depth=10, min_samples_split=5
        ),
        "SVM": SVC(
            kernel="rbf", probability=True, random_state=42,
            class_weight="balanced", C=1.0, gamma="scale"
        ),
        "XGBoost": XGBClassifier(
            n_estimators=100, random_state=42,
            scale_pos_weight=(len(y_train[y_train==0]) / len(y_train[y_train==1])),
            eval_metric="logloss", use_label_encoder=False,
            max_depth=6, learning_rate=0.1
        )
    }

    for name in models:
        print(f"  ✓ {name}")

    return models

def train_models(models, X_train, y_train, X_val, y_val, save_dir="../models"):
    """Train all models and save them to disk."""
    print("\n" + "="*70)
    print("STEP 6: MODEL TRAINING")
    print("="*70)

    os.makedirs(save_dir, exist_ok=True)

    trained_models = {}
    results = {}

    for name, model in models.items():
        print(f"\n  Training {name}...")
        model.fit(X_train, y_train)

        # Save model
        filename = f"{name.lower().replace(' ', '_')}_model.pkl"
        file_path = os.path.join(save_dir, filename)
        with open(file_path, "wb") as file:
            pickle.dump(model, file)

        trained_models[name] = model

        # Predictions
        y_pred = model.predict(X_val)
        y_pred_proba = model.predict_proba(X_val)[:, 1] if hasattr(model, "predict_proba") else None

        # Calculate metrics
        results[name] = {
            "accuracy": accuracy_score(y_val, y_pred),
            "precision": precision_score(y_val, y_pred, zero_division=0),
            "recall": recall_score(y_val, y_pred, zero_division=0),
            "f1_score": f1_score(y_val, y_pred, zero_division=0),
            "roc_auc": roc_auc_score(y_val, y_pred_proba) if y_pred_proba is not None else None,
            "confusion_matrix": confusion_matrix(y_val, y_pred),
            "y_pred": y_pred,
            "y_pred_proba": y_pred_proba
        }

        print(f"    ✓ Accuracy: {results[name]['accuracy']:.4f}")
        print(f"    ✓ F1-Score: {results[name]['f1_score']:.4f}")

    return trained_models, results


def create_best_model_summary_table(results, save_dir="../figure"):
    """Create a best-model summary table and print the winner with percentages."""
    os.makedirs(save_dir, exist_ok=True)

    summary_df = pd.DataFrame([
        {
            'Model': name,
            'Accuracy (%)': results[name]['accuracy'] * 100,
            'Precision (%)': results[name]['precision'] * 100,
            'Recall (%)': results[name]['recall'] * 100,
            'F1-Score (%)': results[name]['f1_score'] * 100,
            'ROC-AUC (%)': results[name]['roc_auc'] * 100 if results[name]['roc_auc'] is not None else 0
        }
        for name in results
    ])

    best_model = summary_df.loc[summary_df['F1-Score (%)'].idxmax()]

    print("\n" + "="*70)
    print("BEST MODEL SUMMARY")
    print("="*70)
    print(f"Best model: {best_model['Model']}")
    print(f"  • Accuracy: {best_model['Accuracy (%)']:.2f}%")
    print(f"  • Precision: {best_model['Precision (%)']:.2f}%")
    print(f"  • Recall: {best_model['Recall (%)']:.2f}%")
    print(f"  • F1-Score: {best_model['F1-Score (%)']:.2f}%")
    print(f"  • ROC-AUC: {best_model['ROC-AUC (%)']:.2f}%")
    print("\nModel performance table (percentages):")
    print(summary_df.to_string(index=False, float_format='%.2f'))

    summary_df.to_csv(os.path.join(save_dir, 'best_model_summary.csv'), index=False)
    print(f"  ✓ Saved: best_model_summary.csv")

    return best_model['Model'], summary_df


def create_best_model_modal(results, save_dir="../figure"):
    """Create a focused modal-style figure for the highest scoring model."""
    os.makedirs(save_dir, exist_ok=True)

    metrics = ['accuracy', 'precision', 'recall', 'f1_score', 'roc_auc']
    display_labels = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC']

    best_model_name = max(results, key=lambda x: results[x]['f1_score'])
    best_values = [results[best_model_name][metric] if results[best_model_name][metric] is not None else 0 for metric in metrics]

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.axis('off')

    title = f"Highest Scoring Model: {best_model_name}"
    subtitle = "Top performance across validation metrics"

    ax.text(0.02, 0.9, title, fontsize=22, fontweight='bold', color='#1F2937', va='top')
    ax.text(0.02, 0.82, subtitle, fontsize=12, color='#4B5563', va='top')

    for i, (label, value) in enumerate(zip(display_labels, best_values)):
        y = 0.65 - i * 0.12
        ax.text(0.05, y, f"{label}", fontsize=14, fontweight='bold', color='#111827', va='center')
        ax.text(0.85, y, f"{value * 100:.2f}%", fontsize=14, fontweight='bold', color='#0B6E99', va='center', ha='right')
        ax.hlines(y - 0.02, 0.05, 0.95, color='#D1D5DB', linewidth=1, alpha=0.5)

    ax.text(0.02, 0.08, "This figure highlights only the best performing model based on F1 score, providing a concise single-model summary.",
            fontsize=10, color='#6B7280', va='bottom')

    plt.tight_layout()
    path = os.path.join(save_dir, 'best_model_modal.png')
    plt.savefig(path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved: best_model_modal.png")

    return path





# ============================================================================
# SECTION 3: ADVANCED VISUALIZATIONS (DIFFERENT STYLES FOR EACH MODEL)
# ============================================================================

def create_advanced_confusion_matrix(y_true, y_pred, model_name, save_dir="../figure"):
    """Create unique confusion matrix visualizations for each model with different styles."""
    os.makedirs(save_dir, exist_ok=True)
    cm = confusion_matrix(y_true, y_pred)
    cm_percentage = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100

    fig, ax = plt.subplots(figsize=(10, 8))

    # Different style based on model
    if model_name == 'Logistic Regression':
        # Style 1: Circular bubbles
        colors = MODEL_COLORS[model_name]['gradient']
        max_val = cm.max()
        for i in range(2):
            for j in range(2):
                size = (cm[i, j] / max_val) * 3000
                circle = plt.Circle((j, i), radius=np.sqrt(size)/20,
                                   color=colors[min(int(cm[i, j]/max_val*len(colors)), len(colors)-1)],
                                   alpha=0.7, ec='black', linewidth=2)
                ax.add_patch(circle)
                ax.text(j, i, f'{cm[i, j]}\n({cm_percentage[i, j]:.1f}%)',
                       ha='center', va='center', fontsize=14, fontweight='bold')

        ax.set_xlim(-0.5, 1.5)
        ax.set_ylim(-0.5, 1.5)
        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(['Low Risk', 'High Risk'], fontsize=12)
        ax.set_yticklabels(['Low Risk', 'High Risk'], fontsize=12)
        ax.set_xlabel('Predicted', fontsize=13, fontweight='bold')
        ax.set_ylabel('Actual', fontsize=13, fontweight='bold')
        ax.set_title(f'{model_name}\nBubble Confusion Matrix', fontsize=15, fontweight='bold', pad=20)
        ax.grid(False)

    elif model_name == 'Random Forest':
        # Style 2: Gradient heatmap with custom annotations
        sns.heatmap(cm, annot=False, fmt='d', cmap=MODEL_COLORS[model_name]['heatmap'],
                   xticklabels=['Low Risk', 'High Risk'],
                   yticklabels=['Low Risk', 'High Risk'],
                   ax=ax, cbar_kws={'label': 'Count', 'shrink': 0.8},
                   square=True, linewidths=2, linecolor='white')

        for i in range(2):
            for j in range(2):
                color = 'white' if cm[i, j] > cm.max()/2 else 'black'
                ax.text(j+0.5, i+0.5, f'{cm[i, j]}\n({cm_percentage[i, j]:.1f}%)',
                       ha='center', va='center', fontsize=13, fontweight='bold', color=color)

        ax.set_title(f'{model_name}\nGradient Confusion Matrix', fontsize=15, fontweight='bold', pad=20)

    elif model_name == 'SVM':
        # Style 3: Donut chart style
        from matplotlib.patches import Wedge

        total = cm.sum()
        angles = [0]
        current = 0
        for i in range(2):
            for j in range(2):
                current += cm[i, j]
                angles.append(360 * current / total)

        colors_donut = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4']
        labels = ['True Negatives', 'False Positives', 'False Negatives', 'True Positives']

        for i in range(4):
            wedge = Wedge((0.5, 0.5), 0.4, angles[i], angles[i+1],
                         facecolor=colors_donut[i], edgecolor='white', linewidth=2)
            ax.add_patch(wedge)
            mid_angle = np.radians((angles[i] + angles[i+1]) / 2)
            label_radius = 0.33
            label_x = 0.5 + label_radius * np.cos(mid_angle)
            label_y = 0.5 + label_radius * np.sin(mid_angle)
            ax.text(label_x, label_y,
                   f'{labels[i]}\n{cm.flatten()[i]}\n({cm_percentage.flatten()[i]:.1f}%)',
                   ha='center', va='center', fontsize=9, fontweight='bold', color='white')

        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_aspect('equal')
        ax.set_title(f'{model_name}\nDonut Confusion Matrix', fontsize=15, fontweight='bold', pad=20)
        ax.axis('off')

    else:  # XGBoost
        # Style 4: Purple gradient heatmap for clarity
        sns.heatmap(
            cm,
            annot=np.array([[f'{cm[i, j]}\n({cm_percentage[i, j]:.1f}%)' for j in range(2)] for i in range(2)]),
            fmt='',
            cmap='Purples',
            xticklabels=['Low Risk', 'High Risk'],
            yticklabels=['Low Risk', 'High Risk'],
            ax=ax,
            cbar_kws={'label': 'Count', 'shrink': 0.8},
            square=True,
            linewidths=2,
            linecolor='white',
            annot_kws={"fontsize": 12, "fontweight": "bold"}
        )

        ax.set_title(f'{model_name}\nPurple Heatmap Confusion Matrix', fontsize=15, fontweight='bold', pad=20)
        ax.set_xlabel('Predicted', fontsize=13, fontweight='bold')
        ax.set_ylabel('Actual', fontsize=13, fontweight='bold')
        ax.set_xticklabels(ax.get_xticklabels(), fontsize=11)
        ax.set_yticklabels(ax.get_yticklabels(), fontsize=11)
        ax.grid(False)

    plt.tight_layout()
    plt.savefig(f"{save_dir}/advanced_cm_{model_name.lower().replace(' ', '_')}.png",
                dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved: advanced_cm_{model_name.lower().replace(' ', '_')}.png")

def create_radar_chart(results, save_dir="../figure"):
    """Create beautiful radar chart comparing all models."""
    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))

    metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC']
    metric_keys = ['accuracy', 'precision', 'recall', 'f1_score', 'roc_auc']

    angles = [n / float(len(metrics)) * 2 * pi for n in range(len(metrics))]
    angles += angles[:1]

    colors_radar = ['#FF6B6B', '#95E77E', '#F59E0B', '#EF4444']

    for idx, (name, color) in enumerate(zip(results.keys(), colors_radar)):
        values = [results[name][key] if results[name][key] is not None else 0 for key in metric_keys]
        values += values[:1]

        ax.plot(angles, values, 'o-', linewidth=2.5, label=name, color=color, markersize=8)
        ax.fill(angles, values, alpha=0.15, color=color)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(metrics, fontsize=12, fontweight='bold')
    ax.set_ylim(0, 1)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels(['0.2', '0.4', '0.6', '0.8', '1.0'], fontsize=9)
    ax.set_title('Model Performance Radar Chart\nComprehensive Comparison',
                fontsize=16, fontweight='bold', pad=30)
    ax.legend(loc='upper right', bbox_to_anchor=(1.2, 1.1), fontsize=10, framealpha=0.95)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f"{save_dir}/radar_chart_comparison.png", dpi=300, bbox_inches='tight')
    plt.close()
    print("  ✓ Saved: radar_chart_comparison.png")

def create_parallel_coordinates(results, y_val, save_dir="../figure"):
    """Create parallel coordinates plot for model predictions."""
    # Prepare data
    plot_data = pd.DataFrame()
    plot_data['True Label'] = y_val.values

    for name in results.keys():
        if results[name]['y_pred_proba'] is not None:
            plot_data[f'{name}'] = results[name]['y_pred_proba']

    plot_data['True Label'] = plot_data['True Label'].map({0: 'Low Risk', 1: 'High Risk'})

    fig, ax = plt.subplots(figsize=(14, 7))

    # Create parallel coordinates manually
    model_names = [col for col in plot_data.columns if col != 'True Label']
    n_models = len(model_names)

    colors_parallel = {'Low Risk': '#2ECC71', 'High Risk': '#E74C3C'}

    for label, color in colors_parallel.items():
        subset = plot_data[plot_data['True Label'] == label]
        for _, row in subset.iterrows():
            values = [row[model] for model in model_names]
            ax.plot(range(n_models), values, color=color, alpha=0.3, linewidth=0.8)

    # Add mean lines
    for label, color in colors_parallel.items():
        subset = plot_data[plot_data['True Label'] == label]
        means = [subset[model].mean() for model in model_names]
        ax.plot(range(n_models), means, color=color, linewidth=3,
               label=f'{label} (Mean)', marker='o', markersize=8)

    ax.set_xticks(range(n_models))
    ax.set_xticklabels(model_names, rotation=45, ha='right', fontsize=10)
    ax.set_ylabel('Prediction Probability', fontsize=12, fontweight='bold')
    ax.set_title('Model Predictions Parallel Coordinates\nHow Models Behave Across Samples',
                fontsize=14, fontweight='bold', pad=20)
    ax.legend(loc='upper left', fontsize=10, framealpha=0.95)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_ylim(-0.05, 1.05)

    plt.tight_layout()
    plt.savefig(f"{save_dir}/parallel_coordinates.png", dpi=300, bbox_inches='tight')
    plt.close()
    print("  ✓ Saved: parallel_coordinates.png")

def create_ridge_plot(results, y_val, save_dir="../figure"):
    """Create ridge plot (joyplot) for probability distributions."""
    fig, ax = plt.subplots(figsize=(12, 8))

    models_list = list(results.keys())
    colors_ridge = ['#FF6B6B', '#4ECDC4', '#F59E0B', '#EF4444']

    for idx, (name, color) in enumerate(zip(models_list, colors_ridge)):
        if results[name]['y_pred_proba'] is not None:
            # Separate by true label
            proba_low = results[name]['y_pred_proba'][y_val == 0]
            proba_high = results[name]['y_pred_proba'][y_val == 1]

            # KDE for low risk
            if len(proba_low) > 1:
                kde_low = gaussian_kde(proba_low)
                x_range = np.linspace(0, 1, 100)
                y_low = kde_low(x_range) / kde_low(x_range).max() + idx * 1.5
                ax.fill_between(x_range, idx * 1.5, y_low, alpha=0.4, color=color)
                ax.plot(x_range, y_low, color=color, linewidth=2)

            # KDE for high risk
            if len(proba_high) > 1:
                kde_high = gaussian_kde(proba_high)
                y_high = kde_high(x_range) / kde_high(x_range).max() + idx * 1.5 + 0.4
                ax.fill_between(x_range, idx * 1.5 + 0.4, y_high, alpha=0.4, color=color, linestyle='--')
                ax.plot(x_range, y_high, color=color, linewidth=2, linestyle='--')

    ax.set_yticks([i * 1.5 + 0.7 for i in range(len(models_list))])
    ax.set_yticklabels(models_list, fontsize=11, fontweight='bold')
    ax.set_xlabel('Prediction Probability', fontsize=12, fontweight='bold')
    ax.set_title('Model Prediction Distributions (Ridge Plot)\nSeparated by True Label',
                fontsize=14, fontweight='bold', pad=20)
    ax.grid(True, alpha=0.3, axis='x', linestyle='--')
    ax.set_xlim(0, 1)

    # Add legend
    low_patch = mpatches.Patch(color='gray', alpha=0.4, label='Low Risk Distribution')
    high_patch = mpatches.Patch(color='gray', alpha=0.4, label='High Risk Distribution', linestyle='--')
    ax.legend(handles=[low_patch, high_patch], loc='upper right', fontsize=10)

    plt.tight_layout()
    plt.savefig(f"{save_dir}/ridge_plot_distributions.png", dpi=300, bbox_inches='tight')
    plt.close()
    print("  ✓ Saved: ridge_plot_distributions.png")

def create_violin_plots(results, y_val, save_dir="../figure"):
    """Create beautiful violin plots for model predictions."""
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    axes_flat = axes.flatten()

    colors_violin = ['#2ECC71', '#E74C3C']

    for idx, (name, ax) in enumerate(zip(results.keys(), axes_flat)):
        if results[name]['y_pred_proba'] is not None:
            # Prepare data
            data_to_plot = [
                results[name]['y_pred_proba'][y_val == 0],
                results[name]['y_pred_proba'][y_val == 1]
            ]

            # Create violin plot
            parts = ax.violinplot(data_to_plot, positions=[1, 2],
                                 showmeans=True, showmedians=True, showextrema=True)

            # Style the violins
            for i, pc in enumerate(parts['bodies']):
                pc.set_facecolor(colors_violin[i])
                pc.set_alpha(0.7)
                pc.set_edgecolor('black')
                pc.set_linewidth(1.5)

            # Style mean and median
            parts['cmeans'].set_color('darkblue')
            parts['cmeans'].set_linewidth(2)
            parts['cmedians'].set_color('gold')
            parts['cmedians'].set_linewidth(2)

            # Add individual points (jitter)
            for i, data in enumerate(data_to_plot):
                x_jitter = np.random.normal(i+1, 0.04, size=len(data))
                ax.scatter(x_jitter, data, alpha=0.3, s=20, color=colors_violin[i])

            ax.set_xticks([1, 2])
            ax.set_xticklabels(['Low Risk\n(Benign)', 'High Risk\n(Malignant)'], fontsize=11)
            ax.set_ylabel('Prediction Probability', fontsize=11, fontweight='bold')
            ax.set_title(f'{name}\nPrediction Distribution by True Label', fontsize=12, fontweight='bold')
            ax.set_ylim(-0.05, 1.05)
            ax.grid(True, alpha=0.3, axis='y', linestyle='--')

            # Add statistical annotation
            median_low = np.median(data_to_plot[0])
            median_high = np.median(data_to_plot[1])
            ax.text(1.5, 0.95, f'Median Diff: {abs(median_high - median_low):.3f}',
                   ha='center', fontsize=9, bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    plt.suptitle('Model Prediction Distributions - Violin Plots',
                fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(f"{save_dir}/violin_plots.png", dpi=300, bbox_inches='tight')
    plt.close()
    print("  ✓ Saved: violin_plots.png")

def create_correlation_heatmap(df, save_dir="../figure", voltage_labels=None):
    """Heatmap showing correlations between biomarkers/DPV features and the target."""
    if voltage_labels is not None:
        dpv_cols = [c for c in df.columns if c.startswith('V') and c[1:].isdigit()]
        corr_with_target = df[dpv_cols].corrwith(df['high_risk']).abs().sort_values(ascending=False)
        top_n = min(20, len(corr_with_target))
        top_features = corr_with_target.head(top_n).index.tolist()
        corr_cols = top_features + ['PSA_pg_per_ml', 'AFP_pg_per_ml', 'CA125_U_per_ml', 'high_risk']
        corr_labels = [voltage_labels.get(c, c) for c in top_features] + ['PSA', 'AFP', 'CA125', 'High Risk']
    else:
        corr_cols = ['PSA_pg_per_ml', 'AFP_pg_per_ml', 'CA125_U_per_ml', 'high_risk']
        corr_labels = ['PSA', 'AFP', 'CA125', 'High Risk']

    corr_matrix = df[corr_cols].corr()

    fig, ax = plt.subplots(figsize=(16, 14))
    sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='RdBu_r',
                xticklabels=corr_labels, yticklabels=corr_labels,
                vmin=-1, vmax=1, center=0, square=True,
                linewidths=0.5, linecolor='white', ax=ax,
                annot_kws={'size': 8},
                cbar_kws={'label': 'Pearson Correlation', 'shrink': 0.8})

    ax.set_title('DPV Feature & Biomarker Correlation Heatmap', fontsize=18, fontweight='bold', pad=15)
    plt.xticks(rotation=45, ha='right', fontsize=10)
    plt.yticks(fontsize=10)
    plt.tight_layout()
    plt.savefig(f"{save_dir}/correlation_heatmap.png", dpi=300, bbox_inches='tight')
    plt.close()
    print("  ✓ Saved: correlation_heatmap.png")


def create_model_performance_bar_chart(results, save_dir="../figure"):
    """Create a grouped bar chart for the full model performance summary."""
    os.makedirs(save_dir, exist_ok=True)
    metrics = ['accuracy', 'precision', 'recall', 'f1_score', 'roc_auc']
    labels = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC']

    chart_data = {
        name: [results[name][metric] if results[name][metric] is not None else 0 for metric in metrics]
        for name in results
    }
    df = pd.DataFrame(chart_data, index=labels)

    fig, ax = plt.subplots(figsize=(14, 8))
    df.plot(kind='bar', ax=ax, width=0.8)

    ax.set_title('Model Performance Summary', fontsize=16, fontweight='bold', pad=18)
    ax.set_ylabel('Score', fontsize=12, fontweight='bold')
    ax.set_ylim(0, 1)
    ax.set_xticklabels(labels, rotation=0, fontsize=11)
    ax.legend(title='Model', fontsize=10, title_fontsize=11, loc='upper right')
    ax.grid(axis='y', alpha=0.3, linestyle='--')

    for patch in ax.patches:
        height = patch.get_height()
        if height is not None:
            ax.annotate(f'{height:.2f}',
                        (patch.get_x() + patch.get_width() / 2, height),
                        ha='center', va='bottom', fontsize=9, xytext=(0, 4), textcoords='offset points')

    plt.tight_layout()
    plt.savefig(f"{save_dir}/model_performance_summary.png", dpi=300, bbox_inches='tight')
    plt.close()
    print("  ✓ Saved: model_performance_summary.png")


def create_roc_curves_enhanced(results, y_val, save_dir="../figure"):
    """Create enhanced ROC curves with confidence intervals and annotations."""
    fig, ax = plt.subplots(figsize=(12, 8))

    colors_roc = ['#FF6B6B', '#4ECDC4', '#F59E0B', '#EF4444']

    for (name, result), color in zip(results.items(), colors_roc):
        if result["y_pred_proba"] is not None:
            fpr, tpr, _ = roc_curve(y_val, result["y_pred_proba"])
            auc = result["roc_auc"]

            # Plot ROC curve
            ax.plot(fpr, tpr, linewidth=2.5, label=f'{name} (AUC = {auc:.3f})',
                   color=color)

            # Add shaded area under curve
            ax.fill_between(fpr, 0, tpr, alpha=0.1, color=color)

            # Mark optimal point (closest to top-left)
            distances = np.sqrt(fpr**2 + (1-tpr)**2)
            optimal_idx = np.argmin(distances)
            ax.plot(fpr[optimal_idx], tpr[optimal_idx], 'o', color=color,
                   markersize=10, markeredgecolor='black', markeredgewidth=1.5)

    # Plot diagonal
    ax.plot([0, 1], [0, 1], 'k--', label='Random Classifier', linewidth=2, alpha=0.7)

    ax.set_xlabel('False Positive Rate (1 - Specificity)', fontsize=12, fontweight='bold')
    ax.set_ylabel('True Positive Rate (Sensitivity)', fontsize=12, fontweight='bold')
    ax.set_title('ROC Curves - Enhanced Comparison', fontsize=14, fontweight='bold', pad=20)
    ax.legend(loc='lower right', fontsize=11, framealpha=0.95)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_xlim([-0.02, 1.02])
    ax.set_ylim([-0.02, 1.02])

    # Add text box with interpretation
    best_model = max(results, key=lambda x: results[x]['roc_auc'] if results[x]['roc_auc'] is not None else 0)
    best_auc = results[best_model]['roc_auc']
    interpretation = f"Best Model: {best_model}\nAUC = {best_auc:.3f}\nExcellent discrimination" if best_auc > 0.8 else "Moderate discrimination"

    ax.text(0.02, 0.02, interpretation, transform=ax.transAxes, fontsize=10,
           verticalalignment='bottom', bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

    plt.tight_layout()
    plt.savefig(f"{save_dir}/roc_curves_enhanced.png", dpi=300, bbox_inches='tight')
    plt.close()
    print("  ✓ Saved: roc_curves_enhanced.png")

# ============================================================================
# SECTION 4: MAIN PIPELINE
# ============================================================================

def run_complete_pipeline():
    """Run the complete machine learning pipeline with all visualizations."""

    print("\n" + "="*70)
    print("PROSTATE CANCER RISK CLASSIFICATION PIPELINE")
    print("="*70)
    print("\nThis pipeline uses DPV voltammetry features (200-point")
    print("current measurements) to predict prostate cancer risk")
    print("based on PSA clinical cutoff (>4 ng/mL).")

    # 1. Load target data
    df_target = load_data("../data/Raw_data_dpv.xlsx", sheet_name="Target_Concentrations")

    # 2. Load DPV features and merge
    df, voltage_labels = load_dpv_features("../data/Raw_data_dpv.xlsx", df_target)

    # 3. Create target variable
    df = create_target_variable(df, psa_column="PSA_pg_per_ml", cutoff=4000)

    # 4. Preprocess features (all 200 DPV current measurements)
    feature_columns = [f'V{i}' for i in range(200)]
    X_scaled, y, scaler = preprocess_features(df, feature_columns)

    # 5. Split data
    X_train, X_val, y_train, y_val = split_data(X_scaled, y, test_size=0.30, random_state=42)

    # 6. Initialize models
    models = initialize_models(X_train, y_train)

    # 7. Train models
    trained_models, results = train_models(models, X_train, y_train, X_val, y_val)

    save_dir = "../models"
    os.makedirs(save_dir, exist_ok=True)

    # 7b. Train GNN model
    print("\n" + "="*70)
    print("STEP 7b: GRAPH NEURAL NETWORK TRAINING")
    print("="*70)
    gnn_results, gnn_model, edge_index = train_gnn(X_train, y_train, X_val, y_val, feature_columns, X_scaled)
    if gnn_model is not None:
        # Save GNN model package
        gnn_package = {
            "model": gnn_model,
            "feature_names": feature_columns,
            "edge_index": edge_index
        }
        gnn_path = os.path.join(save_dir, "gnn_model.pkl")
        with open(gnn_path, "wb") as f:
            pickle.dump(gnn_package, f)
        print(f"  ✓ Saved: gnn_model.pkl")
        trained_models['GNN'] = gnn_model
        results['GNN'] = gnn_results

    # Save scaler and feature columns for the backend server
    with open(os.path.join(save_dir, "scaler.pkl"), "wb") as f:
        pickle.dump(scaler, f)
    with open(os.path.join(save_dir, "feature_columns.pkl"), "wb") as f:
        pickle.dump(feature_columns, f)
    print(f"\n  ✓ Saved: scaler.pkl and feature_columns.pkl")

    # 8. Generate all advanced visualizations
    print("\n" + "="*70)
    print("STEP 8: GENERATING ADVANCED VISUALIZATIONS")
    print("="*70)

    # Create figure directory
    os.makedirs("../figure", exist_ok=True)

    # 7.1 Advanced confusion matrices (different style per model)
    print("\n  📊 Creating advanced confusion matrices...")
    for name in results.keys():
        create_advanced_confusion_matrix(y_val, results[name]['y_pred'], name, "../figure")

    # 7.2 Radar chart
    print("\n  📈 Creating radar chart...")
    create_radar_chart(results, "../figure")

    # 7.3 Parallel coordinates
    print("\n  🔀 Creating parallel coordinates plot...")
    create_parallel_coordinates(results, y_val, "../figure")

    # 7.4 Ridge plot
    print("\n  ⛰️ Creating ridge plot...")
    create_ridge_plot(results, y_val, "../figure")

    # 7.5 Violin plots
    print("\n  🎻 Creating violin plots...")
    create_violin_plots(results, y_val, "../figure")

    # 7.6 Correlation heatmap
    print("\n  🔥 Creating correlation heatmap...")
    create_correlation_heatmap(df, "../figure", voltage_labels)

    # 7.7 Model performance summary chart
    print("\n  📊 Creating model performance summary chart...")
    create_model_performance_bar_chart(results, "../figure")

    # 7.8 Best model modal
    print("\n  🏅 Creating best model modal...")
    create_best_model_modal(results, "../figure")

    # 7.10 Enhanced ROC curves
    print("\n  📉 Creating enhanced ROC curves...")
    create_roc_curves_enhanced(results, y_val, "../figure")

    # 9. Find best model and print summary
    best_model_name = max(results, key=lambda x: results[x]['f1_score'])

    print("\n" + "="*70)
    print("PIPELINE COMPLETED SUCCESSFULLY")
    print("="*70)
    print(f"\n🏆 BEST MODEL: {best_model_name}")
    print(f"   F1-Score: {results[best_model_name]['f1_score']:.4f}")
    print(f"   Accuracy: {results[best_model_name]['accuracy']:.4f}")
    print(f"   ROC-AUC: {results[best_model_name]['roc_auc']:.4f}")

    # Create best model summary table
    create_best_model_summary_table(results, "../figure")

    print("\n📁 Output files saved to:")
    print("   • ../models/ - Trained models")
    print("   • ../figure/ - All visualizations")

    # Print performance table
    print("\n" + "="*70)
    print("MODEL PERFORMANCE SUMMARY")
    print("="*70)
    comparison_df = pd.DataFrame({
        name: {
            'Accuracy': results[name]['accuracy'],
            'Precision': results[name]['precision'],
            'Recall': results[name]['recall'],
            'F1-Score': results[name]['f1_score'],
            'ROC-AUC': results[name]['roc_auc']
        } for name in results
    }).T
    print(comparison_df.round(4))

    return trained_models, results, y_val

# ============================================================================
# RUN THE PIPELINE
# ============================================================================

if __name__ == "__main__":
    trained_models, results, y_val = run_complete_pipeline()
