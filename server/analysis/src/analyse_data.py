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
    average_precision_score,
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

import feature_extraction
import regression_pipeline
from cnn_model import train_cnn, train_bilstm
from console_reports import (
    print_fold_cv_metrics,
    print_biophysical_mapping,
    print_risk_stratification,
    print_optimal_threshold,
    print_top_features,
    print_ensemble_agreement,
    print_patient_prediction,
)

warnings.filterwarnings('ignore')

# Set global style for beautiful plots
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans']
plt.rcParams['axes.facecolor'] = 'white'
plt.rcParams['figure.facecolor'] = 'white'
plt.rcParams['axes.edgecolor'] = '#cccccc'
plt.rcParams['axes.linewidth'] = 1.0
plt.rcParams['grid.alpha'] = 0.3
plt.rcParams['grid.color'] = '#cccccc'
plt.rcParams['grid.linestyle'] = '--'

# Consistent publication-grade font sizing across ALL figures
plt.rcParams['font.size'] = 14
plt.rcParams['axes.titlesize'] = 16
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12
plt.rcParams['legend.fontsize'] = 12
plt.rcParams['figure.titlesize'] = 18
plt.rcParams['savefig.dpi'] = 300

# Unified professional palette (no red) — matches Figure1 fingerprint theme
PALETTE = ['#1f77b4', '#2ca02c', '#9467bd', '#ff7f0e', '#17becf', '#7f7f7f']

# Custom color palettes for each model (blue/teal/green/purple family, no red)
MODEL_COLORS = {
    'Logistic Regression': {
        'primary': '#1f77b4',
        'secondary': '#17becf',
        'gradient': ['#1f77b4', '#4a90c2', '#7fb3d5'],
        'heatmap': 'Blues'
    },
    'Random Forest': {
        'primary': '#2ca02c',
        'secondary': '#3B82F6',
        'gradient': ['#2ca02c', '#5fc24a', '#8fd97a'],
        'heatmap': 'Greens'
    },
    'SVM': {
        'primary': '#ff7f0e',
        'secondary': '#8B5CF6',
        'gradient': ['#ff7f0e', '#ffa94d', '#ffc078'],
        'heatmap': 'Oranges'
    },
    'XGBoost': {
        'primary': '#9467bd',
        'secondary': '#17becf',
        'gradient': ['#9467bd', '#b08cd4', '#cbb5e6'],
        'heatmap': 'Purples'
    },
    '1D-CNN': {
        'primary': '#17becf',
        'secondary': '#3B82F6',
        'gradient': ['#17becf', '#4cc9d6', '#8adde5'],
        'heatmap': 'Blues'
    },
    'BiLSTM': {
        'primary': '#7f7f7f',
        'secondary': '#3B82F6',
        'gradient': ['#7f7f7f', '#9c9c9c', '#c0c0c0'],
        'heatmap': 'Greys'
    }
}

# ============================================================================
# SECTION 1: DATA LOADING AND PREPROCESSING
# ============================================================================

def load_data(file_path):
    """Load the CSV data file."""
    print("\n" + "="*70)
    print("STEP 1: LOADING DATA")
    print("="*70)

    df = pd.read_csv(file_path)
    print(f"✓ Data loaded successfully")
    print(f"  Shape: {df.shape}")
    # print(f"  Columns: {df.columns.tolist()}")
    return df


def load_dpv_features(df):
    """Extract DPV voltammetry feature names from the CSV dataframe."""
    print("\n" + "="*70)
    print("STEP: LOADING DPV VOLTAMMETRY FEATURES")
    print("="*70)

    dpv_cols = [c for c in df.columns if c.startswith('curr_')]
    print(f"  Found {len(dpv_cols)} DPV current measurements per sample")
    print(f"  DPV feature matrix: {df[['sample_id'] + dpv_cols].shape}")

    voltage_labels = {c: c.replace('curr_', '').replace('mV', ' mV') for c in dpv_cols}
    return df, voltage_labels


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
    """Preprocess features: handle missing values, log transform (biomarkers only), and scale."""
    print("\n" + "="*70)
    print("STEP 3: FEATURE PREPROCESSING")
    print("="*70)

    # Remove missing values
    df_clean = df[feature_columns + ["high_risk"]].dropna()
    print(f"  Samples after removing missing values: {len(df_clean)}")

    # Separate features and target
    X = df_clean[feature_columns].copy()
    y = df_clean["high_risk"].copy()

    # Log transformation (biomarker columns only — DPV currents have negative values)
    X_log = X.copy()
    biomarker_cols = ["AFP_pg_per_ml", "CA125_U_per_ml"]
    dpv_cols = [c for c in feature_columns if c.startswith('curr_')]

    print(f"\n  Log transformation applied to biomarker columns:")
    for col in biomarker_cols:
        if col in X.columns:
            X_log[col] = np.log1p(X_log[col])
            print(f"    {col}: [{X[col].min():.2f}, {X[col].max():.2f}] → [{X_log[col].min():.2f}, {X_log[col].max():.2f}]")

    if dpv_cols:
        print(f"    Skipped log for {len(dpv_cols)} DPV current columns (contain negative values)")

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
            'ROC-AUC (%)': results[name]['roc_auc'] * 100 if results[name]['roc_auc'] is not None else 0,
            'PR-AUC (%)': results[name].get('pr_auc', 0) * 100 if results[name].get('pr_auc') is not None else 0
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
    print(f"  • PR-AUC: {best_model['PR-AUC (%)']:.2f}%")
    print("Model performance table (percentages):")
    print(summary_df.to_string(index=False, float_format='%.2f'))

    # CSV export removed — results shown on console only

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

    ax.text(0.02, 0.9, title, va='top')
    ax.text(0.02, 0.82, subtitle, va='top')

    for i, (label, value) in enumerate(zip(display_labels, best_values)):
        y = 0.65 - i * 0.12
        ax.text(0.05, y, f"{label}", va='center')
        ax.text(0.85, y, f"{value * 100:.2f}%", va='center', ha='right')
        ax.hlines(y - 0.02, 0.05, 0.95, linewidth=1, alpha=0.5)

    ax.text(0.02, 0.08, "This figure highlights only the best performing model based on F1 score, providing a concise single-model summary.", va='bottom')

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
    os.makedirs(save_dir, exist_ok=True)
    cm = confusion_matrix(y_true, y_pred)
    cm_percentage = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100

    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Use standard seaborn heatmap for ALL models
    cmap = 'Blues'
    
    sns.heatmap(
        cm,
        annot=np.array([[f'{cm[i, j]}\n({cm_percentage[i, j]:.1f}%)' for j in range(2)] for i in range(2)]),
        fmt='',
        cmap=cmap,
        xticklabels=['Low Risk', 'High Risk'],
        yticklabels=['Low Risk', 'High Risk'],
        ax=ax,
        cbar_kws={'label': 'Count', 'shrink': 0.8},
        square=True,
        linewidths=2,
        linecolor='white',
        annot_kws={'size': 14}
    )

    ax.set_title(f'{model_name}\nConfusion Matrix')
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')
    ax.set_xticklabels(ax.get_xticklabels())
    ax.set_yticklabels(ax.get_yticklabels())
    ax.grid(False)

    plt.tight_layout()
    plt.savefig(f"{save_dir}/advanced_cm_{model_name.lower().replace(' ', '_')}.png",
                dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved: advanced_cm_{model_name.lower().replace(' ', '_')}.png")

def create_radar_chart(results, save_dir="../figure"):
    """Create beautiful radar chart comparing all models."""
    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))

    metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC', 'PR-AUC']
    metric_keys = ['accuracy', 'precision', 'recall', 'f1_score', 'roc_auc', 'pr_auc']

    angles = [n / float(len(metrics)) * 2 * pi for n in range(len(metrics))]
    angles += angles[:1]

    colors_radar = PALETTE[:6]

    for idx, (name, color) in enumerate(zip(results.keys(), colors_radar)):
        values = [results[name].get(key, 0) if results[name].get(key) is not None else 0 for key in metric_keys]
        values += values[:1]

        ax.plot(angles, values, 'o-', linewidth=3, label=name, color=color, markersize=10,
                markeredgecolor='white', markeredgewidth=1.5)
        ax.fill(angles, values, alpha=0.06, color=color)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(metrics)
    ax.set_ylim(0, 1)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels(['0.2', '0.4', '0.6', '0.8', '1.0'])
    ax.set_title('Model Performance Radar Chart\nComprehensive Comparison')
    ax.legend(loc='upper right', bbox_to_anchor=(1.25, 1.12), framealpha=0.95)
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

    colors_parallel = {'Low Risk': '#2ca02c', 'High Risk': '#1f77b4'}

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
    ax.set_xticklabels(model_names, rotation=45, ha='right')
    ax.set_ylabel('Prediction Probability')
    ax.set_title('Model Predictions Parallel Coordinates\nHow Models Behave Across Samples')
    ax.legend(loc='upper left', framealpha=0.95)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_ylim(-0.05, 1.05)

    plt.tight_layout()
    plt.savefig(f"{save_dir}/parallel_coordinates.png", dpi=300, bbox_inches='tight')
    plt.close()
    print("  ✓ Saved: parallel_coordinates.png")

def create_ridge_plot(results, y_val, save_dir="../figure"):
    """Create ridge plot (joyplot) for probability distributions using Seaborn."""
    
    records = []
    for name in results.keys():
        if results[name]['y_pred_proba'] is not None:
            for true_label, prob in zip(y_val, results[name]['y_pred_proba']):
                records.append({'Model': name, 'True Label': 'High Risk' if true_label == 1 else 'Low Risk', 'Prediction Probability': prob})
                
    df_ridge = pd.DataFrame(records)
    if len(df_ridge) == 0:
        return
        
    # Use seaborn facet grid to simulate ridge plot
    sns.set_theme(style="white", rc={"axes.facecolor": (0, 0, 0, 0)})
    
    g = sns.FacetGrid(df_ridge, row="Model", hue="True Label", aspect=10, height=1.2, 
                      palette={'Low Risk': '#2ca02c', 'High Risk': '#1f77b4'})
    
    g.map_dataframe(sns.kdeplot, x="Prediction Probability", fill=True, alpha=0.5, linewidth=2)
    g.map_dataframe(sns.kdeplot, x="Prediction Probability", color="black", lw=1)
    
    # Pass axes to refline manually
    for ax in g.axes.flat:
        ax.axhline(y=0, lw=2, clip_on=False, color="black")
    
    # Label the model names on the left of each row (avoiding hue label tangling)
    for ax, name in zip(g.axes.flat, g.row_names):
        ax.text(0.01, 0.2, name, fontweight="bold", color="black",
                ha="left", va="center", transform=ax.transAxes)
                
    g.set_titles("")
    g.set(yticks=[], ylabel="")
    g.despine(bottom=True, left=True)
    g.fig.subplots_adjust(hspace=-0.25)
    g.set_axis_labels("Prediction Probability", "")
    g.fig.suptitle('Model Prediction Distributions (Ridge Plot)', y=1.02)
    
    # Add a custom legend for the hues since we cleared titles
    g.add_legend(title='True Label', bbox_to_anchor=(0.95, 0.95))
    
    plt.savefig(f"{save_dir}/ridge_plot_distributions.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # Reset seaborn theme to whitegrid for the rest of the script
    sns.set_theme(style="whitegrid", font_scale=1.2)
    print("  ✓ Saved: ridge_plot_distributions.png")

def create_violin_plots(results, y_val, save_dir="../figure"):
    """Create beautiful violin plots for model predictions using Seaborn."""
    
    # Gather data into a DataFrame for seaborn
    records = []
    for name in results.keys():
        if results[name]['y_pred_proba'] is not None:
            for true_label, prob in zip(y_val, results[name]['y_pred_proba']):
                records.append({'Model': name, 'True Label': 'High Risk' if true_label == 1 else 'Low Risk', 'Prediction Probability': prob})
    
    df_violin = pd.DataFrame(records)
    if len(df_violin) == 0:
        return
        
    fig, ax = plt.subplots(figsize=(15, 8))
    
    sns.violinplot(
        data=df_violin, 
        x='Model', 
        y='Prediction Probability', 
        hue='True Label', 
        split=True, 
        inner='quartile',
        palette={'Low Risk': '#2ca02c', 'High Risk': '#1f77b4'},
        ax=ax,
        linewidth=1.5
    )

    ax.set_title('Model Prediction Distributions by True Label')
    ax.set_ylim(-0.05, 1.05)
    
    plt.tight_layout()
    plt.savefig(f"{save_dir}/violin_plots.png", dpi=300, bbox_inches='tight')
    plt.close()
    print("  ✓ Saved: violin_plots.png")

def create_correlation_heatmap(df, save_dir="../figure", voltage_labels=None):
    """Heatmap showing correlations between biomarkers/DPV features and the target."""
    if voltage_labels is not None:
        dpv_cols = [c for c in df.columns if c.startswith('curr_')]
        corr_with_target = df[dpv_cols].corrwith(df['high_risk']).abs().sort_values(ascending=False)
        top_n = min(20, len(corr_with_target))
        top_features = corr_with_target.head(top_n).index.tolist()
        corr_cols = top_features + ['PSA_pg_per_ml', 'AFP_pg_per_ml', 'CA125_U_per_ml', 'high_risk']
        corr_labels = [voltage_labels.get(c, c) for c in top_features] + ['PSA', 'AFP', 'CA125', 'High Risk']
    else:
        corr_cols = ['PSA_pg_per_ml', 'AFP_pg_per_ml', 'CA125_U_per_ml', 'high_risk']
        corr_labels = ['PSA', 'AFP', 'CA125', 'High Risk']

    corr_matrix = df[corr_cols].corr()

    fig, ax = plt.subplots(figsize=(16, 10))
    sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='RdBu_r',
                xticklabels=corr_labels, yticklabels=corr_labels,
                vmin=-1, vmax=1, center=0, square=True,
                linewidths=0.5, linecolor='white', ax=ax,
                annot_kws={'size': 8},
                cbar_kws={'label': 'Pearson Correlation', 'shrink': 0.8})

    ax.set_title('DPV Feature & Biomarker Correlation Heatmap')
    plt.xticks(rotation=45, ha='right')
    plt.yticks()
    ax.figure.axes[-1].set_ylabel('Pearson Correlation')
    plt.tight_layout()
    plt.savefig(f"{save_dir}/correlation_heatmap.png", dpi=300, bbox_inches='tight')
    plt.close()
    print("  ✓ Saved: correlation_heatmap.png")


def create_model_performance_bar_chart(results, save_dir="../figure"):
    """Create a grouped bar chart for the full model performance summary."""
    os.makedirs(save_dir, exist_ok=True)
    metrics = ['accuracy', 'precision', 'recall', 'f1_score', 'roc_auc', 'pr_auc']
    labels = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC', 'PR-AUC']

    chart_data = {
        name: [results[name].get(metric, 0) if results[name].get(metric) is not None else 0 for metric in metrics]
        for name in results
    }
    df = pd.DataFrame(chart_data, index=labels)
    df.index.name = 'Metric'
    df_reset = df.reset_index()
    df_melt = df_reset.melt(id_vars='Metric', var_name='Model', value_name='Score')

    fig, ax = plt.subplots(figsize=(16, 7))
    sns.barplot(data=df_melt, x='Metric', y='Score', hue='Model', palette=PALETTE, edgecolor='black', linewidth=1, ax=ax)

    ax.set_title('Model Performance Summary')
    ax.set_ylabel('Score')
    ax.set_ylim(0, 1.15)
    ax.legend(title='Model', loc='upper right', bbox_to_anchor=(1.0, 1.0))
    ax.grid(axis='y', alpha=0.3, linestyle='--')

    # Add percentages on bars
    for patch in ax.patches:
        height = patch.get_height()
        if height is not None and height > 0:
            ax.annotate(f'{height*100:.1f}%',
                        (patch.get_x() + patch.get_width() / 2, height),
                        fontsize=9,
                        ha='center', va='bottom', xytext=(0, 4), textcoords='offset points')

    plt.tight_layout()
    plt.savefig(f"{save_dir}/model_performance_summary.png", dpi=300, bbox_inches='tight')
    plt.close()
    print("  ✓ Saved: model_performance_summary.png")


def create_roc_curves_enhanced(results, y_val, save_dir="../figure"):
    """Create enhanced ROC curves with confidence intervals and annotations."""
    fig, ax = plt.subplots(figsize=(13, 7))

    colors_roc = PALETTE[:6]

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

    ax.set_xlabel('False Positive Rate (1 - Specificity)')
    ax.set_ylabel('True Positive Rate (Sensitivity)')
    ax.set_title('ROC Curves - Enhanced Comparison')
    ax.legend(loc='lower right', framealpha=0.95)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_xlim([-0.02, 1.02])
    ax.set_ylim([-0.02, 1.02])

    # Add text box with interpretation
    best_model = max(results, key=lambda x: results[x]['roc_auc'] if results[x]['roc_auc'] is not None else 0)
    best_auc = results[best_model]['roc_auc']
    interpretation = f"Best Model: {best_model}\nAUC = {best_auc:.3f}\nExcellent discrimination" if best_auc > 0.8 else "Moderate discrimination"

    ax.text(0.02, 0.02, interpretation, transform=ax.transAxes,
           verticalalignment='bottom', bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

    plt.tight_layout()
    plt.savefig(f"{save_dir}/roc_curves_enhanced.png", dpi=300, bbox_inches='tight')
    plt.close()
    print("  ✓ Saved: roc_curves_enhanced.png")

def create_dpv_fingerprint_plot(df, save_dir="../figure"):
    """Figure 1: DPV fingerprint comparison between low-risk and high-risk cohorts."""
    curr_cols = [c for c in df.columns if c.startswith('curr_')]
    voltages = np.array([float(c.replace('curr_', '').replace('mV', '')) for c in curr_cols])
    X = df[curr_cols].values
    y = df['high_risk'].values

    mean_low = X[y == 0].mean(axis=0)
    std_low = X[y == 0].std(axis=0)
    mean_high = X[y == 1].mean(axis=0)
    std_high = X[y == 1].std(axis=0)

    fig, ax = plt.subplots(figsize=(11, 6), dpi=300)
    ax.plot(voltages, mean_low, label=f'Baseline Low-Risk Cohort (n={(y==0).sum()})', linewidth=2)
    ax.fill_between(voltages, mean_low - std_low, mean_low + std_low, alpha=0.15)
    ax.plot(voltages, mean_high, label=f'High-Risk Cohort (n={(y==1).sum()})', linewidth=2)
    ax.fill_between(voltages, mean_high - std_high, mean_high + std_high, alpha=0.15)

    peaks = [
        (-468, 'PSA Peak\n(-468 mV)', '#2ca02c', 'top'),
        (365, 'AFP Peak\n(365 mV)', '#ff7f0e', 'bottom'),
        (968, 'CA125 Peak\n(968 mV)', '#9467bd', 'top'),
    ]
    ymax = max(mean_low.max(), mean_high.max())
    ymin = min(mean_low.min(), mean_high.min())
    for volt, label, color, pos in peaks:
        ax.axvline(x=volt, color=color, linestyle='--', linewidth=1.5, alpha=0.8)
        if pos == 'top':
            ax.text(volt + 15, ymax - 0.3, label, color=color, verticalalignment='top')
        else:
            ax.text(volt + 15, ymin + 0.3, label, color=color, verticalalignment='bottom')

    ax.set_xlabel('Applied Potential E (mV)')
    ax.set_ylabel('Differential Current i (μA)')
    ax.set_title('Multiplexed Voltammetric Fingerprints across Screening Cohorts')
    ax.legend(loc='upper right', frameon=True)
    fig.tight_layout()
    fig.savefig(os.path.join(save_dir, 'Figure1_DPV_Fingerprints.png'))
    plt.close()
    print(f"  ✓ Saved: Figure1_DPV_Fingerprints.png")


def create_feature_importance_map(trained_models, feature_columns, save_dir="../figure"):
    """Figure 3: Feature importance map using the best tree-based model's importances."""
    curr_cols = [c for c in feature_columns if c.startswith('curr_')]
    voltages = np.array([float(c.replace('curr_', '').replace('mV', '')) for c in curr_cols])

    best_name = None
    best_importances = None
    for name, model in trained_models.items():
        if hasattr(model, 'feature_importances_') and len(model.feature_importances_) == len(feature_columns):
            best_name = name
            dpv_idx = [i for i, c in enumerate(feature_columns) if c.startswith('curr_')]
            best_importances = model.feature_importances_[dpv_idx]
            break

    if best_importances is None:
        print("  ⚠️ No model with feature_importances_ found, using coeffs")
        for name, model in trained_models.items():
            if hasattr(model, 'coef_'):
                coefs = np.abs(model.coef_[0])
                dpv_idx = [i for i, c in enumerate(feature_columns) if c.startswith('curr_')]
                best_importances = coefs[dpv_idx]
                best_name = name
                break

    if best_importances is None:
        print("  ⚠️ Could not generate feature importance map — no suitable model")
        return

    fig, ax = plt.subplots(figsize=(11, 5.5), dpi=300)
    ax.plot(voltages, best_importances, color='#2b5c8f', linewidth=2, label=f'{best_name} Feature Importance')
    ax.fill_between(voltages, 0, best_importances, color='#2b5c8f', alpha=0.25)

    # Shaded biomarker windows with labels placed above (legend-free to avoid tangling)
    ax.axvspan(-468, -448, color='#2ca02c', alpha=0.18)
    ax.axvspan(365, 385, color='#ff7f0e', alpha=0.18)
    ax.axvspan(958, 978, color='#9467bd', alpha=0.18)
    for (x0, x1, text, color) in [(-468, -448, 'PSA Window', '#2ca02c'),
                                  (365, 385, 'AFP Window', '#ff7f0e'),
                                  (958, 978, 'CA125 Window', '#9467bd')]:
        mid = (x0 + x1) / 2
        ax.text(mid, ax.get_ylim()[1] * 0.97, text, ha='center', va='top', color=color)

    ax.legend(loc='upper right', frameon=True)
    ax.set_xlabel('Applied Potential E (mV)')
    ax.set_ylabel('Feature Importance')
    ax.set_title('Feature Importance Map & Physical Redox Mapping')
    fig.tight_layout()
    fig.savefig(os.path.join(save_dir, 'Figure3_Feature_Importance_Map.png'))
    plt.close()
    print(f"  ✓ Saved: Figure3_Feature_Importance_Map.png")


def create_workflow_diagram(save_dir="../figure"):
    """Figure 1: Overall biosensor + AI workflow diagram (graphical abstract)."""
    fig, ax = plt.subplots(figsize=(18, 7), dpi=300)
    ax.axis("off")

    steps = [
        ("1. Serum sample", "1,000 spiked samples\nknown PSA / AFP / CA125", "#1f77b4"),
        ("2. Multiplexed electrode", "validated sensing surface\nfor PSA, AFP, CA125", "#2ca02c"),
        ("3. DPV measurement", "potential sweep -750 to +1250 mV\n200 current points", "#ff7f0e"),
        ("4. Digital fingerprint", "200-point current-potential\nvector per sample", "#9467bd"),
        ("5. AI decoder", "regression hierarchy\nXGBoost best (multi-output)", "#17becf"),
        ("6. Outputs", "PSA  |  AFP  |  CA125\nquantitative concentrations", "#7f7f7f"),
    ]

    n = len(steps)
    box_w, box_h = 0.13, 0.5
    gap = 0.025
    start_x = 0.02

    for i, (title, sub, color) in enumerate(steps):
        x0 = start_x + i * (box_w + gap)
        y0 = 0.28
        box = plt.Rectangle((x0, y0), box_w, box_h, facecolor=color, edgecolor="white",
                            linewidth=2, alpha=0.92, zorder=3)
        ax.add_patch(box)
        ax.text(x0 + box_w / 2, y0 + box_h - 0.07, title, ha="center", va="top", color="white", zorder=4, fontsize=11, fontweight='bold')
        ax.text(x0 + box_w / 2, y0 + 0.12, sub, ha="center", va="center", color="white", zorder=4, fontsize=10)

        if i < n - 1:
            ax.annotate("", xy=(x0 + box_w + gap - 0.012, y0 + box_h / 2),
                        xytext=(x0 + box_w + 0.012, y0 + box_h / 2),
                        arrowprops=dict(arrowstyle="->", color="#333333", lw=2, zorder=5))

    ax.text(0.5, 0.93, "AI-Assisted Decoding of Multiplexed Electrochemical Fingerprints",
            ha="center", va="top")
    ax.text(0.5, 0.86, "One DPV scan in -> PSA, AFP, and CA125 concentrations out",
            ha="center", va="top")

    ax.text(0.5, 0.18, "Optional secondary layer: PSA-threshold classification (4,000 pg/ml cutoff)",
            ha="center", va="top", style="italic")

    fig.tight_layout()
    path = os.path.join(save_dir, "workflow_overview.png")
    fig.savefig(path, bbox_inches="tight")
    plt.close()
    print(f"  ✓ Saved: workflow_overview.png")
    return path


def create_pr_curves(results, y_val, save_dir="../figure"):
    """Figure 2 companion: Precision-Recall curves for all models (across CV folds)."""
    fig, ax = plt.subplots(figsize=(13, 7), dpi=300)
    colors = PALETTE[:6]

    for idx, (name, result) in enumerate(results.items()):
        if result["y_pred_proba"] is None:
            continue
        precision, recall, _ = precision_recall_curve(y_val, result["y_pred_proba"])
        pr_auc = average_precision_score(y_val, result["y_pred_proba"])
        color = colors[idx % len(colors)]
        ax.plot(recall, precision, linewidth=2.5, label=f'{name} (PR-AUC = {pr_auc:.3f})', color=color)
        ax.fill_between(recall, 0, precision, alpha=0.08, color=color)

    ax.axhline(y_val.values.mean(), color='k', linestyle='--', linewidth=1.5, alpha=0.6,
               label=f'Baseline (prevalence = {y_val.values.mean():.2f})')
    ax.set_xlabel('Recall (Sensitivity)')
    ax.set_ylabel('Precision (Positive Predictive Value)')
    ax.set_title('Precision-Recall Curves — 5-Fold Cross-Validation')
    ax.legend(loc='upper right', framealpha=0.95)
    ax.grid(alpha=0.3, linestyle='--')
    ax.set_xlim([-0.02, 1.02])
    ax.set_ylim([0, 1.05])
    fig.tight_layout()
    fig.savefig(os.path.join(save_dir, 'pr_curves.png'))
    plt.close()
    print(f"  ✓ Saved: pr_curves.png")


def create_shap_plot(trained_models, feature_columns, X_scaled, y_true, save_dir="../figure"):
    """Figure 3: SHAP summary confirming the -468 mV PSA redox window as primary driver."""
    try:
        import shap
    except ImportError:
        print("  ⚠️ SHAP not installed — skipping SHAP plot")
        return

    curr_cols = [c for c in feature_columns if c.startswith('curr_')]
    voltages = np.array([float(c.replace('curr_', '').replace('mV', '')) for c in curr_cols])
    dpv_idx = [feature_columns.index(c) for c in curr_cols]

    if len(dpv_idx) == 0 or len(X_scaled) == 0:
        print("  ⚠️ No DPV features available for SHAP")
        return

    X_dpv = np.asarray(X_scaled)[:, dpv_idx]

    explainer = None
    model = None
    model_name = None
    name_map = {
        "1D-CNN": ["1D-CNN", "1d-cnn", "1d_cnn", "CNN"],
        "BiLSTM": ["BiLSTM", "bilstm", "BILSTM"],
        "XGBoost": ["XGBoost", "xgboost", "Xgboost"],
        "Random Forest": ["Random Forest", "random_forest"],
    }
    for pretty, keys in name_map.items():
        for k in keys:
            if k in trained_models and trained_models[k] is not None:
                model = trained_models[k]
                model_name = pretty
                break
        if model is not None:
            break

    if model is None:
        print("  ⚠️ No suitable model for SHAP")
        return

    if model_name in ["1D-CNN", "BiLSTM"]:
        try:
            import torch
            background = torch.FloatTensor(X_dpv[:100])
            explainer = shap.GradientExplainer(model, background)
            shap_values = explainer.shap_values(torch.FloatTensor(X_dpv[:100]))
            if isinstance(shap_values, list):
                shap_values = shap_values[1] if len(shap_values) > 1 else shap_values[0]
            shap_values = np.asarray(shap_values).squeeze()
        except Exception as e:
            print(f"  ⚠️ GradientExplainer failed ({e}) — falling back to XGBoost")
            explainer = None

    if explainer is None:
        xgb = trained_models.get("XGBoost")
        if xgb is None:
            print("  ⚠️ No XGBoost available for SHAP fallback")
            return
        explainer = shap.TreeExplainer(xgb)
        # XGBoost was trained on ALL feature_columns (incl. biomarkers), so pass full X_scaled
        shap_values = explainer.shap_values(np.asarray(X_scaled)[:100])
        if isinstance(shap_values, list):
            shap_values = shap_values[1] if len(shap_values) > 1 else shap_values[0]
        shap_values = np.asarray(shap_values)
        # Keep only the DPV columns to match the voltage axis
        if shap_values.ndim >= 2 and shap_values.shape[-1] == len(feature_columns):
            shap_values = shap_values[:, dpv_idx]
        model_name = "XGBoost"

    shap_values = np.asarray(shap_values)
    if shap_values.ndim == 3:
        shap_values = shap_values[..., 1] if shap_values.shape[-1] == 2 else shap_values.mean(axis=-1)
    shap_values = np.asarray(shap_values).squeeze()
    if shap_values.ndim == 1:
        shap_values = shap_values.reshape(1, -1)

    mean_abs = np.mean(np.abs(shap_values), axis=0)

    # Plot bars on the voltage axis so windows and bars share the same coordinate space
    bar_width = voltages[1] - voltages[0] if len(voltages) > 1 else 1.0

    fig, ax = plt.subplots(figsize=(12, 6), dpi=300)
    ax.bar(voltages, mean_abs, color='#2b5c8f', alpha=0.85, width=bar_width * 0.9)

    # Shaded biomarker windows on the same voltage axis
    ax.axvspan(-468, -448, color='#2ca02c', alpha=0.18)
    ax.axvspan(365, 385, color='#ff7f0e', alpha=0.18)
    ax.axvspan(958, 978, color='#9467bd', alpha=0.18)

    # Window labels placed above the shaded regions, near the top edge
    for (x0, x1, text, color) in [(-468, -448, 'PSA Window', '#2ca02c'),
                                  (365, 385, 'AFP Window', '#ff7f0e'),
                                  (958, 978, 'CA125 Window', '#9467bd')]:
        mid = (x0 + x1) / 2
        ax.text(mid, ax.get_ylim()[1] * 0.97, text, ha='center', va='top', color=color)

    # Nicer tick positions on the voltage axis (9 evenly spaced voltage labels)
    tick_positions = np.linspace(voltages[0], voltages[-1], 9)
    ax.set_xticks(tick_positions)
    ax.set_xticklabels([f'{v:.0f}' for v in tick_positions])
    ax.set_xlabel('Applied Potential E (mV)')
    ax.set_ylabel('Mean |SHAP Value|')
    # ax.set_title(f'SHAP Feature Importance Map ({model_name}) — PSA Redox Window Dominance')
    ax.set_title('SHAP Feature Importance Map — PSA Redox Window Dominance')
    ax.grid(alpha=0.3, axis='y', linestyle='--')
    fig.tight_layout()
    fig.savefig(os.path.join(save_dir, 'shap_importance.png'))
    plt.close()
    print(f"  ✓ Saved: shap_importance.png")


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

    # 1. Load data from CSV
    df = load_data("../data/Raw_DPV_Dataset_for_Cancer_biomarker.csv")

    # 2. Extract DPV feature names
    df, voltage_labels = load_dpv_features(df)

    # 3. Create target variable
    df = create_target_variable(df, psa_column="PSA_pg_per_ml", cutoff=4000)

    # 4. Prepare raw features (log1p biomarkers only — scaling happens inside folds)
    feature_columns = ["AFP_pg_per_ml", "CA125_U_per_ml"] + [c for c in df.columns if c.startswith('curr_')]
    dpv_cols = [c for c in df.columns if c.startswith('curr_')]
    dpv_idx = [feature_columns.index(c) for c in dpv_cols]

    df_clean = df[feature_columns + ["high_risk"]].dropna()
    X_raw = df_clean[feature_columns].copy()
    y = df_clean["high_risk"].copy()

    X_raw["AFP_pg_per_ml"] = np.log1p(X_raw["AFP_pg_per_ml"])
    X_raw["CA125_U_per_ml"] = np.log1p(X_raw["CA125_U_per_ml"])
    print(f"  Final feature matrix shape (before CV): {X_raw.shape}")

    # 5. Stratified 5-Fold Cross-Validation with localized scaling
    print("\n" + "="*70)
    print("STEP 5: STRATIFIED 5-FOLD CROSS-VALIDATION (localized scaling)")
    print("="*70)

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    model_names = ["Logistic Regression", "Random Forest", "SVM", "XGBoost", "1D-CNN", "BiLSTM"]

    n_samples = len(y)
    oof_pred = {name: np.zeros(n_samples, dtype=int) for name in model_names}
    oof_proba = {name: np.zeros(n_samples) for name in model_names}

    # Per-fold metric collection for the fold-wise CV report
    metric_keys = ["accuracy", "precision", "recall", "f1_score", "roc_auc"]
    fold_scores = {name: {m: [] for m in metric_keys} for name in model_names}

    for fold, (train_idx, val_idx) in enumerate(skf.split(X_raw, y)):
        X_train_raw = X_raw.iloc[train_idx].values
        X_val_raw = X_raw.iloc[val_idx].values
        y_train = y.iloc[train_idx]
        y_val_fold = y.iloc[val_idx]
        y_val_fold_np = y_val_fold.values if hasattr(y_val_fold, "values") else y_val_fold

        # LOCALIZED SCALING — fit on train fold only to prevent data leakage
        scaler = RobustScaler().fit(X_train_raw)
        X_train = scaler.transform(X_train_raw)
        X_val = scaler.transform(X_val_raw)

        # Sequence scaling for CNN/BiLSTM (DPV currents only, localized)
        seq_scaler = RobustScaler().fit(X_train_raw[:, dpv_idx])
        X_train_seq = seq_scaler.transform(X_train_raw[:, dpv_idx])
        X_val_seq = seq_scaler.transform(X_val_raw[:, dpv_idx])

        # Baseline classifiers
        models = initialize_models(X_train, y_train)
        for name in ["Logistic Regression", "Random Forest", "SVM", "XGBoost"]:
            model = models[name]
            model.fit(X_train, y_train)
            y_pred = model.predict(X_val)
            y_proba = model.predict_proba(X_val)[:, 1] if hasattr(model, "predict_proba") else y_pred
            oof_pred[name][val_idx] = y_pred
            oof_proba[name][val_idx] = y_proba
            fold_scores[name]["accuracy"].append(accuracy_score(y_val_fold, y_pred))
            fold_scores[name]["precision"].append(precision_score(y_val_fold, y_pred, zero_division=0))
            fold_scores[name]["recall"].append(recall_score(y_val_fold, y_pred, zero_division=0))
            fold_scores[name]["f1_score"].append(f1_score(y_val_fold, y_pred, zero_division=0))
            fold_scores[name]["roc_auc"].append(roc_auc_score(y_val_fold, y_proba))

        # Sequence-aware models: 1D-CNN and BiLSTM
        for name, train_fn in [("1D-CNN", train_cnn), ("BiLSTM", train_bilstm)]:
            res, _ = train_fn(X_train_seq, y_train, X_val_seq, y_val_fold, input_length=len(dpv_cols), epochs=50)
            oof_pred[name][val_idx] = res["y_pred"]
            oof_proba[name][val_idx] = res["y_pred_proba"]
            fold_scores[name]["accuracy"].append(res["accuracy"])
            fold_scores[name]["precision"].append(res["precision"])
            fold_scores[name]["recall"].append(res["recall"])
            fold_scores[name]["f1_score"].append(res["f1_score"])
            fold_scores[name]["roc_auc"].append(res["roc_auc"])

        print(f"  ✓ Fold {fold+1}/5 complete")

    # 6. Aggregate out-of-fold predictions into final results
    print("\n" + "="*70)
    print("STEP 6: AGGREGATED CROSS-VALIDATION METRICS")
    print("="*70)

    results = {}
    trained_models = {}
    y_val = y  # full target (out-of-fold predictions span all samples)

    for name in model_names:
        y_pred_all = oof_pred[name]
        y_proba_all = oof_proba[name]
        y_all = y.values
        results[name] = {
            "accuracy": accuracy_score(y_all, y_pred_all),
            "precision": precision_score(y_all, y_pred_all, zero_division=0),
            "recall": recall_score(y_all, y_pred_all, zero_division=0),
            "f1_score": f1_score(y_all, y_pred_all, zero_division=0),
            "roc_auc": roc_auc_score(y_all, y_proba_all),
            "pr_auc": average_precision_score(y_all, y_proba_all),
            "confusion_matrix": confusion_matrix(y_all, y_pred_all),
            "y_pred": y_pred_all,
            "y_pred_proba": y_proba_all,
        }

    # Console reports (computed from real data, see console_reports.py)
    print_fold_cv_metrics(fold_scores)
    print_biophysical_mapping(df, dpv_cols)

    # Print confusion matrix counts table (TN/FP/FN/TP) — computed from real OOF predictions
    print("\n" + "="*70)
    print("CONFUSION MATRIX COUNTS (out-of-fold predictions)")
    print("="*70)
    print(f"{'Model':<22}{'TN':>6}{'FP':>6}{'FN':>6}{'TP':>6}")
    print("-" * 46)
    for name in model_names:
        tn, fp, fn, tp = results[name]["confusion_matrix"].ravel()
        print(f"{name:<22}{tn:>6}{fp:>6}{fn:>6}{tp:>6}")

    # Persist OOF metrics so the server displays the SAME numbers as the terminal
    import json
    metrics_out = {
        "model_names": model_names,
        "models": {
            name: {
                "accuracy": results[name]["accuracy"],
                "precision": results[name]["precision"],
                "recall": results[name]["recall"],
                "f1_score": results[name]["f1_score"],
                "roc_auc": results[name]["roc_auc"],
                "pr_auc": results[name]["pr_auc"],
                "confusion_matrix": results[name]["confusion_matrix"].ravel().tolist(),
            }
            for name in model_names
        },
    }
    with open(os.path.join("../models", "performance_summary.json"), "w") as f:
        json.dump(metrics_out, f)
    print(f"\n  ✓ Saved: performance_summary.json (OOF metrics for server)")

    # 7. Retrain final deployment models on full data
    save_dir = "../models"
    os.makedirs(save_dir, exist_ok=True)

    print("\n" + "="*70)
    print("STEP 7: TRAINING FINAL DEPLOYMENT MODELS (full data)")
    print("="*70)

    scaler = RobustScaler().fit(X_raw.values)
    X_scaled = scaler.transform(X_raw.values)
    seq_scaler = RobustScaler().fit(X_raw.values[:, dpv_idx])
    X_seq_full = seq_scaler.transform(X_raw.values[:, dpv_idx])

    models = initialize_models(X_scaled, y)
    for name, model in models.items():
        model.fit(X_scaled, y)
        filename = f"{name.lower().replace(' ', '_')}_model.pkl"
        file_path = os.path.join(save_dir, filename)
        with open(file_path, "wb") as file:
            pickle.dump(model, file)
        trained_models[name] = model
        print(f"  ✓ Saved: {filename}")

    for name, train_fn in [("1D-CNN", train_cnn), ("BiLSTM", train_bilstm)]:
        res, model = train_fn(X_seq_full, y, X_seq_full, y, input_length=len(dpv_cols), epochs=50)
        filename = f"{name.lower()}_model.pkl"
        file_path = os.path.join(save_dir, filename)
        with open(file_path, "wb") as file:
            pickle.dump(model, file)
        trained_models[name] = model
        print(f"  ✓ Saved: {filename}")

    with open(os.path.join(save_dir, "scaler.pkl"), "wb") as f:
        pickle.dump(scaler, f)
    with open(os.path.join(save_dir, "feature_columns.pkl"), "wb") as f:
        pickle.dump(feature_columns, f)
    print(f"\n  ✓ Saved: scaler.pkl and feature_columns.pkl")

    # Print the per-patient prediction summary (same numbers as Electron /predict)
    print_patient_prediction(trained_models, scaler, feature_columns, X_scaled[:1])

    # 8. Generate all advanced visualizations
    print("\n" + "="*70)
    print("STEP 8: GENERATING ADVANCED VISUALIZATIONS")
    print("="*70)

    # Create figure directory
    os.makedirs("../figure", exist_ok=True)

    # 7.1 Advanced confusion matrices (different style per model)
    print("\n Creating advanced confusion matrices...")
    for name in results.keys():
        create_advanced_confusion_matrix(y_val, results[name]['y_pred'], name, "../figure")

    # 7.2 Radar chart
    print("\n Creating radar chart...")
    create_radar_chart(results, "../figure")

    # 7.3 Parallel coordinates
    print("\n Creating parallel coordinates plot...")
    create_parallel_coordinates(results, y_val, "../figure")

    # 7.4 Ridge plot
    print("\n Creating ridge plot...")
    create_ridge_plot(results, y_val, "../figure")

    # 7.5 Violin plots
    print("\n Creating violin plots...")
    create_violin_plots(results, y_val, "../figure")

    # 7.6 Correlation heatmap
    print("\n Creating correlation heatmap...")
    create_correlation_heatmap(df, "../figure", voltage_labels)

    # 7.7 Model performance summary chart
    print("\n Creating model performance summary chart...")
    create_model_performance_bar_chart(results, "../figure")

    # 7.8 Best model modal
    print("\n Creating best model modal...")
    create_best_model_modal(results, "../figure")

    # 7.10 Enhanced ROC curves
    print("\n Creating enhanced ROC curves...")
    create_roc_curves_enhanced(results, y_val, "../figure")

    # 7.10b Precision-Recall curves (Figure 2 companion)
    print("\n Creating Precision-Recall curves...")
    create_pr_curves(results, y_val, "../figure")

    # 7.10c SHAP explanation plot
    print("\n Creating SHAP explanation plot...")
    create_shap_plot(trained_models, feature_columns, X_scaled, y.values, "../figure")

    # 7.11 DPV Fingerprint Comparison Plot
    print("\n Creating DPV fingerprint comparison plot...")
    create_dpv_fingerprint_plot(df, "../figure")

    # 7.12 Feature Importance Map
    print("\n Creating feature importance map...")
    create_feature_importance_map(trained_models, feature_columns, "../figure")

    # 7.13 Overall biosensor + AI workflow diagram (Figure 1 / graphical abstract)
    print("\n Creating biosensor + AI workflow diagram...")
    create_workflow_diagram("../figure")

    # 9. Find best model and print summary
    best_model_name = max(results, key=lambda x: results[x]['f1_score'])

    print("\n" + "="*70)
    print("PIPELINE COMPLETED SUCCESSFULLY")
    print("="*70)
    print(f"\n BEST MODEL: {best_model_name}")
    print(f"   F1-Score: {results[best_model_name]['f1_score']:.4f}")
    print(f"   Accuracy: {results[best_model_name]['accuracy']:.4f}")
    print(f"   ROC-AUC: {results[best_model_name]['roc_auc']:.4f}")
    print(f"   PR-AUC: {results[best_model_name]['pr_auc']:.4f}")

    # Create best model summary table
    create_best_model_summary_table(results, "../figure")

    print("\n Output files saved to:")
    print("   • ../models/ - Trained models")
    print("   • ../figure/ - All visualizations")

    # Print performance table
    print("\n" + "="*70)
    print("MODEL PERFORMANCE SUMMARY (5-Fold CV, out-of-fold)")
    print("="*70)
    comparison_df = pd.DataFrame({
        name: {
            'Accuracy': results[name]['accuracy'],
            'Precision': results[name]['precision'],
            'Recall': results[name]['recall'],
            'F1-Score': results[name]['f1_score'],
            'ROC-AUC': results[name]['roc_auc'],
            'PR-AUC': results[name]['pr_auc']
        } for name in results
    }).T
    print(comparison_df.round(4))

    # Advanced console reports (best model + ensemble level)
    best_proba = results[best_model_name]["y_pred_proba"]
    print_risk_stratification(best_proba, y_val.values, best_model_name)
    print_optimal_threshold(best_proba, y_val.values, best_model_name)
    print_top_features(trained_models, feature_columns, dpv_cols)
    print_ensemble_agreement(results, n_samples, model_names)

    return trained_models, results, y_val

# ============================================================================
# RUN THE PIPELINE
# ============================================================================

if __name__ == "__main__":
    trained_models, results, y_val = run_complete_pipeline()

    # Quantitative multi-biomarker regression (P1-P10) - the central experiment.
    # Engineering features are regenerated first, then the regression pipeline
    # reads the new feature file (data_with_features_engineered.csv).
    print("\n" + "=" * 70)
    print("QUANTITATIVE MULTI-BIOMARKER REGRESSION (PSA / AFP / CA125)")
    print("=" * 70)
    feature_extraction.main()
    regression_pipeline.main()


    