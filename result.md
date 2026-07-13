python analyse_data.py 

======================================================================
PROSTATE CANCER RISK CLASSIFICATION PIPELINE
======================================================================

This pipeline uses DPV voltammetry features (200-point
current measurements) to predict prostate cancer risk
based on PSA clinical cutoff (>4 ng/mL).

======================================================================
STEP 1: LOADING DATA
======================================================================
✓ Data loaded successfully
  Shape: (1000, 4)
  Columns: ['sample_id', 'PSA_pg_per_ml', 'AFP_pg_per_ml', 'CA125_U_per_ml']

======================================================================
STEP: LOADING DPV VOLTAMMETRY FEATURES
======================================================================
  Found 1000 sample sheets with DPV data
  DPV feature matrix: (1000, 201) (200 current measurements per sample)
  Merged dataset: (1000, 204)

======================================================================
STEP 2: TARGET VARIABLE CREATION
======================================================================
  Definition: PSA_pg_per_ml > 4000 pg/mL indicates high risk
  Distribution:
    Low Risk (0): 881 (88.1%)
    High Risk (1): 119 (11.9%)

======================================================================
STEP 3: FEATURE PREPROCESSING
======================================================================
  Samples after removing missing values: 1000

  Skipping log transform — features contain negative values

  Scaling: RobustScaler applied
  Final feature matrix shape: (1000, 200)

======================================================================
STEP 4: DATA SPLITTING
======================================================================
  Training set: 700 samples (70.0%)
  Validation set: 300 samples (30.0%)

  Training distribution:
    Low Risk: 617 (88.1%)
    High Risk: 83 (11.9%)

======================================================================
STEP 5: MODEL INITIALIZATION
======================================================================
  ✓ Logistic Regression
  ✓ Random Forest
  ✓ SVM
  ✓ XGBoost

======================================================================
STEP 6: MODEL TRAINING
======================================================================

  Training Logistic Regression...
    ✓ Accuracy: 0.9367
    ✓ F1-Score: 0.7654

  Training Random Forest...
    ✓ Accuracy: 0.9500
    ✓ F1-Score: 0.7619

  Training SVM...
    ✓ Accuracy: 0.9533
    ✓ F1-Score: 0.7879

  Training XGBoost...
    ✓ Accuracy: 0.9467
    ✓ F1-Score: 0.7895

======================================================================
STEP 7b: GRAPH NEURAL NETWORK TRAINING
======================================================================

Initializing Graph Neural Network (GNN)...
Building correlation graph from biomarkers...
Creating GNN training data...

Training Graph Neural Network...
Epoch  10 | Train Loss: 0.0082 | Val Loss: 5.4534
Epoch  20 | Train Loss: 0.0015 | Val Loss: 6.1792
Epoch  30 | Train Loss: 0.0006 | Val Loss: 6.6227
Epoch  40 | Train Loss: 0.0003 | Val Loss: 6.9491
Epoch  50 | Train Loss: 0.0002 | Val Loss: 7.2119

Evaluating GNN on validation set...

✅ GNN Model trained successfully!
  Validation Accuracy: 0.9367
  Validation F1-Score: 0.7595
  Validation ROC-AUC: 0.9663
  ✓ Saved: gnn_model.pkl

  ✓ Saved: scaler.pkl and feature_columns.pkl

======================================================================
STEP 8: GENERATING ADVANCED VISUALIZATIONS
======================================================================

  📊 Creating advanced confusion matrices...
  ✓ Saved: advanced_cm_logistic_regression.png
  ✓ Saved: advanced_cm_random_forest.png
  ✓ Saved: advanced_cm_svm.png
  ✓ Saved: advanced_cm_xgboost.png
  ✓ Saved: advanced_cm_gnn.png

  📈 Creating radar chart...
  ✓ Saved: radar_chart_comparison.png

  🔀 Creating parallel coordinates plot...
  ✓ Saved: parallel_coordinates.png

  ⛰️ Creating ridge plot...
  ✓ Saved: ridge_plot_distributions.png

  🎻 Creating violin plots...
  ✓ Saved: violin_plots.png

  🔥 Creating correlation heatmap...
  ✓ Saved: correlation_heatmap.png

  📊 Creating model performance summary chart...
  ✓ Saved: model_performance_summary.png

  🏅 Creating best model modal...
  ✓ Saved: best_model_modal.png

  📉 Creating enhanced ROC curves...
  ✓ Saved: roc_curves_enhanced.png

======================================================================
PIPELINE COMPLETED SUCCESSFULLY
======================================================================

🏆 BEST MODEL: XGBoost
   F1-Score: 0.7895
   Accuracy: 0.9467
   ROC-AUC: 0.9871

======================================================================
BEST MODEL SUMMARY
======================================================================
Best model: XGBoost
  • Accuracy: 94.67%
  • Precision: 75.00%
  • Recall: 83.33%
  • F1-Score: 78.95%
  • ROC-AUC: 98.71%

Model performance table (percentages):
              Model  Accuracy (%)  Precision (%)  Recall (%)  F1-Score (%)  ROC-AUC (%)
Logistic Regression         93.67          68.89       86.11         76.54        98.47
      Random Forest         95.00          88.89       66.67         76.19        98.61
                SVM         95.33          86.67       72.22         78.79        98.33
            XGBoost         94.67          75.00       83.33         78.95        98.71
                GNN         93.67          69.77       83.33         75.95        96.63
  ✓ Saved: best_model_summary.csv

📁 Output files saved to:
   • ../models/ - Trained models
   • ../figure/ - All visualizations

======================================================================
MODEL PERFORMANCE SUMMARY
======================================================================
                     Accuracy  Precision  Recall  F1-Score  ROC-AUC
Logistic Regression    0.9367     0.6889  0.8611    0.7654   0.9847
Random Forest          0.9500     0.8889  0.6667    0.7619   0.9861
SVM                    0.9533     0.8667  0.7222    0.7879   0.9833
XGBoost                0.9467     0.7500  0.8333    0.7895   0.9871
GNN                    0.9367     0.6977  0.8333    0.7595   0.9663
@Saboor ➜ src git(features) 

-----
