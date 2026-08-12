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
  Shape: (1000, 204)

======================================================================
STEP: LOADING DPV VOLTAMMETRY FEATURES
======================================================================
  Found 200 DPV current measurements per sample
  DPV feature matrix: (1000, 201)

======================================================================
STEP 2: TARGET VARIABLE CREATION
======================================================================
  Definition: PSA_pg_per_ml > 4000 pg/mL indicates high risk
  Distribution:
    Low Risk (0): 881 (88.1%)
    High Risk (1): 119 (11.9%)
  Final feature matrix shape (before CV): (1000, 202)

======================================================================
STEP 5: STRATIFIED 5-FOLD CROSS-VALIDATION (localized scaling)
======================================================================

======================================================================
STEP 5: MODEL INITIALIZATION
======================================================================
  ✓ Logistic Regression
  ✓ Random Forest
  ✓ SVM
  ✓ XGBoost

Initializing 1D-CNN (sequence-aware)...
  ✅ 1D-CNN trained! Accuracy: 0.9500 | F1: 0.8148 | ROC-AUC: 0.9912

Initializing BiLSTM (sequence-aware)...
  ✅ BiLSTM trained! Accuracy: 0.9150 | F1: 0.6792 | ROC-AUC: 0.9128
  ✓ Fold 1/5 complete

======================================================================
STEP 5: MODEL INITIALIZATION
======================================================================
  ✓ Logistic Regression
  ✓ Random Forest
  ✓ SVM
  ✓ XGBoost

Initializing 1D-CNN (sequence-aware)...
  ✅ 1D-CNN trained! Accuracy: 0.9150 | F1: 0.7213 | ROC-AUC: 0.9747

Initializing BiLSTM (sequence-aware)...
  ✅ BiLSTM trained! Accuracy: 0.8100 | F1: 0.5250 | ROC-AUC: 0.9339
  ✓ Fold 2/5 complete

======================================================================
STEP 5: MODEL INITIALIZATION
======================================================================
  ✓ Logistic Regression
  ✓ Random Forest
  ✓ SVM
  ✓ XGBoost

Initializing 1D-CNN (sequence-aware)...
  ✅ 1D-CNN trained! Accuracy: 0.9200 | F1: 0.7333 | ROC-AUC: 0.9830

Initializing BiLSTM (sequence-aware)...
  ✅ BiLSTM trained! Accuracy: 0.9300 | F1: 0.7200 | ROC-AUC: 0.9375
  ✓ Fold 3/5 complete

======================================================================
STEP 5: MODEL INITIALIZATION
======================================================================
  ✓ Logistic Regression
  ✓ Random Forest
  ✓ SVM
  ✓ XGBoost

Initializing 1D-CNN (sequence-aware)...
  ✅ 1D-CNN trained! Accuracy: 0.8750 | F1: 0.6269 | ROC-AUC: 0.9621

Initializing BiLSTM (sequence-aware)...
  ✅ BiLSTM trained! Accuracy: 0.8700 | F1: 0.6286 | ROC-AUC: 0.9453
  ✓ Fold 4/5 complete

======================================================================
STEP 5: MODEL INITIALIZATION
======================================================================
  ✓ Logistic Regression
  ✓ Random Forest
  ✓ SVM
  ✓ XGBoost

Initializing 1D-CNN (sequence-aware)...
  ✅ 1D-CNN trained! Accuracy: 0.8450 | F1: 0.5974 | ROC-AUC: 0.9467

Initializing BiLSTM (sequence-aware)...
  ✅ BiLSTM trained! Accuracy: 0.9200 | F1: 0.7037 | ROC-AUC: 0.9515
  ✓ Fold 5/5 complete

======================================================================
STEP 6: AGGREGATED CROSS-VALIDATION METRICS
======================================================================

======================================================================
FOLD-WISE CROSS-VALIDATION METRICS (mean ± std over 5 folds)
======================================================================
Model                       Accuracy     Precision        Recall      F1-Score       ROC-AUC
--------------------------------------------------------------------------------------------
Logistic Regression      0.978±0.007   0.875±0.064   0.958±0.037   0.913±0.026   0.997±0.002
Random Forest            0.969±0.006   0.926±0.045   0.807±0.034   0.861±0.025   0.992±0.003
SVM                      0.973±0.009   0.847±0.081   0.958±0.037   0.896±0.033   0.997±0.003
XGBoost                  0.978±0.007   0.902±0.041   0.916±0.039   0.908±0.033   0.996±0.002
1D-CNN                   0.901±0.037   0.568±0.097   0.925±0.031   0.699±0.078   0.972±0.016
BiLSTM                   0.889±0.045   0.556±0.114   0.823±0.062   0.651±0.070   0.936±0.013


======================================================================
BIOPHYSICAL PEAK MAPPING (DPV vs biomarker concentration)
======================================================================
Biomarker    Peak (mV)   Measured (mV)    r (log-conc)
------------------------------------------------------
PSA               -468            -468         -0.8624
AFP                365             365         -0.8577
CA125              968             968         -0.8367


======================================================================
CONFUSION MATRIX COUNTS (out-of-fold predictions)
======================================================================
Model                     TN    FP    FN    TP
----------------------------------------------
Logistic Regression      864    17     5   114
Random Forest            873     8    23    96
SVM                      859    22     5   114
XGBoost                  869    12    10   109
1D-CNN                   791    90     9   110
BiLSTM                   791    90    21    98

  ✓ Saved: performance_summary.json (OOF metrics for server)

======================================================================
STEP 7: TRAINING FINAL DEPLOYMENT MODELS (full data)
======================================================================

======================================================================
STEP 5: MODEL INITIALIZATION
======================================================================
  ✓ Logistic Regression
  ✓ Random Forest
  ✓ SVM
  ✓ XGBoost
  ✓ Saved: logistic_regression_model.pkl
  ✓ Saved: random_forest_model.pkl
  ✓ Saved: svm_model.pkl
  ✓ Saved: xgboost_model.pkl

Initializing 1D-CNN (sequence-aware)...
  ✅ 1D-CNN trained! Accuracy: 1.0000 | F1: 1.0000 | ROC-AUC: 1.0000
  ✓ Saved: 1d-cnn_model.pkl

Initializing BiLSTM (sequence-aware)...
  ✅ BiLSTM trained! Accuracy: 0.8980 | F1: 0.6600 | ROC-AUC: 0.9466
  ✓ Saved: bilstm_model.pkl

  ✓ Saved: scaler.pkl and feature_columns.pkl

======================================================================
PATIENT RISK PREDICTION (ensemble, matches Electron /predict)
======================================================================
  Logistic Regression       0.0%
  Random Forest             0.0%
  SVM                       0.0%
  XGBoost                   0.3%
  1D-CNN                    0.3%
  BiLSTM                   39.5%
----------------------------------------
  Risk Score:        0.0669
  Prediction:        Negative
  Model Agreement:   93.3%


======================================================================
STEP 8: GENERATING ADVANCED VISUALIZATIONS
======================================================================

 Creating advanced confusion matrices...
  ✓ Saved: advanced_cm_logistic_regression.png
  ✓ Saved: advanced_cm_random_forest.png
  ✓ Saved: advanced_cm_svm.png
  ✓ Saved: advanced_cm_xgboost.png
  ✓ Saved: advanced_cm_1d-cnn.png
  ✓ Saved: advanced_cm_bilstm.png

 Creating radar chart...
  ✓ Saved: radar_chart_comparison.png

 Creating parallel coordinates plot...
  ✓ Saved: parallel_coordinates.png

 Creating ridge plot...
  ✓ Saved: ridge_plot_distributions.png

 Creating violin plots...
  ✓ Saved: violin_plots.png

 Creating correlation heatmap...
  ✓ Saved: correlation_heatmap.png

 Creating model performance summary chart...
  ✓ Saved: model_performance_summary.png

 Creating best model modal...
  ✓ Saved: best_model_modal.png

 Creating enhanced ROC curves...
  ✓ Saved: roc_curves_enhanced.png

 Creating Precision-Recall curves...
  ✓ Saved: pr_curves.png

 Creating SHAP explanation plot...
  ⚠️ GradientExplainer failed (Input type (torch.FloatTensor) and weight type (torch.cuda.FloatTensor) should be the same or input should be a MKLDNN tensor and weight is a dense tensor) — falling back to XGBoost
  ✓ Saved: shap_importance.png

 Creating DPV fingerprint comparison plot...
  ✓ Saved: Figure1_DPV_Fingerprints.png

 Creating feature importance map...
  ✓ Saved: Figure3_Feature_Importance_Map.png

======================================================================
PIPELINE COMPLETED SUCCESSFULLY
======================================================================

 BEST MODEL: Logistic Regression
   F1-Score: 0.9120
   Accuracy: 0.9780
   ROC-AUC: 0.9975
   PR-AUC: 0.9817

======================================================================
BEST MODEL SUMMARY
======================================================================
Best model: Logistic Regression
  • Accuracy: 97.80%
  • Precision: 87.02%
  • Recall: 95.80%
  • F1-Score: 91.20%
  • ROC-AUC: 99.75%
  • PR-AUC: 98.17%

Model performance table (percentages):
              Model  Accuracy (%)  Precision (%)  Recall (%)  F1-Score (%)  ROC-AUC (%)  PR-AUC (%)
Logistic Regression         97.80          87.02       95.80         91.20        99.75       98.17
      Random Forest         96.90          92.31       80.67         86.10        99.22       94.34
                SVM         97.30          83.82       95.80         89.41        99.63       97.37
            XGBoost         97.80          90.08       91.60         90.83        99.61       97.63
             1D-CNN         90.10          55.00       92.44         68.97        97.30       83.03
             BiLSTM         88.90          52.13       82.35         63.84        92.86       75.24

 Output files saved to:
   • ../models/ - Trained models
   • ../figure/ - All visualizations

======================================================================
MODEL PERFORMANCE SUMMARY (5-Fold CV, out-of-fold)
======================================================================
                     Accuracy  Precision  Recall  F1-Score  ROC-AUC  PR-AUC
Logistic Regression     0.978     0.8702  0.9580    0.9120   0.9975  0.9817
Random Forest           0.969     0.9231  0.8067    0.8610   0.9922  0.9434
SVM                     0.973     0.8382  0.9580    0.8941   0.9963  0.9737
XGBoost                 0.978     0.9008  0.9160    0.9083   0.9961  0.9763
1D-CNN                  0.901     0.5500  0.9244    0.6897   0.9730  0.8303
BiLSTM                  0.889     0.5213  0.8235    0.6384   0.9286  0.7524

======================================================================
RISK STRATIFICATION (Logistic Regression)
======================================================================
  Safe (<45%):        866
  Moderate (45-60%):  9
  High (60-75%):      11
  Critical (>75%):    114
  Total:              1000


======================================================================
OPTIMAL DECISION THRESHOLD (Logistic Regression)
======================================================================
  Optimal threshold:  0.45
  F1 at threshold:    0.9249
  Precision:          0.8731
  Recall:             0.9832


======================================================================
TOP-10 DISCRIMINATIVE VOLTAGE STEPS (feature importance)
======================================================================
  Source model: Logistic Regression
  #  Voltage (mV)    Importance
-------------------------------
  1          -498        1.4471
  2          -448        1.4019
  3          -458        1.3268
  4          -468        1.3104
  5          -428        1.2843
  6          -438        1.2718
  7          -478        1.2671
  8          -418        0.9682
  9          -488        0.9081
 10          -508        0.6492


======================================================================
ENSEMBLE CONSENSUS AGREEMENT
======================================================================
  Models compared:     6
  All models agree:    809 (80.9%)
  Split decision:      191 (19.1%)


======================================================================
QUANTITATIVE MULTI-BIOMARKER REGRESSION (PSA / AFP / CA125)
======================================================================

======================================================================
ENGINEERED FEATURE AUDIT
======================================================================
  peak_anodic_current        OK
  peak_anodic_potential      OK
  peak_cathodic_current      OK
  peak_cathodic_potential    OK
  area_under_curve           OK
  peak_separation            OK

✓ Saved engineered features to A:\Master class\XAI - Concer recogination\biomarkers\server\analysis\src\..\data\data_with_features_engineered.csv
  Shape: (1000, 10)

======================================================================
DATA AUDIT REPORT
======================================================================
  Samples:                1000
  Unique sample IDs:      1000
  Duplicate sample IDs:   0
  DPV measurements/sample:200
  Missing values:         0
  Potential range (mV):   -750 to 1250
  Potential step (mV):    11.0

  Concentration ranges:
    PSA_pg_per_ml    min=0.00124  max=9.927e+04  log10 range=[-2.9,5.0]
    AFP_pg_per_ml    min=0.0004894  max=1.724e+04  log10 range=[-3.3,4.2]
    CA125_U_per_ml   min=0.02825  max=203.9  log10 range=[-1.5,2.3]

======================================================================
BIOMARKER-SPECIFIC CORRELATION (DPV vs log-concentration)
======================================================================
  PSA_pg_per_ml    strongest |r| = -0.8162 at -458 mV (Spearman -0.8193)
  AFP_pg_per_ml    strongest |r| = -0.7984 at 375 mV (Spearman -0.8078)
  CA125_U_per_ml   strongest |r| = -0.8024 at 968 mV (Spearman -0.7967)

======================================================================
P4 - SINGLE-BIOMARKER REGRESSION (log10 concentration)
======================================================================
  PSA_pg_per_ml    Linear       R2=0.8453 MAE=0.6531 RMSE=0.8385 r=0.9205 rho=0.9395
  PSA_pg_per_ml    PLS          R2=0.8453 MAE=0.6532 RMSE=0.8386 r=0.9205 rho=0.9395
  PSA_pg_per_ml    RandomForest R2=0.8745 MAE=0.5606 RMSE=0.7553 r=0.9361 rho=0.9461
  PSA_pg_per_ml    SVR          R2=0.8623 MAE=0.6049 RMSE=0.7911 r=0.9288 rho=0.9442
  PSA_pg_per_ml    XGBoost      R2=0.8812 MAE=0.5263 RMSE=0.7347 r=0.9389 rho=0.9455
  PSA_pg_per_ml    MLP          R2=0.8368 MAE=0.6476 RMSE=0.8611 r=0.9155 rho=0.9243
  PSA_pg_per_ml    1D-CNN       R2=0.3256 MAE=1.4280 RMSE=1.7506 r=0.5714 rho=0.5438
  AFP_pg_per_ml    Linear       R2=0.7876 MAE=0.6097 RMSE=0.7931 r=0.8895 rho=0.9171
  AFP_pg_per_ml    PLS          R2=0.7875 MAE=0.6098 RMSE=0.7933 r=0.8895 rho=0.9170
  AFP_pg_per_ml    RandomForest R2=0.8299 MAE=0.5037 RMSE=0.7097 r=0.9128 rho=0.9317
  AFP_pg_per_ml    SVR          R2=0.8099 MAE=0.5546 RMSE=0.7503 r=0.9002 rho=0.9221
  AFP_pg_per_ml    XGBoost      R2=0.8373 MAE=0.4882 RMSE=0.6942 r=0.9154 rho=0.9330
  AFP_pg_per_ml    MLP          R2=0.7784 MAE=0.5975 RMSE=0.8102 r=0.8842 rho=0.9039
  AFP_pg_per_ml    1D-CNN       R2=0.2036 MAE=1.2443 RMSE=1.5358 r=0.4533 rho=0.4469
  CA125_U_per_ml   Linear       R2=0.8517 MAE=0.2258 RMSE=0.3004 r=0.9237 rho=0.9479
  CA125_U_per_ml   PLS          R2=0.8517 MAE=0.2258 RMSE=0.3004 r=0.9237 rho=0.9479
  CA125_U_per_ml   RandomForest R2=0.8530 MAE=0.2184 RMSE=0.2991 r=0.9281 rho=0.9436
  CA125_U_per_ml   SVR          R2=0.8681 MAE=0.2079 RMSE=0.2833 r=0.9320 rho=0.9546
  CA125_U_per_ml   XGBoost      R2=0.8720 MAE=0.2037 RMSE=0.2791 r=0.9348 rho=0.9488
  CA125_U_per_ml   MLP          R2=0.7914 MAE=0.2764 RMSE=0.3564 r=0.8945 rho=0.9082
  CA125_U_per_ml   1D-CNN       R2=0.3542 MAE=0.4958 RMSE=0.6269 r=0.5984 rho=0.5737

======================================================================
P5 - MULTI-OUTPUT REGRESSION (single fingerprint -> PSA+AFP+CA125)
======================================================================
  Linear       PSA R2=0.8453 AFP R2=0.7876 CA125 R2=0.8517
  PLS          PSA R2=0.8453 AFP R2=0.7875 CA125 R2=0.8517
  RandomForest PSA R2=0.8745 AFP R2=0.8299 CA125 R2=0.8530
  SVR          PSA R2=0.8623 AFP R2=0.8099 CA125 R2=0.8681
  XGBoost      PSA R2=0.8812 AFP R2=0.8373 CA125 R2=0.8720
  MLP          PSA R2=0.8368 AFP R2=0.7784 CA125 R2=0.7914
  1D-CNN       PSA R2=0.3063 AFP R2=0.2140 CA125 R2=0.1999

======================================================================
P6 - RAW DPV vs ENGINEERED FEATURES vs BIOMARKER REGIONS
======================================================================
  full_dpv_200 PSA_pg_per_ml    R2=0.8453 r=0.9205
  full_dpv_200 AFP_pg_per_ml    R2=0.7876 r=0.8895
  full_dpv_200 CA125_U_per_ml   R2=0.8517 r=0.9237
  engineered_6 PSA_pg_per_ml    R2=0.8331 r=0.9128
  engineered_6 AFP_pg_per_ml    R2=0.3359 r=0.5797
  engineered_6 CA125_U_per_ml   R2=0.0040 r=0.0879
  regions_only PSA_pg_per_ml    R2=0.6963 r=0.8344
  regions_only AFP_pg_per_ml    R2=0.6910 r=0.8313
  regions_only CA125_U_per_ml   R2=0.7427 r=0.8618

======================================================================
P7 - ABLATION STUDY
======================================================================
  PSA_region_only  avg R2=0.2175
  AFP_region_only  avg R2=0.2168
  CA125_region_only avg R2=0.2354
  PSA+AFP          avg R2=0.4506
  PSA+CA125        avg R2=0.4664
  AFP+CA125        avg R2=0.4705
  All_three_regions avg R2=0.7100
  Full_200_point   avg R2=0.8282

======================================================================
P8 - CROSS-BIOMARKER INTERFERENCE
======================================================================
  PSA    with AFP    region: R2=0.6892 r=0.8302
  PSA    with CA125  region: R2=0.6829 r=0.8264
  AFP    with PSA    region: R2=0.6775 r=0.8231
  AFP    with CA125  region: R2=0.6806 r=0.8250
  CA125  with PSA    region: R2=0.7232 r=0.8504
  CA125  with AFP    region: R2=0.7361 r=0.8580

======================================================================
P9 - STRICT SPLIT + BOOTSTRAP UNCERTAINTY
======================================================================
  PSA_pg_per_ml    TEST R2=0.8877 (95% CI [0.8552,0.9132]) MAE=0.5371 RMSE=0.7476 r=0.9436
  AFP_pg_per_ml    TEST R2=0.8530 (95% CI [0.8050,0.8920]) MAE=0.4547 RMSE=0.6515 r=0.9237
  CA125_U_per_ml   TEST R2=0.8704 (95% CI [0.8376,0.9012]) MAE=0.2144 RMSE=0.2900 r=0.9335

======================================================================
P10 - SHAP PER BIOMARKER
======================================================================
  PSA_pg_per_ml    top importance voltages: [-458.0, -468.0, -448.0, -478.0, -659.0]
  AFP_pg_per_ml    top importance voltages: [365.0, 375.0, 385.0, 395.0, 355.0]
  CA125_U_per_ml   top importance voltages: [968.0, 978.0, 988.0, 958.0, 998.0]
  ✓ Saved: P15_pred_meas_PSA.png
  ✓ Saved: P15_pred_meas_AFP.png
  ✓ Saved: P15_pred_meas_CA125.png
  ✓ Saved: P20A_multi_output.png
  ✓ Saved: P20B_representations.png
  ✓ Saved: P20C_ablation.png
  ✓ Saved: P20D_shap_vs_potential.png

======================================================================
REGRESSION PIPELINE COMPLETE
======================================================================
Saved: P3/P4/P5/P6/P7/P8/P9 CSV reports + P2/P15/P20 figures
