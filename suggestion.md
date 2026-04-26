# Cancer Biomarker AI Suite — Feature Suggestions

A prioritized backlog of enhancements extractable from the existing trained models, dataset, and architecture.

---

## 🔥 High Priority (High ROI, Low Effort)

### 1. Real SHAP Values

- **What:** Replace the current heuristic SHAP approximation with exact Shapley values computed via the `shap` library.
- **How:** Use `shap.TreeExplainer` for XGBoost and Random Forest, `shap.LinearExplainer` for Logistic Regression. Models are already loaded in memory.
- **Impact:** Transforms the SHAP Waterfall from a clinical demo into a scientifically valid explainability artifact. Major credibility upgrade.
- **Backend:** New `/shap` POST endpoint using real model + scaler pipeline.

### 2. Dynamic Threshold Slider

- **What:** Allow users to change the classification threshold (currently hardcoded at 0.5) via a slider in the UI.
- **How:** Pass the chosen threshold to the predict endpoint; recalculate Positive/Negative verdict and all derived metrics (Precision, Recall, etc.) on the fly.
- **Impact:** Shifts the app from a demo to a real clinical decision-support tool. Oncologists may want to lower threshold to 0.3 to maximize sensitivity (catch more true cases at the cost of more false alarms).
- **UI:** Threshold slider in the dashboard prediction panel, live-updating the risk verdict.

### 3. Real ROC / PR / Calibration Curves

- **What:** Replace the synthetic curve data in the ROC, PR, and Calibration tabs with curves computed from the actual validation split.
- **How:**
  - ROC: `sklearn.metrics.roc_curve`
  - PR: `sklearn.metrics.precision_recall_curve`
  - Calibration: `sklearn.calibration.calibration_curve`
- **Impact:** 3 charts become scientifically valid. The calibration curve in particular reveals if XGBoost is overconfident or underconfident in its probabilities.
- **Backend:** Extend the `/metrics` endpoint to compute and return real curve data from `X_val / y_val`.

---

## 🧬 From the Dataset

### 4. Dynamic Confusion Matrix

- **What:** Replace the hardcoded Confusion Matrix numbers with values computed dynamically from the validation set at the current threshold.
- **How:** `sklearn.metrics.confusion_matrix(y_val, y_pred)` after thresholding `predict_proba` output.
- **Impact:** CM becomes accurate, threshold-aware, and updates alongside the threshold slider.

### 5. Biomarker Correlation Matrix

- **What:** A heatmap showing the Pearson correlation between AFP, CA125, and PSA across the cohort.
- **How:** `df[feature_columns].corr()` on the raw dataset, returned as a matrix.
- **Impact:** Reveals multicollinearity and feature redundancy. Explains why some models agree or disagree.
- **UI:** New "Correlation Matrix" tab under Deep Discovery.

### 6. Real t-SNE / UMAP Projection

- **What:** Replace the synthetic t-SNE scatter points with a real dimensionality reduction of the patient cohort.
- **How:** `sklearn.manifold.TSNE` or `umap-learn` applied to `X_scaled` from the full dataset. Cache the result on server startup.
- **Impact:** Reveals whether the two classes (Positive / Negative) actually form separable clusters, directly explaining model performance limits.
- **Note:** UMAP is significantly faster than t-SNE and arguably more diagnostically meaningful.

### 7. Class Imbalance Statistics Panel

- **What:** A small panel showing cohort-level stats: total patients, class ratio (Positive vs Negative), and how imbalance affects model metrics.
- **How:** Compute from `y_true` distribution in the dataset.
- **Impact:** Explains why Recall is high but Precision is low across all models — the dataset is heavily imbalanced.

---

## 🏥 Clinical Intelligence

### 8. Risk Stratification Bands

- **What:** Replace binary Positive/Negative with 4-tier clinical risk: `Low / Borderline / High / Critical`.
- **How:** Map ensemble probability score to risk tiers with configurable PSA confidence intervals.
- **Impact:** Far more actionable for a clinician than a binary verdict.

### 9. Model Disagreement Detector

- **What:** When models disagree (e.g., XGBoost says Positive, LR says Negative), flag the patient as "High Uncertainty — Manual Review Required."
- **How:** Calculate variance across model probabilities. If variance > threshold, raise the flag.
- **Impact:** Surfaces the cases most in need of human clinical judgment. Essential for a clinical audit tool.

### 10. Threshold Sensitivity Report

- **What:** A table showing how Precision, Recall, F1, and the number of True/False Positives change as the threshold sweeps from 0.1 to 0.9.
- **How:** Loop over threshold values on the backend and compute metrics at each step.
- **Impact:** Gives clinicians a full picture of the performance trade-off space, not just a single operating point.

---

## 🚀 Architecture / Production

### 11. Model Versioning Panel

- **What:** When new `.pkl` artifacts are dropped in, show a version log: model name, file timestamp, last evaluation time.
- **How:** Read file `mtime` from the artifacts directory and surface it in the UI.
- **Impact:** Important for GxP/audit compliance. Traceability of which model version produced which prediction.

### 12. Export to PDF / CSV

- **What:** Allow the user to export the Audit Registry, Performance Summary, and patient-level predictions as a signed PDF or CSV report.
- **How:** Use `jsPDF` or Electron's `dialog` + `fs` for CSV; a print-to-PDF approach for the clinical report.
- **Impact:** Makes the app deployable in a clinical or research reporting workflow.

### 13. UMAP Toggle (vs. t-SNE)

- **What:** Add a toggle in the Latent Space tab to switch between t-SNE and UMAP projections.
- **How:** Run both on server startup (cached), expose via two separate endpoints or a `?method=umap` query param.
- **Impact:** UMAP is faster, more scalable, and often better for clinical cohort visualization.

---

## Priority Order Summary

| #   | Feature                      | Effort | Impact          |
| --- | ---------------------------- | ------ | --------------- |
| 1   | Real SHAP Values             | Low    | 🔴 Critical     |
| 2   | Dynamic Threshold Slider     | Medium | 🔴 Critical     |
| 3   | Real ROC / PR / Calibration  | Low    | 🔴 Critical     |
| 4   | Dynamic Confusion Matrix     | Low    | 🟠 High         |
| 8   | Risk Stratification Bands    | Medium | 🟠 High         |
| 9   | Model Disagreement Detector  | Low    | 🟠 High         |
| 5   | Biomarker Correlation Matrix | Low    | 🟡 Medium       |
| 6   | Real t-SNE / UMAP            | Medium | 🟡 Medium       |
| 7   | Class Imbalance Panel        | Low    | 🟡 Medium       |
| 10  | Threshold Sensitivity Report | Medium | 🟡 Medium       |
| 11  | Model Versioning Panel       | Low    | 🟢 Nice-to-have |
| 12  | Export to PDF / CSV          | High   | 🟢 Nice-to-have |
| 13  | UMAP Toggle                  | Medium | 🟢 Nice-to-have |
