# Deep Dive Documentation: Prostate Cancer Biomarker Analysis (`analysis.py`)

This document provides a highly technical, deep-dive examination of the `analysis.py` script. It details the underlying mathematical concepts, defensive programming mechanisms, model hyperparameters, graph neural network topology, and the comprehensive visualization subsystem used for clinical risk stratification of prostate cancer.

---

## 1. Dataset Processing & Integrity Mechanisms

### Target Definition & Feature Engineering
- **Target Extraction:** The clinical cutoff for high-risk prostate cancer is strictly defined as `PSA_pg_per_ml > 4000 pg/mL` (equivalent to 4 ng/mL). The boolean result is cast to an integer to create a binary classification target (`high_risk`).
- **Data Leakage Prevention:** The feature matrix explicitly drops `PSA_pg_per_ml` to ensure the model does not train on the deterministic variable used to create the target. The retained feature set strictly consists of independent biomarkers: `AFP_pg_per_ml` and `CA125_U_per_ml`.
- **Log Transformation Math:** Because biomarker concentrations often span multiple orders of magnitude, a natural logarithm transformation is applied using `np.log1p(x)`. This computes $log_e(1 + x)$, which inherently handles zero-concentration values without approaching negative infinity.
- **Robust Scaling:** Standard normalization (Z-score) is highly sensitive to extreme clinical outliers. The script employs `sklearn.preprocessing.RobustScaler`, which removes the median and scales the data according to the Interquartile Range (IQR, 25th to 75th percentile).

### Defensive Programming (Data Alignment)
The script includes specialized blocks (e.g., `ALIGNING DATA FOR PLOTTING`) designed to seamlessly transition between PyTorch Tensors, NumPy arrays, and Pandas DataFrames.
- It proactively intercepts edge cases where tensor-to-numpy conversions disrupt index alignment. 
- A fallback recovery mechanism is implemented: if a length mismatch is detected between `X_scaled` and `y`, it explicitly truncates or padding-restores the sequence by fully recreating `y` directly from the original `df_clean` dataframe.

---

## 2. Advanced Model Architectures & Hyperparameters

The pipeline deploys a heterogenous ensemble of models to ensure robustness across different feature topologies. Class imbalance is handled automatically by calculating array weights using `compute_class_weight("balanced")`.

### 2.1 Classical & Ensemble Models
- **Logistic Regression:** Serves as the linear baseline. Initialized with `max_iter=1000` to guarantee convergence on scaled log-data, `C=1.0` (L2 regularization), and balanced class weights.
- **Random Forest:** Configured with `n_estimators=100`, bounded with `max_depth=10` to prevent overfitting on the minor feature space, and `min_samples_split=5`.
- **Support Vector Machine (SVM):** Utilizes a Radial Basis Function (`rbf`) kernel to map non-linear relationships. `probability=True` is enforced to allow Platt scaling for risk probability outputs. `gamma="scale"` calculates the kernel coefficient dynamically based on the number of features and variance.
- **XGBoost:** 
  - `scale_pos_weight` is explicitly calculated mathematically as `(Negative Samples / Positive Samples)` rather than relying on standard balancing.
  - Constrained with `max_depth=6` and a modest `learning_rate=0.1`.
  - Optimized specifically for probabilistic accuracy using `eval_metric="logloss"`.

---

## 3. GNN (Graph Neural Network) Technical Internals

The script implements a cutting-edge representation learning paradigm using PyTorch Geometric (PyG), mapping patient biomarker profiles as graph structures.

### Graph Topology Construction
- **Nodes as Features:** The graph nodes do not represent patients; rather, they represent individual *biomarkers*.
- **Edges as Biological Correlation:** A Pearson correlation matrix is computed from the scaled training data. Bidirectional edges `[i, j]` and `[j, i]` are created strictly between biomarkers exhibiting a strong correlation ($|r| > 0.3$). If no edges meet the threshold, a minimum connectivity fallback `[[0, 1], [1, 0]]` is enforced to prevent unroutable tensors.

### Node Feature Mapping & Forward Pass
- **Feature Broadcasting:** For any given patient, the input tensor `x` expands the patient's entire feature vector to an $(N, N)$ matrix where $N$ is the number of features. This allows every node (biomarker) to contextually observe the entire patient profile during message passing.
- **Architecture:** 
  1. `GCNConv(num_features, 32)` $\to$ ReLU activation
  2. `GCNConv(32, 32)` $\to$ ReLU activation
  3. `global_mean_pool` $\to$ Aggregates the graph embedding down to a single vector representing the patient.
  4. `Linear(32, 2)` $\to$ Outputs unnormalized logits for the binary classes.
- **Optimization:** Trained over 50 epochs using the `Adam` optimizer with a learning rate of $0.01$ and a `CrossEntropyLoss` criterion in mini-batches of 16.

### Application Integration Wrapper
Because raw PyTorch models require their class definitions to be imported at load-time, the script implements a dynamic path appending technique (`sys.path.append('../../')`). It imports the application's native `GNNClassifier` from `logic.model_manager`, injects the trained `gnn_model` weights and `feature_names` into it, and triggers its internal `_build_graph` before pickling. This guarantees seamless deserialization in the Electron backend.

---

## 4. Extensive Evaluation & Risk Stratification

Predictions are made at a base $0.5$ threshold, but the pipeline evaluates the raw continuous probabilities to construct clinical cohorts.

- **Threshold Optimization Loop:** The script programmatically sweeps through 100 threshold thresholds between $0.0$ and $1.0$. It calculates the precision, recall, and F1-score at every bin to determine the mathematical optimal cutoff point (visualized by a purple dashed line on the UI).
- **Stratification Zones:** 
  - **Critical Radius:** $> 75\%$ probability. High urgency intervention.
  - **Urgent Zone:** $45\% - 75\%$ probability.
  - **Stable Cohort:** $< 45\%$ probability. Benign profiles.
- **Validation Fallback:** A safety check is implemented to emit a `CRITICAL WARNING` to standard output if the highest-performing model's accuracy degrades below 50%.

---

## 5. Visualization Subsystem

A comprehensive suite of matplotlib/seaborn plots are generated for deep-dive exploratory data analysis (EDA) and model diagnostics.

### Dimensionality Reduction
- **t-SNE & PCA:** The high-dimensional feature space is reduced to 2 dimensions. 
  - PCA runs first, dynamically capping its components via $min(2, num\_features)$.
  - t-SNE projects the data utilizing a perplexity of 30 over 1000 iterations.
  - Four subplots map these embeddings against True Labels, Best Model Predictions, and Continuous Prediction Probabilities.

### Advanced Diagnostic Charts
- **Polar (Radar) Spider Chart:** Maps Accuracy, Precision, Recall, F1, and AUC simultaneously across radial axes, filled with an alpha=0.15 overlay. 
- **Cumulative Gains Chart:** Evaluates the models against a random baseline. It sorts patients by predicted risk in descending order and plots the percentage of true positives captured against the percentage of the patient pool evaluated.
- **Calibration Curve:** Utilizes `calibration_curve` across 10 bins to measure the reliability of probabilistic outputs. A perfectly calibrated model should fall exactly on the diagonal `y=x` line, meaning when the model predicts $60\%$ risk, $60\%$ of those patients are actually positive.

---

## 6. Artifact Pipeline & Persistence

The script executes a controlled tear-down, ensuring data persistence for downstream Electron application processes:

- **Pickled Weights:** Models are serialized strictly via `pickle.dump` inside the `../models/` directory.
- **CSVs & Aggregations:** 
  - `model_comparison_all_models.csv` stores all floating-point evaluation metrics.
  - `validation_predictions.csv` dumps a granular, patient-by-patient log of true labels juxtaposed against individual model raw probabilities and boolean predictions.
  - `analysis_summary.csv` captures metadata, class imbalance percentages, overall accuracy stats, and the mean/standard-deviation resulting from the rigorous `StratifiedKFold` 5-fold Cross Validation.
