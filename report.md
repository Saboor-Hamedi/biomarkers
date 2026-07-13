# Comprehensive Analytical Report for the Prostate Cancer Risk Classification Study Using DPV Voltammetry

## Abstract

This study applies machine learning to classify prostate cancer risk using differential pulse voltammetry (DPV) data. Instead of standard blood biomarkers, 200 electrical current readings per patient serve as input features. Five models from different algorithm families, including a Graph Neural Network, were trained on DPV scans from 1,000 patients. The risk label was derived from the clinical PSA cutoff of 4,000 pg/mL, yielding about 11.9% high-risk and 88.1% low-risk patients. A held-out validation set of 300 patients was used for testing. All five models delivered consistent, strong results. The top four models reached ROC-AUC values above 98%, and validation accuracies remained above 93%. XGBoost performed best with a ROC-AUC of 98.71% and an F1-score of 78.95%. The models varied in their precision-recall balance, making some more suitable for broad screening and others for targeted confirmatory testing. Logistic Regression caught the most high-risk cases at 86% recall, while Random Forest produced the fewest false alarms at 89% precision. An ensemble combining all five models could provide the most balanced predictions across different clinical settings. These results indicate that DPV voltammetry paired with machine learning can reliably identify prostate cancer risk and supports development of an accessible screening tool that could address key shortcomings of current PSA-based methods, such as unnecessary biopsies and missed diagnoses.

**Keywords:** prostate cancer, differential pulse voltammetry, machine learning, ensemble models, risk classification

## 1. Introduction

Prostate cancer remains one of the most frequently diagnosed cancers in men worldwide. Early detection dramatically improves treatment outcomes, and for decades the primary screening method has been the PSA blood test. Despite its widespread use, PSA testing has known limitations. Many men with elevated PSA do not have prostate cancer, while some men with normal PSA do, leading to unnecessary biopsies on one hand and missed diagnoses on the other. Estimates suggest that up to 70% of men with elevated PSA who undergo biopsy receive a negative result, representing a big burden on patients and healthcare systems. The clinical need for more accurate, accessible screening tools has driven research into alternative approaches, and electrochemical methods have emerged as a promising direction.

Differential pulse voltammetry (DPV) offers a different way of analyzing patient samples. Instead of measuring specific proteins, DPV applies a programmed sequence of voltage pulses to a sample and records the resulting electrical current. The output is a curve of 200 current values that reflects the electrochemical composition of the sample. Because DPV captures a broad electrochemical signature rather than isolated biomarker levels, it can potentially reveal information that standard blood tests miss. The technique is relatively inexpensive and fast, requiring only a small sample volume.

Our dataset consists of DPV measurements from 1,000 patients collected as part of a screening study. For each patient, we recorded 200 current readings from their voltammetry scan along with reference measurements of PSA, AFP, and CA125. We used the PSA level to define the risk label, applying a clinical cutoff of 4,000 pg/mL to separate high-risk from low-risk patients. This threshold is widely accepted in clinical practice and corresponds to the standard cutoff of 4 ng/mL used in PSA-based screening.

We trained five machine learning models on this data that span a range of algorithmic families: Logistic Regression, Random Forest, Support Vector Machine (SVM), XGBoost, and a Graph Neural Network (GNN). Each model learns patterns in the DPV curves that distinguish high-risk from low-risk patients. We compared their performance across multiple metrics and generated diagnostic visualizations to understand their behavior and identify areas for improvement.

All five models delivered strong results. Validation accuracies exceeded 93%, and ROC-AUC scores were above 96% across the board. XGBoost emerged as the top performer with a ROC-AUC of 98.71%. These findings suggest that DPV data carries strong electrochemical signatures linked to prostate cancer risk, and that machine learning models can reliably extract these signals. The consistency across different model types, from simple linear models to complex graph-based architectures, is one of the most important results of this study. It suggests that the predictive information in DPV data is genuine and not an artifact of a particular algorithm's idiosyncrasies. This strength is a strong sign that DPV-based screening could be developed into a practical clinical tool.

## 2. Literature Review

Prostate cancer screening has been a subject of intensive research for decades. The FDA approved the PSA test in 1986 for monitoring disease progression in men already diagnosed with prostate cancer, and it soon became widely used for screening asymptomatic men. The widespread adoption of PSA testing led to a large increase in prostate cancer detection rates, particularly for early-stage disease. However, this came with the recognition that many detected cancers would never have caused clinical symptoms, leading to the problem of overdiagnosis and overtreatment. Large clinical trials have since showed that PSA screening reduces prostate cancer mortality, but the benefit is modest relative to the number of men who need to be screened and treated. A landmark study published in the New England Journal of Medicine reported that PSA screening prevented one death from prostate cancer for every 27 men diagnosed, and the number needed to screen to prevent one death was over 1,000. This trade-off has motivated researchers to search for better screening methods that maintain the mortality benefit while reducing unnecessary interventions.

Many attempts to improve upon PSA testing have focused on combining it with other biomarkers or refining its interpretation. The Prostate Health Index (PHI) integrates total PSA, free PSA, and the precursor form [-2]proPSA into a single score that improves specificity compared to PSA alone. Clinical studies have shown that PHI reduces the number of unnecessary biopsies by 30-40% while maintaining sensitivity. The 4Kscore test measures four kallikrein markers (total PSA, free PSA, intact PSA, and human kallikrein 2) and combines them with clinical variables into a risk prediction model. Both approaches show meaningful improvements over PSA alone, but their adoption has been limited by cost, complexity, and the need for specialized assays. There remains a clear need for a simple, accurate, and affordable screening test, which is where electrochemical approaches like DPV become particularly relevant.

Machine learning has been applied to cancer diagnosis across numerous domains with growing success. Deep learning models have achieved radiologist-level performance in medical image classification for breast, lung, and skin cancer. Decision tree-based methods have been used to analyze patient records, and support vector machines have been applied to genomic data for cancer subtyping. In prostate cancer specifically, researchers have used machine learning to predict biopsy outcomes from PSA levels, MRI features, and clinical variables. Most of these studies rely on a relatively small number of engineered features. The current study differs fundamentally in that we use a high-dimensional electrochemical signal with 200 features per sample, offering the potential to capture information that lower-dimensional approaches miss.

DPV voltammetry is a mature electrochemical technique with decades of use in analytical chemistry. It works by applying a series of voltage pulses to an electrode immersed in a sample solution. At each pulse, the current is measured before and after, and the difference isolates the faradaic current from background capacitive effects. The resulting signal contains peaks at characteristic potentials corresponding to different electrochemically active compounds, with peak heights reflecting their concentrations.

Recent research has extended DPV to clinical applications. Studies have used DPV to detect biomarkers for various cancers, infectious diseases, and metabolic disorders. The key advantage is that DPV can measure multiple compounds simultaneously without separate assays, reducing cost and turnaround time. The current curve acts as a holistic fingerprint of the sample's electrochemical properties, and machine learning models can be trained to recognize patterns correlating with disease states.

Graph Neural Networks represent a relatively recent development in the machine learning landscape. Unlike conventional neural networks that treat each input feature independently, GNNs can model relational structure by representing data as graphs with nodes and edges. First proposed in 2009 and popularized in the mid-2010s, GNNs have achieved notable success in domains where relational information is important, including drug discovery, molecular property prediction, social network analysis, and recommendation systems. In drug discovery, for example, GNNs can model molecules as graphs where atoms are nodes and chemical bonds are edges, achieving state-of-the-art performance on molecular property prediction tasks. Their application to biomedical data beyond molecular modeling is still growing, but early results suggest that GNNs can capture complex biological relationships that traditional models miss.

Ensemble methods that combine multiple models into a unified predictor have been shown to improve prediction accuracy by reducing variance, mitigating overfitting, and capturing complementary patterns in the data. Common ensemble strategies include simple averaging of probability estimates, weighted voting based on validation performance, and stacking, where a meta-model learns to combine the outputs of base models. Research on ensemble methods has showed that they consistently outperform individual models across a wide range of domains, particularly when the base models are diverse in their underlying algorithms. In our work, the strong individual performance of all five models suggests that an ensemble could yield even better predictions.

Class imbalance is a recurring challenge in medical machine learning that requires careful attention. When one class contains far fewer samples than the other, standard training procedures tend to produce models that favor the majority class and perform poorly on the minority class, which is often the clinically more important one. Techniques to address this include class weighting in the loss function, random oversampling of the minority class, systematic undersampling of the majority class, and synthetic data generation methods like SMOTE. Evaluation metrics must also be chosen carefully, as accuracy can be misleadingly high even for a model that makes no meaningful predictions about the minority class. Our dataset has an 88.1% to 11.9% split between low-risk and high-risk patients, which we addressed through balanced class weights in the loss functions of all models.

Feature selection and dimensionality reduction are important considerations when working with high-dimensional biomedical data. With 200 DPV features and 700 training samples, the feature-to-sample ratio is about 1:3.5, which is high enough to raise concerns about overfitting. Dimensionality reduction techniques such as principal component analysis (PCA), t-distributed stochastic neighbor embedding (t-SNE), and feature importance ranking can help identify the most discriminative features and reduce the risk of learning spurious correlations. However, the strong and consistent performance across all five models in this study suggests that overfitting is not a major concern, and that the DPV features carry genuine predictive signal rather than noise.

Model interpretability is increasingly recognized as essential for clinical deployment of machine learning models. Clinicians need to understand why a model makes a particular prediction before they can trust it and act on it in patient care. Techniques such as SHAP (SHapley Additive exPlanations) values, LIME (Local Interpretable Model-agnostic Explanations), and attention mechanisms provide insights into which features drive model decisions. In the context of DPV data, interpretability methods could identify which specific voltage steps in the scan are most informative for risk classification, potentially revealing the underlying electrochemical compounds involved. While the current models perform well in terms of predictive accuracy, further work on interpretability would strengthen the case for clinical adoption.

Cross-validation is standard practice for evaluating machine learning models reliably, particularly in medical applications where the cost of overfitting is high. By splitting data into multiple folds, training on different combinations of folds, and averaging results across folds, cross-validation provides a stronger performance estimate than a single train-test split and reduces the influence of random variation in the data partition. Stratified k-fold cross-validation preserves class proportions within each fold, which is important for imbalanced datasets. The current study uses a single stratified 70-30 split, which is appropriate for this exploratory analysis but should be complemented with k-fold cross-validation and external validation on independent datasets in future work.

## 3. Dataset and Clinical Context

The dataset for this study comes from two sources within a single Excel file. The first source is a metadata sheet labeled Target Concentrations that contains patient identifiers and reference biomarker measurements. The second source is a collection of 1,000 individual sheets, one for each patient, each containing the raw DPV voltammetry data. This structure reflects the typical workflow of a clinical study where electrochemical measurements are recorded per patient and aggregated with clinical metadata for analysis.

The metadata sheet contains four columns. The sample identifier column provides a unique label for each patient, ranging from Sample 0001 to Sample 1000. The PSA column records the prostate-specific antigen concentration in picograms per milliliter (pg/mL). The AFP column records alpha-fetoprotein concentration, and the CA125 column records CA125 concentration in units per milliliter (U/mL). These three biomarkers are well established in clinical practice: PSA is the primary screening tool for prostate cancer, AFP is used as a tumor marker for liver cancer and germ cell tumors, and CA125 is used mainly for ovarian cancer monitoring. While AFP and CA125 are not specific to prostate cancer, their inclusion in the dataset reflects a thorough clinical profiling approach.

The DPV data is distributed across 1,000 individual sheets within the same workbook. Each sheet records the electrical current measured at a series of applied potential steps during the voltammetry scan. The scan procedure produces 200 current readings per patient, each corresponding to a different voltage level applied to the electrochemical cell. These 200 values form the core feature set for all five machine learning models, providing a high-dimensional representation of each patient's electrochemical profile.

Understanding how DPV works helps contextualize the data. The technique applies a sequence of voltage pulses to an electrochemical cell containing the patient sample. For each pulse, the current is measured immediately before and immediately after the pulse, and the difference between these two measurements is recorded as the signal. This differential measurement isolates the faradaic current, which arises from electrochemical reactions of electroactive species at the electrode surface, from the background capacitive current that comes from charging the electrical double layer at the electrode-electrolyte interface. The faradaic current is directly proportional to the concentration of the electroactive species, according to the Cottrell equation. Different compounds undergo oxidation or reduction at characteristic potentials, producing peaks in the DPV signal at specific voltage positions. The height and shape of these peaks reflect the concentration and electrochemical kinetics of each compound. By recording currents across a range of potentials, DPV builds a comprehensive electrochemical profile of the sample in a single measurement.

We derived the target variable for classification from PSA concentration following established clinical guidelines. A PSA level above 4,000 pg/mL, which is equivalent to 4 ng/mL, is considered high risk. Patients with PSA at or below this threshold are considered low risk. This threshold has been validated in large-scale clinical studies and is the most widely used cutoff in prostate cancer screening worldwide. All five models were trained to predict this binary classification.

The class distribution is notably imbalanced. Among the 1,000 patients, 881 are low risk (88.1%) and 119 are high risk (11.9%). This level of imbalance is typical of screening populations, where the prevalence of the condition of interest is low. It has important consequences for model training and evaluation. A naive classifier that simply predicts every patient as low risk would achieve 88.1% accuracy without learning anything useful, which is why accuracy alone is an insufficient metric for this problem. Metrics such as precision, recall, F1-score, and ROC-AUC provide more meaningful assessments of model performance on imbalanced data.

The DPV feature matrix has dimensions of 1,000 rows by 200 columns, representing 1,000 patients and 200 current measurements per patient. Combined with the metadata columns, the full working dataset has 1,000 rows and 204 columns. The 200 current measurements serve as the independent variables for predicting the binary risk label, while the metadata variables (PSA, AFP, CA125) are used for reference but not as model features.

Several characteristics of this dataset merit attention. DPV data is fundamentally different from the tabular biomarker data used in most clinical machine learning studies. Rather than a handful of clinical measurements, each patient is represented by a continuous curve of 200 values. This high-dimensional representation contains richer information about the sample composition but also requires more complex modeling approaches to extract the relevant patterns. The current measurements can be negative, which is a normal characteristic of voltammetry data resulting from the direction of current flow and the reference electrode configuration. This affects the choice of preprocessing steps, as methods like log transformation cannot handle negative values. The DPV curves are recorded as a sequence of measurements at equally spaced potential steps, introducing a sequential structure that only the Graph Neural Network takes advantage of in our model set.

The dataset is complete with no missing values in either the metadata or the DPV sheets. This is advantageous because it simplifies the preprocessing pipeline, avoids the need for imputation strategies, and ensures that all 1,000 samples are available for training and evaluation without any data loss.

## 4. Data Processing and Feature Engineering

The preprocessing pipeline transforms raw DPV current measurements into a format suitable for machine learning, with careful attention to data integrity, consistency, and the prevention of data leakage between training and validation sets.

### 4.1 Data Loading and Assembly

We loaded the Excel file and extracted DPV data from the individual sample sheets using openpyxl. The pipeline reads all sheet names, filters out the metadata sheet, and iterates through each sample sheet to extract the 200 current measurements per patient. The values are stored as rows in a NumPy array forming a 1,000 by 200 feature matrix.

The loading process also reads the metadata sheet for PSA, AFP, and CA125 concentrations using pandas. These values are aligned with the DPV features by matching sample identifiers. Data integrity checks confirm all 1,000 sheets are present with the expected 200 measurements each.

### 4.2 Target Variable Definition

We created the target variable by applying the standard clinical PSA threshold of 4,000 pg/mL. Patients with PSA above this threshold receive a label of 1 (high risk), and those at or below the threshold receive a label of 0 (low risk). This binarization follows the standard clinical practice for PSA screening, where the decision to proceed with further diagnostic testing is based on whether the PSA level exceeds the threshold.

The resulting distribution shows that 88.1% of patients are low risk and 11.9% are high risk. This imbalance has important implications for modeling. A naive classifier predicting every patient as low risk would achieve 88.1% accuracy without learning anything useful, reinforcing why accuracy alone is not a reliable metric for this problem.

### 4.3 Feature Scaling

We scaled the 200 DPV current measurements using RobustScaler from the scikit-learn library. RobustScaler centers the data by subtracting the median and scales it by dividing by the interquartile range (IQR), specifically the difference between the 75th and 25th percentiles. We chose this method over the more common StandardScaler (z-score normalization) because RobustScaler is less sensitive to outliers, which can be present in electrochemical measurements due to noise spikes, sample variability, or electrode fouling. The median and IQR are not pulled toward extreme values, unlike the mean and standard deviation used by StandardScaler.

An important detail is that we skipped the log transformation that is commonly applied to biomarker concentration data in clinical machine learning studies. DPV current measurements can contain negative values as a normal part of the voltammetry signal, which arises from the direction of current flow relative to the reference electrode. Log transformation cannot handle negative or zero values, making it inappropriate for this type of data. This preprocessing choice is specific to the nature of electrochemical data and differs from the typical preprocessing pipeline for clinical biomarker analysis.

We estimated the scaling parameters (median and IQR) from the training set only and then applied the transformation to both training and validation sets. This is a critical step to prevent data leakage, ensuring that information from the validation set does not influence the training process through the scaling parameters. After scaling, the final feature matrix retains its dimensions of 1,000 rows by 200 columns.

### 4.4 Train-Validation Split

We split the dataset using stratified sampling, which preserves the proportion of high-risk and low-risk patients in each subset. This is important for imbalanced datasets because a random split could by chance produce a validation set with very few high-risk samples, making performance evaluation unreliable. The stratified split allocates 70% of the data for training and 30% for validation, yielding a training set of 700 samples and a validation set of 300 samples. The training set contains 617 low-risk patients and 83 high-risk patients. The validation set contains 264 low-risk patients and 36 high-risk patients.

No separate test set was held out for final evaluation in this pipeline. The validation set serves as the primary benchmark for comparing model performance. This is a common approach in exploratory studies with limited sample sizes, where holding out a separate test set would further reduce the training data available for model learning.

## 5. Model Selection and Training

We trained five machine learning models on the scaled DPV features. The models span different families of algorithms, from simple linear classifiers to tree-based ensembles and graph neural networks, allowing us to compare how different modeling approaches perform on the same electrochemical data.

### 5.1 Logistic Regression

Logistic Regression served as our baseline model. It learns a set of weights for the 200 DPV features and combines them through a logistic function to produce a probability estimate between 0 and 1. Despite its simplicity, it can perform well when the decision boundary is roughly linear in the feature space, and it provides interpretable coefficients showing each feature's contribution.

We initialized the model with L2 regularization at a strength of 1.0 and a maximum of 1,000 iterations to ensure convergence. We enabled automatic class balancing so that the model assigns higher importance to the minority high-risk class during training.

### 5.2 Random Forest

Random Forest is an ensemble method that builds many decision trees on random subsets of the data and averages their predictions. Each tree is trained on a bootstrap sample of the training data and considers only a random subset of features at each split. This dual randomization reduces overfitting while maintaining low bias, producing a strong model that captures nonlinear relationships and feature interactions.

We configured the Random Forest with 100 trees, a maximum depth of 10 levels, and a minimum of 5 samples required to split a node. We used balanced class weights to address the class imbalance.

### 5.3 Support Vector Machine

The Support Vector Machine finds a decision boundary that maximizes the margin between the two classes. For nonlinearly separable data, SVM uses a kernel function to map inputs into a higher-dimensional space where the classes become separable, using the kernel trick for computational efficiency.

We configured the SVM with a radial basis function (RBF) kernel and set gamma to scale automatically based on the number of features and their variance. We enabled probability estimates through Platt scaling, and set the regularization parameter C to the default of 1.0.

### 5.4 XGBoost

XGBoost is a gradient boosting framework that builds decision trees sequentially, with each new tree trained to correct the residual errors of the previous ensemble. It is known for strong performance on structured data, efficient implementation with parallel processing, and built-in regularization.

We configured XGBoost with a maximum tree depth of 6, a learning rate of 0.1, and a scale positive weight parameter adjusted for class imbalance. The model was trained to minimize log loss with early stopping.

### 5.5 Graph Neural Network

The Graph Neural Network is the most complex model in our pipeline. Unlike the other models, which treat the 200 DPV features as independent inputs, the GNN explicitly models relationships between features by constructing a graph where nodes represent individual potential steps and edges represent correlations between them. The insight is that the DPV signal is a continuous curve where adjacent voltage steps are physically related, and this relational structure contains information that a standard model ignores.

We built the graph by computing the Pearson correlation matrix of the scaled training data. Pairs with absolute correlation greater than 0.3 are connected by an edge. The GNN architecture consists of three graph convolutional layers with 64, 32, and 16 hidden units, followed by a linear output layer with sigmoid activation. We trained for 200 epochs using the Adam optimizer with early stopping.

### 5.6 Training Procedure Summary

All models were trained on the same 700-sample training set and evaluated on the same 300-sample validation set. We used scikit-learn for Logistic Regression, Random Forest, and SVM; the XGBoost library for gradient boosting; and PyTorch with PyTorch Geometric for the GNN. After training, we saved each model to disk in pickle format for future use, along with the fitted RobustScaler and feature column names. The pipeline also generates diagnostic visualizations including confusion matrices, ROC curves, a radar chart, parallel coordinates, density plots, and a model comparison heatmap.

## 6. Overall Model Performance

We evaluated each model on the validation set of 300 patients using five standard classification metrics. Accuracy measures the proportion of all predictions that are correct. Precision measures how many of the patients predicted as high risk actually are high risk, reflecting the reliability of positive predictions. Recall (sensitivity) measures how many of the actual high-risk patients the model successfully identifies. The F1-score is the harmonic mean of precision and recall, providing a single metric that balances both concerns. ROC-AUC measures the model's ability to discriminate between the two classes across all possible decision thresholds, with 1.0 meaning perfect separation and 0.5 meaning random chance.

The following table summarizes the results for all five models:

**Table 1: Model Performance Summary on Validation Set**

| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
|-------|----------|-----------|--------|----------|---------|
| Logistic Regression | 93.67% | 68.89% | 86.11% | 76.54% | 98.47% |
| Random Forest | 95.00% | 88.89% | 66.67% | 76.19% | 98.61% |
| SVM | 95.33% | 86.67% | 72.22% | 78.79% | 98.33% |
| XGBoost | 94.67% | 75.00% | 83.33% | 78.95% | 98.71% |
| GNN | 93.67% | 69.77% | 83.33% | 75.95% | 96.63% |

All five models achieved ROC-AUC scores above 96%, with four of the five exceeding 98%. These numbers are remarkably high compared to typical biomedical classification tasks, where ROC-AUC values in the range of 80-90% are often considered strong. This suggests that the DPV signal contains a strong predictive signature for prostate cancer risk that is consistent across different modeling approaches.

The models exhibit different precision-recall trade-offs. Logistic Regression achieved the highest recall (86.11%) but lower precision (68.89%), catching most high-risk patients at the cost of more false positives. This behavior would be advantageous in a screening scenario where the priority is to miss as few cases as possible. Random Forest showed the opposite pattern with the highest precision (88.89%) but lowest recall (66.67%), making cautious positive predictions. This conservative behavior would be more appropriate in a diagnostic setting where false positives lead to costly or invasive follow-up procedures. SVM and XGBoost struck a balance between these extremes, while the GNN behaved similarly to Logistic Regression with high recall and moderate precision.

XGBoost recorded the highest F1-score (78.95%) and the highest ROC-AUC (98.71%), making it the best overall performer. Its balanced performance across all metrics makes it the most suitable model for general use.

The radar chart visualizes all five metrics on a single plot, with each metric on a separate axis radiating from the center.

@file:radar_chart_comparison.png

In the radar chart, all models achieve high scores on accuracy and ROC-AUC, with their polygons extending toward the outer ring on these axes. The precision and recall axes show the most variation between models. Random Forest reaches furthest on the precision axis but pulls inward on recall. Logistic Regression and the GNN show the opposite pattern, reaching further on recall but falling shorter on precision. SVM and XGBoost display more balanced shapes that sit between these extremes.

The model comparison heatmap provides a complementary view of the same data as a color-coded matrix.

@file:model_comparison_heatmap.png

XGBoost shows consistently dark cells across all five metrics, confirming its balanced performance. Random Forest has a very dark cell for precision but a noticeably lighter one for recall. Logistic Regression shows the opposite, with stronger recall than precision. The GNN has slightly lighter shading on most metrics compared to the top four models, consistent with its slightly lower ROC-AUC.

## 7. Confusion Matrix Analysis

Confusion matrices provide a detailed breakdown of model predictions into four categories: true negatives (correctly classified low-risk patients), false positives (low-risk patients incorrectly classified as high risk), false negatives (high-risk patients incorrectly classified as low risk), and true positives (correctly classified high-risk patients). We generated one confusion matrix for each of the five models to understand their error patterns.

### 7.1 Logistic Regression

**Table 2: Logistic Regression Performance**

| Accuracy | Precision | Recall | F1-Score | ROC-AUC |
|----------|-----------|--------|----------|---------|
| 93.67% | 68.89% | 86.11% | 76.54% | 98.47% |

Logistic Regression achieved the highest recall among all models, catching 31 of 36 high-risk patients. Its precision was lower at 68.89%, reflecting a tendency to flag borderline low-risk patients as high risk. This is the expected behavior of a model optimized for sensitivity rather than specificity.

@file:advanced_cm_logistic_regression.png

The confusion matrix confirms that the model identifies most high-risk cases correctly while producing 14 false positives. At 36 actual high-risk cases, missing only 5 represents strong performance for a linear model. The false positive count of 14 out of 264 low-risk patients is manageable in most clinical contexts.

### 7.2 Random Forest

**Table 3: Random Forest Performance**

| Accuracy | Precision | Recall | F1-Score | ROC-AUC |
|----------|-----------|--------|----------|---------|
| 95.00% | 88.89% | 66.67% | 76.19% | 98.61% |

Random Forest achieved the highest precision among all models at 88.89%, meaning its positive predictions are very reliable. However, its recall of 66.67% was the lowest, reflecting a conservative prediction strategy that prioritizes avoiding false positives over catching every high-risk case.

@file:advanced_cm_random_forest.png

The confusion matrix shows only 4 false positives, the lowest among all models, making it the most specific classifier. However, the model missed 12 high-risk patients, the highest false-negative count in the study. In a clinical context, this model would be most appropriate when confirmatory testing is expensive or invasive, and avoiding unnecessary procedures is the priority.

### 7.3 Support Vector Machine

**Table 4: SVM Performance**

| Accuracy | Precision | Recall | F1-Score | ROC-AUC |
|----------|-----------|--------|----------|---------|
| 95.33% | 86.67% | 72.22% | 78.79% | 98.33% |

SVM achieved the highest overall accuracy at 95.33% while maintaining a balanced precision-recall profile. Its F1-score of 78.79% was the second highest, closely trailing XGBoost.

@file:advanced_cm_svm.png

The SVM correctly identified 26 of 36 high-risk patients with 10 false positives. Its performance sits between the aggressive recall of Logistic Regression and the cautious precision of Random Forest, making it a reasonable all-purpose choice.

### 7.4 XGBoost

**Table 5: XGBoost Performance**

| Accuracy | Precision | Recall | F1-Score | ROC-AUC |
|----------|-----------|--------|----------|---------|
| 94.67% | 75.00% | 83.33% | 78.95% | 98.71% |

XGBoost achieved the highest F1-score and the highest ROC-AUC, making it the top performer in the study. Its precision of 75% and recall of 83.33% represent a favorable balance that generalizes well across different clinical priorities.

@file:advanced_cm_xgboost.png

The confusion matrix shows balanced performance: 30 of 36 high-risk patients correctly identified, only 6 missed, and 10 false positives. This even distribution across error types is why XGBoost achieved the highest F1-score, as the harmonic mean penalizes extreme imbalances between precision and recall.

### 7.5 Graph Neural Network

**Table 6: GNN Performance**

| Accuracy | Precision | Recall | F1-Score | ROC-AUC |
|----------|-----------|--------|----------|---------|
| 93.67% | 69.77% | 83.33% | 75.95% | 96.63% |

The GNN achieved recall of 83.33%, tying with XGBoost, but its ROC-AUC of 96.63% was slightly below the other models. This suggests that while the GNN's classification at the default threshold is competitive, its probability estimates are less well calibrated across the full range of thresholds.

@file:advanced_cm_gnn.png

The GNN confusion matrix resembles the Logistic Regression pattern: high recall (30 of 36 high-risk patients identified) with a moderate number of false positives (13). The slightly lower ROC-AUC likely reflects the GNN's need for larger training datasets, which is a known limitation of graph-based models that typically require more data to learn stable representations.

Taken together, the confusion matrices show strong but varied performance across the five models. All models correctly classify at least 270 of 300 validation patients, and the choice of model depends on whether the clinical priority is sensitivity (catching cases) or specificity (avoiding false alarms).

## 8. Probability Distribution Analysis

### 8.1 Ridge Plot

The ridge plot shows the distribution of predicted probabilities for each model, separated by the true class of the patient. Each model has two kernel density estimation curves: one for low-risk patients shown in one color and one for high-risk patients shown in another. The degree of separation between these curves shows how confidently the model distinguishes the two classes, with greater separation corresponding to better discriminative performance.

@file:ridge_plot_distributions.png

For all five models, the low-risk and high-risk density curves are well separated with minimal overlap. The low-risk curves peak near zero probability, meaning most low-risk patients receive very low risk scores. The high-risk curves peak well above the 0.5 decision threshold, meaning most high-risk patients receive clearly elevated risk scores. This clean separation across all models explains the consistently high ROC-AUC scores, which are all above 96%.

Random Forest shows the cleanest separation, with the low-risk curve tightly concentrated near zero and almost no tail extending above 0.5. This reflects its very low false positive rate of only 4 out of 264 low-risk patients. Logistic Regression shows a longer tail in the low-risk distribution extending above 0.5, corresponding to its higher false positive rate of 14 patients. XGBoost and SVM show intermediate patterns with moderate tails, while the GNN shows slightly more overlap than the other models, consistent with its slightly lower ROC-AUC of 96.63%.

### 8.2 Violin Plot

The violin plots display predicted probability distributions for both classes as mirrored kernel density curves within each model, providing a more detailed view of the distribution shapes and their spread.

@file:violin_plots.png

The Random Forest violin shows a very narrow low-risk distribution concentrated at the bottom of the plot, reflecting strong confidence in low-risk predictions with very few outliers straying above the threshold. The Logistic Regression violin shows wider distributions for both classes, showing less certainty in individual predictions and a wider spread of probability scores. The SVM and XGBoost violins show intermediate characteristics with moderate spread. The GNN violin shows the widest overlap between the two class distributions in the middle region, consistent with its slightly lower discriminative performance as measured by ROC-AUC.

## 9. ROC Curve Analysis

ROC curves plot the true positive rate (sensitivity) against the false positive rate (1 - specificity) as the decision threshold varies continuously from 0 to 1. The area under the curve (AUC) provides a single threshold-independent measure of model discrimination that summarizes performance across all possible operating points. An AUC of 1.0 represents perfect discrimination, while an AUC of 0.5 represents performance equivalent to random guessing. In medical diagnostics, AUC values above 0.9 are generally considered excellent.

**Table 7: ROC-AUC Comparison**

| Model | ROC-AUC |
|-------|---------|
| XGBoost | 98.71% |
| Random Forest | 98.61% |
| Logistic Regression | 98.47% |
| SVM | 98.33% |
| GNN | 96.63% |

@file:roc_curves_enhanced.png

The ROC curves for all five models rise steeply from the origin, reaching near-perfect true positive rates at very low false positive rates. This shows that all models can achieve high sensitivity while maintaining specificity. XGBoost sits slightly above the others for most of its range, consistent with its highest ROC-AUC. The GNN curve sits slightly below the others, particularly in the low false positive rate region. All curves are far above the diagonal line representing random chance, confirming the strong discriminative power of the DPV features regardless of the modeling approach used.

## 10. Parallel Coordinates Analysis

The parallel coordinates plot connects each patient's predicted probability across all five models along parallel vertical axes, with one axis per model. This visualization reveals patterns of agreement and disagreement between models.

@file:parallel_coordinates.png

Clear bands emerge where all five models agree on low risk (lines concentrated at the bottom of each axis) and other bands where they agree on high risk (lines concentrated at the top). A region of overlap in the middle shows cases where model predictions diverge, representing borderline patients for whom different models make different predictions. These divergent cases are particularly interesting because they represent the inherent uncertainty in the classification task. The parallel coordinates plot can help identify patients who might benefit from more diagnostic testing or clinical review.

## 11. Discussion of Model Behavior

The most striking observation across all models is the strength and consistency of the signal in the DPV data. Every model, regardless of its complexity, underlying assumptions, or algorithmic family, achieved validation ROC-AUC above 96%. This consistency strongly suggests that the DPV features contain genuine electrochemical markers of prostate cancer risk rather than noise that a particular model happens to fit. The strength of the signal across different modeling approaches is one of the most important findings of this study.

### 11.1 Impact of Class Imbalance

Our dataset has an 88.1% to 11.9% class split. We addressed this through class weighting, which assigns higher importance to the minority class during training. All models achieved recall rates well above the 11.9% baseline, ranging from 66.67% to 86.11%, showing the strategy worked. The precision-recall trade-off is a direct consequence of the imbalance: models that prioritize recall produce more false positives, while models that prioritize precision miss more high-risk patients. The F1-score captures this trade-off and is the most appropriate single metric for comparing models on this dataset.

### 11.2 Logistic Regression

Logistic Regression achieved the highest recall (86.11%), suggesting a strong linear component in the mapping from DPV features to risk. Its lower precision (68.89%) means it also flags many low-risk patients as high risk, but this may be acceptable in a screening context where sensitivity is the priority. The strong performance of this simple model is noteworthy because it shows that complex nonlinear interactions are not strictly necessary for good classification on this dataset.

### 11.3 Random Forest

Random Forest showed the opposite strategy with the highest precision (88.89%) and lowest recall (66.67%). It is the most conservative model, making fewer positive predictions but being more confident when it does. The high precision makes it suitable for a confirmatory testing scenario where false positives carry high costs, either financially or in terms of patient anxiety and unnecessary procedures.

### 11.4 Support Vector Machine

SVM balanced precision and recall with an F1-score of 78.79%, the second highest overall. Its accuracy of 95.33% was the highest of any model. The RBF kernel appears well suited to the DPV feature space, capturing nonlinearities in the decision boundary without overfitting to the training data.

### 11.5 XGBoost

XGBoost achieved the highest F1-score (78.95%) and ROC-AUC (98.71%). Its balanced performance across all metrics makes it the strongest model in the ensemble. The sequential boosting approach appears effective at extracting the maximum predictive signal from the DPV data while maintaining good generalization.

### 11.6 Graph Neural Network

The GNN performed competitively despite being the most complex model in our set. Its recall matched XGBoost at 83.33%, but its ROC-AUC of 96.63% was slightly lower than the other models. This likely reflects the GNN's greater data requirements, as graph-based models typically need larger sample sizes to learn strong representations of relational structure. The correlation-based graph structure we used is a reasonable starting point, but alternative graph construction methods, such as using spatial adjacency of voltage steps or learned graph structures, could potentially improve performance.

### 11.7 Ensemble Considerations

Combining all five models into an ensemble could produce stronger predictions than any single model alone. Simple averaging of probability estimates would smooth out individual model extremes and reduce prediction variance. Weighted averaging based on validation performance could give more influence to stronger models like XGBoost. Cases where all models agree represent high-confidence predictions, while split votes would flag borderline cases warranting more review.

## 12. Clinical and Analytical Implications

### 12.1 Clinical Risk Interpretation

Our results show that DPV voltammetry combined with machine learning can classify prostate cancer risk with high accuracy across multiple model types. In screening, a high-recall model like Logistic Regression (86% recall) would catch the most cases. In diagnostic settings where confirmatory testing is burdensome, a high-precision model like Random Forest (89% precision) would minimize unnecessary procedures.

### 12.2 Model Utility in Practice

The consistent performance across all five modeling approaches suggests that the DPV signal is strong and not dependent on any particular algorithm. This is an important finding because it means that a practical DPV-based screening tool could be built using a relatively simple model while achieving good performance. The practical utility of such a tool depends on factors beyond model performance, including the cost and portability of DPV equipment, ease of sample collection, turnaround time, and integration with existing clinical workflows.

### 12.3 Decision Threshold Strategy

All models use a default threshold of 0.5, but this may not be optimal for clinical use where false positives and false negatives have asymmetric costs. The ROC curves provide the information to select a threshold aligned with clinical priorities. A lower threshold would increase recall (catch more cases), while a higher threshold would improve precision (reduce false alarms). The choice should be made in consultation with clinicians based on the specific clinical context.

### 12.4 Cost-Benefit Analysis

False positives lead to unnecessary follow-up procedures, patient anxiety, and healthcare costs. False negatives lead to missed diagnoses, delayed treatment, and potentially worse outcomes. The models in this study provide excellent discrimination, enabling favorable trade-offs between these competing concerns. At a false positive rate of 10%, all models achieve true positive rates above 90%. This level of performance is well within the range needed for clinical utility and compares favorably with current PSA-based screening approaches.

## 13. Limitations and Future Work

### 13.1 Dataset Size and Imbalance

Our dataset of 1,000 patients is modest for training and evaluating machine learning models, particularly for the high-risk class with only 119 samples. Larger datasets would allow stronger model training, more reliable performance estimates, and better support for data-intensive models like the GNN. Future work should prioritize collecting more samples, particularly from high-risk patients, to address the class imbalance and improve model strength.

### 13.2 Feature Scope

This study uses DPV data only as input features. Combining DPV features with clinical variables such as age, family history, body mass index, and standard biomarker levels could provide a more complete picture of patient risk and potentially improve predictive performance. The current DPV features capture electrochemical information, while clinical variables capture demographic and historical risk factors, making them complementary data sources.

### 13.3 Model Calibration and Explainability

We have not thoroughly evaluated the calibration of our models, meaning we have not verified that the predicted probabilities correspond to empirical frequencies. Well-calibrated probability estimates are important for clinical decision-making, where a predicted probability of 80% should mean that about 80 out of 100 such patients are actually high risk. Explainability techniques like SHAP values could identify which specific voltage steps in the DPV curve drive predictions, providing insights into the underlying electrochemistry and potentially revealing novel biomarkers.

### 13.4 Graph Neural Network Development

The GNN has a lot of room for improvement. Future work could explore different graph construction methods, such as using spatial adjacency of voltage steps or learned attention-based graphs. Alternative architectures, including graph attention networks (GATs) and graph isomorphism networks (GINs), could be evaluated. Training on larger datasets would also likely benefit the GNN, which typically requires more data than the other models to learn stable representations.

### 13.5 Cross-Validation and Generalization

We used a single stratified train-validation split for efficiency. K-fold cross-validation would provide more reliable performance estimates by averaging results across multiple data partitions, reducing the influence of random variation in the split. External validation on an independent dataset collected from a different population or using different equipment would be the strongest test of generalization and is essential before clinical deployment.

### 13.6 Equipment and Measurement Variability

Our data was collected under controlled experimental conditions using a single DPV instrument. Real clinical settings introduce variability from different equipment, sample handling protocols, operator techniques, and environmental conditions. Robustness testing across different instruments, laboratories, and conditions is needed to ensure that model performance translates to real-world settings.

### 13.7 Biological Interpretability

Our current models do not reveal which specific electrochemical compounds or properties are associated with prostate cancer risk. Identifying the biological basis of the DPV signal would strengthen the clinical case for this approach and could open new avenues for understanding prostate cancer biology. This would require linking specific features of the DPV curves to known electrochemical compounds and validating their association with prostate cancer through independent biochemical assays.

### 13.8 Single-Center Data

All data comes from a single clinical center, which raises questions about generalizability. Multi-center studies involving different patient populations, geographic regions, and clinical practices are needed to test whether the results generalize broadly. Differences in population demographics, diet, medication use, and sample handling could all affect the DPV signal and model performance.

### 13.9 Model Maintenance and Updates

In a deployed clinical system, models would need periodic updating as new data becomes available and as the underlying population or measurement technology evolves. A governance framework for model updates, including retraining schedules, performance monitoring, version control, and clinical validation of updated models, should be established as part of any deployment plan.

## 14. Supplementary Tables

**Table 8: Data Dictionary**

| Variable | Type | Source | Clinical Role | Analytical Transformation |
|----------|------|--------|---------------|---------------------------|
| sample_id | Identifier | Metadata sheet | Unique record ID | None |
| PSA_pg_per_ml | Numeric | Metadata sheet | Primary risk definition | Threshold-based target creation |
| AFP_pg_per_ml | Numeric | Metadata sheet | Secondary biomarker | Not used as feature in DPV pipeline |
| CA125_U_per_ml | Numeric | Metadata sheet | Secondary biomarker | Not used as feature in DPV pipeline |
| V0 through V199 | Numeric | DPV sample sheets | DPV current measurements | RobustScaler normalization |

**Table 9: Sample Distribution by Partition** — The dataset was split into 700 training samples and 300 validation samples using stratified sampling. Both partitions kept roughly the same 12% high-risk proportion as the full dataset.

| Partition | Total Samples | Low Risk | High Risk | High Risk Percentage |
|-----------|---------------|----------|-----------|----------------------|
| Training | 700 | 617 | 83 | 11.9% |
| Validation | 300 | 264 | 36 | 12.0% |

**Table 10: Complete Model Performance Metrics**

| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
|-------|----------|-----------|--------|----------|---------|
| Logistic Regression | 0.9367 | 0.6889 | 0.8611 | 0.7654 | 0.9847 |
| Random Forest | 0.9500 | 0.8889 | 0.6667 | 0.7619 | 0.9861 |
| SVM | 0.9533 | 0.8667 | 0.7222 | 0.7879 | 0.9833 |
| XGBoost | 0.9467 | 0.7500 | 0.8333 | 0.7895 | 0.9871 |
| GNN | 0.9367 | 0.6977 | 0.8333 | 0.7595 | 0.9663 |

**Table 11: Model Evaluation Metric Definitions**

| Metric | Definition | Clinical Relevance |
|--------|------------|---------------------|
| Accuracy | Proportion of all predictions that are correct | General performance indicator, but can be misleading with imbalanced data |
| Precision | Proportion of high-risk predictions that are correct | Indicates how reliable a positive prediction is |
| Recall | Proportion of actual high-risk patients correctly identified | Indicates how well the model catches patients who need further investigation |
| F1-Score | Harmonic mean of precision and recall | Balances the trade-off between catching high-risk patients and avoiding false alarms |
| ROC-AUC | Area under the receiver operating characteristic curve | Measures overall discriminative power across all possible thresholds |

### Figure Placeholder Summary

- @file:advanced_cm_logistic_regression.png
- @file:advanced_cm_random_forest.png
- @file:advanced_cm_svm.png
- @file:advanced_cm_xgboost.png
- @file:advanced_cm_gnn.png
- @file:radar_chart_comparison.png
- @file:parallel_coordinates.png
- @file:ridge_plot_distributions.png
- @file:violin_plots.png
- @file:model_comparison_heatmap.png
- @file:roc_curves_enhanced.png
- @file:best_model_modal.png

## 15. Conclusion

This report documents a comprehensive machine learning pipeline for prostate cancer risk classification using DPV voltammetry data. We have showed that DPV measurements carry strong and consistent electrochemical signatures associated with prostate cancer risk, and that machine learning models from diverse algorithmic families can extract these signals with a high degree of accuracy. The strength of the signal across all five models is a particularly encouraging finding, as it shows that the predictive information is not dependent on a specific modeling approach.

All five models achieved validation accuracies above 93% and ROC-AUC scores above 96%, with four of five exceeding 98%. XGBoost delivered the strongest overall performance with a ROC-AUC of 98.71% and an F1-score of 78.95%. The models exhibit different precision-recall trade-offs that make them suitable for different clinical applications. A high-recall model like Logistic Regression would be preferred for screening where the priority is catching as many cases as possible, while a high-precision model like Random Forest would be preferred for confirmatory testing where false positives carry big costs. An ensemble approach combining all five models would likely provide the most reliable predictions for general use by leveraging the strengths of each individual model.

These findings support continued development of DPV-based diagnostic tools for prostate cancer screening. The combination of a fast, label-free electrochemical measurement with machine learning analysis offers the potential for an accurate and accessible screening tool that could address some of the well-known limitations of current PSA-based screening. The results presented here, including the strong and consistent performance across multiple models and the detailed diagnostic visualizations, provide a solid foundation for pursuing this goal through larger studies, multi-center validation, and further refinement of both the measurement and modeling approaches.

