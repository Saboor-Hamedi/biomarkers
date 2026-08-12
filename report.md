AI-Assisted Decoding of Multiplexed Electrochemical Fingerprints for Simultaneous Quantification of PSA, AFP, and CA125 in Serum
Abdul Saboor Hamedi
Saboorhamedi49@gmail.com

Supervisors:
Prof. Murni & Dr. Jagadeesh Suriyaprakash

Abstract
This study asks whether a single differential pulse voltammetry (DPV) scan can be computationally decoded into the concentrations of three cancer biomarkers at the same time. A multiplexed electrochemical biosensor was used to record one 200-point current-potential fingerprint from each of 1,000 serum samples, and every sample was spiked with a known concentration of PSA, AFP, and CA125. Because the true concentrations are known, the AI task is quantitative decoding rather than classification. A controlled hierarchy of regression models, from linear and PLS regression through random forest, SVR, and XGBoost to an MLP and a 1D-CNN, was evaluated with 5-fold cross-validation using out-of-fold predictions and a strict untouched test set. XGBoost was the strongest decoder, reaching cross-validation R² values of 0.881, 0.837, and 0.872 for PSA, AFP, and CA125, and test-set R² values of 0.888, 0.853, and 0.870. A single multi-output model decoded all three biomarkers from one fingerprint with the same accuracy as three separate models, which confirms the multiplexing information survives in the raw signal. Ablation shows the experimentally established redox windows carry most of the information, while the full fingerprint adds a small but real amount on top. SHAP analysis shows the models concentrate on the same potential regions the experimental team already identified for PSA, AFP, and CA125. As a secondary application, the decoded PSA was converted into high and low classes using the predefined clinical cutoff of 4,000 pg·mL⁻¹, and the best classifier reached a ROC-AUC of 99.75%. The central result is that a multiplexed electrochemical fingerprint can be decoded quantitatively by AI without inventing any new chemistry.
Keywords: differential pulse voltammetry, multiplexed biosensor, machine learning, multi-output regression, PSA, AFP, CA125, explainable AI

1. Introduction
1.1 Need for multiplexed biomarker detection
Prostate cancer is one of the most common cancers in men worldwide, and finding it early improves the chance of recovery (Hunt & Slaughter, 2025). The main screening test has long been the PSA blood test. PSA is simple and cheap, but not specific enough: some men with high PSA do not have cancer, and some with normal PSA do, so it can cause unnecessary biopsies and miss cases that need treatment (Westerlinck, 2025). Biomarkers such as AFP and CA125 are also used clinically, and a panel of several biomarkers usually gives more reliable information than any single one (Ye et al., 2021).
1.2 Limitations of conventional multi-analyte interpretation
Conventional laboratory quantification relies on optical assays that are costly, need specialized instrumentation, take hours to days, and usually measure one analyte per assay, so a multi-biomarker panel needs several sequential measurements. Combined markers such as the Prostate Health Index (PHI) and the 4Kscore improve specificity over PSA alone, but their adoption has been limited by cost and complexity (Dong et al., 2023).
1.3 DPV as a rich electrochemical fingerprint
Differential pulse voltammetry (DPV) is a mature electrochemical technique (York, 1981; Alyamni et al., 2024). A series of small voltage pulses is applied to an electrode in the sample, and the current is measured at each step, isolating the faradaic current from background effects. The result is a curve of roughly 200 current values acting as a fingerprint of the sample's electrochemistry (Hunt & Slaughter, 2025), and because the whole curve is recorded in one scan, it can in principle carry information about several analytes at once. The experimental team has developed a multiplexed electrode producing a response for PSA, AFP, and CA125 in a single scan, with 1,000 serum samples of known concentrations.
1.4 Potential of machine learning for signal decoding
Machine learning has been applied to cancer research across many domains (Paul et al., 2024; Gogoshin & Rodin, 2023). In this study the input is a high-dimensional fingerprint with 200 features per sample. Instead of manually picking one peak and reading its height, a model can learn how the whole curve relates to the known biomarker concentrations, and SHAP can show which voltage steps the model actually uses (Gao et al., 2025).
1.5 Research gap
The experimental system contains three biomarkers, yet previous analysis reduced this multiplexed experiment to a single binary PSA high/low classification. What is not yet established is whether one DPV fingerprint can estimate all three biomarker concentrations at once.
1.6 Hypothesis and objective
The hypothesis is that a single multiplexed DPV fingerprint can be decoded into simultaneous quantitative estimates of PSA, AFP, and CA125. The objective is to test how well different models reconstruct the three concentrations and whether the voltage regions used match the experimentally established sensing regions.

2. Experimental Section
This section is provided by the experimental and material-science team, who fabricated the electrode, characterized the material, established the electrochemical behaviour and the multiplexed sensing protocol, prepared the serum samples, and recorded the DPV scans. The AI work described here uses the dataset they generated and does not modify the experimental chemistry. The relevant experimental steps are: electrode fabrication (2.1), material characterization (2.2), electrochemical characterization (2.3), the multiplexed sensing protocol (2.4), serum preparation (2.5), the PSA/AFP/CA125 concentration design (2.6), and DPV acquisition (2.7).

3. Dataset and AI Methodology
3.1 Dataset construction
The dataset consists of 1,000 experimental serum samples. Human serum was obtained from known commercial suppliers, and each sample was spiked with a known amount of PSA, AFP, and CA125. Different samples were spiked with different ratios, so the exact concentration of every biomarker in every sample is known by design. The spiked PSA concentrations range from about 0.001 to 99,265 pg·mL⁻¹, AFP from about 0.0005 to 17,237 pg·mL⁻¹, and CA125 from about 0.03 to 204 U·mL⁻¹. The concentrations span several orders of magnitude, so log-transformed concentrations are used as regression targets. A data audit confirmed 1,000 unique sample IDs, 200 DPV measurements per sample, no duplicate rows, no missing values, and correct alignment between concentrations and DPV curves. Table 1 summarises the dataset.

Table 1: Dataset characteristics
Samples: 1,000
Unique sample IDs: 1,000
DPV measurements per sample: 200
Potential range: −750 to +1,250 mV
Potential step: 11 mV
Missing values: 0
PSA range: 0.001–99,265 pg·mL⁻¹
AFP range: 0.0005–17,237 pg·mL⁻¹
CA125 range: 0.03–204 U·mL⁻¹
Regression target: log10 concentration
3.2 DPV representation
For each sample, a DPV voltammogram was recorded by sweeping the applied potential from −750 mV to +1,250 mV with a step of 11 mV, producing 200 current readings. The input to every model is this full 200-point fingerprint. DPV currents can be negative, which is normal for voltammetry, so no log transform is applied to them.
3.3 Electrochemical feature extraction
In addition to the raw fingerprint, six engineered electrochemical features were computed from the same measurements: anodic peak current, anodic peak potential, cathodic peak current, cathodic peak potential, area under the curve, and peak separation. These are derived representations of the same DPV signal and are not independent experiments. Table 2 defines how each feature was calculated.

Table 2: Electrochemical feature definitions
Feature: peak_anodic_current — anodic peak current detected in the DPV scan
Feature: peak_anodic_potential — potential of the anodic peak
Feature: peak_cathodic_current — cathodic peak current detected in the DPV scan
Feature: peak_cathodic_potential — potential of the cathodic peak
Feature: area_under_curve — numerical integral of the current over potential
Feature: peak_separation — difference between anodic and cathodic peak potentials
(Detection used the validated peak windows; no sample was missing a peak, so no imputation was needed.)
3.4 Data preprocessing
All preprocessing was fitted on training data only. Scaling was applied inside each cross-validation fold using a robust scaler that centres on the median and scales by the interquartile range, which is less sensitive to outliers than standard z-score scaling. This prevents data leakage.
3.5 Train, validation, and test strategy
The dataset was handled in two complementary ways. First, a 5-fold cross-validation produced out-of-fold predictions for every sample. Second, a strict split held out a 20% test set that was never used for feature selection, model selection, hyperparameter tuning, threshold selection, or fitting of any preprocessing step. The test set was used only to report final performance.
3.6 Regression models
A controlled hierarchy of models was used rather than a fixed set of competitors: linear regression and partial least squares regression as baselines; random forest, support vector regression, and XGBoost as classical machine learning; and an MLP and a 1D-CNN as neural approaches. All models were compared on identical folds and the same metrics. A simpler model performing as well as a deep one is an important result in itself.
3.7 Multi-output model
The central experiment is multi-output regression, where one model takes the 200-point fingerprint as input and returns PSA, AFP, and CA125 together. This is the computational equivalent of decoding the multiplexed electrochemical signal in one step.
3.8 Classification model (secondary)
As a secondary application, the decoded PSA concentration was converted into two classes using the predefined experimental threshold of 4,000 pg·mL⁻¹ (4 ng·mL⁻¹). This is classification according to the predefined experimental PSA threshold. It is not a clinical prostate-cancer diagnosis, because the dataset contains no biopsy-confirmed cancer status or other clinical ground truth.
3.9 Explainable AI
After selecting the best regression model, SHAP was used to ask which voltage regions contribute to each biomarker's prediction. The AI-important regions were then compared with the experimentally established redox windows. The correct interpretation of this comparison is that AI identifies predictive electrochemical regions that are consistent with the experimentally established sensing responses. SHAP does not by itself establish the chemical identity or mechanism of a peak; that was established experimentally.
3.10 Statistical analysis
For each model and biomarker, R², MAE, RMSE, Pearson r, and Spearman correlation were computed on out-of-fold and test predictions. Uncertainty was assessed with bootstrap 95% confidence intervals on the test R² for each biomarker. Predicted-versus-measured and residual plots were produced for every biomarker.

4. Results
4.1 DPV fingerprint characteristics
Figure 2 shows representative raw DPV curves. The fingerprints share the same overall shape across samples, with differences concentrated in three potential regions. The peaks are reproducible across the 1,000 scans, which is the expected behaviour of a well-characterized multiplexed electrode.
[Figure 2 placeholder: Representative raw DPV fingerprints — to be inserted]
4.2 Biomarker-specific electrochemical response
The DPV signal was compared with each biomarker's log-concentration at the experimentally established potential regions. The strongest correlations are listed below and shown in Figure 3. All three biomarkers map to distinct potential regions: PSA near −468 mV, AFP near 365 mV, and CA125 near 968 mV. These are the same regions the experimental team validated. Figure 4 shows the full three-biomarker fingerprint map.
Pearson correlation at validated regions: PSA r = −0.8624 at −468 mV; AFP r = −0.8577 at 365 mV; CA125 r = −0.8367 at 968 mV.
Strongest single-step correlation (log-concentration): PSA r = −0.8162 at −458 mV; AFP r = −0.7984 at 375 mV; CA125 r = −0.8024 at 968 mV.
[Figure 3 placeholder: Biomarker concentration versus electrochemical response — to be inserted]
[Figure 4 placeholder: Three-biomarker electrochemical fingerprint map — to be inserted]
4.3 Quantitative relationship between DPV and PSA
The DPV current at the PSA window correlates with log-PSA concentration with a Pearson r of −0.8624 at −468 mV. The correlation is negative, meaning higher PSA levels produce a localized current change at this potential.
4.4 Quantitative relationship between DPV and AFP
The AFP window at 365 mV correlates with log-AFP concentration with a Pearson r of −0.8577.
4.5 Quantitative relationship between DPV and CA125
The CA125 window at 968 mV correlates with log-CA125 concentration with a Pearson r of −0.8367.
4.6 Single-biomarker regression
Each biomarker was predicted from the full fingerprint with every model in the hierarchy, using out-of-fold predictions. XGBoost was the best single-biomarker decoder, with R² of 0.881 for PSA, 0.837 for AFP, and 0.872 for CA125. The linear and PLS baselines were close behind, which is consistent with the signal being largely linear in log-concentration. The 1D-CNN performed far worse on this dataset, which is expected because deep networks need much more data than the 1,000 samples available here. Table 3 reports the full single-biomarker comparison. Predicted-versus-measured and residual plots for each biomarker are shown in Figures 5, 6, and 7.

Table 3: Single-biomarker regression performance (R², out-of-fold)
Model   PSA   AFP   CA125
Linear  0.845 0.788 0.852
PLS     0.845 0.788 0.852
Random Forest 0.875 0.830 0.853
SVR     0.862 0.810 0.868
XGBoost 0.881 0.837 0.872
MLP     0.837 0.778 0.791
1D-CNN  0.326 0.204 0.354
[Figure 5 placeholder: Predicted vs measured PSA — to be inserted]
[Figure 6 placeholder: Predicted vs measured AFP — to be inserted]
[Figure 7 placeholder: Predicted vs measured CA125 — to be inserted]
4.7 Multi-output simultaneous prediction
The central experiment used one fingerprint to predict all three biomarkers at once. The multi-output results, shown in Table 4 and Figure 8, are essentially the same as the single-biomarker results: XGBoost reached R² of 0.881, 0.837, and 0.872 for PSA, AFP, and CA125. This means decoding all three biomarkers together does not degrade accuracy, and the multiplexing information is genuinely present in the raw fingerprint.

Table 4: Multi-output regression performance (R², one fingerprint → PSA + AFP + CA125)
Model   PSA   AFP   CA125
Linear  0.845 0.788 0.852
PLS     0.845 0.788 0.852
Random Forest 0.875 0.830 0.853
SVR     0.862 0.810 0.868
XGBoost 0.881 0.837 0.872
MLP     0.837 0.778 0.791
1D-CNN  0.306 0.214 0.200
[Figure 8 placeholder: Multi-output prediction performance — to be inserted]
4.8 Raw fingerprint versus engineered features
Three input representations were compared: the full 200-point fingerprint, the six engineered features, and the three experimentally validated regions only. The results are shown in Table 5 and Figure 9. The full fingerprint performed best overall, with R² of 0.845, 0.788, and 0.852 for PSA, AFP, and CA125. The engineered features worked well for PSA (R² 0.833) but almost completely failed for AFP (0.336) and CA125 (0.004). The regions-only representation reached 0.696, 0.691, and 0.743. The full fingerprint therefore contains information beyond the six hand-crafted descriptors and slightly beyond the three named regions.

Table 5: Raw DPV versus engineered features (R², Linear)
Representation  PSA   AFP   CA125
Full DPV (200 points)  0.845 0.788 0.852
Engineered features (6) 0.833 0.336 0.004
Validated regions only 0.696 0.691 0.743
[Figure 9 placeholder: Raw DPV vs engineered feature performance — to be inserted]
4.9 Ablation analysis
An ablation study removed potential regions one at a time to see how much information each contributes. Using a single biomarker region gives an average R² of about 0.22. Adding a second region roughly doubles this to about 0.46, and all three regions together reach 0.71. The full 200-point fingerprint reaches 0.83. The results, in Table 6 and Figure 10, show that the three experimentally validated regions carry most of the decodable information, and that the remaining points add a smaller but real amount.

Table 6: Ablation results (average R² across biomarkers)
PSA region only   0.218
AFP region only   0.217
CA125 region only 0.235
PSA + AFP        0.451
PSA + CA125      0.466
AFP + CA125      0.471
All three regions 0.710
Full 200-point fingerprint 0.828
[Figure 10 placeholder: Ablation analysis — to be inserted]
4.10 Cross-biomarker interference analysis
Because each sample varies PSA, AFP, and CA125 independently, a controlled check tested whether one biomarker distorts the reading of another. Each target was predicted from its own region combined with one other region, giving R² values between 0.68 and 0.74 depending on the pair. These values sit between the single-region and three-region results, which means the regions add information independently rather than corrupting one another. The three biomarkers can therefore be decoded from one scan without strong cross-interference.
4.11 Explainable AI
SHAP was used on the best regression model to find which voltage steps matter for each biomarker. For PSA the highest-importance potentials are −458, −468, and −448 mV; for AFP they are 365, 375, and 385 mV; and for CA125 they are 968, 978, and 988 mV. These fall inside the experimentally established redox windows, as shown in Figure 11. The AI models therefore use information located in the same regions the experimental team identified. This is consistent with the sensing response established experimentally; SHAP itself does not prove the chemical mechanism.
[Figure 11 placeholder: SHAP/feature importance mapped against potential — to be inserted]
4.12 PSA-threshold classification as a secondary application
As a secondary result, the PSA concentration was converted into high and low classes using the predefined experimental threshold of 4,000 pg·mL⁻¹. This produced 881 low-risk (88.1%) and 119 high-risk (11.9%) samples. The classification results are reported in Table 7 and the ROC analysis in Figure 12. Logistic Regression was the strongest classifier with a ROC-AUC of 99.75%, a PR-AUC of 98.17%, and an F1-score of 91.20%. At the optimized threshold of 0.45, recall reached 98.32%. These numbers show the DPV signal is strongly associated with the predefined PSA threshold, but they do not demonstrate clinical prostate-cancer diagnosis, which would require clinical ground-truth data that this dataset does not contain.

Table 7: Secondary PSA-threshold classification performance (%, out-of-fold)
Model   Accuracy Precision Recall F1-Score ROC-AUC PR-AUC
Logistic Regression 97.80 87.02 95.80 91.20 99.75 98.17
Random Forest   96.90 92.31 80.67 86.10 99.22 94.34
SVM 97.30 83.82 95.80 89.41 99.63 97.37
XGBoost 97.80 90.08 91.60 90.83 99.61 97.63
1D-CNN  90.10 55.00 92.44 68.97 97.30 83.03
BiLSTM  88.90 52.13 82.35 63.84 92.86 75.24
Optimal threshold for Logistic Regression: 0.45; recall at threshold 98.32%, F1 0.92.
[Figure 12 placeholder: Secondary PSA-threshold ROC analysis — to be inserted]

5. Discussion
5.1 Why the multiplexed fingerprint contains quantitative information
The strongest evidence is that one fingerprint predicts all three biomarker concentrations simultaneously, and the multi-output model performs as well as three separate single-biomarker models. If the three signals were tangled together, the shared model would be worse than the dedicated ones. It is not, which means each biomarker's electrochemical information remains separable in the raw curve.
5.2 Why full DPV versus engineered features performs differently
The six engineered descriptors capture the overall peak shape but lose detail. They work for PSA, whose redox signal is large, but fail for AFP and CA125, whose signals are smaller and more subtle. The full 200-point fingerprint retains this detail, which is why it is the more robust input. This justifies using the raw signal rather than only hand-crafted features.
5.3 Whether AI can computationally resolve multiplexed information
Yes, within this dataset. XGBoost reconstructs the three log-concentrations with R² between 0.84 and 0.89 on untouched test samples, with bootstrap confidence intervals of roughly ±0.03. The accuracy is not perfect, and the remaining error is visible in the residual plots, but the quantitative signal is clearly decodable.
5.4 Which electrochemical regions are most informative
The ablation shows the three experimentally validated windows carry most of the information, and SHAP confirms the models concentrate on exactly these potentials. The full fingerprint adds a small but consistent improvement, so there is some useful information outside the named windows. Importantly, the AI did not invent new peak assignments; it used the regions the experimental team already established.
5.5 Advantages of a single-scan multiplexed measurement
A single DPV scan carries PSA, AFP, and CA125 information without strong cross-interference, as the interference analysis shows. This is the practical value of the multiplexed biosensor: one fast, low-cost measurement could support a multi-biomarker readout in one step, subject to the limitations below and to proper validation of equivalence with established laboratory assays.
5.6 Limitations
The dataset is experimental spiked serum, not clinical samples, so no clinical diagnostic claim is made. The 1D-CNN and MLP did not match XGBoost, and with 1,000 samples that is expected; it does not mean deep learning is inferior in general. The regression targets were log-transformed concentrations, so the reported R² is for log scale. The interference check is a controlled analysis on this dataset and does not cover every possible matrix effect. SHAP identifies predictive regions but does not by itself establish electrochemistry.
5.7 What is required for future clinical validation
Before any clinical use, the approach needs independent multi-centre validation on real patient samples with confirmed clinical outcomes, including histopathology and staging. Electrode-to-electrode and batch-to-batch calibration, drift monitoring, and regulatory approval would also be required. None of that is claimed here.

6. Conclusion
This study asked whether a single multiplexed DPV fingerprint from an experimentally validated biosensor can be decoded into simultaneous quantitative estimates of PSA, AFP, and CA125. On this 1,000-sample experimental dataset, the answer is yes.
The evidence comes from several independent analyses that agree with each other. First, the DPV signal carries quantitative biomarker information at the experimentally established redox windows: the correlations with log-concentration are r = −0.86 for PSA at −468 mV, r = −0.86 for AFP at 365 mV, and r = −0.84 for CA125 at 968 mV. Second, quantitative decoding works: XGBoost reached cross-validation R² values of 0.881, 0.837, and 0.872 for PSA, AFP, and CA125, and these held on a strict 20% hold-out test set at 0.888, 0.853, and 0.870, with bootstrap 95% confidence intervals of roughly ±0.03. Third, the multiplexing information genuinely survives in one scan: a single multi-output model decoded all three biomarkers together with the same R² as three separate single-biomarker models, and the cross-biomarker interference check (R² between 0.68 and 0.74 for region pairs) shows the three signals add information independently rather than corrupting each other.
The analyses also establish what the fingerprint information is and where it lives. The ablation study shows that the three experimentally validated regions carry most of the decodable information, while the full 200-point fingerprint adds a smaller but real amount on top. The raw full fingerprint also outperforms the six engineered descriptors, which work for PSA but almost completely fail for AFP and CA125. SHAP analysis shows the models concentrate their predictions on the same potential regions the experimental team established (PSA near −458 to −448 mV, AFP near 365 to 385 mV, CA125 near 968 to 988 mV), meaning the AI reads the same electrochemical signal the biosensor was designed to produce. The 1D-CNN underperformed on this dataset, which is expected with only 1,000 samples and confirms that a simpler, well-chosen model is the better decoder here.
As a secondary application, the decoded PSA supports classification according to the predefined experimental threshold of 4,000 pg·mL⁻¹. Logistic Regression reached a ROC-AUC of 99.75%, a PR-AUC of 98.17%, and an F1-score of 91.20%. These numbers show the DPV signal is strongly associated with the predefined PSA threshold, but they are not a clinical prostate-cancer diagnosis, which would require clinical ground-truth data this dataset does not contain.
The overall result is that AI can decode a multiplexed electrochemical fingerprint quantitatively without destroying the scientific meaning of the biosensor: the models use the same potential regions the experimental team validated, and they reconstruct the three biomarker concentrations with enough accuracy that the multiplexing information is preserved in one scan. This is the strongest possible outcome for the system, and it provides a clear basis for future work with real clinical samples.

References
Akinwumi, P. O., Ojo, S., Nathaniel, T. I., Wanliss, J., Karunwi, O., & Sulaiman, M. (2025). Evaluating machine learning models for stroke prediction based on clinical variables. Frontiers in Neurology, 16(September). https://doi.org/10.3389/fneur.2025.1668420
Alyamni, N., Abot, J. L., & Zestos, A. G. (2024). Perspective—Advances in Voltammetric Methods for the Measurement of Biomolecules. ECS Sensors Plus, 3(2). https://doi.org/10.1149/2754-2726/ad3c4f
Calza-Metre, M., & Borzì, L. (2026). Machine learning-based automatic stress detection: Performance and generalization across datasets. Smart Health, 41(March), 100693. https://doi.org/10.1016/j.smhl.2026.100693
Chavan, S. G., Rathod, P. R., Koyappayil, A., Hwang, S., & Lee, M. H. (2025). Recent advances of electrochemical and optical point-of-care biosensors for detecting neurotransmitter serotonin biomarkers. Biosensors and Bioelectronics, 267(September 2024), 116743. https://doi.org/10.1016/j.bios.2024.116743
Dong, T., Matos Pires, N. M., Yang, Z., & Jiang, Z. (2023). Advances in Electrochemical Biosensors Based on Nanomaterials for Protein Biomarker Detection in Saliva. Advanced Science, 10(6). https://doi.org/10.1002/advs.202205429
Gao, L., Feng, J., Gao, Y., Luo, L., Jiang, H., Yang, Q., Lu, J., & Guo, L. (2025). XGBoost-based model for predicting PICC occlusion risk in cancer patients: Insights from SHAP analysis. Alexandria Engineering Journal, 123(March), 436–447. https://doi.org/10.1016/j.aej.2025.03.089
Gogoshin, G., & Rodin, A. S. (2023). Graph Neural Networks in Cancer and Oncology Research: Emerging and Future Trends. Cancers, 15(24). https://doi.org/10.3390/cancers15245858
Hunt, A., & Slaughter, G. (2025). Electrochemical Detection of Prostate Cancer—Associated miRNA-141 Using a Low-Cost Disposable Biosensor. Biosensors, 15(6), 1–13. https://doi.org/10.3390/bios15060364
Jafari, E., Dadgar, H., Zarei, A., Samimi, R., Manafi-Farid, R., Divband, G. A., Nikkholgh, B., Fallahi, B., Amini, H. R., Ahmadzadehfar, H., Keshavarz, A., & Assadi, M. (2024). The role of [68Ga]Ga-PSMA PET/CT in primary staging of newly diagnosed prostate cancer: predictive value of PET-derived parameters for risk stratification through machine learning. Clinical and Translational Imaging, 12(6), 669–682. https://doi.org/10.1007/s40336-024-00666-9
Lee, S. J., Yu, S. H., Kim, Y., Kim, J. K., Hong, J. H., Kim, C. S., Seo, S. Il, Byun, S. S., Jeong, C. W., Lee, J. Y., & Choi, I. Y. (2020). Prediction system for prostate cancer recurrence using machine learning. Applied Sciences (Switzerland), 10(4), 1–9. https://doi.org/10.3390/app10041333
Mitchell B. Max, M.D., Sue A. Lynch, M.D., Joanne Muir, R.N., M.S., Susan E. Shoaf, Ph.D., Bruce Smoller, M.D., and Ronald Dubner, D.D.S., Ph. D. (1993). The New England Journal of Medicine is produced by NEJM Group, a division of the Massachusetts Medical Society. The New England Journal of Medicine, 326(19), 1250–1256.
Passaro, A., Al Bakir, M., Hamilton, E. G., Diehn, M., André, F., Roy-Chowdhuri, S., Mountzios, G., Wistuba, I. I., Swanton, C., & Peters, S. (2024). Cancer biomarkers: Emerging trends and clinical implications for personalized treatment. Cell, 187(7), 1617–1635. https://doi.org/10.1016/j.cell.2024.02.041
Paul, S. G., Saha, A., Hasan, Z., Rashed, S., Noori, H., & Moustafa, A. (2024). A Systematic Review of Graph Neural Network in Healthcare-Based Applications: Recent Advances, Trends, and Future Directions. IEEE Access, 12(January), 15145–15170. https://doi.org/10.1109/ACCESS.2024.3354809
Qiu, Y., Liu, W., Wang, J., & Li, R. (2024). PAGE: Parametric Generative Explainer for Graph Neural Network. Frontiers in Artificial Intelligence and Applications, 392, 858–865. https://doi.org/10.3233/FAIA240572
Roy-chowdhuri, S., Passaro, A., Bakir, M. Al, Hamilton, E. G., Diehn, M., Andre, F., Mountzios, G., Wistuba, I. I., Swanton, C., & Peters, S. (2024). ll Cancer biomarkers: Emerging trends and clinical implications for personalized treatment. https://doi.org/10.1016/j.cell.2024.02.041
Westerlinck, P. (2025). Comparative Analysis of Predictive Models for Individual Cancer Risk: Approaches and Applications. Onco, 5(2), 1–18. https://doi.org/10.3390/onco5020029
Ye, C., Liang, D., Ruan, Y., Lin, X., Yu, Y., Nan, R., Yi, Y., & Sun, W. (2021). Photonic crystal barcode: An emerging tool for cancer diagnosis. Smart Materials in Medicine, 2(June), 182–195. https://doi.org/10.1016/j.smaim.2021.06.003
York, N. (1981). Pulse voltammetric methods of analysis. 6, 315–326.
