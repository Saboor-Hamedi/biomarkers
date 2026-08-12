STUDENT INSTRUCTION DOCUMENT
AI Analysis of a Multiplexed Electrochemical Biosensor
Required correction and step-by-step research plan
Working title
AI-Assisted Decoding of Multiplexed Electrochemical Fingerprints for Simultaneous Quantification of PSA, AFP, and CA125 in Serum
________________________________________
1. First understand what experiment you are analysing
The experimental/material-science team has already:
• fabricated the electrode;
• characterized the material/electrode;
• established the electrochemical behaviour;
• established the multiplexed sensing system;
• generated the experimental dataset;
• prepared 1,000 serum samples with known PSA, AFP and CA125 concentrations;
• recorded a DPV response for each sample.
The dataset contains approximately:
1,000 samples × 200 electrochemical current measurements
over the experimentally defined potential range.
The existing report itself describes the dataset this way.
Your role
Your role is to answer:
Can artificial intelligence computationally decode the information contained in the experimentally generated multiplexed DPV fingerprint?
You are NOT being asked to redesign the biosensor.
You are NOT being asked to invent a new electrochemical mechanism.
You are NOT being asked to claim clinical prostate-cancer diagnosis.
________________________________________
2. Understand the basic biosensor architecture before coding
You must be able to explain:
Biological target
PSA / AFP / CA125
↓
Recognition/sensing event
Established experimentally by the biosensor team
↓
Electrochemical response
The electrode produces a current response as a function of applied potential.
↓
DPV
Differential pulse voltammetry generates a current–potential curve.
↓
Digital fingerprint
The measured curve is represented numerically by approximately 200 current values.
↓
AI
The AI model learns relationships between the electrochemical fingerprint and the experimentally known biomarker concentrations.
This is the architecture of the project.
________________________________________
3. Major problems in the current manuscript
The current report must NOT be submitted in its present form.
Problem 1 — It repeatedly calls experimental serum samples “patients”
The report says:
“1,000 patients”
and refers to “high-risk patients.”
This is scientifically incorrect for the current dataset.
The report itself states that the serum samples were prepared by spiking serum with known amounts of PSA, AFP and CA125.
Therefore use:
1,000 experimental serum samples
not:
1,000 patients.
________________________________________
4. Problem 2 — The present study does NOT demonstrate prostate-cancer diagnosis
The current report converts PSA concentration into:
high risk vs low risk
using 4 ng/mL.
It then describes the resulting classification as prostate-cancer risk classification.
This is too strong.
The dataset contains known PSA concentrations.
It does NOT contain:
• biopsy-confirmed prostate cancer status;
• histopathology;
• cancer/no-cancer clinical diagnosis;
• patient clinical outcome;
• BPH status;
• Gleason score;
• clinical staging.
Therefore:
You may report
PSA-threshold classification
You may NOT report
AI diagnosis of prostate cancer
unless appropriate clinical ground-truth data are available.
________________________________________
5. Problem 3 — The current analysis wastes the multiplexed dataset
The present manuscript makes PSA high/low classification the central endpoint.
The current report gives Logistic Regression:
ROC-AUC = 99.75%
and other models above 97%.
That is interesting, but it is NOT sufficient to represent the whole experiment.
Why?
Because the experimental system contains:
PSA + AFP + CA125
yet the principal AI target is:
PSA high/low.
That reduces a multiplexed quantitative sensing experiment to a single binary classification problem.
We must recover the multiplexing information.
________________________________________
6. Problem 4 — Do not call the AI result a “digital surrogate assay” yet
The current abstract claims that the system:
“replicates the information content of a multi-biomarker laboratory panel”
and supports decentralized point-of-care screening.
Those are very strong claims.
They are NOT demonstrated merely by a PSA classification model.
Remove these claims unless they are experimentally and computationally demonstrated.
Use:
AI-assisted decoding of multiplexed electrochemical fingerprints
instead.
________________________________________
7. Problem 5 — The current report overinterprets SHAP
The current report says that SHAP:
“mathematically confirms” the physical chemistry.
This is incorrect scientific interpretation.
SHAP can tell us:
which input features contribute to the model prediction.
SHAP does NOT independently establish:
the chemical identity or electrochemical mechanism of a peak.
The electrochemical/material-science experiments establish the mechanism.
AI can then show whether the model uses information located in those experimentally established regions.
________________________________________
8. Problem 6 — The current report assumes that high AUC proves genuine electrochemical/clinical validity
The report states that the strong performance suggests the DPV features contain genuine electrochemical markers of prostate-cancer risk.
This conclusion is too strong.
The current classification target was created directly from the known PSA concentration.
Therefore a high AUC primarily demonstrates:
the DPV signal contains information associated with the experimentally defined PSA threshold.
It does NOT demonstrate:
prostate cancer risk in real patients.
________________________________________
9. Problem 7 — The current report is overly focused on model competition
The report compares:
• Logistic Regression
• Random Forest
• SVM
• XGBoost
• 1D-CNN
• BiLSTM
and emphasizes which gives the highest AUC.
This is not enough for a high-level biosensor paper.
The scientific question is not:
Which AI model gives the biggest number?
The scientific question is:
What electrochemical information is contained in the multiplexed fingerprint, and can AI decode it reliably?
________________________________________
10. New central research question
The entire analysis must now be reorganized around:
Can a single multiplexed DPV fingerprint generated by the experimentally validated biosensor be computationally decoded into simultaneous quantitative estimates of PSA, AFP and CA125?
This is the central question.
________________________________________
PHASE 1 — DATA AUDIT
Step 1. Create a formal data dictionary
Document:
• Sample ID
• PSA concentration
• AFP concentration
• CA125 concentration
• potential values
• current values
Clearly define:
X
Electrochemical information.
Y
Known experimental biomarker concentration.
________________________________________
Step 2. Verify sample identity
Confirm:
• exactly 1,000 samples;
• unique sample IDs;
• exactly 200 current measurements per sample;
• no accidental row duplication;
• correct alignment between biomarker concentrations and DPV curves.
Save an audit report.
________________________________________
Step 3. Preserve the original data
Never modify the original CSV.
Create:
RAW_DATA
and work only on copies.
Never overwrite the experimental data.
________________________________________
PHASE 2 — UNDERSTAND THE DPV SIGNAL
Step 4. Plot raw DPV curves
Produce:
Figure A
20–50 representative individual DPV curves.
Figure B
All 1,000 curves as a transparent overlay/density representation.
Figure C
Mean ± SD DPV fingerprint.
Figure D
PSA concentration groups.
Figure E
AFP concentration groups.
Figure F
CA125 concentration groups.
The purpose is to understand the sensor before applying AI.
________________________________________
PHASE 3 — VALIDATE THE BIOMARKER INFORMATION
The experimental team has already established the electrochemical regions.
Do NOT invent new peak assignments.
Use the experimentally validated potential regions for:
• PSA
• AFP
• CA125
Then calculate:
PSA
DPV signal vs PSA concentration.
AFP
DPV signal vs AFP concentration.
CA125
DPV signal vs CA125 concentration.
Use:
• Pearson correlation;
• Spearman correlation;
• appropriate regression;
• confidence intervals;
• residual analysis.
Do not blindly assume linearity.
Because the concentration ranges span several orders of magnitude, investigate whether log concentration is scientifically appropriate for regression.
________________________________________
PHASE 4 — DO NOT THROW AWAY THE RAW SIGNAL
You have two representations:
Representation 1
Full 200-point DPV fingerprint
Representation 2
Engineered electrochemical features
The engineered feature file contains descriptors such as:
• anodic peak current;
• anodic peak potential;
• cathodic peak current;
• cathodic peak potential;
• area under curve;
• peak separation.
These are derived representations of the same electrochemical measurements.
They are NOT independent experiments.
________________________________________
PHASE 5 — AUDIT THE ENGINEERED FEATURES
For every engineered feature document:
How was it calculated?
For example:
• peak detection algorithm;
• peak window;
• baseline correction;
• numerical integration method;
• peak-separation formula;
• missing-peak handling.
Do not simply accept the feature file.
We need reproducibility.
________________________________________
PHASE 6 — CHECK MISSING FEATURES
If a peak was not detected in some samples:
DO NOT automatically replace the missing value with the mean.
First determine:
Why was the peak not detected?
Possible causes include:
• weak signal;
• true absence;
• peak overlap;
• noise;
• algorithm failure;
• baseline problem.
Document the reason.
________________________________________
PHASE 7 — PRIMARY AI TASK: QUANTITATIVE DECODING
This is now the main analysis.
Task 1 — PSA regression
Input:
DPV fingerprint
Output:
PSA concentration
Report:
• R²
• MAE
• RMSE
• Pearson r
• Spearman correlation
• predicted vs measured plot
• residual plot
________________________________________
Task 2 — AFP regression
Input:
DPV fingerprint
Output:
AFP concentration
Same metrics.
________________________________________
Task 3 — CA125 regression
Input:
DPV fingerprint
Output:
CA125 concentration
Same metrics.
________________________________________
PHASE 8 — MULTI-OUTPUT REGRESSION
This is the most important computational experiment.
Input:
one 200-point DPV fingerprint
Output:
PSA + AFP + CA125
Conceptually:
DPV fingerprint
↓
AI decoder
↓
PSA prediction
AFP prediction
CA125 prediction
This is the computational equivalent of decoding the multiplexed electrochemical signal.
________________________________________
PHASE 9 — COMPARE SIMPLE AND ADVANCED MODELS
Use a controlled hierarchy.
Baseline
1.  Linear Regression
2.  PLS Regression
Classical machine learning
3.  Random Forest
4.  SVR
5.  XGBoost
Advanced
6.  Neural Network
7.  1D-CNN
Do not assume deep learning is superior.
If a simpler model performs equally well, that is an important scientific result.
________________________________________
PHASE 10 — RAW DPV VS ENGINEERED FEATURES
This comparison is mandatory.
Model A
200 raw DPV points.
Model B
Six engineered electrochemical features.
Model C
Biomarker-specific electrochemical regions.
Compare their predictive performance.
Prepare:
Representation  Inputs  PSA AFP CA125
Full DPV  200
Engineered features 6
PSA region  selected
AFP region  selected
CA125 region  selected
This tells us whether the full fingerprint really contains additional information.
________________________________________
PHASE 11 — ABLATION STUDY
Perform:
A
PSA region only.
B
AFP region only.
C
CA125 region only.
D
PSA + AFP.
E
PSA + CA125.
F
AFP + CA125.
G
All three regions.
H
Full 200-point DPV.
This is important because it demonstrates whether the AI is genuinely using the multiplexed fingerprint.
________________________________________
PHASE 12 — TEST FOR CROSS-BIOMARKER INTERFERENCE
Because the samples contain independently varied biomarkers, investigate:
Does changing AFP alter PSA prediction?
Does changing CA125 alter PSA prediction?
Does changing PSA alter AFP prediction?
Does changing the other biomarkers alter CA125 prediction?
Create controlled interaction analyses.
This can become an important part of the multiplexing argument.
________________________________________
PHASE 13 — STRICT DATA SPLITTING
Before ANY model fitting:
Create:
Training set
Validation set
Untouched test set
The test set must never be used for:
• feature selection;
• model selection;
• hyperparameter tuning;
• threshold selection;
• preprocessing fitting.
________________________________________
PHASE 14 — PREVENT DATA LEAKAGE
All of the following must occur inside the training data/folds:
• scaling;
• normalization;
• feature selection;
• PCA;
• feature engineering where learned from data;
• hyperparameter optimization.
The current manuscript states that RobustScaler was fitted within cross-validation folds, which is good practice.
But the new analysis must maintain this discipline for every preprocessing operation.
________________________________________
PHASE 15 — USE OUT-OF-SAMPLE PREDICTIONS
Do not report only training performance.
Generate:
Out-of-fold predictions
and:
Untouched test-set predictions.
Plot:
Measured vs predicted
for each biomarker.
________________________________________
PHASE 16 — UNCERTAINTY
Do not give only one R² number.
Where appropriate report:
• confidence intervals;
• bootstrap intervals;
• prediction intervals;
• uncertainty distributions.
Show where the model performs poorly.
________________________________________
PHASE 17 — EXPLAINABLE AI
After selecting the best model:
Use SHAP or another appropriate method.
Ask:
Which voltage regions contribute to PSA prediction?
Which voltage regions contribute to AFP prediction?
Which voltage regions contribute to CA125 prediction?
Then compare the AI-important regions with the experimentally established electrochemical regions.
The correct interpretation is:
AI identifies predictive electrochemical regions consistent with experimentally established sensing responses.
Do NOT say:
SHAP proves the chemical mechanism.
________________________________________
PHASE 18 — RETURN TO THE EXISTING PSA CLASSIFICATION RESULT
Do not delete the current PSA classification result.
Move it to a secondary application.
The workflow becomes:
DPV fingerprint
↓
AI quantitative decoder
↓
PSA / AFP / CA125
↓
PSA threshold classification
The current report’s Logistic Regression ROC-AUC of 99.75% can therefore become a secondary result rather than the entire paper.
But the classification must be described accurately as:
classification according to the predefined experimental PSA threshold
not clinical prostate-cancer diagnosis.
________________________________________
PHASE 19 — BUILD THE FINAL AI-BIOSENSOR PROTOTYPE
Create a conceptual digital workflow:
INPUT
Serum sample
↓
SENSOR
Validated multiplexed electrochemical electrode
↓
MEASUREMENT
DPV
↓
DIGITAL SIGNAL
200-point current–potential fingerprint
↓
AI DECODER
Preprocessing
feature representation
trained model
↓
OUTPUT
PSA
AFP
CA125
↓
OPTIONAL DECISION LAYER
PSA threshold classification
This should be the main graphical abstract.
________________________________________
PHASE 20 — MANUSCRIPT STRUCTURE
1. Introduction
1.1
Need for multiplexed biomarker detection.
1.2
Limitations of conventional multi-analyte electrochemical interpretation.
1.3
DPV as a rich electrochemical fingerprint.
1.4
Potential of machine learning for signal decoding.
1.5
Research gap.
1.6
Our hypothesis and objective.
________________________________________
2. Experimental Section
The material-science team supplies the validated:
2.1 Electrode fabrication
2.2 Material characterization
2.3 Electrochemical characterization
2.4 Multiplexed sensing protocol
2.5 Serum preparation
2.6 PSA/AFP/CA125 concentration design
2.7 DPV acquisition
The IT student must NOT rewrite the experimental chemistry from memory.
Use the actual experimental protocol.
________________________________________
3. Dataset and AI Methodology
3.1 Dataset construction
1,000 experimental serum samples.
3.2 DPV representation
200 current measurements per sample.
3.3 Electrochemical feature extraction
Six engineered features.
3.4 Data preprocessing
3.5 Train/validation/test strategy
3.6 Regression models
3.7 Multi-output model
3.8 Classification model
3.9 Explainable AI
3.10 Statistical analysis
________________________________________
4. Results
4.1 DPV fingerprint characteristics
4.2 Biomarker-specific electrochemical response
4.3 Quantitative relationship between DPV and PSA
4.4 Quantitative relationship between DPV and AFP
4.5 Quantitative relationship between DPV and CA125
4.6 Single-biomarker regression
4.7 Multi-output simultaneous prediction
4.8 Raw fingerprint versus engineered features
4.9 Ablation analysis
4.10 Cross-biomarker interference analysis
4.11 Explainable AI
4.12 PSA-threshold classification as secondary application
________________________________________
5. Discussion
Discuss:
5.1
Why the multiplexed fingerprint contains quantitative information.
5.2
Why full DPV versus engineered features performs differently.
5.3
Whether AI can computationally resolve multiplexed information.
5.4
Which electrochemical regions are most informative.
5.5
Advantages of a single-scan multiplexed measurement.
5.6
Limitations.
5.7
What is required for future clinical validation.
________________________________________
6. Conclusion
The conclusion must focus on:
experimentally validated multiplexed electrochemical sensing + AI-assisted digital decoding.
Do not conclude:
“AI diagnoses prostate cancer”
unless genuine clinical ground truth exists.
________________________________________
PHASE 21 — FIGURES TO PRODUCE
At minimum:
Figure 1
Overall biosensor + AI workflow.
Figure 2
Representative raw DPV fingerprints.
Figure 3
Biomarker concentration versus electrochemical response.
Figure 4
Three-biomarker electrochemical fingerprint map.
Figure 5
Predicted vs measured PSA.
Figure 6
Predicted vs measured AFP.
Figure 7
Predicted vs measured CA125.
Figure 8
Multi-output prediction performance.
Figure 9
Raw DPV vs engineered feature performance.
Figure 10
Ablation analysis.
Figure 11
SHAP/feature importance mapped against potential.
Figure 12
Secondary PSA-threshold ROC analysis.
Do NOT create 20 decorative AI figures.
Every figure must answer a scientific question.
________________________________________
PHASE 22 — TABLES
Table 1
Dataset characteristics.
Table 2
Electrochemical feature definitions.
Table 3
Single-biomarker regression performance.
Table 4
Multi-output regression performance.
Table 5
Raw DPV versus engineered features.
Table 6
Ablation results.
Table 7
Secondary PSA classification performance.
________________________________________
PHASE 23 — WHAT YOU ARE NOT ALLOWED TO DO
Do NOT:
1.  Call spiked serum samples patients.
2.  Call PSA-threshold classification prostate-cancer diagnosis.
3.  Add clinical claims not supported by the dataset.
4.  Invent clinical outcomes.
5.  Invent patient information.
6.  Change experimental concentrations.
7.  Modify raw electrochemical measurements.
8.  Delete inconvenient samples without scientific justification.
9.  Fill missing values blindly.
10. Use biomarker concentrations as model inputs when predicting those same biomarkers.
11. Use the test set for model development.
12. Select the best result and hide weaker results.
13. Claim that deep learning is superior without statistical evidence.
14. Claim SHAP proves chemical mechanism.
15. Claim AI discovered the experimentally established electrochemical peaks.
16. Present derived features as independent experimental measurements.
17. Call the model a clinical diagnostic device.
18. Claim equivalence to ELISA or laboratory diagnostics unless experimentally demonstrated.
19. Use “digital surrogate assay” unless the required validation is actually demonstrated.
20. Optimize the model simply to obtain a larger AUC.
________________________________________
PHASE 24 — REQUIRED STUDENT DELIVERABLES
Before the manuscript is rewritten, submit these files:
Deliverable 1
Data audit report.
Deliverable 2
Data dictionary.
Deliverable 3
Raw DPV visualization package.
Deliverable 4
Electrochemical feature-generation documentation.
Deliverable 5
Biomarker-specific correlation analysis.
Deliverable 6
PSA regression.
Deliverable 7
AFP regression.
Deliverable 8
CA125 regression.
Deliverable 9
Multi-output PSA/AFP/CA125 regression.
Deliverable 10
Raw DPV versus engineered-feature comparison.
Deliverable 11
Ablation study.
Deliverable 12
Cross-biomarker interference analysis.
Deliverable 13
Strict cross-validation results.
Deliverable 14
Untouched test-set results.
Deliverable 15
Uncertainty/confidence analysis.
Deliverable 16
Explainable-AI analysis.
Deliverable 17
Secondary PSA-threshold classification.
Deliverable 18
Final figures.
Deliverable 19
Final tables.
Deliverable 20
Rewritten Results.
Deliverable 21
Rewritten Discussion.
Deliverable 22
Final manuscript.
________________________________________
THE MOST IMPORTANT RULE
Do not optimize for the highest AI score.
Optimize for the strongest scientific explanation.
The final paper must allow a reader to follow this chain:
REAL BIOSENSOR
↓
REAL ELECTROCHEMICAL RESPONSE
↓
REAL MULTIPLEXED DPV FINGERPRINT
↓
DIGITAL REPRESENTATION
↓
AI DECODING
↓
BIOMARKER QUANTIFICATION
↓
SECONDARY DECISION/CLASSIFICATION
That is the research.
Not:
CSV → Random Forest → 99% → prostate cancer.
________________________________________
GO/NO-GO CHECKPOINTS
Do not proceed automatically from one phase to the next.
CHECKPOINT 1
Can you explain the DPV data correctly?
YES → continue
NO → study electrochemical fundamentals
________________________________________
CHECKPOINT 2
Can you reproduce every engineered feature?
YES → continue
NO → stop
________________________________________
CHECKPOINT 3
Can the DPV fingerprint predict each biomarker quantitatively?
YES → continue
NO → investigate why
________________________________________
CHECKPOINT 4
Can one fingerprint predict PSA + AFP + CA125 simultaneously?
YES → this becomes the central result
NO → report honestly and determine whether the paper should focus on classification/fingerprint recognition instead
________________________________________
CHECKPOINT 5
Does full DPV outperform engineered features?
YES → strong evidence for fingerprint-based AI
NO → engineered electrochemical descriptors may be sufficient and should be emphasized
________________________________________
CHECKPOINT 6
Do AI-important voltage regions agree with experimentally established sensing regions?
YES → strong integrated electrochemical + AI interpretation
NO → investigate before making mechanistic claims
________________________________________
CHECKPOINT 7
Does performance remain strong on untouched samples?
YES → credible computational validation
NO → investigate generalization/overfitting
________________________________________
FINAL SCIENTIFIC MESSAGE
The experimental team has already created the difficult part:
a multiplexed electrochemical sensing system and a large 1,000-sample experimental dataset.
Your job is not to make the paper look more “AI.”
Your job is to demonstrate rigorously that:
AI can decode the experimentally generated electrochemical fingerprint without destroying the scientific meaning of the biosensor.
If you do that properly, the resulting work will be much stronger than a simple AI classification paper.
