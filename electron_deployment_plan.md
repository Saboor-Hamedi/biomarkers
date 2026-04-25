# 🧬 XAI Biomarker Hub: Electron Deployment Blueprint

## 1. Project Objective

Transition the existing Python-based Cancer Biomarker Research Hub into a professional, standalone **Electron Desktop Application**. The goal is to provide a premium, industrial-grade interface that combines high-performance React frontend with the robust GNN/XGBoost analytical engine.

## 2. Technical Stack

- **Shell**: Electron (v30+)
- **Frontend**: React + Vite + Tailwind CSS
- **Analytical Engine**: Python 3.11+ (FastAPI Backend)
- **State Management**: Zustand (for clinical session persistence)
- **AI Committee**: 5-Model Ensemble (GNN, XGBoost, Random Forest, SVM, Logistic Regression)

## 3. UI/UX Design System (Industrial Grade)

- **Aesthetic**: Dark Clinical Mode (#0e1117 background).
- **Metrics**: High-contrast **Pure White Cards** with dark text for maximum diagnostic readability.
- **Layout**:
  - **Sidebar**: Collapsible navigation for "Individual Audit", "Batch Forensic Audit", and "Artifact Gallery".
  - **Header**: Real-time system status (Model Calibration, Dataset Scope, Forensic Mode).
- **Sanitization**: Strict professional tone (No emojis, no decorative clutter).

## 4. Integration Roadmap

### Phase 1: Environment Setup

1.  Install Tailwind CSS and PostCSS.
2.  Configure `electron-vite` to handle Python subprocesses.

### Phase 2: The "Brain" (Backend Bridge)

1.  Create `server/main.py` using FastAPI.
2.  Expose `/predict` and `/audit` endpoints.
3.  Load `.pkl` artifacts (including GNN dictionary unpacking logic).
4.  Implement `np.log1p` and `RobustScaler` synchronization.

### Phase 3: The "Face" (Frontend Dashboard)

1.  Build **Metric Tiles** (Risk Score, Prediction, Consensus).
2.  Build **AI Committee Peer Review** (Live multi-model status).
3.  Build **Sensitivity Simulator** (What-If analysis).
4.  Build **Batch Audit Table** (Top 50 high-risk clinical registry).

### Phase 4: Artifact Gallery

1.  Map static PNGs (ROC, Feature Importance, Trajectory) into React components.
2.  Implement high-fidelity plot rendering using Recharts/D3 if dynamic plots are requested.

## 5. Deployment Parity

Ensure that results in the Electron app match the **analysis.ipynb** (70/30 split, 1000 records) with 100% mathematical precision.

---

**Status**: Ready for initialization in the next session.
