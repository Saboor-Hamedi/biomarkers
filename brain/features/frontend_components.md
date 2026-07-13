# Frontend Components

React-based UI rendered inside the Electron window.

## Component Tree

```
App.jsx
├── Sidebar (ArtifactPicker)
├── Main Panel
│   ├── ForensicInput.jsx
│   ├── StatCards
│   │   ├── RiskScoreCard
│   │   ├── PredictionCard
│   │   └── ConsensusCard
│   ├── Visual.jsx
│   │   └── visual/
│   │       ├── AnalyticView.jsx       ← Reusable wrapper
│   │       ├── Counterfactual.jsx
│   │       ├── Heatmap.jsx
│   │       ├── Boundaries.jsx
│   │       ├── Shap.jsx
│   │       ├── Trajectory.jsx
│   │       ├── Roc.jsx
│   │       ├── Pr.jsx
│   │       ├── Calibration.jsx
│   │       ├── Cm.jsx
│   │       ├── Tsne.jsx
│   │       ├── Importance.jsx
│   │       ├── Distribution.jsx
│   │       └── CalibrationRiskDist.jsx
│   └── [other panels]
└── Navigation
    ├── Audit Panel
    ├── Artifact Management
    └── Settings
```

## `App.jsx`

Root component (~625 lines). Central state management.

**State**:
```javascript
const [inputs, setInputs] = useState({ sample_id: 'Sample_0001' });
const [predictResult, setPredictResult] = useState(null);
const [loading, setLoading] = useState(false);
const [trajectoryData, setTrajectoryData] = useState(null);
const [shapData, setShapData] = useState(null);
const [counterfactualData, setCounterfactualData] = useState(null);
const [activeView, setActiveView] = useState('audit');
```

**Key Handlers**:
| Handler | Action |
|---------|--------|
| `handlePredict()` | POST `/predict` + `/trajectory` + `/shap` + `/counterfactual` |
| `handleInputChange(name, value)` | Updates `inputs` state |
| `handleReset()` | Calls IPC `reset-artifacts` (no-op) |
| `onPurge()` | Calls IPC `reset-artifacts` (no-op) |
| `handleSync()` | Calls IPC `sync-artifacts` |
| `handleConfirm()` | Opens confirm dialog via IPC |

**API Calls**: All target `http://localhost:8001/{endpoint}`.

**Import**: `import Visual from './components/Visual'` (use `<Visual>` in JSX)

## `ForensicInput.jsx`

Single-patient query interface.

```jsx
const ForensicInput = ({ inputs, onInputChange, onPredict, loading }) => (
  <div>
    <input
      type="text"
      value={inputs.sample_id}
      onChange={(e) => onInputChange('sample_id', e.target.value)}
      placeholder="Sample ID (e.g., Sample_0001)"
    />
    <button onClick={onPredict} disabled={loading}>
      {loading ? 'Analyzing...' : 'Audit'}
    </button>
  </div>
);
```

- Text input (not a dropdown) for `sample_id`
- "Audit" button triggers prediction

## `ArtifactPicker.jsx`

Sidebar/navigation panel for selecting analytical views.

**Modes**: Tabs/categories for different artifact types (models, visualizations, reports).

## `Visual.jsx`

Container component (251 lines) that replaces the old monolithic `VisualAnalytics.jsx` (1508 lines).

**Props**: Receives all analytic data from `App.jsx`.

**Responsibilities**:
- Prepares all `useMemo` table data (shapTableData, trajectoryModels, rocTableData, prTableData, cmTableData, tsneTableData, importanceTableData, etc.)
- Delegates rendering to 14 sub-components in `visual/`

**Sub-component invocation**:
```jsx
return (
  <>
    <Counterfactual activeTab={activeTab} counterfactualData={counterfactualData} />
    <Heatmap activeTab={activeTab} heatmapData={heatmapData} />
    <Boundaries activeTab={activeTab} boundariesData={boundariesData} />
    <Shap activeTab={activeTab} shapData={shapData} shapTableData={shapTableData} />
    <Trajectory activeTab={activeTab} trajectoryData={trajectoryData} ... />
    <Roc activeTab={activeTab} rocTableData={rocTableData} metrics={metrics} />
    <Pr activeTab={activeTab} prTableData={prTableData} metrics={metrics} />
    <Calibration activeTab={activeTab} calibrationTableData={calibrationTableData} metrics={metrics} />
    <Cm activeTab={activeTab} metrics={metrics} cmTableData={cmTableData} />
    <Tsne activeTab={activeTab} tsneData={tsneData} ... />
    <Importance activeTab={activeTab} ... />
    <Distribution activeTab={activeTab} distributionData={distributionData} ... inputs={inputs} />
    <CalibrationRiskDist activeTab={activeTab} calibrationRiskData={calibrationRiskData} />
  </>
)
```

## `visual/` — View Components

### `AnalyticView.jsx`

Reusable wrapper used by all 14 sub-components.

**Props**: `title`, `icon`, `explanation`, `children`, `tableData`, `columns`, `extraAction`.

Layout: Header with title/icon/explanation → grid split (8/4): chart on left, table on right.

### Sub-component reference

| Component | Tab key | Depends on |
|-----------|---------|------------|
| `Counterfactual` | `counterfactual` | `counterfactualData` |
| `Heatmap` | `heatmap` | `heatmapData` |
| `Boundaries` | `boundaries` | `boundariesData` |
| `Shap` | `shap` | `shapData`, `shapTableData` |
| `Trajectory` | `trajectory` | `trajectoryData`, `trajectoryModels`, `trajectoryColors`, `trajectoryTableData` |
| `Roc` | `roc` | `rocTableData`, `metrics` |
| `Pr` | `pr` | `prTableData`, `metrics` |
| `Calibration` | `calibration` | `calibrationTableData`, `metrics` |
| `Cm` | `cm` | `metrics`, `cmTableData` |
| `Tsne` | `tsne` | `tsneData`, `tsneSubsets`, `tsneView`, `tsneTableData`, `setTsneView` |
| `Importance` | `importance` | `importancePrimaryModel`, `importanceTableData`, `importanceChartData` |
| `Distribution` | `distribution` | `distributionData`, `distributionTableData`, `distEntries`, `inputs` |
| `CalibrationRiskDist` | `calibration-risk` | `calibrationRiskData` |

### Notable implementations

- **Tsne.jsx** (357 lines) — Largest component. Two views via `tsneView` state: "standard" (single scatter) and "audit" (4 plots: t-SNE by ground truth, by model prediction, PCA, risk topography).
- **CalibrationRiskDist.jsx** (316 lines) — 4-panel dashboard: calibration curves, risk distribution histogram, threshold optimization (precision/recall/F1 sweep), risk stratification summary bars.
- **Distribution.jsx** — Density plots with patient reference marker line. `inputs` prop used to draw a pulsing red marker at patient's value.

## `StatCards`

Three stat cards below the input showing single-patient results:

1. **Risk Score** — Mean probability across models (e.g., "0.87")
2. **Prediction** — Binary {0, 1} from `/predict`
3. **Consensus** — Majority vote across 5 models

## Data Flow

```
User types sample_id
  ↓
Clicks "Audit"
  ↓
App.jsx: handlePredict()
  ↓
POST http://localhost:8001/predict     ← model predictions
POST http://localhost:8001/trajectory  ← trajectory data
POST http://localhost:8001/shap        ← SHAP values
POST http://localhost:8001/counterfactual ← counterfactual
  ↓
setPredictResult(response)
setTrajectoryData(response)
setShapData(response)
setCounterfactualData(response)
  ↓
Visual re-renders (passes data to sub-components)
StatCards update with risk_score / prediction / consensus
```

## Notes

- `inputs` is destructured without `= {}` default in both `Visual.jsx:60` and `Distribution.jsx:7` — safe because `App.jsx` always passes the object.
