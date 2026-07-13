# DPV Voltammetry Data Pipeline

Loads, parses, and preprocesses Differential Pulse Voltammetry data from Excel for model training and inference.

## Source File

`server/analysis/data/Raw_data_dpv.xlsx`

## Excel Structure

### `Target_Concentrations` Sheet

Metadata sheet with columns:

| Column | Description |
|--------|-------------|
| `Sample_ID` | Patient identifier (`Sample_0001` … `Sample_1000`) |
| `PSA` | PSA concentration (pg/mL) |
| `AFP` | AFP concentration |
| `CA125` | CA125 concentration |
| `CA19_9` | CA19-9 concentration |
| `Label` | 1 = high risk, 0 = low risk |

**Risk threshold**: `Label = 1` when PSA > 4000 pg/mL.

### `Sample_*` Sheets

Each of the 1000 patient sheets contains columns:

| Column | Description |
|--------|-------------|
| `Potential (V)` | Applied potential step |
| `Current (µA)` | Measured current at this potential |
| … (V0–V199) | 200 potential steps per patient |

## Processing (`analyse_data.py`)

### 1. Sheet Discovery

```python
excel_file = pd.ExcelFile(path)
sample_sheets = [s for s in excel_file.sheet_names if s != "Target_Concentrations"]
```

### 2. Feature Extraction

For each sample sheet, extract the full `Current (µA)` column → 200 values per patient → stored into feature columns `V0` through `V199`.

```python
def load_dpv_data(excel_path):
    excel_file = pd.ExcelFile(excel_path)
    sample_sheets = [s for s in excel_file.sheet_names if s != "Target_Concentrations"]
    data = []
    for sheet in sample_sheets:
        df = excel_file.parse(sheet)
        currents = df["Current (µA)"].values[:200]
        data.append(currents)
    return np.array(data)  # shape: (n_samples, 200)
```

### 3. Target Alignment

Target labels are extracted from `Target_Concentrations` sheet, ordered by Sample_ID.

### 4. Train/Test Split

- 70% training, 30% holdout test
- Training set further split: 80% train / 20% validation
- Final shapes: (560, 200) train, (140, 200) val, (300, 200) test

### 5. Feature Scaling

`RobustScaler` — uses median and IQR, less sensitive to outliers.

```python
scaler = RobustScaler()
X_train_scaled = scaler.fit_transform(X_train)
```

Scaler saved to `models/scaler.pkl`.

### 6. Feature Columns

List of column names `["V0", "V1", …, "V199"]` saved to `models/feature_columns.pkl`.

## Server-Side Inference

In `server/main.py`:

```python
def load_dpv_data():
    """Returns (sample_ids, X, target_labels, concentrations)"""

def preprocess_for_inference(sample_id):
    """Loads single patient's DPV currents, scales, returns DataFrame."""
```

The endpoint `POST /predict` accepts `{"sample_id": "Sample_0001"}`, looks up the patient's 200 DPV values from the Excel file, scales them, and runs all 5 models.
