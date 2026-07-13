# Graph Neural Network Model

A PyTorch-based Graph Neural Network that models DPV feature correlations as a graph for classification.

## Source File

`server/analysis/src/gnn_model.py`

## Architecture

```
Input: 200 DPV features per patient
            │
      ┌─────┴─────┐
      │  GNNConv   │  ← Edge_index defines which features are connected
      │  (64 dim)  │
      └─────┬─────┘
            │
      ┌─────┴─────┐
      │  GNNConv   │
      │  (32 dim)  │
      └─────┬─────┘
            │
      ┌─────┴─────┐
      │  GNNConv   │
      │  (16 dim)  │
      └─────┬─────┘
            │
      ┌─────┴─────┐
      │  Linear   │
      │  (1 dim)  │
      └─────┬─────┘
            │
        Sigmoid
            │
        Binary output
```

## Class: `TorchGNN`

Defined at module level inside `try/except ImportError` to handle missing PyTorch.

```python
class TorchGNN(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, edge_index):
        self.conv1 = GNNConv(in_channels, hidden_channels)
        self.conv2 = GNNConv(hidden_channels, hidden_channels // 2)
        self.conv3 = GNNConv(hidden_channels // 2, out_channels)
        self.lin = nn.Linear(out_channels, 1)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index).relu()
        x = self.conv2(x, edge_index).relu()
        x = self.conv3(x, edge_index).relu()
        x = self.lin(x)
        return torch.sigmoid(x).squeeze()
```

## Graph Construction

The edge_index defining which features connect is computed from **correlation**:

```python
corr = np.corrcoef(X_train_scaled.T)  # (200, 200) correlation matrix
edge_index = np.argwhere(np.abs(corr) > 0.3)  # edges where |r| > 0.3
# Convert to torch.LongTensor of shape (2, num_edges)
```

This creates a graph where highly-correlated potential steps share edges.

## Training: `train_gnn()`

```python
def train_gnn(X_train, y_train, X_val, y_val, feature_names):
    # Convert data to PyTorch tensors
    # Build correlation-based edge_index
    # Create TorchGNN instance
    # Train with Adam optimizer + BCE loss
    # Return (results_dict, model, edge_index)
```

| Hyperparameter | Value |
|---------------|-------|
| Epochs | 200 |
| Optimizer | Adam (lr=0.01) |
| Loss | BCEWithLogitsLoss |
| Early stopping | Patience = 20 |

## Pickle Serialization

Saved as `models/gnn_model.pkl` with dict:

```python
{
    "model": <TorchGNN>,
    "feature_names": ["V0", …, "V199"],
    "edge_index": <torch.LongTensor (2, E)>
}
```

## Loading for Inference

In `server/main.py`:

```python
import sys
sys.path.insert(0, os.path.join(ANALYSIS_DIR, "src"))
from gnn_model import TorchGNN

with open("gnn_model.pkl", "rb") as f:
    data = pickle.load(f)
    gnn_model = data["model"]
    edge_index = data["edge_index"]
```

> ⚠️ `TorchGNN` class must be importable at module level for pickle to deserialize. Solution: combine with `sys.path.insert` + `try/except ImportError` in `gnn_model.py` that provides a stub `GNNModel`.

## Performance Note

- GNN is **skipped** in `/stats` and `/metrics` endpoints because loading PyTorch adds ~30s startup delay.
- GNN is only used in `/predict`, `/ensemble`, and related single-patient endpoints.
