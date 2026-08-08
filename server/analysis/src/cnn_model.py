import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
)

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


class CNN1D(nn.Module):
    """1D Convolutional Neural Network for sequence-aware DPV fingerprinting.

    Treats the 200 ordered current steps as a 1D signal and extracts
    local peak-width / slope variations via stacked convolutions.
    """

    def __init__(self, input_length=200, hidden_channels=64, num_classes=2):
        super(CNN1D, self).__init__()
        self.input_length = input_length
        self.conv1 = nn.Conv1d(1, hidden_channels, kernel_size=7, padding=3)
        self.bn1 = nn.BatchNorm1d(hidden_channels)
        self.conv2 = nn.Conv1d(hidden_channels, hidden_channels * 2, kernel_size=5, padding=2)
        self.bn2 = nn.BatchNorm1d(hidden_channels * 2)
        self.conv3 = nn.Conv1d(hidden_channels * 2, hidden_channels * 2, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm1d(hidden_channels * 2)
        self.pool = nn.MaxPool1d(2)
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Linear(hidden_channels * 2, num_classes)

    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.global_pool(x).squeeze(-1)
        x = self.dropout(x)
        return self.fc(x)

    def predict(self, X):
        return self.predict_proba(X).argmax(axis=1)

    def predict_proba(self, X):
        self.eval()
        X = np.asarray(X, dtype=np.float32)
        if X.ndim == 1:
            X = X.reshape(1, -1)
        if X.shape[1] != self.input_length:
            X = X[:, -self.input_length:]
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(device)
        with torch.no_grad():
            out = self(torch.FloatTensor(X).to(device))
        return F.softmax(out, dim=1).cpu().numpy()


class BiLSTM(nn.Module):
    """Bidirectional LSTM for sequence-aware DPV fingerprinting."""

    def __init__(self, input_length=200, hidden_size=64, num_layers=2, num_classes=2):
        super(BiLSTM, self).__init__()
        self.input_length = input_length
        self.lstm = nn.LSTM(
            input_size=1,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=0.3 if num_layers > 1 else 0.0,
        )
        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Linear(hidden_size * 2, num_classes)

    def forward(self, x):
        x = x.unsqueeze(-1)
        out, _ = self.lstm(x)
        out = out.mean(dim=1)
        out = self.dropout(out)
        return self.fc(out)

    def predict(self, X):
        return self.predict_proba(X).argmax(axis=1)

    def predict_proba(self, X):
        self.eval()
        X = np.asarray(X, dtype=np.float32)
        if X.ndim == 1:
            X = X.reshape(1, -1)
        if X.shape[1] != self.input_length:
            X = X[:, -self.input_length:]
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(device)
        with torch.no_grad():
            out = self(torch.FloatTensor(X).to(device))
        return F.softmax(out, dim=1).cpu().numpy()


def _train_pytorch_model(model, X_train, y_train, X_val, y_val, epochs=50, lr=0.001, batch_size=64, seed=42, patience=8):
    torch.manual_seed(seed)
    np.random.seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    y_train = np.asarray(y_train) if isinstance(y_train, (np.ndarray, list)) else np.asarray(y_train.values)
    y_val = np.asarray(y_val) if isinstance(y_val, (np.ndarray, list)) else np.asarray(y_val.values)

    X_train_t = torch.FloatTensor(np.asarray(X_train, dtype=np.float32)).to(device)
    X_val_t = torch.FloatTensor(np.asarray(X_val, dtype=np.float32)).to(device)
    y_train_t = torch.LongTensor(y_train).to(device)
    y_val_t = torch.LongTensor(y_val).to(device)

    classes = np.unique(y_train)
    weights = np.array([len(y_train) / (2.0 * np.bincount(y_train)[c]) for c in classes])
    class_weight = torch.FloatTensor(weights).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)

    n = len(X_train_t)
    best_val_loss = float("inf")
    best_state = None
    epochs_no_improve = 0

    for epoch in range(epochs):
        model.train()
        perm = torch.randperm(n, device=device)
        total_loss = 0.0
        nb = 0
        for i in range(0, n, batch_size):
            idx = perm[i:i + batch_size]
            xb, yb = X_train_t[idx], y_train_t[idx]
            optimizer.zero_grad()
            out = model(xb)
            loss = criterion(out, yb)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            nb += 1

        model.eval()
        with torch.no_grad():
            out_val = model(X_val_t)
            val_loss = criterion(out_val, y_val_t).item()
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    model.to(torch.device("cpu"))
    return model


def _evaluate_pytorch_model(model, X_val, y_val):
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    with torch.no_grad():
        X_val_t = torch.FloatTensor(np.asarray(X_val, dtype=np.float32)).to(device)
        out = model(X_val_t)
        probs = F.softmax(out, dim=1)
        y_pred = probs.argmax(dim=1).cpu().numpy()
        y_proba = probs[:, 1].cpu().numpy()
    model.to(torch.device("cpu"))

    return {
        "accuracy": accuracy_score(y_val, y_pred),
        "precision": precision_score(y_val, y_pred, zero_division=0),
        "recall": recall_score(y_val, y_pred, zero_division=0),
        "f1_score": f1_score(y_val, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_val, y_proba),
        "pr_auc": average_precision_score(y_val, y_proba),
        "confusion_matrix": confusion_matrix(y_val, y_pred),
        "y_pred": y_pred,
        "y_pred_proba": y_proba,
    }


def train_cnn(X_train, y_train, X_val, y_val, input_length=200, epochs=50, lr=0.001, seed=42, batch_size=64, patience=8):
    """Train the 1D-CNN on DPV sequences."""
    if not TORCH_AVAILABLE:
        print("\u26a0\ufe0f  PyTorch not installed. 1D-CNN skipped.")
        return None, None

    print("\nInitializing 1D-CNN (sequence-aware)...")
    model = CNN1D(input_length=input_length, num_classes=2)
    _train_pytorch_model(model, X_train, y_train, X_val, y_val, epochs=epochs, lr=lr, seed=seed, batch_size=batch_size, patience=patience)
    results = _evaluate_pytorch_model(model, X_val, y_val)
    print(f"  \u2705 1D-CNN trained! Accuracy: {results['accuracy']:.4f} | F1: {results['f1_score']:.4f} | ROC-AUC: {results['roc_auc']:.4f}")
    return results, model


def train_bilstm(X_train, y_train, X_val, y_val, input_length=200, epochs=50, lr=0.001, seed=42, batch_size=64, patience=8):
    """Train the BiLSTM on DPV sequences."""
    if not TORCH_AVAILABLE:
        print("\u26a0\ufe0f  PyTorch not installed. BiLSTM skipped.")
        return None, None

    print("\nInitializing BiLSTM (sequence-aware)...")
    model = BiLSTM(input_length=input_length, num_classes=2)
    _train_pytorch_model(model, X_train, y_train, X_val, y_val, epochs=epochs, lr=lr, seed=seed, batch_size=batch_size, patience=patience)
    results = _evaluate_pytorch_model(model, X_val, y_val)
    print(f"  \u2705 BiLSTM trained! Accuracy: {results['accuracy']:.4f} | F1: {results['f1_score']:.4f} | ROC-AUC: {results['roc_auc']:.4f}")
    return results, model
