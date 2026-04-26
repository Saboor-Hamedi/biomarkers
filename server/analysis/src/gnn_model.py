import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
)

try:
    import torch
    from torch_geometric.nn import GCNConv, global_mean_pool

    class GNNModel(torch.nn.Module):
        def __init__(self, num_features, hidden_channels, num_classes):
            super(GNNModel, self).__init__()
            self.conv1 = GCNConv(num_features, hidden_channels)
            self.conv2 = GCNConv(hidden_channels, hidden_channels)
            self.lin = torch.nn.Linear(hidden_channels, num_classes)

        def forward(self, x, edge_index, batch):
            x = self.conv1(x, edge_index)
            x = x.relu()
            x = self.conv2(x, edge_index)
            x = x.relu()
            x = global_mean_pool(x, batch)
            x = self.lin(x)
            return x
except ImportError:
    class GNNModel:
        pass

def train_gnn(X_train, y_train, X_val, y_val, feature_columns, X_scaled):
    print("\nInitializing Graph Neural Network (GNN)...")
    # GNN requires PyTorch and PyG libraries
    try:
        import torch
        import torch.nn.functional as F
        from torch_geometric.data import Data, DataLoader
        # GNNModel is now defined at the top level


        # Build correlation-based graph from features
        print("Building correlation graph from biomarkers...")
        corr_matrix = pd.DataFrame(X_scaled, columns=feature_columns).corr()
        edges = []
        for i in range(len(corr_matrix.columns)):
            for j in range(i + 1, len(corr_matrix.columns)):
                if abs(corr_matrix.iloc[i, j]) > 0.3:  # Only keep strong correlations
                    edges.append([i, j])
                    edges.append([j, i])  # Bidirectional

        if not edges:
            edges = [[0, 1], [1, 0]]  # Minimum connectivity

        edge_index = torch.LongTensor(edges).t().contiguous()

        # Create graph data objects
        print("Creating GNN training data...")
        train_data_list = []
        for idx in range(len(X_train)):
            x = (
                torch.FloatTensor(X_train[idx])
                .unsqueeze(0)
                .expand(len(feature_columns), -1)
            )
            y = torch.LongTensor(
                [y_train.iloc[idx] if hasattr(y_train, "iloc") else y_train[idx]]
            )
            data = Data(x=x, edge_index=edge_index, y=y)
            train_data_list.append(data)

        val_data_list = []
        for idx in range(len(X_val)):
            x = torch.FloatTensor(X_val[idx]).unsqueeze(0).expand(len(feature_columns), -1)
            y = torch.LongTensor(
                [y_val.iloc[idx] if hasattr(y_val, "iloc") else y_val[idx]]
            )
            data = Data(x=x, edge_index=edge_index, y=y)
            val_data_list.append(data)

        train_loader = DataLoader(train_data_list, batch_size=16, shuffle=True)
        val_loader = DataLoader(val_data_list, batch_size=16, shuffle=False)

        # Initialize and train GNN
        print("\nTraining Graph Neural Network...")
        gnn_model = GNNModel(
            num_features=len(feature_columns), hidden_channels=32, num_classes=2
        )
        optimizer = torch.optim.Adam(gnn_model.parameters(), lr=0.01)
        criterion = torch.nn.CrossEntropyLoss()

        best_val_loss = float("inf")
        for epoch in range(50):
            gnn_model.train()
            train_loss = 0
            for batch in train_loader:
                optimizer.zero_grad()
                out = gnn_model(
                    batch.x,
                    batch.edge_index,
                    (
                        batch.batch
                        if hasattr(batch, "batch")
                        else torch.zeros(batch.x.size(0), dtype=torch.long)
                    ),
                )
                loss = criterion(out, batch.y.squeeze())
                loss.backward()
                optimizer.step()
                train_loss += loss.item()

            gnn_model.eval()
            val_loss = 0
            with torch.no_grad():
                for batch in val_loader:
                    out = gnn_model(
                        batch.x,
                        batch.edge_index,
                        (
                            batch.batch
                            if hasattr(batch, "batch")
                            else torch.zeros(batch.x.size(0), dtype=torch.long)
                        ),
                    )
                    loss = criterion(out, batch.y.squeeze())
                    val_loss += loss.item()

            if (epoch + 1) % 10 == 0:
                print(
                    f"Epoch {epoch+1:3d} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}"
                )

        # GNN Predictions
        print("\nEvaluating GNN on validation set...")
        gnn_model.eval()
        y_gnn_pred = []
        y_gnn_proba = []
        with torch.no_grad():
            for batch in val_loader:
                out = gnn_model(
                    batch.x,
                    batch.edge_index,
                    (
                        batch.batch
                        if hasattr(batch, "batch")
                        else torch.zeros(batch.x.size(0), dtype=torch.long)
                    ),
                )
                probs = F.softmax(out, dim=1)
                y_gnn_pred.extend(probs.argmax(dim=1).numpy())
                y_gnn_proba.extend(probs[:, 1].numpy())

        y_gnn_pred = np.array(y_gnn_pred)
        y_gnn_proba = np.array(y_gnn_proba)

        # Calculate GNN metrics
        gnn_results = {
            "accuracy": accuracy_score(y_val, y_gnn_pred),
            "precision": precision_score(y_val, y_gnn_pred, zero_division=0),
            "recall": recall_score(y_val, y_gnn_pred, zero_division=0),
            "f1_score": f1_score(y_val, y_gnn_pred, zero_division=0),
            "roc_auc": roc_auc_score(y_val, y_gnn_proba),
            "confusion_matrix": confusion_matrix(y_val, y_gnn_pred),
            "y_pred": y_gnn_pred,
            "y_pred_proba": y_gnn_proba,
        }

        print("\n✅ GNN Model trained successfully!")
        print(f"  Validation Accuracy: {gnn_results['accuracy']:.4f}")
        print(f"  Validation F1-Score: {gnn_results['f1_score']:.4f}")
        print(f"  Validation ROC-AUC: {gnn_results['roc_auc']:.4f}")
        


        return gnn_results, gnn_model

    except ImportError:
        print("⚠️  PyTorch or PyG not installed. GNN training skipped.")
        print("   Install with: pip install torch torch-geometric")
        return None, None
