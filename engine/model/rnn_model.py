import numpy as np
import pandas as pd

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset
    import torch.optim as optim
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    class nn:
        Module = object

class FormGRU(nn.Module):
    def __init__(self, input_dim, hidden_dim=32, num_layers=1, num_classes=3):
        super().__init__()
        if not HAS_TORCH: return
        self.gru = nn.GRU(input_dim, hidden_dim, num_layers, batch_first=True, dropout=0.0 if num_layers == 1 else 0.2)
        
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim * 2, 16),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(16, num_classes)
        )
        
    def forward(self, x_home, x_away):
        _, h_home = self.gru(x_home)
        out_home = h_home[-1]
        
        _, h_away = self.gru(x_away)
        out_away = h_away[-1]
        
        merged = torch.cat((out_home, out_away), dim=1)
        out = self.fc(merged)
        return out

class GRUClassifierWrapper:
    """Wrapper that mimics sklearn interface for our ensemble"""
    def __init__(self, epochs=25, batch_size=32, lr=0.005):
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.model = None
        self.is_trained = False
        self.windows = [20, 10, 5] # The sequence order (Oldest -> Newest)
        self.base_metrics = ['pts', 'gf', 'ga', 'xg_for', 'xg_against', 'poss', 'sot_for']

    def _build_sequences(self, X):
        """Converts wide dataframe into (N, Seq, Features) based on the rolling windows"""
        N = len(X)
        seq_len = len(self.windows)
        num_feat = len(self.base_metrics)

        X_home_seq = np.zeros((N, seq_len, num_feat))
        X_away_seq = np.zeros((N, seq_len, num_feat))

        for s_idx, w in enumerate(self.windows):
            for f_idx, m_name in enumerate(self.base_metrics):
                h_col = f"home_roll_{m_name}_{w}"
                a_col = f"away_roll_{m_name}_{w}"
                
                if h_col in X.columns: X_home_seq[:, s_idx, f_idx] = X[h_col].values
                if a_col in X.columns: X_away_seq[:, s_idx, f_idx] = X[a_col].values

        return X_home_seq, X_away_seq

    def fit(self, X, y):
        if not HAS_TORCH:
            print("PyTorch not installed. GRU training skipped.")
            return
            
        X_home, X_away = self._build_sequences(X)
        self.model = FormGRU(input_dim=len(self.base_metrics))
        
        # Datasets
        tensor_h = torch.FloatTensor(X_home)
        tensor_a = torch.FloatTensor(X_away)
        tensor_y = torch.LongTensor(y)
        dataset = TensorDataset(tensor_h, tensor_a, tensor_y)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        
        self.model.train()
        for epoch in range(self.epochs):
            for batch_h, batch_a, batch_y in loader:
                optimizer.zero_grad()
                outputs = self.model(batch_h, batch_a)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                
        self.is_trained = True

    def predict_proba(self, X):
        if not HAS_TORCH or not self.is_trained:
            return np.ones((len(X), 3)) / 3.0
            
        X_home, X_away = self._build_sequences(X)
        tensor_h = torch.FloatTensor(X_home)
        tensor_a = torch.FloatTensor(X_away)
        
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(tensor_h, tensor_a)
            probs = torch.softmax(outputs, dim=1).numpy()
            
        return probs
