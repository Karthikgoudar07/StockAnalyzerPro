import torch
import torch.nn as nn
from torch.utils.data import Dataset

class AttentionLayer(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1)
        )

    def forward(self, hidden_states):
        attention_weights = self.attention(hidden_states)
        attention_weights = torch.softmax(attention_weights, dim=1)
        attended = torch.sum(attention_weights * hidden_states, dim=1)
        return attended

class EnhancedStockBiLSTM(nn.Module):
    def __init__(self, input_size=5, hidden_size=128, num_layers=3, output_size=30):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.lstm1 = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, bidirectional=True)
        self.attention = AttentionLayer(hidden_size * 2)
        
        self.dropout = nn.Dropout(0.3)
        self.norm1 = nn.LayerNorm(hidden_size * 2)
        
        self.fc1 = nn.Linear(hidden_size * 2, hidden_size)
        self.relu = nn.ReLU()
        self.norm2 = nn.LayerNorm(hidden_size)
        
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        device = next(self.parameters()).device
        h0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_size).to(device)
        c0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_size).to(device)
        
        out, _ = self.lstm1(x, (h0, c0))
        out = self.norm1(out)
        
        attended = self.attention(out)
        attended = self.dropout(attended)
        
        out = self.fc1(attended)
        out = self.relu(out)
        out = self.norm2(out)
        out = self.dropout(out)
        
        out = self.fc2(out)
        return out

class StockDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y)
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]
