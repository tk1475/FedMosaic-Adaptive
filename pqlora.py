import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class PQLoRALayer(nn.Module):
    def __init__(self, in_features, out_features, r=16, alpha=16, dropout=0.1):
        super().__init__()
        self.r = r
        self.alpha = alpha
        self.scaling = alpha / r
        
        # Low-Rank Matrices
        self.P_local = nn.Parameter(torch.eye(r, in_features)) # Identity init
        self.Q_local = nn.Parameter(torch.zeros(out_features, r)) # Zero init
        
        # Gating parameter (controls Local vs Global mix)
        self.beta = nn.Parameter(torch.tensor(0.5)) 
        
        # Global Buffers (Frozen, updated by Server)
        self.register_buffer("P_global", torch.zeros(r, in_features))
        self.register_buffer("Q_global", torch.zeros(out_features, r))
        
        self.dropout = nn.Dropout(p=dropout)
        
    def update_global_weights(self, P, Q):
        with torch.no_grad():
            self.P_global.copy_(P)
            self.Q_global.copy_(Q)
            
    def forward(self, x):
        # Local Path
        local_out = (x @ self.P_local.T) @ self.Q_local.T
        
        # Global Path
        global_out = (x @ self.P_global.T) @ self.Q_global.T
        
        # RELA Equation: Mix based on Beta
        combined = (1 - self.beta) * local_out + self.beta * global_out
        
        return self.dropout(combined) * self.scaling