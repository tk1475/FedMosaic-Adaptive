import torch
import torch.nn as nn
import torch.nn.functional as F

class PQLoRALayer(nn.Module):
    def __init__(self, in_dim, out_dim, r=16, alpha=16):
        super().__init__()
        self.r = r
        self.scaling = alpha / r

        # --- The Frozen Adapters (Model-Specific Dimensions) ---
        # A: Projects down (in_dim -> r)
        # B: Projects up (r -> out_dim)
        # Initialized Orthogonally (Theorem 1 of the paper)
        self.A = nn.Linear(in_dim, r, bias=False)
        self.B = nn.Linear(r, out_dim, bias=False)
        
        # Orthogonal Init
        nn.init.orthogonal_(self.A.weight)
        nn.init.orthogonal_(self.B.weight)
        
        # Freeze A and B immediately
        self.A.weight.requires_grad = False
        self.B.weight.requires_grad = False

        # --- The Shared Adapters (Dimension-Invariant) ---
        # P: (r, r) - Mixing matrix
        # Q: (r, 1) - Bias term
        # Trainable & Shared across all clients
        self.P = nn.Parameter(torch.eye(r), requires_grad=True)
        self.Q = nn.Parameter(torch.zeros(r), requires_grad=True)

    def forward(self, x):
        # x shape: (batch, seq, in_dim)
        
        # 1. Down Projection (Frozen)
        # Output: (batch, seq, r)
        x_down = self.A(x)

        # 2. Shared Transformation (Trainable)
        # P maps r -> r. Q adds bias.
        # x_transformed = (x_down @ P) + Q
        x_transformed = torch.matmul(x_down, self.P) + self.Q

        # 3. Up Projection (Frozen)
        # Output: (batch, seq, out_dim)
        x_up = self.B(x_transformed)

        return x_up * self.scaling