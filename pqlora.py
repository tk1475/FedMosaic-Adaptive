import torch
import torch.nn as nn
import torch.nn.functional as F

class PQLoRALayer(nn.Module):
    def __init__(self, in_dim, out_dim, r=16, alpha=16):
        super().__init__()
        self.r = r
        self.scaling = alpha / r

        # --- 1. Frozen Projection Matrices (A & B) ---
        # "Initialized with orthogonal sets" (Sec 4.2.2)
        self.A = nn.Linear(in_dim, r, bias=False)
        self.B = nn.Linear(r, out_dim, bias=False)
        
        # Orthogonal Initialization
        nn.init.orthogonal_(self.A.weight)
        nn.init.orthogonal_(self.B.weight)
        
        # Freeze A and B immediately (Theorem 2)
        self.A.weight.requires_grad = False
        self.B.weight.requires_grad = False

        # --- 2. Local Adapter (Trainable) ---
        # L_i in the paper
        self.P_local = nn.Parameter(torch.eye(r), requires_grad=True)
        self.Q_local = nn.Parameter(torch.zeros(r), requires_grad=True)

        # --- 3. Global Adapter (Frozen) ---
        # G_i in the paper (Personalized Global Model)
        # We start with identity/zeros until first aggregation
        self.P_global = nn.Parameter(torch.eye(r), requires_grad=False)
        self.Q_global = nn.Parameter(torch.zeros(r), requires_grad=False)

        # --- 4. Gating Parameter (beta) ---
        # Eq 7: Balances Local vs Global knowledge
        self.beta = nn.Parameter(torch.tensor(0.0), requires_grad=True)

    def forward(self, x):
        # x shape: (batch, seq, in_dim)
        
        # 1. Down Projection (Shared A)
        x_down = self.A(x) # (batch, seq, r)

        # 2. Local Pathway (L_i)
        # x_L = x_down @ P_local + Q_local
        x_L = torch.matmul(x_down, self.P_local) + self.Q_local
        
        # 3. Global Pathway (G_i) - Frozen
        with torch.no_grad():
            x_G = torch.matmul(x_down, self.P_global) + self.Q_global

        # 4. Gating (Eq 7)
        # h_out = (1 - sigmoid(beta)) * h_L + sigmoid(beta) * h_G
        gate = torch.sigmoid(self.beta)
        x_combined = (1 - gate) * x_L + gate * x_G

        # 5. Up Projection (Shared B)
        x_up = self.B(x_combined) # (batch, seq, out_dim)

        return x_up * self.scaling
        
    def update_global_weights(self, new_P, new_Q):
        """Updates the frozen global component after aggregation."""
        with torch.no_grad():
            self.P_global.copy_(new_P)
            self.Q_global.copy_(new_Q)