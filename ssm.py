import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class S6Block(nn.Module):
    """S6 (Selective State Space) Block"""
    def __init__(self, d_model, d_state=64, expand_factor=2):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_inner = expand_factor * d_model
        
        # Input projections
        self.in_proj = nn.Linear(d_model, self.d_inner * 2)
        
        # Selective parameters
        self.x_proj = nn.Linear(self.d_inner, d_state + d_state + self.d_inner)
        self.dt_proj = nn.Linear(d_state, self.d_inner)
        
        # State space parameters
        self.A_log = nn.Parameter(torch.randn(self.d_inner, d_state))
        self.D = nn.Parameter(torch.randn(self.d_inner))
        
        # Output projection
        self.out_proj = nn.Linear(self.d_inner, d_model)
        
        # Normalization
        self.norm = nn.LayerNorm(d_model)
        
        self._init_parameters()
    
    def _init_parameters(self):
        """Initialize parameters for stability"""
        # Initialize A to be stable
        nn.init.uniform_(self.A_log, -4.0, -2.0)
        # Initialize D
        nn.init.normal_(self.D, mean=1.0, std=0.1)
    
    def forward(self, x):
        """
        Args:
            x: (B, L, D) input sequence
        Returns:
            (B, L, D) output sequence
        """
        B, L, D = x.shape
        residual = x
        
        # Input projection
        x_and_res = self.in_proj(x)  # (B, L, 2*d_inner)
        x, res = x_and_res.split([self.d_inner, self.d_inner], dim=-1)
        
        # Apply SiLU activation
        x = F.silu(x)
        
        # Selective scan
        x = self.selective_scan(x)
        
        # Apply residual connection
        x = x * F.silu(res)
        
        # Output projection
        x = self.out_proj(x)
        
        # Residual connection and normalization
        out = self.norm(x + residual)
        return out
    
    def selective_scan(self, x):
        """
        Selective scan operation (core of S6) - FIXED VERSION
        Args:
            x: (B, L, d_inner)
        Returns:
            (B, L, d_inner)
        """
        B, L, d_inner = x.shape
        
        # Compute selective parameters
        x_dbl = self.x_proj(x)  # (B, L, d_state + d_state + d_inner)
        delta, B_sel, C_sel = x_dbl.split([self.d_state, self.d_state, self.d_inner], dim=-1)
        
        # Discretization step
        delta = F.softplus(self.dt_proj(delta))  # (B, L, d_inner)
        
        # State space matrices
        A = -torch.exp(self.A_log.float())  # (d_inner, d_state)
        
        # Discretize A and B
        deltaA = torch.exp(delta.unsqueeze(-1) * A.unsqueeze(0).unsqueeze(0))  # (B, L, d_inner, d_state)
        deltaB = delta.unsqueeze(-1) * B_sel.unsqueeze(-2)  # (B, L, d_inner, d_state)
        
        # Initialize state
        h = torch.zeros(B, d_inner, self.d_state, device=x.device, dtype=x.dtype)  # (B, d_inner, d_state)
        
        # Selective scan (FIXED implementation)
        outputs = []
        
        for i in range(L):
            # FIXED: Use correct indexing and broadcasting
            # Update state: h = A * h + B * x
            x_i = x[:, i]  # (B, d_inner)
            deltaA_i = deltaA[:, i]  # (B, d_inner, d_state)
            deltaB_i = deltaB[:, i]  # (B, d_inner, d_state)
            
            # State update with correct broadcasting
            h = deltaA_i * h + deltaB_i * x_i.unsqueeze(-1)  # (B, d_inner, d_state)
            
            # Output computation: y = C * h
            # FIXED: Use einsum for clear matrix multiplication
            C_i = C_sel[:, i]  # (B, d_inner)
            
            # We want to compute y[b, d] = sum_s(C[b, d] * h[b, d, s])
            # This is equivalent to element-wise multiplication followed by sum over state dimension
            y_i = torch.sum(C_i.unsqueeze(-1) * h, dim=-1)  # (B, d_inner)
            
            outputs.append(y_i)
        
        y = torch.stack(outputs, dim=1)  # (B, L, d_inner)
        
        # Add skip connection - FIXED: proper broadcasting
        y = y + x * self.D.view(1, 1, -1)  # (B, L, d_inner)
        
        return y

class S6SSMEncoder(nn.Module):
    """S6-based SSM Encoder for sequence modeling"""
    def __init__(self, d_model=768, n_layers=3, d_state=64):
        super().__init__()
        
        self.layers = nn.ModuleList([
            S6Block(d_model, d_state) for _ in range(n_layers)
        ])
        
        self.final_norm = nn.LayerNorm(d_model)
        
        print(f"🚀 S6SSM initialized with {n_layers} layers")
        
    def forward(self, x):
        """
        Args:
            x: (B, T, D) input features from HuBERT
        Returns:
            (B, T, D) encoded features
        """
        for layer in self.layers:
            x = layer(x)
        
        return self.final_norm(x)