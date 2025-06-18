#!/usr/bin/env python3
"""
Test script for the fixed voice conversion model
"""

import torch
import torch.nn as nn
from transformers import HubertModel

# Fixed SSM implementation
import torch.nn.functional as F
import math

class S6Block(nn.Module):
    """S6 (Selective State Space) Block - FIXED VERSION"""
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
        nn.init.uniform_(self.A_log, -4.0, -2.0)
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
        Selective scan operation - FIXED VERSION
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
        h = torch.zeros(B, d_inner, self.d_state, device=x.device, dtype=x.dtype)
        
        # Selective scan - FIXED
        outputs = []
        
        for i in range(L):
            # Correct indexing and broadcasting
            x_i = x[:, i]  # (B, d_inner)
            deltaA_i = deltaA[:, i]  # (B, d_inner, d_state)
            deltaB_i = deltaB[:, i]  # (B, d_inner, d_state)
            
            # State update
            h = deltaA_i * h + deltaB_i * x_i.unsqueeze(-1)  # (B, d_inner, d_state)
            
            # Output computation
            C_i = C_sel[:, i]  # (B, d_inner)
            y_i = torch.sum(C_i.unsqueeze(-1) * h, dim=-1)  # (B, d_inner)
            
            outputs.append(y_i)
        
        y = torch.stack(outputs, dim=1)  # (B, L, d_inner)
        
        # Add skip connection - FIXED
        y = y + x * self.D.view(1, 1, -1)  # (B, L, d_inner)
        
        return y

class S6SSMEncoder(nn.Module):
    """S6-based SSM Encoder"""
    def __init__(self, d_model=768, n_layers=3, d_state=64):
        super().__init__()
        
        self.layers = nn.ModuleList([
            S6Block(d_model, d_state) for _ in range(n_layers)
        ])
        
        self.final_norm = nn.LayerNorm(d_model)
        
    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return self.final_norm(x)

# Simplified test model (without Flow Matching for now)
class SimpleTestModel(nn.Module):
    def __init__(self, d_model=768, n_speakers=4):
        super().__init__()
        
        # Mock HuBERT (we'll just use a simple linear layer for testing)
        self.feature_extractor = nn.Linear(16384, d_model * 50)  # Mock feature extraction
        
        # SSM Encoder
        self.ssm_encoder = S6SSMEncoder(d_model=d_model, n_layers=3)
        
        # Speaker embedding
        self.speaker_embedding = nn.Embedding(n_speakers, d_model)
        
        # Simple output projection
        self.output_proj = nn.Linear(d_model, 16384)
        
    def forward(self, source_waveform, target_speaker_id):
        """
        Simplified forward pass for testing shapes
        """
        B = source_waveform.size(0)
        
        # Convert stereo to mono for processing
        if source_waveform.dim() == 3:  # (B, C, L)
            source_mono = source_waveform.mean(dim=1)  # (B, L)
        else:
            source_mono = source_waveform
        
        # Mock feature extraction (simulating HuBERT output)
        features = self.feature_extractor(source_mono)  # (B, d_model * 50)
        features = features.view(B, 50, -1)  # (B, 50, d_model)
        
        print(f"[Test] Features shape: {features.shape}")
        
        # SSM encoding
        encoded = self.ssm_encoder(features)  # (B, 50, d_model)
        print(f"[Test] Encoded shape: {encoded.shape}")
        
        # Speaker embedding
        speaker_emb = self.speaker_embedding(target_speaker_id)  # (B, d_model)
        speaker_emb = speaker_emb.unsqueeze(1).expand(-1, 50, -1)  # (B, 50, d_model)
        
        # Combine
        combined = encoded + speaker_emb  # (B, 50, d_model)
        
        # Simple output (pool and project)
        pooled = combined.mean(dim=1)  # (B, d_model)
        output = self.output_proj(pooled)  # (B, 16384)
        
        print(f"[Test] Output shape: {output.shape}")
        
        return output

def test_fixed_model():
    """Test the fixed model with dummy data"""
    print("🧪 Testing fixed model...")
    
    # Configuration
    batch_size = 2
    waveform_length = 16384
    channels = 2
    n_speakers = 4
    d_model = 768
    
    # Create dummy data
    if channels == 2:
        source_waveform = torch.randn(batch_size, channels, waveform_length)
    else:
        source_waveform = torch.randn(batch_size, waveform_length)
    
    target_speaker_id = torch.randint(0, n_speakers, (batch_size,))
    
    print(f"📊 Input shapes:")
    print(f"   Source: {source_waveform.shape}")
    print(f"   Speaker ID: {target_speaker_id.shape}")
    
    # Initialize model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🔧 Device: {device}")
    
    model = SimpleTestModel(d_model=d_model, n_speakers=n_speakers).to(device)
    source_waveform = source_waveform.to(device)
    target_speaker_id = target_speaker_id.to(device)
    
    # Test forward pass
    try:
        output = model(source_waveform, target_speaker_id)
        print(f"✅ Forward pass successful!")
        print(f"📊 Final output: {output.shape}")
        # Test backward pass
        loss = output.sum()
        loss.backward()
        print(f"✅ Backward pass successful!")
    except Exception as e:
        print(f"❌ Error: {type(e).__name__}: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_fixed_model()