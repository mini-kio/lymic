import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class FlowMatching(nn.Module):
    """
    Flow Matching for FULL WAVEFORM generation
    Based on "Flow Matching for Generative Modeling" paper
    """
    def __init__(self, dim=16384, condition_dim=768, steps=100, sigma_min=1e-4):
        super().__init__()
        self.dim = dim  # Full waveform length!
        self.condition_dim = condition_dim
        self.steps = steps
        self.sigma_min = sigma_min
        
        # Vector field network for waveform generation
        self.vector_field = VectorFieldNet(
            dim=dim, 
            condition_dim=condition_dim,
            hidden_dim=512
        )
        
    def compute_loss(self, x1, condition, num_samples=None):
        """
        Compute flow matching loss for waveform generation
        Args:
            x1: (B, waveform_length) target waveforms  
            condition: (B, condition_dim) conditioning features
        """
        B, waveform_length = x1.shape
        
        if num_samples is None:
            num_samples = B
        
        # Sample time uniformly
        t = torch.rand(num_samples, device=x1.device, dtype=x1.dtype)
        
        # Sample noise
        x0 = torch.randn_like(x1[:num_samples])
        x1_sample = x1[:num_samples]
        condition_sample = condition[:num_samples]
        
        # Interpolate
        t_expanded = t.unsqueeze(-1)  # (num_samples, 1)
        x_t = (1 - t_expanded) * x0 + t_expanded * x1_sample
        
        # Target velocity (conditional flow)
        u_t = x1_sample - x0
        
        # Predict velocity
        v_t = self.vector_field(x_t, t, condition_sample)
        
        # MSE loss
        loss = F.mse_loss(v_t, u_t)
        
        return loss
    
    def sample(self, condition, num_steps=None, x0=None, method='fast_inverse'):
        """
        🚀 Optimized sampling with multiple fast methods
        Args:
            condition: (B, condition_dim) conditioning features
            num_steps: number of integration steps
            x0: initial noise (if None, sample from Gaussian)
            method: 'fast_inverse', 'ode_adaptive', 'ode', 'euler'
        """
        if num_steps is None:
            # Default faster steps
            if method == 'fast_inverse':
                num_steps = 8  # 🔥 Very fast
            elif method == 'ode_adaptive':
                num_steps = 15  # Adaptive, so fewer nominal steps
            else:
                num_steps = self.steps
        
        B = condition.size(0)
        device = condition.device
        
        if x0 is None:
            x0 = torch.randn(B, self.dim, device=device)
        
        if method == 'ode_adaptive':
            return self._sample_ode_adaptive(condition, x0, num_steps)
        elif method == 'ode':
            return self._sample_ode(condition, x0, num_steps)
        elif method == 'fast_inverse':
            return self._sample_fast_inverse(condition, x0, num_steps)
        else:  # euler
            return self._sample_euler(condition, x0, num_steps)
    
    def _sample_euler(self, condition, x0, num_steps):
        """Standard Euler integration"""
        dt = 1.0 / num_steps
        t = 0.0
        x = x0
        
        for step in range(num_steps):
            t_tensor = torch.full((x.size(0),), t, device=x.device)
            
            # Predict velocity
            v = self.vector_field(x, t_tensor, condition)
            
            # Update
            x = x + dt * v
            t += dt
        
        return x
    
    def _sample_ode(self, condition, x0, num_steps):
        """ODE solver using adaptive step size"""
        try:
            from scipy.integrate import solve_ivp
            import numpy as np
        except ImportError:
            # Fallback to Euler if scipy not available
            return self._sample_euler(condition, x0, num_steps)
        
        device = x0.device
        B = x0.size(0)
        
        def ode_func(t, x_flat):
            x_tensor = torch.from_numpy(x_flat.reshape(B, -1)).float().to(device)
            t_tensor = torch.full((B,), t, device=device)
            
            with torch.no_grad():
                v = self.vector_field(x_tensor, t_tensor, condition)
            
            return v.cpu().numpy().flatten()
        
        # Solve ODE
        sol = solve_ivp(
            ode_func, 
            [0, 1], 
            x0.cpu().numpy().flatten(),
            method='RK45',
            rtol=1e-3,  # Faster tolerance
            atol=1e-6
        )
        
        result = torch.from_numpy(sol.y[:, -1]).float().to(device)
        return result.view(B, self.dim)
    
    def _sample_fast_inverse(self, condition, x0, num_steps):
        """
        🚀 Ultra-fast inverse flow sampling
        Uses adaptive step size + higher-order methods
        """
        # Use even fewer steps with smart step sizing
        effective_steps = max(num_steps // 2, 3)  # Minimum 3 steps
        
        # Adaptive step schedule (larger steps early, smaller steps late)
        step_schedule = self._get_adaptive_schedule(effective_steps)
        
        x = x0
        t = 0.0
        
        for i, dt in enumerate(step_schedule):
            t_tensor = torch.full((x.size(0),), t, device=x.device)
            
            if i == 0:
                # First step: simple Euler
                v = self.vector_field(x, t_tensor, condition)
                x = x + dt * v
            else:
                # Higher-order steps: 4th order Runge-Kutta for critical final steps
                if i >= effective_steps - 2:  # Last 2 steps use RK4
                    x = self._rk4_step(x, t, dt, condition)
                else:
                    # Middle steps: Heun's method (RK2)
                    v1 = self.vector_field(x, t_tensor, condition)
                    x_pred = x + dt * v1
                    
                    t_next = torch.full((x.size(0),), t + dt, device=x.device)
                    v2 = self.vector_field(x_pred, t_next, condition)
                    
                    x = x + dt * 0.5 * (v1 + v2)
            
            t += dt
        
        return x
    
    def _get_adaptive_schedule(self, num_steps):
        """Generate adaptive step schedule"""
        # Exponential decay: larger steps early, smaller steps late
        steps = []
        total_time = 1.0
        
        # Generate exponentially decreasing steps
        alpha = 0.3  # Controls decay rate
        weights = [math.exp(-alpha * i) for i in range(num_steps)]
        total_weight = sum(weights)
        
        # Normalize to sum to 1.0
        steps = [w * total_time / total_weight for w in weights]
        
        return steps
    
    def _rk4_step(self, x, t, dt, condition):
        """4th order Runge-Kutta step for high accuracy"""
        t_tensor = torch.full((x.size(0),), t, device=x.device)
        
        k1 = self.vector_field(x, t_tensor, condition)
        
        t_mid1 = torch.full((x.size(0),), t + dt/2, device=x.device)
        k2 = self.vector_field(x + dt/2 * k1, t_mid1, condition)
        
        k3 = self.vector_field(x + dt/2 * k2, t_mid1, condition)
        
        t_end = torch.full((x.size(0),), t + dt, device=x.device)
        k4 = self.vector_field(x + dt * k3, t_end, condition)
        
        return x + dt/6 * (k1 + 2*k2 + 2*k3 + k4)
    
    def _sample_ode_adaptive(self, condition, x0, num_steps):
        """
        🎯 Adaptive ODE solver with error control
        """
        try:
            from scipy.integrate import solve_ivp
            import numpy as np
        except ImportError:
            print("⚠️ SciPy not available, falling back to fast_inverse")
            return self._sample_fast_inverse(condition, x0, num_steps)
        
        device = x0.device
        B = x0.size(0)
        
        def ode_func(t, x_flat):
            x_tensor = torch.from_numpy(x_flat.reshape(B, -1)).float().to(device)
            t_tensor = torch.full((B,), t, device=device)
            
            with torch.no_grad():
                v = self.vector_field(x_tensor, t_tensor, condition)
            
            return v.cpu().numpy().flatten()
        
        # Adaptive tolerances based on num_steps
        rtol = max(1e-3, 1e-2 / num_steps)  # Looser tolerance for speed
        atol = max(1e-6, 1e-5 / num_steps)
        
        # Solve ODE with adaptive stepping
        sol = solve_ivp(
            ode_func, 
            [0, 1], 
            x0.cpu().numpy().flatten(),
            method='DOP853',  # High-order adaptive method
            rtol=rtol,
            atol=atol,
            max_step=0.1  # Prevent too large steps
        )
        
        result = torch.from_numpy(sol.y[:, -1]).float().to(device)
        return result.view(B, self.dim)
    
    def sample_ode(self, condition, num_steps=50, method='euler'):
        """
        More sophisticated ODE integration
        """
        from scipy.integrate import solve_ivp
        import numpy as np
        
        B, T, _ = condition.shape
        device = condition.device
        
        x0 = torch.randn(B, T, self.dim, device=device)
        
        def ode_func(t, x):
            x_tensor = torch.from_numpy(x).float().to(device).view(B, T, self.dim)
            t_tensor = torch.full((B * T,), t, device=device)
            condition_flat = condition.view(-1, condition.size(-1))
            
            v = self.vector_field(x_tensor.view(-1, self.dim), t_tensor, condition_flat)
            return v.cpu().numpy().flatten()
        
        # Solve ODE
        sol = solve_ivp(
            ode_func, 
            [0, 1], 
            x0.cpu().numpy().flatten(),
            method='RK45',
            rtol=1e-5,
            atol=1e-8
        )
        
        result = torch.from_numpy(sol.y[:, -1]).float().to(device)
        return result.view(B, T, self.dim)

class VectorFieldNet(nn.Module):
    """Neural network for predicting vector field for waveform generation"""
    def __init__(self, dim=16384, condition_dim=768, hidden_dim=512):
        super().__init__()
        
        self.dim = dim
        self.condition_dim = condition_dim
        
        # Time embedding
        self.time_embedding = SinusoidalEmbedding(hidden_dim)
        
        # Waveform projection (compress high-dim waveform)
        self.waveform_proj = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # Condition projection
        self.condition_proj = nn.Linear(condition_dim, hidden_dim)
        
        # Main network - deeper for waveform complexity
        self.net = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, dim)  # Output full waveform velocity
        )
        
    def forward(self, x, t, condition):
        """
        Args:
            x: (B, dim) input waveform
            t: (B,) time
            condition: (B, condition_dim) conditioning
        """
        # Embeddings
        x_emb = self.waveform_proj(x)  # (B, hidden_dim)
        t_emb = self.time_embedding(t)  # (B, hidden_dim)  
        c_emb = self.condition_proj(condition)  # (B, hidden_dim)
        
        # Concatenate
        h = torch.cat([x_emb, t_emb, c_emb], dim=-1)  # (B, hidden_dim * 3)
        
        # Predict velocity
        v = self.net(h)  # (B, dim) - full waveform velocity!
        
        return v

class RetrievalModule(nn.Module):
    """Optional retrieval module for enhanced content features"""
    def __init__(self, feature_dim=768, k=5):
        super().__init__()
        self.feature_dim = feature_dim
        self.k = k
        
        # Feature enhancement network
        self.enhance_net = nn.Sequential(
            nn.Linear(feature_dim * 2, feature_dim),
            nn.SiLU(),
            nn.Linear(feature_dim, feature_dim)
        )
        
        # This would store training features in practice
        self.register_buffer('training_features', torch.empty(0, feature_dim))
        self.register_buffer('speaker_ids', torch.empty(0, dtype=torch.long))
        
    def add_training_features(self, features, speaker_ids):
        """Add features from training data"""
        self.training_features = torch.cat([self.training_features, features.detach()], dim=0)
        self.speaker_ids = torch.cat([self.speaker_ids, speaker_ids], dim=0)
    
    def enhance(self, content_features, target_speaker_id):
        """
        Enhance content features using retrieval
        Args:
            content_features: (B, feature_dim)
            target_speaker_id: (B,) target speaker IDs
        """
        if self.training_features.size(0) == 0:
            return content_features
        
        B = content_features.size(0)
        enhanced_features = []
        
        for i in range(B):
            query = content_features[i:i+1]  # (1, feature_dim)
            spk_id = target_speaker_id[i].item()
            
            # Find features from same speaker
            same_speaker_mask = (self.speaker_ids == spk_id)
            if same_speaker_mask.sum() > 0:
                candidate_features = self.training_features[same_speaker_mask]
                
                # Compute similarity (cosine similarity)
                similarities = F.cosine_similarity(
                    query.unsqueeze(1),  # (1, 1, feature_dim)
                    candidate_features.unsqueeze(0),  # (1, N, feature_dim)
                    dim=-1
                ).squeeze(0)  # (N,)
                
                # Get top-k most similar
                k = min(self.k, similarities.size(0))
                _, top_indices = similarities.topk(k)
                retrieved_features = candidate_features[top_indices].mean(dim=0, keepdim=True)  # (1, feature_dim)
                
                # Enhance with retrieved features
                combined = torch.cat([query, retrieved_features], dim=-1)  # (1, feature_dim * 2)
                enhanced = self.enhance_net(combined)  # (1, feature_dim)
                enhanced_features.append(enhanced)
            else:
                # No features from target speaker, use original
                enhanced_features.append(query)
        
        return torch.cat(enhanced_features, dim=0)  # (B, feature_dim)

class SinusoidalEmbedding(nn.Module):
    """Sinusoidal time embedding"""
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        
    def forward(self, t):
        """
        Args:
            t: (N,) time values in [0, 1]
        """
        device = t.device
        half_dim = self.dim // 2
        
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = t.unsqueeze(-1) * emb.unsqueeze(0)
        
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
        
        if self.dim % 2 == 1:
            emb = F.pad(emb, (0, 1))
            
        return emb