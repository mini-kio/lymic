import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class RectifiedFlow(nn.Module):
    """
    🔥 Rectified Flow for Efficient Waveform Generation
    - 직선적 경로로 더 효율적인 학습
    - 적은 단계로도 고품질 생성
    - FP16 최적화 적용
    """
    def __init__(self, dim=16384, condition_dim=768, steps=20, hidden_dim=512):
        super().__init__()
        self.dim = dim
        self.condition_dim = condition_dim
        self.steps = steps
        
        # 🔥 경량화된 벡터 필드 네트워크
        self.vector_field = RectifiedVectorField(
            dim=dim,
            condition_dim=condition_dim,
            hidden_dim=hidden_dim
        )
        
        # 🚀 최적화 설정
        self._compiled = False
        
    def compile_model(self):
        """벡터 필드 컴파일"""
        if not self._compiled:
            try:
                self.vector_field = torch.compile(self.vector_field, mode='max-autotune')
                self._compiled = True
                print("🚀 RectifiedFlow compiled")
            except Exception as e:
                print(f"⚠️ RectifiedFlow compilation failed: {e}")
    
    def compute_loss(self, x1, condition):
        """
        🔥 Rectified Flow 손실 계산
        더 직선적인 경로로 학습 효율성 향상
        
        Args:
            x1: (B, dim) 타겟 파형
            condition: (B, condition_dim) 조건
        """
        B = x1.size(0)
        device = x1.device
        
        # 시간 샘플링 (균등 분포)
        t = torch.rand(B, device=device, dtype=x1.dtype)
        
        # 노이즈 샘플링 (가우시안)
        x0 = torch.randn_like(x1)
        
        # 🔥 Rectified Flow: 직선적 보간
        t_expanded = t.view(B, 1)
        x_t = (1 - t_expanded) * x0 + t_expanded * x1
        
        # 🔥 타겟 속도 (직선 경로)
        target_velocity = x1 - x0
        
        # 🔥 속도 예측
        predicted_velocity = self.vector_field(x_t, t, condition)
        
        # MSE 손실
        loss = F.mse_loss(predicted_velocity, target_velocity)
        
        return loss
    
    @torch.amp.autocast('cuda')
    def sample(self, condition, num_steps=None, x0=None, method='fast_rectified'):
        """
        🚀 최적화된 샘플링 - 여러 빠른 방법 지원
        """
        if num_steps is None:
            num_steps = max(4, self.steps // 5)  # 기본적으로 매우 빠르게
        
        B = condition.size(0)
        device = condition.device
        
        if x0 is None:
            x0 = torch.randn(B, self.dim, device=device, dtype=condition.dtype)
        
        # 방법에 따른 분기
        if method == 'fast_rectified':
            return self._sample_fast_rectified(condition, x0, num_steps)
        elif method == 'heun':
            return self._sample_heun(condition, x0, num_steps)
        elif method == 'rk4':
            return self._sample_rk4(condition, x0, num_steps)
        else:  # euler
            return self._sample_euler(condition, x0, num_steps)
    
    def _sample_fast_rectified(self, condition, x0, num_steps):
        """
        🚀 Ultra-fast Rectified Flow 샘플링
        - 적응적 단계 크기
        - 고차 정확도
        """
        # 적응적 단계 스케줄
        step_schedule = self._get_optimal_schedule(num_steps)
        
        x = x0
        t = 0.0
        
        for i, dt in enumerate(step_schedule):
            t_tensor = torch.full((x.size(0),), t, device=x.device, dtype=x.dtype)
            
            if i == 0:
                # 첫 단계: Euler
                v = self.vector_field(x, t_tensor, condition)
                x = x + dt * v
            elif i >= len(step_schedule) - 2:
                # 마지막 2단계: RK4로 정확도 향상
                x = self._rk4_step(x, t, dt, condition)
            else:
                # 중간 단계: Heun's method
                v1 = self.vector_field(x, t_tensor, condition)
                x_pred = x + dt * v1
                
                t_next = torch.full((x.size(0),), t + dt, device=x.device, dtype=x.dtype)
                v2 = self.vector_field(x_pred, t_next, condition)
                
                x = x + dt * 0.5 * (v1 + v2)
            
            t += dt
        
        return x
    
    def _get_optimal_schedule(self, num_steps):
        """최적화된 단계 스케줄"""
        # Rectified Flow에 최적화된 스케줄
        # 초기에는 큰 단계, 후반에는 작은 단계
        
        if num_steps <= 1:
            return [1.0]
        
        # 지수적 감소 + 선형 조합
        alpha = 0.2  # 감소율
        steps = []
        
        for i in range(num_steps):
            # 비선형 스케줄링
            progress = i / (num_steps - 1)
            
            # 초기에는 큰 단계, 후반에는 세밀한 단계
            weight = math.exp(-alpha * progress)
            
            steps.append(weight)
        
        # 정규화하여 총합이 1이 되도록
        total = sum(steps)
        steps = [s / total for s in steps]
        
        return steps
    
    def _sample_euler(self, condition, x0, num_steps):
        """표준 Euler 적분"""
        dt = 1.0 / num_steps
        x = x0
        
        for i in range(num_steps):
            t = torch.full((x.size(0),), i * dt, device=x.device, dtype=x.dtype)
            v = self.vector_field(x, t, condition)
            x = x + dt * v
        
        return x
    
    def _sample_heun(self, condition, x0, num_steps):
        """Heun's method (RK2)"""
        dt = 1.0 / num_steps
        x = x0
        
        for i in range(num_steps):
            t = torch.full((x.size(0),), i * dt, device=x.device, dtype=x.dtype)
            
            # Heun's method
            v1 = self.vector_field(x, t, condition)
            x_pred = x + dt * v1
            
            t_next = torch.full((x.size(0),), (i + 1) * dt, device=x.device, dtype=x.dtype)
            v2 = self.vector_field(x_pred, t_next, condition)
            
            x = x + dt * 0.5 * (v1 + v2)
        
        return x
    
    def _sample_rk4(self, condition, x0, num_steps):
        """4차 Runge-Kutta"""
        dt = 1.0 / num_steps
        x = x0
        
        for i in range(num_steps):
            t = i * dt
            x = self._rk4_step(x, t, dt, condition)
        
        return x
    
    def _rk4_step(self, x, t, dt, condition):
        """RK4 단일 단계"""
        t_tensor = torch.full((x.size(0),), t, device=x.device, dtype=x.dtype)
        k1 = self.vector_field(x, t_tensor, condition)
        
        t_mid1 = torch.full((x.size(0),), t + dt/2, device=x.device, dtype=x.dtype)
        k2 = self.vector_field(x + dt/2 * k1, t_mid1, condition)
        
        k3 = self.vector_field(x + dt/2 * k2, t_mid1, condition)
        
        t_end = torch.full((x.size(0),), t + dt, device=x.device, dtype=x.dtype)
        k4 = self.vector_field(x + dt * k3, t_end, condition)
        
        return x + dt/6 * (k1 + 2*k2 + 2*k3 + k4)

class RectifiedVectorField(nn.Module):
    """
    🔥 최적화된 벡터 필드 네트워크
    - FP16 최적화
    - 메모리 효율적 설계
    - 컴파일 최적화 적용
    """
    def __init__(self, dim=16384, condition_dim=768, hidden_dim=512):
        super().__init__()
        
        self.dim = dim
        self.condition_dim = condition_dim
        
        # 🔥 시간 임베딩 (최적화)
        self.time_embedding = OptimizedTimeEmbedding(hidden_dim)
        
        # 🔥 효율적인 파형 프로젝션
        self.waveform_proj = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.SiLU(),
            nn.Dropout(0.05),  # 낮은 드롭아웃
            nn.Linear(hidden_dim, hidden_dim // 4)
        )
        
        # 조건 프로젝션
        self.condition_proj = nn.Linear(condition_dim, hidden_dim // 4)
        
        # 🔥 최적화된 메인 네트워크
        self.net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Dropout(0.05),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, dim)  # 출력: 파형 속도
        )
        
        # 🔥 가중치 초기화 최적화
        self._initialize_weights()
        
    def _initialize_weights(self):
        """효율적인 가중치 초기화"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                # He 초기화 (SiLU에 최적화)
                nn.init.kaiming_normal_(module.weight, mode='fan_in', nonlinearity='relu')
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    @torch.amp.autocast('cuda')
    def forward(self, x, t, condition):
        """
        최적화된 순전파
        Args:
            x: (B, dim) 입력 파형
            t: (B,) 시간
            condition: (B, condition_dim) 조건
        """
        # 임베딩들
        x_emb = self.waveform_proj(x)  # (B, hidden_dim//4)
        t_emb = self.time_embedding(t)  # (B, hidden_dim//2)
        c_emb = self.condition_proj(condition)  # (B, hidden_dim//4)
        
        # 연결 - 차원 맞춤
        h = torch.cat([x_emb, t_emb, c_emb], dim=-1)  # (B, hidden_dim)
        
        # 속도 예측
        velocity = self.net(h)
        
        return velocity

class OptimizedTimeEmbedding(nn.Module):
    """최적화된 시간 임베딩"""
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        
        # 사전 계산된 상수
        half_dim = dim // 4  # hidden_dim의 1/2를 차지하도록 수정
        emb = math.log(10000) / (half_dim - 1) if half_dim > 1 else 0
        self.register_buffer('emb_scale', torch.exp(torch.arange(half_dim) * -emb))
        self.proj = nn.Linear(half_dim * 2, dim // 2)  # 최종 출력 차원 맞춤
        
    @torch.amp.autocast('cuda')
    def forward(self, t):
        """
        빠른 시간 임베딩
        Args:
            t: (N,) 시간 값 [0, 1]
        """
        if len(self.emb_scale) == 0:
            # half_dim이 0인 경우 처리
            return torch.zeros(t.size(0), self.dim // 2, device=t.device, dtype=t.dtype)
            
        emb = t.unsqueeze(-1) * self.emb_scale.unsqueeze(0)
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
        
        # 프로젝션을 통해 차원 맞춤
        emb = self.proj(emb)
            
        return emb

# 🔥 추가 최적화 유틸리티들
class FlowScheduler:
    """동적 스케줄링으로 추론 속도 최적화"""
    
    @staticmethod
    def get_adaptive_steps(quality_target='fast'):
        """품질 목표에 따른 적응적 단계 수"""
        schedules = {
            'ultra_fast': 3,
            'fast': 6,
            'balanced': 12,
            'high_quality': 20,
            'best': 30
        }
        return schedules.get(quality_target, 6)
    
    @staticmethod
    def get_progressive_schedule(max_steps, current_epoch, total_epochs):
        """훈련 중 점진적 단계 증가"""
        # 초기에는 적은 단계, 후반에는 많은 단계
        progress = current_epoch / total_epochs
        min_steps = max(2, max_steps // 10)
        steps = int(min_steps + (max_steps - min_steps) * progress)
        return min(steps, max_steps)

# 🚀 성능 최적화를 위한 컴파일 래퍼
def compile_rectified_flow(model):
    """RectifiedFlow 모델 컴파일"""
    try:
        if hasattr(torch, 'compile'):
            model.vector_field = torch.compile(
                model.vector_field, 
                mode='max-autotune',
                dynamic=True
            )
            print("🚀 RectifiedFlow vector field compiled")
        else:
            print("⚠️ torch.compile not available")
    except Exception as e:
        print(f"⚠️ Compilation failed: {e}")
    
    return model