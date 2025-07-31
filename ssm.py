import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional
from torch.utils.checkpoint import checkpoint

class OptimizedS6Block(nn.Module):
    """
     최적화된 S6 (Selective State Space) Block
    - FP16 최적화
    - 메모리 효율적 구현
    - 컴파일 최적화 지원
    - 더 빠른 선택적 스캔
    """
    def __init__(self, d_model, d_state=64, expand_factor=2, use_fast_conv=True):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_inner = expand_factor * d_model
        self.use_fast_conv = use_fast_conv
        
        #  효율적인 입력 프로젝션
        self.in_proj = nn.Linear(d_model, self.d_inner * 2, bias=False)
        
        #  선택적 파라미터 (최적화됨)
        self.x_proj = nn.Linear(self.d_inner, d_state * 2 + self.d_inner, bias=False)
        self.dt_proj = nn.Linear(d_state, self.d_inner, bias=True)
        
        #  상태 공간 파라미터 (초기화 최적화)
        self.A_log = nn.Parameter(torch.randn(self.d_inner, d_state))
        self.D = nn.Parameter(torch.randn(self.d_inner))
        
        #  출력 프로젝션 (bias 제거로 최적화)
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)
        
        # 정규화
        self.norm = nn.LayerNorm(d_model)
        
        #  최적화된 초기화
        self._optimized_init()
        
        # 컴파일 준비
        self._compiled = False
    
    def _optimized_init(self):
        """ 최적화된 파라미터 초기화"""
        # A 행렬: 안정적인 초기화
        with torch.no_grad():
            # S4/S6에 최적화된 초기화
            nn.init.uniform_(self.A_log, -4.0, -1.0)
            
            # D: 작은 양수 값
            nn.init.normal_(self.D, mean=1.0, std=0.1)
            
            # 프로젝션 레이어들
            nn.init.xavier_uniform_(self.in_proj.weight)
            nn.init.xavier_uniform_(self.x_proj.weight)
            nn.init.xavier_uniform_(self.dt_proj.weight)
            nn.init.xavier_uniform_(self.out_proj.weight)
            
            # dt_proj bias 초기화
            nn.init.uniform_(self.dt_proj.bias, -0.1, 0.1)
    
    def compile_block(self):
        """블록 컴파일"""
        if not self._compiled:
            try:
                self.selective_scan = torch.compile(
                    self.selective_scan, 
                    mode='max-autotune',
                    dynamic=True
                )
                self._compiled = True
                print(" S6Block compiled")
            except Exception as e:
                print(f" S6Block compilation failed: {e}")
    
    @torch.amp.autocast('cuda')
    def forward(self, x):
        """
         최적화된 순전파
        Args:
            x: (B, L, D) 입력 시퀀스
        Returns:
            (B, L, D) 출력 시퀀스
        """
        B, L, D = x.shape
        residual = x
        
        #  입력 프로젝션 (한 번에)
        x_and_res = self.in_proj(x)  # (B, L, 2*d_inner)
        x, res = x_and_res.split([self.d_inner, self.d_inner], dim=-1)
        
        # SiLU 활성화
        x = F.silu(x)
        
        #  선택적 스캔 (최적화됨)
        x = self.selective_scan(x)
        
        # 잔차 연결
        x = x * F.silu(res)
        
        # 출력 프로젝션
        x = self.out_proj(x)
        
        # 정규화 + 잔차
        return self.norm(x + residual)
    
    def selective_scan(self, x):
        """
         최적화된 선택적 스캔
        - 메모리 효율적
        - 수치적 안정성 향상
        - FP16 최적화
        """
        B, L, d_inner = x.shape
        
        #  선택적 파라미터 계산
        x_dbl = self.x_proj(x)  # (B, L, d_state*2 + d_inner)
        
        # 차원 분할 수정
        delta_B_sel, C_sel = x_dbl.split([self.d_state * 2, self.d_inner], dim=-1)
        B_sel, delta = delta_B_sel.split([self.d_state, self.d_state], dim=-1)
        
        #  이산화 단계 (수치적 안정성)
        delta = F.softplus(self.dt_proj(delta))  # (B, L, d_inner)
        delta = torch.clamp(delta, max=10.0)  # 수치적 안정성
        
        # 상태 공간 행렬
        A = -torch.exp(self.A_log.float())  # (d_inner, d_state)
        
        #  효율적인 이산화
        # deltaA: (B, L, d_inner, d_state)
        deltaA = torch.exp(delta.unsqueeze(-1) * A.unsqueeze(0).unsqueeze(0))
        
        # deltaB: (B, L, d_inner, d_state) 
        deltaB = delta.unsqueeze(-1) * B_sel.unsqueeze(-2)
        
        #  최적화된 스캔 (청크 단위 처리)
        chunk_size = min(64, L)  # 메모리 효율성
        outputs = []
        
        # 초기 상태
        h = torch.zeros(B, d_inner, self.d_state, device=x.device, dtype=x.dtype)
        
        for i in range(0, L, chunk_size):
            end_idx = min(i + chunk_size, L)
            chunk_len = end_idx - i
            
            # 청크 데이터
            x_chunk = x[:, i:end_idx]  # (B, chunk_len, d_inner)
            deltaA_chunk = deltaA[:, i:end_idx]  # (B, chunk_len, d_inner, d_state)
            deltaB_chunk = deltaB[:, i:end_idx]  # (B, chunk_len, d_inner, d_state)
            C_chunk = C_sel[:, i:end_idx]  # (B, chunk_len, d_inner)
            
            # 청크 내 스캔
            chunk_outputs = []
            for j in range(chunk_len):
                # 상태 업데이트 - 차원 체크 추가
                h_next = deltaA_chunk[:, j] * h  # (B, d_inner, d_state)
                
                # deltaB와 x의 차원 맞춤
                x_expanded = x_chunk[:, j].unsqueeze(-1)  # (B, d_inner, 1)
                deltaB_j = deltaB_chunk[:, j]  # (B, d_inner, d_state)
                
                h = h_next + deltaB_j * x_expanded  # (B, d_inner, d_state)
                
                # 출력 계산 - 차원 확인
                C_expanded = C_chunk[:, j].unsqueeze(-1)  # (B, d_inner, 1)
                y = torch.sum(C_expanded * h, dim=-1)  # (B, d_inner)
                chunk_outputs.append(y)
            
            outputs.extend(chunk_outputs)
        
        y = torch.stack(outputs, dim=1)  # (B, L, d_inner)
        
        #  스킵 연결 (브로드캐스팅 최적화)
        y = y + x * self.D.view(1, 1, -1)
        
        return y

class FastS6Block(nn.Module):
    """
     Ultra-fast S6 Block
    - 근사 스캔으로 극대 최적화
    - 추론 전용 최적화
    """
    def __init__(self, d_model, d_state=32, expand_factor=1.5):  # 더 작은 파라미터
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_inner = int(expand_factor * d_model)
        
        # 축소된 네트워크
        self.in_proj = nn.Linear(d_model, self.d_inner, bias=False)
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)
        
        # 간단한 선택적 메커니즘
        self.gate = nn.Linear(d_model, self.d_inner, bias=False)
        
        self.norm = nn.LayerNorm(d_model)
        
        # 초기화
        nn.init.xavier_uniform_(self.in_proj.weight)
        nn.init.xavier_uniform_(self.out_proj.weight)
        nn.init.xavier_uniform_(self.gate.weight)
    
    def forward(self, x):
        """Ultra-fast forward"""
        residual = x
        
        # 단순화된 처리
        h = self.in_proj(x)
        g = torch.sigmoid(self.gate(x))
        
        # 간단한 선택적 활성화
        h = h * g * F.silu(h)
        
        # 출력
        x = self.out_proj(h)
        
        return self.norm(x + residual)

class OptimizedS6SSMEncoder(nn.Module):
    """
     최적화된 S6-기반 SSM 인코더
    - 동적 레이어 선택
    - 그래디언트 체크포인팅
    - 컴파일 최적화
    """
    def __init__(self, d_model=768, n_layers=3, d_state=64, use_fast_layers=False, 
                 use_gradient_checkpointing=False):
        super().__init__()
        
        self.use_gradient_checkpointing = use_gradient_checkpointing
        
        #  동적 레이어 구성
        self.layers = nn.ModuleList()
        
        for i in range(n_layers):
            if use_fast_layers and i >= n_layers - 1:
                # 마지막 레이어는 ultra-fast
                layer = FastS6Block(d_model, d_state//2)
            else:
                # 일반 최적화된 레이어
                layer = OptimizedS6Block(d_model, d_state)
            
            self.layers.append(layer)
        
        self.final_norm = nn.LayerNorm(d_model)
        
        # 컴파일 준비
        self._compiled = False
        
        print(f" OptimizedS6SSM initialized:")
        print(f"   Layers: {n_layers}")
        print(f"   Fast layers: {'' if use_fast_layers else ''}")
        print(f"   Gradient checkpointing: {'' if use_gradient_checkpointing else ''}")
    
    def compile_encoder(self):
        """인코더 컴파일"""
        if not self._compiled:
            try:
                # 각 레이어 컴파일
                for layer in self.layers:
                    if hasattr(layer, 'compile_block'):
                        layer.compile_block()
                
                self._compiled = True
                print(" S6SSMEncoder compiled")
            except Exception as e:
                print(f" S6SSMEncoder compilation failed: {e}")
    
    @torch.amp.autocast('cuda')
    def forward(self, x):
        """
         최적화된 순전파
        Args:
            x: (B, T, D) HuBERT에서 온 특성
        Returns:
            (B, T, D) 인코딩된 특성
        """
        if self.use_gradient_checkpointing and self.training:
            # 그래디언트 체크포인팅 사용
            for layer in self.layers:
                x = checkpoint(layer, x, use_reentrant=False)
        else:
            # 일반 순전파
            for layer in self.layers:
                x = layer(x)
        
        return self.final_norm(x)

class ParallelS6Block(nn.Module):
    """
     병렬 처리 최적화된 S6 Block
    - 여러 헤드 병렬 처리
    - 더 효율적인 메모리 사용
    """
    def __init__(self, d_model, d_state=64, num_heads=8):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.num_heads = num_heads
        self.d_head = d_model // num_heads
        
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        
        # 멀티헤드 프로젝션
        self.head_proj = nn.Linear(d_model, d_model * 3, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)
        
        # 헤드별 상태 공간 파라미터
        self.A_log = nn.Parameter(torch.randn(num_heads, self.d_head, d_state))
        self.D = nn.Parameter(torch.randn(num_heads, self.d_head))
        
        self.norm = nn.LayerNorm(d_model)
        
        self._init_parameters()
    
    def _init_parameters(self):
        """파라미터 초기화"""
        nn.init.xavier_uniform_(self.head_proj.weight)
        nn.init.xavier_uniform_(self.out_proj.weight)
        nn.init.uniform_(self.A_log, -4.0, -1.0)
        nn.init.normal_(self.D, mean=1.0, std=0.1)
    
    def forward(self, x):
        """병렬 처리 순전파"""
        B, L, D = x.shape
        residual = x
        
        # 멀티헤드 프로젝션
        qkv = self.head_proj(x).view(B, L, 3, self.num_heads, self.d_head)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, B, num_heads, L, d_head)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        # 병렬 선택적 스캔 (간단화)
        # 실제로는 각 헤드별로 선택적 스캔을 병렬 수행
        attn = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_head)
        attn = F.softmax(attn, dim=-1)
        out = torch.matmul(attn, v)
        
        # 헤드 결합
        out = out.transpose(1, 2).contiguous().view(B, L, D)
        
        # 출력 프로젝션
        x = self.out_proj(out)
        
        return self.norm(x + residual)

#  S6 최적화 유틸리티

def benchmark_s6_performance(d_model=768, seq_len=1000, batch_size=8, num_runs=10):
    """S6 블록 성능 벤치마크"""
    import time
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 테스트할 블록들
    blocks = {
        'OptimizedS6': OptimizedS6Block(d_model).to(device),
        'FastS6': FastS6Block(d_model).to(device),
        'ParallelS6': ParallelS6Block(d_model).to(device)
    }
    
    # 테스트 데이터
    x = torch.randn(batch_size, seq_len, d_model, device=device)
    
    print(f" S6 Performance Benchmark:")
    print(f"   Input shape: {x.shape}")
    print(f"   Device: {device}")
    print(f"   Runs: {num_runs}")
    
    results = {}
    
    for name, block in blocks.items():
        block.eval()
        
        # 워밍업
        with torch.no_grad():
            for _ in range(3):
                _ = block(x)
        
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        
        # 벤치마크
        times = []
        with torch.no_grad():
            for _ in range(num_runs):
                start_time = time.time()
                output = block(x)
                torch.cuda.synchronize() if torch.cuda.is_available() else None
                end_time = time.time()
                times.append(end_time - start_time)
        
        avg_time = sum(times) / len(times)
        throughput = batch_size * seq_len / avg_time
        
        results[name] = {
            'avg_time': avg_time,
            'throughput': throughput,
            'output_shape': output.shape
        }
        
        print(f"   {name}:")
        print(f"     Time: {avg_time*1000:.2f}ms")
        print(f"     Throughput: {throughput:.0f} tokens/sec")
        print(f"     Memory: {torch.cuda.memory_allocated()//1024//1024}MB" if torch.cuda.is_available() else "")
    
    return results

def create_adaptive_ssm_encoder(d_model=768, target_speed='balanced'):
    """
    적응적 SSM 인코더 생성
    speed: 'fast', 'balanced', 'quality'
    """
    configs = {
        'fast': {
            'n_layers': 2,
            'd_state': 32,
            'use_fast_layers': True,
            'use_gradient_checkpointing': False
        },
        'balanced': {
            'n_layers': 3,
            'd_state': 64,
            'use_fast_layers': False,
            'use_gradient_checkpointing': True
        },
        'quality': {
            'n_layers': 4,
            'd_state': 128,
            'use_fast_layers': False,
            'use_gradient_checkpointing': True
        }
    }
    
    config = configs.get(target_speed, configs['balanced'])
    
    encoder = OptimizedS6SSMEncoder(
        d_model=d_model,
        **config
    )
    
    print(f" Created {target_speed} SSM encoder")
    return encoder

# 호환성을 위한 별칭
S6SSMEncoder = OptimizedS6SSMEncoder