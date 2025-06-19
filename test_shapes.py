"""
Shape 테스트 코드 - flow_matching.py와 ssm.py
"""
import torch
import torch.nn as nn
from flow_matching import RectifiedFlow, RectifiedVectorField, OptimizedTimeEmbedding
from ssm import OptimizedS6Block, OptimizedS6SSMEncoder

def test_flow_matching_shapes():
    """Flow Matching 모듈들의 shape 테스트"""
    print("🧪 Testing Flow Matching Shapes...")
    
    # 테스트 파라미터
    batch_size = 4
    dim = 16384
    condition_dim = 768
    hidden_dim = 512
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    try:
        # 1. OptimizedTimeEmbedding 테스트
        time_emb = OptimizedTimeEmbedding(hidden_dim).to(device)
        t = torch.rand(batch_size, device=device)
        t_emb = time_emb(t)
        print(f"✅ TimeEmbedding: {t.shape} -> {t_emb.shape}")
        assert t_emb.shape == (batch_size, hidden_dim // 2), f"Expected {(batch_size, hidden_dim // 2)}, got {t_emb.shape}"
        
        # 2. RectifiedVectorField 테스트
        vector_field = RectifiedVectorField(dim=dim, condition_dim=condition_dim, hidden_dim=hidden_dim).to(device)
        x = torch.randn(batch_size, dim, device=device)
        condition = torch.randn(batch_size, condition_dim, device=device)
        
        velocity = vector_field(x, t, condition)
        print(f"✅ VectorField: x{x.shape} + t{t.shape} + c{condition.shape} -> {velocity.shape}")
        assert velocity.shape == (batch_size, dim), f"Expected {(batch_size, dim)}, got {velocity.shape}"
        
        # 3. RectifiedFlow 전체 테스트
        flow = RectifiedFlow(dim=dim, condition_dim=condition_dim, hidden_dim=hidden_dim).to(device)
        
        # 손실 계산 테스트
        x1 = torch.randn(batch_size, dim, device=device)
        loss = flow.compute_loss(x1, condition)
        print(f"✅ Flow Loss: {loss.item():.4f}")
        
        # 샘플링 테스트
        with torch.no_grad():
            samples = flow.sample(condition, num_steps=4)
            print(f"✅ Flow Sampling: {condition.shape} -> {samples.shape}")
            assert samples.shape == (batch_size, dim), f"Expected {(batch_size, dim)}, got {samples.shape}"
            
        print("✅ Flow Matching shapes: ALL PASSED")
        
    except Exception as e:
        print(f"❌ Flow Matching error: {e}")
        import traceback
        traceback.print_exc()

def test_ssm_shapes():
    """SSM 모듈들의 shape 테스트"""
    print("\n🧪 Testing SSM Shapes...")
    
    # 테스트 파라미터
    batch_size = 4
    seq_len = 100
    d_model = 768
    d_state = 64
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    try:
        # 1. OptimizedS6Block 테스트
        s6_block = OptimizedS6Block(d_model=d_model, d_state=d_state).to(device)
        x = torch.randn(batch_size, seq_len, d_model, device=device)
        
        output = s6_block(x)
        print(f"✅ S6Block: {x.shape} -> {output.shape}")
        assert output.shape == (batch_size, seq_len, d_model), f"Expected {(batch_size, seq_len, d_model)}, got {output.shape}"
          # 2. OptimizedS6SSMEncoder 테스트 - 최적화 기능 활성화
        encoder = OptimizedS6SSMEncoder(
            d_model=d_model, 
            n_layers=3, 
            d_state=d_state,
            use_fast_layers=True,  # ✅ Fast layers 활성화
            use_gradient_checkpointing=True  # ✅ Gradient checkpointing 활성화
        ).to(device)
        
        encoded = encoder(x)
        print(f"✅ S6Encoder: {x.shape} -> {encoded.shape}")
        assert encoded.shape == (batch_size, seq_len, d_model), f"Expected {(batch_size, seq_len, d_model)}, got {encoded.shape}"
        
        print("✅ SSM shapes: ALL PASSED")
        
    except Exception as e:
        print(f"❌ SSM error: {e}")
        import traceback
        traceback.print_exc()

def test_memory_usage():
    """메모리 사용량 테스트"""
    print("\n🧪 Testing Memory Usage...")
    
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        initial_memory = torch.cuda.memory_allocated()
          # 작은 모델로 테스트 - 최적화 기능들 활성화
        flow = RectifiedFlow(dim=1024, condition_dim=256, hidden_dim=128).cuda()
        encoder = OptimizedS6SSMEncoder(
            d_model=256, 
            n_layers=2, 
            d_state=32,
            use_fast_layers=True,  # ✅ Fast layers 활성화
            use_gradient_checkpointing=True  # ✅ Gradient checkpointing 활성화
        ).cuda()
        
        current_memory = torch.cuda.memory_allocated()
        print(f"✅ Memory usage: {(current_memory - initial_memory) / 1024 / 1024:.2f} MB")
        
        torch.cuda.empty_cache()
    else:
        print("✅ CPU mode - no GPU memory test")

if __name__ == "__main__":
    print("🚀 Shape Testing Started...")
    
    test_flow_matching_shapes()
    test_ssm_shapes()
    test_memory_usage()
    
    print("\n🎉 All shape tests completed!")
