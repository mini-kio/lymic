"""
간단한 벤치마크 실행 스크립트
"""
import torch
from flow_matching import RectifiedFlow
from ssm import OptimizedS6Block, FastS6Block

def simple_benchmark():
    """간단한 성능 테스트"""
    print(" Simple Performance Check")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    # Flow Matching 테스트
    print("\n Flow Matching Test:")
    flow = RectifiedFlow(dim=1024, condition_dim=768, hidden_dim=256).to(device)
    condition = torch.randn(2, 768, device=device)
    x1 = torch.randn(2, 1024, device=device)
    
    import time
    start = time.time()
    loss = flow.compute_loss(x1, condition)
    print(f"  Loss computation: {(time.time() - start)*1000:.2f}ms")
    
    start = time.time()
    with torch.no_grad():
        samples = flow.sample(condition, num_steps=4)
    print(f"  Sampling: {(time.time() - start)*1000:.2f}ms")
    
    # SSM 테스트
    print("\n SSM Test:")
    s6_block = OptimizedS6Block(d_model=256, d_state=32).to(device)
    fast_block = FastS6Block(d_model=256, d_state=16).to(device)
    
    x = torch.randn(2, 100, 256, device=device)
    
    start = time.time()
    out1 = s6_block(x)
    print(f"  OptimizedS6: {(time.time() - start)*1000:.2f}ms")
    
    start = time.time()
    out2 = fast_block(x)
    print(f"  FastS6: {(time.time() - start)*1000:.2f}ms")
    
    # 메모리 사용량
    if torch.cuda.is_available():
        memory_mb = torch.cuda.memory_allocated() / 1024 / 1024
        print(f"\n Memory usage: {memory_mb:.2f}MB")
    
    print("\n Simple benchmark completed!")

if __name__ == "__main__":
    simple_benchmark()
