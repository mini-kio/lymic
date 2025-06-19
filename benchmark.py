"""
🚀 고급 성능 벤치마크 도구
flow_matching.py와 ssm.py 모듈들의 성능을 종합적으로 테스트
"""
import torch
import torch.nn as nn
import time
import psutil
import os
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt
import numpy as np

from flow_matching import RectifiedFlow, FlowScheduler
from ssm import OptimizedS6Block, FastS6Block, ParallelS6Block, create_adaptive_ssm_encoder

class PerformanceBenchmark:
    """🚀 종합 성능 벤치마크"""
    
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.results = {}
        
    def benchmark_flow_matching(self, 
                               dims=[1024, 4096, 16384],
                               batch_sizes=[2, 4, 8],
                               num_runs=10):
        """Flow Matching 성능 벤치마크"""
        print("🧪 Flow Matching Benchmark Started...")
        
        flow_results = {}
        
        for dim in dims:
            for batch_size in batch_sizes:
                key = f"dim_{dim}_batch_{batch_size}"
                print(f"  Testing {key}...")
                
                # 모델 초기화
                flow = RectifiedFlow(
                    dim=dim, 
                    condition_dim=768, 
                    hidden_dim=min(512, dim//32)
                ).to(self.device)
                
                # 테스트 데이터
                condition = torch.randn(batch_size, 768, device=self.device)
                x1 = torch.randn(batch_size, dim, device=self.device)
                
                # 워밍업
                for _ in range(3):
                    with torch.no_grad():
                        _ = flow.compute_loss(x1, condition)
                        _ = flow.sample(condition, num_steps=4)
                
                torch.cuda.synchronize() if torch.cuda.is_available() else None
                
                # 손실 계산 벤치마크
                loss_times = []
                for _ in range(num_runs):
                    start_time = time.time()
                    loss = flow.compute_loss(x1, condition)
                    loss.backward()
                    flow.zero_grad()
                    torch.cuda.synchronize() if torch.cuda.is_available() else None
                    loss_times.append(time.time() - start_time)
                
                # 샘플링 벤치마크
                sample_times = []
                with torch.no_grad():
                    for _ in range(num_runs):
                        start_time = time.time()
                        samples = flow.sample(condition, num_steps=4)
                        torch.cuda.synchronize() if torch.cuda.is_available() else None
                        sample_times.append(time.time() - start_time)
                
                # 메모리 사용량
                memory_mb = torch.cuda.memory_allocated() / 1024 / 1024 if torch.cuda.is_available() else 0
                
                flow_results[key] = {
                    'loss_time_avg': np.mean(loss_times),
                    'loss_time_std': np.std(loss_times),
                    'sample_time_avg': np.mean(sample_times),
                    'sample_time_std': np.std(sample_times),
                    'memory_mb': memory_mb,
                    'dim': dim,
                    'batch_size': batch_size
                }
                
                # 메모리 정리
                del flow, condition, x1
                torch.cuda.empty_cache() if torch.cuda.is_available() else None
        
        self.results['flow_matching'] = flow_results
        print("✅ Flow Matching Benchmark Completed")
        
    def benchmark_ssm_variants(self,
                              d_models=[256, 512, 768],
                              seq_lens=[100, 500, 1000],
                              batch_size=4,
                              num_runs=10):
        """SSM 변형들 성능 벤치마크"""
        print("🧪 SSM Variants Benchmark Started...")
        
        ssm_results = {}
        
        # 테스트할 SSM 블록들
        block_types = {
            'OptimizedS6': OptimizedS6Block,
            'FastS6': FastS6Block,
            'ParallelS6': ParallelS6Block
        }
        
        for d_model in d_models:
            for seq_len in seq_lens:
                for block_name, block_class in block_types.items():
                    key = f"{block_name}_d{d_model}_seq{seq_len}"
                    print(f"  Testing {key}...")
                    
                    try:
                        # 블록 초기화
                        if block_name == 'ParallelS6':
                            block = block_class(d_model, num_heads=min(8, d_model//64)).to(self.device)
                        else:
                            block = block_class(d_model).to(self.device)
                        
                        # 테스트 데이터
                        x = torch.randn(batch_size, seq_len, d_model, device=self.device)
                        
                        # 워밍업
                        for _ in range(3):
                            with torch.no_grad():
                                _ = block(x)
                        
                        torch.cuda.synchronize() if torch.cuda.is_available() else None
                        
                        # 순전파 벤치마크
                        forward_times = []
                        for _ in range(num_runs):
                            start_time = time.time()
                            output = block(x)
                            torch.cuda.synchronize() if torch.cuda.is_available() else None
                            forward_times.append(time.time() - start_time)
                        
                        # 역전파 포함 벤치마크
                        backward_times = []
                        for _ in range(num_runs):
                            x_grad = x.clone().requires_grad_(True)
                            start_time = time.time()
                            output = block(x_grad)
                            loss = output.sum()
                            loss.backward()
                            torch.cuda.synchronize() if torch.cuda.is_available() else None
                            backward_times.append(time.time() - start_time)
                            block.zero_grad()
                        
                        # 처리량 계산
                        tokens_per_sec = (batch_size * seq_len) / np.mean(forward_times)
                        
                        # 메모리 사용량
                        memory_mb = torch.cuda.memory_allocated() / 1024 / 1024 if torch.cuda.is_available() else 0
                        
                        ssm_results[key] = {
                            'forward_time_avg': np.mean(forward_times),
                            'forward_time_std': np.std(forward_times),
                            'backward_time_avg': np.mean(backward_times),
                            'backward_time_std': np.std(backward_times),
                            'tokens_per_sec': tokens_per_sec,
                            'memory_mb': memory_mb,
                            'd_model': d_model,
                            'seq_len': seq_len,
                            'block_type': block_name
                        }
                        
                        # 메모리 정리
                        del block, x
                        torch.cuda.empty_cache() if torch.cuda.is_available() else None
                        
                    except Exception as e:
                        print(f"  ⚠️ {key} failed: {e}")
                        continue
        
        self.results['ssm_variants'] = ssm_results
        print("✅ SSM Variants Benchmark Completed")
    
    def benchmark_memory_scaling(self):
        """메모리 스케일링 테스트"""
        print("🧪 Memory Scaling Benchmark Started...")
        
        if not torch.cuda.is_available():
            print("⚠️ CUDA not available, skipping memory benchmark")
            return
        
        memory_results = {}
        
        # Flow Matching 메모리 스케일링
        dims = [1024, 2048, 4096, 8192, 16384]
        batch_size = 4
        
        for dim in dims:
            torch.cuda.empty_cache()
            initial_memory = torch.cuda.memory_allocated()
            
            try:
                flow = RectifiedFlow(dim=dim, condition_dim=768).cuda()
                condition = torch.randn(batch_size, 768, device='cuda')
                x1 = torch.randn(batch_size, dim, device='cuda')
                
                # 순전파
                loss = flow.compute_loss(x1, condition)
                current_memory = torch.cuda.memory_allocated()
                
                memory_results[f'flow_dim_{dim}'] = {
                    'memory_mb': (current_memory - initial_memory) / 1024 / 1024,
                    'dim': dim,
                    'type': 'flow_matching'
                }
                
                del flow, condition, x1, loss
                
            except RuntimeError as e:
                if "out of memory" in str(e):
                    memory_results[f'flow_dim_{dim}'] = {
                        'memory_mb': float('inf'),
                        'dim': dim,
                        'type': 'flow_matching',
                        'oom': True
                    }
                    break
        
        # SSM 메모리 스케일링
        seq_lens = [100, 500, 1000, 2000, 4000]
        d_model = 768
        
        for seq_len in seq_lens:
            torch.cuda.empty_cache()
            initial_memory = torch.cuda.memory_allocated()
            
            try:
                encoder = create_adaptive_ssm_encoder(d_model=d_model, target_speed='fast').cuda()
                x = torch.randn(batch_size, seq_len, d_model, device='cuda')
                
                # 순전파
                output = encoder(x)
                current_memory = torch.cuda.memory_allocated()
                
                memory_results[f'ssm_seq_{seq_len}'] = {
                    'memory_mb': (current_memory - initial_memory) / 1024 / 1024,
                    'seq_len': seq_len,
                    'type': 'ssm'
                }
                
                del encoder, x, output
                
            except RuntimeError as e:
                if "out of memory" in str(e):
                    memory_results[f'ssm_seq_{seq_len}'] = {
                        'memory_mb': float('inf'),
                        'seq_len': seq_len,
                        'type': 'ssm',
                        'oom': True
                    }
                    break
        
        self.results['memory_scaling'] = memory_results
        print("✅ Memory Scaling Benchmark Completed")
    
    def generate_report(self):
        """성능 보고서 생성"""
        print("\n📊 Performance Benchmark Report")
        print("=" * 50)
        
        # Flow Matching 결과
        if 'flow_matching' in self.results:
            print("\n🔥 Flow Matching Performance:")
            flow_results = self.results['flow_matching']
            
            for key, result in flow_results.items():
                print(f"  {key}:")
                print(f"    Loss computation: {result['loss_time_avg']*1000:.2f}±{result['loss_time_std']*1000:.2f}ms")
                print(f"    Sampling (4 steps): {result['sample_time_avg']*1000:.2f}±{result['sample_time_std']*1000:.2f}ms")
                print(f"    Memory usage: {result['memory_mb']:.2f}MB")
        
        # SSM 결과
        if 'ssm_variants' in self.results:
            print("\n🚀 SSM Variants Performance:")
            ssm_results = self.results['ssm_variants']
            
            # 블록별 요약
            block_summary = {}
            for key, result in ssm_results.items():
                block_type = result['block_type']
                if block_type not in block_summary:
                    block_summary[block_type] = []
                block_summary[block_type].append(result['tokens_per_sec'])
            
            for block_type, throughputs in block_summary.items():
                avg_throughput = np.mean(throughputs)
                print(f"  {block_type}: {avg_throughput:.0f} tokens/sec (avg)")
        
        # 메모리 스케일링 결과
        if 'memory_scaling' in self.results:
            print("\n💾 Memory Scaling:")
            memory_results = self.results['memory_scaling']
            
            print("  Flow Matching:")
            for key, result in memory_results.items():
                if result['type'] == 'flow_matching':
                    if result.get('oom'):
                        print(f"    dim {result['dim']}: OOM")
                    else:
                        print(f"    dim {result['dim']}: {result['memory_mb']:.2f}MB")
            
            print("  SSM Encoder:")
            for key, result in memory_results.items():
                if result['type'] == 'ssm':
                    if result.get('oom'):
                        print(f"    seq_len {result['seq_len']}: OOM")
                    else:
                        print(f"    seq_len {result['seq_len']}: {result['memory_mb']:.2f}MB")
    
    def save_results(self, filename="benchmark_results.txt"):
        """결과를 파일로 저장"""
        with open(filename, 'w', encoding='utf-8') as f:
            f.write("🚀 Lymic Performance Benchmark Results\n")
            f.write(f"Device: {self.device}\n")
            f.write(f"Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("=" * 50 + "\n\n")
            
            # 결과를 텍스트로 저장
            import json
            for category, results in self.results.items():
                f.write(f"\n{category.upper()}:\n")
                for key, result in results.items():
                    f.write(f"  {key}: {json.dumps(result, indent=4)}\n")
        
        print(f"📄 Results saved to {filename}")

def quick_benchmark():
    """빠른 벤치마크 실행"""
    benchmark = PerformanceBenchmark()
    
    print("🚀 Quick Benchmark Starting...")
    
    # 작은 크기로 빠른 테스트
    benchmark.benchmark_flow_matching(
        dims=[1024, 4096], 
        batch_sizes=[2, 4], 
        num_runs=5
    )
    
    benchmark.benchmark_ssm_variants(
        d_models=[256, 512], 
        seq_lens=[100, 500], 
        num_runs=5
    )
    
    benchmark.benchmark_memory_scaling()
    
    benchmark.generate_report()
    benchmark.save_results("quick_benchmark.txt")
    
    return benchmark

if __name__ == "__main__":
    # 명령행 실행
    import argparse
    
    parser = argparse.ArgumentParser(description='Lymic Performance Benchmark')
    parser.add_argument('--quick', action='store_true', help='Run quick benchmark')
    parser.add_argument('--full', action='store_true', help='Run full benchmark')
    
    args = parser.parse_args()
    
    if args.quick or (not args.full):
        quick_benchmark()
    elif args.full:
        benchmark = PerformanceBenchmark()
        
        print("🚀 Full Benchmark Starting (this may take a while)...")
        
        benchmark.benchmark_flow_matching()
        benchmark.benchmark_ssm_variants()
        benchmark.benchmark_memory_scaling()
        
        benchmark.generate_report()
        benchmark.save_results("full_benchmark.txt")
