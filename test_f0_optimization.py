#!/usr/bin/env python3
"""
F0 캐시 최적화 테스트 스크립트
"""
import numpy as np
import torch
import time
from utils import (
    extract_f0, extract_f0_crepe, extract_f0_fast_pyin, extract_f0_hybrid,
    GPUAcceleratedF0Cache, print_f0_methods, get_f0_config,
    CREPE_AVAILABLE, SCIPY_AVAILABLE
)

def test_f0_methods():
    """F0 추출 방법들 성능 테스트"""
    print("Testing F0 extraction methods...")
    print_f0_methods()
    
    # 테스트 오디오 생성 (5초, 44.1kHz)
    sample_rate = 44100
    duration = 5.0
    t = np.linspace(0, duration, int(sample_rate * duration))
    
    # 복합 음성 신호 (F0 변화)
    f0_base = 220  # A3
    f0_variation = 50 * np.sin(2 * np.pi * 2 * t)  # 2Hz 변조
    audio = np.sin(2 * np.pi * (f0_base + f0_variation) * t)
    
    # 노이즈 추가 (현실적인 음성)
    noise = np.random.normal(0, 0.05, len(audio))
    audio = audio + noise
    
    print(f"\nTest audio: {duration}s, {sample_rate}Hz, {len(audio)} samples")
    
    methods_to_test = ['pyin']
    if CREPE_AVAILABLE:
        methods_to_test.extend(['crepe_tiny', 'crepe_small', 'hybrid'])
    
    results = {}
    
    for method in methods_to_test:
        print(f"\nTesting {method}...")
        
        start_time = time.time()
        try:
            f0, vuv = extract_f0(audio, sample_rate, method=method)
            elapsed = time.time() - start_time
            
            # 통계 계산
            voiced_frames = np.sum(vuv > 0.5)
            mean_f0 = np.mean(f0[vuv > 0.5]) if voiced_frames > 0 else 0
            
            results[method] = {
                'time': elapsed,
                'voiced_frames': voiced_frames,
                'mean_f0': mean_f0,
                'success': True
            }
            
            print(f"   Success: {elapsed:.3f}s")
            print(f"   Voiced frames: {voiced_frames}/{len(f0)} ({100*voiced_frames/len(f0):.1f}%)")
            print(f"   Mean F0: {mean_f0:.1f}Hz")
            
        except Exception as e:
            results[method] = {
                'time': float('inf'),
                'error': str(e),
                'success': False
            }
            print(f"   Failed: {e}")
    
    # 성능 비교
    print(f"\nPerformance Comparison:")
    print(f"{'Method':<12} {'Time (s)':<10} {'Speed':<10} {'Accuracy':<10}")
    print("-" * 50)
    
    fastest_time = min(r['time'] for r in results.values() if r['success'])
    
    for method, result in results.items():
        if result['success']:
            speedup = fastest_time / result['time']
            print(f"{method:<12} {result['time']:<10.3f} {speedup:<10.2f}x {'OK':<10}")
        else:
            print(f"{method:<12} {'FAILED':<10} {'-':<10} {'FAIL':<10}")
    
    return results

def test_gpu_cache():
    """GPU 캐시 시스템 테스트"""
    print(f"\nTesting GPU F0 Cache...")
    
    # 가짜 오디오 파일 배치 생성
    batch_size = 8
    audio_batch = []
    
    for i in range(batch_size):
        # 각각 다른 길이와 F0의 테스트 오디오
        duration = 2.0 + i * 0.5  # 2-5.5초
        sample_rate = 44100
        t = np.linspace(0, duration, int(sample_rate * duration))
        
        f0 = 200 + i * 20  # 200-340Hz
        audio = np.sin(2 * np.pi * f0 * t) * np.exp(-t/5)  # 감쇠
        audio_batch.append(audio)
    
    # GPU 캐시 초기화
    gpu_cache = GPUAcceleratedF0Cache(
        use_gpu=torch.cuda.is_available(),
        batch_size=batch_size//2,  # 작은 배치로 테스트
        model_capacity='small'
    )
    
    print(f"Testing batch F0 extraction...")
    start_time = time.time()
    
    try:
        results = gpu_cache.extract_batch_f0_gpu(audio_batch)
        elapsed = time.time() - start_time
        
        print(f"Batch processing successful:")
        print(f"   Files: {len(audio_batch)}")
        print(f"   Time: {elapsed:.3f}s")
        print(f"   Avg per file: {elapsed/len(audio_batch)*1000:.1f}ms")
        
        # 결과 검증
        for i, (f0, vuv) in enumerate(results):
            voiced_ratio = np.mean(vuv > 0.5)
            mean_f0 = np.mean(f0[vuv > 0.5]) if np.any(vuv > 0.5) else 0
            print(f"   File {i+1}: {len(f0)} frames, {voiced_ratio:.2f} voiced, {mean_f0:.1f}Hz mean F0")
        
        return True
        
    except Exception as e:
        print(f"Batch processing failed: {e}")
        return False

def main():
    """메인 테스트 함수"""
    print("F0 Cache Optimization Test")
    print("=" * 50)
    
    # 시스템 정보
    print(f"CREPE available: {'YES' if CREPE_AVAILABLE else 'NO'}")
    print(f"SciPy available: {'YES' if SCIPY_AVAILABLE else 'NO'}")
    print(f"CUDA available: {'YES' if torch.cuda.is_available() else 'NO'}")
    
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name()}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory/1e9:.1f}GB")
    
    print("\n" + "="*50)
    
    # 1. F0 방법들 테스트
    method_results = test_f0_methods()
    
    # 2. GPU 캐시 테스트
    if CREPE_AVAILABLE or torch.cuda.is_available():
        gpu_success = test_gpu_cache()
    else:
        print("\nSkipping GPU cache test (CREPE/CUDA not available)")
        gpu_success = False
    
    # 3. 결과 요약
    print(f"\nTest Summary:")
    print(f"   Method tests: {len([r for r in method_results.values() if r['success']])}/{len(method_results)} passed")
    print(f"   GPU cache test: {'PASS' if gpu_success else 'SKIP'}")
    
    # 4. 권장사항
    print(f"\nRecommendations:")
    if CREPE_AVAILABLE:
        if torch.cuda.is_available():
            print("   - Use 'crepe_small' with GPU acceleration for best balance")
            print("   - Use 'hybrid' for highest quality")
        else:
            print("   - Use 'crepe_small' for good accuracy")
    else:
        print("   - Install CREPE for better performance: pip install crepe tensorflow")
    print("   - Use 'pyin' as fallback on CPU-only systems")

if __name__ == "__main__":
    main()