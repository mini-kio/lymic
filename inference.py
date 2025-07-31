#!/usr/bin/env python3
"""
Ultra-Optimized Voice Conversion Inference
- AMP FP16 mixed precision
- Rectified Flow fast sampling
- F0 conditional generation
- Chunk-based processing
- Compile optimization
"""

import torch
import torchaudio
import argparse
from pathlib import Path
import time
import numpy as np
from tqdm import tqdm
import warnings
from typing import Optional, Tuple, List

from model import VoiceConversionModel
from utils import extract_f0, normalize_f0

warnings.filterwarnings("ignore", category=UserWarning)

class OptimizedInferenceEngine:
    """
    Optimized inference engine
    - AMP FP16 optimization
    - Batch processing
    - Memory efficient
    - Dynamic chunk sizing
    """
    
    def __init__(self, model_path: str, device: str = 'auto', use_amp: bool = True,
                 compile_model: bool = True):
        
        self.device = self._setup_device(device)
        self.use_amp = use_amp and torch.cuda.is_available()
        
        print(f"Initializing OptimizedInferenceEngine:")
        print(f"   Device: {self.device}")
        print(f"   AMP FP16: {'Enabled' if self.use_amp else 'Disabled'}")
        
        self.model, self.config = self._load_model(model_path)
        
        if compile_model:
            self._compile_model()
        
        self.inference_times = []
        self.memory_usage = []
        
        print("Inference engine ready!")
    
    def _setup_device(self, device: str) -> torch.device:
        """Setup device"""
        if device == 'auto':
            if torch.cuda.is_available():
                device = 'cuda'
            else:
                device = 'cpu'
        
        device = torch.device(device)
        
        if device.type == 'cuda':
            torch.backends.cudnn.benchmark = True
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            print(f"CUDA optimizations enabled")
        
        return device
    
    def _load_model(self, model_path: str) -> Tuple[VoiceConversionModel, dict]:
        """Load model"""
        print(f"Loading model from {model_path}")
        
        checkpoint = torch.load(model_path, map_location=self.device)
        config = checkpoint.get('config', {})
        
        model = VoiceConversionModel(
            d_model=config.get('d_model', 768),
            ssm_layers=config.get('ssm_layers', 3),
            flow_steps=config.get('flow_steps', 20),
            n_speakers=config.get('n_speakers', 256),
            waveform_length=config.get('waveform_length', 16384),
            use_retrieval=config.get('use_retrieval', True),
            lora_rank=config.get('lora_rank', 16),
            adapter_dim=config.get('adapter_dim', 64),
            use_f0_conditioning=config.get('use_f0_conditioning', True)
        ).to(self.device)
        
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        
        # Convert to half precision for AMP
        if self.use_amp:
            model = model.half()
            print("Model converted to FP16")
        
        print("Model loaded successfully")
        return model, config
    
    def _compile_model(self):
        """Compile model"""
        try:
            print("Compiling model for optimization...")
            self.model.compile_model()
            print("Model compilation completed")
        except Exception as e:
            print(f"Warning: Model compilation failed: {e}")
    
    @torch.no_grad()
    @torch.cuda.amp.autocast()
    def convert_single_chunk(self, 
                           source_chunk: torch.Tensor,
                           target_speaker_id: int,
                           f0_chunk: Optional[torch.Tensor] = None,
                           vuv_chunk: Optional[torch.Tensor] = None,
                           semitone_shift: float = 0.0,
                           method: str = 'fast_rectified',
                           num_steps: int = 6) -> torch.Tensor:
        """
        Single chunk conversion (optimized)
        """
        if source_chunk.dim() == 2:
            source_chunk = source_chunk.unsqueeze(0)  # 배치 차원 추가
        
        target_speaker_tensor = torch.tensor([target_speaker_id], device=self.device)
        
        f0_target = None
        vuv_target = None
        
        if f0_chunk is not None and vuv_chunk is not None:
            f0_target = f0_chunk.unsqueeze(0) if f0_chunk.dim() == 1 else f0_chunk
            vuv_target = vuv_chunk.unsqueeze(0) if vuv_chunk.dim() == 1 else vuv_chunk
        
        with torch.cuda.amp.autocast(enabled=self.use_amp):
            result = self.model(
                source_waveform=source_chunk,
                target_speaker_id=target_speaker_tensor,
                f0_target=f0_target,
                vuv_target=vuv_target,
                semitone_shift=semitone_shift,
                training=False,
                inference_method=method,
                num_steps=num_steps
            )
        
        return result['converted_waveform'].squeeze(0)
    
    def load_and_preprocess_audio(self, audio_path: str, 
                                target_sample_rate: int = 44100) -> torch.Tensor:
        """Optimized audio loading and preprocessing"""
        print(f"Loading audio: {audio_path}")
        
        waveform, sr = torchaudio.load(audio_path)
        
        if sr != target_sample_rate:
            resampler = torchaudio.transforms.Resample(sr, target_sample_rate)
            waveform = resampler(waveform)
            print(f"Resampled: {sr}Hz -> {target_sample_rate}Hz")
        
        if waveform.size(0) == 1:
            waveform = waveform.repeat(2, 1)
        elif waveform.size(0) > 2:
            waveform = waveform[:2]
        
        waveform = waveform.to(self.device)
        
        if self.use_amp:
            waveform = waveform.half()
        
        duration = waveform.size(1) / target_sample_rate
        print(f"Audio loaded: {waveform.shape}, Duration: {duration:.2f}s")
        
        return waveform
    
    def extract_f0_features(self, waveform: torch.Tensor, 
                          sample_rate: int = 44100) -> Tuple[torch.Tensor, torch.Tensor]:
        """F0 feature extraction"""
        if waveform.dim() == 2:
            mono_waveform = waveform.mean(dim=0)
        else:
            mono_waveform = waveform
        
        audio_np = mono_waveform.cpu().float().numpy()
        
        f0, vuv = extract_f0(
            audio_np,
            sample_rate=sample_rate,
            hop_length=512,
            method='pyin'
        )
        
        f0_normalized = normalize_f0(f0, method='log')
        f0_tensor = torch.from_numpy(f0_normalized).to(self.device)
        vuv_tensor = torch.from_numpy(vuv).to(self.device)
        
        if self.use_amp:
            f0_tensor = f0_tensor.half()
            vuv_tensor = vuv_tensor.half()
        
        return f0_tensor, vuv_tensor
    
    def split_audio_with_f0(self, waveform: torch.Tensor, 
                          f0: torch.Tensor, vuv: torch.Tensor,
                          chunk_length: int = 16384, 
                          overlap: int = 2048) -> List[dict]:
        """Audio splitting with F0"""
        channels, total_length = waveform.shape
        step_size = chunk_length - overlap
        
        chunks = []
        
        start = 0
        while start < total_length:
            end = min(start + chunk_length, total_length)
            
            audio_chunk = waveform[:, start:end]
            
            if audio_chunk.size(1) < chunk_length:
                pad_length = chunk_length - audio_chunk.size(1)
                audio_chunk = torch.cat([
                    audio_chunk, 
                    torch.zeros(channels, pad_length, device=waveform.device, dtype=waveform.dtype)
                ], dim=1)
            hop_length = 512
            f0_start = start // hop_length
            f0_end = min(f0_start + chunk_length // hop_length + 1, f0.size(0))
            
            f0_chunk = f0[f0_start:f0_end]
            vuv_chunk = vuv[f0_start:f0_end]
            
            chunks.append({
                'audio': audio_chunk,
                'f0': f0_chunk,
                'vuv': vuv_chunk,
                'start': start,
                'end': end,
                'actual_length': min(chunk_length, total_length - start)
            })
            
            start += step_size
            
            if end >= total_length:
                break
        
        print(f"Split into {len(chunks)} chunks (overlap: {overlap})")
        return chunks
    
    def merge_chunks_with_crossfade(self, chunks: List[torch.Tensor], 
                                   positions: List[dict],
                                   overlap: int = 2048) -> torch.Tensor:
        """Merge chunks with crossfade"""
        if not chunks:
            return torch.empty(0)
        
        total_length = max(pos['end'] for pos in positions)
        merged_audio = torch.zeros(total_length, device=chunks[0].device, dtype=chunks[0].dtype)
        weights = torch.zeros(total_length, device=chunks[0].device, dtype=chunks[0].dtype)
        
        fade_length = min(overlap // 2, 512)
        
        for i, (chunk, pos) in enumerate(zip(chunks, positions)):
            start, end = pos['start'], pos['end']
            actual_length = pos['actual_length']
            
            if actual_length < chunk.size(0):
                chunk = chunk[:actual_length]
            
            actual_end = min(start + chunk.size(0), total_length)
            chunk_trimmed = chunk[:actual_end - start]
            
            if i > 0 and fade_length > 0:
                fade_in = torch.linspace(0, 1, fade_length, device=chunk.device, dtype=chunk.dtype)
                chunk_trimmed[:fade_length] *= fade_in
            
            if i < len(chunks) - 1 and fade_length > 0:
                fade_out = torch.linspace(1, 0, fade_length, device=chunk.device, dtype=chunk.dtype)
                chunk_trimmed[-fade_length:] *= fade_out
            
            merged_audio[start:actual_end] += chunk_trimmed
            weights[start:actual_end] += 1.0
        
        weights[weights == 0] = 1.0
        merged_audio = merged_audio / weights
        
        return merged_audio
    
    def convert_full_audio(self, 
                         audio_path: str,
                         target_speaker_id: int,
                         output_path: str,
                         chunk_length: int = 16384,
                         overlap: int = 2048,
                         method: str = 'fast_rectified',
                         num_steps: int = 6,
                         use_f0: bool = True,
                         semitone_shift: float = 0.0) -> dict:
        """
        Full audio conversion (optimized)
        """
        start_time = time.time()
        
        waveform = self.load_and_preprocess_audio(audio_path)
        
        f0, vuv = None, None
        if use_f0 and self.config.get('use_f0_conditioning', True):
            print("Extracting F0 features...")
            f0, vuv = self.extract_f0_features(waveform)
            print(f"F0 extracted: {f0.shape}")
        
        if use_f0 and f0 is not None:
            chunks = self.split_audio_with_f0(waveform, f0, vuv, chunk_length, overlap)
        else:
            chunks = self._split_audio_simple(waveform, chunk_length, overlap)
        
        print(f"Converting {len(chunks)} chunks...")
        converted_chunks = []
        
        for i, chunk_data in enumerate(tqdm(chunks, desc="Converting")):
            chunk_start_time = time.time()
            
            if use_f0 and f0 is not None:
                converted_chunk = self.convert_single_chunk(
                    chunk_data['audio'],
                    target_speaker_id,
                    chunk_data['f0'],
                    chunk_data['vuv'],
                    semitone_shift,
                    method,
                    num_steps
                )
            else:
                converted_chunk = self.convert_single_chunk(
                    chunk_data['audio'],
                    target_speaker_id,
                    None,
                    None,
                    semitone_shift,
                    method,
                    num_steps
                )
            
            converted_chunks.append(converted_chunk)
            
            chunk_time = time.time() - chunk_start_time
            self.inference_times.append(chunk_time)
            
            if torch.cuda.is_available():
                memory_mb = torch.cuda.memory_allocated(self.device) / 1024 / 1024
                self.memory_usage.append(memory_mb)
        
        print("Merging chunks...")
        positions = [{'start': c.get('start', 0), 'end': c.get('end', 0), 
                     'actual_length': c.get('actual_length', chunk_length)} 
                    for c in chunks]
        
        merged_audio = self.merge_chunks_with_crossfade(converted_chunks, positions, overlap)
        
        if merged_audio.dim() == 1:
            merged_audio = merged_audio.unsqueeze(0).repeat(2, 1)
        
        print(f"Saving to {output_path}")
        merged_audio_save = merged_audio.cpu().float()  # Convert to CPU + FP32 for saving
        torchaudio.save(output_path, merged_audio_save, 44100)
        
        total_time = time.time() - start_time
        duration = merged_audio.size(1) / 44100
        rtf = duration / total_time
        
        stats = {
            'duration': duration,
            'total_time': total_time,
            'rtf': rtf,
            'chunks': len(chunks),
            'method': method,
            'steps': num_steps,
            'f0_used': use_f0,
            'avg_chunk_time': sum(self.inference_times[-len(chunks):]) / len(chunks),
            'peak_memory_mb': max(self.memory_usage[-len(chunks):]) if self.memory_usage else 0
        }
        
        print(f"Conversion completed!")
        print(f"Stats:")
        print(f"   Duration: {duration:.2f}s")
        print(f"   Total time: {total_time:.2f}s")
        print(f"   RTF: {rtf:.2f}x")
        print(f"   Method: {method} ({num_steps} steps)")
        print(f"   F0 conditioning: {'Yes' if use_f0 else 'No'}")
        print(f"   Semitone shift: {semitone_shift:+.1f}")
        print(f"   Peak memory: {stats['peak_memory_mb']:.1f}MB")
        
        return stats
    
    def _split_audio_simple(self, waveform: torch.Tensor, 
                          chunk_length: int, overlap: int) -> List[dict]:
        """Simple audio splitting without F0"""
        channels, total_length = waveform.shape
        step_size = chunk_length - overlap
        
        chunks = []
        start = 0
        
        while start < total_length:
            end = min(start + chunk_length, total_length)
            
            audio_chunk = waveform[:, start:end]
            
            # 패딩
            if audio_chunk.size(1) < chunk_length:
                pad_length = chunk_length - audio_chunk.size(1)
                audio_chunk = torch.cat([
                    audio_chunk,
                    torch.zeros(channels, pad_length, device=waveform.device, dtype=waveform.dtype)
                ], dim=1)
            
            chunks.append({
                'audio': audio_chunk,
                'start': start,
                'end': end,
                'actual_length': min(chunk_length, total_length - start)
            })
            
            start += step_size
            
            if end >= total_length:
                break
        
        return chunks
    
    def benchmark_performance(self, audio_path: str, target_speaker_id: int,
                            methods: List[str] = None, 
                            step_counts: List[int] = None) -> dict:
        """Performance benchmark"""
        if methods is None:
            methods = ['fast_rectified', 'heun', 'euler']
        if step_counts is None:
            step_counts = [4, 6, 8, 12]
        
        print(f"Benchmarking performance...")
        
        results = {}
        
        for method in methods:
            for steps in step_counts:
                key = f"{method}_{steps}steps"
                print(f"\nTesting {key}...")
                
                try:
                    stats = self.convert_full_audio(
                        audio_path=audio_path,
                        target_speaker_id=target_speaker_id,
                        output_path=f"benchmark_{key}.wav",
                        method=method,
                        num_steps=steps,
                        use_f0=True
                    )
                    
                    results[key] = {
                        'rtf': stats['rtf'],
                        'total_time': stats['total_time'],
                        'avg_chunk_time': stats['avg_chunk_time'],
                        'peak_memory_mb': stats['peak_memory_mb']
                    }
                    
                    print(f"Success {key}: RTF={stats['rtf']:.2f}x")
                    
                except Exception as e:
                    print(f"Failed {key}: {e}")
                    results[key] = {'error': str(e)}
        
        return results

def main():
    parser = argparse.ArgumentParser(description='Optimized Voice Conversion Inference')
    parser.add_argument('--model', '-m', required=True, help='Model checkpoint path')
    parser.add_argument('--input', '-i', required=True, help='Input audio file')
    parser.add_argument('--output', '-o', required=True, help='Output audio file')
    parser.add_argument('--speaker', '-s', type=int, required=True, help='Target speaker ID')
    
    # 최적화 설정
    parser.add_argument('--device', default='auto', help='Device (auto/cuda/cpu)')
    parser.add_argument('--no-amp', action='store_true', help='Disable AMP FP16')
    parser.add_argument('--no-compile', action='store_true', help='Disable model compilation')
    parser.add_argument('--no-f0', action='store_true', help='Disable F0 conditioning')
    
    # 추론 설정
    parser.add_argument('--method', default='fast_rectified',
                       choices=['fast_rectified', 'heun', 'rk4', 'euler'],
                       help='Inference method')
    parser.add_argument('--steps', type=int, default=6, help='Number of inference steps')
    parser.add_argument('--chunk-length', type=int, default=16384, help='Chunk length')
    parser.add_argument('--overlap', type=int, default=2048, help='Chunk overlap')
    
    # Voice control settings
    parser.add_argument('--semitone-shift', type=float, default=0.0, 
                       help='Semitone shift (-12 to +12, singing voice conversion)')
    
    # Benchmark
    parser.add_argument('--benchmark', action='store_true', help='Run performance benchmark')
    
    args = parser.parse_args()
    
    model_path = Path(args.model)
    input_path = Path(args.input)
    output_path = Path(args.output)
    
    if not model_path.exists():
        print(f"Error: Model not found: {model_path}")
        return
    
    if not input_path.exists():
        print(f"Error: Input not found: {input_path}")
        return
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    print("Starting Optimized Voice Conversion")
    print(f"   Model: {model_path}")
    print(f"   Input: {input_path}")
    print(f"   Output: {output_path}")
    print(f"   Speaker: {args.speaker}")
    print(f"   Method: {args.method} ({args.steps} steps)")
    print(f"   F0 conditioning: {'Yes' if not args.no_f0 else 'No'}")
    print(f"   Semitone shift: {args.semitone_shift:+.1f}")
    
    engine = OptimizedInferenceEngine(
        model_path=str(model_path),
        device=args.device,
        use_amp=not args.no_amp,
        compile_model=not args.no_compile
    )
    
    if args.benchmark:
        results = engine.benchmark_performance(
            audio_path=str(input_path),
            target_speaker_id=args.speaker
        )
        
        print("\nBenchmark Results:")
        for key, stats in results.items():
            if 'error' not in stats:
                print(f"   {key}: RTF={stats['rtf']:.2f}x, Memory={stats['peak_memory_mb']:.1f}MB")
    else:
        stats = engine.convert_full_audio(
            audio_path=str(input_path),
            target_speaker_id=args.speaker,
            output_path=str(output_path),
            chunk_length=args.chunk_length,
            overlap=args.overlap,
            method=args.method,
            num_steps=args.steps,
            use_f0=not args.no_f0,
            semitone_shift=args.semitone_shift
        )

if __name__ == '__main__':
    main()

"""
Optimized Inference Examples:

# 1. 기본 빠른 변환
python inference.py -m model.pt -i input.wav -o output.wav -s 1

# 2. 고품질 변환 (더 많은 단계)
python inference.py -m model.pt -i input.wav -o output.wav -s 1 \\
                   --method heun --steps 12

# 3. 초고속 변환 (F0 없이)
python inference.py -m model.pt -i input.wav -o output.wav -s 1 \\
                   --method fast_rectified --steps 4 --no-f0

# 노래 변환 예시들
# 남성 -> 여성 (1옥타브 위)
python inference.py -m model.pt -i song.wav -o song_female.wav -s 1 \\
                   --semitone-shift +12

# 여성 -> 남성 (1옥타브 아래)  
python inference.py -m model.pt -i song.wav -o song_male.wav -s 2 \\
                   --semitone-shift -12

# 미세 조정 (+2 세미톤, 온음 위)
python inference.py -m model.pt -i song.wav -o song_higher.wav -s 1 \\
                   --semitone-shift +2.0

# 4. 성능 벤치마크
python inference.py -m model.pt -i input.wav -o output.wav -s 1 --benchmark

# 5. CPU 추론
python inference.py -m model.pt -i input.wav -o output.wav -s 1 \\
                   --device cpu --no-amp --no-compile

# 6. 메모리 절약 (작은 청크)
python inference.py -m model.pt -i input.wav -o output.wav -s 1 \\
                   --chunk-length 8192 --overlap 1024
"""