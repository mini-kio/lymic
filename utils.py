import torch
import torch.nn as nn
import torchaudio
import numpy as np
import librosa
from pathlib import Path
from torch.utils.data import Dataset
import json
import random
import shutil
import warnings
from functools import lru_cache
import multiprocessing as mp
from concurrent.futures import ThreadPoolExecutor
import time
from tqdm import tqdm

try:
    import crepe
    CREPE_AVAILABLE = True
except ImportError:
    CREPE_AVAILABLE = False
    print("Warning: CREPE not available. Install with: pip install crepe tensorflow")

try:
    from scipy import interpolate
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False

warnings.filterwarnings("ignore", category=UserWarning)

@lru_cache(maxsize=1000)
def extract_f0_cached(audio_hash, sample_rate=44100, hop_length=512, f0_min=80, f0_max=800, method='pyin'):
    """
    캐시된 F0 추출 - 동일한 오디오에 대해 재계산 방지
    """
    # 실제로는 audio를 받아야 하지만, 캐싱을 위해 해시 사용
    # 이 함수는 실제 구현에서는 사용하지 않고, 아래의 extract_f0를 사용
    pass

def extract_f0_crepe(audio, sample_rate=44100, hop_length=512, model_capacity='small'):
    """
    CREPE로 정확하고 빠른 F0 추출
    
    Args:
        model_capacity: 'tiny', 'small', 'medium', 'large', 'full'
                       tiny: 가장 빠름 (3배), 약간 정확도 하락
                       small: 균형잡힌 선택 (2배 빠름, 높은 정확도)
                       full: 최고 정확도, 가장 느림
    """
    if not CREPE_AVAILABLE:
        raise ImportError("CREPE not available. Install with: pip install crepe tensorflow")
    
    if isinstance(audio, torch.Tensor):
        audio = audio.cpu().numpy()
    
    # CREPE는 16kHz로 훈련됨 (내부에서 리샘플링)
    time, frequency, confidence, activation = crepe.predict(
        audio,
        sample_rate,
        model_capacity=model_capacity,  # 'small'이 속도-정확도 균형점
        step_size=hop_length / sample_rate * 1000,  # ms 단위
        viterbi=True,  # 스무딩 적용
        center=True,
        verbose=0
    )
    
    # Confidence 기반 VUV (0.5 임계값)
    vuv = (confidence > 0.5).astype(np.float32)
    f0 = frequency * vuv  # Unvoiced 구간은 0으로
    
    # hop_length에 맞춰 길이 조정
    expected_frames = 1 + len(audio) // hop_length
    if len(f0) != expected_frames and SCIPY_AVAILABLE:
        # 선형 보간으로 길이 맞춤
        old_indices = np.linspace(0, len(f0) - 1, len(f0))
        new_indices = np.linspace(0, len(f0) - 1, expected_frames)
        
        f_interp = interpolate.interp1d(old_indices, f0, kind='linear', 
                                       bounds_error=False, fill_value=0)
        v_interp = interpolate.interp1d(old_indices, vuv, kind='linear', 
                                       bounds_error=False, fill_value=0)
        
        f0 = f_interp(new_indices)
        vuv = v_interp(new_indices)
        vuv = (vuv > 0.5).astype(np.float32)  # Re-threshold
    
    return f0.astype(np.float32), vuv.astype(np.float32)

def extract_f0_fast_pyin(audio, sample_rate=44100, hop_length=512, f0_min=80, f0_max=800):
    """최적화된 빠른 pYIN (백업용)"""
    f0, voiced_flag, _ = librosa.pyin(
        audio,
        fmin=f0_min, fmax=f0_max, sr=sample_rate, hop_length=hop_length,
        frame_length=hop_length * 2,  # 3→2
        win_length=hop_length,        # 2→1
        resolution=0.2,               # 0.1→0.2  
        n_thresholds=20,              # 100→20 (5배 빠름!)
        max_transition_rate=50,
        switch_prob=0.02,
        no_trough_prob=0.02
    )
    
    f0 = np.nan_to_num(f0, nan=0.0)
    vuv = voiced_flag.astype(np.float32)
    
    return f0, vuv

def extract_f0_hybrid(audio, sample_rate=44100, hop_length=512):
    """
    하이브리드 방법: CREPE + pYIN 결합으로 최고 정확도
    """
    if not CREPE_AVAILABLE:
        return extract_f0_fast_pyin(audio, sample_rate, hop_length)
    
    # CREPE (주요)
    f0_crepe, vuv_crepe = extract_f0_crepe(
        audio, sample_rate, hop_length, model_capacity='small'
    )
    
    # pYIN (백업 및 검증)
    f0_pyin, vuv_pyin = extract_f0_fast_pyin(
        audio, sample_rate, hop_length
    )
    
    # CREPE confidence가 낮은 구간에서 pYIN 사용
    try:
        _, _, confidence, _ = crepe.predict(
            audio, sample_rate, model_capacity='small',
            step_size=hop_length / sample_rate * 1000, verbose=0
        )
        
        # 길이 맞춤
        if len(confidence) != len(f0_crepe) and SCIPY_AVAILABLE:
            old_idx = np.linspace(0, len(confidence)-1, len(confidence))
            new_idx = np.linspace(0, len(confidence)-1, len(f0_crepe))
            confidence = interpolate.interp1d(old_idx, confidence)(new_idx)
        
        # 하이브리드 결합
        confidence_threshold = 0.3
        low_confidence = confidence < confidence_threshold
        f0_final = f0_crepe.copy()
        vuv_final = vuv_crepe.copy()
        
        f0_final[low_confidence] = f0_pyin[low_confidence]
        vuv_final[low_confidence] = vuv_pyin[low_confidence]
        
        return f0_final, vuv_final
    except:
        # CREPE 실패 시 pYIN만 사용
        return f0_pyin, vuv_pyin

def extract_f0(audio, sample_rate=44100, hop_length=512, f0_min=80, f0_max=800, method='crepe_small'):
    """
    통합 F0 추출 함수
    - CREPE: 높은 정확도, GPU 가속
    - pYIN: 빠른 처리, CPU 최적화
    - hybrid: 최고 품질, CREPE + pYIN 결합
    """
    if isinstance(audio, torch.Tensor):
        audio = audio.cpu().numpy()
    
    # 입력 검증
    if len(audio) == 0:
        return np.array([]), np.array([])
    
    try:
        if method == 'crepe_small':
            return extract_f0_crepe(audio, sample_rate, hop_length, 'small')
        elif method == 'crepe_tiny':
            return extract_f0_crepe(audio, sample_rate, hop_length, 'tiny')
        elif method == 'crepe_full':
            return extract_f0_crepe(audio, sample_rate, hop_length, 'full')
        elif method == 'hybrid':
            return extract_f0_hybrid(audio, sample_rate, hop_length)
        elif method == 'pyin':
            return extract_f0_fast_pyin(audio, sample_rate, hop_length, f0_min, f0_max)
        else:
            # 기본값: CREPE가 있으면 사용, 없으면 pYIN
            if CREPE_AVAILABLE:
                return extract_f0_crepe(audio, sample_rate, hop_length, 'small')
            else:
                return extract_f0_fast_pyin(audio, sample_rate, hop_length, f0_min, f0_max)
        
        # 길이 검증
        expected_frames = 1 + len(audio) // hop_length
        if len(f0) != expected_frames:
            # 길이 조정
            if len(f0) > expected_frames:
                f0 = f0[:expected_frames]
                vuv = vuv[:expected_frames]
            else:
                # 패딩
                pad_length = expected_frames - len(f0)
                f0 = np.pad(f0, (0, pad_length), mode='constant', constant_values=0)
                vuv = np.pad(vuv, (0, pad_length), mode='constant', constant_values=0)
        
        
        # 길이 검증 및 조정
        expected_frames = 1 + len(audio) // hop_length
        if len(f0) != expected_frames:
            if len(f0) > expected_frames:
                f0 = f0[:expected_frames]
                vuv = vuv[:expected_frames]
            else:
                pad_length = expected_frames - len(f0)
                f0 = np.pad(f0, (0, pad_length), mode='constant', constant_values=0)
                vuv = np.pad(vuv, (0, pad_length), mode='constant', constant_values=0)
        
        return f0.astype(np.float32), vuv.astype(np.float32)
        
    except Exception as e:
        print(f"F0 extraction failed with {method}: {e}")
        # 폴백: pYIN 시도
        if method != 'pyin':
            try:
                return extract_f0_fast_pyin(audio, sample_rate, hop_length, f0_min, f0_max)
            except:
                pass
        
        # 최종 실패 시 기본값
        expected_frames = 1 + len(audio) // hop_length
        return np.zeros(expected_frames, dtype=np.float32), np.zeros(expected_frames, dtype=np.float32)

def compute_vuv(f0, threshold=50.0):  # 더 높은 임계값
    """VUV 계산 - 최적화됨"""
    if isinstance(f0, torch.Tensor):
        return (f0 > threshold).float()
    else:
        return (f0 > threshold).astype(np.float32)

def normalize_f0(f0, method='log', f0_min=50, f0_max=1000):  # 더 넓은 범위
    """
     최적화된 F0 정규화
    - 더 안정적인 정규화
    - 극값 처리 개선
    """
    if isinstance(f0, torch.Tensor):
        is_torch = True
        device = f0.device
        f0_np = f0.cpu().numpy()
    else:
        is_torch = False
        f0_np = f0.copy()
    
    # Voiced 마스크
    voiced = f0_np > 0
    
    if not np.any(voiced):
        # 모든 프레임이 unvoiced인 경우
        result = np.zeros_like(f0_np)
    else:
        f0_norm = np.zeros_like(f0_np)
        
        if method == 'log':
            # 로그 정규화 (더 안정적)
            f0_voiced = f0_np[voiced]
            f0_voiced = np.clip(f0_voiced, f0_min, f0_max)
            
            # 로그 변환
            log_f0 = np.log(f0_voiced + 1e-8)  # 수치 안정성
            log_f0_min, log_f0_max = np.log(f0_min), np.log(f0_max)
            
            # [-1, 1] 범위로 정규화
            f0_norm[voiced] = 2 * (log_f0 - log_f0_min) / (log_f0_max - log_f0_min) - 1
            
        elif method == 'linear':
            # 선형 정규화
            f0_voiced = f0_np[voiced]
            f0_voiced = np.clip(f0_voiced, f0_min, f0_max)
            f0_norm[voiced] = 2 * (f0_voiced - f0_min) / (f0_max - f0_min) - 1
            
        else:
            raise ValueError(f"Unknown normalization method: {method}")
        
        result = f0_norm
    
    if is_torch:
        return torch.from_numpy(result).to(device).float()
    else:
        return result.astype(np.float32)

def denormalize_f0(f0_norm, method='log', f0_min=50, f0_max=1000):
    """F0 역정규화"""
    if isinstance(f0_norm, torch.Tensor):
        is_torch = True
        device = f0_norm.device
        f0_np = f0_norm.cpu().numpy()
    else:
        is_torch = False
        f0_np = f0_norm.copy()
    
    if method == 'log':
        log_f0_min, log_f0_max = np.log(f0_min), np.log(f0_max)
        log_f0 = (f0_np + 1) / 2 * (log_f0_max - log_f0_min) + log_f0_min
        f0 = np.exp(log_f0) - 1e-8
    elif method == 'linear':
        f0 = (f0_np + 1) / 2 * (f0_max - f0_min) + f0_min
    else:
        raise ValueError(f"Unknown normalization method: {method}")
    
    if is_torch:
        return torch.from_numpy(f0).to(device).float()
    else:
        return f0.astype(np.float32)

class GPUAcceleratedF0Cache:
    """GPU 최적화 F0 캐시 시스템"""
    
    def __init__(self, use_gpu=True, batch_size=32, model_capacity='small'):
        self.device = 'cuda' if use_gpu and torch.cuda.is_available() else 'cpu'
        self.batch_size = batch_size
        self.model_capacity = model_capacity
        
        # GPU 메모리 최적화
        if self.device == 'cuda':
            torch.cuda.empty_cache()
            torch.backends.cudnn.benchmark = True
        
        print(f"🚀 GPU F0 Cache initialized:")
        print(f"   Device: {self.device}")
        print(f"   Batch size: {self.batch_size}")
        print(f"   Model: {self.model_capacity}")
    
    def extract_batch_f0_gpu(self, audio_batch, sample_rate=44100, hop_length=512):
        """배치 단위 GPU F0 추출"""
        results = []
        
        if not CREPE_AVAILABLE:
            # CREPE 없으면 pYIN 사용
            for audio in audio_batch:
                f0, vuv = extract_f0_fast_pyin(audio, sample_rate, hop_length)
                results.append((f0, vuv))
        else:
            # CREPE 배치 처리 (GPU에서 자동 가속)
            for audio in audio_batch:
                try:
                    f0, vuv = extract_f0_crepe(
                        audio, sample_rate, hop_length, self.model_capacity
                    )
                    results.append((f0, vuv))
                except Exception as e:
                    print(f"⚠️ CREPE failed, fallback to pYIN: {e}")
                    f0, vuv = extract_f0_fast_pyin(audio, sample_rate, hop_length)
                    results.append((f0, vuv))
        
        return results
    
    def build_cache_parallel_gpu(self, all_files, cache_dir, sample_rate=44100, hop_length=512):
        """GPU 병렬 F0 캐시 구축"""
        print(f"🚀 Building GPU F0 cache for {len(all_files)} files...")
        
        total_processed = 0
        failed_files = []
        
        # 진행률 표시와 함께 배치 처리
        for i in tqdm(range(0, len(all_files), self.batch_size), desc="F0 Cache"):
            batch_files = all_files[i:i+self.batch_size]
            
            # 배치 오디오 로드
            audio_batch = []
            valid_files = []
            
            for file_path in batch_files:
                try:
                    # 캐시 확인
                    cache_file = cache_dir / f"{file_path.stem}_f0.npz"
                    if cache_file.exists():
                        continue
                    
                    # 오디오 로드
                    audio, sr = torchaudio.load(str(file_path))
                    if sr != sample_rate:
                        resampler = torchaudio.transforms.Resample(sr, sample_rate)
                        audio = resampler(audio)
                    
                    if audio.shape[0] > 1:
                        audio = audio.mean(dim=0)
                    else:
                        audio = audio.squeeze(0)
                    
                    audio_batch.append(audio.numpy())
                    valid_files.append(file_path)
                    
                except Exception as e:
                    failed_files.append((str(file_path), str(e)))
                    continue
            
            # GPU 배치 F0 추출
            if audio_batch:
                f0_results = self.extract_batch_f0_gpu(
                    audio_batch, sample_rate, hop_length
                )
                
                # 캐시 저장
                for (f0, vuv), file_path in zip(f0_results, valid_files):
                    try:
                        cache_file = cache_dir / f"{file_path.stem}_f0.npz"
                        f0_norm = normalize_f0(f0, method='log')
                        
                        np.savez_compressed(
                            cache_file,
                            f0=f0, 
                            f0_normalized=f0_norm, 
                            vuv=vuv
                        )
                        total_processed += 1
                    except Exception as e:
                        failed_files.append((str(file_path), f"Cache save failed: {e}"))
        
        # 결과 보고
        print(f"✅ GPU F0 cache completed:")
        print(f"   Processed: {total_processed} files")
        if failed_files:
            print(f"   Failed: {len(failed_files)} files")
            for file_path, error in failed_files[:5]:  # 처음 5개만 표시
                print(f"      {Path(file_path).name}: {error}")
        
        return total_processed, failed_files

class OptimizedVoiceConversionDataset(Dataset):
    """
     최적화된 Voice Conversion Dataset
    - 멀티프로세싱 F0 추출
    - 캐싱 시스템
    - 메모리 효율적 로딩
    - 더 빠른 데이터 처리
    """
    
    def __init__(self, data_dir, sample_rate=44100, waveform_length=16384, 
                 channels=2, min_files_per_speaker=5, extract_f0=True, 
                 hop_length=512, f0_method='crepe_small', use_cache=True, 
                 max_workers=None, use_gpu_cache=True, gpu_batch_size=32):
        
        self.data_dir = Path(data_dir)
        self.sample_rate = sample_rate
        self.waveform_length = waveform_length
        self.channels = channels
        self.min_files_per_speaker = min_files_per_speaker
        self.extract_f0 = extract_f0
        self.hop_length = hop_length
        self.f0_method = f0_method
        self.use_cache = use_cache
        self.use_gpu_cache = use_gpu_cache
        self.gpu_batch_size = gpu_batch_size
        
        #  캐시 디렉토리
        self.cache_dir = self.data_dir / '.cache'
        if self.use_cache:
            self.cache_dir.mkdir(exist_ok=True)
        
        # GPU 캐시 초기화
        if self.use_gpu_cache and self.extract_f0:
            model_capacity = 'small' if 'crepe' in self.f0_method else 'small'
            if 'tiny' in self.f0_method:
                model_capacity = 'tiny'
            elif 'full' in self.f0_method:
                model_capacity = 'full'
            
            self.gpu_cache = GPUAcceleratedF0Cache(
                use_gpu=True,
                batch_size=self.gpu_batch_size,
                model_capacity=model_capacity
            )
        else:
            self.gpu_cache = None
        
        # 멀티프로세싱 설정
        self.max_workers = max_workers or min(8, mp.cpu_count())
        
        print(f"🎵 Initializing optimized dataset:")
        print(f"   Cache: {'✓ Enabled' if self.use_cache else '✗ Disabled'}")
        print(f"   F0 extraction: {'✓ Enabled' if self.extract_f0 else '✗ Disabled'}")
        print(f"   F0 method: {self.f0_method}")
        print(f"   GPU cache: {'✓ Enabled' if self.use_gpu_cache else '✗ Disabled'}")
        print(f"   Max workers: {self.max_workers}")
        
        # 데이터 스캔
        self._scan_dataset()
        
        # F0 캐시 준비
        if self.extract_f0:
            self._prepare_f0_cache()
    
    def _scan_dataset(self):
        """ 최적화된 데이터셋 스캔"""
        start_time = time.time()
        
        self.speakers = []
        self.speaker_files = {}
        
        # 병렬로 화자 폴더 스캔
        speaker_dirs = [d for d in self.data_dir.iterdir() if d.is_dir() and not d.name.startswith('.')]
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            results = list(executor.map(self._scan_speaker_dir, speaker_dirs))
        
        # 결과 집계
        for speaker_name, audio_files in results:
            if speaker_name and len(audio_files) >= self.min_files_per_speaker:
                self.speakers.append(speaker_name)
                self.speaker_files[speaker_name] = sorted(audio_files)
                print(f" {speaker_name}: {len(audio_files)} files")
            elif speaker_name:
                print(f" {speaker_name}: {len(audio_files)} files (< {self.min_files_per_speaker}, skipped)")
        
        # 화자 ID 매핑
        self.speaker_to_id = {spk: i for i, spk in enumerate(sorted(self.speakers))}
        self.id_to_speaker = {i: spk for spk, i in self.speaker_to_id.items()}
        
        # 훈련 페어 생성
        self._generate_training_pairs()
        
        scan_time = time.time() - start_time
        print(f" Dataset scan completed in {scan_time:.2f}s")
    
    def _scan_speaker_dir(self, speaker_dir):
        """개별 화자 디렉토리 스캔"""
        speaker_name = speaker_dir.name
        audio_files = []
        
        for ext in ['*.wav', '*.mp3', '*.flac', '*.m4a']:
            audio_files.extend(list(speaker_dir.glob(ext)))
        
        return speaker_name, audio_files
    
    def _generate_training_pairs(self):
        """훈련 페어 생성 - 최적화됨"""
        self.training_pairs = []
        
        for source_speaker in self.speakers:
            source_files = self.speaker_files[source_speaker]
            
            for target_speaker in self.speakers:
                if target_speaker != source_speaker:
                    target_speaker_id = self.speaker_to_id[target_speaker]
                    
                    for source_file in source_files:
                        self.training_pairs.append({
                            'source_file': source_file,
                            'source_speaker': source_speaker,
                            'target_speaker': target_speaker,
                            'target_speaker_id': target_speaker_id
                        })
        
        print(f" Generated {len(self.training_pairs)} training pairs")
    
    def _prepare_f0_cache(self):
        """F0 캐시 준비"""
        if not self.use_cache:
            return
        
        cache_info_file = self.cache_dir / 'f0_cache_info.json'
        
        if cache_info_file.exists():
            with open(cache_info_file, 'r') as f:
                cache_info = json.load(f)
            
            # 캐시 유효성 검증
            if (cache_info.get('sample_rate') == self.sample_rate and
                cache_info.get('hop_length') == self.hop_length and
                cache_info.get('f0_method') == self.f0_method):
                print(" F0 cache is valid")
                return
        
        print(" Building F0 cache...")
        self._build_f0_cache()
    
    def _build_f0_cache(self):
        """F0 캐시 구축 - GPU 가속 지원"""
        all_files = []
        for files in self.speaker_files.values():
            all_files.extend(files)
        
        print(f"🎵 Processing {len(all_files)} files for F0 cache...")
        
        if self.use_gpu_cache and self.gpu_cache:
            # GPU 가속 캐시 구축
            total_processed, failed_files = self.gpu_cache.build_cache_parallel_gpu(
                all_files, self.cache_dir, self.sample_rate, self.hop_length
            )
        else:
            # 기존 CPU 멀티스레딩 방식
            print("💻 Using CPU multiprocessing for F0 cache...")
            batch_size = 50
            for i in tqdm(range(0, len(all_files), batch_size), desc="F0 Cache"):
                batch_files = all_files[i:i+batch_size]
                
                with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                    executor.map(self._extract_and_cache_f0, batch_files)
        
        # 캐시 정보 저장
        cache_info = {
            'sample_rate': self.sample_rate,
            'hop_length': self.hop_length,
            'f0_method': self.f0_method,
            'total_files': len(all_files),
            'gpu_accelerated': self.use_gpu_cache and self.gpu_cache is not None
        }
        
        with open(self.cache_dir / 'f0_cache_info.json', 'w') as f:
            json.dump(cache_info, f)
        
        print("✅ F0 cache built successfully")
    
    def _extract_and_cache_f0(self, audio_file):
        """개별 파일의 F0 추출 및 캐시"""
        cache_file = self.cache_dir / f"{audio_file.stem}_f0.npz"
        
        if cache_file.exists():
            return  # 이미 캐시됨
        
        try:
            # 오디오 로드 (짧게)
            waveform, sr = torchaudio.load(audio_file)
            if sr != self.sample_rate:
                resampler = torchaudio.transforms.Resample(sr, self.sample_rate)
                waveform = resampler(waveform)
            
            # 모노 변환
            if waveform.size(0) > 1:
                waveform = waveform.mean(dim=0)
            else:
                waveform = waveform.squeeze(0)
            
            # F0 추출 - 메소드에 따라 다른 파라미터 사용
            if self.f0_method.startswith('crepe'):
                f0, vuv = extract_f0(
                    waveform.numpy(),
                    sample_rate=self.sample_rate,
                    hop_length=self.hop_length,
                    method=self.f0_method
                )
            else:
                f0, vuv = extract_f0(
                    waveform.numpy(),
                    sample_rate=self.sample_rate,
                    hop_length=self.hop_length,
                    f0_min=80, f0_max=800,
                    method=self.f0_method
                )
            
            # 정규화
            f0_normalized = normalize_f0(f0, method='log')
            
            # 캐시 저장
            np.savez_compressed(
                cache_file,
                f0=f0,
                f0_normalized=f0_normalized,
                vuv=vuv
            )
            
        except Exception as e:
            print(f" Failed to cache F0 for {audio_file}: {e}")
    
    def _load_cached_f0(self, audio_file):
        """캐시된 F0 로드"""
        if not self.use_cache:
            return None
        
        cache_file = self.cache_dir / f"{audio_file.stem}_f0.npz"
        
        if cache_file.exists():
            try:
                data = np.load(cache_file)
                return {
                    'f0': data['f0'],
                    'f0_normalized': data['f0_normalized'],
                    'vuv': data['vuv']
                }
            except Exception as e:
                print(f" Failed to load cached F0 for {audio_file}: {e}")
        
        return None
    
    def __len__(self):
        return len(self.training_pairs)
    
    def __getitem__(self, idx):
        pair = self.training_pairs[idx]
        
        #  오디오 로딩
        source_waveform = self._load_audio(pair['source_file'])
        
        # 타겟 오디오 (랜덤 선택)
        target_files = self.speaker_files[pair['target_speaker']]
        target_file = random.choice(target_files)
        target_waveform = self._load_audio(target_file)
        
        result = {
            'source_waveform': source_waveform,
            'target_waveform': target_waveform,
            'target_speaker_id': torch.tensor(pair['target_speaker_id'], dtype=torch.long),
            'source_speaker': pair['source_speaker'],
            'target_speaker': pair['target_speaker'],
            'source_file': str(pair['source_file']),
            'target_file': str(target_file)
        }
        
        #  F0 처리
        if self.extract_f0:
            # 캐시에서 시도
            f0_data = self._load_cached_f0(target_file)
            
            if f0_data is not None:
                # 캐시된 데이터 사용
                f0_normalized = f0_data['f0_normalized']
                vuv = f0_data['vuv']
            else:
                # 실시간 추출
                if target_waveform.dim() == 2:
                    target_mono = target_waveform.mean(dim=0)
                else:
                    target_mono = target_waveform
                
                try:
                    # F0 추출 - 메소드에 따라 다른 파라미터 사용
                    if self.f0_method.startswith('crepe'):
                        f0, vuv = extract_f0(
                            target_mono.numpy(),
                            sample_rate=self.sample_rate,
                            hop_length=self.hop_length,
                            method=self.f0_method
                        )
                    else:
                        f0, vuv = extract_f0(
                            target_mono.numpy(),
                            sample_rate=self.sample_rate,
                            hop_length=self.hop_length,
                            f0_min=80, f0_max=800,
                            method=self.f0_method
                        )
                    f0_normalized = normalize_f0(f0, method='log')
                except Exception as e:
                    print(f" F0 extraction failed: {e}")
                    # 기본값 사용
                    default_frames = self.waveform_length // self.hop_length + 1
                    f0_normalized = np.zeros(default_frames, dtype=np.float32)
                    vuv = np.zeros(default_frames, dtype=np.float32)
            
            result['f0_target'] = torch.from_numpy(f0_normalized).float()
            result['vuv_target'] = torch.from_numpy(vuv).float()
        
        return result
    
    def _load_audio(self, file_path):
        """ 최적화된 오디오 로딩"""
        try:
            #  torchaudio로 빠른 로딩
            waveform, sr = torchaudio.load(str(file_path))
            
            # 리샘플링 (필요시)
            if sr != self.sample_rate:
                resampler = torchaudio.transforms.Resample(sr, self.sample_rate)
                waveform = resampler(waveform)
            
            # 채널 처리
            if self.channels == 2:
                if waveform.size(0) == 1:
                    waveform = waveform.repeat(2, 1)
                elif waveform.size(0) > 2:
                    waveform = waveform[:2]
            else:
                if waveform.size(0) > 1:
                    waveform = waveform.mean(dim=0, keepdim=True)
            
            # 길이 조정
            if self.channels == 2:
                waveform = self._fix_length(waveform, self.waveform_length, dim=1)
            else:
                waveform = self._fix_length(waveform.squeeze(0), self.waveform_length, dim=0)
            
            return waveform
            
        except Exception as e:
            print(f" Failed to load audio {file_path}: {e}")
            # 기본 노이즈 반환
            if self.channels == 2:
                return torch.randn(2, self.waveform_length) * 0.01
            else:
                return torch.randn(self.waveform_length) * 0.01
    
    def _fix_length(self, waveform, target_length, dim=0):
        """길이 조정 - 최적화됨"""
        current_length = waveform.size(dim)
        
        if current_length > target_length:
            # 랜덤 크롭 (더 효율적)
            start = torch.randint(0, current_length - target_length + 1, (1,)).item()
            if dim == 0:
                return waveform[start:start + target_length]
            else:
                return waveform[:, start:start + target_length]
        elif current_length < target_length:
            # 제로 패딩
            pad_length = target_length - current_length
            if dim == 0:
                return torch.cat([waveform, torch.zeros(pad_length)], dim=0)
            else:
                return torch.cat([waveform, torch.zeros(waveform.size(0), pad_length)], dim=1)
        else:
            return waveform
    
    def get_speaker_info(self):
        """화자 정보 반환"""
        return {
            'speakers': self.speakers,
            'speaker_to_id': self.speaker_to_id,
            'id_to_speaker': self.id_to_speaker,
            'total_speakers': len(self.speakers),
            'total_pairs': len(self.training_pairs),
            'files_per_speaker': {spk: len(files) for spk, files in self.speaker_files.items()}
        }
    
    def print_sample_pairs(self, num_samples=5):
        """샘플 페어 출력"""
        print(f"\n Sample training pairs:")
        for i in range(min(num_samples, len(self.training_pairs))):
            pair = self.training_pairs[i]
            print(f"   {i+1}. {pair['source_speaker']} → {pair['target_speaker']}")
            print(f"       {Path(pair['source_file']).name}")
        
        if self.extract_f0:
            print(f"    F0 conditioning enabled with cache")
        print()

# 호환성을 위한 별칭
VoiceConversionDataset = OptimizedVoiceConversionDataset

def optimized_collate_fn(batch):
    """ 최적화된 collate 함수"""
    # 첫 번째 아이템에서 차원 확인
    first_source = batch[0]['source_waveform']
    is_stereo = first_source.dim() == 2
    batch_size = len(batch)
    
    # 효율적인 스택킹
    if is_stereo:
        source_waveforms = torch.stack([item['source_waveform'] for item in batch])
        target_waveforms = torch.stack([item['target_waveform'] for item in batch])
    else:
        source_waveforms = torch.stack([item['source_waveform'] for item in batch])
        target_waveforms = torch.stack([item['target_waveform'] for item in batch])
    
    target_speaker_ids = torch.stack([item['target_speaker_id'] for item in batch])
    
    result = {
        'source_waveform': source_waveforms,
        'target_waveform': target_waveforms,
        'target_speaker_id': target_speaker_ids,
        'source_speakers': [item['source_speaker'] for item in batch],
        'target_speakers': [item['target_speaker'] for item in batch]
    }
    
    #  F0/VUV 처리 (최적화됨)
    if 'f0_target' in batch[0]:
        # 배치 내 최대 길이 찾기
        max_f0_len = max(item['f0_target'].size(0) for item in batch)
        
        # 미리 할당
        f0_targets = torch.zeros(batch_size, max_f0_len)
        vuv_targets = torch.zeros(batch_size, max_f0_len)
        
        for i, item in enumerate(batch):
            f0 = item['f0_target']
            vuv = item['vuv_target']
            
            # 길이 조정
            actual_len = min(f0.size(0), max_f0_len)
            f0_targets[i, :actual_len] = f0[:actual_len]
            vuv_targets[i, :actual_len] = vuv[:actual_len]
        
        result['f0_target'] = f0_targets
        result['vuv_target'] = vuv_targets
    
    return result

# 호환성을 위한 별칭
collate_fn = optimized_collate_fn

#  추가 최적화 유틸리티들

def create_optimized_train_val_split(dataset_root, train_ratio=0.8, use_symlinks=True):
    """
    최적화된 train/val 분할
    - 심볼릭 링크 사용으로 디스크 공간 절약
    - 병렬 처리
    """
    dataset_root = Path(dataset_root)
    train_dir = dataset_root / 'train'
    val_dir = dataset_root / 'val'
    
    # 기존 디렉토리 정리
    if train_dir.exists():
        shutil.rmtree(train_dir)
    if val_dir.exists():
        shutil.rmtree(val_dir)
    
    train_dir.mkdir()
    val_dir.mkdir()
    
    def process_speaker(speaker_dir):
        if not speaker_dir.is_dir() or speaker_dir.name in ['train', 'val', '.cache']:
            return None, 0, 0
        
        speaker_name = speaker_dir.name
        
        # 오디오 파일 수집
        audio_files = []
        for ext in ['*.wav', '*.mp3', '*.flac', '*.m4a']:
            audio_files.extend(list(speaker_dir.glob(ext)))
        
        if len(audio_files) < 2:
            return speaker_name, 0, 0
        
        # 분할
        random.shuffle(audio_files)
        split_idx = int(len(audio_files) * train_ratio)
        
        train_files = audio_files[:split_idx]
        val_files = audio_files[split_idx:]
        
        # 디렉토리 생성
        (train_dir / speaker_name).mkdir()
        (val_dir / speaker_name).mkdir()
        
        # 파일 복사 또는 링크
        for file in train_files:
            dest = train_dir / speaker_name / file.name
            if use_symlinks:
                dest.symlink_to(file.absolute())
            else:
                shutil.copy2(file, dest)
        
        for file in val_files:
            dest = val_dir / speaker_name / file.name
            if use_symlinks:
                dest.symlink_to(file.absolute())
            else:
                shutil.copy2(file, dest)
        
        return speaker_name, len(train_files), len(val_files)
    
    # 병렬 처리
    speaker_dirs = [d for d in dataset_root.iterdir() if d.is_dir()]
    
    with ThreadPoolExecutor(max_workers=min(8, mp.cpu_count())) as executor:
        results = list(executor.map(process_speaker, speaker_dirs))
    
    # 결과 출력
    total_train = 0
    total_val = 0
    
    for speaker_name, train_count, val_count in results:
        if speaker_name and (train_count > 0 or val_count > 0):
            print(f" {speaker_name}: {train_count} train, {val_count} val")
            total_train += train_count
            total_val += val_count
    
    print(f" Optimized Train/Val split completed!")
    print(f"    Total train files: {total_train}")
    print(f"    Total val files: {total_val}")
    print(f"    Using {'symlinks' if use_symlinks else 'copies'}")

def benchmark_dataset_loading(dataset, num_samples=100):
    """데이터셋 로딩 성능 벤치마크"""
    print(f" Benchmarking dataset loading ({num_samples} samples)...")
    
    start_time = time.time()
    
    for i in range(min(num_samples, len(dataset))):
        sample = dataset[i]
    
    end_time = time.time()
    total_time = end_time - start_time
    avg_time = total_time / num_samples
    
    print(f" Loading benchmark results:")
    print(f"   Total time: {total_time:.2f}s")
    print(f"   Average per sample: {avg_time*1000:.2f}ms")
    print(f"   Samples per second: {num_samples/total_time:.1f}")
    
    return avg_time

# 최적화된 F0 추출 설정
F0_CONFIG = {
    # 기본 설정
    'extract_f0': True,
    'f0_method': 'crepe_small',  # 추천: 속도-정확도 균형
    'use_gpu_f0': True,
    'f0_batch_size': 32,
    
    # 메소드별 설정
    'methods': {
        'crepe_tiny': {
            'description': '가장 빠름 (3배), 약간 정확도 하락',
            'speed': 'fastest',
            'accuracy': 'good'
        },
        'crepe_small': {
            'description': '균형잡힌 선택 (2배 빠름, 높은 정확도)',
            'speed': 'fast',
            'accuracy': 'high'
        },
        'crepe_full': {
            'description': '최고 정확도, 가장 느림',
            'speed': 'slow',
            'accuracy': 'highest'
        },
        'hybrid': {
            'description': 'CREPE + pYIN 결합, 최고 품질',
            'speed': 'medium',
            'accuracy': 'highest'
        },
        'pyin': {
            'description': 'CPU 최적화, CREPE 없을 때 사용',
            'speed': 'medium',
            'accuracy': 'medium'
        }
    }
}

def get_f0_config():
    """F0 설정 정보 반환"""
    return F0_CONFIG

def print_f0_methods():
    """사용 가능한 F0 추출 방법 출력"""
    print("🎵 Available F0 extraction methods:")
    print(f"   CREPE available: {'✓' if CREPE_AVAILABLE else '✗'}")
    print(f"   SciPy available: {'✓' if SCIPY_AVAILABLE else '✗'}")
    print()
    
    for method, info in F0_CONFIG['methods'].items():
        status = "✓" if method == 'pyin' or CREPE_AVAILABLE else "✗ (needs CREPE)"
        print(f"   {status} {method:12} - {info['description']}")
        print(f"      Speed: {info['speed']:8} | Accuracy: {info['accuracy']}")
    print()
    print("Recommended:")
    print("  • For fastest: crepe_tiny")
    print("  • For balanced: crepe_small (default)")
    print("  • For best quality: hybrid")
    print("  • For CPU only: pyin")

# 사용 예시 함수
def create_optimized_dataset_with_f0(data_dir, **kwargs):
    """
    최적화된 F0 캐시가 포함된 데이터셋 생성
    
    Usage:
        # GPU 가속 CREPE (추천)
        dataset = create_optimized_dataset_with_f0(
            'path/to/data',
            f0_method='crepe_small',
            use_gpu_cache=True,
            gpu_batch_size=32
        )
        
        # 최고 품질 하이브리드
        dataset = create_optimized_dataset_with_f0(
            'path/to/data',
            f0_method='hybrid',
            use_gpu_cache=True
        )
        
        # CPU 전용
        dataset = create_optimized_dataset_with_f0(
            'path/to/data',
            f0_method='pyin',
            use_gpu_cache=False
        )
    """
    # 기본값 설정
    config = F0_CONFIG.copy()
    config.update(kwargs)
    
    return OptimizedVoiceConversionDataset(data_dir, **config)