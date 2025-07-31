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

warnings.filterwarnings("ignore", category=UserWarning)

@lru_cache(maxsize=1000)
def extract_f0_cached(audio_hash, sample_rate=44100, hop_length=512, f0_min=80, f0_max=800, method='pyin'):
    """
    캐시된 F0 추출 - 동일한 오디오에 대해 재계산 방지
    """
    # 실제로는 audio를 받아야 하지만, 캐싱을 위해 해시 사용
    # 이 함수는 실제 구현에서는 사용하지 않고, 아래의 extract_f0를 사용
    pass

def extract_f0(audio, sample_rate=44100, hop_length=512, f0_min=80, f0_max=800, method='pyin'):
    """
     최적화된 F0 추출
    - 더 빠른 파라미터
    - 에러 핸들링 강화
    - 메모리 효율성 향상
    """
    if isinstance(audio, torch.Tensor):
        audio = audio.cpu().numpy()
    
    # 입력 검증
    if len(audio) == 0:
        return np.array([]), np.array([])
    
    try:
        if method == 'pyin':
            #  최적화된 pyin 파라미터
            f0, voiced_flag, voiced_probs = librosa.pyin(
                audio,
                fmin=f0_min,
                fmax=f0_max,
                sr=sample_rate,
                hop_length=hop_length,
                frame_length=hop_length * 3,  # 더 빠르게
                win_length=hop_length * 2,    # 더 빠르게
                resolution=0.1,               # 더 빠르게
                max_transition_rate=35.92,    # 기본값
                switch_prob=0.01,             # 더 빠르게
                no_trough_prob=0.01           # 더 빠르게
            )
            
            # NaN 처리
            valid_mask = ~np.isnan(f0)
            f0 = np.nan_to_num(f0, nan=0.0)
            
            # VUV 플래그 생성
            vuv = valid_mask & (f0 > f0_min/2)  # 더 관대한 임계값
            vuv = vuv.astype(np.float32)
            
        else:
            raise ValueError(f"Unsupported F0 extraction method: {method}")
        
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
        
        return f0.astype(np.float32), vuv.astype(np.float32)
        
    except Exception as e:
        print(f" F0 extraction failed: {e}")
        # 실패 시 기본 길이로 0 반환
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
                 hop_length=512, f0_method='pyin', use_cache=True, 
                 max_workers=None):
        
        self.data_dir = Path(data_dir)
        self.sample_rate = sample_rate
        self.waveform_length = waveform_length
        self.channels = channels
        self.min_files_per_speaker = min_files_per_speaker
        self.extract_f0 = extract_f0
        self.hop_length = hop_length
        self.f0_method = f0_method
        self.use_cache = use_cache
        
        #  캐시 디렉토리
        self.cache_dir = self.data_dir / '.cache'
        if self.use_cache:
            self.cache_dir.mkdir(exist_ok=True)
        
        # 멀티프로세싱 설정
        self.max_workers = max_workers or min(8, mp.cpu_count())
        
        print(f" Initializing optimized dataset:")
        print(f"   Cache: {' Enabled' if self.use_cache else ' Disabled'}")
        print(f"   F0 extraction: {' Enabled' if self.extract_f0 else ' Disabled'}")
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
        """F0 캐시 구축"""
        all_files = []
        for files in self.speaker_files.values():
            all_files.extend(files)
        
        print(f" Processing {len(all_files)} files for F0 cache...")
        
        # 배치 처리로 F0 추출
        batch_size = 50
        for i in range(0, len(all_files), batch_size):
            batch_files = all_files[i:i+batch_size]
            
            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                executor.map(self._extract_and_cache_f0, batch_files)
            
            print(f"Progress: {min(i+batch_size, len(all_files))}/{len(all_files)}")
        
        # 캐시 정보 저장
        cache_info = {
            'sample_rate': self.sample_rate,
            'hop_length': self.hop_length,
            'f0_method': self.f0_method,
            'total_files': len(all_files)
        }
        
        with open(self.cache_dir / 'f0_cache_info.json', 'w') as f:
            json.dump(cache_info, f)
        
        print(" F0 cache built successfully")
    
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
            
            # F0 추출
            f0, vuv = extract_f0(
                waveform.numpy(),
                sample_rate=self.sample_rate,
                hop_length=self.hop_length,
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
                    f0, vuv = extract_f0(
                        target_mono.numpy(),
                        sample_rate=self.sample_rate,
                        hop_length=self.hop_length,
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