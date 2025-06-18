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

def extract_f0(audio, sample_rate=44100, hop_length=512, f0_min=80, f0_max=800, method='pyin'):
    """
    Extract F0 from audio using librosa
    Args:
        audio: (T,) audio waveform
        sample_rate: sampling rate
        hop_length: hop length for STFT
        f0_min, f0_max: F0 range
        method: 'pyin' or 'harvest'
    Returns:
        f0: (T_frames,) F0 sequence
        vuv: (T_frames,) voiced/unvoiced flags
    """
    if isinstance(audio, torch.Tensor):
        audio = audio.cpu().numpy()
    
    if method == 'pyin':
        f0, voiced_flag, voiced_probs = librosa.pyin(
            audio,
            fmin=f0_min,
            fmax=f0_max,
            sr=sample_rate,
            hop_length=hop_length,
            frame_length=hop_length * 4,
            threshold=0.1
        )
        
        # Handle NaN values
        f0 = np.nan_to_num(f0, nan=0.0)
        vuv = ~np.isnan(voiced_flag)
        
    # elif method == 'harvest':
    #     # Alternative: use pyworld for harvest (requires pyworld installation)
    #     try:
    #         import pyworld as pw
    #         f0, t = pw.harvest(
    #             audio.astype(np.float64),
    #             sample_rate,
    #             f0_floor=f0_min,
    #             f0_ceil=f0_max,
    #             frame_period=hop_length / sample_rate * 1000
    #         )
    #         vuv = f0 > 0
    #         f0[~vuv] = 0.0
    #     except ImportError:
    #         print("Warning: pyworld not installed, using pyin instead")
    #         return extract_f0(audio, sample_rate, hop_length, f0_min, f0_max, method='pyin')
    
    else:
        raise ValueError(f"Unknown F0 extraction method: {method}")
    
    return f0, vuv.astype(np.float32)

def compute_vuv(f0, threshold=0.1):
    """
    Compute voiced/unvoiced from F0
    Args:
        f0: F0 sequence
        threshold: threshold for voiced detection
    Returns:
        vuv: voiced/unvoiced binary sequence
    """
    if isinstance(f0, torch.Tensor):
        return (f0 > threshold).float()
    else:
        return (f0 > threshold).astype(np.float32)

def normalize_f0(f0, method='log', f0_min=80, f0_max=800):
    """
    Normalize F0 values
    Args:
        f0: F0 sequence
        method: 'log' or 'linear'
    Returns:
        normalized F0
    """
    if isinstance(f0, torch.Tensor):
        is_torch = True
        device = f0.device
        f0_np = f0.cpu().numpy()
    else:
        is_torch = False
        f0_np = f0
    
    # Mask unvoiced regions
    voiced = f0_np > 0
    
    if method == 'log':
        # Log-scale normalization
        f0_norm = np.zeros_like(f0_np)
        if np.any(voiced):
            f0_voiced = f0_np[voiced]
            f0_voiced = np.clip(f0_voiced, f0_min, f0_max)
            log_f0 = np.log(f0_voiced)
            log_f0_min, log_f0_max = np.log(f0_min), np.log(f0_max)
            f0_norm[voiced] = (log_f0 - log_f0_min) / (log_f0_max - log_f0_min) * 2 - 1
    
    elif method == 'linear':
        # Linear normalization
        f0_norm = np.zeros_like(f0_np)
        if np.any(voiced):
            f0_voiced = f0_np[voiced]
            f0_voiced = np.clip(f0_voiced, f0_min, f0_max)
            f0_norm[voiced] = (f0_voiced - f0_min) / (f0_max - f0_min) * 2 - 1
    
    else:
        raise ValueError(f"Unknown normalization method: {method}")
    
    if is_torch:
        return torch.from_numpy(f0_norm).to(device)
    else:
        return f0_norm

def denormalize_f0(f0_norm, method='log', f0_min=80, f0_max=800):
    """Denormalize F0 values"""
    if isinstance(f0_norm, torch.Tensor):
        is_torch = True
        device = f0_norm.device
        f0_np = f0_norm.cpu().numpy()
    else:
        is_torch = False
        f0_np = f0_norm
    
    if method == 'log':
        log_f0_min, log_f0_max = np.log(f0_min), np.log(f0_max)
        log_f0 = (f0_np + 1) / 2 * (log_f0_max - log_f0_min) + log_f0_min
        f0 = np.exp(log_f0)
    
    elif method == 'linear':
        f0 = (f0_np + 1) / 2 * (f0_max - f0_min) + f0_min
    
    else:
        raise ValueError(f"Unknown normalization method: {method}")
    
    if is_torch:
        return torch.from_numpy(f0).to(device)
    else:
        return f0

class VoiceConversionDataset(Dataset):
    """
    Voice Conversion Dataset - RVC Style Structure
    
    🎯 목표: 음성의 화자만 변경 (음질 향상 + 억양 보존)
    
    Simple dataset structure:
    dataset_root/
    ├── speaker1/
    │   ├── 001.wav
    │   ├── 002.wav
    │   ├── 003.wav
    │   └── ...
    ├── speaker2/
    │   ├── 001.wav
    │   ├── 002.wav
    │   └── ...
    ├── speaker3/
    │   ├── 001.wav
    │   └── ...
    └── ...
    
    🔥 훈련 방식:
    - speaker1의 001.wav를 입력으로 받아서
    - speaker2의 스타일로 변환하도록 학습
    - 모든 화자가 source도 되고 target도 됨!
    """
    
    def __init__(self, data_dir, sample_rate=44100, waveform_length=16384, 
                 channels=2, min_files_per_speaker=5):
        self.data_dir = Path(data_dir)
        self.sample_rate = sample_rate
        self.waveform_length = waveform_length
        self.channels = channels
        self.min_files_per_speaker = min_files_per_speaker
        
        # 🔍 모든 화자 폴더 스캔
        self.speakers = []
        self.speaker_files = {}
        
        for speaker_dir in self.data_dir.iterdir():
            if speaker_dir.is_dir():
                # 화자 폴더 내 오디오 파일 찾기
                audio_files = []
                for ext in ['*.wav', '*.mp3', '*.flac']:
                    audio_files.extend(list(speaker_dir.glob(ext)))
                
                # 최소 파일 수 체크
                if len(audio_files) >= self.min_files_per_speaker:
                    speaker_name = speaker_dir.name
                    self.speakers.append(speaker_name)
                    self.speaker_files[speaker_name] = sorted(audio_files)
                    print(f"📁 {speaker_name}: {len(audio_files)} files")
                else:
                    print(f"⚠️ {speaker_dir.name}: {len(audio_files)} files (< {self.min_files_per_speaker}, skipped)")
        
        # 화자 ID 매핑 생성
        self.speaker_to_id = {spk: i for i, spk in enumerate(sorted(self.speakers))}
        self.id_to_speaker = {i: spk for spk, i in self.speaker_to_id.items()}
        
        # 🎯 훈련 페어 생성: (source_file, target_speaker_id)
        self.training_pairs = []
        
        for source_speaker in self.speakers:
            source_files = self.speaker_files[source_speaker]
            
            for target_speaker in self.speakers:
                if target_speaker != source_speaker:  # 다른 화자로만 변환
                    target_speaker_id = self.speaker_to_id[target_speaker]
                    
                    # 각 source 파일을 target 화자로 변환하는 페어 생성
                    for source_file in source_files:
                        self.training_pairs.append({
                            'source_file': source_file,
                            'source_speaker': source_speaker,
                            'target_speaker': target_speaker,
                            'target_speaker_id': target_speaker_id
                        })
        
        print(f"\n🎯 Dataset Summary:")
        print(f"   👥 Total speakers: {len(self.speakers)}")
        print(f"   📊 Training pairs: {len(self.training_pairs)}")
        print(f"   🎵 Sample rate: {self.sample_rate}Hz")
        print(f"   ⏱️ Waveform length: {self.waveform_length} samples (~{self.waveform_length/self.sample_rate:.1f}s)")
        
        if len(self.speakers) < 2:
            raise ValueError("최소 2명의 화자가 필요합니다!")
    
    def __len__(self):
        return len(self.training_pairs)
    
    def __getitem__(self, idx):
        pair = self.training_pairs[idx]
        
        # 📥 Source audio 로드 (변환할 음성)
        source_waveform = self._load_audio(pair['source_file'])
        
        # 🎯 Target audio 로드 (목표 화자의 음성 - 랜덤 선택)
        target_files = self.speaker_files[pair['target_speaker']]
        target_file = random.choice(target_files)  # 랜덤 선택!
        target_waveform = self._load_audio(target_file)
        
        return {
            'source_waveform': source_waveform,
            'target_waveform': target_waveform,  # 모델이 target 스타일 학습용
            'target_speaker_id': torch.tensor(pair['target_speaker_id'], dtype=torch.long),
            'source_speaker': pair['source_speaker'], 
            'target_speaker': pair['target_speaker'],
            # 🔍 디버깅 정보
            'source_file': str(pair['source_file']),
            'target_file': str(target_file)
        }
    
    def _load_audio(self, file_path):
        """오디오 파일 로드 및 전처리"""
        # 오디오 로드
        waveform, sr = torchaudio.load(file_path)
        
        # 리샘플링
        if sr != self.sample_rate:
            resampler = torchaudio.transforms.Resample(sr, self.sample_rate)
            waveform = resampler(waveform)
        
        # 채널 처리
        if self.channels == 2:
            # 스테레오로 변환
            if waveform.size(0) == 1:
                waveform = waveform.repeat(2, 1)  # 모노 → 스테레오
            elif waveform.size(0) > 2:
                waveform = waveform[:2]  # 첫 2채널만
        else:
            # 모노로 변환
            if waveform.size(0) > 1:
                waveform = waveform.mean(dim=0, keepdim=True)
        
        # 길이 조정
        if self.channels == 2:
            waveform = self._fix_length(waveform, self.waveform_length, dim=1)
        else:
            waveform = self._fix_length(waveform.squeeze(0), self.waveform_length, dim=0)
        
        return waveform
    
    def _fix_length(self, waveform, target_length, dim=0):
        """길이 맞추기: 크롭 또는 패딩"""
        current_length = waveform.size(dim)
        
        if current_length > target_length:
            # 랜덤 크롭
            start = torch.randint(0, current_length - target_length + 1, (1,)).item()
            if dim == 0:
                return waveform[start:start + target_length]
            else:  # dim == 1
                return waveform[:, start:start + target_length]
        elif current_length < target_length:
            # 제로 패딩
            pad_length = target_length - current_length
            if dim == 0:
                return torch.cat([waveform, torch.zeros(pad_length)])
            else:  # dim == 1
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
        print(f"\n🔍 Sample training pairs:")
        for i in range(min(num_samples, len(self.training_pairs))):
            pair = self.training_pairs[i]
            print(f"   {i+1}. {pair['source_speaker']} → {pair['target_speaker']}")
            print(f"      📁 {Path(pair['source_file']).name}")
            print(f"      🎯 Convert to {pair['target_speaker']} style")
        print()
    
    def get_speaker_stats(self):
        """화자별 통계"""
        stats = {}
        for speaker in self.speakers:
            files = self.speaker_files[speaker]
            total_duration = 0
            
            # 대략적인 길이 계산 (정확하지 않지만 추정용)
            for file in files[:10]:  # 처음 10개 파일만 체크 (속도)
                try:
                    info = torchaudio.info(file)
                    duration = info.num_frames / info.sample_rate
                    total_duration += duration
                except:
                    pass
            
            avg_duration = total_duration / min(10, len(files))
            estimated_total = avg_duration * len(files)
            
            stats[speaker] = {
                'files': len(files),
                'estimated_duration_minutes': estimated_total / 60
            }
        
        return stats

def create_train_val_split(dataset_root, train_ratio=0.8, val_ratio=0.2):
    """
    훈련/검증 데이터 분할
    화자별로 파일을 분할해서 train/val 폴더에 복사
    """
    dataset_root = Path(dataset_root)
    train_dir = dataset_root / 'train'
    val_dir = dataset_root / 'val'
    
    # 기존 train/val 디렉토리 제거
    if train_dir.exists():
        shutil.rmtree(train_dir)
    if val_dir.exists():
        shutil.rmtree(val_dir)
    
    train_dir.mkdir()
    val_dir.mkdir()
    
    # 각 화자별로 파일 분할
    for speaker_dir in dataset_root.iterdir():
        if speaker_dir.is_dir() and speaker_dir.name not in ['train', 'val']:
            speaker_name = speaker_dir.name
            
            # 오디오 파일 수집
            audio_files = []
            for ext in ['*.wav', '*.mp3', '*.flac']:
                audio_files.extend(list(speaker_dir.glob(ext)))
            
            if len(audio_files) < 2:
                print(f"⚠️ {speaker_name}: 파일이 너무 적음 ({len(audio_files)})")
                continue
            
            # 셔플 후 분할
            random.shuffle(audio_files)
            split_idx = int(len(audio_files) * train_ratio)
            
            train_files = audio_files[:split_idx]
            val_files = audio_files[split_idx:]
            
            # 디렉토리 생성
            (train_dir / speaker_name).mkdir()
            (val_dir / speaker_name).mkdir()
            
            # 파일 복사
            for file in train_files:
                shutil.copy2(file, train_dir / speaker_name / file.name)
            
            for file in val_files:
                shutil.copy2(file, val_dir / speaker_name / file.name)
            
            print(f"📁 {speaker_name}: {len(train_files)} train, {len(val_files)} val")
    
    print(f"✅ Train/Val split completed!")
    print(f"   📁 Train: {train_dir}")
    print(f"   📁 Val: {val_dir}")

# 사용 예시
def dataset_usage_example():
    """데이터셋 사용 예시"""
    print("🚀 Dataset Usage Example")
    
    # 1. 데이터셋 로드
    dataset = VoiceConversionDataset(
        data_dir='./dataset_root',
        sample_rate=44100,
        waveform_length=16384,  # ~0.37초
        channels=2
    )
    
    # 2. 정보 출력
    dataset.print_sample_pairs()
    speaker_info = dataset.get_speaker_info()
    print(f"화자 목록: {speaker_info['speakers']}")
    
    # 3. 샘플 데이터 확인
    sample = dataset[0]
    print(f"\n📊 Sample data shapes:")
    print(f"   Source: {sample['source_waveform'].shape}")
    print(f"   Target: {sample['target_waveform'].shape}")
    print(f"   Target ID: {sample['target_speaker_id']}")
    
    return dataset
    """
    Dataset for audio and F0 pairs
    Expected data structure:
    data_dir/
    ├── train/
    │   ├── audio/
    │   │   ├── file1.wav
    │   │   └── file2.wav
    │   └── f0/
    │       ├── file1.npy  # F0 array
    │       └── file2.npy
    └── val/
        ├── audio/
        └── f0/
    """
    
    def __init__(self, data_dir, split='train', sample_rate=44100, hop_length=512, 
                 max_length=None, f0_method='pyin', channels=2):
        self.data_dir = Path(data_dir)
        self.split = split
        self.sample_rate = sample_rate
        self.hop_length = hop_length
        self.max_length = max_length
        self.f0_method = f0_method
        self.channels = channels
        
        # Get file list
        self.audio_dir = self.data_dir / split / 'audio'
        self.f0_dir = self.data_dir / split / 'f0'
        
        self.audio_files = sorted(list(self.audio_dir.glob('*.wav')))
        
        # Filter files that have corresponding F0
        self.valid_files = []
        for audio_file in self.audio_files:
            f0_file = self.f0_dir / f"{audio_file.stem}.npy"
            if f0_file.exists():
                self.valid_files.append(audio_file.stem)
        
        print(f"Found {len(self.valid_files)} valid files in {split} split")
    
    def __len__(self):
        return len(self.valid_files)
    
    def __getitem__(self, idx):
        file_stem = self.valid_files[idx]
        
        # Load audio
        audio_file = self.audio_dir / f"{file_stem}.wav"
        waveform, sr = torchaudio.load(audio_file)
        
        # Resample if necessary
        if sr != self.sample_rate:
            resampler = torchaudio.transforms.Resample(sr, self.sample_rate)
            waveform = resampler(waveform)
        
        # Convert to stereo if needed, or keep as is
        if self.channels == 2:
            if waveform.size(0) == 1:
                # Convert mono to stereo by duplicating
                waveform = waveform.repeat(2, 1)
            elif waveform.size(0) > 2:
                # Take first 2 channels
                waveform = waveform[:2]
        else:
            # Convert to mono for mono processing
            if waveform.size(0) > 1:
                waveform = waveform.mean(dim=0, keepdim=True)
        
        # For F0 extraction, always use mono version
        if waveform.size(0) > 1:
            waveform_mono = waveform.mean(dim=0)  # (T,)
        else:
            waveform_mono = waveform.squeeze(0)  # (T,)
        
        # Load or extract F0
        f0_file = self.f0_dir / f"{file_stem}.npy"
        if f0_file.exists():
            f0 = np.load(f0_file)
        else:
            # Extract F0 on the fly using mono version
            f0, _ = extract_f0(
                waveform_mono.numpy(),
                sample_rate=self.sample_rate,
                hop_length=self.hop_length,
                method=self.f0_method
            )
        
        # Compute V/UV
        vuv = compute_vuv(f0)
        
        # Normalize F0
        f0_norm = normalize_f0(f0)
        
        # Truncate or pad
        if self.max_length is not None:
            current_length = waveform.size(-1)  # Time dimension
            if current_length > self.max_length:
                waveform = waveform[:, :self.max_length]  # (channels, max_length)
                f0_frames = self.max_length // self.hop_length
                f0_norm = f0_norm[:f0_frames]
                vuv = vuv[:f0_frames]
            else:
                # Pad waveform
                pad_length = self.max_length - current_length
                if waveform.dim() == 2:  # (channels, time)
                    waveform = torch.cat([waveform, torch.zeros(waveform.size(0), pad_length)], dim=1)
                else:  # (time,) - mono case
                    waveform = torch.cat([waveform, torch.zeros(pad_length)])
                
                # Pad F0 and VUV
                f0_frames = self.max_length // self.hop_length
                if len(f0_norm) < f0_frames:
                    pad_frames = f0_frames - len(f0_norm)
                    f0_norm = np.pad(f0_norm, (0, pad_frames), 'constant')
                    vuv = np.pad(vuv, (0, pad_frames), 'constant')
        
        return {
            'waveform': waveform,
            'f0': torch.from_numpy(f0_norm).float(),
            'vuv': torch.from_numpy(vuv).float(),
            'filename': file_stem
        }

def collate_fn(batch):
    """Custom collate function for voice conversion pairs"""
    # Get dimensions from first item
    first_source = batch[0]['source_waveform']
    first_target = batch[0]['target_waveform']
    
    is_stereo = first_source.dim() == 2
    
    batch_size = len(batch)
    
    if is_stereo:
        # Stereo: (batch, channels, time)
        source_waveforms = torch.stack([item['source_waveform'] for item in batch])
        target_waveforms = torch.stack([item['target_waveform'] for item in batch])
    else:
        # Mono: (batch, time)  
        source_waveforms = torch.stack([item['source_waveform'] for item in batch])
        target_waveforms = torch.stack([item['target_waveform'] for item in batch])
    
    target_speaker_ids = torch.stack([item['target_speaker_id'] for item in batch])
    
    return {
        'source_waveform': source_waveforms,
        'target_waveform': target_waveforms,
        'target_speaker_id': target_speaker_ids,
        'source_speakers': [item['source_speaker'] for item in batch],
        'target_speakers': [item['target_speaker'] for item in batch]
    }

def save_model_config(model, filename):
    """Save model configuration"""
    config = {
        'model_type': model.__class__.__name__,
        'hubert_model_name': getattr(model, 'hubert_model_name', 'ZhenYe234/hubert_base_general_audio'),
        'd_model': model.ssm_encoder.layers[0].d_model if model.ssm_encoder.layers else 768,
        'ssm_layers': len(model.ssm_encoder.layers),
        'flow_steps': model.flow_matching.steps
    }
    
    with open(filename, 'w') as f:
        json.dump(config, f, indent=2)

def load_model_from_config(config_file, checkpoint_file):
    """Load model from config and checkpoint"""
    with open(config_file, 'r') as f:
        config = json.load(f)
    
    from model import VoiceConversionModel
    model = VoiceConversionModel(
        hubert_model_name=config.get('hubert_model_name', 'ZhenYe234/hubert_base_general_audio'),
        d_model=config.get('d_model', 768),
        ssm_layers=config.get('ssm_layers', 4),
        flow_steps=config.get('flow_steps', 100),
        n_speakers=config.get('n_speakers', 256),
        waveform_length=config.get('waveform_length', 16384),
        use_retrieval=config.get('use_retrieval', False)
    )
    
    checkpoint = torch.load(checkpoint_file, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    
    return model