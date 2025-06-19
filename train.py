import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast
import numpy as np
from pathlib import Path
import wandb
from tqdm import tqdm
import time
import warnings

from model import VoiceConversionModel
from utils import VoiceConversionDataset, collate_fn
from flow_matching import FlowScheduler

warnings.filterwarnings("ignore", category=UserWarning)

class OptimizedVoiceConversionTrainer:
    """
    🚀 최적화된 Voice Conversion Trainer
    - AMP FP16 혼합 정밀도
    - Rectified Flow 
    - F0 조건부 생성
    - 컴파일 최적화
    """
    
    def __init__(self, config):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 🔥 AMP 스케일러 초기화
        self.scaler = GradScaler()
        self.use_amp = config.get('use_amp', True) and torch.cuda.is_available()
        
        print(f"🚀 Initializing trainer with:")
        print(f"   Device: {self.device}")
        print(f"   AMP FP16: {'✅ Enabled' if self.use_amp else '❌ Disabled'}")
        
        # 🔥 모델 초기화
        self.model = VoiceConversionModel(
            d_model=config.get('d_model', 768),
            ssm_layers=config.get('ssm_layers', 3),
            flow_steps=config.get('flow_steps', 20),  # Rectified Flow는 더 적은 단계
            n_speakers=config.get('n_speakers', 256),
            waveform_length=config.get('waveform_length', 16384),
            use_retrieval=config.get('use_retrieval', True),
            lora_rank=config.get('lora_rank', 16),
            adapter_dim=config.get('adapter_dim', 64),
            use_f0_conditioning=config.get('use_f0_conditioning', True)
        ).to(self.device)
        
        # 🚀 모델 컴파일 (PyTorch 2.0+)
        if config.get('compile_model', True):
            self.model.compile_model()
        
        print(f"🎵 F0 conditioning: {'✅ Enabled' if config.get('use_f0_conditioning', True) else '❌ Disabled'}")
        print(f"🔍 Retrieval: {'✅ Enabled' if config.get('use_retrieval', True) else '❌ Disabled'}")
        
        # 🔥 최적화된 옵티마이저
        self._setup_optimizer()
        
        # 스케줄러
        self.scheduler = optim.lr_scheduler.OneCycleLR(
            self.optimizer,
            max_lr=config.get('lr', 1e-4),
            total_steps=config.get('max_epochs', 80) * config.get('steps_per_epoch', 1000),
            pct_start=0.1,  # 10% 웜업
            div_factor=25,  # 초기 LR = max_lr / 25
            final_div_factor=10000  # 최종 LR = max_lr / 10000
        )
        
        # 손실 가중치
        self.flow_weight = config.get('flow_weight', 1.0)
        self.f0_weight = config.get('f0_weight', 0.1)
        self.vuv_weight = config.get('vuv_weight', 0.1)
        
        # 🔥 동적 스케줄링
        self.flow_scheduler = FlowScheduler()
        
        # 로깅
        if config.get('use_wandb', False):
            wandb.init(
                project="voice-conversion-optimized",
                config=config,
                name=f"optimized_{config.get('experiment_name', 'default')}"
            )
        
        # 성능 메트릭
        self.training_times = []
        self.memory_usage = []
        
    def _setup_optimizer(self):
        """최적화된 옵티마이저 설정"""
        # 파라미터 그룹 분리
        no_decay = ['bias', 'LayerNorm.weight', 'layer_norm.weight']
        
        optimizer_grouped_parameters = [
            {
                'params': [p for n, p in self.model.named_parameters() 
                          if not any(nd in n for nd in no_decay) and p.requires_grad],
                'weight_decay': self.config.get('weight_decay', 1e-5),
                'lr': self.config.get('lr', 1e-4)
            },
            {
                'params': [p for n, p in self.model.named_parameters() 
                          if any(nd in n for nd in no_decay) and p.requires_grad],
                'weight_decay': 0.0,
                'lr': self.config.get('lr', 1e-4)
            }
        ]
        
        # 🔥 AdamW with fused optimization
        self.optimizer = optim.AdamW(
            optimizer_grouped_parameters,
            lr=self.config.get('lr', 1e-4),
            betas=(0.9, 0.999),
            eps=1e-8,
            fused=True if torch.cuda.is_available() else False  # CUDA 최적화
        )
        
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in self.model.parameters())
        
        print(f"⚙️ Optimizer setup:")
        print(f"   Total parameters: {total_params:,}")
        print(f"   Trainable parameters: {trainable_params:,}")
        print(f"   Trainable ratio: {trainable_params/total_params*100:.1f}%")
        print(f"   Fused AdamW: {'✅ Enabled' if torch.cuda.is_available() else '❌ Disabled'}")
    
    def train_epoch(self, train_loader, epoch):
        """🔥 AMP FP16 최적화된 훈련 에포크"""
        self.model.train()
        
        total_loss = 0
        flow_loss_sum = 0
        f0_loss_sum = 0
        vuv_loss_sum = 0
        
        # 동적 추론 단계 계산
        inference_steps = self.flow_scheduler.get_progressive_schedule(
            max_steps=self.config.get('flow_steps', 20),
            current_epoch=epoch,
            total_epochs=self.config.get('max_epochs', 80)
        )
        
        epoch_start_time = time.time()
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}")
        for batch_idx, batch in enumerate(pbar):
            batch_start_time = time.time()
            
            # 데이터 GPU 이동
            source_waveform = batch['source_waveform'].to(self.device, non_blocking=True)
            target_waveform = batch['target_waveform'].to(self.device, non_blocking=True)
            target_speaker_id = batch['target_speaker_id'].to(self.device, non_blocking=True)
            
            # F0/VUV 데이터
            f0_target = batch.get('f0_target')
            vuv_target = batch.get('vuv_target')
            if f0_target is not None:
                f0_target = f0_target.to(self.device, non_blocking=True)
            if vuv_target is not None:
                vuv_target = vuv_target.to(self.device, non_blocking=True)
            
            # 🔥 AMP 혼합 정밀도 순전파
            with autocast(enabled=self.use_amp):
                outputs = self.model(
                    source_waveform=source_waveform,
                    target_speaker_id=target_speaker_id,
                    target_waveform=target_waveform,
                    f0_target=f0_target,
                    vuv_target=vuv_target,
                    training=True,
                    inference_method='fast_rectified',
                    num_steps=inference_steps
                )
                
                loss = outputs['total_loss']
                flow_loss = outputs['flow_loss']
            
            # 🔥 AMP 스케일된 역전파
            self.optimizer.zero_grad(set_to_none=True)  # 메모리 효율성
            if self.use_amp:
                self.scaler.scale(loss).backward()
                
                # 그래디언트 클리핑 (스케일된 그래디언트에)
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()
            
            self.scheduler.step()
            
            # 메트릭 수집
            total_loss += loss.item()
            flow_loss_sum += flow_loss.item()
            
            aux_loss_sum = 0
            if 'f0_loss' in outputs:
                f0_loss_sum += outputs['f0_loss'].item()
                aux_loss_sum += outputs['f0_loss'].item()
            if 'vuv_loss' in outputs:
                vuv_loss_sum += outputs['vuv_loss'].item()
                aux_loss_sum += outputs['vuv_loss'].item()
            
            # 성능 메트릭
            batch_time = time.time() - batch_start_time
            
            # 진행률 표시
            pbar_dict = {
                'Loss': f'{loss.item():.4f}',
                'Flow': f'{flow_loss.item():.4f}',
                'Steps': f'{inference_steps}',
                'LR': f'{self.optimizer.param_groups[0]["lr"]:.2e}',
                'Time': f'{batch_time:.2f}s'
            }
            
            if aux_loss_sum > 0:
                pbar_dict['F0+VUV'] = f'{aux_loss_sum:.4f}'
            
            if self.use_amp:
                pbar_dict['Scale'] = f'{self.scaler.get_scale():.0f}'
            
            pbar.set_postfix(pbar_dict)
            
            # Wandb 로깅
            if self.config.get('use_wandb', False) and batch_idx % 50 == 0:
                log_dict = {
                    'train/loss': loss.item(),
                    'train/flow_loss': flow_loss.item(),
                    'train/lr': self.optimizer.param_groups[0]['lr'],
                    'train/inference_steps': inference_steps,
                    'train/batch_time': batch_time,
                    'train/epoch': epoch + 1
                }
                
                if 'f0_loss' in outputs:
                    log_dict['train/f0_loss'] = outputs['f0_loss'].item()
                if 'vuv_loss' in outputs:
                    log_dict['train/vuv_loss'] = outputs['vuv_loss'].item()
                if self.use_amp:
                    log_dict['train/grad_scale'] = self.scaler.get_scale()
                
                wandb.log(log_dict)
            
            # 메모리 사용량 체크 (가끔씩)
            if batch_idx % 100 == 0 and torch.cuda.is_available():
                memory_allocated = torch.cuda.memory_allocated(self.device) / 1024**3
                self.memory_usage.append(memory_allocated)
        
        # 에포크 성능 기록
        epoch_time = time.time() - epoch_start_time
        self.training_times.append(epoch_time)
        
        num_batches = len(train_loader)
        metrics = {
            'total_loss': total_loss / num_batches,
            'flow_loss': flow_loss_sum / num_batches,
            'f0_loss': f0_loss_sum / num_batches if f0_loss_sum > 0 else 0,
            'vuv_loss': vuv_loss_sum / num_batches if vuv_loss_sum > 0 else 0,
            'epoch_time': epoch_time,
            'inference_steps': inference_steps
        }
        
        return metrics
    
    @torch.no_grad()
    def validate(self, val_loader, epoch):
        """🚀 최적화된 검증"""
        self.model.eval()
        
        total_loss = 0
        flow_loss_sum = 0
        f0_loss_sum = 0
        vuv_loss_sum = 0
        
        # 검증은 더 적은 단계로
        val_steps = max(4, self.config.get('flow_steps', 20) // 4)
        
        with torch.cuda.amp.autocast(enabled=self.use_amp):
            for batch in tqdm(val_loader, desc="Validation", leave=False):
                source_waveform = batch['source_waveform'].to(self.device, non_blocking=True)
                target_waveform = batch['target_waveform'].to(self.device, non_blocking=True)
                target_speaker_id = batch['target_speaker_id'].to(self.device, non_blocking=True)
                
                f0_target = batch.get('f0_target')
                vuv_target = batch.get('vuv_target')
                if f0_target is not None:
                    f0_target = f0_target.to(self.device, non_blocking=True)
                if vuv_target is not None:
                    vuv_target = vuv_target.to(self.device, non_blocking=True)
                
                outputs = self.model(
                    source_waveform=source_waveform,
                    target_speaker_id=target_speaker_id,
                    target_waveform=target_waveform,
                    f0_target=f0_target,
                    vuv_target=vuv_target,
                    training=True,
                    inference_method='fast_rectified',
                    num_steps=val_steps
                )
                
                loss = outputs['total_loss']
                flow_loss = outputs['flow_loss']
                
                total_loss += loss.item()
                flow_loss_sum += flow_loss.item()
                
                if 'f0_loss' in outputs:
                    f0_loss_sum += outputs['f0_loss'].item()
                if 'vuv_loss' in outputs:
                    vuv_loss_sum += outputs['vuv_loss'].item()
        
        num_batches = len(val_loader)
        val_metrics = {
            'total_loss': total_loss / num_batches,
            'flow_loss': flow_loss_sum / num_batches,
            'f0_loss': f0_loss_sum / num_batches if f0_loss_sum > 0 else 0,
            'vuv_loss': vuv_loss_sum / num_batches if vuv_loss_sum > 0 else 0
        }
        
        if self.config.get('use_wandb', False):
            wandb.log({f'val/{k}': v for k, v in val_metrics.items()})
        
        return val_metrics
    
    def train(self, train_loader, val_loader=None):
        """메인 훈련 루프"""
        print(f"\n🚀 Starting optimized training:")
        print(f"   Rectified Flow steps: {self.config.get('flow_steps', 20)}")
        print(f"   AMP FP16: {'✅' if self.use_amp else '❌'}")
        print(f"   Model compilation: {'✅' if self.config.get('compile_model', True) else '❌'}")
        
        best_val_loss = float('inf')
        
        for epoch in range(self.config.get('max_epochs', 80)):
            epoch_start = time.time()
            
            print(f"\n🔥 Epoch {epoch+1}/{self.config.get('max_epochs', 80)}")
            
            # 검색 데이터 수집 (첫 에포크)
            if epoch == 0 and self.model.use_retrieval:
                self._collect_retrieval_data(train_loader)
            
            # 훈련
            train_metrics = self.train_epoch(train_loader, epoch)
            
            print(f"📊 Train - Loss: {train_metrics['total_loss']:.4f}, " +
                  f"Flow: {train_metrics['flow_loss']:.4f}, " +
                  f"Time: {train_metrics['epoch_time']:.1f}s")
            
            if train_metrics['f0_loss'] > 0:
                print(f"        🎵 F0: {train_metrics['f0_loss']:.4f}, " +
                      f"VUV: {train_metrics['vuv_loss']:.4f}")
            
            # 검증
            if val_loader is not None:
                val_metrics = self.validate(val_loader, epoch)
                print(f"📊 Val   - Loss: {val_metrics['total_loss']:.4f}, " +
                      f"Flow: {val_metrics['flow_loss']:.4f}")
                
                if val_metrics['f0_loss'] > 0:
                    print(f"        🎵 F0: {val_metrics['f0_loss']:.4f}, " +
                          f"VUV: {val_metrics['vuv_loss']:.4f}")
                
                # 최고 모델 저장
                if val_metrics['total_loss'] < best_val_loss:
                    best_val_loss = val_metrics['total_loss']
                    self.save_checkpoint(f"best_model_epoch_{epoch+1}.pt", epoch)
                    print(f"💾 New best model! (Loss: {best_val_loss:.4f})")
            
            # 정기 체크포인트
            if (epoch + 1) % self.config.get('save_every', 10) == 0:
                self.save_checkpoint(f"checkpoint_epoch_{epoch+1}.pt", epoch)
            
            # 성능 리포트
            if (epoch + 1) % 20 == 0:
                self._print_performance_stats(epoch + 1)
        
        print(f"\n🎉 Training completed!")
        self._print_final_stats()
    
    def _collect_retrieval_data(self, train_loader):
        """검색 모듈용 데이터 수집"""
        if not self.model.use_retrieval:
            return
        
        print("🔍 Collecting retrieval features...")
        self.model.eval()
        
        with torch.no_grad(), torch.cuda.amp.autocast(enabled=self.use_amp):
            for batch in tqdm(train_loader, desc="Collecting features", leave=False):
                source_waveform = batch['source_waveform'].to(self.device, non_blocking=True)
                target_speaker_id = batch['target_speaker_id'].to(self.device, non_blocking=True)
                
                # HuBERT 특성 추출
                if source_waveform.dim() == 3:
                    source_mono = source_waveform.mean(dim=1)
                else:
                    source_mono = source_waveform
                
                hubert_output = self.model.hubert(source_mono)
                content_repr = hubert_output.last_hidden_state
                encoded_content = self.model.ssm_encoder(content_repr)
                
                # 스피커 임베딩
                speaker_emb = self.model.speaker_embedding(target_speaker_id)
                speaker_emb = speaker_emb.unsqueeze(1).expand(-1, encoded_content.size(1), -1)
                
                # 결합 및 풀링
                condition = encoded_content + speaker_emb
                condition = self.model.speaker_adapter(condition)
                condition_pooled = condition.mean(dim=1)
                
                # 검색 모듈에 추가
                self.model.retrieval_module.add_training_features(
                    condition_pooled, target_speaker_id
                )
        
        print("✅ Retrieval data collection completed")
        self.model.train()
    
    def save_checkpoint(self, filename, epoch):
        """최적화된 체크포인트 저장"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'scaler_state_dict': self.scaler.state_dict() if self.use_amp else None,
            'config': self.config,
            'training_times': self.training_times,
            'memory_usage': self.memory_usage
        }
        torch.save(checkpoint, filename)
        print(f"💾 Checkpoint saved: {filename}")
    
    def _print_performance_stats(self, epoch):
        """성능 통계 출력"""
        if self.training_times:
            avg_time = sum(self.training_times[-10:]) / min(10, len(self.training_times))
            print(f"⚡ Avg epoch time (last 10): {avg_time:.1f}s")
        
        if self.memory_usage and torch.cuda.is_available():
            max_memory = max(self.memory_usage[-50:]) if len(self.memory_usage) >= 50 else max(self.memory_usage)
            print(f"🖥️ Peak GPU memory: {max_memory:.2f} GB")
    
    def _print_final_stats(self):
        """최종 통계 출력"""
        print(f"\n📊 Final Training Statistics:")
        if self.training_times:
            total_time = sum(self.training_times)
            avg_time = total_time / len(self.training_times)
            print(f"   Total training time: {total_time/3600:.2f} hours")
            print(f"   Average epoch time: {avg_time:.1f}s")
        
        if self.memory_usage:
            avg_memory = sum(self.memory_usage) / len(self.memory_usage)
            max_memory = max(self.memory_usage)
            print(f"   Average GPU memory: {avg_memory:.2f} GB")
            print(f"   Peak GPU memory: {max_memory:.2f} GB")
        
        print(f"   AMP FP16 used: {'Yes' if self.use_amp else 'No'}")
        print(f"   Model compiled: {'Yes' if self.config.get('compile_model', True) else 'No'}")

def main():
    """최적화된 훈련 메인 함수"""
    config = {
        # 🔥 최적화 설정
        'use_amp': True,  # AMP FP16 활성화
        'compile_model': True,  # PyTorch 2.0 컴파일
        
        # 모델 설정
        'd_model': 768,
        'ssm_layers': 3,
        'flow_steps': 20,  # Rectified Flow로 더 적은 단계
        'n_speakers': 256,
        'waveform_length': 16384,
        'use_retrieval': True,
        'lora_rank': 16,
        'adapter_dim': 64,
        'use_f0_conditioning': True,  # 🎵 F0 조건부 생성
        
        # 훈련 설정
        'batch_size': 12,  # FP16으로 더 큰 배치 가능
        'lr': 1e-4,
        'weight_decay': 1e-5,
        'max_epochs': 80,
        'save_every': 10,
        
        # 데이터 설정
        'data_dir': './data',
        'sample_rate': 44100,
        'channels': 2,
        'extract_f0': True,  # F0 추출 활성화
        'f0_method': 'pyin',
        'f0_weight': 0.1,
        'vuv_weight': 0.1,
        
        # 로깅
        'use_wandb': False,
        'experiment_name': 'rectified_flow_f0'
    }
    
    # 데이터셋 로드
    print("📁 Loading optimized dataset...")
    
    if Path(config['data_dir']).is_dir() and not (Path(config['data_dir']) / 'train').exists():
        full_dataset = VoiceConversionDataset(
            data_dir=config['data_dir'],
            sample_rate=config['sample_rate'],
            waveform_length=config['waveform_length'],
            channels=config['channels'],
            extract_f0=config['extract_f0'],
            hop_length=512,
            f0_method=config['f0_method']
        )
        
        total_size = len(full_dataset)
        train_size = int(0.8 * total_size)
        val_size = total_size - train_size
        
        train_dataset, val_dataset = torch.utils.data.random_split(
            full_dataset, [train_size, val_size]
        )
        
    else:
        train_dataset = VoiceConversionDataset(
            data_dir=Path(config['data_dir']) / 'train',
            sample_rate=config['sample_rate'],
            waveform_length=config['waveform_length'],
            channels=config['channels'],
            extract_f0=config['extract_f0'],
            hop_length=512,
            f0_method=config['f0_method']
        )
        
        val_dataset = VoiceConversionDataset(
            data_dir=Path(config['data_dir']) / 'val',
            sample_rate=config['sample_rate'],
            waveform_length=config['waveform_length'],
            channels=config['channels'],
            extract_f0=config['extract_f0'],
            hop_length=512,
            f0_method=config['f0_method']
        )
    
    # 데이터셋 정보
    if hasattr(train_dataset, 'dataset'):
        dataset_info = train_dataset.dataset.get_speaker_info()
        train_dataset.dataset.print_sample_pairs()
    else:
        dataset_info = train_dataset.get_speaker_info()
        train_dataset.print_sample_pairs()
    
    # 화자 수 업데이트
    config['n_speakers'] = dataset_info['total_speakers']
    config['steps_per_epoch'] = len(train_dataset) // config['batch_size']
    
    print(f"\n🎯 Optimized Dataset Info:")
    print(f"   👥 Speakers: {dataset_info['total_speakers']}")
    print(f"   📊 Training pairs: {len(train_dataset)}")
    print(f"   🔍 Validation pairs: {len(val_dataset)}")
    print(f"   🎵 F0 conditioning: ✅ Enabled")
    print(f"   🚀 Rectified Flow: ✅ Enabled")
    
    # 🔥 최적화된 데이터 로더
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=8,  # 더 많은 워커
        pin_memory=True,
        persistent_workers=True,  # 워커 재사용
        collate_fn=collate_fn,
        prefetch_factor=2,  # 미리 가져오기
        drop_last=True  # 안정적인 배치 크기
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True,
        collate_fn=collate_fn,
        drop_last=False
    )
    
    # 트레이너 초기화 및 훈련
    trainer = OptimizedVoiceConversionTrainer(config)
    trainer.train(train_loader, val_loader)

if __name__ == "__main__":
    main()