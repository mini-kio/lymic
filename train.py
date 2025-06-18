import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
import torchaudio
from pathlib import Path
import wandb
from tqdm import tqdm

from model import VoiceConversionModel
from utils import extract_f0, compute_vuv, VoiceConversionDataset, collate_fn

class VoiceConversionTrainer:
    def __init__(self, config):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Initialize model
        self.model = VoiceConversionModel(
            d_model=config.get('d_model', 768),
            ssm_layers=config.get('ssm_layers', 3),  # Optimized depth
            flow_steps=config.get('flow_steps', 50),  # Faster inference
            n_speakers=config.get('n_speakers', 256),
            waveform_length=config.get('waveform_length', 16384),
            use_retrieval=config.get('use_retrieval', True),  # Enable by default
            lora_rank=config.get('lora_rank', 16),
            adapter_dim=config.get('adapter_dim', 64)
        ).to(self.device)
        
        print(f"🚀 Model initialized with optimized settings:")
        print(f"   SSM layers: {config.get('ssm_layers', 3)} (reduced for speed)")
        print(f"   Flow steps: {config.get('flow_steps', 50)} (optimized)")
        print(f"   Retrieval: {'✅ Enabled' if config.get('use_retrieval', True) else '❌ Disabled'}")
        
        # Optimizer
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=config.get('lr', 1e-4),
            weight_decay=config.get('weight_decay', 1e-5)
        )
        
        # Scheduler
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=config.get('max_epochs', 100),
            eta_min=config.get('min_lr', 1e-6)
        )
        
        # Loss weights
        self.flow_weight = config.get('flow_weight', 1.0)
        self.vuv_weight = config.get('vuv_weight', 0.5)
        
        # Logging
        if config.get('use_wandb', False):
            wandb.init(project="f0-prediction", config=config)
    
    def train_epoch(self, train_loader):
        self.model.train()
        total_loss = 0
        flow_loss_sum = 0
        vuv_loss_sum = 0
        
        pbar = tqdm(train_loader, desc="Training")
        for batch_idx, batch in enumerate(pbar):
            source_waveform = batch['source_waveform'].to(self.device)
            target_waveform = batch['target_waveform'].to(self.device)
            target_speaker_id = batch['target_speaker_id'].to(self.device)
            
            # Optional F0 and V/UV targets for auxiliary loss
            f0_target = batch.get('f0_target', None)
            vuv_target = batch.get('vuv_target', None)
            if f0_target is not None:
                f0_target = f0_target.to(self.device)
            if vuv_target is not None:
                vuv_target = vuv_target.to(self.device)
            
            # Forward pass
            outputs = self.model(
                source_waveform=source_waveform,
                target_speaker_id=target_speaker_id,
                target_waveform=target_waveform,
                f0_target=f0_target,
                vuv_target=vuv_target,
                training=True
            )
            
            # Get loss
            loss = outputs['total_loss']
            flow_loss = outputs['flow_loss']
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            
            self.optimizer.step()
            
            # Logging
            total_loss += loss.item()
            flow_loss_sum += flow_loss.item()
            
            # Optional auxiliary losses
            aux_loss_sum = 0
            if 'f0_loss' in outputs:
                aux_loss_sum += outputs['f0_loss'].item()
            if 'vuv_loss' in outputs:
                aux_loss_sum += outputs['vuv_loss'].item()
            
            pbar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'Flow': f'{flow_loss.item():.4f}',
                'Aux': f'{aux_loss_sum:.4f}'
            })
            
            if self.config.get('use_wandb', False):
                log_dict = {
                    'train/loss': loss.item(),
                    'train/flow_loss': flow_loss.item(),
                    'train/lr': self.optimizer.param_groups[0]['lr']
                }
                if 'f0_loss' in outputs:
                    log_dict['train/f0_loss'] = outputs['f0_loss'].item()
                if 'vuv_loss' in outputs:
                    log_dict['train/vuv_loss'] = outputs['vuv_loss'].item()
                wandb.log(log_dict)
        
        return {
            'total_loss': total_loss / len(train_loader),
            'flow_loss': flow_loss_sum / len(train_loader)
        }
    
    def validate(self, val_loader):
        self.model.eval()
        total_loss = 0
        flow_loss_sum = 0
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validation"):
                source_waveform = batch['source_waveform'].to(self.device)
                target_waveform = batch['target_waveform'].to(self.device)
                target_speaker_id = batch['target_speaker_id'].to(self.device)
                
                outputs = self.model(
                    source_waveform=source_waveform,
                    target_speaker_id=target_speaker_id,
                    target_waveform=target_waveform,
                    training=True,
                    inference_method='fast_inverse',  # 🚀 Fast validation
                    num_steps=6  # Even faster for validation
                )
                
                flow_loss = outputs['flow_loss']
                loss = outputs['total_loss']
                
                total_loss += loss.item()
                flow_loss_sum += flow_loss.item()
        
        val_metrics = {
            'total_loss': total_loss / len(val_loader),
            'flow_loss': flow_loss_sum / len(val_loader)
        }
        
        if self.config.get('use_wandb', False):
            wandb.log({f'val/{k}': v for k, v in val_metrics.items()})
        
        return val_metrics
    
    def train(self, train_loader, val_loader=None):
        best_val_loss = float('inf')
        
        for epoch in range(self.config.get('max_epochs', 80)):
            print(f"\n🚀 Epoch {epoch+1}/{self.config.get('max_epochs', 80)}")
            
            # Set epoch for retrieval data collection
            self.epoch = epoch
            
            # Training
            train_metrics = self.train_epoch(train_loader)
            print(f"Train Loss: {train_metrics['total_loss']:.4f}")
            
            # Validation
            if val_loader is not None:
                val_metrics = self.validate(val_loader)
                print(f"Val Loss: {val_metrics['total_loss']:.4f}")
                
                # Save best model
                if val_metrics['total_loss'] < best_val_loss:
                    best_val_loss = val_metrics['total_loss']
                    self.save_checkpoint(f"best_model_epoch_{epoch+1}.pt")
            
            # Scheduler step
            self.scheduler.step()
            
            # Save checkpoint
            if (epoch + 1) % self.config.get('save_every', 5) == 0:
                self.save_checkpoint(f"checkpoint_epoch_{epoch+1}.pt")
    
    def save_checkpoint(self, filename):
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'config': self.config
        }
        torch.save(checkpoint, filename)
        print(f"Checkpoint saved: {filename}")
    
    def load_checkpoint(self, filename):
        checkpoint = torch.load(filename, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        print(f"Checkpoint loaded: {filename}")

def main():
    # Configuration (Optimized)
    config = {
        'batch_size': 6,  # Slightly larger due to optimizations
        'lr': 1e-4,
        'weight_decay': 1e-5,
        'max_epochs': 80,  # Fewer epochs due to better architecture
        'min_lr': 1e-6,
        'd_model': 768,
        'ssm_layers': 3,  # 🔥 Optimized: 6 → 3 layers
        'flow_steps': 50,  # 🔥 Optimized: 100 → 50 steps
        'n_speakers': 256,
        'waveform_length': 16384,
        'use_retrieval': True,  # 🔍 Enable retrieval by default
        'lora_rank': 16,
        'adapter_dim': 64,
        'save_every': 5,  # Save more frequently
        'use_wandb': False,
        'data_dir': './data',
        'sample_rate': 44100,
        'hop_length': 512,
        'channels': 2,
        
        # 🚀 Fast inference settings
        'inference_method': 'fast_inverse',
        'inference_steps': 8
    }
    
    # Create datasets (RVC-style structure)
    print("📁 Loading RVC-style dataset...")
    
    # Option 1: 전체 데이터셋 사용 (train/val 구분 없음)
    if Path(config['data_dir']).is_dir() and not (Path(config['data_dir']) / 'train').exists():
        print("🔄 Using single dataset directory (no train/val split)")
        
        full_dataset = VoiceConversionDataset(
            data_dir=config['data_dir'],
            sample_rate=config['sample_rate'],
            waveform_length=config['waveform_length'],
            channels=config['channels']
        )
        
        # 랜덤 train/val split (80/20)
        total_size = len(full_dataset)
        train_size = int(0.8 * total_size)
        val_size = total_size - train_size
        
        train_dataset, val_dataset = torch.utils.data.random_split(
            full_dataset, [train_size, val_size]
        )
        
        print(f"📊 Dataset split: {train_size} train, {val_size} val")
        
    else:
        # Option 2: train/val 폴더 구조 사용
        print("📁 Using train/val directory structure")
        
        train_dataset = VoiceConversionDataset(
            data_dir=config['data_dir'] / 'train' if isinstance(config['data_dir'], Path) else Path(config['data_dir']) / 'train',
            sample_rate=config['sample_rate'],
            waveform_length=config['waveform_length'],
            channels=config['channels']
        )
        
        val_dataset = VoiceConversionDataset(
            data_dir=config['data_dir'] / 'val' if isinstance(config['data_dir'], Path) else Path(config['data_dir']) / 'val',
            sample_rate=config['sample_rate'],
            waveform_length=config['waveform_length'],
            channels=config['channels']
        )
    
    # 데이터셋 정보 출력
    if hasattr(train_dataset, 'dataset'):  # random_split인 경우
        dataset_info = train_dataset.dataset.get_speaker_info()
        train_dataset.dataset.print_sample_pairs()
    else:
        dataset_info = train_dataset.get_speaker_info()
        train_dataset.print_sample_pairs()
    
    print(f"\n🎯 Dataset Info:")
    print(f"   👥 Total speakers: {dataset_info['total_speakers']}")
    print(f"   📊 Training pairs: {len(train_dataset)}")
    print(f"   🔍 Validation pairs: {len(val_dataset)}")
    
    # 화자 목록 출력 (처음 10명만)
    speakers = list(dataset_info['speakers'])[:10]
    print(f"   👤 Speakers (first 10): {speakers}")
    if dataset_info['total_speakers'] > 10:
        print(f"       ... and {dataset_info['total_speakers'] - 10} more")
    
    # 모델 설정 업데이트 (화자 수에 맞춰)
    config['n_speakers'] = dataset_info['total_speakers']
    print(f"   🔧 Updated n_speakers to {config['n_speakers']}")
    
    # Data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        collate_fn=collate_fn
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        collate_fn=collate_fn
    )
    
    # Initialize trainer
    trainer = VoiceConversionTrainer(config)
    
    # Train
    trainer.train(train_loader, val_loader)

if __name__ == "__main__":
    main()