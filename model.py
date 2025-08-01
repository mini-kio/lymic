import torch
import torch.nn as nn
from transformers import HubertModel
from ssm import S6SSMEncoder
from flow_matching import RectifiedFlow
import torch.nn.functional as F
import math
import sys
import os
from torch.autograd import Function
sys.path.append(os.path.join(os.path.dirname(__file__), 'dcae_vocoder'))
from music_dcae_pipeline import MusicDCAE

class GradReverse(Function):
    """Gradient Reversal Layer - flips gradient sign during backprop"""
    @staticmethod
    def forward(ctx, x, lambd):
        ctx.lambd = lambd
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.lambd * grad_output, None  # sign flip

class GRL(nn.Module):
    """Gradient Reversal Layer wrapper"""
    def __init__(self, lambd=1.0):
        super().__init__()
        self.lambd = lambd
        
    def forward(self, x):
        return GradReverse.apply(x, self.lambd)
    
    def set_lambda(self, lambd):
        """Update lambda for scheduling"""
        self.lambd = lambd

class DualHeadmHuBERT(nn.Module):
    """
    Dual-Head mHuBERT for Content-Speaker separation
    - ContentHead: for linguistic content (target branch)
    - SpeakerHead: for speaker characteristics (reference branch)
    - GRL: prevents speaker info leakage to content
    """
    def __init__(self, hubert_model_name="utter-project/mHuBERT-147", d_content=256, d_speaker=256):
        super().__init__()
        
        # Load single mHuBERT (shared backbone)
        self.hubert = HubertModel.from_pretrained(hubert_model_name)
        
        # Freeze initially (will unfreeze top layers later)
        for param in self.hubert.parameters():
            param.requires_grad = False
        
        hidden_size = self.hubert.config.hidden_size  # 768
        
        # Content Head - for linguistic content
        self.content_head = nn.Sequential(
            nn.Linear(hidden_size, d_content),
            nn.LayerNorm(d_content),
            nn.Dropout(0.1)
        )
        
        # Speaker Head - for speaker characteristics  
        self.speaker_head = nn.Sequential(
            nn.Linear(hidden_size, d_speaker),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(d_speaker, d_speaker)
        )
        
        # Gradient Reversal Layer (applied before speaker head in content branch)
        self.grl = GRL(lambd=0.0)  # Start with 0, gradually increase
        
        print(f"DualHead mHuBERT initialized:")
        print(f"   Backbone: {hubert_model_name}")
        print(f"   Content dim: {d_content}")
        print(f"   Speaker dim: {d_speaker}")
        print(f"   GRL lambda: {self.grl.lambd}")
    
    def unfreeze_top_layers(self, num_layers=4):
        """Unfreeze top N layers for fine-tuning"""
        # mHuBERT has encoder.layers
        total_layers = len(self.hubert.encoder.layers)
        for i in range(total_layers - num_layers, total_layers):
            for param in self.hubert.encoder.layers[i].parameters():
                param.requires_grad = True
        print(f"Unfroze top {num_layers} layers of mHuBERT")
    
    def set_grl_lambda(self, lambd):
        """Update GRL lambda for scheduling"""
        self.grl.set_lambda(lambd)
    
    def forward(self, audio, branch="content"):
        """
        Forward pass with branch selection
        Args:
            audio: (B, T) waveform
            branch: "content" or "speaker"
        """
        # Extract mHuBERT features
        with torch.cuda.amp.autocast(enabled=False):
            if branch == "content" and self.training:
                # Content branch - allow gradients for unfrozen layers
                hubert_output = self.hubert(audio.float())
            else:
                # Speaker branch or inference - no gradients
                with torch.no_grad():
                    hubert_output = self.hubert(audio.float())
                    
        features = hubert_output.last_hidden_state  # (B, T, 768)
        
        if branch == "content":
            # Content branch - apply GRL to prevent speaker leakage
            content_features = self.content_head(self.grl(features))
            return content_features  # (B, T, d_content)
        
        elif branch == "speaker":
            # Speaker branch - extract speaker characteristics
            speaker_features = self.speaker_head(features)  # (B, T, d_speaker)
            speaker_features = F.normalize(speaker_features, dim=-1)  # L2 normalize
            return speaker_features  # (B, T, d_speaker)
        
        else:
            raise ValueError(f"Unknown branch: {branch}")

def orthogonality_loss(content_features, speaker_features):
    """
    Orthogonality loss to ensure content and speaker features are uncorrelated
    Args:
        content_features: (B, T, d_content)
        speaker_features: (B, T, d_speaker)  
    Returns:
        loss: scalar tensor
    """
    # Compute cosine similarity at each time step
    cos_sim = F.cosine_similarity(content_features, speaker_features, dim=-1)  # (B, T)
    
    # Minimize absolute cosine similarity (push towards orthogonal)
    orth_loss = cos_sim.abs().mean()
    
    return orth_loss

class F0Embedding(nn.Module):
    """F0 information embedding for voice conversion"""
    def __init__(self, d_model=768):
        super().__init__()
        self.f0_proj = nn.Sequential(
            nn.Linear(1, d_model // 4),
            nn.SiLU(),
            nn.Linear(d_model // 4, d_model // 2)
        )
        self.vuv_proj = nn.Sequential(
            nn.Linear(1, d_model // 4),
            nn.SiLU(), 
            nn.Linear(d_model // 4, d_model // 2)
        )
        
        self.combine_proj = nn.Linear(d_model, d_model)
        
    def forward(self, f0, vuv, semitone_shift=0.0):
        """
        Args:
            f0: (B, T) 정규화된 F0 값
            vuv: (B, T) voiced/unvoiced 플래그
            semitone_shift: float 또는 (B,) 세미톤 시프트 (-12 ~ +12)
        Returns:
            (B, T, d_model) F0 임베딩
        """
        if semitone_shift != 0.0:
            f0_shifted = self.apply_semitone_shift(f0, vuv, semitone_shift)
        else:
            f0_shifted = f0
        
        f0_emb = self.f0_proj(f0_shifted.unsqueeze(-1))  # (B, T, d_model//2)
        vuv_emb = self.vuv_proj(vuv.unsqueeze(-1))  # (B, T, d_model//2)
        
        f0_emb = f0_emb * vuv.unsqueeze(-1)
        
        combined = torch.cat([f0_emb, vuv_emb], dim=-1)  # (B, T, d_model)
        return self.combine_proj(combined)
    
    def apply_semitone_shift(self, f0, vuv, semitone_shift):
        """
        Apply semitone shift to F0
        Args:
            f0: (B, T) F0 values (Hz or log scale)
            vuv: (B, T) voiced/unvoiced mask
            semitone_shift: float or (B,) semitone shift (-12 ~ +12)
        Returns:
            (B, T) shifted F0 values
        """
        if isinstance(semitone_shift, (int, float)):
            if semitone_shift == 0.0:
                return f0
            semitone_shift = torch.tensor(semitone_shift, device=f0.device, dtype=f0.dtype)
        
        if semitone_shift.dim() == 0:
            semitone_shift = semitone_shift.unsqueeze(0).expand(f0.size(0))
        
        # Log scale semitone shift: 1 semitone = 1/12 in log2 scale
        shift_factor = semitone_shift.unsqueeze(1) * (1.0 / 12.0)  # (B, 1)
        
        f0_shifted = f0 + shift_factor
        
        f0_shifted = f0_shifted * vuv + f0 * (1 - vuv)
        
        return f0_shifted

class LoRALayer(nn.Module):
    """Lightweight LoRA layer"""
    def __init__(self, in_features, out_features, rank=16, alpha=16):
        super().__init__()
        self.rank = rank
        self.alpha = alpha
        
        self.lora_A = nn.Parameter(torch.randn(rank, in_features) * (1.0 / math.sqrt(rank)))
        self.lora_B = nn.Parameter(torch.zeros(out_features, rank))
        
        self.scaling = alpha / rank
        
    def forward(self, x):
        lora_weight = self.lora_B @ self.lora_A * self.scaling
        return F.linear(x, lora_weight)

class ReferenceEncoder(nn.Module):
    """Reference Audio Encoder for Speaker Characteristics"""
    def __init__(self, d_model=768, hidden_dim=256):
        super().__init__()
        
        # Temporal pooling and encoding
        self.temporal_pool = nn.AdaptiveAvgPool1d(1)
        
        # Speaker characteristic encoder
        self.encoder = nn.Sequential(
            nn.Linear(d_model, hidden_dim),
            nn.SiLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, d_model)
        )
        
        self.layer_norm = nn.LayerNorm(d_model)
        
        # Initialize weights
        for module in self.encoder:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)
    
    def forward(self, reference_features):
        """
        Extract speaker characteristics from reference audio
        Args:
            reference_features: (B, T, d_model) - mHuBERT features from reference audio
        Returns:
            (B, d_model) - Speaker embedding
        """
        # Temporal pooling: (B, T, d_model) -> (B, d_model)
        pooled = reference_features.mean(dim=1)  # Global average pooling
        
        # Encode speaker characteristics
        speaker_emb = self.encoder(pooled)
        speaker_emb = self.layer_norm(speaker_emb)
        
        return speaker_emb

class ASVLeakageMetric(nn.Module):
    """
    ASV-based leakage metric to measure speaker identity leakage
    Uses a pretrained speaker verification model to compute leakage score
    """
    def __init__(self):
        super().__init__()
        # This would typically load a pretrained ASV model
        # For now, we'll use a simple placeholder
        self.enabled = False  # Set to True when ASV model is available
        
    def compute_leakage(self, source_audio, converted_audio, reference_audio):
        """
        Compute speaker leakage score
        Args:
            source_audio: Original source audio
            converted_audio: Generated audio  
            reference_audio: Target reference audio
        Returns:
            leakage_score: Higher means more source speaker leakage
        """
        if not self.enabled:
            return torch.tensor(0.0, device=source_audio.device)
            
        # TODO: Implement with actual ASV model
        # logprob_src = asv_model(source_audio, converted_audio)
        # logprob_tgt = asv_model(reference_audio, converted_audio)  
        # leakage_score = torch.clamp_min(logprob_src - logprob_tgt, 0)
        
        return torch.tensor(0.0, device=source_audio.device)

class VoiceConversionModel(nn.Module):
    def __init__(self, 
                 hubert_model_name="utter-project/mHuBERT-147",
                 d_model=768,
                 d_content=256,
                 d_speaker=256,
                 ssm_layers=3,
                 flow_steps=20,
                 waveform_length=16384,
                 use_retrieval=False,
                 use_f0_conditioning=True,
                 orthogonality_weight=0.2,
                 grl_schedule="cosine"):
        super().__init__()
        
        self.use_f0_conditioning = use_f0_conditioning
        self.orthogonality_weight = orthogonality_weight
        self.grl_schedule = grl_schedule
        self.training_epoch = 0
        
        # Dual-Head mHuBERT for Content-Speaker separation
        self.dual_hubert = DualHeadmHuBERT(
            hubert_model_name=hubert_model_name,
            d_content=d_content,
            d_speaker=d_speaker
        )
            
        # SSM Encoder for content processing (adapts to d_content)
        self.ssm_encoder = S6SSMEncoder(d_model=d_content, n_layers=ssm_layers)
        
        # DCAE-Vocoder Pipeline
        self.dcae_vocoder = MusicDCAE(
            source_sample_rate=44100,
            dcae_checkpoint_path=os.path.join(os.path.dirname(__file__), 'dcae_vocoder', 'dcae'),
            vocoder_checkpoint_path=os.path.join(os.path.dirname(__file__), 'dcae_vocoder', 'vocoder')
        )
        
        # Freeze DCAE-Vocoder (pretrained)
        for param in self.dcae_vocoder.parameters():
            param.requires_grad = False
        
        # MEL Spectrogram dimensions (128 mel bins)
        mel_dim = 128
        
        # Rectified Flow for MEL generation
        condition_dim = d_content + d_speaker  # Content + Speaker features
        if use_f0_conditioning:
            condition_dim += d_content  # F0 embedding (use d_content for consistency)
            
        self.rectified_flow = RectifiedFlow(
            dim=mel_dim,  # Generate MEL spectrogram instead of raw audio
            condition_dim=condition_dim,
            steps=flow_steps,
            hidden_dim=256  # smaller for MEL
        )
        
        # Speaker feature aggregation (replaces ReferenceEncoder)
        self.speaker_aggregator = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),  # Pool over time dimension
            nn.Flatten(),
            nn.Linear(d_speaker, d_speaker),
            nn.ReLU(),
            nn.Linear(d_speaker, d_speaker),
            nn.LayerNorm(d_speaker)
        )
        
        # F0 related modules (adapted for d_content)
        if use_f0_conditioning:
            self.f0_embedding = F0Embedding(d_content)
            
        # F0/VUV prediction heads (from content features)
        self.f0_proj = nn.Sequential(
            nn.Linear(d_content, 128),
            nn.SiLU(),
            nn.Linear(128, 1)
        )
        
        self.vuv_proj = nn.Sequential(
            nn.Linear(d_content, 128),
            nn.SiLU(),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )
        
        # Remove retrieval module - not needed for reference-based conversion
        self.use_retrieval = use_retrieval
        
        # ASV-based leakage metric
        self.asv_metric = ASVLeakageMetric()
        
        # Optimization flags
        self.enable_compile = True  # torch.compile 활성화
        self._is_compiled = False
        
    def compile_model(self):
        """PyTorch 2.0 컴파일 최적화"""
        if self.enable_compile and not self._is_compiled:
            try:
                # 핵심 모듈들 컴파일
                self.ssm_encoder = torch.compile(self.ssm_encoder, mode='max-autotune')
                self.rectified_flow = torch.compile(self.rectified_flow, mode='max-autotune')
                if self.use_f0_conditioning:
                    self.f0_embedding = torch.compile(self.f0_embedding, mode='max-autotune')
                
                self._is_compiled = True
                print("Model compiled for optimization")
            except Exception as e:
                print(f"Warning: Compilation failed: {e}")
                
    def freeze_base_model(self):
        """Freeze base model for fine-tuning"""
        for param in self.ssm_encoder.parameters():
            param.requires_grad = False
        for param in self.rectified_flow.parameters():
            param.requires_grad = False
        print("Base model frozen for fine-tuning")
    
    def get_trainable_parameters(self):
        """Return only trainable parameters"""
        trainable_params = []
        
        # Dual-head parameters (always trainable)
        trainable_params.extend(self.dual_hubert.content_head.parameters())
        trainable_params.extend(self.dual_hubert.speaker_head.parameters())
        
        # Other modules
        trainable_params.extend(self.speaker_aggregator.parameters())
        trainable_params.extend(self.f0_proj.parameters())
        trainable_params.extend(self.vuv_proj.parameters())
        
        if self.use_f0_conditioning:
            trainable_params.extend(self.f0_embedding.parameters())
            
        return trainable_params
    
    def set_training_epoch(self, epoch):
        """Update training epoch for GRL scheduling"""
        self.training_epoch = epoch
        
        # Update GRL lambda based on schedule
        if self.grl_schedule == "cosine":
            if epoch < 5:
                # Warm-up: 0 -> 0.5
                lambd = 0.5 * epoch / 5
            elif epoch < 20:
                # Main training: 0.5 -> 1.0
                progress = (epoch - 5) / 15
                lambd = 0.5 + 0.5 * (1 - math.cos(math.pi * progress)) / 2
            else:
                # Fine-tuning: keep at 1.0
                lambd = 1.0
        else:
            # Linear schedule
            lambd = min(1.0, epoch / 20)
            
        self.dual_hubert.set_grl_lambda(lambd)
    
    def unfreeze_top_layers(self, num_layers=4):
        """Unfreeze top layers of mHuBERT backbone"""
        self.dual_hubert.unfreeze_top_layers(num_layers)
    
    def _convert_to_mono(self, waveform):
        """Convert stereo to mono"""
        if waveform.dim() == 3:
            return waveform.mean(dim=1)
        return waveform
    
    def _interpolate_f0_to_content(self, f0, content_length):
        """Interpolate F0 to match content length"""
        if f0.size(1) == content_length:
            return f0
        return F.interpolate(
            f0.unsqueeze(1), 
            size=content_length, 
            mode='linear', 
            align_corners=False
        ).squeeze(1)
    
    @torch.cuda.amp.autocast()
    def forward(self, 
                source_waveform, 
                reference_waveform,
                target_waveform=None, 
                f0_target=None, 
                vuv_target=None, 
                semitone_shift=0.0,
                training=True,
                inference_method='fast_rectified',
                num_steps=8):
        """
        Optimized forward pass with AMP
        """
        if source_waveform.dim() == 3:
            source_mono = source_waveform.mean(dim=1)
        else:
            source_mono = source_waveform
            
        # Extract content features using dual-head mHuBERT
        content_features = self.dual_hubert(source_mono, branch="content")  # (B, T, d_content)
        
        # Process content through SSM encoder
        if training and hasattr(self, '_is_finetuning') and self._is_finetuning:
            with torch.no_grad():
                encoded_content = self.ssm_encoder(content_features)
        else:
            encoded_content = self.ssm_encoder(content_features)
        
        # Reference audio processing
        if reference_waveform.dim() == 3:
            reference_mono = reference_waveform.mean(dim=1)
        else:
            reference_mono = reference_waveform
            
        # Extract speaker features using dual-head mHuBERT
        speaker_features = self.dual_hubert(reference_mono, branch="speaker")  # (B, T, d_speaker)
        
        # Aggregate speaker features over time
        speaker_emb = self.speaker_aggregator(speaker_features.transpose(1, 2)).squeeze(-1)  # (B, d_speaker)
        
        # Expand speaker embedding to match content sequence length
        speaker_emb_expanded = speaker_emb.unsqueeze(1).expand(-1, encoded_content.size(1), -1)  # (B, T, d_speaker)
        
        # Combine content and speaker features
        combined_features = torch.cat([encoded_content, speaker_emb_expanded], dim=-1)  # (B, T, d_content + d_speaker)
        
        # F0 conditional generation
        if self.use_f0_conditioning and f0_target is not None and vuv_target is not None:
            f0_resized = self._interpolate_f0_to_content(f0_target, encoded_content.size(1))
            vuv_resized = self._interpolate_f0_to_content(vuv_target, encoded_content.size(1))
            
            # F0 embedding with semitone shift
            f0_emb = self.f0_embedding(f0_resized, vuv_resized, semitone_shift)
            
            combined_features = torch.cat([combined_features, f0_emb], dim=-1)
        
        # Pool condition for flow matching
        condition_pooled = combined_features.mean(dim=1)
        
        if training and target_waveform is not None:
            # Training mode - convert target audio to MEL spectrogram
            target_mono = self._convert_to_mono(target_waveform)
            
            # Extract MEL spectrogram from target audio using DCAE vocoder
            with torch.no_grad():
                target_mel = self.dcae_vocoder.vocoder.encode(target_mono.unsqueeze(1))  # Add channel dim
            
            if hasattr(self, '_is_finetuning') and self._is_finetuning:
                with torch.no_grad():
                    flow_loss = self.rectified_flow.compute_loss(target_mel, condition_pooled)
                flow_loss = flow_loss.detach()
            else:
                flow_loss = self.rectified_flow.compute_loss(target_mel, condition_pooled)
            
            results = {'flow_loss': flow_loss}
            
            # Orthogonality loss - prevent content-speaker leakage
            if self.training:
                # Extract both content and speaker features from same audio for orthogonality
                source_speaker_features = self.dual_hubert(source_mono, branch="speaker")
                orth_loss = orthogonality_loss(content_features, source_speaker_features)
                results['orthogonality_loss'] = orth_loss
            
            # Auxiliary losses
            if f0_target is not None:
                f0_pred = self.f0_proj(encoded_content).squeeze(-1)
                f0_pred_resized = self._interpolate_f0_to_content(f0_pred, f0_target.size(1))
                f0_loss = F.mse_loss(f0_pred_resized, f0_target)
                results['f0_loss'] = f0_loss
                
            if vuv_target is not None:
                vuv_pred = self.vuv_proj(encoded_content).squeeze(-1)
                vuv_pred_resized = self._interpolate_f0_to_content(vuv_pred, vuv_target.size(1))
                vuv_loss = F.binary_cross_entropy(vuv_pred_resized, vuv_target.float())
                results['vuv_loss'] = vuv_loss
            
            # Total loss computation
            total_loss = torch.tensor(0.0, device=flow_loss.device, dtype=flow_loss.dtype)
            if not (hasattr(self, '_is_finetuning') and self._is_finetuning):
                total_loss += flow_loss
            if 'orthogonality_loss' in results:
                total_loss += self.orthogonality_weight * results['orthogonality_loss']
            if 'f0_loss' in results:
                total_loss += 0.1 * results['f0_loss']
            if 'vuv_loss' in results:
                total_loss += 0.1 * results['vuv_loss']
                
            results['total_loss'] = total_loss
            return results
            
        else:
            # Inference mode - Generate MEL then convert to audio
            # Determine target length based on source audio
            source_length = source_mono.size(-1)
            target_mel_length = source_length // 512  # hop_length = 512
            
            # Generate MEL spectrogram using Rectified Flow
            generated_mel = self.rectified_flow.sample(
                condition=condition_pooled,
                target_length=target_mel_length,
                num_steps=num_steps,
                method=inference_method
            )
            
            # Convert MEL to audio using DCAE-Vocoder
            with torch.no_grad():
                # generated_mel: (B, T, 128) -> need to transpose for vocoder
                mel_transposed = generated_mel.transpose(1, 2)  # (B, 128, T)
                converted_waveform = self.dcae_vocoder.vocoder.decode(mel_transposed).squeeze(1)
            
            f0_pred = self.f0_proj(encoded_content).squeeze(-1)
            vuv_pred = self.vuv_proj(encoded_content).squeeze(-1)
            
            return {
                'converted_waveform': converted_waveform,
                'f0_pred': f0_pred,
                'vuv_pred': vuv_pred,
                'generated_mel': generated_mel
            }

class RetrievalModule(nn.Module):
    """Optimized retrieval module"""
    def __init__(self, feature_dim=768, k=5):
        super().__init__()
        self.feature_dim = feature_dim
        self.k = k
        
        self.enhance_net = nn.Sequential(
            nn.Linear(feature_dim * 2, feature_dim),
            nn.SiLU(),
            nn.Dropout(0.05),
            nn.Linear(feature_dim, feature_dim)
        )
        
        self.register_buffer('training_features', torch.empty(0, feature_dim))
        self.register_buffer('speaker_ids', torch.empty(0, dtype=torch.long))
        
        self._cache = {}
        self._cache_size = 100
        
    def add_training_features(self, features, speaker_ids):
        """Add training features"""
        features = features.detach().cpu()
        speaker_ids = speaker_ids.detach().cpu()
        
        self.training_features = torch.cat([self.training_features, features], dim=0)
        self.speaker_ids = torch.cat([self.speaker_ids, speaker_ids], dim=0)
        
        self._cache.clear()
    
    @torch.cuda.amp.autocast()
    def enhance(self, content_features, target_speaker_id):
        """Enhance content features"""
        if self.training_features.size(0) == 0:
            return content_features
        
        B = content_features.size(0)
        device = content_features.device
        enhanced_features = []
        
        for i in range(B):
            query = content_features[i:i+1]
            spk_id = target_speaker_id[i].item()
            
            cache_key = f"{spk_id}_{hash(query.cpu().numpy().tobytes())}"
            if cache_key in self._cache:
                enhanced_features.append(self._cache[cache_key])
                continue
            
            same_speaker_mask = (self.speaker_ids == spk_id)
            if same_speaker_mask.sum() > 0:
                candidate_features = self.training_features[same_speaker_mask].to(device)
                
                similarities = F.cosine_similarity(
                    query.unsqueeze(1),
                    candidate_features.unsqueeze(0),
                    dim=-1
                ).squeeze(0)
                
                k = min(self.k, similarities.size(0))
                _, top_indices = similarities.topk(k)
                retrieved_features = candidate_features[top_indices].mean(dim=0, keepdim=True)
                
                combined = torch.cat([query, retrieved_features], dim=-1)
                enhanced = self.enhance_net(combined)
                
                if len(self._cache) < self._cache_size:
                    self._cache[cache_key] = enhanced
                
                enhanced_features.append(enhanced)
            else:
                enhanced_features.append(query)
        
        return torch.cat(enhanced_features, dim=0)