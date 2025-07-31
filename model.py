import torch
import torch.nn as nn
from transformers import HubertModel
from ssm import S6SSMEncoder
from flow_matching import RectifiedFlow
import torch.nn.functional as F
import math

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

class SpeakerAdapter(nn.Module):
    """Optimized Speaker Adapter"""
    def __init__(self, d_model=768, adapter_dim=64):
        super().__init__()
        
        self.adapter = nn.Sequential(
            nn.Linear(d_model, adapter_dim),
            nn.SiLU(),
            nn.Linear(adapter_dim, d_model)
        )
        
        self.layer_norm = nn.LayerNorm(d_model)
        
        for module in self.adapter:
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, std=0.01)
                nn.init.zeros_(module.bias)
    
    def forward(self, x):
        return self.layer_norm(x + self.adapter(x))

class VoiceConversionModel(nn.Module):
    def __init__(self, 
                 hubert_model_name="utter-project/mHuBERT-147",
                 d_model=768,
                 ssm_layers=3,
                 flow_steps=20,
                 n_speakers=256,
                 waveform_length=16384,
                 use_retrieval=True,
                 lora_rank=16,
                 adapter_dim=64,
                 use_f0_conditioning=True):
        super().__init__()
        
        self.use_f0_conditioning = use_f0_conditioning
        
        # mHuBERT-147 (FROZEN) - multilingual support
        self.hubert = HubertModel.from_pretrained(hubert_model_name)
        for param in self.hubert.parameters():
            param.requires_grad = False
        
        print(f"Loaded mHuBERT-147: Multilingual speech representation model")
        print(f"   Model: {hubert_model_name}")
        print(f"   Hidden size: {self.hubert.config.hidden_size}")
        print(f"   Languages: 147+ languages supported")
            
        # SSM Encoder (FROZEN during fine-tuning)
        self.ssm_encoder = S6SSMEncoder(d_model=d_model, n_layers=ssm_layers)
        
        # Rectified Flow (more efficient)
        condition_dim = d_model
        if use_f0_conditioning:
            condition_dim += d_model  # F0 embedding
            
        self.rectified_flow = RectifiedFlow(
            dim=waveform_length,
            condition_dim=condition_dim,
            steps=flow_steps,
            hidden_dim=512  # reduced for memory efficiency
        )
        
        # TRAINABLE COMPONENTS
        self.speaker_embedding = nn.Embedding(n_speakers, d_model)
        self.speaker_adapter = SpeakerAdapter(d_model, adapter_dim)
        self.speaker_lora = LoRALayer(d_model, d_model, rank=lora_rank)
        
        # F0 related modules
        if use_f0_conditioning:
            self.f0_embedding = F0Embedding(d_model)
            
        # F0/VUV prediction heads
        self.f0_proj = nn.Sequential(
            nn.Linear(d_model, 128),
            nn.SiLU(),
            nn.Linear(128, 1)
        )
        
        self.vuv_proj = nn.Sequential(
            nn.Linear(d_model, 128),
            nn.SiLU(),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )
        
        # Retrieval module
        self.use_retrieval = use_retrieval
        if use_retrieval:
            self.retrieval_module = RetrievalModule(d_model)
        
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
        trainable_params.extend(self.speaker_embedding.parameters())
        trainable_params.extend(self.speaker_adapter.parameters())
        trainable_params.extend(self.speaker_lora.parameters())
        trainable_params.extend(self.f0_proj.parameters())
        trainable_params.extend(self.vuv_proj.parameters())
        
        if self.use_f0_conditioning:
            trainable_params.extend(self.f0_embedding.parameters())
        if self.use_retrieval:
            trainable_params.extend(self.retrieval_module.parameters())
            
        return trainable_params
    
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
                target_speaker_id, 
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
            
        # HuBERT extraction (keep FP32)
        with torch.cuda.amp.autocast(enabled=False):
            with torch.no_grad():
                hubert_output = self.hubert(source_mono.float())
                content_repr = hubert_output.last_hidden_state
        
        # SSM encoding
        if training and hasattr(self, '_is_finetuning') and self._is_finetuning:
            with torch.no_grad():
                encoded_content = self.ssm_encoder(content_repr)
        else:
            encoded_content = self.ssm_encoder(content_repr)
        
        # Speaker processing
        speaker_emb = self.speaker_embedding(target_speaker_id)
        speaker_emb = speaker_emb.unsqueeze(1).expand(-1, encoded_content.size(1), -1)
        
        speaker_emb_lora = self.speaker_lora(speaker_emb)
        
        condition = encoded_content + speaker_emb + speaker_emb_lora
        condition = self.speaker_adapter(condition)
        
        # F0 conditional generation
        if self.use_f0_conditioning and f0_target is not None and vuv_target is not None:
            f0_resized = self._interpolate_f0_to_content(f0_target, encoded_content.size(1))
            vuv_resized = self._interpolate_f0_to_content(vuv_target, encoded_content.size(1))
            
            # F0 embedding with semitone shift
            f0_emb = self.f0_embedding(f0_resized, vuv_resized, semitone_shift)
            
            condition = torch.cat([condition, f0_emb], dim=-1)
        
        # Retrieval enhancement
        if self.use_retrieval and hasattr(self, 'retrieval_module'):
            condition_pooled = condition.mean(dim=1)
            condition_pooled = self.retrieval_module.enhance(condition_pooled, target_speaker_id)
        else:
            condition_pooled = condition.mean(dim=1)
        
        if training and target_waveform is not None:
            # Training mode
            target_mono = self._convert_to_mono(target_waveform)
            
            if hasattr(self, '_is_finetuning') and self._is_finetuning:
                with torch.no_grad():
                    flow_loss = self.rectified_flow.compute_loss(target_mono, condition_pooled)
                flow_loss = flow_loss.detach()
            else:
                flow_loss = self.rectified_flow.compute_loss(target_mono, condition_pooled)
            
            results = {'flow_loss': flow_loss}
            
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
            
            total_loss = torch.tensor(0.0, device=flow_loss.device, dtype=flow_loss.dtype)
            if not (hasattr(self, '_is_finetuning') and self._is_finetuning):
                total_loss += flow_loss
            if 'f0_loss' in results:
                total_loss += 0.1 * results['f0_loss']
            if 'vuv_loss' in results:
                total_loss += 0.1 * results['vuv_loss']
                
            results['total_loss'] = total_loss
            return results
            
        else:
            # Inference mode - Rectified Flow sampling
            converted_waveform = self.rectified_flow.sample(
                condition=condition_pooled,
                num_steps=num_steps,
                method=inference_method
            )
            
            f0_pred = self.f0_proj(encoded_content).squeeze(-1)
            vuv_pred = self.vuv_proj(encoded_content).squeeze(-1)
            
            return {
                'converted_waveform': converted_waveform,
                'f0_pred': f0_pred,
                'vuv_pred': vuv_pred
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