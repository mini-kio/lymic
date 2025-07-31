import torch
import torch.nn as nn
from transformers import HubertModel
from ssm import S6SSMEncoder
from flow_matching import RectifiedFlow
import torch.nn.functional as F
import math

class F0Embedding(nn.Module):
    """F0 정보를 임베딩으로 변환하여 실제 생성에 활용"""
    def __init__(self, d_model=768):
        super().__init__()
        # F0와 VUV를 효율적으로 임베딩
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
        
        # 결합 네트워크
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
        # 🎵 세미톤 시프트 적용
        if semitone_shift != 0.0:
            f0_shifted = self.apply_semitone_shift(f0, vuv, semitone_shift)
        else:
            f0_shifted = f0
        
        f0_emb = self.f0_proj(f0_shifted.unsqueeze(-1))  # (B, T, d_model//2)
        vuv_emb = self.vuv_proj(vuv.unsqueeze(-1))  # (B, T, d_model//2)
        
        # F0는 voiced 영역에서만 활성화
        f0_emb = f0_emb * vuv.unsqueeze(-1)
        
        # 결합
        combined = torch.cat([f0_emb, vuv_emb], dim=-1)  # (B, T, d_model)
        return self.combine_proj(combined)
    
    def apply_semitone_shift(self, f0, vuv, semitone_shift):
        """
        🎵 세미톤 시프트 적용
        Args:
            f0: (B, T) F0 값 (Hz 단위 또는 log scale)
            vuv: (B, T) voiced/unvoiced 마스크
            semitone_shift: float 또는 (B,) 세미톤 수 (-12 ~ +12)
        Returns:
            (B, T) 시프트된 F0 값
        """
        if isinstance(semitone_shift, (int, float)):
            if semitone_shift == 0.0:
                return f0
            semitone_shift = torch.tensor(semitone_shift, device=f0.device, dtype=f0.dtype)
        
        # 배치별 서로 다른 시프트 지원
        if semitone_shift.dim() == 0:
            semitone_shift = semitone_shift.unsqueeze(0).expand(f0.size(0))
        
        # Log scale에서 세미톤 시프트 (더 정확함)
        # 1 semitone = log2(2^(1/12)) = 1/12 in log2 scale
        shift_factor = semitone_shift.unsqueeze(1) * (1.0 / 12.0)  # (B, 1)
        
        # F0가 이미 log scale이라고 가정하고 시프트
        f0_shifted = f0 + shift_factor
        
        # Voiced 영역에서만 시프트 적용
        f0_shifted = f0_shifted * vuv + f0 * (1 - vuv)
        
        return f0_shifted

class LoRALayer(nn.Module):
    """경량화된 LoRA 레이어"""
    def __init__(self, in_features, out_features, rank=16, alpha=16):
        super().__init__()
        self.rank = rank
        self.alpha = alpha
        
        # 효율적인 초기화
        self.lora_A = nn.Parameter(torch.randn(rank, in_features) * (1.0 / math.sqrt(rank)))
        self.lora_B = nn.Parameter(torch.zeros(out_features, rank))
        
        self.scaling = alpha / rank
        
    def forward(self, x):
        lora_weight = self.lora_B @ self.lora_A * self.scaling
        return F.linear(x, lora_weight)

class SpeakerAdapter(nn.Module):
    """최적화된 Speaker Adapter"""
    def __init__(self, d_model=768, adapter_dim=64):
        super().__init__()
        
        # 더 효율적인 구조
        self.adapter = nn.Sequential(
            nn.Linear(d_model, adapter_dim),
            nn.SiLU(),
            nn.Linear(adapter_dim, d_model)
        )
        
        self.layer_norm = nn.LayerNorm(d_model)
        
        # 작은 가중치로 초기화
        for module in self.adapter:
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, std=0.01)
                nn.init.zeros_(module.bias)
    
    def forward(self, x):
        return self.layer_norm(x + self.adapter(x))

class VoiceConversionModel(nn.Module):
    def __init__(self, 
                 hubert_model_name="utter-project/mHuBERT-147",  # 🌍 다국어 HuBERT-147
                 d_model=768,
                 ssm_layers=3,
                 flow_steps=20,  # 🔥 Rectified Flow는 더 적은 단계로도 고품질
                 n_speakers=256,
                 waveform_length=16384,
                 use_retrieval=True,
                 lora_rank=16,
                 adapter_dim=64,
                 use_f0_conditioning=True):  # 🎵 F0 조건부 생성
        super().__init__()
        
        self.use_f0_conditioning = use_f0_conditioning
        
        # 🔒 mHuBERT-147 (FROZEN) - 다국어 지원
        self.hubert = HubertModel.from_pretrained(hubert_model_name)
        for param in self.hubert.parameters():
            param.requires_grad = False
        
        print(f"🌍 Loaded mHuBERT-147: Multilingual speech representation model")
        print(f"   Model: {hubert_model_name}")
        print(f"   Hidden size: {self.hubert.config.hidden_size}")
        print(f"   Languages: 147+ languages supported")
            
        # 🔒 SSM Encoder (FROZEN during fine-tuning)
        self.ssm_encoder = S6SSMEncoder(d_model=d_model, n_layers=ssm_layers)
        
        # 🔥 Rectified Flow (더 효율적)
        condition_dim = d_model
        if use_f0_conditioning:
            condition_dim += d_model  # F0 임베딩 추가
            
        self.rectified_flow = RectifiedFlow(
            dim=waveform_length,
            condition_dim=condition_dim,
            steps=flow_steps,
            hidden_dim=512  # 메모리 효율성을 위해 축소
        )
        
        # 🔥 TRAINABLE COMPONENTS
        self.speaker_embedding = nn.Embedding(n_speakers, d_model)
        self.speaker_adapter = SpeakerAdapter(d_model, adapter_dim)
        self.speaker_lora = LoRALayer(d_model, d_model, rank=lora_rank)
        
        # 🎵 F0 관련 모듈
        if use_f0_conditioning:
            self.f0_embedding = F0Embedding(d_model)
            
        # F0/VUV 예측 헤드 (더 효율적)
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
        
        # 🔍 Retrieval module
        self.use_retrieval = use_retrieval
        if use_retrieval:
            self.retrieval_module = RetrievalModule(d_model)
        
        # 🚀 최적화 플래그
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
                print("🚀 Model compiled for optimization")
            except Exception as e:
                print(f"⚠️ Compilation failed: {e}")
                
    def freeze_base_model(self):
        """Fine-tuning을 위한 기본 모델 고정"""
        for param in self.ssm_encoder.parameters():
            param.requires_grad = False
        for param in self.rectified_flow.parameters():
            param.requires_grad = False
        print("🔒 Base model frozen for fine-tuning")
    
    def get_trainable_parameters(self):
        """훈련 가능한 파라미터만 반환"""
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
        """스테레오를 모노로 변환"""
        if waveform.dim() == 3:
            return waveform.mean(dim=1)
        return waveform
    
    def _interpolate_f0_to_content(self, f0, content_length):
        """F0 길이를 content 길이에 맞춤"""
        if f0.size(1) == content_length:
            return f0
        return F.interpolate(
            f0.unsqueeze(1), 
            size=content_length, 
            mode='linear', 
            align_corners=False
        ).squeeze(1)
    
    @torch.cuda.amp.autocast()  # 🔥 AMP 자동 적용
    def forward(self, 
                source_waveform, 
                target_speaker_id, 
                target_waveform=None, 
                f0_target=None, 
                vuv_target=None, 
                semitone_shift=0.0,  # 🎵 세미톤 시프트 추가
                training=True,
                inference_method='fast_rectified',
                num_steps=8):
        """
        🔥 최적화된 forward pass with AMP
        """
        # 스테레오 -> 모노 변환
        if source_waveform.dim() == 3:
            source_mono = source_waveform.mean(dim=1)
        else:
            source_mono = source_waveform
            
        # 🔒 HuBERT 추출 (FP32 유지)
        with torch.cuda.amp.autocast(enabled=False):
            with torch.no_grad():
                hubert_output = self.hubert(source_mono.float())
                content_repr = hubert_output.last_hidden_state
        
        # 🔒 SSM 인코딩
        if training and hasattr(self, '_is_finetuning') and self._is_finetuning:
            with torch.no_grad():
                encoded_content = self.ssm_encoder(content_repr)
        else:
            encoded_content = self.ssm_encoder(content_repr)
        
        # 🔥 Speaker 처리
        speaker_emb = self.speaker_embedding(target_speaker_id)
        speaker_emb = speaker_emb.unsqueeze(1).expand(-1, encoded_content.size(1), -1)
        
        # LoRA 적용
        speaker_emb_lora = self.speaker_lora(speaker_emb)
        
        # 결합
        condition = encoded_content + speaker_emb + speaker_emb_lora
        condition = self.speaker_adapter(condition)
        
        # 🎵 F0 조건부 생성
        if self.use_f0_conditioning and f0_target is not None and vuv_target is not None:
            # F0 길이를 content 길이에 맞춤
            f0_resized = self._interpolate_f0_to_content(f0_target, encoded_content.size(1))
            vuv_resized = self._interpolate_f0_to_content(vuv_target, encoded_content.size(1))
            
            # F0 임베딩 (세미톤 시프트 포함)
            f0_emb = self.f0_embedding(f0_resized, vuv_resized, semitone_shift)
            
            # 조건에 F0 정보 추가
            condition = torch.cat([condition, f0_emb], dim=-1)
        
        # 🔍 Retrieval 강화
        if self.use_retrieval and hasattr(self, 'retrieval_module'):
            condition_pooled = condition.mean(dim=1)
            condition_pooled = self.retrieval_module.enhance(condition_pooled, target_speaker_id)
        else:
            condition_pooled = condition.mean(dim=1)
        
        if training and target_waveform is not None:
            # 🔥 훈련 모드
            target_mono = self._convert_to_mono(target_waveform)
            
            # Rectified Flow 손실
            if hasattr(self, '_is_finetuning') and self._is_finetuning:
                with torch.no_grad():
                    flow_loss = self.rectified_flow.compute_loss(target_mono, condition_pooled)
                flow_loss = flow_loss.detach()
            else:
                flow_loss = self.rectified_flow.compute_loss(target_mono, condition_pooled)
            
            results = {'flow_loss': flow_loss}
            
            # 🎵 보조 손실들
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
            
            # 총 손실
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
            # 🚀 추론 모드 - Rectified Flow 샘플링
            converted_waveform = self.rectified_flow.sample(
                condition=condition_pooled,
                num_steps=num_steps,
                method=inference_method
            )
            
            # F0/VUV 예측
            f0_pred = self.f0_proj(encoded_content).squeeze(-1)
            vuv_pred = self.vuv_proj(encoded_content).squeeze(-1)
            
            return {
                'converted_waveform': converted_waveform,
                'f0_pred': f0_pred,
                'vuv_pred': vuv_pred
            }

class RetrievalModule(nn.Module):
    """최적화된 검색 모듈"""
    def __init__(self, feature_dim=768, k=5):
        super().__init__()
        self.feature_dim = feature_dim
        self.k = k
        
        # 더 효율적인 강화 네트워크
        self.enhance_net = nn.Sequential(
            nn.Linear(feature_dim * 2, feature_dim),
            nn.SiLU(),
            nn.Dropout(0.05),  # 낮은 드롭아웃
            nn.Linear(feature_dim, feature_dim)
        )
        
        # 훈련 특성 저장소
        self.register_buffer('training_features', torch.empty(0, feature_dim))
        self.register_buffer('speaker_ids', torch.empty(0, dtype=torch.long))
        
        # 캐시 최적화
        self._cache = {}
        self._cache_size = 100
        
    def add_training_features(self, features, speaker_ids):
        """훈련 특성 추가"""
        features = features.detach().cpu()
        speaker_ids = speaker_ids.detach().cpu()
        
        self.training_features = torch.cat([self.training_features, features], dim=0)
        self.speaker_ids = torch.cat([self.speaker_ids, speaker_ids], dim=0)
        
        # 캐시 초기화
        self._cache.clear()
    
    @torch.cuda.amp.autocast()
    def enhance(self, content_features, target_speaker_id):
        """컨텐츠 특성 강화"""
        if self.training_features.size(0) == 0:
            return content_features
        
        B = content_features.size(0)
        device = content_features.device
        enhanced_features = []
        
        for i in range(B):
            query = content_features[i:i+1]
            spk_id = target_speaker_id[i].item()
            
            # 캐시 확인
            cache_key = f"{spk_id}_{hash(query.cpu().numpy().tobytes())}"
            if cache_key in self._cache:
                enhanced_features.append(self._cache[cache_key])
                continue
            
            # 동일 화자 특성 찾기
            same_speaker_mask = (self.speaker_ids == spk_id)
            if same_speaker_mask.sum() > 0:
                candidate_features = self.training_features[same_speaker_mask].to(device)
                
                # 코사인 유사도 계산
                similarities = F.cosine_similarity(
                    query.unsqueeze(1),
                    candidate_features.unsqueeze(0),
                    dim=-1
                ).squeeze(0)
                
                # Top-k 선택
                k = min(self.k, similarities.size(0))
                _, top_indices = similarities.topk(k)
                retrieved_features = candidate_features[top_indices].mean(dim=0, keepdim=True)
                
                # 강화
                combined = torch.cat([query, retrieved_features], dim=-1)
                enhanced = self.enhance_net(combined)
                
                # 캐시 저장 (크기 제한)
                if len(self._cache) < self._cache_size:
                    self._cache[cache_key] = enhanced
                
                enhanced_features.append(enhanced)
            else:
                enhanced_features.append(query)
        
        return torch.cat(enhanced_features, dim=0)