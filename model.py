import torch
import torch.nn as nn
from transformers import HubertModel
from ssm import S6SSMEncoder
from flow_matching import FlowMatching
import torch.nn.functional as F

class LoRALayer(nn.Module):
    """LoRA (Low-Rank Adaptation) layer"""
    def __init__(self, in_features, out_features, rank=16, alpha=16):
        super().__init__()
        self.rank = rank
        self.alpha = alpha
        
        # Low-rank matrices
        self.lora_A = nn.Parameter(torch.randn(rank, in_features) * 0.01)
        self.lora_B = nn.Parameter(torch.zeros(out_features, rank))
        
        # Scaling factor
        self.scaling = alpha / rank
        
    def forward(self, x):
        # LoRA: W + (B @ A) * scaling
        lora_weight = self.lora_B @ self.lora_A * self.scaling
        return F.linear(x, lora_weight)

class SpeakerAdapter(nn.Module):
    """Speaker-specific adapter module"""
    def __init__(self, d_model=768, adapter_dim=64):
        super().__init__()
        
        # Down-projection
        self.down_proj = nn.Linear(d_model, adapter_dim)
        
        # Non-linearity
        self.activation = nn.ReLU()
        
        # Up-projection
        self.up_proj = nn.Linear(adapter_dim, d_model)
        
        # Layer norm
        self.layer_norm = nn.LayerNorm(d_model)
        
        # Initialize small weights
        nn.init.normal_(self.down_proj.weight, std=0.01)
        nn.init.normal_(self.up_proj.weight, std=0.01)
        nn.init.zeros_(self.down_proj.bias)
        nn.init.zeros_(self.up_proj.bias)
    
    def forward(self, x):
        residual = x
        
        # Adapter forward pass
        x = self.down_proj(x)
        x = self.activation(x)
        x = self.up_proj(x)
        
        # Residual connection + layer norm
        return self.layer_norm(residual + x)

class VoiceConversionModel(nn.Module):
    def __init__(self, 
                 hubert_model_name="ZhenYe234/hubert_base_general_audio",
                 d_model=768,
                 ssm_layers=3,  # 🔥 6 → 3 for speed
                 flow_steps=50,  # 🔥 100 → 50 for faster inference
                 n_speakers=256,
                 waveform_length=16384,
                 use_retrieval=True,  # 🔥 Enable retrieval by default
                 lora_rank=16,
                 adapter_dim=64):
        super().__init__()
        
        # 🔒 HuBERT pretrained model (FROZEN)
        self.hubert = HubertModel.from_pretrained(hubert_model_name)
        for param in self.hubert.parameters():
            param.requires_grad = False
            
        # 🔒 SSM Encoder (FROZEN during fine-tuning)
        self.ssm_encoder = S6SSMEncoder(
            d_model=d_model,
            n_layers=ssm_layers  # Reduced from 6 to 3
        )
        
        # 🔒 Flow Matching (FROZEN during fine-tuning)
        self.flow_matching = FlowMatching(
            dim=waveform_length,
            condition_dim=d_model,
            steps=flow_steps
        )
        
        # 🔥 TRAINABLE COMPONENTS (for fine-tuning)
        
        # Speaker embedding (trainable)
        self.speaker_embedding = nn.Embedding(n_speakers, d_model)
        
        # Speaker adapters (trainable)
        self.speaker_adapter = SpeakerAdapter(d_model, adapter_dim)
        
        # LoRA for speaker conditioning (trainable)
        self.speaker_lora = LoRALayer(d_model, d_model, rank=lora_rank)
        
        # Optional: F0 and V/UV prediction heads (trainable)
        self.f0_proj = nn.Sequential(
            nn.Linear(d_model, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )
        
        self.vuv_proj = nn.Sequential(
            nn.Linear(d_model, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )
        
        # 🔍 Retrieval module (trainable)
        self.use_retrieval = use_retrieval
        if use_retrieval:
            self.retrieval_module = RetrievalModule(d_model)
    
    def freeze_base_model(self):
        """Freeze base components for efficient fine-tuning"""
        # HuBERT already frozen
        
        # Freeze SSM encoder
        for param in self.ssm_encoder.parameters():
            param.requires_grad = False
            
        # Freeze Flow Matching
        for param in self.flow_matching.parameters():
            param.requires_grad = False
            
        print("🔒 Base model components frozen for fine-tuning")
    
    def unfreeze_all(self):
        """Unfreeze all components for full training"""
        for param in self.parameters():
            param.requires_grad = True
            
        # Keep HuBERT frozen
        for param in self.hubert.parameters():
            param.requires_grad = False
            
        print("🔓 All trainable components unfrozen")
    
    def get_trainable_parameters(self):
        """Get only trainable parameters for fine-tuning"""
        trainable_params = []
        
        # Speaker embedding
        trainable_params.extend(self.speaker_embedding.parameters())
        
        # Speaker adapter
        trainable_params.extend(self.speaker_adapter.parameters())
        
        # Speaker LoRA
        trainable_params.extend(self.speaker_lora.parameters())
        
        # F0/VUV heads
        trainable_params.extend(self.f0_proj.parameters())
        trainable_params.extend(self.vuv_proj.parameters())
        
        # Retrieval module
        if self.use_retrieval:
            trainable_params.extend(self.retrieval_module.parameters())
            
        return trainable_params
    
    def forward(self, 
                source_waveform, 
                target_speaker_id, 
                target_waveform=None, 
                f0_target=None, 
                vuv_target=None, 
                training=True,
                inference_method='fast_inverse',  # 🔥 Default to fast inference
                num_steps=10):  # 🔥 Fast steps
        """
        Args:
            source_waveform: (B, L) or (B, C, L) source audio
            target_speaker_id: (B,) target speaker IDs
            target_waveform: (B, target_length) target waveform for training
            inference_method: 'fast_inverse', 'ode', 'euler'
            num_steps: number of inference steps
        """
        # Convert stereo to mono for HuBERT if needed
        if source_waveform.dim() == 3:
            source_mono = source_waveform.mean(dim=1)
        else:
            source_mono = source_waveform
        
        # 🔒 HuBERT feature extraction (frozen)
        with torch.no_grad():
            hubert_output = self.hubert(source_mono)
            content_repr = hubert_output.last_hidden_state
            print(f"[Model] HuBERT out: {content_repr.shape}")
        
        # 🔒 SSM encoding (frozen during fine-tuning)
        if training and hasattr(self, '_is_finetuning') and self._is_finetuning:
            with torch.no_grad():
                encoded_content = self.ssm_encoder(content_repr)
        else:
            encoded_content = self.ssm_encoder(content_repr)
        print(f"[Model] SSM out: {encoded_content.shape}")
        
        # 🔥 Speaker processing (trainable)
        speaker_emb = self.speaker_embedding(target_speaker_id)  # (B, D)
        speaker_emb = speaker_emb.unsqueeze(1).expand(-1, encoded_content.size(1), -1)
        
        # Apply LoRA to speaker embedding
        speaker_emb_lora = self.speaker_lora(speaker_emb)
        
        # Combine content + speaker
        condition = encoded_content + speaker_emb + speaker_emb_lora
        
        # Apply speaker adapter
        condition = self.speaker_adapter(condition)
        
        # 🔍 Retrieval enhancement (if enabled)
        if self.use_retrieval and hasattr(self, 'retrieval_module'):
            condition_pooled = condition.mean(dim=1)  # (B, D)
            condition_pooled = self.retrieval_module.enhance(condition_pooled, target_speaker_id)
        else:
            condition_pooled = condition.mean(dim=1)
        
        if training and target_waveform is not None:
            # Training mode
            if hasattr(self, '_is_finetuning') and self._is_finetuning:
                # Fine-tuning: freeze flow matching
                with torch.no_grad():
                    flow_loss = self.flow_matching.compute_loss(
                        x1=target_waveform,
                        condition=condition_pooled
                    )
                # Don't backprop through flow loss during fine-tuning
                flow_loss = flow_loss.detach()
            else:
                # Full training
                flow_loss = self.flow_matching.compute_loss(
                    x1=target_waveform,
                    condition=condition_pooled
                )
            
            results = {'flow_loss': flow_loss}
            
            # Auxiliary losses (always trainable)
            if f0_target is not None:
                f0_pred = self.f0_proj(encoded_content).squeeze(-1)
                f0_loss = nn.MSELoss()(f0_pred, f0_target)
                results['f0_loss'] = f0_loss
                results['f0_pred'] = f0_pred
            
            if vuv_target is not None:
                vuv_pred = self.vuv_proj(encoded_content).squeeze(-1)
                vuv_loss = nn.BCELoss()(vuv_pred, vuv_target.float())
                results['vuv_loss'] = vuv_loss
                results['vuv_pred'] = vuv_pred
            
            # Total loss
            total_loss = torch.tensor(0.0, device=flow_loss.device)
            if not (hasattr(self, '_is_finetuning') and self._is_finetuning):
                total_loss += flow_loss
            if 'f0_loss' in results:
                total_loss += 0.1 * results['f0_loss']
            if 'vuv_loss' in results:
                total_loss += 0.1 * results['vuv_loss']
                
            results['total_loss'] = total_loss
            return results
            
        else:
            # Inference mode - use fast methods
            converted_waveform = self.flow_matching.sample(
                condition=condition_pooled,
                num_steps=num_steps,
                method=inference_method  # fast_inverse, ode, euler
            )
            
            # Optional predictions
            f0_pred = self.f0_proj(encoded_content).squeeze(-1)
            vuv_pred = self.vuv_proj(encoded_content).squeeze(-1)
            
            return {
                'converted_waveform': converted_waveform,
                'f0_pred': f0_pred,
                'vuv_pred': vuv_pred
            }

class RetrievalModule(nn.Module):
    """Enhanced retrieval module with FAISS indexing"""
    def __init__(self, feature_dim=768, k=5, use_faiss=True):
        super().__init__()
        self.feature_dim = feature_dim
        self.k = k
        self.use_faiss = use_faiss
        
        # Feature enhancement network
        self.enhance_net = nn.Sequential(
            nn.Linear(feature_dim * 2, feature_dim),
            nn.ReLU(),
            nn.Linear(feature_dim, feature_dim),
            nn.Dropout(0.1)
        )
        
        # Storage for training features
        self.register_buffer('training_features', torch.empty(0, feature_dim))
        self.register_buffer('speaker_ids', torch.empty(0, dtype=torch.long))
        
        # FAISS index (will be built dynamically)
        self.faiss_index = None
        self._index_built = False
    
    def add_training_features(self, features, speaker_ids):
        """Add features from training data"""
        features = features.detach().cpu()
        speaker_ids = speaker_ids.detach().cpu()
        
        self.training_features = torch.cat([self.training_features, features], dim=0)
        self.speaker_ids = torch.cat([self.speaker_ids, speaker_ids], dim=0)
        
        # Rebuild index
        self._index_built = False
    
    def build_faiss_index(self):
        """Build FAISS index for fast retrieval"""
        if self.training_features.size(0) == 0:
            return
            
        try:
            import faiss
            
            # Convert to numpy
            features_np = self.training_features.cpu().numpy().astype('float32')
            
            # Normalize features for cosine similarity
            faiss.normalize_L2(features_np)
            
            # Build index
            self.faiss_index = faiss.IndexFlatIP(self.feature_dim)  # Inner product (cosine sim)
            self.faiss_index.add(features_np)
            
            self._index_built = True
            print(f"🔍 FAISS index built with {features_np.shape[0]} features")
            
        except ImportError:
            print("⚠️ FAISS not available, using PyTorch similarity search")
            self.use_faiss = False
    
    def enhance(self, content_features, target_speaker_id):
        """
        Enhance content features using retrieval
        Args:
            content_features: (B, feature_dim)
            target_speaker_id: (B,) target speaker IDs
        """
        if self.training_features.size(0) == 0:
            return content_features
        
        B = content_features.size(0)
        device = content_features.device
        enhanced_features = []
        
        # Build FAISS index if not built
        if self.use_faiss and not self._index_built:
            self.build_faiss_index()
        
        for i in range(B):
            query = content_features[i:i+1]  # (1, feature_dim)
            spk_id = target_speaker_id[i].item()
            
            # Find features from same speaker
            same_speaker_mask = (self.speaker_ids == spk_id)
            if same_speaker_mask.sum() > 0:
                if self.use_faiss and self._index_built:
                    # Use FAISS for fast search
                    query_np = query.cpu().numpy().astype('float32')
                    faiss.normalize_L2(query_np)
                    
                    # Search all features first
                    similarities, indices = self.faiss_index.search(query_np, min(self.k * 10, self.training_features.size(0)))
                    
                    # Filter by speaker
                    valid_indices = []
                    for idx in indices[0]:
                        if self.speaker_ids[idx] == spk_id:
                            valid_indices.append(idx)
                        if len(valid_indices) >= self.k:
                            break
                    
                    if valid_indices:
                        retrieved_features = self.training_features[valid_indices].to(device)
                        retrieved_features = retrieved_features.mean(dim=0, keepdim=True)
                    else:
                        retrieved_features = query
                        
                else:
                    # Fallback to PyTorch similarity
                    candidate_features = self.training_features[same_speaker_mask].to(device)
                    
                    # Compute cosine similarity
                    similarities = F.cosine_similarity(
                        query.unsqueeze(1),
                        candidate_features.unsqueeze(0),
                        dim=-1
                    ).squeeze(0)
                    
                    # Get top-k
                    k = min(self.k, similarities.size(0))
                    _, top_indices = similarities.topk(k)
                    retrieved_features = candidate_features[top_indices].mean(dim=0, keepdim=True)
                
                # Enhance with retrieved features
                combined = torch.cat([query, retrieved_features], dim=-1)
                enhanced = self.enhance_net(combined)
                enhanced_features.append(enhanced)
            else:
                # No features from target speaker
                enhanced_features.append(query)
        
        return torch.cat(enhanced_features, dim=0)
        
    def forward(self, 
                source_waveform, 
                target_speaker_id, 
                target_waveform=None, 
                f0_target=None, 
                vuv_target=None, 
                training=True):
        """
        Args:
            source_waveform: (B, L) or (B, C, L) source audio
            target_speaker_id: (B,) target speaker IDs
            target_waveform: (B, target_length) target waveform for training
            f0_target: (B, T) target F0 (optional, for auxiliary loss)
            vuv_target: (B, T) target V/UV (optional, for auxiliary loss)
        """
        # Convert stereo to mono for HuBERT if needed
        if source_waveform.dim() == 3:  # (B, C, L) - stereo
            source_mono = source_waveform.mean(dim=1)  # (B, L)
        else:  # (B, L) - already mono
            source_mono = source_waveform
        
        # HuBERT feature extraction
        with torch.no_grad():
            hubert_output = self.hubert(source_mono)
            content_repr = hubert_output.last_hidden_state  # (B, T, D)

        # SSM encoding
        encoded_content = self.ssm_encoder(content_repr)  # (B, T, D)
        
        # Speaker embedding
        speaker_emb = self.speaker_embedding(target_speaker_id)  # (B, D)
        
        # Expand speaker embedding to sequence length
        speaker_emb = speaker_emb.unsqueeze(1).expand(-1, encoded_content.size(1), -1)  # (B, T, D)
        
        # Combine content + speaker for conditioning
        condition = encoded_content + speaker_emb  # (B, T, D)
        
        # Optional: Retrieval enhancement
        if self.use_retrieval and hasattr(self, 'retrieval_module'):
            condition = self.retrieval_module.enhance(condition, target_speaker_id)
        
        # Pool condition for waveform generation (or use attention)
        condition_pooled = condition.mean(dim=1)  # (B, D) - simple pooling
        
        if training and target_waveform is not None:
            # Training: Flow matching loss for FULL WAVEFORM
            flow_loss = self.flow_matching.compute_loss(
                x1=target_waveform,  # (B, waveform_length) - TARGET WAVEFORM!
                condition=condition_pooled
            )
            
            results = {'flow_loss': flow_loss}
            
            # Optional auxiliary losses (F0, V/UV)
            if f0_target is not None:
                f0_pred = self.f0_proj(encoded_content).squeeze(-1)  # (B, T)
                f0_loss = nn.MSELoss()(f0_pred, f0_target)
                results['f0_loss'] = f0_loss
                results['f0_pred'] = f0_pred
            
            if vuv_target is not None:
                vuv_pred = self.vuv_proj(encoded_content).squeeze(-1)  # (B, T)
                vuv_loss = nn.BCELoss()(vuv_pred, vuv_target.float())
                results['vuv_loss'] = vuv_loss
                results['vuv_pred'] = vuv_pred
            
            # Total loss
            total_loss = flow_loss
            if 'f0_loss' in results:
                total_loss += 0.1 * results['f0_loss']  # Auxiliary loss weight
            if 'vuv_loss' in results:
                total_loss += 0.1 * results['vuv_loss']  # Auxiliary loss weight
                
            results['total_loss'] = total_loss
            return results
            
        else:
            # Inference: Generate target waveform using Flow Matching
            converted_waveform = self.flow_matching.sample(
                condition=condition_pooled,
                num_steps=20,  # Fast inference
                method='ode'  # Use ODE solver
            )
            
            # Optional: predict F0 and V/UV for analysis
            f0_pred = self.f0_proj(encoded_content).squeeze(-1)  # (B, T)
            vuv_pred = self.vuv_proj(encoded_content).squeeze(-1)  # (B, T)
            
            return {
                'converted_waveform': converted_waveform,  # Main output!
                'f0_pred': f0_pred,
                'vuv_pred': vuv_pred
            }
    
    def extract_features(self, waveform):
        """Extract HuBERT features for analysis"""
        # Convert stereo to mono for HuBERT if needed
        if waveform.dim() == 3:  # (B, C, L) - stereo
            waveform_mono = waveform.mean(dim=1)  # (B, L)
        else:  # (B, L) - already mono
            waveform_mono = waveform
            
        with torch.no_grad():
            hubert_output = self.hubert(waveform_mono)
            return hubert_output.last_hidden_state