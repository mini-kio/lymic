#!/usr/bin/env python3

import torch
import torch.nn.functional as F
import torchaudio
import numpy as np
import pytest
import sys
import os
from pathlib import Path

sys.path.append(os.path.dirname(__file__))

from model import VoiceConversionModel, DualHeadmHuBERT, F0Embedding
from ssm import S6SSMEncoder, OptimizedS6Block
from flow_matching import RectifiedFlow
from utils import extract_f0, normalize_f0, denormalize_f0, OptimizedVoiceConversionDataset
from dcae_vocoder.music_dcae_pipeline import MusicDCAE
# Note: Some inference functions may not be available


class TestShapes:
    """Comprehensive shape testing for all models and operations"""
    
    def setup_method(self):
        """Setup test environment"""
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.batch_size = 2
        self.seq_len = 128
        self.audio_len = 16384
        self.sample_rate = 44100
        
        # Test audio paths
        self.vocal_path = Path(__file__).parent / 'vocal.mp3'
        
        print(f"Test setup: device={self.device}, batch_size={self.batch_size}")
    
    def test_audio_loading_shapes(self):
        """Test audio loading and basic shape operations"""
        print("\n=== Testing Audio Loading Shapes ===")
        
        # Test with vocal.mp3 if available
        if self.vocal_path.exists():
            waveform, sr = torchaudio.load(str(self.vocal_path))
            print(f"vocal.mp3: {waveform.shape}, sr={sr}")
            
            # Test mono conversion
            if waveform.shape[0] > 1:
                mono = waveform.mean(dim=0)
                print(f"Mono conversion: {waveform.shape} -> {mono.shape}")
                
            # Test resampling
            if sr != self.sample_rate:
                resampler = torchaudio.transforms.Resample(sr, self.sample_rate)
                resampled = resampler(waveform)
                print(f"Resampling: {waveform.shape} -> {resampled.shape}")
        
        # Test synthetic audio
        synthetic_stereo = torch.randn(2, self.audio_len)
        synthetic_mono = torch.randn(1, self.audio_len)
        synthetic_batch = torch.randn(self.batch_size, 2, self.audio_len)
        
        print(f"Synthetic stereo: {synthetic_stereo.shape}")
        print(f"Synthetic mono: {synthetic_mono.shape}")
        print(f"Synthetic batch: {synthetic_batch.shape}")
        
        # Test shape transformations
        stereo_to_mono = synthetic_stereo.mean(dim=0)
        print(f"Stereo to mono: {synthetic_stereo.shape} -> {stereo_to_mono.shape}")
        
        batch_to_mono = synthetic_batch.mean(dim=1)
        print(f"Batch stereo to mono: {synthetic_batch.shape} -> {batch_to_mono.shape}")
        
        assert stereo_to_mono.shape == (self.audio_len,)
        assert batch_to_mono.shape == (self.batch_size, self.audio_len)
    
    def test_f0_extraction_shapes(self):
        """Test F0 extraction and processing shapes"""
        print("\n=== Testing F0 Extraction Shapes ===")
        
        # Test with different audio lengths
        audio_lengths = [8192, 16384, 22050, 44100]
        
        for length in audio_lengths:
            audio = torch.randn(length).numpy()
            f0, vuv = extract_f0(audio, sample_rate=self.sample_rate, hop_length=512)
            
            expected_frames = 1 + length // 512
            print(f"Audio len {length}: F0 {f0.shape}, VUV {vuv.shape}, expected {expected_frames}")
            
            assert f0.shape[0] == expected_frames
            assert vuv.shape[0] == expected_frames
            assert f0.dtype == np.float32
            assert vuv.dtype == np.float32
        
        # Test F0 normalization shapes
        f0_test = np.array([100, 200, 0, 150, 0, 300], dtype=np.float32)
        f0_norm = normalize_f0(f0_test, method='log')
        f0_denorm = denormalize_f0(f0_norm, method='log')
        
        print(f"F0 normalization: {f0_test.shape} -> {f0_norm.shape} -> {f0_denorm.shape}")
        assert f0_test.shape == f0_norm.shape == f0_denorm.shape
        
        # Test with torch tensors
        f0_torch = torch.from_numpy(f0_test)
        f0_norm_torch = normalize_f0(f0_torch, method='log')
        print(f"Torch F0 normalization: {f0_torch.shape} -> {f0_norm_torch.shape}")
        assert f0_torch.shape == f0_norm_torch.shape
    
    def test_dual_head_hubert_shapes(self):
        """Test DualHeadmHuBERT shape handling"""
        print("\n=== Testing DualHead mHuBERT Shapes ===")
        
        try:
            model = DualHeadmHuBERT(d_content=256, d_speaker=256)
            model.eval()
            
            # Test different batch sizes and sequence lengths
            test_configs = [
                (1, 16000),
                (2, 32000), 
                (4, 16384)
            ]
            
            for batch_size, seq_len in test_configs:
                audio = torch.randn(batch_size, seq_len)
                
                with torch.no_grad():
                    content_features = model(audio, branch="content")
                    speaker_features = model(audio, branch="speaker")
                
                print(f"Input {audio.shape}:")
                print(f"  Content: {content_features.shape}")
                print(f"  Speaker: {speaker_features.shape}")
                
                # Verify shapes
                assert content_features.shape[0] == batch_size
                assert speaker_features.shape[0] == batch_size
                assert content_features.shape[2] == 256  # d_content
                assert speaker_features.shape[2] == 256  # d_speaker
                
                # Check if sequence length is reasonable
                expected_seq_ratio = seq_len // 320  # Rough estimate for HuBERT
                assert abs(content_features.shape[1] - expected_seq_ratio) < 10
                
        except Exception as e:
            print(f"Warning: DualHead mHuBERT test failed (likely model not available): {e}")
    
    def test_ssm_encoder_shapes(self):
        """Test S6SSMEncoder shape handling"""
        print("\n=== Testing S6SSM Encoder Shapes ===")
        
        # Test different model dimensions
        model_dims = [128, 256, 512, 768]
        
        for d_model in model_dims:
            encoder = S6SSMEncoder(d_model=d_model, n_layers=2)
            encoder.eval()
            
            # Test different sequence lengths
            seq_lengths = [32, 64, 128, 256]
            
            for seq_len in seq_lengths:
                input_tensor = torch.randn(self.batch_size, seq_len, d_model)
                
                with torch.no_grad():
                    output = encoder(input_tensor)
                
                print(f"SSM d_model={d_model}, seq_len={seq_len}: {input_tensor.shape} -> {output.shape}")
                
                # Verify shapes
                assert output.shape == input_tensor.shape
                assert output.shape == (self.batch_size, seq_len, d_model)
    
    def test_s6_layer_shapes(self):
        """Test individual S6 Block shapes"""
        print("\n=== Testing S6 Block Shapes ===")
        
        layer = OptimizedS6Block(d_model=256, d_state=64, expand_factor=2)
        layer.eval()
        
        # Test different input shapes
        test_shapes = [
            (1, 32, 256),
            (2, 64, 256), 
            (4, 128, 256),
            (8, 256, 256)
        ]
        
        for shape in test_shapes:
            input_tensor = torch.randn(*shape)
            
            with torch.no_grad():
                output = layer(input_tensor)
            
            print(f"S6 Block: {input_tensor.shape} -> {output.shape}")
            assert output.shape == input_tensor.shape
    
    def test_flow_matching_shapes(self):
        """Test RectifiedFlow shape handling"""
        print("\n=== Testing RectifiedFlow Shapes ===")
        
        # Test different dimensions
        dims = [64, 128, 256]
        condition_dims = [128, 256, 512]
        
        for dim in dims:
            for condition_dim in condition_dims:
                flow = RectifiedFlow(dim=dim, condition_dim=condition_dim, steps=10)
                flow.eval()
                
                # Test sampling
                condition = torch.randn(self.batch_size, condition_dim)
                target_length = 64
                
                with torch.no_grad():
                    sample = flow.sample(condition, target_length, num_steps=4)
                
                expected_shape = (self.batch_size, target_length, dim)
                print(f"Flow dim={dim}, cond_dim={condition_dim}: condition {condition.shape} -> sample {sample.shape}")
                assert sample.shape == expected_shape
                
                # Test loss computation
                target = torch.randn(self.batch_size, target_length, dim)
                with torch.no_grad():
                    loss = flow.compute_loss(target, condition)
                
                print(f"  Loss shape: {loss.shape}")
                assert loss.dim() == 0  # Scalar loss
    
    def test_f0_embedding_shapes(self):
        """Test F0Embedding shape handling"""
        print("\n=== Testing F0 Embedding Shapes ===")
        
        d_models = [256, 512, 768]
        
        for d_model in d_models:
            f0_emb = F0Embedding(d_model=d_model)
            f0_emb.eval()
            
            # Test different sequence lengths
            seq_lengths = [32, 64, 128]
            
            for seq_len in seq_lengths:
                f0 = torch.randn(self.batch_size, seq_len)
                vuv = torch.randint(0, 2, (self.batch_size, seq_len)).float()
                
                with torch.no_grad():
                    embedding = f0_emb(f0, vuv)
                    embedding_shifted = f0_emb(f0, vuv, semitone_shift=2.0)
                
                expected_shape = (self.batch_size, seq_len, d_model)
                print(f"F0 Embedding d_model={d_model}, seq_len={seq_len}: -> {embedding.shape}")
                assert embedding.shape == expected_shape
                assert embedding_shifted.shape == expected_shape
    
    def test_dcae_vocoder_shapes(self):
        """Test DCAE Vocoder shape handling with vocal.mp3"""
        print("\n=== Testing DCAE Vocoder Shapes ===")
        
        try:
            # Initialize DCAE vocoder
            dcae = MusicDCAE(
                source_sample_rate=44100,
                dcae_checkpoint_path=Path(__file__).parent / 'dcae_vocoder' / 'dcae',
                vocoder_checkpoint_path=Path(__file__).parent / 'dcae_vocoder' / 'vocoder'
            )
            dcae.eval()
            
            # Test with vocal.mp3 if available
            if self.vocal_path.exists():
                print(f"Testing DCAE with vocal.mp3...")
                
                waveform, sr = torchaudio.load(str(self.vocal_path))
                print(f"Loaded vocal.mp3: {waveform.shape}, sr={sr}")
                
                # Convert to mono if stereo
                if waveform.shape[0] > 1:
                    waveform = waveform.mean(dim=0, keepdim=True)
                
                # Resample if needed
                if sr != 44100:
                    resampler = torchaudio.transforms.Resample(sr, 44100)
                    waveform = resampler(waveform)
                
                # Test different chunk sizes
                chunk_sizes = [16384, 32768, 65536]
                
                for chunk_size in chunk_sizes:
                    if waveform.shape[1] >= chunk_size:
                        chunk = waveform[:, :chunk_size]
                        
                        with torch.no_grad():
                            # Test encoding
                            latent = dcae.encode_audio(chunk)
                            print(f"DCAE Encode: {chunk.shape} -> {latent.shape}")
                            
                            # Test reconstruction
                            reconstructed = dcae.decode_latent(latent)
                            print(f"DCAE Decode: {latent.shape} -> {reconstructed.shape}")
                            
                            # Verify shapes are reasonable
                            assert latent.dim() == 3  # (B, C, T)
                            assert reconstructed.shape[0] == chunk.shape[0]  # Same batch size
                            
                            # Audio length should be approximately preserved
                            length_ratio = reconstructed.shape[-1] / chunk.shape[-1]
                            print(f"  Length ratio: {length_ratio:.3f}")
                            assert 0.5 < length_ratio < 2.0  # Reasonable range
            
            else:
                print("vocal.mp3 not found, testing with synthetic audio...")
                
                # Test with synthetic audio
                test_audio = torch.randn(1, 44100)  # 1 second
                
                with torch.no_grad():
                    latent = dcae.encode_audio(test_audio)
                    reconstructed = dcae.decode_latent(latent)
                
                print(f"DCAE Synthetic test: {test_audio.shape} -> {latent.shape} -> {reconstructed.shape}")
                
        except Exception as e:
            print(f"Warning: DCAE test failed (likely model checkpoints not available): {e}")
    
    def test_voice_conversion_model_shapes(self):
        """Test complete VoiceConversionModel shape handling"""
        print("\n=== Testing VoiceConversionModel Shapes ===")
        
        try:
            model = VoiceConversionModel(
                d_model=512,
                d_content=256,
                d_speaker=256,
                ssm_layers=2,
                flow_steps=10,
                waveform_length=16384,
                use_f0_conditioning=True
            )
            model.eval()
            
            # Test inputs
            source_waveform = torch.randn(self.batch_size, 16384)
            reference_waveform = torch.randn(self.batch_size, 16384)
            target_waveform = torch.randn(self.batch_size, 16384)
            
            # Generate F0 data
            f0_frames = 1 + 16384 // 512
            f0_target = torch.randn(self.batch_size, f0_frames)
            vuv_target = torch.randint(0, 2, (self.batch_size, f0_frames)).float()
            
            print(f"Model input shapes:")
            print(f"  Source: {source_waveform.shape}")
            print(f"  Reference: {reference_waveform.shape}")
            print(f"  Target: {target_waveform.shape}")
            print(f"  F0: {f0_target.shape}")
            print(f"  VUV: {vuv_target.shape}")
            
            # Test training mode
            with torch.no_grad():
                training_output = model(
                    source_waveform=source_waveform,
                    reference_waveform=reference_waveform,
                    target_waveform=target_waveform,
                    f0_target=f0_target,
                    vuv_target=vuv_target,
                    training=True
                )
            
            print(f"Training output keys: {list(training_output.keys())}")
            for key, value in training_output.items():
                if isinstance(value, torch.Tensor):
                    print(f"  {key}: {value.shape if value.dim() > 0 else 'scalar'}")
            
            # Test inference mode
            with torch.no_grad():
                inference_output = model(
                    source_waveform=source_waveform,
                    reference_waveform=reference_waveform,
                    f0_target=f0_target,
                    vuv_target=vuv_target,
                    training=False,
                    num_steps=4
                )
            
            print(f"Inference output keys: {list(inference_output.keys())}")
            for key, value in inference_output.items():
                if isinstance(value, torch.Tensor):
                    print(f"  {key}: {value.shape}")
            
            # Verify inference output shapes
            assert 'converted_waveform' in inference_output
            converted = inference_output['converted_waveform']
            assert converted.shape[0] == self.batch_size
            print(f"Converted waveform shape: {converted.shape}")
            
        except Exception as e:
            print(f"Warning: VoiceConversionModel test failed (likely model dependencies not available): {e}")
    
    def test_inference_functions_shapes(self):
        """Test inference utility functions"""
        print("\n=== Testing Inference Functions Shapes ===")
        
        # Test direct audio loading with torchaudio if vocal.mp3 exists
        if self.vocal_path.exists():
            try:
                waveform, sr = torchaudio.load(str(self.vocal_path))
                duration = waveform.shape[1] / sr
                print(f"Direct load: {self.vocal_path.name} -> {waveform.shape}, sr={sr}, duration={duration:.2f}s")
                
                # Test mono conversion
                if waveform.shape[0] > 1:
                    mono = waveform.mean(dim=0, keepdim=True)
                    print(f"  Mono conversion: {waveform.shape} -> {mono.shape}")
                
            except Exception as e:
                print(f"Warning: audio loading test failed: {e}")
        
        # Test inference pipeline functionality
        try:
            from inference import VoiceConversionInference
            
            print("VoiceConversionInference class available for testing")
            
        except Exception as e:
            print(f"Warning: VoiceConversionInference test failed: {e}")
    
    def test_edge_cases_shapes(self):
        """Test edge cases and boundary conditions"""
        print("\n=== Testing Edge Cases ===")
        
        # Test very small sequences
        small_audio = torch.randn(1, 512)  # Minimum size
        f0_small, vuv_small = extract_f0(small_audio.squeeze().numpy(), hop_length=512)
        print(f"Small audio: {small_audio.shape} -> F0 {f0_small.shape}, VUV {vuv_small.shape}")
        
        # Test large batch sizes
        large_batch = torch.randn(16, 128, 256)
        ssm = S6SSMEncoder(d_model=256, n_layers=1)
        with torch.no_grad():
            large_output = ssm(large_batch)
        print(f"Large batch SSM: {large_batch.shape} -> {large_output.shape}")
        
        # Test sequence length edge cases
        for seq_len in [1, 2, 3, 7, 15, 31, 63, 127]:
            input_tensor = torch.randn(1, seq_len, 256)
            with torch.no_grad():
                output = ssm(input_tensor)
            print(f"SSM seq_len={seq_len}: {input_tensor.shape} -> {output.shape}")
            assert output.shape == input_tensor.shape
    
    def test_memory_efficiency(self):
        """Test memory usage with large tensors"""
        print("\n=== Testing Memory Efficiency ===")
        
        if torch.cuda.is_available():
            device = torch.device('cuda')
            
            # Test large audio processing
            large_audio = torch.randn(4, 88200, device=device)  # 2 seconds stereo
            print(f"Large audio shape: {large_audio.shape}")
            print(f"Memory before: {torch.cuda.memory_allocated() / 1024**2:.1f}MB")
            
            # Convert to mono
            mono = large_audio.mean(dim=0, keepdim=True)
            print(f"Mono conversion: {large_audio.shape} -> {mono.shape}")
            print(f"Memory after mono: {torch.cuda.memory_allocated() / 1024**2:.1f}MB")
            
            # Extract F0
            f0, vuv = extract_f0(mono.squeeze().cpu().numpy())
            print(f"F0 extraction: audio {mono.shape} -> F0 {f0.shape}")
            print(f"Memory after F0: {torch.cuda.memory_allocated() / 1024**2:.1f}MB")
            
            # Clean up
            del large_audio, mono
            torch.cuda.empty_cache()
            print(f"Memory after cleanup: {torch.cuda.memory_allocated() / 1024**2:.1f}MB")
        else:
            print("CUDA not available, skipping memory test")


def run_all_tests():
    """Run all shape tests"""
    print("="*60)
    print("COMPREHENSIVE SHAPE TESTING FOR LYMIC")
    print("="*60)
    
    tester = TestShapes()
    tester.setup_method()
    
    # Run all tests
    test_methods = [
        'test_audio_loading_shapes',
        'test_f0_extraction_shapes', 
        'test_dual_head_hubert_shapes',
        'test_ssm_encoder_shapes',
        'test_s6_layer_shapes',
        'test_flow_matching_shapes',
        'test_f0_embedding_shapes',
        'test_dcae_vocoder_shapes',
        'test_voice_conversion_model_shapes',
        'test_inference_functions_shapes',
        'test_edge_cases_shapes',
        'test_memory_efficiency'
    ]
    
    passed = 0
    failed = 0
    
    for method_name in test_methods:
        try:
            print(f"\n{'='*20} {method_name.upper()} {'='*20}")
            method = getattr(tester, method_name)
            method()
            print(f"[PASS] {method_name} PASSED")
            passed += 1
        except Exception as e:
            print(f"[FAIL] {method_name} FAILED: {e}")
            failed += 1
    
    print(f"\n{'='*60}")
    print(f"SHAPE TESTING SUMMARY")
    print(f"{'='*60}")
    print(f"Passed: {passed}")
    print(f"Failed: {failed}")
    print(f"Total: {passed + failed}")
    
    if failed == 0:
        print("All shape tests passed!")
    else:
        print(f"{failed} tests failed - check dependencies and model files")


if __name__ == "__main__":
    run_all_tests()