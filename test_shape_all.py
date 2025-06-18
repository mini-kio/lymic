#!/usr/bin/env python3
"""
전체 shape 오류 테스트: SSM, FlowMatching, VoiceConversionModel
"""
import torch

try:
    from ssm import S6SSMEncoder
except ImportError as e:
    print(f"❌ SSM import 실패: {e}")
    S6SSMEncoder = None

try:
    from flow_matching import FlowMatching
except ImportError as e:
    print(f"❌ FlowMatching import 실패: {e}")
    FlowMatching = None

try:
    from model import VoiceConversionModel
except ImportError as e:
    print(f"❌ VoiceConversionModel import 실패: {e}")
    VoiceConversionModel = None


def test_ssm_encoder():
    if S6SSMEncoder is None:
        print("⚠️ S6SSMEncoder 없음, 테스트 스킵")
        return

    print("\n[SSM] S6SSMEncoder shape test")
    B, T, D = 2, 50, 768
    x = torch.randn(B, T, D)
    model = S6SSMEncoder(d_model=D, n_layers=3)
    y = model(x)
    print(f"  입력: {x.shape} -> 출력: {y.shape}")
    assert y.shape == (B, T, D)
    print("  ✅ SSM shape OK")


def test_flow_matching():
    if FlowMatching is None:
        print("⚠️ FlowMatching 없음, 테스트 스킵")
        return

    print("\n[FlowMatching] shape test")
    B, waveform_length, cond_dim = 2, 16384, 768
    x1 = torch.randn(B, waveform_length)
    condition = torch.randn(B, cond_dim)
    model = FlowMatching(dim=waveform_length, condition_dim=cond_dim, steps=8)
    
    try:
        # loss
        loss = model.compute_loss(x1, condition)
        print(f"  Loss: {loss.item():.4f}")
    except Exception as e:
        print(f"❌ FlowMatching loss 계산 실패: {e}")
        return

    try:
        # sample
        out = model.sample(condition, num_steps=4, method='fast_inverse')
        print(f"  샘플 shape: {out.shape}")
        assert out.shape == (B, waveform_length)
    except Exception as e:
        print(f"❌ FlowMatching 샘플링 실패: {e}")
        return

    print("  ✅ FlowMatching shape OK")


def test_voice_conversion_model():
    if VoiceConversionModel is None:
        print("⚠️ VoiceConversionModel 없음, 테스트 스킵")
        return

    print("\n[VoiceConversionModel] shape test")
    B, L, n_speakers = 2, 16384, 8
    source_waveform = torch.randn(B, L)
    target_speaker_id = torch.randint(0, n_speakers, (B,))
    model = VoiceConversionModel(d_model=768, n_speakers=n_speakers, waveform_length=L, use_retrieval=False)
    
    try:
        with torch.no_grad():
            result = model(source_waveform, target_speaker_id, target_waveform=source_waveform, training=True)
        print(f"  결과 keys: {list(result.keys())}")
        print(f"  flow_loss: {result['flow_loss'].item():.4f}")
    except Exception as e:
        print(f"❌ VoiceConversionModel 실행 실패: {e}")
        return

    print("  ✅ VoiceConversionModel shape OK")


if __name__ == "__main__":
    print("🚀 전체 shape 테스트 시작")
    test_ssm_encoder()
    test_flow_matching()
    test_voice_conversion_model()
