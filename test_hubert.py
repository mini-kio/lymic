#!/usr/bin/env python3
"""
HuBERT 출력 차원 확인 스크립트
"""

import torch
from transformers import HubertModel, HubertConfig

def test_hubert_dimensions():
    """HuBERT의 입력/출력 차원 확인"""
    print("🔍 Testing HuBERT dimensions...")

    # 장치 설정
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"   🖥️  Using device: {device}")

    # 모델 로딩 (에러 처리 포함)
    try:
        hubert = HubertModel.from_pretrained("ZhenYe234/hubert_base_general_audio").to(device)
        hubert.eval()
        print("   ✅ HuBERT 모델 로딩 성공")
    except Exception as e:
        print(f"❌ 모델 로딩 실패: {e}")
        return

    # 테스트 입력 생성 (스테레오 → 모노)
    batch_size = 2
    channels = 2
    length = 16000  # 1초 기준

    stereo_input = torch.randn(batch_size, channels, length).to(device)
    mono_input = stereo_input.mean(dim=1)  # (batch_size, sequence_length)

    print(f"   📊 스테레오 입력 shape: {stereo_input.shape}")
    print(f"   📊 모노 입력 shape: {mono_input.shape}")

    # HuBERT forward
    with torch.no_grad():
        try:
            # HuggingFace 모델은 input_values 키워드 인자 사용
            hubert_output = hubert(input_values=mono_input)
            content_repr = hubert_output.last_hidden_state
        except Exception as e:
            print(f"❌ 모델 추론 실패: {e}")
            return

    # 출력 확인
    print(f"   📊 HuBERT 출력 shape: {content_repr.shape}")
    print(f"   📊 시퀀스 길이: {content_repr.size(1)}")
    print(f"   📊 특성 차원: {content_repr.size(2)}")

    return content_repr.shape

if __name__ == "__main__":
    test_hubert_dimensions()
