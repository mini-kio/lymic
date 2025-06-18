#!/usr/bin/env python3
"""
🚀 Voice Conversion Inference Example
Ultra-fast inference with optimized settings
"""

import torch
import torchaudio
import argparse
from pathlib import Path
import time

from model import VoiceConversionModel

def load_model(checkpoint_path, device='cuda'):
    """Load trained model from checkpoint"""
    print(f"📦 Loading model from {checkpoint_path}")
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    config = checkpoint.get('config', {})
    
    # Initialize model with saved config
    model = VoiceConversionModel(
        d_model=config.get('d_model', 768),
        ssm_layers=config.get('ssm_layers', 3),
        flow_steps=config.get('flow_steps', 50),
        n_speakers=config.get('n_speakers', 256),
        waveform_length=config.get('waveform_length', 16384),
        use_retrieval=config.get('use_retrieval', True),
        lora_rank=config.get('lora_rank', 16),
        adapter_dim=config.get('adapter_dim', 64)
    ).to(device)
    
    # Load weights
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    print("✅ Model loaded successfully")
    return model, config

def preprocess_audio(audio_path, target_length=16384, sample_rate=44100):
    """Preprocess input audio"""
    print(f"🎵 Loading audio from {audio_path}")
    
    # Load audio
    waveform, sr = torchaudio.load(audio_path)
    
    # Resample if needed
    if sr != sample_rate:
        print(f"🔄 Resampling from {sr}Hz to {sample_rate}Hz")
        resampler = torchaudio.transforms.Resample(sr, sample_rate)
        waveform = resampler(waveform)
    
    # Convert to stereo if mono
    if waveform.size(0) == 1:
        waveform = waveform.repeat(2, 1)
    elif waveform.size(0) > 2:
        waveform = waveform[:2]  # Take first 2 channels
    
    # Fix length
    current_length = waveform.size(1)
    if current_length > target_length:
        # Random crop
        start = torch.randint(0, current_length - target_length + 1, (1,)).item()
        waveform = waveform[:, start:start + target_length]
    elif current_length < target_length:
        # Pad with zeros
        pad_length = target_length - current_length
        waveform = torch.cat([waveform, torch.zeros(2, pad_length)], dim=1)
    
    print(f"✅ Audio preprocessed: {waveform.shape}")
    return waveform

def convert_voice(model, source_audio, target_speaker_id, 
                  method='fast_inverse', num_steps=6, device='cuda'):
    """Perform voice conversion"""
    print(f"🔄 Converting voice with method '{method}', steps={num_steps}")
    
    # Prepare inputs
    source_audio = source_audio.unsqueeze(0).to(device)  # Add batch dimension
    target_speaker_id = torch.tensor([target_speaker_id], device=device)
    
    # Measure inference time
    start_time = time.time()
    
    with torch.no_grad():
        result = model(
            source_waveform=source_audio,
            target_speaker_id=target_speaker_id,
            training=False,
            inference_method=method,
            num_steps=num_steps
        )
    
    inference_time = time.time() - start_time
    converted_audio = result['converted_waveform']
    
    print(f"⚡ Inference completed in {inference_time:.3f}s")
    print(f"📊 Output shape: {converted_audio.shape}")
    
    return converted_audio.cpu(), result

def main():
    parser = argparse.ArgumentParser(description='Voice Conversion Inference')
    parser.add_argument('--model', '-m', required=True, 
                       help='Path to model checkpoint')
    parser.add_argument('--input', '-i', required=True,
                       help='Path to input audio file')
    parser.add_argument('--output', '-o', required=True,
                       help='Path to output audio file')
    parser.add_argument('--speaker', '-s', type=int, required=True,
                       help='Target speaker ID (check speaker_mapping.json or dataset info)')
    parser.add_argument('--list-speakers', action='store_true',
                       help='List available speakers and their IDs')
    parser.add_argument('--method', default='fast_inverse',
                       choices=['fast_inverse', 'ode_adaptive', 'ode', 'euler'],
                       help='Inference method')
    parser.add_argument('--steps', type=int, default=6,
                       help='Number of inference steps')
    parser.add_argument('--device', default='cuda',
                       help='Device to use (cuda/cpu)')
    parser.add_argument('--length', type=int, default=16384,
                       help='Target waveform length')
    
    args = parser.parse_args()
    
    # List speakers option
    if args.list_speakers:
        model_path = Path(args.model)
        if model_path.exists():
            device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
            model, config = load_model(model_path, device)
            
            print("📋 Available Speakers:")
            # Try to get speaker info from config
            if 'speaker_info' in config:
                speaker_to_id = config['speaker_info']['speaker_to_id']
                for speaker, id in speaker_to_id.items():
                    print(f"   {id}: {speaker}")
            else:
                print("   Speaker mapping not found in checkpoint")
                print("   Check your dataset for speaker names")
        else:
            print(f"❌ Model file not found: {model_path}")
        return
    
    # Check paths
    model_path = Path(args.model)
    input_path = Path(args.input)
    output_path = Path(args.output)
    
    if not model_path.exists():
        print(f"❌ Model file not found: {model_path}")
        return
    
    if not input_path.exists():
        print(f"❌ Input file not found: {input_path}")
        return
    
    # Create output directory
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    print("🚀 Starting Voice Conversion Inference")
    print(f"   Model: {model_path}")
    print(f"   Input: {input_path}")
    print(f"   Output: {output_path}")
    print(f"   Target Speaker: {args.speaker}")
    print(f"   Method: {args.method}")
    print(f"   Steps: {args.steps}")
    
    # Load model
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    model, config = load_model(model_path, device)
    
    # Preprocess audio
    source_audio = preprocess_audio(input_path, args.length)
    
    # Convert voice
    converted_audio, result = convert_voice(
        model, source_audio, args.speaker,
        method=args.method, num_steps=args.steps, device=device
    )
    
    # Save result
    print(f"💾 Saving converted audio to {output_path}")
    torchaudio.save(output_path, converted_audio, 44100)
    
    # Optional: save additional info
    if 'f0_pred' in result:
        f0_pred = result['f0_pred'].cpu()
        vuv_pred = result['vuv_pred'].cpu()
        print(f"📈 F0 range: {f0_pred.min():.2f} - {f0_pred.max():.2f}")
        print(f"📊 Voiced ratio: {vuv_pred.mean():.2%}")
    
    print("🎉 Voice conversion completed!")

def batch_convert(model_path, input_dir, output_dir, target_speaker_id, 
                  method='fast_inverse', num_steps=6):
    """Batch conversion example"""
    print("🔄 Batch conversion mode")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model, config = load_model(model_path, device)
    
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Find all audio files
    audio_files = list(input_dir.glob('*.wav')) + list(input_dir.glob('*.mp3'))
    
    print(f"📂 Found {len(audio_files)} audio files")
    
    total_time = 0
    for i, audio_file in enumerate(audio_files, 1):
        print(f"\n[{i}/{len(audio_files)}] Processing {audio_file.name}")
        
        try:
            # Preprocess
            source_audio = preprocess_audio(audio_file)
            
            # Convert
            start_time = time.time()
            converted_audio, _ = convert_voice(
                model, source_audio, target_speaker_id,
                method=method, num_steps=num_steps, device=device
            )
            conversion_time = time.time() - start_time
            total_time += conversion_time
            
            # Save
            output_file = output_dir / f"converted_{audio_file.stem}.wav"
            torchaudio.save(output_file, converted_audio, 44100)
            
            print(f"✅ Saved to {output_file}")
            
        except Exception as e:
            print(f"❌ Error processing {audio_file.name}: {e}")
    
    avg_time = total_time / len(audio_files)
    print(f"\n🎉 Batch conversion completed!")
    print(f"📊 Average time per file: {avg_time:.3f}s")
    print(f"📊 Total time: {total_time:.1f}s")

if __name__ == '__main__':
    main()

# Example usage:
"""
🚀 Voice Conversion Inference Examples

# 1. 화자 목록 확인
python inference.py -m checkpoint.pt --list-speakers

# 2. 단일 파일 변환 (RVC-style!)
python inference.py -m checkpoint.pt -i my_voice.wav -o converted.wav -s 2

# 3. 초고속 변환 (실시간 수준)
python inference.py -m checkpoint.pt -i input.wav -o output.wav -s 1 \\
                   --method fast_inverse --steps 4

# 4. 고품질 변환
python inference.py -m checkpoint.pt -i input.wav -o output.wav -s 3 \\
                   --method ode_adaptive --steps 20

# 5. 배치 변환 (프로그래밍 방식)
python -c "
from inference import batch_convert
batch_convert(
    model_path='checkpoint.pt',
    input_dir='./test_voices/',
    output_dir='./converted/', 
    target_speaker_id=2,  # 'bob'로 변환
    method='fast_inverse',
    num_steps=6
)
"

📋 데이터셋에서 화자 ID 확인하는 법:
- dataset_root/alice/ → speaker_id = 0
- dataset_root/bob/ → speaker_id = 1  
- dataset_root/charlie/ → speaker_id = 2
- ...

💡 알파벳 순서로 자동 할당됩니다!
"""