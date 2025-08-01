#!/usr/bin/env python3
"""
Standalone DCAE + Vocoder Audio Processing Script
Usage: python process_audio.py 1.mp3
"""

import argparse
import os
import sys
import time
import torch
import torchaudio
from music_dcae_pipeline import MusicDCAE


def main():
    parser = argparse.ArgumentParser(description="Process audio file with DCAE and Vocoder")
    parser.add_argument("input_file", help="Input audio file (e.g., 1.mp3)")
    parser.add_argument("--output", "-o", help="Output file name (default: input_reconstructed.wav)")
    parser.add_argument("--dcae-path", help="Path to DCAE checkpoint")
    parser.add_argument("--vocoder-path", help="Path to vocoder checkpoint")
    parser.add_argument("--device", default="auto", help="Device to use (cpu/cuda/auto)")
    parser.add_argument("--chunk-size", type=int, default=256, help="Chunk size for decoding (default: 256)")
    parser.add_argument("--overlap-size", type=int, default=8, help="Overlap size for chunking (default: 8)")
    parser.add_argument("--no-chunking", action="store_true", help="Disable chunked decoding")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.input_file):
        print(f"Error: Input file '{args.input_file}' not found")
        sys.exit(1)
    
    # Determine device
    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device
    print(f"Using device: {device}")
    
    # Set output filename
    if args.output:
        output_file = args.output
    else:
        base_name = os.path.splitext(args.input_file)[0]
        output_file = f"{base_name}_reconstructed.wav"
    
    try:
        print("Initializing DCAE model...")
        model = MusicDCAE(
            source_sample_rate=44100,  # Set to 44100 to avoid resampling
            dcae_checkpoint_path=args.dcae_path,
            vocoder_checkpoint_path=args.vocoder_path
        )
        model = model.to(device)
        model.eval()
        
        print(f"Loading audio: {args.input_file}")
        audio, sr = model.load_audio(args.input_file)
        print(f"Audio shape: {audio.shape}, Sample rate: {sr}")
        
        # Move to device and add batch dimension
        audio = audio.to(device)
        audio_lengths = torch.tensor([audio.shape[1]]).to(device)
        audios = audio.unsqueeze(0)  # Add batch dimension
        
        print("Encoding audio to latents...")
        start_time = time.time()
        latents, latent_lengths = model.encode(audios, audio_lengths, sr)
        encoding_time = time.time() - start_time
        print(f"Encoding completed in {encoding_time:.2f} seconds")
        print(f"Latent shape: {latents.shape}")
        
        print("Decoding latents back to audio...")
        start_time = time.time()
        use_chunking = not args.no_chunking
        # Decode at 44100Hz first, then resample to original sr if needed
        output_sr, pred_wavs = model.decode(
            latents, 
            None,  # Don't pass audio_lengths to avoid truncation
            44100,  # Keep vocoder's native sample rate
            use_chunking=use_chunking,
            chunk_size=args.chunk_size,
            overlap_size=args.overlap_size
        )
        
        # Resample to original sample rate if different
        if sr != 44100:
            resampler = torchaudio.transforms.Resample(44100, sr)
            pred_wavs = [resampler(wav.float()) for wav in pred_wavs]
            output_sr = sr
        decoding_time = time.time() - start_time
        print(f"Decoding completed in {decoding_time:.2f} seconds")
        
        print(f"Saving reconstructed audio: {output_file}")
        torchaudio.save(output_file, pred_wavs[0], output_sr)
        
        total_time = encoding_time + decoding_time
        
        print("Processing completed successfully!")
        print(f"Input:  {args.input_file}")
        print(f"Output: {output_file}")
        print(f"Original duration: {audio.shape[1] / sr:.2f}s")
        print(f"Reconstructed duration: {pred_wavs[0].shape[1] / output_sr:.2f}s")
        print("\n--- Timing Summary ---")
        print(f"Encoding time:  {encoding_time:.2f} seconds")
        print(f"Decoding time:  {decoding_time:.2f} seconds")
        print(f"Total time:     {total_time:.2f} seconds")
        print(f"Real-time factor: {(audio.shape[1] / sr) / total_time:.2f}x")
        
    except Exception as e:
        print(f"Error during processing: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()