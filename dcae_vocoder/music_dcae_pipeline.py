"""
Standalone Music DCAE Pipeline for encoding and decoding audio
"""

import os
import torch
from diffusers import AutoencoderDC
import torchaudio
import torchvision.transforms as transforms
from diffusers.models.modeling_utils import ModelMixin
from diffusers.loaders import FromOriginalModelMixin
from diffusers.configuration_utils import ConfigMixin, register_to_config

from music_vocoder import ADaMoSHiFiGANV1


class MusicDCAE(ModelMixin, ConfigMixin, FromOriginalModelMixin):
    @register_to_config
    def __init__(
        self,
        source_sample_rate=None,
        dcae_checkpoint_path=None,
        vocoder_checkpoint_path=None,
    ):
        super(MusicDCAE, self).__init__()

        if dcae_checkpoint_path is None:
            dcae_checkpoint_path = "./dcae"
        if vocoder_checkpoint_path is None:
            vocoder_checkpoint_path = "./vocoder"

        self.dcae = AutoencoderDC.from_pretrained(dcae_checkpoint_path, local_files_only=True)
        self.vocoder = ADaMoSHiFiGANV1.from_pretrained(vocoder_checkpoint_path, local_files_only=True)

        if source_sample_rate is None:
            source_sample_rate = 44100  # Set to 44100 to avoid resampling

        self.source_sample_rate = source_sample_rate
        # Only create resampler if source rate is different from target rate
        if source_sample_rate != 44100:
            self.resampler = torchaudio.transforms.Resample(source_sample_rate, 44100)
        else:
            self.resampler = None

        self.transform = transforms.Compose(
            [
                transforms.Normalize(0.5, 0.5),
            ]
        )
        self.min_mel_value = -11.0
        self.max_mel_value = 3.0
        self.audio_chunk_size = int(round((1024 * 512 / 44100 * 48000)))
        self.mel_chunk_size = 1024
        self.time_dimention_multiple = 8
        self.latent_chunk_size = self.mel_chunk_size // self.time_dimention_multiple
        self.scale_factor = 0.1786
        self.shift_factor = -1.9091

    def load_audio(self, audio_path):
        audio, sr = torchaudio.load(audio_path)
        if audio.shape[0] == 1:
            audio = audio.repeat(2, 1)
        return audio, sr

    def forward_mel(self, audios):
        mels = []
        for i in range(len(audios)):
            image = self.vocoder.mel_transform(audios[i])
            mels.append(image)
        mels = torch.stack(mels)
        return mels

    @torch.no_grad()
    def encode(self, audios, audio_lengths=None, sr=None):
        if audio_lengths is None:
            audio_lengths = torch.tensor([audios.shape[2]] * audios.shape[0])
            audio_lengths = audio_lengths.to(audios.device)

        device = audios.device
        dtype = audios.dtype

        if sr is None:
            sr = self.source_sample_rate
        
        # Only resample if necessary
        if sr != 44100:
            if self.resampler is not None:
                resampler = self.resampler.to(device).to(dtype)
            else:
                resampler = torchaudio.transforms.Resample(sr, 44100).to(device).to(dtype)
            audio = resampler(audios)
        else:
            audio = audios

        max_audio_len = audio.shape[-1]
        if max_audio_len % (8 * 512) != 0:
            audio = torch.nn.functional.pad(
                audio, (0, 8 * 512 - max_audio_len % (8 * 512))
            )

        mels = self.forward_mel(audio)
        mels = (mels - self.min_mel_value) / (self.max_mel_value - self.min_mel_value)
        mels = self.transform(mels)
        latents = []
        for mel in mels:
            latent = self.dcae.encoder(mel.unsqueeze(0))
            latents.append(latent)
        latents = torch.cat(latents, dim=0)
        latent_lengths = (
            audio_lengths / sr * 44100 / 512 / self.time_dimention_multiple
        ).long()
        latents = (latents - self.shift_factor) * self.scale_factor
        return latents, latent_lengths

    @torch.no_grad()
    def decode(self, latents, audio_lengths=None, sr=None, use_chunking=True, chunk_size=128, overlap_size=16):
        latents = latents / self.scale_factor + self.shift_factor

        pred_wavs = []

        for latent in latents:
            # Ensure latent has the right shape (C, H, W)
            if latent.dim() == 3:
                # latent is already (C, H, W)
                current_latent = latent
            else:
                # If somehow it's 4D, squeeze batch dimension
                current_latent = latent.squeeze(0) if latent.dim() == 4 else latent
                
            if use_chunking and current_latent.shape[-1] > chunk_size:
                # Use chunked processing for large latents
                wav = self._decode_chunked(current_latent, chunk_size, overlap_size, sr)
            else:
                # Use original method for small latents
                wav = self._decode_single(current_latent, sr)
            
            pred_wavs.append(wav)

        # Don't crop the audio - let it be full length
        # The audio_lengths calculation was incorrect, causing premature truncation
        return sr if sr is not None else 44100, pred_wavs

    def _decode_single(self, latent, sr=None):
        """Original single-pass decoding"""
        mels = self.dcae.decoder(latent.unsqueeze(0))
        mels = mels * 0.5 + 0.5
        mels = mels * (self.max_mel_value - self.min_mel_value) + self.min_mel_value

        wav_ch1 = self.vocoder.decode(mels[:,0,:,:]).squeeze(1).cpu()
        wav_ch2 = self.vocoder.decode(mels[:,1,:,:]).squeeze(1).cpu()
        wav = torch.cat([wav_ch1, wav_ch2], dim=0)

        if sr is not None and sr != 44100:
            resampler = torchaudio.transforms.Resample(44100, sr)
            wav = resampler(wav.cpu().float())
        
        return wav

    def _decode_chunked(self, latent, chunk_size, overlap_size, sr=None):
        """Chunked decoding with overlap-add for better performance"""
        device = latent.device
        total_length = latent.shape[-1]
        hop_size = chunk_size - 2 * overlap_size
        
        if hop_size <= 0:
            hop_size = chunk_size // 2
            overlap_size = (chunk_size - hop_size) // 2
        
        # Calculate chunk positions
        chunk_starts = list(range(0, total_length, hop_size))
        
        # Handle the last chunk more carefully to avoid duplication
        if len(chunk_starts) > 1 and chunk_starts[-1] < total_length:
            last_start = chunk_starts[-1]
            remaining = total_length - last_start
            
            # If remaining is very small, just extend the previous chunk
            if remaining <= overlap_size:
                chunk_starts.pop()  # Remove the last chunk start
            # If remaining is substantial but less than full chunk, adjust position
            elif remaining < chunk_size and total_length - chunk_starts[-2] - hop_size > chunk_size:
                chunk_starts[-1] = total_length - chunk_size
        
        # Remove any duplicate positions
        chunk_starts = list(dict.fromkeys(chunk_starts))  # Preserves order while removing duplicates
        
        wav_chunks = []
        
        for i, start in enumerate(chunk_starts):
            end = min(start + chunk_size, total_length)
            # latent is (C, H, W), so slice the last dimension
            chunk_latent = latent[:, :, start:end]
            
            # Store original chunk size for later use
            original_chunk_size = chunk_latent.shape[-1]
            
            # Pad if chunk is too small (but remember original size)
            if chunk_latent.shape[-1] < chunk_size:
                pad_size = chunk_size - chunk_latent.shape[-1]
                chunk_latent = torch.nn.functional.pad(chunk_latent, (0, pad_size), mode='replicate')
            
            # Decode chunk
            chunk_mels = self.dcae.decoder(chunk_latent.unsqueeze(0))
            chunk_mels = chunk_mels * 0.5 + 0.5
            chunk_mels = chunk_mels * (self.max_mel_value - self.min_mel_value) + self.min_mel_value
            
            # Process each channel separately to save memory
            if chunk_mels.dim() == 4:
                # chunk_mels is (B, C, H, W)
                chunk_wav_ch1 = self.vocoder.decode(chunk_mels[:,0,:,:]).squeeze(1)
                chunk_wav_ch2 = self.vocoder.decode(chunk_mels[:,1,:,:]).squeeze(1)
            else:
                # chunk_mels is (C, H, W)
                chunk_wav_ch1 = self.vocoder.decode(chunk_mels[0,:,:].unsqueeze(0)).squeeze(1)
                chunk_wav_ch2 = self.vocoder.decode(chunk_mels[1,:,:].unsqueeze(0)).squeeze(1)
            chunk_wav = torch.cat([chunk_wav_ch1, chunk_wav_ch2], dim=0)
            
            # Crop to original size to avoid padding artifacts
            expected_wav_length = original_chunk_size * self.time_dimention_multiple * 512
            if chunk_wav.shape[-1] > expected_wav_length:
                chunk_wav = chunk_wav[:, :expected_wav_length]
            
            wav_chunks.append(chunk_wav.cpu())
            
            # Clear GPU memory
            del chunk_latent, chunk_mels, chunk_wav_ch1, chunk_wav_ch2, chunk_wav
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
        
        # Overlap-add reconstruction with proper windowing and volume normalization
        if len(wav_chunks) == 1:
            final_wav = wav_chunks[0]
        else:
            # Calculate total length to match original MEL spectrogram
            total_audio_length = total_length * self.time_dimention_multiple * 512
            final_wav = torch.zeros(wav_chunks[0].shape[0], total_audio_length)
            overlap_audio_size = overlap_size * self.time_dimention_multiple * 512
            
            # Normalize volume across chunks to prevent volume variations
            chunk_rms = []
            for chunk in wav_chunks:
                rms = torch.sqrt(torch.mean(chunk ** 2))
                chunk_rms.append(rms)
            
            # Use median RMS for normalization to avoid outliers
            target_rms = torch.median(torch.stack(chunk_rms))
            
            # Normalize chunks to consistent volume
            normalized_chunks = []
            for i, chunk in enumerate(wav_chunks):
                if chunk_rms[i] > 1e-8:  # Avoid division by zero
                    normalized_chunk = chunk * (target_rms / chunk_rms[i])
                else:
                    normalized_chunk = chunk
                normalized_chunks.append(normalized_chunk)
            wav_chunks = normalized_chunks
                
            for i, chunk in enumerate(wav_chunks):
                start_pos = i * hop_size * self.time_dimention_multiple * 512
                end_pos = start_pos + chunk.shape[-1]
                
                if end_pos <= total_audio_length:
                    if i == 0:
                        # First chunk: no fade-in needed
                        final_wav[:, start_pos:end_pos] = chunk
                    else:
                        # Apply crossfade for overlapping regions
                        non_overlap_start = start_pos + overlap_audio_size
                        
                        if non_overlap_start < end_pos:
                            # Handle overlap region with crossfade
                            if overlap_audio_size > 0 and start_pos < final_wav.shape[-1]:
                                overlap_end = min(start_pos + overlap_audio_size, final_wav.shape[-1], end_pos)
                                overlap_len = overlap_end - start_pos
                                
                                if overlap_len > 0:
                                    # Use energy-preserving equal-power crossfade
                                    fade_out = torch.cos(torch.linspace(0, torch.pi/2, overlap_len))
                                    fade_in = torch.sin(torch.linspace(0, torch.pi/2, overlap_len))
                                    
                                    # Apply equal-power crossfade (preserves energy: cos²+sin²=1)
                                    final_wav[:, start_pos:overlap_end] *= fade_out
                                    chunk_overlap = chunk[:, :overlap_len] * fade_in
                                    final_wav[:, start_pos:overlap_end] += chunk_overlap
                            
                            # Add non-overlapping part
                            if non_overlap_start < end_pos:
                                chunk_start_idx = overlap_audio_size
                                final_wav[:, non_overlap_start:end_pos] = chunk[:, chunk_start_idx:chunk_start_idx + (end_pos - non_overlap_start)]
                        else:
                            # No non-overlapping part, just handle overlap
                            if overlap_audio_size > 0 and start_pos < final_wav.shape[-1]:
                                overlap_end = min(end_pos, final_wav.shape[-1])
                                overlap_len = overlap_end - start_pos
                                
                                if overlap_len > 0:
                                    fade_out = torch.cos(torch.linspace(0, torch.pi/2, overlap_len))
                                    fade_in = torch.sin(torch.linspace(0, torch.pi/2, overlap_len))
                                    
                                    final_wav[:, start_pos:overlap_end] *= fade_out
                                    chunk_overlap = chunk[:, :overlap_len] * fade_in
                                    final_wav[:, start_pos:overlap_end] += chunk_overlap
                else:
                    # Handle final chunk that might exceed total length
                    remaining = total_audio_length - start_pos
                    if remaining > 0:
                        if i > 0 and overlap_audio_size > 0:
                            # Apply fade for the overlap region
                            overlap_len = min(overlap_audio_size, remaining, chunk.shape[-1])
                            if overlap_len > 0:
                                fade_out = torch.cos(torch.linspace(0, torch.pi/2, overlap_len))
                                fade_in = torch.sin(torch.linspace(0, torch.pi/2, overlap_len))
                                
                                final_wav[:, start_pos:start_pos + overlap_len] *= fade_out
                                chunk_overlap = chunk[:, :overlap_len] * fade_in
                                final_wav[:, start_pos:start_pos + overlap_len] += chunk_overlap
                                
                                # Add remaining part if any
                                if overlap_len < remaining and overlap_len < chunk.shape[-1]:
                                    final_wav[:, start_pos + overlap_len:total_audio_length] = chunk[:, overlap_len:remaining]
                        else:
                            final_wav[:, start_pos:total_audio_length] = chunk[:, :remaining]
        
        if sr is not None and sr != 44100:
            resampler = torchaudio.transforms.Resample(44100, sr)
            final_wav = resampler(final_wav.float())
        
        return final_wav

    def forward(self, audios, audio_lengths=None, sr=None):
        latents, latent_lengths = self.encode(
            audios=audios, audio_lengths=audio_lengths, sr=sr
        )
        sr, pred_wavs = self.decode(latents=latents, audio_lengths=audio_lengths, sr=sr)
        return sr, pred_wavs, latents, latent_lengths