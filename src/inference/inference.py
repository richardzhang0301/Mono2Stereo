# src/interface/interface.py

import torch
import torchaudio
import torch.nn as nn
from pathlib import Path

from configs.config import Config
from models.unet import UNetAudio

def convert_mono_to_stereo(input_mono_path, output_stereo_path):
    # Load configurations
    config = Config()

    # Load the trained model
    model = UNetAudio()
    try:
        model.load_state_dict(torch.load(config.MODEL_SAVE_PATH, map_location=config.DEVICE))
        model.to(config.DEVICE)
        model.eval()
    except FileNotFoundError:
        print(f"Model file not found at {config.MODEL_SAVE_PATH}. Please train the model first.")
        return
    except Exception as e:
        print(f"Error loading the model: {e}")
        return

    # Use Path to handle file paths
    input_mono_path = Path(input_mono_path)
    output_stereo_path = Path(output_stereo_path)

    # Load mono audio file
    try:
        mono_waveform, sr = torchaudio.load(str(input_mono_path))
    except Exception as e:
        print(f"Error loading mono audio file: {e}")
        return

    # Resample if necessary
    if sr != config.SAMPLE_RATE:
        mono_waveform = torchaudio.functional.resample(mono_waveform, sr, config.SAMPLE_RATE)

    # Ensure mono waveform has one channel
    if mono_waveform.shape[0] != 1:
        mono_waveform = mono_waveform.mean(dim=0, keepdim=True)

    # Process in chunks to handle large files
    total_length = mono_waveform.shape[1]
    stereo_waveform = []

    with torch.no_grad():
        for start in range(0, total_length, config.CHUNK_SIZE):
            end = min(start + config.CHUNK_SIZE, total_length)
            mono_chunk = mono_waveform[:, start:end]

            # Pad if necessary
            if mono_chunk.shape[1] < config.CHUNK_SIZE:
                padding = config.CHUNK_SIZE - mono_chunk.shape[1]
                mono_chunk = nn.functional.pad(mono_chunk, (0, padding))

            mono_chunk = mono_chunk.to(config.DEVICE)
            mono_chunk = mono_chunk.unsqueeze(0)  # [1, 1, chunk_size]

            # Generate stereo chunk
            try:
                stereo_chunk = model(mono_chunk)
            except Exception as e:
                print(f"Error during model inference: {e}")
                return

            stereo_chunk = stereo_chunk.squeeze(0).cpu()  # [2, chunk_size]

            # Remove padding
            stereo_chunk = stereo_chunk[:, :end - start]

            stereo_waveform.append(stereo_chunk)

    # Concatenate all stereo chunks
    if stereo_waveform:
        stereo_waveform = torch.cat(stereo_waveform, dim=1)
    else:
        print("No audio data found in the input file.")
        return

    # Save stereo audio file
    try:
        torchaudio.save(str(output_stereo_path), stereo_waveform, sample_rate=config.SAMPLE_RATE)
        print(f'Stereo audio saved to {output_stereo_path}')
    except Exception as e:
        print(f"Error saving stereo audio file: {e}")
