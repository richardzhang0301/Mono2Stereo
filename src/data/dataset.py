# your_project/data/dataset.py

import os
import torch
import torchaudio
import torch.nn as nn
from torch.utils.data import Dataset
from pathlib import Path
from configs.config import Config
import random

def get_relative_paths(root_dir):
    paths = []
    root_path = Path(root_dir)
    for path in root_path.rglob('*.wav'):
        if path.is_file():
            rel_path = path.relative_to(root_path)
            paths.append(rel_path)
    return sorted(paths)

class MonoStereoDataset(Dataset):
    config = Config()
    
    def __init__(self, mono_dir, stereo_dir, transform=None, sample_rate=config.SAMPLE_RATE, chunk_size=config.CHUNK_SIZE):
        """
        Custom Dataset for loading mono and stereo audio files.

        Parameters:
        - mono_dir (str or Path): Path to the directory containing mono audio files.
        - stereo_dir (str or Path): Path to the directory containing stereo audio files.
        - transform (callable, optional): Optional transform to be applied on a sample.
        - sample_rate (int): Sample rate to which all audio files will be resampled.
        - chunk_size (int): Number of samples per audio chunk (e.g., sample_rate * duration_in_seconds).
        """
        self.mono_dir = Path(mono_dir)
        self.stereo_dir = Path(stereo_dir)
        self.transform = transform
        self.sample_rate = sample_rate
        self.chunk_size = chunk_size  # Fixed chunk size (e.g., 5 seconds)

        # Get relative paths
        self.mono_files_rel = get_relative_paths(self.mono_dir)
        self.stereo_files_rel = get_relative_paths(self.stereo_dir)

        # Get common relative paths
        self.common_files_rel = sorted(list(set(self.mono_files_rel) & set(self.stereo_files_rel)))

        # Ensure there are common files
        assert len(self.common_files_rel) > 0, "No common files found between mono and stereo directories."

        # Get full paths
        self.mono_files = [self.mono_dir / f for f in self.common_files_rel]
        self.stereo_files = [self.stereo_dir / f for f in self.common_files_rel]

    def __len__(self):
        return len(self.common_files_rel)

    def __getitem__(self, idx):
        mono_file = self.mono_files[idx]
        stereo_file = self.stereo_files[idx]

        try:
            # Load audio files
            mono_waveform, sample_rate_mono = torchaudio.load(str(mono_file))
            stereo_waveform, sample_rate_stereo = torchaudio.load(str(stereo_file))
        except Exception as e:
            print(f'Error loading files:\nMono: {mono_file}\nStereo: {stereo_file}\nError: {e}')
            raise e

        # Resample if necessary
        if sample_rate_mono != self.sample_rate:
            mono_waveform = torchaudio.functional.resample(mono_waveform, sample_rate_mono, self.sample_rate)
        if sample_rate_stereo != self.sample_rate:
            stereo_waveform = torchaudio.functional.resample(stereo_waveform, sample_rate_stereo, self.sample_rate)

        # Convert stereo to two channels if necessary
        if stereo_waveform.shape[0] != 2:
            stereo_waveform = stereo_waveform.repeat(2, 1)

        # Ensure both audio files have the same length
        min_length = min(mono_waveform.shape[1], stereo_waveform.shape[1])

        # Check if the audio is shorter than the chunk size
        if min_length < self.chunk_size:
            # If the audio is shorter, pad it to the chunk size
            pad_amount = self.chunk_size - min_length
            mono_waveform = nn.functional.pad(mono_waveform, (0, pad_amount))
            stereo_waveform = nn.functional.pad(stereo_waveform, (0, pad_amount))
        else:
            # Randomly select a chunk from the audio
            max_offset = min_length - self.chunk_size
            offset = random.randint(0, max_offset)
            mono_waveform = mono_waveform[:, offset:offset + self.chunk_size]
            stereo_waveform = stereo_waveform[:, offset:offset + self.chunk_size]

        # Optional transformations
        if self.transform:
            mono_waveform = self.transform(mono_waveform)
            stereo_waveform = self.transform(stereo_waveform)

        return mono_waveform, stereo_waveform
