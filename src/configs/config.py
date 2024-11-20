# src/configs/config.py

import json
from pathlib import Path
import torch

class Config:
    def __init__(self, config_path: str = "config.json"):
        """
        Loads configuration from a JSON file.

        Parameters:
        - config_path (str): Relative path to the config.json file from the location of this script.
        """
        # Resolve the path relative to the current file
        config_file = Path(__file__).resolve().parent.parent.parent / config_path

        if not config_file.exists():
            raise FileNotFoundError(f"Configuration file not found at {config_file}")

        with open(config_file, 'r') as f:
            config = json.load(f)

        # Load configurations with defaults
        self.SAMPLE_RATE = config.get("sample_rate", 48000)
        self.CHUNK_SIZE = config.get("chunk_size", self.SAMPLE_RATE * 5)
        self.NUM_EPOCHS = config.get("num_epochs", 50)
        self.BATCH_SIZE = config.get("batch_size", 4)
        self.LEARNING_RATE = config.get("learning_rate", 0.001)
        self.MONO_DIR = config.get("mono_dir", "./mono")
        self.STEREO_DIR = config.get("stereo_dir", "./stereo")
        self.MODEL_SAVE_FILENAME = config.get("model_save_path", "model/unet_audio_model.pth")
        self.DEVICE_STR = config.get("device", "cuda") if torch.cuda.is_available() else "cpu"

        # Device configuration
        self.DEVICE = torch.device(self.DEVICE_STR)

        # Construct full path for model saving
        self.MODEL_SAVE_PATH = Path(__file__).resolve().parent.parent.parent / self.MODEL_SAVE_FILENAME

        # Ensure the model directory exists
        self.MODEL_SAVE_PATH.parent.mkdir(parents=True, exist_ok=True)
