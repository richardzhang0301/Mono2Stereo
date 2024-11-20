# preprocess.py

import json
import subprocess
import sys
from pathlib import Path

def load_config(config_path: str = "config.json") -> dict:
    """
    Loads the configuration from a JSON file.

    Parameters:
    - config_path (str): Path to the config.json file.

    Returns:
    - config (dict): Dictionary containing configuration parameters.
    """
    config_file = Path(__file__).resolve().parent / config_path
    if not config_file.exists():
        print(f"Configuration file not found at {config_file}")
        sys.exit(1)
    
    with open(config_file, 'r') as f:
        config = json.load(f)
    
    return config

def run_stereo2mono(original_dir: str, mono_dir: str, stereo_dir: str, sample_rate: int):
    """
    Executes the stereo2mono.py script with the provided directories and sample rate.

    Parameters:
    - original_dir (str): Path to the original dataset directory.
    - mono_dir (str): Path to the output mono directory.
    - stereo_dir (str): Path to the output stereo directory.
    - sample_rate (int): Desired sample rate for output files.
    """
    stereo2mono_script = Path(__file__).resolve().parent / "tools" / "stereo2mono" / "stereo2mono.py"
    
    if not stereo2mono_script.exists():
        print(f"stereo2mono.py not found at {stereo2mono_script}")
        sys.exit(1)
    
    # Construct the command
    cmd = [
        sys.executable,  # Path to the Python interpreter
        str(stereo2mono_script),
        original_dir,
        mono_dir,
        stereo_dir,
        '--sample_rate', str(sample_rate)
    ]
    
    print("Starting preprocessing using stereo2mono.py...")
    try:
        result = subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        print(result.stdout)
        print("Preprocessing completed successfully.")
    except subprocess.CalledProcessError as e:
        print("An error occurred during preprocessing:")
        print(e.stderr)
        sys.exit(1)

def main():
    # Load configurations
    config = load_config()
    
    original_dataset_dir = config.get("original_dataset_dir")
    mono_dir = config.get("mono_dir")
    stereo_dir = config.get("stereo_dir")
    sample_rate = config.get("sample_rate", 48000)  # Default to 48000Hz if not specified
    
    if not all([original_dataset_dir, mono_dir, stereo_dir]):
        print("Please ensure 'original_dataset_dir', 'mono_dir', and 'stereo_dir' are specified in config.json.")
        sys.exit(1)
    
    run_stereo2mono(original_dataset_dir, mono_dir, stereo_dir, sample_rate)

if __name__ == "__main__":
    main()
