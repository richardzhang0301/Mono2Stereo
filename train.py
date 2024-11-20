# train.py

import sys
from pathlib import Path

def main():
    # Add 'src' directory to the system path to allow imports from src/
    src_path = Path(__file__).resolve().parent / 'src'
    sys.path.insert(0, str(src_path))

    try:
        from training.train import train_model
    except ImportError as e:
        print(f"Error importing train_model: {e}")
        sys.exit(1)

    # Invoke the training function
    train_model()

if __name__ == '__main__':
    main()
