# inference.py

import sys
from pathlib import Path

def main():
    # Add 'src' directory to the system path to allow imports from src/
    src_path = Path(__file__).resolve().parent / 'src'
    sys.path.insert(0, str(src_path))

    try:
        from interface.interface import convert_mono_to_stereo
    except ImportError as e:
        print(f"Error importing convert_mono_to_stereo: {e}")
        sys.exit(1)

    import argparse

    # Setup command-line argument parsing
    parser = argparse.ArgumentParser(description='Mono to Stereo Audio Conversion')
    parser.add_argument('--input', type=str, required=True, help='Path to input mono .wav file')
    parser.add_argument('--output', type=str, required=True, help='Path to output stereo .wav file')

    args = parser.parse_args()

    # Invoke the conversion function
    convert_mono_to_stereo(args.input, args.output)

if __name__ == '__main__':
    main()
