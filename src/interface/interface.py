# your_project/interface/interface.py

import argparse
from inference.inference import convert_mono_to_stereo
from training.train import train_model

def main():
    parser = argparse.ArgumentParser(description='Mono to Stereo Audio Conversion')
    subparsers = parser.add_subparsers(dest='command')

    # Training command
    train_parser = subparsers.add_parser('train', help='Train the model')

    # Inference command
    infer_parser = subparsers.add_parser('infer', help='Convert mono audio to stereo')
    infer_parser.add_argument('--input', type=str, required=True, help='Path to input mono .wav file')
    infer_parser.add_argument('--output', type=str, required=True, help='Path to output stereo .wav file')

    args = parser.parse_args()

    if args.command == 'train':
        train_model()
    elif args.command == 'infer':
        convert_mono_to_stereo(args.input, args.output)
    else:
        parser.print_help()

if __name__ == '__main__':
    main()
