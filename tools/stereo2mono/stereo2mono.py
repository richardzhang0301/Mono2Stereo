# tools/stereo2mono/stereo2mono.py

import os
import sys
import argparse
import subprocess

def main():
    parser = argparse.ArgumentParser(description='Create mono and stereo .wav files from input folder with sample rate handling.')
    parser.add_argument('input_folder', help='Input folder containing audio files.')
    parser.add_argument('mono_output_folder', help='Output folder for mono .wav files.')
    parser.add_argument('stereo_output_folder', help='Output folder for stereo .wav files.')
    parser.add_argument('--sample_rate', type=int, default=48000, help='Desired sample rate for output files. Defaults to 48000Hz.')
    args = parser.parse_args()

    input_folder = os.path.abspath(args.input_folder)
    mono_output_folder = os.path.abspath(args.mono_output_folder)
    stereo_output_folder = os.path.abspath(args.stereo_output_folder)
    desired_sample_rate = args.sample_rate

    check_ffmpeg()

    # Walk through input folder
    for root, dirs, files in os.walk(input_folder):
        for file in files:
            # Check if file is an audio file by extension
            if file.lower().endswith(('.wav', '.mp3', '.m4a', '.flac', '.ape', '.wma')):
                input_file = os.path.join(root, file)
                # Compute relative path
                rel_path = os.path.relpath(root, input_folder)
                # Prepare output paths
                mono_output_dir = os.path.join(mono_output_folder, rel_path)
                stereo_output_dir = os.path.join(stereo_output_folder, rel_path)
                # Ensure output directories exist
                os.makedirs(mono_output_dir, exist_ok=True)
                os.makedirs(stereo_output_dir, exist_ok=True)
                # Output file names (change extension to .wav)
                base_name = os.path.splitext(file)[0]
                mono_output_file = os.path.join(mono_output_dir, base_name + '.wav')
                stereo_output_file = os.path.join(stereo_output_dir, base_name + '.wav')
                print(f'Processing {input_file}', flush=True)
                # Get input file's sample rate
                input_sample_rate = get_sample_rate(input_file)
                if input_sample_rate is None:
                    print(f"Skipping {input_file} due to sample rate retrieval failure.", flush=True)
                    continue
                # Determine if resampling is needed
                resample_flag = []
                if input_sample_rate != desired_sample_rate:
                    print(f'Resampling {input_file} from {input_sample_rate}Hz to {desired_sample_rate}Hz', flush=True)
                    resample_flag = ['-ar', str(desired_sample_rate)]
                # Generate mono wav file
                create_mono_wav(input_file, mono_output_file, resample_flag)
                # Generate stereo wav file
                create_stereo_wav(input_file, stereo_output_file, resample_flag)

def get_sample_rate(file_path):
    """
    Retrieves the sample rate of an audio file using ffprobe.

    Parameters:
    - file_path (str): Path to the audio file.

    Returns:
    - sample_rate (int): Sample rate of the audio file in Hz.
    """
    cmd = [
        'ffprobe', '-v', 'error', '-select_streams', 'a:0',
        '-show_entries', 'stream=sample_rate', '-of', 'default=noprint_wrappers=1:nokey=1',
        file_path
    ]
    try:
        result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=True, encoding='utf-8', errors='replace')
        sample_rate_str = result.stdout.strip()
        if sample_rate_str.isdigit():
            return int(sample_rate_str)
        else:
            print(f'Invalid sample rate value for {file_path}: "{sample_rate_str}"', flush=True)
            return None
    except subprocess.CalledProcessError as e:
        print(f'Error retrieving sample rate for {file_path}:\n{e.stderr}', flush=True)
        return None

def create_mono_wav(input_file, output_file, resample_flag):
    """
    Uses ffmpeg to create a mono .wav file from the input audio file.
    Resamples if resample_flag is provided.

    Parameters:
    - input_file (str): Path to the input audio file.
    - output_file (str): Path to the output mono .wav file.
    - resample_flag (list): FFmpeg flags for resampling, e.g., ['-ar', '48000'] or empty list.
    """
    # Use ffmpeg to mix left and right channels and output mono .wav
    cmd = [
        'ffmpeg', '-y', '-i', input_file
    ] + resample_flag + [
        '-ac', '1',
        '-filter_complex', 'pan=mono|c0=0.5*c0+0.5*c1',
        output_file
    ]
    try:
        result = subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True, encoding='utf-8', errors='replace')
        if result.returncode != 0:
            print(f'Error processing {input_file} for mono output:\n{result.stderr}', flush=True)
    except Exception as e:
        print(f'Unexpected error processing {input_file} for mono output: {e}', flush=True)

def create_stereo_wav(input_file, output_file, resample_flag):
    """
    Uses ffmpeg to convert an audio file to stereo .wav format.
    Resamples if resample_flag is provided.

    Parameters:
    - input_file (str): Path to the input audio file.
    - output_file (str): Path to the output stereo .wav file.
    - resample_flag (list): FFmpeg flags for resampling, e.g., ['-ar', '48000'] or empty list.
    """
    # Use ffmpeg to convert to stereo .wav
    cmd = [
        'ffmpeg', '-y', '-i', input_file
    ] + resample_flag + [
        '-ac', '2',
        output_file
    ]
    try:
        result = subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True, encoding='utf-8', errors='replace')
        if result.returncode != 0:
            print(f'Error processing {input_file} for stereo output:\n{result.stderr}', flush=True)
    except Exception as e:
        print(f'Unexpected error processing {input_file} for stereo output: {e}', flush=True)

def check_ffmpeg():
    """
    Checks if ffmpeg is installed and accessible.
    Exits the script if ffmpeg is not found.
    """
    try:
        subprocess.run(['ffmpeg', '-version'], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
    except FileNotFoundError:
        print('ffmpeg is not installed or not found in system PATH.', flush=True)
        sys.exit(1)
    except subprocess.CalledProcessError:
        print('ffmpeg is installed but encountered an error.', flush=True)
        sys.exit(1)

if __name__ == '__main__':
    main()
