#!/usr/bin/env python3

"""
preprocessing.py

A script for preparing raw audio data for the SoundSight project.
Features:
- Consistent sample rate
- Mono channel
- Trim leading/trailing silence
- Loudness normalization
- Random augmentation (pitch shift, time stretch, background noise)
- Organized batch processing and metadata logging

Usage:
  python preprocessing.py --input_dir data --output_dir data_preprocessed
    [--sample_rate 16000] [--augment] [--augment_prob 0.5]
    [--normalize] [--noise_dir data/noise]

Example:
  python preprocessing.py --input_dir data --output_dir data_preprocessed --sample_rate 16000 --augment --normalize
"""

import os
import sys
import argparse
import random
import csv
import numpy as np
import librosa
import soundfile as sf
from pydub import AudioSegment


def preprocess_audio(
    input_path: str,
    sample_rate: int = 16000,
    top_db: int = 30,
    loudness_normalize: bool = False
):
    """
    Load an audio file, resample, convert to mono, trim silence,
    and optionally normalize loudness.

    Returns:
        y (np.ndarray): Processed audio samples.
        sr (int): The sample rate of the processed audio.
    """
    # 1. Load audio (as float32 array)
    y, sr = librosa.load(input_path, sr=None, mono=True)

    # 2. Resample if needed
    if sr != sample_rate:
        y = librosa.resample(y, orig_sr=sr, target_sr=sample_rate)
        sr = sample_rate

    # 3. Trim leading and trailing silence
    y, _ = librosa.effects.trim(y, top_db=top_db)

    # 4. Optional: Loudness normalization using pydub
    if loudness_normalize:
        # Convert numpy array -> pydub AudioSegment for normalization
        temp_wav_path = "temp_norm.wav"
        sf.write(temp_wav_path, y, sr)
        seg = AudioSegment.from_wav(temp_wav_path)
        # Normalize to -20 dBFS (common reference)
        change_in_dBFS = -20.0 - seg.dBFS
        seg = seg.apply_gain(change_in_dBFS)
        # Back to numpy
        seg.export(temp_wav_path, format="wav")
        y, sr = librosa.load(temp_wav_path, sr=None, mono=True)
        # Clean up
        os.remove(temp_wav_path)

    return y, sr


def augment_audio(
    y: np.ndarray,
    sr: int,
    noise_files: list = None,
    noise_prob: float = 0.3,
    pitch_range: int = 2,
    time_stretch_range: tuple = (0.9, 1.1)
):
    """
    Apply random audio augmentations:
    - Pitch shift in range [-pitch_range, pitch_range].
    - Time stretch in [time_stretch_range[0], time_stretch_range[1]].
    - Optional background noise overlay from a list of noise_files.

    Returns:
        y_aug (np.ndarray): Augmented audio data.
    """
    # 1. Time stretch
    rate = random.uniform(*time_stretch_range)  # e.g. between 0.9 and 1.1
    y_stretch = librosa.effects.time_stretch(y, rate)

    # 2. Pitch shift
    steps = random.randint(-pitch_range, pitch_range)  # e.g. -2 to 2 semitones
    y_shift = librosa.effects.pitch_shift(y_stretch, sr, n_steps=steps)

    # 3. (Optional) Background noise overlay
    if noise_files and random.random() < noise_prob:
        noise_file = random.choice(noise_files)
        # Load noise at sr
        noise_wav, noise_sr = librosa.load(noise_file, sr=sr, mono=True)
        # Trim or repeat noise to match length of y_shift
        if len(noise_wav) < len(y_shift):
            # Tile the noise to match length
            repeats = int(len(y_shift) / len(noise_wav)) + 1
            noise_wav = np.tile(noise_wav, repeats)
        noise_wav = noise_wav[:len(y_shift)]

        # Mix noise at a random SNR (e.g., -5 to 5 dB)
        # The ratio is random, can be adjusted
        snr_dB = random.uniform(-5, 5)
        # Convert snr_dB to a linear scale
        snr_linear = 10 ** (snr_dB / 20)
        # Make sure volumes are properly scaled
        y_shift_norm = y_shift / np.max(np.abs(y_shift) + 1e-8)
        noise_wav_norm = noise_wav / np.max(np.abs(noise_wav) + 1e-8) * (1 / snr_linear)

        y_mixed = y_shift_norm + noise_wav_norm
        # Normalize to avoid clipping
        if np.max(np.abs(y_mixed)) > 0:
            y_mixed /= np.max(np.abs(y_mixed))
        y_aug = y_mixed
    else:
        y_aug = y_shift

    return y_aug


def batch_preprocess(
    input_dir: str = "data",
    output_dir: str = "data_preprocessed",
    sample_rate: int = 16000,
    top_db: int = 30,
    loudness_normalize: bool = False,
    do_augment: bool = False,
    augment_prob: float = 0.5,
    noise_dir: str = None
):
    """
    Walk through subdirectories of input_dir, preprocess each .wav/.mp3/.flac file,
    and save the result in a mirrored folder structure under output_dir.
    Optionally perform augmentation at a given probability (augment_prob).
    Logs metadata to a CSV file in output_dir.
    """

    # 1. Gather noise files (if any)
    noise_files = []
    if noise_dir and os.path.exists(noise_dir):
        for f in os.listdir(noise_dir):
            if f.lower().endswith(('.wav', '.mp3', '.flac')):
                noise_files.append(os.path.join(noise_dir, f))
        print(f"[INFO] Found {len(noise_files)} noise files in {noise_dir}.")

    # 2. Prepare output directory
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 3. Prepare metadata logger
    metadata_file = os.path.join(output_dir, "metadata_preprocessed.csv")
    fieldnames = [
        "input_file", "output_file", "label",
        "sample_rate", "duration_sec",
        "augmented", "augment_details"
    ]

    # If the metadata file already exists, we append; otherwise, create a new one.
    write_header = not os.path.exists(metadata_file)
    with open(metadata_file, mode="a", newline='', encoding='utf-8') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()

        # 4. Traverse input directory
        for root, dirs, files in os.walk(input_dir):
            # The subdir relative to input_dir
            relative_path = os.path.relpath(root, input_dir)
            # Create mirrored path in output_dir
            out_subdir = os.path.join(output_dir, relative_path)
            if not os.path.exists(out_subdir):
                os.makedirs(out_subdir, exist_ok=True)

            for fname in files:
                if fname.lower().endswith(('.wav', '.mp3', '.flac')):
                    input_path = os.path.join(root, fname)
                    # Build an output filename with .wav extension
                    base_name = os.path.splitext(fname)[0]
                    output_filename = base_name + ".wav"
                    output_path = os.path.join(out_subdir, output_filename)

                    # 4a. Determine label (assume folder structure => category name)
                    label = os.path.basename(root)

                    # 4b. Basic preprocessing
                    y_clean, sr_clean = preprocess_audio(
                        input_path,
                        sample_rate=sample_rate,
                        top_db=top_db,
                        loudness_normalize=loudness_normalize
                    )
                    duration_clean = len(y_clean) / sr_clean

                    # 4c. Save the "clean" version
                    sf.write(output_path, y_clean, sr_clean)

                    # 4d. Log metadata for the clean version
                    writer.writerow({
                        "input_file": input_path,
                        "output_file": output_path,
                        "label": label,
                        "sample_rate": sr_clean,
                        "duration_sec": round(duration_clean, 3),
                        "augmented": False,
                        "augment_details": ""
                    })

                    # 4e. Optional augmentation
                    if do_augment and random.random() < augment_prob:
                        y_aug = augment_audio(y_clean, sr_clean, noise_files=noise_files)
                        duration_aug = len(y_aug) / sr_clean

                        # Construct augmented filename
                        output_aug_filename = base_name + "_aug.wav"
                        output_aug_path = os.path.join(out_subdir, output_aug_filename)
                        sf.write(output_aug_path, y_aug, sr_clean)

                        # Log metadata for the augmented version
                        writer.writerow({
                            "input_file": input_path,
                            "output_file": output_aug_path,
                            "label": label,
                            "sample_rate": sr_clean,
                            "duration_sec": round(duration_aug, 3),
                            "augmented": True,
                            "augment_details": "pitch/time_stretch/noise"
                        })

    print(f"[DONE] Preprocessing complete. Metadata logged to {metadata_file}")


def parse_arguments():
    parser = argparse.ArgumentParser(description="Preprocess and optionally augment audio files.")
    parser.add_argument("--input_dir", type=str, default="data", help="Path to the input directory with raw audio.")
    parser.add_argument("--output_dir", type=str, default="data_preprocessed", help="Path to the output directory.")
    parser.add_argument("--sample_rate", type=int, default=16000, help="Target sample rate for output audio.")
    parser.add_argument("--top_db", type=int, default=30, help="Threshold (in dB) below reference to consider as silence.")
    parser.add_argument("--normalize", action="store_true", help="Perform loudness normalization to -20 dBFS.")
    parser.add_argument("--augment", action="store_true", help="If set, creates augmented copies of audio.")
    parser.add_argument("--augment_prob", type=float, default=0.5, help="Probability of performing augmentation on a file.")
    parser.add_argument("--noise_dir", type=str, default=None, help="Directory containing noise files for overlay.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()

    # Print config
    print(f"=== Preprocessing Configuration ===")
    print(f"Input Directory     : {args.input_dir}")
    print(f"Output Directory    : {args.output_dir}")
    print(f"Sample Rate         : {args.sample_rate}")
    print(f"Top dB (silence)    : {args.top_db}")
    print(f"Loudness Normalize  : {args.normalize}")
    print(f"Augment Data        : {args.augment}")
    print(f"Augment Probability : {args.augment_prob}")
    print(f"Noise Directory     : {args.noise_dir}")
    print("====================================\n")

    batch_preprocess(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        sample_rate=args.sample_rate,
        top_db=args.top_db,
        loudness_normalize=args.normalize,
        do_augment=args.augment,
        augment_prob=args.augment_prob,
        noise_dir=args.noise_dir
    )
