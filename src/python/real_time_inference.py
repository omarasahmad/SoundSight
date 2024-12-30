#!/usr/bin/env python3
"""
real_time_inference.py

Perform real-time sound classification using a microphone on a PC environment.
Loads a trained SoundCNN model, listens to short audio segments, and prints predictions.

Usage:
  (venv) $ python src/python/real_time_inference.py
"""

import os
import queue
import sys
import time
import numpy as np
import torch
import torch.nn as nn
import torchaudio
import torchaudio.transforms as T
import sounddevice as sd

from train_model import SoundCNN  # Reuse the CNN architecture

# -------------------
# Config
# -------------------
MODEL_PATH = "models/best_sound_model.pth"  # Must match your training output
SAMPLE_RATE = 16000
NUM_CLASSES = 6  # Adjust to match your model
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# You must use the same category ordering you had in train_model
# If you used dataset.class_to_idx with categories sorted, replicate them here:
CATEGORIES = [
    "car_horn",
    "footsteps",
    "laughing",
    "rain",
    "siren",
    "wind"
]
# Make sure this order is exactly how your final model expects them.

# Length of audio chunk to record in seconds
CHUNK_DURATION = 1.0

# -------------------
# Load Model
# -------------------
def load_model(model_path: str) -> nn.Module:
    model = SoundCNN(num_classes=NUM_CLASSES)
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.eval()
    model.to(DEVICE)
    print(f"Loaded model from {model_path}")
    return model

# -------------------
# Audio Processing Helpers
# -------------------
def audio_chunk_to_melspec(audio_chunk: np.ndarray, sr: int = SAMPLE_RATE) -> torch.Tensor:
    """
    Convert a NumPy audio chunk to a MelSpectrogram torch.Tensor
    matching the pipeline in train_model.py
    """
    # Convert to torch tensor
    waveform = torch.from_numpy(audio_chunk).float().unsqueeze(0)  # shape: [1, samples]

    # Create Mel Spectrogram
    mel_transform = T.MelSpectrogram(
        sample_rate=sr,
        n_mels=64,
        n_fft=1024,
        hop_length=512
    )
    mel_spectrogram = mel_transform(waveform)  # [1, 64, time_frames]

    # Convert amplitude to dB
    db_transform = T.AmplitudeToDB(top_db=80)
    mel_db = db_transform(mel_spectrogram)  # shape: [1, 64, time_frames]

    return mel_db

# -------------------
# Real-Time Audio Inference
# -------------------
def run_inference(model: nn.Module):
    """
    Continuously capture audio in chunks of CHUNK_DURATION seconds,
    classify, and print the predicted category.
    Press Ctrl+C to stop.
    """
    print("Starting real-time inference. Press Ctrl+C to stop.")

    audio_q = queue.Queue()

    def audio_callback(indata, frames, time_info, status):
        """Callback function for sounddevice to put recorded data into a queue."""
        if status:
            print(status, file=sys.stderr)
        # indata is shape [frames, channels]; convert to mono if necessary
        if indata.shape[1] > 1:
            mono_data = np.mean(indata, axis=1)
        else:
            mono_data = indata[:, 0]
        audio_q.put(mono_data)

    # Start recording stream
    with sd.InputStream(callback=audio_callback,
                        channels=1,
                        samplerate=SAMPLE_RATE,
                        blocksize=int(SAMPLE_RATE * CHUNK_DURATION)):
        try:
            while True:
                # Get next chunk from the queue
                chunk = audio_q.get()
                # chunk is 1D NumPy array of length ~ SAMPLE_RATE * CHUNK_DURATION

                # Convert chunk to Mel Spectrogram
                mel_db = audio_chunk_to_melspec(chunk, SAMPLE_RATE)
                mel_db = mel_db.unsqueeze(0).to(DEVICE)  # shape: [B=1, 1, 64, time_frames]

                # Inference
                with torch.no_grad():
                    outputs = model(mel_db)
                    probs = torch.softmax(outputs, dim=1)
                    pred_class = torch.argmax(probs, dim=1).item()
                    confidence = probs[0, pred_class].item()

                predicted_label = CATEGORIES[pred_class]
                print(f"Detected: {predicted_label} ({confidence*100:.1f}% confidence)")

                # Sleep briefly to avoid printing too rapidly
                # (We're chunking at CHUNK_DURATION-second intervals, so this is optional.)
                time.sleep(0.1)

        except KeyboardInterrupt:
            print("Stopping real-time inference.")

def main():
    if not os.path.exists(MODEL_PATH):
        print(f"Model file not found: {MODEL_PATH}")
        sys.exit(1)

    # Load the trained model
    model = load_model(MODEL_PATH)

    # Start microphone-based inference
    run_inference(model)

if __name__ == "__main__":
    main()
