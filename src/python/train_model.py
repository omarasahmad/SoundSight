#!/usr/bin/env python3
"""
train_model.py

Train a CNN-based sound classification model using data in data_preprocessed/.

Usage (from SoundSight/):
  (venv) $ python src/python/train_model.py
"""

import os
import glob
import random
import numpy as np
from typing import List, Tuple

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

import librosa
import torchaudio
import torchaudio.transforms as T

# -------------------
# Hyperparameters and Global Config
# -------------------
SAMPLE_RATE = 16000  # Must match your Phase 3 preprocessing
NUM_CLASSES = 6      # Adjust based on your categories
EPOCHS = int(os.environ.get("SOUNDSIGHT_EPOCHS", 20))
BATCH_SIZE = int(os.environ.get("SOUNDSIGHT_BATCH_SIZE", 16))
LEARNING_RATE = 0.001

# Path settings
DATA_ROOT = "data_preprocessed"  # Folder containing category subfolders
MODEL_DIR = "models"             # Where we'll save the trained model
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -------------------
# Dataset Class
# -------------------
class SoundDataset(Dataset):
    """
    A Dataset that loads preprocessed audio from folder structure:
      data_preprocessed/<class_name>/*.wav
    Creates a Mel Spectrogram tensor for each audio file.
    """

    def __init__(self, data_root: str, sample_rate: int, transform=None):
        super().__init__()
        self.data_root = data_root
        self.sample_rate = sample_rate
        self.transform = transform

        # Get subfolders as categories
        self.categories = sorted([
            d for d in os.listdir(self.data_root)
            if os.path.isdir(os.path.join(self.data_root, d))
        ])

        # Map category to index
        self.class_to_idx = {cat: i for i, cat in enumerate(self.categories)}
        print(f"Detected categories: {self.class_to_idx}")

        # Gather all audio file paths with labels
        self.audio_files = []
        for cat in self.categories:
            cat_path = os.path.join(self.data_root, cat)
            for file in glob.glob(os.path.join(cat_path, "*.wav")):
                self.audio_files.append((file, cat))

        # Shuffle data
        random.shuffle(self.audio_files)

    def __len__(self):
        return len(self.audio_files)

    def __getitem__(self, idx: int):
        file_path, cat_name = self.audio_files[idx]
        label_idx = self.class_to_idx[cat_name]

        # Load audio (waveform)
        waveform, sr = torchaudio.load(file_path)
        # Convert stereo to mono if needed
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)

        # Resample if necessary
        if sr != self.sample_rate:
            resampler = T.Resample(orig_freq=sr, new_freq=self.sample_rate)
            waveform = resampler(waveform)

        # Create Mel Spectrogram
        mel_transform = T.MelSpectrogram(
            sample_rate=self.sample_rate,
            n_mels=64,
            n_fft=1024,
            hop_length=512
        )
        mel_spectrogram = mel_transform(waveform)  # shape: [1, 64, time_frames]

        # Convert amplitude to dB
        db_transform = T.AmplitudeToDB(top_db=80)
        mel_db = db_transform(mel_spectrogram)  # shape: [1, 64, time_frames]

        if self.transform:
            mel_db = self.transform(mel_db)

        # Return spectrogram and label
        # We'll consider mel_db as a "image" of shape [1, freq_bins, time_frames]
        return mel_db, label_idx

# -------------------
# CNN Model
# -------------------
class SoundCNN(nn.Module):
    """
    A simple CNN for audio classification via Mel Spectrogram input
    """

    def __init__(self, num_classes: int):
        super(SoundCNN, self).__init__()
        # Input shape: [Batch, 1, 64, T]
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        self.pool1 = nn.MaxPool2d(kernel_size=2)  # Reduces freq to 32, time to T/2

        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        self.pool2 = nn.MaxPool2d(kernel_size=2)  # Reduces freq to 16, time to T/4

        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(64)
        self.pool3 = nn.MaxPool2d(kernel_size=2)  # Reduces freq to 8, time to T/8

        # Adaptive pooling to ensure a fixed size regardless of input
        self.adaptive_pool = nn.AdaptiveAvgPool2d((8, 8))  # [Batch, 64, 8, 8]

        # Fully connected layers
        self.fc1 = nn.Linear(64 * 8 * 8, 128)  # Adjusted to match adaptive pool
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        # x: [B, 1, 64, T]
        x = self.pool1(torch.relu(self.bn1(self.conv1(x))))  # [B,16,32,T/2]
        x = self.pool2(torch.relu(self.bn2(self.conv2(x))))  # [B,32,16,T/4]
        x = self.pool3(torch.relu(self.bn3(self.conv3(x))))  # [B,64,8,T/8]
        x = self.adaptive_pool(x)                            # [B,64,8,8]

        # Flatten
        x = x.view(x.size(0), -1)  # [B, 64*8*8=4096]
        x = torch.relu(self.fc1(x))  # [B, 128]
        x = self.fc2(x)              # [B, num_classes]
        return x

# -------------------
# Train and Validation Functions
# -------------------
def train_one_epoch(model, dataloader, criterion, optimizer, device=DEVICE):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for i, (inputs, labels) in enumerate(dataloader):
        inputs = inputs.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)
        _, predicted = torch.max(outputs, 1)
        correct += (predicted == labels).sum().item()
        total += labels.size(0)

    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc

def validate(model, dataloader, criterion, device=DEVICE):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for i, (inputs, labels) in enumerate(dataloader):
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)

            loss = criterion(outputs, labels)
            running_loss += loss.item() * inputs.size(0)

            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc

# -------------------
# Main Training Routine
# -------------------
def main():
    # 1. Prepare the dataset
    dataset = SoundDataset(DATA_ROOT, SAMPLE_RATE)

    # 2. Split into train/val (80/20 for example)
    total_len = len(dataset)
    val_len = int(0.2 * total_len)
    train_len = total_len - val_len

    train_data, val_data = torch.utils.data.random_split(dataset, [train_len, val_len])

    train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_data, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

    # 3. Create model, loss, optimizer
    model = SoundCNN(num_classes=NUM_CLASSES).to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # 4. Training loop
    best_val_acc = 0.0
    os.makedirs(MODEL_DIR, exist_ok=True)

    for epoch in range(EPOCHS):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, DEVICE)
        val_loss, val_acc = validate(model, val_loader, criterion, DEVICE)

        print(f"Epoch [{epoch+1}/{EPOCHS}] | "
              f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} | "
              f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")

        # Save model if validation improves
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            model_path = os.path.join(MODEL_DIR, "best_sound_model.pth")
            torch.save(model.state_dict(), model_path)
            print(f"  Model improved. Saved to {model_path}")

    print("Training complete. Best Val Acc = {:.4f}".format(best_val_acc))

if __name__ == "__main__":
    main()
