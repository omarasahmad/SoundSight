#!/usr/bin/env python3
"""
main.py

Provides a command-line interface (CLI) for SoundSight.
Usage examples:

  # Train the model
  python main.py train --epochs 30

  # Run real-time microphone inference
  python main.py infer

Requirements:
  - Must be run within the 'SoundSight' folder
  - The venv must be activated (unless you package it as an executable)
"""

import argparse
import sys
import os

# We will import the functions we need directly from your existing scripts.
# Adjust the paths as needed if you reorganize your code.
from train_model import main as train_main
from real_time_inference import main as infer_main

def parse_args():
    parser = argparse.ArgumentParser(
        description="SoundSight CLI - Train model, run inference, etc."
    )

    subparsers = parser.add_subparsers(dest="command", help="Commands")

    # ---- Subparser for TRAIN ----
    train_parser = subparsers.add_parser("train", help="Train the sound classification model")
    train_parser.add_argument("--epochs", type=int, default=20, help="Number of training epochs")
    train_parser.add_argument("--batch_size", type=int, default=16, help="Training batch size")

    # ---- Subparser for INFER ----
    infer_parser = subparsers.add_parser("infer", help="Run real-time microphone inference")

    return parser.parse_args()

def main():
    args = parse_args()

    if args.command == "train":
        # We can pass arguments to train_model.py via environment variables or directly modify its code
        print(f"Starting training with epochs={args.epochs}, batch_size={args.batch_size}")
        # One approach: override EPOCHS or BATCH_SIZE in train_model.py if you made them global.
        # Alternatively, refactor train_model.py to accept arguments in a function call.
        os.environ["SOUNDSIGHT_EPOCHS"] = str(args.epochs)
        os.environ["SOUNDSIGHT_BATCH_SIZE"] = str(args.batch_size)
        train_main()

    elif args.command == "infer":
        print("Starting real-time inference...")
        infer_main()

    else:
        print("No valid command provided.")
        sys.exit(1)

if __name__ == "__main__":
    main()
