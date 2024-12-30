# SoundSight

SoundSight is a PC-based sound classification system that uses a Convolutional Neural Network (CNN) to classify sounds in real-time. This project is designed to recognize sounds such as sirens, footsteps, laughter, and more, leveraging machine learning and audio data processing.

## Table of Contents
1. [Project Overview](#project-overview)
2. [Features](#features)
3. [Dependencies](#dependencies)
4. [Setup Instructions](#setup-instructions)
5. [Usage Instructions](#usage-instructions)
   - [Train the Model](#train-the-model)
   - [Run Real-Time Inference](#run-real-time-inference)
6. [Data Preprocessing Pipeline](#data-preprocessing-pipeline)
7. [Training Details](#training-details)
8. [Real-Time Inference](#real-time-inference)
9. [Future Improvements](#future-improvements)
10. [Contributing](#contributing)
11. [License](#license)

---

## Project Overview

SoundSight provides a command-line interface (CLI) for:
- Training a CNN-based model on custom audio datasets.
- Running real-time sound classification using a PC microphone.

The project is designed for easy customization and extensibility, making it suitable for both research and practical applications.

---

## Features

- **Custom Sound Classification**: Train on user-defined sound categories.
- **Real-Time Inference**: Classify sounds in real-time from a PC microphone.
- **Preprocessing Tools**: Automatically process audio data for training.
- **Command-Line Interface**: Simple commands to train models and run inference.

---

## Dependencies

The project relies on Python 3.8+ and the following libraries:
- `torch`
- `torchaudio`
- `librosa`
- `sounddevice`
- `numpy`

All dependencies are listed in `requirements.txt`.

---

## Setup Instructions

1. **Clone the Repository**
   ```bash
   git clone https://github.com/omarasahmad/SoundSight.git
   cd SoundSight
   ```

2. **Set Up a Virtual Environment**
   ```bash
   python -m venv venv
   source venv/bin/activate      # On macOS/Linux
   venv\Scripts\activate       # On Windows
   ```

3. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

---

## Usage Instructions

### Train the Model

1. Prepare your dataset in `data/` with subfolders for each category (e.g., `data/siren/`, `data/footsteps/`, etc.). Ensure your audio files are preprocessed or run preprocessing (see below).
2. Train the model:
   ```bash
   python src/python/main.py train --epochs 20 --batch_size 16
   ```
3. The trained model will be saved to `models/best_sound_model.pth`.

### Run Real-Time Inference

1. Ensure your microphone is connected.
2. Run the real-time inference script:
   ```bash
   python src/python/main.py infer
   ```
3. Speak or play audio near the microphone, and the system will output the predicted category with confidence scores.

---

## Data Preprocessing Pipeline

SoundSight includes tools for preprocessing audio data:
- **Resampling**: Converts all audio to a consistent sample rate (e.g., 16kHz).
- **Silence Trimming**: Removes leading and trailing silence.
- **Mel Spectrogram Generation**: Converts waveforms to Mel Spectrograms for model training.

Preprocessing is automated via `preprocessing.py`. To preprocess your dataset:
```bash
python src/python/preprocessing.py
```
The output will be saved in `data_preprocessed/`.

---

## Training Details

The model uses a Convolutional Neural Network (CNN) with the following architecture:
1. **Input**: Mel Spectrograms generated from audio files.
2. **Convolutional Layers**: Three layers with ReLU activation and max-pooling.
3. **Fully Connected Layers**: Two layers, culminating in the final classification layer.
4. **Loss Function**: Cross-Entropy Loss.
5. **Optimizer**: Adam.

### Hyperparameters
- Batch size: `16`
- Learning rate: `0.001`
- Epochs: `20` (configurable via CLI)

Training progress, including loss and accuracy, is displayed during execution.

---

## Real-Time Inference

The `real_time_inference.py` script:
1. Captures 1-second audio chunks from the microphone.
2. Converts each chunk to a Mel Spectrogram.
3. Classifies the sound using the trained model.
4. Prints the detected category and confidence score.

Example output:
```
Detected: siren (85.4% confidence)
Detected: footsteps (72.1% confidence)
```

---

## Future Improvements

- **Model Optimization**: Experiment with deeper CNNs or transfer learning.
- **Multi-Label Classification**: Support overlapping sound categories.
- **Cross-Platform Inference**: Explore Docker for improved portability.
- **Augmented Training Data**: Add noise, pitch shifts, or other augmentation.

---

## Contributing

Contributions are welcome! To contribute:
1. Fork the repository.
2. Create a feature branch.
3. Submit a pull request.

Please ensure your changes are well-documented and tested.

---

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.
