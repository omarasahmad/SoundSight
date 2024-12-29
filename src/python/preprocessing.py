import os
import librosa
import soundfile as sf

# Constants
INPUT_DIR = "data"
OUTPUT_DIR = "data_preprocessed"
CATEGORIES = ["Siren", "Car_Horn", "Laughing", "Footsteps", "Rain", "Wind"]
TARGET_SAMPLE_RATE = 16000  # Consistent sample rate for all audio files
DURATION = 5.0  # Expected duration of audio files (in seconds)


def preprocess_audio(input_path, output_path, sample_rate=TARGET_SAMPLE_RATE, duration=DURATION):
    """
    Preprocess a single audio file by resampling, converting to mono, and trimming silence.

    Args:
        input_path (str): Path to the input audio file.
        output_path (str): Path to save the preprocessed audio file.
        sample_rate (int): Target sample rate for resampling.
        duration (float): Target duration of the audio file in seconds.
    """
    try:
        # Load audio
        y, sr = librosa.load(input_path, sr=None, mono=True)
        print(f"Processing: {input_path}")

        # Resample to target sample rate if necessary
        if sr != sample_rate:
            y = librosa.resample(y, orig_sr=sr, target_sr=sample_rate)

        # Trim silence
        y, _ = librosa.effects.trim(y, top_db=30)

        # Ensure the audio length is exactly 'duration' seconds
        target_length = int(sample_rate * duration)
        if len(y) < target_length:
            # Pad if too short
            y = librosa.util.fix_length(y, target_length)
        else:
            # Trim if too long
            y = y[:target_length]

        # Save processed audio
        sf.write(output_path, y, sample_rate)
        print(f"Saved: {output_path}")

    except Exception as e:
        print(f"Error processing {input_path}: {e}")


def batch_preprocess(input_dir=INPUT_DIR, output_dir=OUTPUT_DIR, categories=CATEGORIES):
    """
    Preprocess all audio files in the specified categories.

    Args:
        input_dir (str): Path to the root directory containing the raw audio files.
        output_dir (str): Path to save the preprocessed audio files.
        categories (list): List of category names to process.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for category in categories:
        category_input_path = os.path.join(input_dir, category)
        category_output_path = os.path.join(output_dir, category)

        if not os.path.exists(category_output_path):
            os.makedirs(category_output_path)

        # Process all .wav files in the category folder
        for filename in os.listdir(category_input_path):
            if filename.endswith(".wav"):
                input_path = os.path.join(category_input_path, filename)
                output_path = os.path.join(category_output_path, filename)
                preprocess_audio(input_path, output_path)


if __name__ == "__main__":
    print("Starting batch preprocessing...")
    batch_preprocess()
    print("Batch preprocessing complete!")
