# Data Pipeline for SoundSight

## Raw Data
- Located in `data/`, organized by category folders.
- Some files from UrbanSound8K, others self-recorded.

## Preprocessing
1. Resample to 16kHz
2. Convert stereo → mono
3. Trim leading/trailing silence
4. Loudness normalization (planned)

## Augmentation
- Random pitch shifts in [-2, 2] semitones
- Time stretch in [0.9, 1.1] range
- 50% probability to create augmented duplicates

## Output
- Stored in `data_preprocessed/`
- Each file has a name: `<original>_aug.wav` if augmented
- Metadata in `metadata.csv`
