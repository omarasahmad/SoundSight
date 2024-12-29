# Data Pipeline for SoundSight

## Raw Data
- Located in `data/`, organized by category folders.
- Some files from ESC-50, others self-recorded.

## Preprocessing
1. Resample to 16kHz
2. Convert stereo → mono
3. Trim leading/trailing silence

## Output
- Stored in `data_preprocessed/`
