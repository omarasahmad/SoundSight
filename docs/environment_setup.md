# SoundSight Environment Setup

## Python Environment
- Using Python 3.9 with a venv
- Libraries: librosa, pydub, torch, etc.

## AR Setup
- **Unity** version 2021.3 LTS
  - AR Foundation, ARCore XR Plugin, ARKit XR Plugin

## CI
- GitHub Actions for Python linting
- Additional workflows planned for Unity builds

## Notes
- For local dev, ensure:
  - `venv/` activated
  - `.gitignore` includes environment folders
- For AR dev, run minimal test scene to confirm plugin installation.
