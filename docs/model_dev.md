# Model Development & Integration

## Steps Completed
1. Created a CNN-based classifier (train_model.py).
2. Trained the CNN model for 20 epochs, achieving a best validation accuracy of 94.29%.
3. Implemented real_time_inference.py for PC microphone input.
4. Implemented `AdaptiveAvgPool2d` to ensure consistent feature map sizes.

## Observations
- The model now trains without dimensionality errors.
- Validation accuracy improves over epochs, indicating effective learning.

## Next Steps
- Run `real_time_inference.py` to test the model in a real-time environment.
- Collect more data or apply advanced augmentation techniques to improve robustness.
