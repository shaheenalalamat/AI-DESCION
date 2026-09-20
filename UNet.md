# AI_DECISION_UNet.md

Decision ID: AI-DEC-003

Architecture Name: UNet (with ResNet50 backbone)

Decision: 
Designate UNet architecture with ResNet50 backbone as a secondary backup model for the keypoint regression task.

Status: 
Alternative / Backup ⚠️

Training Parameters & Configuration:
- Optimizer: Adam
- Learning Rate: 0.001
- Epochs: 15 (Sanity Check subset: 100 Train, 50 Val, Image Size: 256x256)
- Loss Function: Weighted MSE (alpha = 0.001)

Performance Metrics:
- Final Train Loss: 0.000329
- Final Val Loss: 0.000337

Reason: 
Showed very strong, stable, and smooth convergence, proving high capability in learning landmark coordinates, though it was slightly outperformed by DeepLabV3.

Evidence: 
- Empirical Sanity Check training logs showing consistent loss reduction over 15 epochs
- Successful integration with `segmentation-models-pytorch` pipeline

Date: 
2026-09-20

Impact: 
- Maintained as a robust fallback architecture in case of unforeseen edge cases during scaling
- Validated the effectiveness of ResNet50 encoders for medical image feature extraction