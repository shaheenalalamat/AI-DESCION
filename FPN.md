# AI_DECISION_FPN.md

Decision ID: AI-DEC-002

Architecture Name: FPN (Feature Pyramid Network with ResNet50 backbone)

Decision: 
Reject FPN architecture for the core echocardiogram keypoint regression pipeline.

Status: 
Rejected ❌

Training Parameters & Configuration:
- Optimizer: Adam
- Learning Rate: 0.001
- Epochs: 15 (Sanity Check subset: 100 Train, 50 Val, Image Size: 256x256)
- Loss Function: Weighted MSE (alpha = 0.001)

Performance Metrics:
- Final Train Loss: 0.001198
- Final Val Loss: 0.001198 (Flatline / Saturation)

Reason: 
The model experienced early training saturation (flatline) at a relatively high loss value starting from the second epoch without continuous convergence or proper learning progression on Gaussian heatmaps.

Evidence: 
- Empirical Sanity Check training logs (15 epochs on subset data)
- Direct observation of loss stagnation

Date: 
2026-09-20

Impact: 
- Excluded FPN from full-dataset training candidates
- Shifted focus to alternative segmentation architectures