# AI_DECISION_DeepLabV3_Improved.md

Decision ID: AI-DEC-008

Architecture Name: DeepLabV3 (with Custom Smart Regression Head & Adaptive Pooling)

Decision: 
Evaluate the modified DeepLabV3 architecture designed to preserve spatial features before regression, solving the "bottleneck" issue of the previous global pooling approach.

Status: 
Evaluated ✅ (Successfully Improved LVIDd Prediction)

Training Parameters & Configuration:
- Optimizer: Adam
- Learning Rate: 0.0003
- Epochs: 25 (Early Stopping triggered. Subset: 100 Train, 50 Val, Image Size: 256x256)
- Loss Function: MSE Loss
- Augmentations: RandomResizedCrop, RandomRotation (15°), ColorJitter, AddGaussianNoise

Performance Metrics:
- Best Val Loss: 0.2249 (Achieved at Epoch 10)
- Final Train Loss: 0.2293 (At Epoch 25)
- MAE IVSd (Septal Wall): ± 0.1947
- MAE LVIDd (LV Dimension): ± 0.5664
- MAE LVPWd (Posterior Wall): ± 0.1655

Reason: 
The custom architectural modification (Smart Regression Head) successfully reduced the error margin for the most challenging metric, LVIDd, bringing it down to ± 0.5664 (compared to ± 0.5971 in the baseline DeepLabV3). This proves that preserving a 2x2 spatial grid before flattening retains crucial geometrical context for large chamber measurements.

Date: 
2026-09-24

Impact: 
- Validates that DeepLabV3 can be highly effective for Direct Regression when equipped with an appropriate spatial-preserving head.