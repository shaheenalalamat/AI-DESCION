# AI_DECISION_DeepLabV3.md

Decision ID: AI-DEC-004

Architecture Name: DeepLabV3 (with ResNet50 backbone and atrous convolutions)

Decision: 
Select DeepLabV3 as the primary, official architecture for full-dataset keypoint regression.

Status: 
Confirmed / Selected ✅ (Gold Standard)

Training Parameters & Configuration:
- Optimizer: Adam
- Learning Rate: 0.001
- Epochs: 15 (Sanity Check subset: 100 Train, 50 Val, Image Size: 256x256)
- Loss Function: Weighted MSE (alpha = 0.001)

Performance Metrics:
- Final Train Loss: 0.000187
- Final Val Loss: 0.000184

Reason: 
Achieved the lowest error metrics among all tested models with clean, consistent convergence. Furthermore, it perfectly aligns with official EchoNet-LVH literature and the project's reference technical report[cite: 3].

Evidence: 
- Duffy et al., JAMA Cardiology 2022 (Official EchoNet-LVH paper)[cite: 3]
- Project reference technical report (`COR-LVH AI Architecture Recommendation`)[cite: 3]
- Empirical Sanity Check results showing superior performance and stability

Date: 
2026-09-20

Impact: 
- Selected as the definitive architecture for scaling up to the full dataset (11,578 videos/frames)[cite: 3]
- Proceeding to configure final training parameters (Early Stopping, Gaussian Augmentation)[cite: 3]