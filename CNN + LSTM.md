# AI_DECISION_CNN_LSTM.md

Decision ID: AI-DEC-009

Architecture Name: CNN + LSTM (ResNet18 Feature Extractor + Bidirectional LSTM)

Decision: 
Evaluate a novel approach treating single echocardiogram frames as a spatial sequence of patches, processed by a Bidirectional LSTM to capture contextual relationships across the heart's axis.

Status: 
Evaluated 🔬 (Strong Wall Detection / Weak Global Dimension)

Training Parameters & Configuration:
- Optimizer: Adam
- Learning Rate: 0.0003
- Epochs: 17 (Early Stopping triggered. Subset: 100 Train, 50 Val, Image Size: 256x256)
- Loss Function: MSE Loss
- Augmentations: RandomResizedCrop, RandomRotation (15°), ColorJitter, AddGaussianNoise

Performance Metrics:
- Best Val Loss: 0.2481 (Achieved at Epoch 05)
- Final Train Loss: 0.1195 (At Epoch 15)
- MAE IVSd (Septal Wall): ± 0.1882
- MAE LVIDd (LV Dimension): ± 0.6034
- MAE LVPWd (Posterior Wall): ± 0.1443

Reason: 
The sequence-based processing proved highly effective at identifying the distinct boundaries of the heart walls (IVSd and LVPWd achieved excellent MAEs of 0.1882 and 0.1443). However, the sequential nature struggled to capture the full global context required for the large inner chamber diameter (LVIDd error increased to 0.6034).

Date: 
2026-09-24

Impact: 
- Proves CNN+LSTM is viable for single-frame medical imaging but is better suited for localized feature extraction rather than global dimensional regression.