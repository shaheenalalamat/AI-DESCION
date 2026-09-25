# AI_DECISION_ResNet50_Benchmark.md

Decision ID: AI-DEC-010

Architecture Name: ResNet50 (Benchmark Re-evaluation)

Decision: 
Re-evaluate ResNet50 alongside the modified DeepLabV3 and CNN+LSTM architectures using the exact same hyperparameters (LR=0.0003) for a fair baseline comparison.

Status: 
Evaluated 📊 (Consistent & Highly Competitive)

Training Parameters & Configuration:
- Optimizer: Adam
- Learning Rate: 0.0003
- Epochs: 43 (Early Stopping triggered. Subset: 100 Train, 50 Val, Image Size: 256x256)
- Loss Function: MSE Loss
- Augmentations: RandomResizedCrop, RandomRotation (15°), ColorJitter, AddGaussianNoise

Performance Metrics:
- Best Val Loss: 0.2531 (Achieved at Epoch 30)
- Final Train Loss: 0.0246 (At Epoch 40)
- MAE IVSd (Septal Wall): ± 0.1913
- MAE LVIDd (LV Dimension): ± 0.5721
- MAE LVPWd (Posterior Wall): ± 0.1366

Reason: 
ResNet50 demonstrated remarkable stability, training deeper into the epochs (43 epochs) without immediate catastrophic overfitting. It achieved the best overall error for the Posterior Wall (LVPWd = 0.1366) across all models in this trial, remaining a top-tier contender for the final full-scale training.

Date: 
2026-09-24

Impact: 
- Confirms ResNet50's robustness. The final choice for the 11,578 dataset training will be a strategic decision between ResNet50's stability and DeepLabV3_Improved's slightly better handling of the LVIDd dimension.