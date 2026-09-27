# COR-LVH — AI Decision Log

This document records the major technical decisions made by the COR-LVH AI team.

The purpose of this file is to maintain a clear history of:
- What decision was made
- Why it was made
- What evidence supported it
- What impact it has on the project

Experiments are documented separately.

The relationship between experiments and decisions is:

Experiment
→ Evidence
→ Decision
→ Final Methodology

Old decisions must not be deleted.

If a decision is replaced by a newer decision, its status must be changed to:

SUPERSEDED

and the new decision must be documented as a new entry.

---

# Decision Statuses

The following statuses are used:

- CONFIRMED  
  The decision has been adopted by the team and is currently active.

- UNDER EVALUATION  
  The decision is currently being tested and has not yet been finalized.

- PLANNED  
  The decision represents the intended approach but still requires implementation or validation.

- OPEN  
  No final decision has been made yet.

- OUT OF SCOPE  
  The item is intentionally excluded from the current project scope.

- SUPERSEDED  
  The decision was previously active but has been replaced by a newer decision.

---

# AI-DEC-001 — Dataset Selection

**Decision:**  
Use EchoNet-LVH as the primary dataset for the COR-LVH AI pipeline.

**Status:**  
CONFIRMED

**Reason:**  
EchoNet-LVH contains PLAX echocardiography videos and the cardiac measurements required by COR-LVH, including IVSd, LVIDd, and LVPWd.

**Evidence:**  
Official access to EchoNet-LVH was obtained through Stanford AIMI / Redivis, and the dataset was successfully inspected and prepared by the AI team.

**Impact:**  
All current measurement-model training and evaluation are based on EchoNet-LVH.

**Date:**  
[DATE]

**Owner:**  
AI Team

---

# AI-DEC-002 — Target Measurements

**Decision:**  
Use the following three measurements as the primary AI targets:

- IVSd
- LVIDd
- LVPWd

**Status:**  
CONFIRMED

**Reason:**  
These measurements are required by the COR-LVH measurement-based LVH assessment pipeline.

**Impact:**  
The AI model must return all three measurements successfully before the backend continues to LV Mass and LVMI calculations.

**Date:**  
[DATE]

**Owner:**  
AI Team

---

# AI-DEC-003 — Measurement Units

**Decision:**  
Keep IVSd, LVIDd, and LVPWd measurements in centimeters (cm) throughout the current AI pipeline.

For academic comparison with studies reporting results in millimeters:

MAE_mm = MAE_cm × 10

**Status:**  
CONFIRMED

**Reason:**  
The inspected CalcValue measurements are treated as centimeters, and the backend LV Mass formula requires the measurements in centimeters.

**Impact:**  
Duplicate unit conversion must be avoided.

The future AI API must explicitly return the measurement unit.

**Date:**  
[DATE]

**Owner:**  
AI Team / Backend Team

---

# AI-DEC-004 — Dataset Cleaning

**Decision:**  
Exclude corrupted and unusable samples identified during Phase 1 dataset inspection.

**Status:**  
CONFIRMED

**Reason:**  
Several videos contained corrupted or shifted metadata that could produce invalid training data.

**Final Clean Dataset Size:**  
11,578 samples.

**Impact:**  
Excluded samples must not be reintroduced into later training or evaluation.

**Date:**  
[DATE]

**Owner:**  
AI Team

---

# AI-DEC-005 — Dataset Split

**Decision:**  
Preserve the train / validation / test split provided with the current EchoNet-LVH public dataset release.

**Status:**  
CONFIRMED

**Current Split After Cleaning:**

- Train: 10,109
- Validation: 1,132
- Test: 337

**Reason:**  
The dataset already provides an official split column.

Creating a new random split without evidence would introduce unnecessary changes to the dataset organization.

**Known Limitation:**  
Video-level overlap has been checked and is zero.

Patient-level separation has not been independently verified because the currently available metadata does not contain a documented independent PatientId or SubjectId.

**Impact:**  
The Test split must not be used for model tuning.

**Date:**  
[DATE]

**Owner:**  
AI Team

---

# AI-DEC-006 — Training Frame Selection

**Decision:**  
Use the dataset-provided annotated End-Diastolic frame for measurement-model training.

**Status:**  
CONFIRMED

**Reason:**  
The purpose of the current measurement-model experiment is to evaluate cardiac measurement estimation when the correct ED frame is already known.

**Important Limitation:**  
This decision does not solve automatic End-Diastole detection for new doctor-uploaded videos.

**Impact:**  
Automatic ED detection is handled separately in a later experiment.

**Date:**  
[DATE]

**Owner:**  
AI Team

---

# AI-DEC-007 — Image Size

**Decision:**  
Resize extracted ED frames to:

256 × 256 pixels.

**Status:**  
CONFIRMED

**Reason:**  
The original videos have different spatial resolutions.

A standardized input size is required for the current training pipeline.

Measurement coordinates were transformed using the corresponding width and height scaling ratios.

**Impact:**  
The current measurement-model pipeline expects 256 × 256 input images.

**Date:**  
[DATE]

**Owner:**  
AI Team

---

# AI-DEC-008 — Pixel Normalization

**Decision:**  
Normalize image pixel values dynamically during PyTorch data loading using:

pixel_value / 255

**Status:**  
CONFIRMED

**Reason:**  
Normalization can be applied during runtime and does not require storing an additional normalized image dataset.

**Impact:**  
Normalization must remain consistent during training and inference.

**Date:**  
[DATE]

**Owner:**  
AI Team

---

# AI-DEC-009 — Architecture Pilot

**Decision:**  
Evaluate multiple candidate architectures on a small controlled subset before full-dataset training.

**Status:**  
CONFIRMED

**Evaluated Architectures:**

- DeepLabV3 Improved
- ResNet50
- CNN + LSTM

**Reason:**  
A small pilot allows the team to compare model behavior and identify promising architectures before committing computational resources to full-dataset training.

**Impact:**  
Pilot results are considered screening evidence only and do not independently establish the final model.

**Date:**  
[DATE]

**Owner:**  
AI Team

---

# AI-DEC-010 — Previous Architecture Pilot

**Decision:**  
The previous pilot involving the earlier DeepLabV3, UNet, and FPN comparison is no longer used as the current architecture-selection basis.

**Status:**  
SUPERSEDED

**Reason:**  
The AI team later repeated the architecture comparison using a revised and more relevant group of models and updated experimental settings.

**Replaced By:**  
AI-DEC-009

**Impact:**  
Previous results remain preserved for historical documentation but should not be presented as the current model-selection evidence.

**Date:**  
[DATE]

**Owner:**  
AI Team

---

# AI-DEC-011 — DeepLabV3 Full-Dataset Training

**Decision:**  
Proceed with full-dataset training of the improved DeepLabV3 measurement architecture.

**Status:**  
UNDER EVALUATION

**Experiment:**  
EXP-M-001

**Training Outcome:**

- Maximum scheduled epochs: 150
- Training stopped: Epoch 52
- Early Stopping: Enabled
- Patience: 20
- Best checkpoint: Epoch 32
- Best Validation Loss: 0.0217
- Checkpoint file:
  `best_deeplabv3_exp_m_001.pth`

**Reason:**  
DeepLabV3 showed strong results during the architecture pilot and was selected for full-dataset evaluation.

**Important:**  
Completion of training does not automatically make DeepLabV3 the final COR-LVH measurement model.

Final confirmation depends on measurement evaluation, including:

- MAE_IVSd
- MAE_LVIDd
- MAE_LVPWd

**Impact:**  
The saved best checkpoint will be used for final EXP-M-001 evaluation.

**Date:**  
[DATE]

**Owner:**  
AI Team

---

# AI-DEC-012 — Automatic End-Diastole Detection Approach

**Decision:**  
Use an LVID temporal curve as the primary planned baseline for automatic End-Diastole detection from full PLAX videos.

**Status:**  
PLANNED / UNDER EVALUATION

**Intended Pipeline:**

PLAX Video  
→ Split into Frames  
→ Estimate LVID for each frame  
→ Build LVID temporal curve  
→ Apply smoothing  
→ Detect local peaks  
→ Identify ED candidate(s)

**Reason:**  
This approach allows ED detection to be based on temporal changes in LVID without immediately introducing a second dedicated neural network.

**Important Limitation:**  
The feasibility of applying the measurement model frame-by-frame to non-ED frames must first be experimentally verified.

**Impact:**  
This decision will be evaluated in EXP-ED-001.

**Date:**  
[DATE]

**Owner:**  
AI Team

---

# AI-DEC-013 — Multiple End-Diastole Candidates

**Decision:**  
The final strategy for handling multiple ED candidates in videos containing multiple cardiac cycles has not yet been selected.

**Status:**  
OPEN

**Possible Strategies:**

- Largest valid LVID peak
- Best candidate
- Representative cardiac cycle
- Mean measurement
- Median measurement
- Other evidence-based strategy

**Reason:**  
The final strategy must be selected based on experimental results rather than assumption.

**Impact:**  
No final multiple-heartbeat handling rule should currently be documented as confirmed.

**Date:**  
[DATE]

**Owner:**  
AI Team

---

# AI-DEC-014 — Automatic PLAX View Validation

**Decision:**  
Automatic confirmation that an uploaded video is a valid PLAX view is not currently part of the confirmed core AI pipeline.

**Status:**  
PLANNED / FEASIBILITY-DEPENDENT

**Reason:**  
The feature may improve input quality control but should not become a dependency that prevents completion of the core graduation-project objectives.

**Impact:**  
The system must still support technical file validation independently of automatic PLAX classification.

**Date:**  
[DATE]

**Owner:**  
AI Team

---

# AI-DEC-015 — AI Failure Handling

**Decision:**  
AI measurement failures must use explicit failure statuses and must not return fabricated measurement values.

**Status:**  
CONFIRMED

**Prohibited Example:**

- IVSd = -1
- LVIDd = -1
- LVPWd = -1

**Expected Conceptual Failure:**

status = failed

measurements = null

errorCode = appropriate failure code

**Reason:**  
Magic or sentinel measurement values may accidentally be interpreted as real cardiac measurements.

**Impact:**  
Required downstream calculations must not execute when required AI measurements are unavailable.

**Date:**  
[DATE]

**Owner:**  
AI Team / Backend Team

---

# Open AI Decisions

The following decisions remain open and must be resolved through experiments:

- Final measurement-model architecture
- Final EXP-M-001 measurement performance
- Final augmentation configuration
- Final optimizer and learning-rate strategy
- Final ED smoothing method
- Final peak-detection method
- Multiple-ED handling
- Final AI API contract
- Automatic PLAX validation feasibility
- Final full-video inference strategy

---

# Decision Change Rule

When a decision changes:

1. Do not delete the previous decision.
2. Change its status to:

SUPERSEDED

3. Add:

Superseded By: AI-DEC-XXX

4. Create a new decision entry containing:
   - The new decision
   - Reason for the change
   - Experimental evidence
   - Impact on the project