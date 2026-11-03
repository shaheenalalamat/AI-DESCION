"""
Step 2: Model Architecture (COR-LVH AI Validation gate)

- Load MobileNetV3-Small from torchvision (ImageNet pretrained)
- Replace the 1000-class layer with a 1-node Linear layer + Sigmoid (binary classification)

The model OUTPUTS A PROBABILITY in (0, 1): close to 1 = valid echo, close to 0 = invalid.
Because Sigmoid is inside the model, use nn.BCELoss in Step 3 (not BCEWithLogitsLoss).

Run this file directly to self-test it on 1 valid + 1 invalid image from the dataset of Step 1.
"""
import sys
from pathlib import Path

import pandas as pd
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as T
from PIL import Image

# ----------------------------- CONFIG ---------------------------------------
DATASET_DIR = Path("gate_dataset_v2")   # output folder of Step 1
INPUT_SIZE = 224                        # MobileNetV3 input size
PRETRAINED = True                       # needs internet the first time (downloads ImageNet weights)
# -----------------------------------------------------------------------------

MEAN, STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]   # ImageNet statistics


def build_model(pretrained=True):
    weights = models.MobileNet_V3_Small_Weights.DEFAULT if pretrained else None
    model = models.mobilenet_v3_small(weights=weights)

    # classifier = [Linear(576,1024), Hardswish, Dropout, Linear(1024,1000)]
    last = model.classifier[-1]
    assert isinstance(last, nn.Linear) and last.out_features == 1000, "unexpected classifier layout"
    model.classifier[-1] = nn.Linear(last.in_features, 1)     # 1000 classes -> 1 node
    model.classifier.append(nn.Sigmoid())                     # probability in (0, 1)
    return model


def make_eval_transform(size=INPUT_SIZE):
    """Same preprocessing to use for val / test / inference (no augmentation)."""
    return T.Compose([T.Resize((size, size)), T.ToTensor(), T.Normalize(MEAN, STD)])


def count_params(model):
    total = sum(p.numel() for p in model.parameters())
    train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, train


def self_test():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)

    model = build_model(PRETRAINED).to(device).eval()
    total, train = count_params(model)
    print(f"parameters: {total:,} total | {train:,} trainable")
    print("head:", model.classifier)

    tf = make_eval_transform()

    # shape check on a random batch
    with torch.no_grad():
        out = model(torch.randn(4, 3, INPUT_SIZE, INPUT_SIZE, device=device))
    assert out.shape == (4, 1), f"expected (4, 1), got {tuple(out.shape)}"
    assert ((out > 0) & (out < 1)).all(), "output is not a probability in (0, 1)"
    print("random batch OK, output shape:", tuple(out.shape))

    # forward on real images (1 valid + 1 invalid) from Step 1
    mpath = DATASET_DIR / "manifest.csv"
    if not mpath.exists():
        print(f"{mpath} not found, skipping the real-image test (set DATASET_DIR).")
        return
    m = pd.read_csv(mpath)
    for label, name in ((1, "valid"), (0, "invalid")):
        row = m[(m.split == "train") & (m.label == label)].iloc[0]
        with Image.open(DATASET_DIR / row.file) as im:
            x = tf(im.convert("RGB")).unsqueeze(0).to(device)
        with torch.no_grad():
            p = model(x).item()
        print(f"{name:7s} image ({row.file}): output = {p:.4f}")

    print("\nOK. The model is untrained for this task, so the outputs above are NOT meaningful yet.")
    print("They only prove the forward pass works and gives a number between 0 and 1.")


if __name__ == "__main__":
    try:
        self_test()
    except AssertionError as e:
        sys.exit(f"CHECK FAILED: {e}")

