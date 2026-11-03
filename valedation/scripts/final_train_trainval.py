"""
Final training of the validation gate on train + val (team decision: after the epoch count is fixed on val,
val is added to the training data so the model learns from it).

- same model, same settings and same loop as Exp 1 (Step 3): MobileNetV3-Small, AdamW lr 1e-4, wd 1e-4,
  batch 16, BCE loss, seed 42, no augmentation, ImageNet weights as the starting point
- EPOCHS = 5 is fixed from Exp 1 (best epoch on val); there is no validation any more, so no best-epoch choice:
  the weights after the last epoch are saved
- the TEST split is not read here, not even to check a number
- the threshold (0.1782, chosen in Exp 3 with the model that never saw val) is not touched here

Needs step2_model.py and gate_dataset_v2 in the same folder. Does not touch exp1/.
Not re-runnable on purpose: if exp_final/final.pth exists it stops (delete the folder to train again).
"""
import hashlib
import json
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from step2_model import build_model, make_eval_transform

# ----------------------------- CONFIG ---------------------------------------
DATASET_DIR = Path("gate_dataset_v2")
OUT_DIR = Path("exp_final")
EPOCHS = 5            # best epoch of Exp 1
BATCH_SIZE = 16
LR = 1e-4
WEIGHT_DECAY = 1e-4
SEED = 42
NUM_WORKERS = 0
# -----------------------------------------------------------------------------


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


class GateDataset(Dataset):
    def __init__(self, df, root, transform):
        self.files = df["file"].tolist()
        self.y = df["label"].to_numpy(np.float32)
        self.root = Path(root)
        self.tf = transform

    def __len__(self):
        return len(self.files)

    def __getitem__(self, i):
        with Image.open(self.root / self.files[i]) as im:
            x = self.tf(im.convert("RGB"))
        return x, torch.tensor([self.y[i]])


def main():
    if (OUT_DIR / "final.pth").exists():
        print(f"{OUT_DIR / 'final.pth'} already exists: not training again. Delete {OUT_DIR}/ if you really want to retrain.")
        return
    seed_everything(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)

    m = pd.read_csv(DATASET_DIR / "manifest.csv")
    tr = m[m.split.isin(["train", "val"])].reset_index(drop=True)
    te_videos = set(m[(m.split == "test") & (m.label == 1)].source_video)
    assert te_videos.isdisjoint(set(tr[tr.label == 1].source_video)), "LEAKAGE: a test video is in train/val"
    assert not set(tr.file) & set(m[m.split == "test"].file), "a test file is in the training data"
    assert len(tr) == 680 and int(tr.label.sum()) == 340, "unexpected size of train + val"
    print(f"train + val: {len(tr)} images ({int(tr.label.sum())} valid, {int((tr.label == 0).sum())} invalid); the test split is not loaded")

    loader = DataLoader(GateDataset(tr, DATASET_DIR, make_eval_transform()), batch_size=BATCH_SIZE,
                        shuffle=True, num_workers=NUM_WORKERS)
    model = build_model(pretrained=True).to(device)
    criterion = nn.BCELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    history = []
    for epoch in range(1, EPOCHS + 1):
        t0 = time.time()
        model.train()
        total, n, correct = 0.0, 0, 0
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            p = model(x)
            loss = criterion(p, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total += loss.item() * x.size(0)
            n += x.size(0)
            correct += int(((p.detach() >= 0.5).float() == y).sum().item())
        history.append(dict(epoch=epoch, train_loss=total / n, train_acc=correct / n))
        print(f"epoch {epoch}/{EPOCHS} | train loss {total / n:.4f} acc {correct / n:.3f} | {time.time() - t0:.0f}s")

    OUT_DIR.mkdir(exist_ok=True)
    torch.save({"model": model.state_dict(), "epoch": EPOCHS, "trained_on": "train+val", "input_size": 224},
               OUT_DIR / "final.pth")
    pd.DataFrame(history).to_csv(OUT_DIR / "history.csv", index=False)

    # fit check on the training images (NOT an evaluation: these images were trained on)
    model.eval()
    ps, ys = [], []
    with torch.no_grad():
        for x, y in DataLoader(GateDataset(tr, DATASET_DIR, make_eval_transform()), batch_size=32, num_workers=NUM_WORKERS):
            ps.append(model(x.to(device)).cpu().numpy().ravel())
            ys.append(y.numpy().ravel())
    ps, ys = np.concatenate(ps), np.concatenate(ys)
    fit = {"valid_prob_min": float(ps[ys == 1].min()), "invalid_prob_max": float(ps[ys == 0].max())}
    print(f"\nfit check on the training images (not an evaluation): lowest valid prob {fit['valid_prob_min']:.4f}, "
          f"highest invalid prob {fit['invalid_prob_max']:.4f}")

    json.dump({"trained_on": "train+val", "images": int(len(tr)), "epochs": EPOCHS, "batch_size": BATCH_SIZE, "lr": LR,
               "weight_decay": WEIGHT_DECAY, "seed": SEED, "torch": torch.__version__, "history": history,
               "fit_check_on_training_images": fit,
               "sha256": {"final_pth": sha256(OUT_DIR / "final.pth"), "manifest_csv": sha256(DATASET_DIR / "manifest.csv")},
               "note": "val is part of the training data: no held-out numbers here; the test split was not used"},
              open(OUT_DIR / "final_summary.json", "w"), indent=2)
    print(f"\nOK: saved final.pth, history.csv, final_summary.json in {OUT_DIR}/ (final.pth sha256 starts {sha256(OUT_DIR / 'final.pth')[:16]})")


if __name__ == "__main__":
    main()

