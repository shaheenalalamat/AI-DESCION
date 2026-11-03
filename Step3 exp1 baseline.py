"""
Step 3 / Exp 1: Baseline Training (3-5 epochs, BCE Loss)

- trains on the 'train' split, evaluates on the 'val' split after every epoch
- the 'test' split is NOT touched here (kept for the final evaluation)
- saves the best weights (lowest val loss), the history, and a loss/accuracy curve

Needs step2_model.py in the same folder and gate_dataset_v2 from Step 1.
Label meaning: 1 = valid echo, 0 = invalid image.
"""
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
OUT_DIR = Path("exp1")
EPOCHS = 5            # slide: 3-5 epochs
BATCH_SIZE = 16
LR = 1e-4             # AdamW, same lr for the whole network (pretrained backbone + new head)
WEIGHT_DECAY = 1e-4
THRESHOLD = 0.5       # decision threshold for this baseline (tuned later in Exp 3)
SEED = 42
NUM_WORKERS = 0       # 0 is the safe choice on Windows / notebooks
# -----------------------------------------------------------------------------


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


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
        return x, torch.tensor([self.y[i]])          # target shape (1,) to match model output (B, 1)


def run_epoch(model, loader, criterion, device, optimizer=None):
    training = optimizer is not None
    model.train(training)
    total, n, probs, ys = 0.0, 0, [], []
    with torch.set_grad_enabled(training):
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            p = model(x)
            loss = criterion(p, y)
            if training:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            total += loss.item() * x.size(0)
            n += x.size(0)
            probs.append(p.detach().cpu().numpy().ravel())
            ys.append(y.cpu().numpy().ravel())
    return total / n, np.concatenate(probs), np.concatenate(ys)


def confusion(probs, ys, thr):
    pred = (probs >= thr).astype(int)
    y = ys.astype(int)
    tp = int(((pred == 1) & (y == 1)).sum())   # valid accepted
    tn = int(((pred == 0) & (y == 0)).sum())   # invalid rejected
    fp = int(((pred == 1) & (y == 0)).sum())   # invalid accepted  (bad image passes the gate)
    fn = int(((pred == 0) & (y == 1)).sum())   # valid rejected
    return dict(tp=tp, tn=tn, fp=fp, fn=fn, acc=(tp + tn) / len(y))


def main():
    seed_everything(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    OUT_DIR.mkdir(exist_ok=True)
    print("device:", device)

    # ---- data + integrity checks -------------------------------------------
    m = pd.read_csv(DATASET_DIR / "manifest.csv")
    tr, va = m[m.split == "train"].reset_index(drop=True), m[m.split == "val"].reset_index(drop=True)
    assert set(tr.file).isdisjoint(set(va.file)), "same file in train and val"
    tv, vv = set(tr[tr.label == 1].source_video), set(va[va.label == 1].source_video)
    assert tv.isdisjoint(vv), "LEAKAGE: a video is in both train and val"
    print(f"train: {len(tr)} (valid {int(tr.label.sum())}) | val: {len(va)} (valid {int(va.label.sum())})")

    tf = make_eval_transform()          # Resize 224 + ImageNet normalization, no augmentation
    train_loader = DataLoader(GateDataset(tr, DATASET_DIR, tf), batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=NUM_WORKERS)
    val_loader = DataLoader(GateDataset(va, DATASET_DIR, tf), batch_size=BATCH_SIZE * 2, shuffle=False,
                            num_workers=NUM_WORKERS)

    x0, y0 = next(iter(train_loader))
    assert x0.shape[1:] == (3, 224, 224) and y0.shape[1:] == (1,), (x0.shape, y0.shape)

    # ---- model / loss / optimizer ------------------------------------------
    model = build_model(pretrained=True).to(device)
    criterion = nn.BCELoss()            # model already ends with Sigmoid
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    # ---- training loop -----------------------------------------------------
    history, best_val = [], float("inf")
    for epoch in range(1, EPOCHS + 1):
        t0 = time.time()
        tr_loss, tr_p, tr_y = run_epoch(model, train_loader, criterion, device, optimizer)
        va_loss, va_p, va_y = run_epoch(model, val_loader, criterion, device)
        tr_acc = confusion(tr_p, tr_y, THRESHOLD)["acc"]
        c = confusion(va_p, va_y, THRESHOLD)
        history.append(dict(epoch=epoch, train_loss=tr_loss, val_loss=va_loss, train_acc=tr_acc,
                            val_acc=c["acc"], val_fp=c["fp"], val_fn=c["fn"]))
        flag = ""
        if va_loss < best_val:
            best_val = va_loss
            torch.save({"model": model.state_dict(), "epoch": epoch, "val_loss": va_loss,
                        "threshold": THRESHOLD, "input_size": 224}, OUT_DIR / "best.pth")
            flag = "  <- best"
        print(f"epoch {epoch}/{EPOCHS} | train loss {tr_loss:.4f} acc {tr_acc:.3f} | "
              f"val loss {va_loss:.4f} acc {c['acc']:.3f} | FP {c['fp']} FN {c['fn']} | "
              f"{time.time() - t0:.0f}s{flag}")

    pd.DataFrame(history).to_csv(OUT_DIR / "history.csv", index=False)

    # ---- final report on val with the best weights ---------------------------
    ck = torch.load(OUT_DIR / "best.pth", map_location=device, weights_only=False)
    model.load_state_dict(ck["model"])
    va_loss, va_p, va_y = run_epoch(model, val_loader, criterion, device)
    c = confusion(va_p, va_y, THRESHOLD)
    print(f"\nBEST epoch {ck['epoch']} | val loss {va_loss:.4f} | val acc {c['acc']:.4f} (threshold {THRESHOLD})")
    print("confusion on val (positive = valid echo):")
    print(f"                    predicted valid   predicted invalid")
    print(f"  actual valid      {c['tp']:>10}        {c['fn']:>10}   <- FN = good image rejected")
    print(f"  actual invalid    {c['fp']:>10}        {c['tn']:>10}   <- FP = bad image PASSED the gate")

    va = va.copy()
    va["prob"] = va_p
    va["pred"] = (va_p >= THRESHOLD).astype(int)
    va["correct"] = (va["pred"] == va["label"]).astype(int)
    print("\nval accuracy per image kind:")
    print(va.groupby("kind")["correct"].agg(["mean", "count"]).round(4).to_string())
    print("\nval probability range per class (valid should be near 1, invalid near 0):")
    print(va.groupby("label")["prob"].agg(["min", "mean", "max"]).round(4).to_string())
    va.to_csv(OUT_DIR / "val_predictions.csv", index=False)

    # ---- curves (optional) ---------------------------------------------------
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        h = pd.DataFrame(history)
        fig, ax = plt.subplots(1, 2, figsize=(10, 3.5))
        ax[0].plot(h.epoch, h.train_loss, marker="o", label="train")
        ax[0].plot(h.epoch, h.val_loss, marker="o", label="val")
        ax[0].set_title("BCE loss"); ax[0].set_xlabel("epoch"); ax[0].legend()
        ax[1].plot(h.epoch, h.train_acc, marker="o", label="train")
        ax[1].plot(h.epoch, h.val_acc, marker="o", label="val")
        ax[1].set_title("accuracy"); ax[1].set_xlabel("epoch"); ax[1].legend()
        fig.tight_layout()
        fig.savefig(OUT_DIR / "curves.png", dpi=120)
        print(f"\nsaved {OUT_DIR / 'curves.png'}")
    except Exception as e:
        print("curves not saved:", e)

    json.dump({"epochs": EPOCHS, "batch_size": BATCH_SIZE, "lr": LR, "weight_decay": WEIGHT_DECAY,
               "threshold": THRESHOLD, "seed": SEED, "best_epoch": ck["epoch"], "best_val_loss": ck["val_loss"],
               "val_confusion": c}, open(OUT_DIR / "exp1_summary.json", "w"), indent=2)
    print(f"saved best.pth, history.csv, val_predictions.csv, exp1_summary.json in {OUT_DIR}/")


if __name__ == "__main__":
    main()
