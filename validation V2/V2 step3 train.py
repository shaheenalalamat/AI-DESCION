"""
V2.0 / step 3: training of the gate with the out-of-distribution invalid images (slide: Phase 1 V2.0)

- train = gate_dataset_v2 train + gate_dataset_v2_ood train, val = same for val
- the TEST rows of both manifests are dropped when the manifests are read: they are never loaded here
- same model and settings as Exp 1: MobileNetV3-Small (ImageNet weights), AdamW lr 1e-4, wd 1e-4, batch 16,
  BCE loss, seed 42, 5 epochs, no augmentation
- the invalid class is now ~3.5x bigger than the valid one, so every epoch draws the images with a
  class-balanced sampler (about half valid, half invalid)
- best epoch = lowest class-balanced val loss (mean of the loss on valid images and on invalid images,
  because val has many more invalid images). Nothing else is used to choose it.
- threshold = rule of Exp 3 on val: middle of [highest invalid prob, lowest valid prob above it]
- the old model and the old API are not touched

Needs step2_model.py, gate_dataset_v2 and gate_dataset_v2_ood in the same folder.
Run:  python v2_step3_train.py            (real run, needs both kinds in the ood manifest)
      python v2_step3_train.py --smoke    (1 epoch on ~100 images, only to check that the script runs)
Not re-runnable on purpose: if OUT_DIR/best.pth exists it stops (delete the folder to train again).
"""
import hashlib
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

from step2_model import build_model, make_eval_transform

# ----------------------------- CONFIG ---------------------------------------
OLD_DIR = Path("gate_dataset_v2")
OOD_DIR = Path("gate_dataset_v2_ood")
REQUIRED_KINDS = ("random_images", "other_medical")   # a real run needs both; edit this line to train without one
EPOCHS = 5
BATCH_SIZE = 16
LR = 1e-4
WEIGHT_DECAY = 1e-4
SEED = 42
NUM_WORKERS = 0
PRETRAINED = True
SMOKE = "--smoke" in sys.argv
OUT_DIR = Path("exp_v2_smoke" if SMOKE else "exp_v2")
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
    def __init__(self, df, transform):
        self.paths = df["path"].tolist()
        self.y = df["label"].to_numpy(np.float32)
        self.tf = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        with Image.open(self.paths[i]) as im:
            x = self.tf(im.convert("RGB"))
        return x, torch.tensor([self.y[i]])


def load_manifests():
    old = pd.read_csv(OLD_DIR / "manifest.csv")
    ood = pd.read_csv(OOD_DIR / "manifest.csv")
    old["path"] = [str(OLD_DIR / f) for f in old.file]
    ood["path"] = [str(OOD_DIR / f) for f in ood.file]
    m = pd.concat([old, ood], ignore_index=True)
    return m[m.split.isin(["train", "val"])].reset_index(drop=True)      # test rows are dropped here


def balanced_sampler(y):
    y = np.asarray(y)
    w = np.where(y == 1, 0.5 / (y == 1).sum(), 0.5 / (y == 0).sum())
    g = torch.Generator().manual_seed(SEED)
    return WeightedRandomSampler(torch.tensor(w, dtype=torch.double), num_samples=len(y), replacement=True, generator=g)


def predict(model, loader, device):
    model.eval()
    out = []
    with torch.no_grad():
        for x, _ in loader:
            out.append(model(x.to(device)).cpu().numpy().ravel())
    return np.concatenate(out)


def balanced_loss(p, y):
    p = np.clip(p, 1e-7, 1 - 1e-7)
    l = -(y * np.log(p) + (1 - y) * np.log(1 - p))
    return float(0.5 * l[y == 1].mean() + 0.5 * l[y == 0].mean())


def counts(p, y, thr):
    acc = p >= thr
    return dict(fp=int((acc & (y == 0)).sum()), fn=int((~acc & (y == 1)).sum()))


def exp3_threshold(p, y):
    """Rule of Exp 3: smallest FN with FP = 0, threshold in the middle of the allowed interval."""
    lo = float(p[y == 0].max())
    above = p[(y == 1) & (p > lo)]
    if len(above) == 0:
        return None
    hi = float(above.min())
    return dict(highest_invalid_prob=lo, lowest_valid_prob_above_it=hi, margin=hi - lo,
                chosen_threshold=round((lo + hi) / 2, 4), fn_min=int(((p <= lo) & (y == 1)).sum()))


def main():
    if (OUT_DIR / "best.pth").exists():
        print(f"{OUT_DIR / 'best.pth'} already exists: not training again. Delete {OUT_DIR}/ if you really want to retrain.")
        return
    seed_everything(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device, "| smoke run" if SMOKE else "")

    m = load_manifests()
    assert "test" not in set(m.split), "a test row is in the data"
    if not SMOKE:
        missing = [k for k in REQUIRED_KINDS if k not in set(m.kind)]
        if missing:
            sys.exit(f"missing kinds in the ood manifest: {missing}. Build them first (v2_step1_build_ood_data.py), "
                     f"or edit REQUIRED_KINDS if you really want to train without them.")
    tr, va = m[m.split == "train"].reset_index(drop=True), m[m.split == "val"].reset_index(drop=True)
    assert set(tr.path).isdisjoint(set(va.path)), "same file in train and val"
    assert set(tr[tr.label == 1].source_video).isdisjoint(set(va[va.label == 1].source_video)), "LEAKAGE: a video is in train and val"
    assert tr.source_file.dropna().isin(set(va.source_file.dropna())).sum() == 0, "LEAKAGE: same ood source image in train and val"
    if SMOKE:
        tr = pd.concat([g.sample(min(len(g), 32), random_state=SEED) for _, g in tr.groupby("label")], ignore_index=True)
        va = pd.concat([g.sample(min(len(g), 16), random_state=SEED) for _, g in va.groupby("label")], ignore_index=True)
    print("train:", len(tr), "| val:", len(va))
    print("train per kind:", tr.groupby("kind").size().to_dict())
    print("val per kind:  ", va.groupby("kind").size().to_dict())

    tf = make_eval_transform()
    train_loader = DataLoader(GateDataset(tr, tf), batch_size=BATCH_SIZE, sampler=balanced_sampler(tr.label),
                              num_workers=NUM_WORKERS)
    val_loader = DataLoader(GateDataset(va, tf), batch_size=BATCH_SIZE * 2, shuffle=False, num_workers=NUM_WORKERS)
    yv = va.label.to_numpy(np.float32)

    model = build_model(pretrained=PRETRAINED).to(device)
    criterion = nn.BCELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    OUT_DIR.mkdir(exist_ok=True)
    epochs = 1 if SMOKE else EPOCHS
    history, best = [], float("inf")
    for epoch in range(1, epochs + 1):
        t0 = time.time()
        model.train()
        total, n = 0.0, 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            loss = criterion(model(x), y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total += loss.item() * x.size(0)
            n += x.size(0)
        pv = predict(model, val_loader, device)
        vl = balanced_loss(pv, yv)
        c = counts(pv, yv, 0.5)
        history.append(dict(epoch=epoch, train_loss=total / n, val_balanced_loss=vl, val_fp_at_0_5=c["fp"], val_fn_at_0_5=c["fn"],
                            highest_invalid_prob=float(pv[yv == 0].max()), lowest_valid_prob=float(pv[yv == 1].min())))
        flag = ""
        if vl < best:
            best = vl
            torch.save({"model": model.state_dict(), "epoch": epoch, "val_balanced_loss": vl, "input_size": 224},
                       OUT_DIR / "best.pth")
            flag = "  <- best"
        print(f"epoch {epoch}/{epochs} | train loss {total / n:.4f} | val balanced loss {vl:.4f} | at 0.5: FP {c['fp']} FN {c['fn']} | "
              f"highest invalid {history[-1]['highest_invalid_prob']:.4f} lowest valid {history[-1]['lowest_valid_prob']:.4f} | "
              f"{time.time() - t0:.0f}s{flag}")
    pd.DataFrame(history).to_csv(OUT_DIR / "history.csv", index=False)

    # val report with the best weights
    ck = torch.load(OUT_DIR / "best.pth", map_location=device, weights_only=False)
    model.load_state_dict(ck["model"])
    pv = predict(model, val_loader, device)
    va = va.copy()
    va["prob"] = pv
    va.drop(columns=["path"]).to_csv(OUT_DIR / "val_predictions.csv", index=False)
    rule = exp3_threshold(pv, yv)
    summary = {"smoke": SMOKE, "epochs": epochs, "batch_size": BATCH_SIZE, "lr": LR, "weight_decay": WEIGHT_DECAY, "seed": SEED,
               "sampler": "class balanced (half valid, half invalid per epoch)", "best_epoch": ck["epoch"],
               "best_val_balanced_loss": ck["val_balanced_loss"], "train_per_kind": tr.groupby("kind").size().to_dict(),
               "val_per_kind": va.groupby("kind").size().to_dict(), "threshold_rule": rule, "torch": torch.__version__,
               "sha256": {"best_pth": sha256(OUT_DIR / "best.pth"), "old_manifest": sha256(OLD_DIR / "manifest.csv"),
                          "ood_manifest": sha256(OOD_DIR / "manifest.csv")},
               "note": "test rows were never loaded; threshold and best epoch were chosen on val only"}
    print(f"\nBEST epoch {ck['epoch']} | val balanced loss {ck['val_balanced_loss']:.4f}")
    if rule is None:
        print("NO threshold: no valid val image has a probability above the highest invalid one. The gate does not separate on val.")
    else:
        thr = rule["chosen_threshold"]
        c = counts(pv, yv, thr)
        summary["at_chosen_threshold"] = c
        print(f"threshold (rule of Exp 3): {thr} | highest invalid {rule['highest_invalid_prob']:.4f} | "
              f"lowest valid above it {rule['lowest_valid_prob_above_it']:.4f} | margin {rule['margin']:.4f}")
        print(f"at {thr}: FP {c['fp']} (invalid accepted), FN {c['fn']} (valid rejected, minimum possible {rule['fn_min']})")
        print("\nval, per kind at the chosen threshold (invalid kinds: accepted; echo: rejected):")
        for kind, g in va.groupby("kind"):
            acc = int((g.prob >= thr).sum())
            print(f"  {kind:14s} n={len(g):4d}  " + (f"rejected {len(g) - acc}" if kind == "echo" else f"accepted {acc}"))
    json.dump(summary, open(OUT_DIR / "v2_summary.json", "w"), indent=2)
    print(f"\nOK: saved best.pth, history.csv, val_predictions.csv, v2_summary.json in {OUT_DIR}/ (the test split was not used)")


if __name__ == "__main__":
    main()

