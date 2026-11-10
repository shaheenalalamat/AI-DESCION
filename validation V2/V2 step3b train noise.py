"""
V2.0 / step 3b: same training as step 3, plus light Gaussian noise during training (augmentation)

Why: step 3 gave FP 0 on val, but its robustness check (step 4) rejected echo with a little noise
(noise 0.05: 60 of 60 echo rejected, the first model rejected 0). Here every TRAIN image, valid or invalid, gets
Gaussian noise with probability NOISE_PROB and a random std in [0, MAX_NOISE] (pixel scale 0-1, added after ToTensor,
before Normalize, same place as in Exp 2). Both classes get it, so noise says nothing about the class.
Val stays clean. Everything else is identical to step 3 (same data, sampler, settings, best-epoch rule, threshold rule).

Written BEFORE running (decision rule):
  1. val at the chosen threshold: FP = 0 for every invalid kind (black, white, noise, blur, random_images, other_medical)
  2. step 4 (python v2_step4_robustness.py exp_v2b): middle frames 31 of 31 accepted, blur r15 and r25: 0 accepted,
     noise 0.03 and 0.05: FP = 0 and FN = 0 (what the first model did)
  If 1 or 2 fails, the result is reported as it is. The threshold is not changed afterwards. The test split is not read.
Note: noise 0.03 and 0.05 are inside the training range now, noise 0.10 is above it (MAX_NOISE = 0.08).

step3 (exp_v2) is not touched. Output goes to exp_v2b.
Run:  python v2_step3b_train_noise.py            (real run)
      python v2_step3b_train_noise.py --smoke    (1 epoch on ~100 images, only to check that the script runs)
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

import torchvision.transforms as T

from step2_model import INPUT_SIZE, MEAN, STD, build_model

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
OUT_DIR = Path("exp_v2c_smoke" if SMOKE else "exp_v2c")
MAX_NOISE = 0.05       # random noise std in [0, MAX_NOISE] on train images
NOISE_PROB = 0.5       # share of train images that get noise
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
    def __init__(self, df, noise_max=0.0, noise_prob=0.0):
        self.paths = df["path"].tolist()
        self.y = df["label"].to_numpy(np.float32)
        self.resize, self.to_t, self.norm = T.Resize((INPUT_SIZE, INPUT_SIZE)), T.ToTensor(), T.Normalize(MEAN, STD)
        self.noise_max, self.noise_prob = noise_max, noise_prob

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        with Image.open(self.paths[i]) as im:
            x = self.to_t(self.resize(im.convert("RGB")))
        if self.noise_max > 0 and random.random() < self.noise_prob:
            x = x + torch.randn_like(x) * random.uniform(0, self.noise_max)
        return self.norm(x), torch.tensor([self.y[i]])


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

    train_loader = DataLoader(GateDataset(tr, MAX_NOISE, NOISE_PROB), batch_size=BATCH_SIZE, sampler=balanced_sampler(tr.label),
                              num_workers=NUM_WORKERS)
    val_loader = DataLoader(GateDataset(va), batch_size=BATCH_SIZE * 2, shuffle=False, num_workers=NUM_WORKERS)
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
               "sampler": "class balanced (half valid, half invalid per epoch)",
               "noise_augmentation": {"max_std": MAX_NOISE, "prob": NOISE_PROB, "applies_to": "all train images, val stays clean"}, "best_epoch": ck["epoch"],
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
