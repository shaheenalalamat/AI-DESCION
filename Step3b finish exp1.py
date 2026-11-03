"""
Step 3 / Exp 1 (finish): writes curves.png and exp1_summary.json from the files
that the training run already saved. Does NOT import torch (avoids the OpenMP crash)
and does NOT retrain. Re-run safe: overwrites only curves.png and exp1_summary.json.
"""
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

DATASET_DIR = Path("gate_dataset_v2")
OUT_DIR = Path("exp1")
THRESHOLD = 0.5
CFG = dict(epochs=5, batch_size=16, lr=1e-4, weight_decay=1e-4, seed=42)   # same values as the training script


def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


for f in ("best.pth", "history.csv", "val_predictions.csv"):
    assert (OUT_DIR / f).exists(), f"missing {OUT_DIR / f}"
assert (DATASET_DIR / "manifest.csv").exists(), "missing manifest.csv"

h = pd.read_csv(OUT_DIR / "history.csv")
v = pd.read_csv(OUT_DIR / "val_predictions.csv")
assert len(h) == CFG["epochs"], f"history has {len(h)} epochs, expected {CFG['epochs']}"

best = h.loc[h.val_loss.idxmin()]
tp = int(((v.pred == 1) & (v.label == 1)).sum())
tn = int(((v.pred == 0) & (v.label == 0)).sum())
fp = int(((v.pred == 1) & (v.label == 0)).sum())
fn = int(((v.pred == 0) & (v.label == 1)).sum())
acc = (tp + tn) / len(v)
assert ((v.prob >= THRESHOLD).astype(int) == v.pred).all(), "pred column does not match prob and threshold"
assert abs(acc - best.val_acc) < 1e-9, "val accuracy differs from history at the best epoch"
assert fp == best.val_fp and fn == best.val_fn, "FP/FN differ from history at the best epoch"

fig, ax = plt.subplots(1, 2, figsize=(10, 3.5))
ax[0].plot(h.epoch, h.train_loss, marker="o", label="train")
ax[0].plot(h.epoch, h.val_loss, marker="o", label="val")
ax[0].set_title("BCE loss"); ax[0].set_xlabel("epoch"); ax[0].legend()
ax[1].plot(h.epoch, h.train_acc, marker="o", label="train")
ax[1].plot(h.epoch, h.val_acc, marker="o", label="val")
ax[1].set_title("accuracy"); ax[1].set_xlabel("epoch"); ax[1].legend()
fig.tight_layout()
fig.savefig(OUT_DIR / "curves.png", dpi=120)

summary = dict(CFG, threshold=THRESHOLD, best_epoch=int(best.epoch), best_val_loss=float(best.val_loss),
               val_confusion=dict(tp=tp, tn=tn, fp=fp, fn=fn, acc=acc),
               val_prob_range={str(k): [float(g.prob.min()), float(g.prob.mean()), float(g.prob.max())]
                               for k, g in v.groupby("label")},
               sha256=dict(best_pth=sha256(OUT_DIR / "best.pth"),
                           manifest_csv=sha256(DATASET_DIR / "manifest.csv")))
(OUT_DIR / "exp1_summary.json").write_text(json.dumps(summary, indent=2))

print(f"best epoch {summary['best_epoch']} | val loss {summary['best_val_loss']:.4f} | val acc {acc:.4f}")
print(f"TP {tp}  TN {tn}  FP {fp}  FN {fn}")
print("best.pth sha256:", summary["sha256"]["best_pth"][:16])
print("manifest.csv sha256:", summary["sha256"]["manifest_csv"][:16])
print("OK: saved curves.png and exp1_summary.json")
