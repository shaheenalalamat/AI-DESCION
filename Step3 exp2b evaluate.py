"""
Step 3 / Exp 2 (part B): robustness evaluation of the Exp 1 model. No retraining, no plots.

Needs: step2_model.py, exp1/best.pth, exp1/exp1_summary.json, gate_dataset_v2, exp2/sets (part A).
Everything is on VAL data. The test split is not used.

Sets:
  val_clean              the 120 val images, must reproduce exp1/val_predictions.csv
  val_noise_0.03/0.05/0.10   same images + Gaussian noise (std on the 0-1 pixel scale, before normalization)
  midframe               middle frame of the val videos, expected valid
  blur_r2, blur_r5       mildly blurred real echo, no expected label (reported only)
  blur_r15, blur_r25     extremely blurred real echo, expected invalid
Saves exp2/exp2_results.csv, exp2/exp2_probs.csv (used by Exp 3), exp2/exp2_summary.json.
"""
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torchvision.transforms as T
from PIL import Image

from step2_model import INPUT_SIZE, MEAN, STD, build_model

# ----------------------------- CONFIG ---------------------------------------
DATASET_DIR = Path("gate_dataset_v2")
EXP1_DIR = Path("exp1")
SETS_DIR = Path("exp2/sets")
OUT_DIR = Path("exp2")
THRESHOLD = 0.5
NOISE_STDS = (0.03, 0.05, 0.10)
BATCH = 32
SEED = 42
# -----------------------------------------------------------------------------


def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def predict(model, files, noise_std=0.0, seed=SEED):
    """Probability of 'valid' for every file. Noise (if any) is added after ToTensor, before Normalize."""
    resize, to_t, norm = T.Resize((INPUT_SIZE, INPUT_SIZE)), T.ToTensor(), T.Normalize(MEAN, STD)
    g = torch.Generator().manual_seed(seed)
    out = []
    for i in range(0, len(files), BATCH):
        xs = []
        for f in files[i:i + BATCH]:
            with Image.open(f) as im:
                x = to_t(resize(im.convert("RGB")))
            if noise_std > 0:
                x = x + torch.randn(x.shape, generator=g) * noise_std
            xs.append(norm(x))
        with torch.no_grad():
            out.append(model(torch.stack(xs)).numpy().ravel())
    return np.concatenate(out)


def score(prob, expected):
    """expected: 1 valid, 0 invalid, -1 no label. Returns the metrics that make sense for the set."""
    pass_rate = float((prob >= THRESHOLD).mean())
    r = dict(n=len(prob), pass_rate=round(pass_rate, 4), p_min=round(float(prob.min()), 4),
             p_mean=round(float(prob.mean()), 4), p_max=round(float(prob.max()), 4))
    e = np.asarray(expected)
    lab = e >= 0
    if lab.any():
        pred = (prob >= THRESHOLD).astype(int)
        r["acc"] = round(float((pred[lab] == e[lab]).mean()), 4)
        r["fp"] = int(((pred == 1) & (e == 0)).sum())     # invalid image passed the gate
        r["fn"] = int(((pred == 0) & (e == 1)).sum())     # valid image rejected
    return r


def main():
    torch.manual_seed(SEED)
    summary1 = json.load(open(EXP1_DIR / "exp1_summary.json"))
    h = sha256(EXP1_DIR / "best.pth")
    assert h == summary1["sha256"]["best_pth"], "best.pth is not the checkpoint recorded in exp1_summary.json"
    print("best.pth sha256 matches exp1_summary.json:", h[:16])

    ck = torch.load(EXP1_DIR / "best.pth", map_location="cpu", weights_only=False)
    assert ck["input_size"] == INPUT_SIZE and ck["threshold"] == THRESHOLD
    model = build_model(pretrained=False)
    model.load_state_dict(ck["model"])
    model.eval()

    m = pd.read_csv(DATASET_DIR / "manifest.csv")
    val = m[m.split == "val"].reset_index(drop=True)
    sm = pd.read_csv(SETS_DIR / "exp2_sets_manifest.csv")
    val_files = [DATASET_DIR / f for f in val.file]

    probs = []          # long table: file, set, prob
    results = []        # one row per set

    def add(name, files, prob, expected, names):
        probs.append(pd.DataFrame(dict(file=names, set=name, prob=prob)))
        results.append(dict(set=name, **score(prob, expected)))

    # 1) clean val, must match Exp 1
    p0 = predict(model, val_files)
    ref = pd.read_csv(EXP1_DIR / "val_predictions.csv")
    assert list(ref.file) == list(val.file)
    diff = float(np.abs(p0 - ref.prob.to_numpy()).max())
    print(f"clean val vs exp1/val_predictions.csv: max abs diff {diff:.2e}")
    assert diff < 1e-4, "cannot reproduce Exp 1 predictions, stop here"
    add("val_clean", val_files, p0, val.label.to_numpy(), list(val.file))

    # 2) noise
    noise_kind = {}
    for s in NOISE_STDS:
        p = predict(model, val_files, noise_std=s)
        name = f"val_noise_{s:.2f}"
        add(name, val_files, p, val.label.to_numpy(), list(val.file))
        ok = ((p >= THRESHOLD).astype(int) == val.label.to_numpy())
        noise_kind[name] = pd.Series(ok).groupby(val.kind.to_numpy()).mean().round(4).to_dict()

    # 3) middle frames and blurred real echo
    for name, g in sm.groupby("set", sort=False):
        files = [SETS_DIR / f for f in g.file]
        assert all(f.exists() for f in files), f"missing images of {name}"
        add(name, files, predict(model, files), g.expected.to_numpy(), list(g.file))

    res = pd.DataFrame(results)
    allp = pd.concat(probs, ignore_index=True)
    res.to_csv(OUT_DIR / "exp2_results.csv", index=False)
    allp.to_csv(OUT_DIR / "exp2_probs.csv", index=False)

    pd.set_option("display.width", 200)
    print(f"\nthreshold {THRESHOLD}. pass_rate = share with prob >= threshold (accepted by the gate)")
    print(res.to_string(index=False))
    print("\naccuracy per image kind under noise:")
    print(pd.DataFrame(noise_kind).to_string())

    mid = allp[allp.set == "midframe"].sort_values("prob")
    print("\nlowest 5 middle-frame probabilities (valid expected):")
    print(mid.head(5).to_string(index=False))
    for r in (15, 25):
        b = allp[allp.set == f"blur_r{r}"].sort_values("prob", ascending=False)
        print(f"\nhighest 5 probabilities in blur_r{r} (invalid expected):")
        print(b.head(5).to_string(index=False))

    json.dump({"threshold": THRESHOLD, "noise_stds": list(NOISE_STDS), "seed": SEED,
               "torch": torch.__version__, "clean_val_max_abs_diff_vs_exp1": diff,
               "sha256": {"best_pth": h, "sets_manifest": sha256(SETS_DIR / "exp2_sets_manifest.csv"),
                          "manifest_csv": sha256(DATASET_DIR / "manifest.csv")},
               "results": results, "noise_accuracy_per_kind": noise_kind},
              open(OUT_DIR / "exp2_summary.json", "w"), indent=2)
    print(f"\nOK: saved exp2_results.csv, exp2_probs.csv, exp2_summary.json in {OUT_DIR}/")


if __name__ == "__main__":
    main()
