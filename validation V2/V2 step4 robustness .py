"""
V2.0 / step 4: robustness of a V2 model on the same sets as Exp 2. No retraining. VAL data only, the test split is not used.

Which model is checked:
  - by default MODEL_DIR below (exp_v2b = the model trained with noise)
  - or pass the folder name:  python V2_step4_robustness.py exp_v2
  Results go to <model folder>/robustness/, so exp_v2/robustness is never overwritten.

- model = <folder>/best.pth (checked against <folder>/v2_summary.json), threshold = the one chosen in step 3/3b (on val)
- sets (made in Exp 2, nothing is rebuilt here):
    val_clean                  the 120 val images of gate_dataset_v2 (must match <folder>/val_predictions.csv)
    val_noise_0.03/0.05/0.10   same images + Gaussian noise
    midframe                   middle frame of the 31 val videos = what the API sends, expected valid
    blur_r2, blur_r5           mildly blurred real echo, no expected label (only the number accepted is reported)
    blur_r15, blur_r25         extremely blurred real echo, expected invalid
- the threshold is not changed after seeing these results

Needs step2_model.py, gate_dataset_v2, exp2/sets and the model folder.
Re-run safe: if the results exist it only compares and prints OK or CHANGED.
"""
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torchvision.transforms as T
from PIL import Image

from step2_model import INPUT_SIZE, MEAN, STD, build_model

# ----------------------------- CONFIG ---------------------------------------
MODEL_DIR = "exp_v2c"      # default model folder (change here, or pass it as the first argument)
DATASET_DIR = Path("gate_dataset_v2")
SETS_DIR = Path("exp2/sets")
V2_DIR = Path(sys.argv[1] if len(sys.argv) > 1 else MODEL_DIR)
OUT_DIR = V2_DIR / "robustness"
NOISE_STDS = (0.03, 0.05, 0.10)
REFERENCE_THRESHOLD = 0.5
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
    """Probability of 'valid'. Noise (if any) is added after ToTensor, before Normalize (same as Exp 2)."""
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


def score(prob, expected, thr):
    """expected: 1 valid, 0 invalid, -1 no label."""
    e = np.asarray(expected)
    acc = prob >= thr
    r = dict(accepted=int(acc.sum()))
    if (e >= 0).any():
        r["fp"] = int((acc & (e == 0)).sum())      # invalid image accepted
        r["fn"] = int((~acc & (e == 1)).sum())     # valid image rejected
    return r


def main():
    assert V2_DIR.is_dir(), f"model folder not found: {V2_DIR}"
    torch.manual_seed(SEED)
    print(f"model folder: {V2_DIR}")
    summary1 = json.load(open(V2_DIR / "v2_summary.json"))
    rule = summary1["threshold_rule"]
    assert rule is not None, "step 3 found no threshold"
    thr = rule["chosen_threshold"]
    assert sha256(V2_DIR / "best.pth") == summary1["sha256"]["best_pth"], "best.pth is not the checkpoint recorded in v2_summary.json"
    print(f"best.pth matches v2_summary.json | threshold from training: {thr}")

    ck = torch.load(V2_DIR / "best.pth", map_location="cpu", weights_only=False)
    model = build_model(pretrained=False)
    model.load_state_dict(ck["model"])
    model.eval()

    m = pd.read_csv(DATASET_DIR / "manifest.csv")
    val = m[m.split == "val"].reset_index(drop=True)
    sm = pd.read_csv(SETS_DIR / "exp2_sets_manifest.csv")
    val_files = [DATASET_DIR / f for f in val.file]
    yv = val.label.to_numpy()

    probs, results = [], []

    def add(name, prob, expected, names):
        probs.append(pd.DataFrame(dict(file=names, set=name, prob=prob)))
        a, b = score(prob, expected, thr), score(prob, expected, REFERENCE_THRESHOLD)
        results.append(dict(set=name, n=len(prob), **{k: a[k] for k in a},
                            **{f"{k}_at_0_5": b[k] for k in b},
                            p_min=round(float(prob.min()), 4), p_mean=round(float(prob.mean()), 4), p_max=round(float(prob.max()), 4)))

    # 1) clean val, must match the training run
    p0 = predict(model, val_files)
    ref = pd.read_csv(V2_DIR / "val_predictions.csv").set_index("file").loc[list(val.file)]
    diff = float(np.abs(p0 - ref.prob.to_numpy()).max())
    print(f"clean val vs {V2_DIR}/val_predictions.csv: max abs diff {diff:.2e}")
    assert diff < 1e-4, "cannot reproduce the training-run predictions, stop here"
    add("val_clean", p0, yv, list(val.file))

    # 2) noise
    for s in NOISE_STDS:
        add(f"val_noise_{s:.2f}", predict(model, val_files, noise_std=s), yv, list(val.file))

    # 3) middle frames and blurred real echo
    for name, g in sm.groupby("set", sort=False):
        files = [SETS_DIR / f for f in g.file]
        assert all(f.exists() for f in files), f"missing images of {name}"
        add(name, predict(model, files), g.expected.to_numpy(), list(g.file))

    res = pd.DataFrame(results)
    for c in ("fp", "fn", "fp_at_0_5", "fn_at_0_5"):
        res[c] = res[c].astype("Int64")
    allp = pd.concat(probs, ignore_index=True)
    summary = {"model_dir": str(V2_DIR), "threshold": thr, "reference_threshold": REFERENCE_THRESHOLD,
               "noise_stds": list(NOISE_STDS), "seed": SEED,
               "clean_val_max_abs_diff_vs_training": diff, "results": json.loads(res.to_json(orient="records")),
               "sha256": {"best_pth": sha256(V2_DIR / "best.pth"), "sets_manifest": sha256(SETS_DIR / "exp2_sets_manifest.csv"),
                          "manifest_csv": sha256(DATASET_DIR / "manifest.csv")}}

    rp, pp, sp = OUT_DIR / "robustness_results.csv", OUT_DIR / "robustness_probs.csv", OUT_DIR / "robustness_summary.json"
    if sp.exists() and pp.exists():
        old = pd.read_csv(pp)
        same = list(old.file) == list(allp.file) and np.allclose(old.prob, allp.prob, atol=1e-5) \
            and json.loads(sp.read_text())["results"] == summary["results"]
        print("OK: same results as the saved ones" if same else "CHANGED: results differ from the saved ones (nothing overwritten)")
        return
    OUT_DIR.mkdir(exist_ok=True)
    res.to_csv(rp, index=False)
    allp.to_csv(pp, index=False)
    sp.write_text(json.dumps(summary, indent=2))

    pd.set_option("display.width", 220)
    print(f"\naccepted = prob >= {thr} | FP = invalid accepted | FN = valid rejected (labeled sets only)")
    print(res[["set", "n", "accepted", "fp", "fn", "accepted_at_0_5", "p_min", "p_mean", "p_max"]].to_string(index=False))
    mid = allp[allp.set == "midframe"].sort_values("prob")
    print("\nlowest 5 middle-frame probabilities (valid expected):")
    print(mid.head(5).to_string(index=False))
    for r in (15, 25):
        b = allp[allp.set == f"blur_r{r}"].sort_values("prob", ascending=False)
        print(f"\nhighest 5 probabilities in blur_r{r} (invalid expected):")
        print(b.head(5).to_string(index=False))
    print(f"\nOK: saved robustness_results.csv, robustness_probs.csv, robustness_summary.json in {OUT_DIR}/")


if __name__ == "__main__":
    main()