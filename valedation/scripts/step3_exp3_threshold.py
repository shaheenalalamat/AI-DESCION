"""
Step 3 / Exp 3: Threshold Tuning (slide: adjust to minimize False Negatives)

Uses only the probabilities saved by Exp 2 (exp2/exp2_probs.csv). No torch, no images, VAL data only.
Positive class = valid echo (same as Exp 1/2):
  FN = valid image rejected      FP = invalid (Class 0) image passed the gate

Labeled sets: val_clean, val_noise_*, midframe (valid) and blur_r15, blur_r25 (invalid).
Rule: the smallest number of FN such that no invalid image passes (FP = 0 on every labeled set).
FN never decreases when the threshold goes up, so the optimum is just above the highest invalid probability.
The chosen value is the middle of the allowed interval [highest invalid prob, lowest valid prob above it].
blur_r2 and blur_r5 have no label: only their pass count is reported.

Re-run safe: if exp3/ already has the outputs it only prints OK or CHANGED.
"""
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

# ----------------------------- CONFIG ---------------------------------------
PROBS = Path("exp2/exp2_probs.csv")
MANIFEST = Path("gate_dataset_v2/manifest.csv")
OUT_DIR = Path("exp3")
GRID = (0.01, 0.02, 0.05, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.99)
REPORT_AT = (0.5, 0.6)        # 0.5 = default used so far, 0.6 = value written on slide 5
# -----------------------------------------------------------------------------


def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def main():
    p = pd.read_csv(PROBS)
    m = pd.read_csv(MANIFEST)
    lab = dict(zip(m.file, m.label))

    def expected(row):
        if row.set.startswith("val_"):
            return lab[row.file]
        if row.set == "midframe":
            return 1
        if row.set in ("blur_r15", "blur_r25"):
            return 0
        return -1

    p["expected"] = [expected(r) for r in p.itertuples()]
    assert p.groupby("set").size().to_dict() == {
        "blur_r15": 60, "blur_r2": 60, "blur_r25": 60, "blur_r5": 60, "midframe": 31,
        "val_clean": 120, "val_noise_0.03": 120, "val_noise_0.05": 120, "val_noise_0.10": 120}, "unexpected set sizes"
    L = p[p.expected >= 0]
    sets = list(L.set.unique())

    def counts(t):
        row = {"threshold": t}
        for s in sets:
            g = L[L.set == s]
            pa = g.prob.to_numpy() >= t
            e = g.expected.to_numpy()
            row[f"{s}_fp"] = int((pa & (e == 0)).sum())
            row[f"{s}_fn"] = int((~pa & (e == 1)).sum())
        row["FP_total"] = sum(row[f"{s}_fp"] for s in sets)
        row["FN_total"] = sum(row[f"{s}_fn"] for s in sets)
        for s in ("blur_r2", "blur_r5"):
            row[f"{s}_pass_of_60"] = int((p[p.set == s].prob.to_numpy() >= t).sum())
        return row

    # exact optimum
    inv = L[L.expected == 0].prob.to_numpy()
    val_ = L[L.expected == 1].prob.to_numpy()
    lo = float(inv.max())                          # highest probability of an invalid image
    above = val_[val_ > lo]
    hi = float(above.min())                        # lowest valid probability above lo
    chosen = (lo + hi) / 2
    fn_min = int((val_ <= lo).sum())               # valid images that cannot be saved without letting an invalid one pass
    c = counts(chosen)
    assert c["FP_total"] == 0 and c["FN_total"] == fn_min, "optimum check failed"

    grid = pd.DataFrame([counts(t) for t in sorted(set(GRID) | {round(chosen, 4)})])
    summary = {
        "rule": "min FN subject to FP = 0 on all labeled sets (val only)",
        "highest_invalid_prob": lo, "lowest_valid_prob_above_it": hi, "margin": hi - lo,
        "chosen_threshold": round(chosen, 4), "FN_at_chosen": c["FN_total"], "FP_at_chosen": c["FP_total"],
        "labeled_images": {"valid": int(len(val_)), "invalid": int(len(inv))},
        "unlabeled_pass_at_chosen": {"blur_r2": c["blur_r2_pass_of_60"], "blur_r5": c["blur_r5_pass_of_60"]},
        "at_reference_thresholds": {str(t): {k: v for k, v in counts(t).items() if k in
                                    ("FP_total", "FN_total", "blur_r2_pass_of_60", "blur_r5_pass_of_60")}
                                    for t in REPORT_AT},
        "sha256": {"exp2_probs": sha256(PROBS), "manifest_csv": sha256(MANIFEST)},
    }

    table_path, json_path = OUT_DIR / "threshold_table.csv", OUT_DIR / "exp3_summary.json"
    if table_path.exists() and json_path.exists():
        # compare with line endings normalized: on Windows pandas writes CRLF, read_text() returns LF
        norm = lambda t: t.replace("\r\n", "\n")
        same = norm(table_path.read_text()) == norm(grid.to_csv(index=False)) and \
            json.loads(json_path.read_text()) == json.loads(json.dumps(summary))
        print("OK: same results as the saved ones" if same else "CHANGED: results differ from the saved ones (nothing overwritten)")
        return
    OUT_DIR.mkdir(exist_ok=True)
    grid.to_csv(table_path, index=False)
    json_path.write_text(json.dumps(summary, indent=2))

    pd.set_option("display.width", 200)
    show = grid[["threshold", "FP_total", "FN_total", "blur_r2_pass_of_60", "blur_r5_pass_of_60"]]
    print("labeled sets: FP = invalid passed, FN = valid rejected (summed over the 7 labeled sets)")
    print(show.to_string(index=False))
    print(f"\nhighest invalid prob {lo:.4f} | lowest valid prob above it {hi:.4f} | margin {hi - lo:.4f}")
    print(f"chosen threshold {chosen:.4f} -> FP {c['FP_total']}, FN {c['FN_total']}")
    for t in REPORT_AT:
        r = summary["at_reference_thresholds"][str(t)]
        print(f"threshold {t}: FP {r['FP_total']}, FN {r['FN_total']}")
    print(f"\nOK: saved {table_path} and {json_path}")


if __name__ == "__main__":
    main()
