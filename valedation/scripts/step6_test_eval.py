"""
Final test evaluation of the validation gate (run ONCE, threshold fixed before looking at the test split).

- uses only the 'test' rows of gate_dataset_v2/manifest.csv (60 valid + 60 invalid images)
- model = step4/gate.onnx (checked against step4_summary.json), threshold = exp3/exp3_summary.json
- same preprocessing as the API (numpy), no torch needed
- nothing is tuned here: the threshold is not changed after seeing these results

Re-run safe: if test_eval/ already has the results it only compares and prints OK or CHANGED.
"""
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import onnxruntime as ort
import pandas as pd
from PIL import Image

# ----------------------------- CONFIG ---------------------------------------
DATASET_DIR = Path("gate_dataset_v2")
ONNX_PATH = Path("step4/gate.onnx")
STEP4_SUMMARY = Path("step4/step4_summary.json")
EXP3_SUMMARY = Path("exp3/exp3_summary.json")
OUT_DIR = Path("test_eval")
REFERENCE_THRESHOLD = 0.5      # reported only, next to the chosen one
# -----------------------------------------------------------------------------

MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)


def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def prep(rgb):
    im = Image.fromarray(rgb).resize((224, 224), Image.BILINEAR)
    x = (np.asarray(im, np.float32) / 255.0 - MEAN) / STD
    return np.ascontiguousarray(x.transpose(2, 0, 1)[None])


def wilson(k, n, z=1.96):
    """95% Wilson interval for a proportion k/n."""
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return max(0.0, c - h), min(1.0, c + h)


def metrics(prob, y, thr):
    pred = (prob >= thr).astype(int)
    tp = int(((pred == 1) & (y == 1)).sum())
    tn = int(((pred == 0) & (y == 0)).sum())
    fp = int(((pred == 1) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum())
    lo, hi = wilson(tp + tn, len(y))
    return dict(threshold=thr, tp=tp, tn=tn, fp=fp, fn=fn, correct=tp + tn, n=int(len(y)),
                accuracy=(tp + tn) / len(y), accuracy_ci95=[round(lo, 4), round(hi, 4)])


def main():
    m = pd.read_csv(DATASET_DIR / "manifest.csv")
    te = m[m.split == "test"].reset_index(drop=True)
    assert len(te) == 120 and int(te.label.sum()) == 60, "unexpected test split size"
    tv = set(te[te.label == 1].source_video)
    for s in ("train", "val"):
        other = set(m[(m.split == s) & (m.label == 1)].source_video)
        assert tv.isdisjoint(other), f"LEAKAGE: a test video is also in {s}"
    print(f"test: {len(te)} images ({int(te.label.sum())} valid from {len(tv)} videos, {int((te.label == 0).sum())} invalid)")

    assert sha256(ONNX_PATH) == json.load(open(STEP4_SUMMARY))["onnx_sha256"], "gate.onnx is not the one recorded in Step 4"
    thr = float(json.load(open(EXP3_SUMMARY))["chosen_threshold"])
    print(f"gate.onnx matches step4_summary.json | threshold from Exp 3: {thr}")

    opts = ort.SessionOptions()
    opts.intra_op_num_threads = 1
    s = ort.InferenceSession(str(ONNX_PATH), sess_options=opts, providers=["CPUExecutionProvider"])
    iname, oname = s.get_inputs()[0].name, s.get_outputs()[0].name
    prob = []
    for f in te.file:
        with Image.open(DATASET_DIR / f) as im:
            prob.append(float(s.run([oname], {iname: prep(np.asarray(im.convert("RGB")))})[0].ravel()[0]))
    prob = np.array(prob)
    y = te.label.to_numpy()

    pred = te[["file", "label", "kind", "source_video"]].copy()
    pred["prob"] = prob
    pred["pred"] = (prob >= thr).astype(int)
    pred["correct"] = (pred.pred == pred.label).astype(int)

    main_m, ref_m = metrics(prob, y, thr), metrics(prob, y, REFERENCE_THRESHOLD)
    per_kind = pred.groupby("kind")["correct"].agg(["sum", "count"]).rename(columns={"sum": "correct", "count": "n"})
    summary = dict(
        note="single evaluation on the test split, threshold fixed before (Exp 3), nothing tuned here",
        chosen=main_m, reference=ref_m,
        per_kind={k: dict(correct=int(r.correct), n=int(r.n)) for k, r in per_kind.iterrows()},
        highest_invalid_prob=float(prob[y == 0].max()), lowest_valid_prob=float(prob[y == 1].min()),
        sha256=dict(onnx=sha256(ONNX_PATH), manifest_csv=sha256(DATASET_DIR / "manifest.csv"),
                    exp3_summary=sha256(EXP3_SUMMARY)))

    pp, sp = OUT_DIR / "test_predictions.csv", OUT_DIR / "test_summary.json"
    if pp.exists() and sp.exists():
        old = pd.read_csv(pp)
        same = list(old.file) == list(pred.file) and np.allclose(old.prob, pred.prob, atol=1e-6) \
            and json.loads(sp.read_text()) == json.loads(json.dumps(summary))
        print("OK: same results as the saved ones" if same else "CHANGED: results differ from the saved ones (nothing overwritten)")
        return
    OUT_DIR.mkdir(exist_ok=True)
    pred.to_csv(pp, index=False)
    sp.write_text(json.dumps(summary, indent=2))

    for name, r in (("chosen threshold", main_m), ("reference", ref_m)):
        print(f"\n{name} {r['threshold']}: accuracy {r['correct']}/{r['n']} = {100 * r['accuracy']:.2f}% "
              f"(95% CI {100 * r['accuracy_ci95'][0]:.1f}-{100 * r['accuracy_ci95'][1]:.1f}%)")
        print(f"  TP {r['tp']}  TN {r['tn']}  FP {r['fp']} (invalid passed)  FN {r['fn']} (valid rejected)")
    print("\naccuracy per kind at the chosen threshold:")
    print(per_kind.to_string())
    print(f"\nhighest invalid prob {summary['highest_invalid_prob']:.4f} | lowest valid prob {summary['lowest_valid_prob']:.4f} | threshold {thr}")
    wrong = pred[pred.correct == 0]
    if len(wrong):
        print("\nwrongly classified at the chosen threshold:")
        print(wrong[["file", "label", "prob"]].to_string(index=False))
    print(f"\nOK: saved test_predictions.csv and test_summary.json in {OUT_DIR}/")


if __name__ == "__main__":
    main()
