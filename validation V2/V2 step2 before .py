"""
V2.0 / step 2: "before" measurement. How many out-of-distribution images does the CURRENT gate accept?

- images: gate_dataset_v2_ood (all of them are invalid, so every accepted image is a false positive)
- models: step4/gate.onnx (first model) and step4_final/gate_final.onnx (final model, the one in the API),
  each checked against its step4_summary.json
- thresholds: 0.1782 (Exp 3, the one in the API) and 0.5 for reference
- nothing is trained or tuned here. Same numpy preprocessing as the API.

Re-run safe: if v2_before/ already has the results it only compares and prints OK or CHANGED.
"""
import hashlib
import json
from pathlib import Path

import numpy as np
import onnxruntime as ort
import pandas as pd
from PIL import Image

# ----------------------------- CONFIG ---------------------------------------
OOD_DIR = Path("gate_dataset_v2_ood")
MODELS = {
    "first": (Path("step4/gate.onnx"), Path("step4/step4_summary.json")),
    "final": (Path("step4_final/gate_final.onnx"), Path("step4_final/step4_summary.json")),
}
EXP3_SUMMARY = Path("exp3/exp3_summary.json")
OUT_DIR = Path("v2_before")
REFERENCE_THRESHOLD = 0.5
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


def score(onnx_path, files):
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = 1
    s = ort.InferenceSession(str(onnx_path), sess_options=opts, providers=["CPUExecutionProvider"])
    iname, oname = s.get_inputs()[0].name, s.get_outputs()[0].name
    out = []
    for f in files:
        with Image.open(OOD_DIR / f) as im:
            out.append(float(s.run([oname], {iname: prep(np.asarray(im.convert("RGB")))})[0].ravel()[0]))
    return np.array(out)


def main():
    m = pd.read_csv(OOD_DIR / "manifest.csv")
    assert (m.label == 0).all(), "the ood manifest must only have invalid images"
    thr = float(json.load(open(EXP3_SUMMARY))["chosen_threshold"])
    print(f"{len(m)} ood images | threshold from Exp 3: {thr}")

    probs, summary = {}, {"threshold": thr, "reference_threshold": REFERENCE_THRESHOLD, "models": {}}
    for name, (onnx_path, step4_summary) in MODELS.items():
        assert sha256(onnx_path) == json.load(open(step4_summary))["onnx_sha256"], f"{onnx_path} is not the one recorded in {step4_summary}"
        p = score(onnx_path, m.file)
        probs[name] = p
        res = {}
        for sp, g in m.assign(prob=p).groupby("split"):
            res[sp] = dict(n=int(len(g)), accepted_at_thr=int((g.prob >= thr).sum()),
                           accepted_at_0_5=int((g.prob >= REFERENCE_THRESHOLD).sum()),
                           p_min=round(float(g.prob.min()), 4), p_mean=round(float(g.prob.mean()), 4),
                           p_max=round(float(g.prob.max()), 4))
        res["all"] = dict(n=int(len(m)), accepted_at_thr=int((p >= thr).sum()),
                          accepted_at_0_5=int((p >= REFERENCE_THRESHOLD).sum()),
                          p_min=round(float(p.min()), 4), p_mean=round(float(p.mean()), 4), p_max=round(float(p.max()), 4))
        summary["models"][name] = res
        print(f"\n{name} model ({onnx_path.name}): ood images accepted by the gate (should be 0)")
        print(pd.DataFrame(res).T[["n", "accepted_at_thr", "accepted_at_0_5", "p_min", "p_mean", "p_max"]].to_string())

    table = m[["file", "split", "kind", "source_file"]].copy()
    for name, p in probs.items():
        table[f"prob_{name}"] = p
    summary["sha256"] = {"manifest_csv": sha256(OOD_DIR / "manifest.csv"),
                         "exp3_summary": sha256(EXP3_SUMMARY),
                         **{f"onnx_{k}": sha256(v[0]) for k, v in MODELS.items()}}

    pp, sp_ = OUT_DIR / "probs.csv", OUT_DIR / "summary.json"
    if pp.exists() and sp_.exists():
        old = pd.read_csv(pp)
        same = list(old.file) == list(table.file) \
            and all(np.allclose(old[f"prob_{k}"], table[f"prob_{k}"], atol=1e-6) for k in MODELS) \
            and json.loads(sp_.read_text()) == json.loads(json.dumps(summary))
        print("\nOK: same results as the saved ones" if same else "\nCHANGED: results differ from the saved ones (nothing overwritten)")
        return
    OUT_DIR.mkdir(exist_ok=True)
    table.to_csv(pp, index=False)
    sp_.write_text(json.dumps(summary, indent=2))

    worst = table.sort_values("prob_final", ascending=False).head(10)
    print("\n10 ood images with the highest probability (final model), look at them:")
    print(worst[["file", "prob_first", "prob_final"]].to_string(index=False))
    print(f"\nOK: saved probs.csv and summary.json in {OUT_DIR}/")


if __name__ == "__main__":
    main()

