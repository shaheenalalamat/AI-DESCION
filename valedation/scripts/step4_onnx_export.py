"""
Step 4: Optimization & ONNX Export (slide: convert .pth to ONNX, measure latency in milliseconds)

1. checks best.pth is the checkpoint recorded in exp1_summary.json
2. exports step4/gate.onnx (input 1x3x224x224, dynamic batch, output = probability of 'valid')
3. checks the ONNX file (onnx.checker) and compares ONNX Runtime with PyTorch on every image of Exp 2
   (val, middle frames, blurred): max probability difference and number of decision flips
4. checks that the numpy preprocessing (the one the API will use) equals the torchvision one used in training
5. measures latency on CPU, batch 1: preprocessing, model, and total

Needs: torch, onnx, onnxruntime, onnxscript, step2_model.py, exp1/, exp2/, exp3/.
Re-run safe: an existing gate.onnx is not re-exported, and the summary is written only once.
"""
import hashlib
import json
import platform
import os
import time
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
EXP3_DIR = Path("exp3")
SETS_DIR = Path("exp2/sets")
OUT_DIR = Path("step4")
ONNX_PATH = OUT_DIR / "gate.onnx"
OPSET = 17
PARITY_TOL = 1e-4          # max allowed |prob_torch - prob_onnx|
PREP_TOL = 1e-4            # max allowed difference numpy preprocessing vs torchvision
WARMUP, RUNS = 20, 300     # latency measurement
# -----------------------------------------------------------------------------

MEAN_A = np.array(MEAN, np.float32)
STD_A = np.array(STD, np.float32)


def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def prep(rgb):
    """RGB uint8 array (H, W, 3) -> float32 (1, 3, 224, 224). This is the preprocessing for the API (Step 5)."""
    im = Image.fromarray(rgb).resize((INPUT_SIZE, INPUT_SIZE), Image.BILINEAR)
    x = (np.asarray(im, np.float32) / 255.0 - MEAN_A) / STD_A
    return np.ascontiguousarray(x.transpose(2, 0, 1)[None])


def load_rgb(path):
    with Image.open(path) as im:
        return np.asarray(im.convert("RGB"))


def stats_ms(t):
    t = np.asarray(t) * 1000
    return dict(median=round(float(np.median(t)), 3), mean=round(float(t.mean()), 3),
                p95=round(float(np.percentile(t, 95)), 3), min=round(float(t.min()), 3),
                max=round(float(t.max()), 3))


def export(model):
    dummy = torch.randn(2, 3, INPUT_SIZE, INPUT_SIZE)
    names = dict(input_names=["input"], output_names=["prob"])
    try:        # attempt 1: classic exporter
        torch.onnx.export(model, dummy, str(ONNX_PATH), opset_version=OPSET, dynamo=False,
                          dynamic_axes={"input": {0: "batch"}, "prob": {0: "batch"}}, **names)
        return "torchscript exporter"
    except Exception as e1:
        print("attempt 1 (dynamo=False) failed:", repr(e1)[:300])
    try:        # attempt 2: default exporter of this torch version
        ONNX_PATH.unlink(missing_ok=True)
        torch.onnx.export(model, (dummy,), str(ONNX_PATH), opset_version=OPSET, external_data=False,
                          dynamic_shapes={"x": {0: torch.export.Dim("batch")}}, **names)
        return "dynamo exporter"
    except Exception as e2:
        raise SystemExit(f"both export attempts failed.\nattempt 2: {repr(e2)[:500]}")


def main():
    import onnx
    import onnxruntime as ort

    OUT_DIR.mkdir(exist_ok=True)
    torch.set_grad_enabled(False)

    # 1) checkpoint identity
    s1 = json.load(open(EXP1_DIR / "exp1_summary.json"))
    h_pth = sha256(EXP1_DIR / "best.pth")
    assert h_pth == s1["sha256"]["best_pth"], "best.pth is not the checkpoint recorded in exp1_summary.json"
    print("best.pth sha256 matches exp1_summary.json:", h_pth[:16])
    ck = torch.load(EXP1_DIR / "best.pth", map_location="cpu", weights_only=False)
    model = build_model(pretrained=False)
    model.load_state_dict(ck["model"])
    model.eval()

    # 2) export
    if ONNX_PATH.exists():
        how = "existing file, not re-exported"
        print(f"{ONNX_PATH} exists, using it")
    else:
        how = export(model)
        print(f"exported with the {how}")
    onnx.checker.check_model(onnx.load(str(ONNX_PATH)))
    print("onnx.checker: OK")
    assert not list(OUT_DIR.glob("*.data")), "weights were saved in a separate .data file; the API needs one single .onnx file"
    h_onnx, size_mb = sha256(ONNX_PATH), ONNX_PATH.stat().st_size / 1e6
    print(f"gate.onnx: {size_mb:.2f} MB, sha256 {h_onnx[:16]}")

    sess = ort.InferenceSession(str(ONNX_PATH), providers=["CPUExecutionProvider"])
    in_name, out_name = sess.get_inputs()[0].name, sess.get_outputs()[0].name
    print("input:", sess.get_inputs()[0].shape, "output:", sess.get_outputs()[0].shape)

    # 3) parity on every image of Exp 2 (torchvision preprocessing, batches of 8 -> also tests the dynamic batch)
    m = pd.read_csv(DATASET_DIR / "manifest.csv")
    val = m[m.split == "val"]
    sm = pd.read_csv(SETS_DIR / "exp2_sets_manifest.csv")
    items = [("val_clean", DATASET_DIR / f) for f in val.file] + [(r.set, SETS_DIR / r.file) for r in sm.itertuples()]
    tf = T.Compose([T.Resize((INPUT_SIZE, INPUT_SIZE)), T.ToTensor(), T.Normalize(MEAN, STD)])
    rows, prep_diff = [], 0.0
    for i in range(0, len(items), 8):
        chunk = items[i:i + 8]
        xs = []
        for _, f in chunk:
            with Image.open(f) as im:
                xs.append(tf(im.convert("RGB")))
        x = torch.stack(xs)
        pt = model(x).numpy().ravel()
        po = sess.run([out_name], {in_name: x.numpy()})[0].ravel()
        for (name, f), a, b, xi in zip(chunk, pt, po, xs):
            prep_diff = max(prep_diff, float(np.abs(prep(load_rgb(f))[0] - xi.numpy()).max()))
            rows.append(dict(set=name, file=Path(f).name, prob_torch=float(a), prob_onnx=float(b), diff=abs(float(a) - float(b))))
    par = pd.DataFrame(rows)
    thr = json.load(open(EXP3_DIR / "exp3_summary.json"))["chosen_threshold"]
    max_diff = float(par["diff"].max())
    flips = {str(t): int(((par.prob_torch >= t) != (par.prob_onnx >= t)).sum()) for t in (0.5, thr)}
    print(f"\nparity on {len(par)} images: max |torch - onnx| = {max_diff:.2e}, decision flips {flips}")
    print(f"numpy preprocessing vs torchvision: max difference {prep_diff:.2e}")
    assert max_diff < PARITY_TOL, "ONNX output differs from PyTorch"
    assert all(v == 0 for v in flips.values()), "a decision changes between PyTorch and ONNX"
    assert prep_diff < PREP_TOL, "numpy preprocessing is not equal to the training preprocessing"
    par.to_csv(OUT_DIR / "step4_parity.csv", index=False)

    # 4) latency, batch 1, CPU, one real frame (256x256 RGB array, like a frame from cv2)
    frame = load_rgb(DATASET_DIR / val[val.label == 1].file.iloc[0])
    lat = {}
    for label, threads in (("default_threads", 0), ("1_thread", 1)):
        so = ort.SessionOptions()
        so.intra_op_num_threads = threads
        s = ort.InferenceSession(str(ONNX_PATH), sess_options=so, providers=["CPUExecutionProvider"])
        x = prep(frame)
        for _ in range(WARMUP):
            s.run([out_name], {in_name: x})
        t_prep, t_model, t_all = [], [], []
        for _ in range(RUNS):
            a = time.perf_counter()
            x = prep(frame)
            b = time.perf_counter()
            s.run([out_name], {in_name: x})
            c = time.perf_counter()
            t_prep.append(b - a); t_model.append(c - b); t_all.append(c - a)
        lat[label] = dict(preprocess_ms=stats_ms(t_prep), model_ms=stats_ms(t_model), total_ms=stats_ms(t_all))
    print(f"\nlatency, CPU, batch 1, {RUNS} runs after {WARMUP} warm-up (ms)")
    for label, d in lat.items():
        print(f"  {label}:")
        for k, v in d.items():
            print(f"    {k:14s} median {v['median']:8.3f} | mean {v['mean']:8.3f} | p95 {v['p95']:8.3f} | max {v['max']:8.3f}")
    if lat["default_threads"]["total_ms"]["median"] >= 1000:
        print("WARNING: median latency is not in the millisecond range")

    summary = dict(
        checkpoint_sha256=h_pth, onnx_sha256=h_onnx, onnx_size_mb=round(size_mb, 3), export=how, opset=OPSET,
        input=dict(name=in_name, shape=sess.get_inputs()[0].shape), output=dict(name=out_name, shape=sess.get_outputs()[0].shape),
        parity=dict(images=int(len(par)), max_abs_diff=max_diff, decision_flips=flips, tolerance=PARITY_TOL),
        numpy_vs_torchvision_preprocess_max_diff=prep_diff, threshold_from_exp3=thr, latency=lat,
        environment=dict(torch=torch.__version__, onnx=onnx.__version__, onnxruntime=ort.__version__,
                         python=platform.python_version(), os=platform.platform(), cpu_count=os.cpu_count(),
                         processor=platform.processor()))
    sp = OUT_DIR / "step4_summary.json"
    if sp.exists():
        print(f"\n{sp} already exists, not overwritten (latency differs a little on every run)")
    else:
        sp.write_text(json.dumps(summary, indent=2))
        print(f"\nOK: saved gate.onnx, step4_parity.csv, step4_summary.json in {OUT_DIR}/")


if __name__ == "__main__":
    main()

