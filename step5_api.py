"""
Step 5: FastAPI Inference Integration (COR-LVH AI Validation gate)

POST /validate  (multipart, field "file" = echo video, .avi)
  1. extract the center frame
  2. pass it to the ONNX model (step4/gate.onnx)
  3. score >= THRESHOLD  -> 200 with the score (the frame is accepted, the next stage takes over)
  4. score <  THRESHOLD  -> HTTP 400 (image rejected)
  unreadable video       -> HTTP 400 too, with a different "error" code
GET /health  -> model loaded + threshold

The frame goes through exactly the same steps as the "midframe" set of Exp 2:
frame n//2 -> RGB -> resize 256x256 (INTER_AREA) -> resize 224 (PIL bilinear) -> ImageNet normalization.

Run from the project folder:   uvicorn step5_api:app --port 8000
Interactive page:              http://127.0.0.1:8000/docs
"""
import os
import tempfile
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort
from fastapi import FastAPI, File, HTTPException, UploadFile
from PIL import Image

# ----------------------------- CONFIG ---------------------------------------
ONNX_PATH = Path(os.environ.get("GATE_ONNX", "step4/gate.onnx"))
THRESHOLD = 0.1782          # from Exp 3 (exp3/exp3_summary.json), not the 0.6 written on the slide
IMG_SIZE = 256              # same as the dataset images
INPUT_SIZE = 224            # model input
# -----------------------------------------------------------------------------

MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)

opts = ort.SessionOptions()
opts.intra_op_num_threads = 1       # measured in Step 4: 1 thread was faster for this small model
session = ort.InferenceSession(str(ONNX_PATH), sess_options=opts, providers=["CPUExecutionProvider"])
IN_NAME = session.get_inputs()[0].name
OUT_NAME = session.get_outputs()[0].name

app = FastAPI(title="COR-LVH validation gate")


def center_frame(video_path):
    cap = cv2.VideoCapture(str(video_path))
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if n < 1:
        cap.release()
        return None, 0, 0
    idx = n // 2
    cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
    ok, frame = cap.read()
    cap.release()
    return (frame if ok else None), idx, n


def to_input(frame_bgr):
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    rgb = cv2.resize(rgb, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA)
    im = Image.fromarray(rgb).resize((INPUT_SIZE, INPUT_SIZE), Image.BILINEAR)
    x = (np.asarray(im, np.float32) / 255.0 - MEAN) / STD
    return np.ascontiguousarray(x.transpose(2, 0, 1)[None])


def score_frame(frame_bgr):
    return float(session.run([OUT_NAME], {IN_NAME: to_input(frame_bgr)})[0].ravel()[0])


@app.get("/health")
def health():
    return {"model": ONNX_PATH.name, "threshold": THRESHOLD}


@app.post("/validate")
async def validate(file: UploadFile = File(...)):
    data = await file.read()
    if not data:
        raise HTTPException(400, detail={"error": "empty_file"})
    fd, tmp = tempfile.mkstemp(suffix=Path(file.filename or "x.avi").suffix or ".avi")
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(data)
        frame, idx, n = center_frame(tmp)
    finally:
        os.remove(tmp)
    if frame is None:
        raise HTTPException(400, detail={"error": "unreadable_video"})
    s = score_frame(frame)
    if s < THRESHOLD:
        raise HTTPException(400, detail={"error": "invalid_image", "score": round(s, 4), "threshold": THRESHOLD})
    return {"valid": True, "score": round(s, 4), "threshold": THRESHOLD, "center_frame": idx, "frame_count": n}

