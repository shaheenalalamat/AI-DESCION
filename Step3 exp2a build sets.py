"""
Step 3 / Exp 2 (part A): builds the robustness test sets. No torch here.

Sets (all from the VAL split only, the test split is not touched):
  midframe  : the middle frame of each of the val videos (what the API will send), expected valid
  blur_r*   : the 60 real val echo frames with Gaussian blur of radius 2, 5, 15, 25
              radius 15 and 25 are "extreme blur" (same range as Step 1, 15-30) -> expected invalid
              radius 2 and 5 are mild -> no expected label, only reported (helps Exp 3)
Noise is added later in part B on the tensors, so nothing is saved for it.

Re-run safe: if exp2/sets already exists it only compares and prints OK or CHANGED.
"""
import hashlib
import shutil
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from PIL import Image, ImageFilter

# ----------------------------- CONFIG ---------------------------------------
VIDEO_DIR = Path(r"C:/Users/rahma/OneDrive/Desktop/COR_LVH/EchoNet-LVH")
DATASET_DIR = Path("gate_dataset_v2")
OUT_DIR = Path("exp2/sets")
BLUR_RADII = (2, 5, 15, 25)
EXTREME_FROM = 15          # radius >= this is expected invalid
IMG_SIZE = 256
# -----------------------------------------------------------------------------


def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def to_square(frame_bgr):
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    return cv2.resize(rgb, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA)


def build(out_dir, val_valid, videos):
    rows = []

    # middle frame of every val video
    for name in sorted(val_valid.source_video.unique()):
        cap = cv2.VideoCapture(str(videos[name]))
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        idx = n // 2
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, fr = cap.read()
        cap.release()
        if not ok:
            sys.exit(f"could not read frame {idx} of {name}")
        g = cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY)
        rel = Path("midframe") / f"{Path(name).stem}_f{idx}.png"
        (out_dir / rel).parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(to_square(fr)).save(out_dir / rel)
        rows.append(dict(file=rel.as_posix(), set="midframe", expected=1, source_video=name,
                         frame_idx=idx, param=0, mean=round(float(g.mean()), 2),
                         sharpness=round(float(cv2.Laplacian(g, cv2.CV_64F).var()), 2)))

    # blurred copies of the real val echo frames
    for r in BLUR_RADII:
        for _, row in val_valid.iterrows():
            with Image.open(DATASET_DIR / row.file) as im:
                out = im.convert("RGB").filter(ImageFilter.GaussianBlur(radius=r))
            rel = Path(f"blur_r{r}") / Path(row.file).name
            (out_dir / rel).parent.mkdir(parents=True, exist_ok=True)
            out.save(out_dir / rel)
            rows.append(dict(file=rel.as_posix(), set=f"blur_r{r}", expected=0 if r >= EXTREME_FROM else -1,
                             source_video=row.source_video, frame_idx=row.frame_idx, param=r,
                             mean=np.nan, sharpness=np.nan))
    return pd.DataFrame(rows)


def main():
    m = pd.read_csv(DATASET_DIR / "manifest.csv")
    val_valid = m[(m.split == "val") & (m.label == 1)].reset_index(drop=True)
    names = sorted(val_valid.source_video.unique())
    print(f"val: {len(val_valid)} real frames from {len(names)} videos")

    videos = {p.name: p for p in VIDEO_DIR.rglob("*") if p.is_file() and p.suffix.lower() == ".avi"}
    missing = [n for n in names if n not in videos]
    if missing:
        sys.exit(f"{len(missing)} val videos not found under {VIDEO_DIR}: {missing[:5]}")
    print(f"all {len(names)} val videos found")

    if OUT_DIR.exists():
        tmp = OUT_DIR.parent / "_check"
        shutil.rmtree(tmp, ignore_errors=True)
        new = build(tmp, val_valid, videos)
        old = pd.read_csv(OUT_DIR / "exp2_sets_manifest.csv")
        same = new.fillna(-1).astype(str).equals(old.fillna(-1).astype(str))
        if same:
            same = all(sha256(tmp / f) == sha256(OUT_DIR / f) for f in new.file)
        shutil.rmtree(tmp, ignore_errors=True)
        print("OK: same sets as before" if same else "CHANGED: rebuilt sets differ from the saved ones (nothing overwritten)")
        return

    df = build(OUT_DIR, val_valid, videos)
    df.to_csv(OUT_DIR / "exp2_sets_manifest.csv", index=False)

    print("\nimages per set:")
    print(df.groupby("set").size().to_string())
    mid = df[df.set == "midframe"]
    print(f"\nmiddle frames: mean brightness {mid['mean'].min():.0f}-{mid['mean'].max():.0f}, "
          f"sharpness {mid.sharpness.min():.0f}-{mid.sharpness.max():.0f} "
          f"(Step 1 filter: brightness 15-200, sharpness >= 20)")
    same_idx = int(sum(((val_valid.source_video == r.source_video) & (val_valid.frame_idx == r.frame_idx)).any()
                       for r in mid.itertuples()))
    print(f"middle frames identical to a frame already in val: {same_idx}")
    assert len(mid) == len(names)
    assert (df.groupby("set").size().drop("midframe") == len(val_valid)).all()
    print(f"\nOK: saved in {OUT_DIR}/ (manifest: exp2_sets_manifest.csv)")


if __name__ == "__main__":
    main()
