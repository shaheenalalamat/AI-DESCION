"""
V2.0 / step 1: out-of-distribution invalid images (slide: Phase 1 V2.0)

Builds the new invalid folders: invalid/random_images (COCO) now, invalid/other_medical (X-ray, MRI) later.
gate_dataset_v2 is NOT touched and its test split is not used. Everything goes to a new folder (OUT_DIR),
with a manifest that has the same columns as gate_dataset_v2/manifest.csv (+ source_file), so both can be joined later.

- images are picked at random (seed 42) and resized to 256x256 like the other images (cv2.INTER_AREA)
- split 70/15/15 by image, per kind (COCO images are independent, there are no videos here)
- re-run safe: if OUT_DIR exists it only compares and prints OK or CHANGED
"""
import hashlib
import json
import math
import random
import shutil
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw

# ----------------------------- CONFIG ---------------------------------------
# kind -> (folder, or list of folders, with the source images; how many images to take in total)
# with a list, every folder gives the same number of images (n / number of folders)
SOURCES = {
    "random_images": (Path(r"C:/Users/rahma/OneDrive/Desktop/valedation/val2017/val2017"), 1000),
    "other_medical": ([Path(r"C:/Users/rahma/OneDrive/Desktop/valedation/archive"),         # chest X-ray
                       Path(r"C:/Users/rahma/OneDrive/Desktop/valedation/archive (1)")],    # brain MRI
                      1000),
}
OUT_DIR = Path("gate_dataset_v2_ood")
SPLIT = (0.70, 0.15, 0.15)
IMG_SIZE = 256
MIN_SIDE = 100             # skip tiny images
EXTS = {".jpg", ".jpeg", ".png"}
SEED = 42
# -----------------------------------------------------------------------------

SPLITS = ("train", "val", "test")


def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def split_counts(total, ratios):
    c = [int(total * r) for r in ratios]
    c[0] += total - sum(c)
    return dict(zip(SPLITS, c))


def list_images(folder):
    return sorted(p for p in folder.rglob("*")
                  if p.is_file() and p.suffix.lower() in EXTS and "__MACOSX" not in p.parts and not p.name.startswith("._"))


def load_square(p):
    try:
        with Image.open(p) as im:
            im = im.convert("RGB")
            if min(im.size) < MIN_SIDE:
                return None
            return cv2.resize(np.asarray(im), (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA)
    except OSError:
        return None


def build(out_dir):
    rows = []
    for ki, (kind, (src, n)) in enumerate(SOURCES.items()):
        folders = list(src) if isinstance(src, (list, tuple)) else [src]
        assert n % len(folders) == 0, f"{kind}: {n} images cannot be shared equally by {len(folders)} folders"
        rng = random.Random(SEED * 100 + ki)
        used, count = set(), 0          # used = content hashes already taken (the same image twice is skipped)
        for folder in folders:
            q = n // len(folders)
            files = list_images(folder)
            if len(files) < q:
                sys.exit(f"{kind}: only {len(files)} images under {folder}, need {q}")
            rng.shuffle(files)
            it = iter(files)
            for s, c in split_counts(q, SPLIT).items():
                got = 0
                for p in it:
                    arr = load_square(p)
                    if arr is None:
                        continue
                    h = sha256(p)
                    if h in used:
                        continue
                    used.add(h)
                    rel = Path(s) / "invalid" / kind / f"{kind}_{count:04d}.png"
                    (out_dir / rel).parent.mkdir(parents=True, exist_ok=True)
                    Image.fromarray(arr).save(out_dir / rel)
                    name = p.relative_to(folder).as_posix()
                    rows.append(dict(file=rel.as_posix(), label=0, split=s, kind=kind, source_video="",
                                     frame_idx=-1, source_file=f"{folder.name}/{name}" if len(folders) > 1 else name))
                    count += 1
                    got += 1
                    if got == c:
                        break
                if got < c:
                    sys.exit(f"{kind}: not enough readable images for {s} in {folder} ({got} < {c})")
    return pd.DataFrame(rows)


def sanity_grid(df, split, out_path, n=32, cols=8):
    sub = df[df.split == split]
    sub = sub.sample(min(n, len(sub)), random_state=SEED)
    rows = math.ceil(len(sub) / cols)
    canvas = Image.new("RGB", (cols * 128, rows * 128 + 24), (30, 30, 30))
    ImageDraw.Draw(canvas).text((6, 6), f"{split} / ood invalid (n={len(sub)})", fill=(255, 255, 255))
    for i, f in enumerate(sub["file"]):
        with Image.open(OUT_DIR / f) as im:
            canvas.paste(im.resize((128, 128)), ((i % cols) * 128, 24 + (i // cols) * 128))
    canvas.save(out_path)


def main():
    for kind, (src, n) in SOURCES.items():
        for folder in (src if isinstance(src, (list, tuple)) else [src]):
            if not folder.is_dir():
                sys.exit(f"{kind}: folder not found: {folder}")

    if OUT_DIR.exists():
        tmp = OUT_DIR.parent / "_check_ood"
        shutil.rmtree(tmp, ignore_errors=True)
        new = build(tmp)
        new.to_csv(tmp / "manifest.csv", index=False)      # same round trip as the saved manifest (empty cells -> NaN)
        new, old = pd.read_csv(tmp / "manifest.csv"), pd.read_csv(OUT_DIR / "manifest.csv")
        same = new.astype(str).equals(old.astype(str))
        if same:
            same = all(sha256(tmp / f) == sha256(OUT_DIR / f) for f in new.file)
        shutil.rmtree(tmp, ignore_errors=True)
        print("OK: same images as before" if same else "CHANGED: rebuilt images differ from the saved ones (nothing overwritten)")
        return

    df = build(OUT_DIR)
    df.to_csv(OUT_DIR / "manifest.csv", index=False)

    assert df.file.is_unique, "duplicate file names"
    for kind in SOURCES:
        assert df[df.kind == kind].source_file.is_unique, f"{kind}: the same source image is used twice"
    assert all((OUT_DIR / f).exists() for f in df.file)

    counts = df.groupby(["kind", "split"]).size().unstack(fill_value=0)[list(SPLITS)]
    (OUT_DIR / "build_report.json").write_text(json.dumps(
        {"seed": SEED, "counts": counts.to_dict(orient="index"), "img_size": IMG_SIZE,
         "sha256_manifest": sha256(OUT_DIR / "manifest.csv")}, indent=2))
    for s in SPLITS:
        sanity_grid(df, s, OUT_DIR / f"sanity_{s}.png")

    print("images per kind/split:\n", counts)
    print(f"\nOK: saved in {OUT_DIR}/ . Open sanity_train.png, sanity_val.png, sanity_test.png and look at them by eye.")


if __name__ == "__main__":
    main()