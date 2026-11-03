"""
Step 1: Synthetic Data Engineering (COR-LVH AI Validation gate)

Class 1 (valid)  : real frames extracted from EchoNet-LVH videos
Class 0 (invalid): images generated here (black, white, noise, extreme blur)

Split is done on VIDEOS (never on frames), so no video appears in two splits.
Invalid images are generated per split with a different seed per split.

Outputs (under OUT_DIR):
    {train,val,test}/{valid,invalid}/*.png
    manifest.csv            file, label, split, kind, source_video, frame_idx
    sanity_{split}.png      visual sanity check grids (look at them with your eyes)
    build_report.json       counts + checks
"""
import json
import math
import random
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFilter
try:
    from tqdm import tqdm
except ImportError:  # progress bar is optional
    def tqdm(it, **kw):
        return it

# ----------------------------- CONFIG ---------------------------------------
VIDEO_DIR = Path(r"C:/Users/rahma/OneDrive/Desktop/COR_LVH/EchoNet-LVH")   # folder with the .avi files
OUT_DIR = Path("gate_dataset_v2")
VIDEO_EXT = ".avi"

N_VALID = 400              # total valid frames (target 300-500)
N_INVALID = 400            # total invalid images (target 300-500)
FRAMES_PER_VIDEO = 2       # frames taken from each chosen video
SPLIT = (0.70, 0.15, 0.15) # train / val / test (by video)
IMG_SIZE = 256             # saved size, same as the resized frames used by the team
SEED = 42

# how many random candidate frames to look at per pick; the sharpest one is kept
CANDIDATES_PER_PICK = 8
EDGE_MARGIN = 0.10         # ignore the first/last 10% of each video
MIN_SHARPNESS = 20.0       # variance of Laplacian, rejects blurry/empty frames
MIN_MEAN, MAX_MEAN = 15, 200   # rejects almost-black / almost-white frames
MIN_FRAME_GAP = 10         # two frames from the same video must be at least this many frames apart

INVALID_KINDS = ("black", "white", "noise", "blur")
# -----------------------------------------------------------------------------

SPLITS = ("train", "val", "test")


def sharpness(gray):
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def read_frame(cap, idx):
    cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
    ok, frame = cap.read()
    return frame if ok else None


def pick_frames(video_path, k, rng):
    """Return up to k (frame_bgr, frame_idx) that are clear. Random positions, sharpest of a few candidates."""
    cap = cv2.VideoCapture(str(video_path))
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if n < 5:
        cap.release()
        return []
    lo, hi = int(n * EDGE_MARGIN), max(int(n * (1 - EDGE_MARGIN)), int(n * EDGE_MARGIN) + 1)
    used, out = set(), []
    for _ in range(k):
        best = None
        for _ in range(CANDIDATES_PER_PICK):
            idx = rng.randrange(lo, hi)
            if any(abs(idx - u) < MIN_FRAME_GAP for u in used):
                continue
            fr = read_frame(cap, idx)
            if fr is None:
                continue
            g = cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY)
            m, s = float(g.mean()), sharpness(g)
            if not (MIN_MEAN <= m <= MAX_MEAN) or s < MIN_SHARPNESS:
                continue
            if best is None or s > best[0]:
                best = (s, fr, idx)
        if best is not None:
            used.add(best[2])
            out.append((best[1], best[2]))
    cap.release()
    return out


def to_square(frame_bgr):
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    return cv2.resize(rgb, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA)


# ----------------------- invalid image generators ----------------------------
def gen_black(rng):
    v = rng.randint(0, 8)
    return np.full((IMG_SIZE, IMG_SIZE, 3), v, np.uint8)


def gen_white(rng):
    v = rng.randint(247, 255)
    return np.full((IMG_SIZE, IMG_SIZE, 3), v, np.uint8)


def gen_noise(rng):
    # grayscale random noise (same value on 3 channels), like the grayscale echo frames
    r = np.random.RandomState(rng.randrange(2**31))
    g = r.randint(0, 256, (IMG_SIZE, IMG_SIZE), dtype=np.uint8)
    return np.stack([g, g, g], axis=-1)


def gen_blur(rng):
    """Random gray shapes (echo-like blobs) then an extreme Gaussian blur."""
    img = Image.new("RGB", (IMG_SIZE, IMG_SIZE), (rng.randint(0, 60),) * 3)
    d = ImageDraw.Draw(img)
    for _ in range(rng.randint(6, 14)):
        x0, y0 = rng.randint(0, IMG_SIZE - 40), rng.randint(0, IMG_SIZE - 40)
        x1, y1 = x0 + rng.randint(20, 120), y0 + rng.randint(20, 120)
        c = (rng.randint(40, 255),) * 3
        (d.ellipse if rng.random() < 0.5 else d.rectangle)([x0, y0, x1, y1], fill=c)
    img = img.filter(ImageFilter.GaussianBlur(radius=rng.uniform(15, 30)))
    return np.array(img)


GEN = {"black": gen_black, "white": gen_white, "noise": gen_noise, "blur": gen_blur}


def save_png(arr_rgb, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(arr_rgb).save(path)


def split_counts(total, ratios):
    """Exact integer counts that sum to total."""
    c = [int(total * r) for r in ratios]
    c[0] += total - sum(c)
    return dict(zip(SPLITS, c))


def sanity_grid(df, split, out_path, per_class=24, cols=8):
    parts = []
    for label, name in ((1, "valid"), (0, "invalid")):
        sub = df[(df.split == split) & (df.label == label)]
        sub = sub.sample(min(per_class, len(sub)), random_state=SEED) if len(sub) else sub
        thumbs = [Image.open(OUT_DIR / f).resize((128, 128)) for f in sub["file"]]
        rows = math.ceil(max(len(thumbs), 1) / cols)
        canvas = Image.new("RGB", (cols * 128, rows * 128 + 24), (30, 30, 30))
        ImageDraw.Draw(canvas).text((6, 6), f"{split} / {name} (n={len(sub)})", fill=(255, 255, 255))
        for i, t in enumerate(thumbs):
            canvas.paste(t, ((i % cols) * 128, 24 + (i // cols) * 128))
        parts.append(canvas)
    w = max(p.width for p in parts)
    full = Image.new("RGB", (w, sum(p.height for p in parts)), (30, 30, 30))
    y = 0
    for p in parts:
        full.paste(p, (0, y))
        y += p.height
    full.save(out_path)


def main():
    if abs(sum(SPLIT) - 1) > 1e-9:
        sys.exit("SPLIT must sum to 1")
    if not VIDEO_DIR.is_dir():
        sys.exit(f"VIDEO_DIR not found: {VIDEO_DIR}")
    videos = sorted(p for p in VIDEO_DIR.rglob("*") if p.is_file() and p.suffix.lower() == VIDEO_EXT)
    if not videos:
        found = {}
        for p in VIDEO_DIR.rglob("*"):
            if p.is_file():
                found[p.suffix.lower() or "(no ext)"] = found.get(p.suffix.lower() or "(no ext)", 0) + 1
        subdirs = [p.name for p in VIDEO_DIR.iterdir() if p.is_dir()][:10]
        sys.exit(f"no {VIDEO_EXT} files under {VIDEO_DIR} (searched all subfolders).\n"
                 f"file types found: {found or 'none, folder is empty'}\n"
                 f"first subfolders: {subdirs}\n"
                 f"Fix VIDEO_DIR or VIDEO_EXT.")
    print(f"found {len(videos)} {VIDEO_EXT} files")
    if len({p.stem for p in videos}) != len(videos):
        sys.exit("duplicate video names across subfolders, names must be unique")
    if OUT_DIR.exists() and any(OUT_DIR.iterdir()):
        sys.exit(f"{OUT_DIR} is not empty. Delete it or change OUT_DIR (avoids mixing runs).")

    rng = random.Random(SEED)

    # 1) choose videos and split them (VIDEO level)
    n_videos = math.ceil(N_VALID / FRAMES_PER_VIDEO)
    pool = videos[:]
    rng.shuffle(pool)
    # take extra videos so unreadable/blurry ones can be skipped
    chosen = pool[: min(len(pool), math.ceil(n_videos * 1.3))]
    need = split_counts(N_VALID, SPLIT)
    print(f"videos available: {len(videos)} | candidates: {len(chosen)} | target valid per split: {need}")

    # videos per split proportional to targets
    vid_split, cursor = {}, 0
    vcounts = split_counts(len(chosen), SPLIT)
    for s in SPLITS:
        for v in chosen[cursor: cursor + vcounts[s]]:
            vid_split[v] = s
        cursor += vcounts[s]

    rows, got = [], {s: 0 for s in SPLITS}

    # 2) valid frames
    for v, s in tqdm(vid_split.items(), desc="extracting"):
        if got[s] >= need[s]:
            continue
        k = min(FRAMES_PER_VIDEO, need[s] - got[s])
        for fr, idx in pick_frames(v, k, rng):
            rel = Path(s) / "valid" / f"{v.stem}_f{idx}.png"
            save_png(to_square(fr), OUT_DIR / rel)
            rows.append(dict(file=rel.as_posix(), label=1, split=s, kind="echo",
                             source_video=v.name, frame_idx=idx))
            got[s] += 1
    for s in SPLITS:
        if got[s] < need[s]:
            print(f"WARNING: {s} valid frames {got[s]} < target {need[s]} "
                  f"(not enough clear frames: lower MIN_SHARPNESS or raise the video pool)")

    # 3) invalid images, per split, own seed per split, balanced against valid count
    for si, s in enumerate(SPLITS):
        n_inv = need_inv = split_counts(N_INVALID, SPLIT)[s]
        r = random.Random(SEED * 1000 + si + 1)
        kinds = [INVALID_KINDS[i % len(INVALID_KINDS)] for i in range(n_inv)]
        for i, kind in enumerate(kinds):
            rel = Path(s) / "invalid" / f"{kind}_{i:04d}.png"
            save_png(GEN[kind](r), OUT_DIR / rel)
            rows.append(dict(file=rel.as_posix(), label=0, split=s, kind=kind,
                             source_video="", frame_idx=-1))

    df = pd.DataFrame(rows)
    df.to_csv(OUT_DIR / "manifest.csv", index=False)

    # 4) checks
    vsets = {s: set(df[(df.split == s) & (df.label == 1)].source_video) for s in SPLITS}
    overlaps = {f"{a}&{b}": len(vsets[a] & vsets[b])
                for i, a in enumerate(SPLITS) for b in SPLITS[i + 1:]}
    counts = df.groupby(["split", "label"]).size().unstack(fill_value=0).rename(columns={0: "invalid", 1: "valid"})
    dup_files = int(df["file"].duplicated().sum())
    report = {
        "seed": SEED, "videos_used": {s: len(vsets[s]) for s in SPLITS},
        "video_overlap_between_splits": overlaps,
        "counts": counts.to_dict(orient="index"),
        "invalid_kinds": df[df.label == 0].groupby(["split", "kind"]).size().unstack(fill_value=0).to_dict(orient="index"),
        "duplicate_filenames": dup_files,
    }
    (OUT_DIR / "build_report.json").write_text(json.dumps(report, indent=2))

    for s in SPLITS:
        sanity_grid(df, s, OUT_DIR / f"sanity_{s}.png")

    print("\ncounts per split/class:\n", counts)
    print("video overlap between splits (must be 0):", overlaps)
    assert all(v == 0 for v in overlaps.values()), "LEAKAGE: a video is in two splits"
    assert dup_files == 0
    print(f"\nDone. Open {OUT_DIR}/sanity_train.png, sanity_val.png, sanity_test.png and check by eye.")


if __name__ == "__main__":
    main()
