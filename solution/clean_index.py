"""Clean the FAISS index by removing duplicate (video_id, frame_idx) entries.
Also cap each video to true keyframe count from the map-keyframes CSV.
"""
import json
import os
from pathlib import Path
import numpy as np
import faiss

ROOT = Path(os.environ.get("AIC_ROOT", str(Path(__file__).resolve().parents[1])))
EXTRACTED = ROOT / "extracted"
OUT = ROOT / "solution" / "index"
CLIP_DIR = EXTRACTED / "clip-features-32-aic25-b1" / "clip-features-32"
MAP_DIR = EXTRACTED / "map-keyframes-aic25-b1" / "map-keyframes"

# Build fresh (video_id, frame_idx) and vector mapping
all_meta = []
all_vecs = []

npy_files = sorted(CLIP_DIR.glob("*.npy"))
for npy in npy_files:
    vid = npy.stem
    feats = np.load(npy).astype(np.float32)
    csv = MAP_DIR / f"{vid}.csv"
    if not csv.exists():
        continue
    frame_idx_arr = []
    with open(csv) as f:
        next(f)
        for line in f:
            parts = line.strip().split(",")
            if len(parts) >= 4:
                frame_idx_arr.append(int(parts[3]))
    T = min(len(frame_idx_arr), feats.shape[0])
    feats = feats[:T]
    frame_idx_arr = frame_idx_arr[:T]
    faiss.normalize_L2(feats)
    all_vecs.append(feats)
    for fi in frame_idx_arr:
        all_meta.append([vid, fi])

mat = np.vstack(all_vecs).astype(np.float32)
print(f"Total vectors (clean): {len(all_meta)}   {mat.shape}")

# Dedupe just in case
seen = set()
keep_idx = []
keep_meta = []
for i, m in enumerate(all_meta):
    key = tuple(m)
    if key in seen:
        continue
    seen.add(key)
    keep_idx.append(i)
    keep_meta.append(m)

mat = mat[keep_idx]
print(f"After dedup: {len(keep_meta)}   {mat.shape}")

# Build fresh index
d = mat.shape[1]
index = faiss.IndexFlatIP(d)
index.add(mat)
print(f"Index ntotal: {index.ntotal}")

faiss.write_index(index, str(OUT / "clip_b32.faiss"))
with open(OUT / "meta.json", "w") as f:
    json.dump(keep_meta, f)
print("Saved.")
