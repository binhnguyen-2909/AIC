"""
Build CLIP search index for AIC 2026 vòng sơ tuyển.
- Load pre-extracted clip-ViT-B-32 features from .npy files
- Build FAISS IP index (cosine via normalize)
- Store frame_id lookup from map-keyframes CSV
- Save index + metadata to disk for fast retrieval
"""
import os
import json
import numpy as np
import faiss
from pathlib import Path

ROOT = Path(os.environ.get("AIC_ROOT", str(Path(__file__).resolve().parents[1])))
EXTRACTED = ROOT / "extracted"
OUT = ROOT / "solution" / "index"
OUT.mkdir(parents=True, exist_ok=True)

CLIP_DIR = EXTRACTED / "clip-features-32-aic25-b1" / "clip-features-32"
MAP_DIR = EXTRACTED / "map-keyframes-aic25-b1" / "map-keyframes"

# video_id -> list of (keyframe_idx, frame_idx)
video_frames = {}
# global index: list of (video_id, frame_idx)
all_meta = []
all_vecs = []

npy_files = sorted(CLIP_DIR.glob("*.npy"))
print(f"Found {len(npy_files)} video npy files")

for npy in npy_files:
    vid = npy.stem  # L21_V001
    # clips
    feats = np.load(npy).astype(np.float32)  # (T, 512)
    # map-keyframes
    csv = MAP_DIR / f"{vid}.csv"
    if not csv.exists():
        print(f"WARN no map for {vid}")
        continue
    # read csv: n,pts_time,fps,frame_idx
    frame_idx_arr = []
    with open(csv) as f:
        next(f)  # header
        for line in f:
            parts = line.strip().split(",")
            if len(parts) >= 4:
                frame_idx_arr.append(int(parts[3]))
    if len(frame_idx_arr) != feats.shape[0]:
        # some videos might differ; truncate
        T = min(len(frame_idx_arr), feats.shape[0])
        feats = feats[:T]
        frame_idx_arr = frame_idx_arr[:T]
    # L2 normalize for cosine-similarity via IP
    faiss.normalize_L2(feats)
    all_vecs.append(feats)
    for fi in frame_idx_arr:
        all_meta.append((vid, fi))

print(f"Total vectors: {len(all_meta)}")
mat = np.vstack(all_vecs).astype(np.float32)
print(f"Mat shape: {mat.shape}")

# Build FAISS index
d = mat.shape[1]
index = faiss.IndexFlatIP(d)
index.add(mat)
print(f"Index built, ntotal={index.ntotal}")

# Save
faiss.write_index(index, str(OUT / "clip_b32.faiss"))
with open(OUT / "meta.json", "w") as f:
    json.dump(all_meta, f)
print(f"Saved to {OUT}")
