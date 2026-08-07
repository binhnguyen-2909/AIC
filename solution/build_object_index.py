"""
Build object-signal index from Faster R-CNN OpenImages V4 detections.

Each frame in each video has up to 100 detections with class entities like
"Tower", "Skyscraper", "Car". We build two indexes:

  (1) per-frame object bag  -> { (video_id, frame_idx): { class: max_score } }
      stored as a flat array so we can score any (video, frame) in O(1).

  (2) per-video TF-IDF over class entities -> { video_id: { class: weight } }
      Used as an aggregated "what's in this video" signal.

  (3) inverted index class -> [(video_id, frame_idx, score)] for object-first search.

Output: solution/ensemble_index/objects.pkl
"""
import os, json, pickle
from pathlib import Path
import numpy as np
from collections import defaultdict
import math

ROOT = Path(os.environ.get("AIC_ROOT", str(Path(__file__).resolve().parents[1])))
EXTRACTED = ROOT / "extracted"
OBJ_DIR = EXTRACTED / "objects-aic25-b1" / "objects"
MAP_DIR = EXTRACTED / "map-keyframes-aic25-b1" / "map-keyframes"
OUT = ROOT / "solution" / "ensemble_index"
OUT.mkdir(parents=True, exist_ok=True)

# Global frame_idx -> (video_id, frame_pos_in_video) map.
# Already exists at solution/index/meta.json but we want frame_pos too.
META_PATH = ROOT / "solution" / "index" / "meta.json"


def build_frame_pos():
    with open(META_PATH) as f:
        meta = json.load(f)
    # frame_pos[i] = (video_id, pos_in_video) where pos_in_video is 0..T-1 for that video
    frame_pos = []
    for vid, fi in meta:
        frame_pos.append((vid, fi))
    return frame_pos


def main():
    frame_pos = build_frame_pos()
    n_frames = len(frame_pos)
    print(f"Total frames: {n_frames}")

    # Object-inverted index: class -> list of (frame_idx, max_score)
    inv_class = defaultdict(list)
    # Per-frame top objects: frame_idx -> {class: max_score}
    frame_objs = []
    # Per-video class aggregate (for video-level scoring)
    video_class_score = defaultdict(lambda: defaultdict(float))
    video_class_count = defaultdict(lambda: defaultdict(int))

    # Iterate videos, gather all keyframe JSONs.
    video_dirs = sorted(OBJ_DIR.iterdir())
    print(f"Video dirs: {len(video_dirs)}")
    n_done = 0
    for vd in video_dirs:
        if not vd.is_dir():
            continue
        vid = vd.name
        json_files = sorted(vd.glob("*.json"), key=lambda p: int(p.stem))
        for pos, jf in enumerate(json_files):
            try:
                with open(jf) as f:
                    d = json.load(f)
            except Exception:
                continue
            classes = d.get("detection_class_entities", [])
            scores = d.get("detection_scores", [])
            if not classes:
                continue
            # Parse scores
            try:
                scores_f = [float(s) for s in scores]
            except Exception:
                scores_f = []
            # cap to top-N by score per frame to avoid noise
            if scores_f:
                order = np.argsort(-np.asarray(scores_f))[:30]
                keep_classes = [classes[i] for i in order if i < len(classes)]
                keep_scores = [scores_f[i] for i in order if i < len(scores_f)]
            else:
                keep_classes = classes[:30]
                keep_scores = [1.0] * len(keep_classes)
            # Map to frame_idx using meta lookup
            # We rely on order: pos in video -> frame_idx in global meta.
            # But faster: we already have json_files in pos order matching map-keyframes order.
            # Build a lookup: video_id -> list of global frame_idx in order.
            # For speed, defer assignment: just record (vid, pos) and resolve later.
            # Easier: precompute vid->frame_idxs
        n_done += 1
        if n_done % 100 == 0:
            print(f"  scanned {n_done} videos")

    # Faster path: build per-video -> list of frame_idxs
    vid_to_fidx = defaultdict(list)
    for fidx, (vid, fi) in enumerate(frame_pos):
        vid_to_fidx[vid].append(fidx)

    # Re-iterate with mapping
    for vd in video_dirs:
        if not vd.is_dir():
            continue
        vid = vd.name
        json_files = sorted(vd.glob("*.json"), key=lambda p: int(p.stem))
        # Do not align by list position.  The map contains repeated original
        # frame indices for some videos, while the production FAISS index
        # keeps one row per unique (video, frame_idx).  Position-based
        # truncation shifts every later object annotation onto the wrong
        # frame.  Resolve each object file through the authoritative CSV.
        map_csv = MAP_DIR / f"{vid}.csv"
        ordinal_to_frame = {}
        if map_csv.exists():
            with open(map_csv) as mf:
                next(mf, None)
                for line in mf:
                    parts = line.strip().split(",")
                    if len(parts) >= 4:
                        ordinal_to_frame[int(parts[0])] = int(parts[3])
        frame_to_fidx = {}
        for fidx in vid_to_fidx.get(vid, []):
            frame_to_fidx.setdefault(int(frame_pos[fidx][1]), fidx)

        for jf in json_files:
            frame_id = ordinal_to_frame.get(int(jf.stem))
            fidx = frame_to_fidx.get(frame_id)
            if fidx is None:
                continue
            try:
                with open(jf) as f:
                    d = json.load(f)
            except Exception:
                continue
            classes = d.get("detection_class_entities", [])
            scores = d.get("detection_scores", [])
            try:
                scores_f = [float(s) for s in scores]
            except Exception:
                scores_f = []
            if not classes:
                continue
            # cap to top 30
            if scores_f:
                order = np.argsort(-np.asarray(scores_f))[:30]
                keep_classes = [classes[i] for i in order if i < len(classes)]
                keep_scores = [scores_f[i] for i in order if i < len(scores_f)]
            else:
                keep_classes = classes[:30]
                keep_scores = [1.0] * len(keep_classes)
            # Per-frame
            fbag = {}
            for cls, sc in zip(keep_classes, keep_scores):
                if cls not in fbag or sc > fbag[cls]:
                    fbag[cls] = sc
            # ensure size
            if len(frame_objs) <= fidx:
                frame_objs.extend([{} for _ in range(fidx + 1 - len(frame_objs))])
            frame_objs[fidx] = fbag
            # Inverted: use the same per-frame max-pooling as ``frame_objs``.
            # Appending every repeated detection box overweights crowded
            # frames and makes object fusion depend on detector duplication.
            for cls, sc in fbag.items():
                inv_class[cls].append((fidx, sc))
                video_class_score[vid][cls] += sc
                video_class_count[vid][cls] += 1

    print(f"Built frames={len(frame_objs)}, classes={len(inv_class)}")

    # Save
    payload = {
        "frame_objs": frame_objs,  # list[dict[class->score]]
        "inv_class": dict(inv_class),  # class -> list[(frame_idx, score)]
        "video_class_score": {v: dict(d) for v, d in video_class_score.items()},
        "video_class_count": {v: dict(d) for v, d in video_class_count.items()},
    }
    with open(OUT / "objects.pkl", "wb") as fp:
        pickle.dump(payload, fp)
    print(f"Saved to {OUT/'objects.pkl'}")


if __name__ == "__main__":
    main()
