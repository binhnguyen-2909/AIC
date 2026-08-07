"""
TRAKE solver for AIC 2026 vong so tuyen.

Task: given a query describing an action/event with N sub-events, return
(video_id, frame_id1, ..., frame_idN) such that each frame_idj is inside the
ground-truth range [sj, ej] (usually < 10 frames wide). Wrong video => 0.

Pipeline (offline, no internet):
  Stage 1 - Retrieval: encode the query with CLIP-ViT-B/32, search FAISS
              IndexFlatIP built from pre-extracted clip-features-32-aic25-b1.
              Pick top-1 video by mode of top-K keyframes (top-3 as backups).
  Stage 2 - Alignment: for each sub-event, re-rank keyframes inside the chosen
              video using CLIP cosine sim between the event phrase embedding and
              each in-video keyframe. Add a small action-specific bias when an
              objects JSON is available.
  Stage 3 - Ordering: enforce strictly increasing frame indices with a small
              margin between consecutive events using a Viterbi-style DP that
              minimizes a cost = -score + lambda * gap_penalty + repeat_penalty.
  Stage 4 - Top-100 generation: for each (video in top-3) we emit N lines that
              are perturbed around the best alignment (offset +/-k and
              micro-perturbations on each event) to fill the 100-slot budget.

Output: JSONL file with rows {query_id, rank, answer} where answer is the
string "video_id, f1, ..., fN".

Usage:
  python trake_solver.py --queries queries.jsonl --out submission.jsonl
                          [--queries-key events] [--top-k-videos 3] [--top-100]
"""
from __future__ import annotations

import argparse
import json
import math
import os
import pickle
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

ROOT = Path(__file__).resolve().parent
EXTRACTED = ROOT.parent / "extracted"
INDEX_DIR = ROOT / "index"
CLIP_DIR = EXTRACTED / "clip-features-32-aic25-b1" / "clip-features-32"
MAP_DIR = EXTRACTED / "map-keyframes-aic25-b1" / "map-keyframes"
OBJ_DIR = EXTRACTED / "objects-aic25-b1" / "objects"

CACHE_PER_VID = ROOT / "index" / "per_video.pkl"

# Lazy import of torch/transformers so --help works without GPU.
_torch = None
_transformers = None


def _load_clip(device: str = "cuda"):
    global _torch, _transformers
    import torch as _t  # noqa
    from transformers import CLIPModel, CLIPProcessor  # noqa
    _torch = _t
    _transformers = (CLIPModel, CLIPProcessor)
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device).eval()
    proc = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    if device == "cuda":
        model = model.half()
    return model, proc, device


# ---------------------------------------------------------------------------
# Per-video bookkeeping
# ---------------------------------------------------------------------------


def build_per_video_cache(meta_path: Path = INDEX_DIR / "meta.json",
                          cache_path: Path = CACHE_PER_VID,
                          clip_dir: Path = CLIP_DIR) -> Dict:
    """Return {video_id: {'frames': sorted list of frame_idx,
                         'rows': np.array of row indices into global meta,
                         'vecs': np.array (T,512) normalized}}.

    If the npy shape differs from the map-keyframes length, the shorter of
    the two is used to keep ``vecs`` and ``frames`` aligned.
    """
    if cache_path.exists():
        with open(cache_path, "rb") as f:
            return pickle.load(f)

    with open(meta_path) as f:
        meta = json.load(f)

    by_vid: Dict[str, List[Tuple[int, int]]] = defaultdict(list)
    for row, (vid, fi) in enumerate(meta):
        by_vid[vid].append((row, int(fi)))

    out: Dict[str, Dict] = {}
    for vid, items in by_vid.items():
        items.sort(key=lambda x: x[1])
        rows = np.array([r for r, _ in items], dtype=np.int64)
        frames = np.array([f for _, f in items], dtype=np.int64)
        npy = clip_dir / f"{vid}.npy"
        vecs = None
        if npy.exists():
            v = np.load(npy).astype(np.float32)
            # The source map can contain repeated original frame indices.
            # ``meta`` keeps the first unique occurrence, so taking v[:T]
            # would shift all vectors after the first duplicate.  Resolve
            # each meta frame back to its keyframe ordinal before slicing.
            frame_to_ordinal = {}
            csv = MAP_DIR / f"{vid}.csv"
            if csv.exists():
                with open(csv) as mf:
                    next(mf, None)
                    for line in mf:
                        parts = line.strip().split(",")
                        if len(parts) >= 4:
                            frame_to_ordinal.setdefault(int(parts[3]), int(parts[0]) - 1)
            ordinals = [frame_to_ordinal.get(int(fi), i)
                        for i, fi in enumerate(frames)]
            keep = [i for i, o in enumerate(ordinals)
                    if 0 <= o < v.shape[0]]
            rows = rows[keep]
            frames = frames[keep]
            ordinals = [ordinals[i] for i in keep]
            v = v[np.asarray(ordinals, dtype=np.int64)]
            norms = np.linalg.norm(v, axis=1, keepdims=True) + 1e-9
            vecs = (v / norms).astype(np.float32)
        out[vid] = {"frames": frames, "rows": rows, "vecs": vecs}

    with open(cache_path, "wb") as f:
        pickle.dump(out, f, protocol=pickle.HIGHEST_PROTOCOL)
    return out


# ---------------------------------------------------------------------------
# CLIP text encoder
# ---------------------------------------------------------------------------


class TextEncoder:
    def __init__(self, device: str = "cuda"):
        self.model, self.proc, self.device = _load_clip(device)
        self._cache: Dict[str, np.ndarray] = {}

    @staticmethod
    def _normalize(x: np.ndarray) -> np.ndarray:
        n = np.linalg.norm(x, axis=-1, keepdims=True) + 1e-9
        return (x / n).astype(np.float32)

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        import torch as _t
        todo = [t for t in texts if t not in self._cache]
        if todo:
            with _t.no_grad():
                inp = self.proc(text=list(todo), return_tensors="pt",
                                padding=True, truncation=True).to(self.device)
                for k in inp:
                    if inp[k].dtype == _t.float32:
                        inp[k] = inp[k].half()
                out = self.model.get_text_features(**inp)
                # ``get_text_features`` returns ``BaseModelOutputWithPooling``
                # whose ``text_embeds`` is the projected L2-normalized
                # embedding (shape [B, projection_dim]).
                feats = getattr(out, "text_embeds", None)
                if feats is None:
                    feats = out[0] if isinstance(out, tuple) else out.pooler_output
                feats = feats / feats.norm(dim=-1, keepdim=True).clamp(min=1e-9)
                arr = feats.cpu().numpy().astype(np.float32)
            for t, v in zip(todo, arr):
                self._cache[t] = v
        return np.stack([self._cache[t] for t in texts], axis=0)


# ---------------------------------------------------------------------------
# TRAKE solver
# ---------------------------------------------------------------------------


def select_video(query_emb: np.ndarray, faiss_index, meta: List[List],
                 k_search: int = 200, top_videos: int = 3) -> Tuple[str, float]:
    """Return (top_video, score). Uses mode of top-K keyframes weighted by score."""
    scores, idx = faiss_index.search(query_emb.astype(np.float32),
                                     min(k_search, faiss_index.ntotal))
    counts: Dict[str, float] = defaultdict(float)
    for s, i in zip(scores[0], idx[0]):
        if i < 0:
            continue
        vid, _ = meta[i]
        counts[vid] += float(s) + 0.5  # score mass + uniform mass for tie break
    ranked = sorted(counts.items(), key=lambda x: -x[1])
    if not ranked:
        return "", 0.0
    return ranked[0][0], ranked[0][1]


def event_score(video_vecs: np.ndarray, event_emb: np.ndarray) -> np.ndarray:
    """Per-frame cosine scores between event phrase and every keyframe in the video."""
    if video_vecs is None or len(video_vecs) == 0:
        return np.zeros(0, dtype=np.float32)
    return video_vecs @ event_emb.reshape(-1).astype(np.float32)


def object_bias(video_id: str, frame_indices: np.ndarray,
                event_keywords: List[str]) -> np.ndarray:
    """Optional bias from Faster R-CNN object JSON. Returns length-T bias array."""
    T = len(frame_indices)
    bias = np.zeros(T, dtype=np.float32)
    vid_dir = OBJ_DIR / video_id
    if not vid_dir.exists():
        return bias
    kw = [k.lower() for k in event_keywords if k]
    if not kw:
        return bias
    # Object files are named by keyframe ordinal (001.json, 002.json, ...),
    # whereas the solver cache stores the original video frame_idx (0, 90,
    # 261, ...).  Resolve through the authoritative map before opening JSON;
    # using frame_idx directly silently missed almost every detection.
    frame_to_ordinal = {}
    map_path = MAP_DIR / f"{video_id}.csv"
    if map_path.exists():
        try:
            with open(map_path) as mf:
                next(mf, None)
                for line in mf:
                    parts = line.strip().split(",")
                    if len(parts) >= 4:
                        # CSV n is one-based; object JSON follows that name.
                        frame_to_ordinal.setdefault(int(parts[3]), int(parts[0]))
        except (OSError, ValueError):
            frame_to_ordinal = {}
    # Read objects JSON for every keyframe; 873 videos * ~150 frames ≈ 130k
    # small files — cheap.  Skip with an early-out if the directory has fewer
    # than 10 JSONs.
    sample = np.linspace(0, T - 1, num=min(T, 32)).astype(int)
    for i in sample:
        ordinal = frame_to_ordinal.get(int(frame_indices[i]), int(i) + 1)
        jp = vid_dir / f"{ordinal:03d}.json"
        if not jp.exists():
            continue
        try:
            with open(jp) as f:
                d = json.load(f)
        except Exception:
            continue
        names = " ".join(d.get("detection_class_entities", [])).lower()
        hits = sum(1 for k in kw if k in names)
        if hits:
            # Spread the local bias to the immediate neighbourhood so that
            # neighbours also receive the boost before DP assignment.
            lo = max(0, i - 2)
            hi = min(T, i + 3)
            bias[lo:hi] = np.maximum(bias[lo:hi], 0.01 * hits)
    return bias


def align_events_dp(scores_per_event: List[np.ndarray],
                    frames_per_event: List[np.ndarray],
                    min_gap: int = 1,
                    repeat_penalty: float = 0.05,
                    back_penalty: float = 0.5) -> List[int]:
    """Exact Viterbi alignment over an ordered sequence of events.

    ``scores_per_event[e][j]`` is the CLIP score for assigning event ``e`` to
    candidate frame ``frames_per_event[e][j]``.  The transition enforces
    strictly increasing frame ids whenever possible.  If no feasible path
    exists (e.g. too few keyframes), equal/backward transitions remain possible
    but receive a penalty.

    Complexity is O(E*T^2).  Typical AIC videos have T=50..300 keyframes and
    E<10, so this is well below 10 ms/query on CPU.
    """
    E = len(scores_per_event)
    if E == 0:
        return []

    # Dynamic-programming table and backpointers. Candidate counts may differ
    # across events, so store one array per event instead of a dense matrix.
    dp: List[np.ndarray] = [scores_per_event[0].astype(np.float64).copy()]
    back: List[Optional[np.ndarray]] = [None]

    for e in range(1, E):
        prev_dp = dp[e - 1]
        prev_frames = frames_per_event[e - 1]
        cur_frames = frames_per_event[e]
        cur_scores = scores_per_event[e].astype(np.float64)

        cur_dp = np.full(len(cur_frames), -np.inf, dtype=np.float64)
        cur_back = np.full(len(cur_frames), -1, dtype=np.int64)

        for j, fj in enumerate(cur_frames):
            gaps = fj - prev_frames
            feasible = gaps >= min_gap
            transition = np.zeros(len(prev_frames), dtype=np.float64)
            # A tiny preference for separation avoids choosing adjacent
            # near-duplicates while preserving fine-grained localization.
            transition[feasible] -= 0.001 / np.sqrt(np.maximum(gaps[feasible], 1))
            # Equal or backward assignments are allowed only as a fallback.
            transition[~feasible] -= (
                repeat_penalty
                + back_penalty * np.minimum(np.abs(gaps[~feasible]), 300)
                / 300.0
            )
            vals = prev_dp + transition
            i = int(np.argmax(vals))
            cur_dp[j] = cur_scores[j] + vals[i]
            cur_back[j] = i

        dp.append(cur_dp)
        back.append(cur_back)

    # Backtrack the maximum-score path.
    idx = int(np.argmax(dp[-1]))
    chosen_idx = [idx]
    for e in range(E - 1, 0, -1):
        idx = int(back[e][idx])
        chosen_idx.append(idx)
    chosen_idx.reverse()
    return [int(frames_per_event[e][chosen_idx[e]]) for e in range(E)]


def assemble_answer(video_id: str, frames: Sequence[int]) -> str:
    return f"{video_id}, " + ", ".join(str(int(f)) for f in frames)


def generate_top100(primary_video: str,
                    primary_frames: List[int],
                    backup_videos: List[Tuple[str, float]],
                    per_video_cache: Dict,
                    n_lines: int = 100) -> List[str]:
    """Emit up to n_lines answers. Slot 0 is the best primary answer.
    Subsequent slots:  (a) small perturbations on the primary,
                      (b) the same alignment for backup videos,
                      (c) coarse temporal shifts on the primary.
    """
    out: List[str] = [assemble_answer(primary_video, primary_frames)]

    # (a) local perturbations: +/-1..5 on each event independently
    perturb_offsets = []
    for off in range(1, 6):
        for e in range(len(primary_frames)):
            new = list(primary_frames)
            new[e] = max(0, new[e] + off)
            perturb_offsets.append(new)
        for e in range(len(primary_frames)):
            new = list(primary_frames)
            new[e] = max(0, new[e] - off)
            perturb_offsets.append(new)

    # Sort perturbations by total absolute shift so closest stay near top.
    def l1shift(seq):
        return sum(abs(seq[i] - primary_frames[i]) for i in range(len(seq)))

    perturb_offsets.sort(key=l1shift)

    for seq in perturb_offsets:
        if len(out) >= n_lines:
            break
        out.append(assemble_answer(primary_video, seq))

    # (b) backup videos: pick their best alignment by mean keyframe score
    for vid, _ in backup_videos:
        if vid == primary_video or vid not in per_video_cache:
            continue
        if len(out) >= n_lines:
            break
        info = per_video_cache[vid]
        # naive: pick equally spaced frames as fallback
        T = len(info["frames"])
        if T == 0:
            continue
        idxs = np.linspace(0, T - 1, num=len(primary_frames)).astype(int)
        seq = [int(info["frames"][i]) for i in idxs]
        out.append(assemble_answer(vid, seq))

    # (c) coarse shifts on the primary video (event 0 shifted by large step)
    big_offsets = [-30, -15, -7, 7, 15, 30, 45]
    for off in big_offsets:
        if len(out) >= n_lines:
            break
        seq = [max(0, f + off) for f in primary_frames]
        out.append(assemble_answer(primary_video, seq))

    # Pad by repeating the primary with tiny shifts to reach 100
    while len(out) < n_lines:
        seq = list(primary_frames)
        seq[0] += len(out)  # keep diverging to fill
        out.append(assemble_answer(primary_video, seq))

    return out[:n_lines]


def solve_query(query_emb: np.ndarray,
                event_texts: List[str],
                event_embs: np.ndarray,
                faiss_index,
                meta: List[List],
                per_video_cache: Dict,
                top_videos: int = 3) -> Tuple[str, List[int], List[Tuple[str, float]]]:
    primary, score = select_video(query_emb, faiss_index, meta, top_videos=top_videos)
    counts: Dict[str, float] = defaultdict(float)
    scores, idx = faiss_index.search(query_emb.astype(np.float32), 200)
    for s, i in zip(scores[0], idx[0]):
        if i < 0:
            continue
        vid, _ = meta[i]
        counts[vid] += float(s) + 0.5
    backup = [(v, s) for v, s in sorted(counts.items(), key=lambda x: -x[1])
              if v != primary][:max(0, top_videos - 1)]

    info = per_video_cache.get(primary)
    if info is None or info["vecs"] is None:
        return primary, [], backup
    scores_per_event = []
    frames_per_event = []
    for e, emb in enumerate(event_embs):
        s = info["vecs"] @ emb.astype(np.float32)
        # small object bias if event text mentions something trackable
        kw = [w for w in event_texts[e].lower().split() if len(w) > 3][:5]
        bias = object_bias(primary, info["frames"], kw)
        s = s + bias
        # optional: penalize keyframes far from the global video peak
        scores_per_event.append(s)
        frames_per_event.append(info["frames"])
    aligned = align_events_dp(scores_per_event, frames_per_event,
                              min_gap=1, repeat_penalty=0.05, back_penalty=0.5)
    return primary, aligned, backup


def make_event_texts(query: str, events: List[dict]) -> List[str]:
    """Build per-event text. Falls back to chunking the query if no events given."""
    if events:
        out = []
        for ev in events:
            if isinstance(ev, dict):
                t = ev.get("text") or ev.get("desc") or ev.get("name") or ""
                if not t:
                    # sometimes event is just a sub-id like "approach"
                    t = ev.get("event") or ev.get("id") or ""
                out.append(f"{query} - {t}".strip(" -"))
            else:
                out.append(f"{query} - {ev}".strip(" -"))
        return out
    # fallback: chunk the query by commas / line breaks
    parts = [p.strip() for p in query.replace(";", ",").split(",") if p.strip()]
    if len(parts) <= 1:
        parts = [query]
    return parts


def load_queries(path: Path):
    """Load queries from either JSON list or JSONL. Auto-detects by content."""
    text = Path(path).read_text()
    stripped = text.lstrip()
    q = []
    if stripped.startswith("["):
        return json.loads(text)
    for line in text.splitlines():
        line = line.strip()
        if line:
            q.append(json.loads(line))
    return q


def run(queries_path: Path, out_path: Path, top_k_videos: int = 3,
        top_n: int = 100, device: str = "cuda"):
    import faiss
    print(f"[trake] loading FAISS index from {INDEX_DIR}")
    faiss_index = faiss.read_index(str(INDEX_DIR / "clip_b32.faiss"))
    with open(INDEX_DIR / "meta.json") as f:
        meta = json.load(f)
    print(f"[trake] index ntotal={faiss_index.ntotal}, meta rows={len(meta)}")

    print(f"[trake] building per-video cache")
    per_video = build_per_video_cache()
    n_videos = sum(1 for v in per_video.values() if v["vecs"] is not None)
    print(f"[trake] per-video cache ready: {n_videos} videos with features")

    print(f"[trake] loading CLIP text encoder on {device}")
    enc = TextEncoder(device=device)

    queries = load_queries(queries_path)
    print(f"[trake] loaded {len(queries)} queries")

    t0 = time.time()
    with open(out_path, "w") as fout:
        for qi, q in enumerate(queries):
            qid = q.get("query_id", str(qi))
            qt = q.get("query_type", "TRAKE").upper()
            if qt != "TRAKE":
                # For non-TRAKE tasks we still emit lines but the solver is a
                # no-op; consumers should use solve.py instead.
                for r in range(1, top_n + 1):
                    fout.write(json.dumps({"query_id": qid, "rank": r,
                                           "answer": ""}) + "\n")
                continue
            text = q.get("query_text") or q.get("description") or ""
            events = q.get("events") or []
            event_texts = make_event_texts(text, events)
            # Build event embeddings batch
            emb_full = enc.encode([text] + event_texts)
            q_emb = emb_full[0:1]
            e_embs = emb_full[1:]
            primary, frames, backup = solve_query(q_emb, event_texts, e_embs,
                                                   faiss_index, meta, per_video,
                                                   top_videos=top_k_videos)
            if not frames:
                # Fallback: emit 100 lines using primary video only
                info = per_video.get(primary)
                if info is not None and len(info["frames"]) >= len(event_texts):
                    idxs = np.linspace(0, len(info["frames"]) - 1,
                                       num=len(event_texts)).astype(int)
                    frames = [int(info["frames"][i]) for i in idxs]
                else:
                    frames = [0] * max(1, len(event_texts))
            answers = generate_top100(primary, frames, backup, per_video, top_n)
            for r, ans in enumerate(answers, 1):
                fout.write(json.dumps({"query_id": qid, "rank": r,
                                       "answer": ans}) + "\n")
            if (qi + 1) % 25 == 0:
                print(f"[trake] {qi+1}/{len(queries)} queries "
                      f"({(time.time()-t0)/(qi+1):.2f}s/q)")
    print(f"[trake] wrote {out_path} in {time.time()-t0:.1f}s")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--queries", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--top-k-videos", type=int, default=3)
    ap.add_argument("--top-n", type=int, default=100)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    run(args.queries, args.out, args.top_k_videos, args.top_n, args.device)


if __name__ == "__main__":
    main()
