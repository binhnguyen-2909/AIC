"""
Ensemble retriever for AIC 2026 KIS / Q&A / TRAKE.

Strategy: BM25/ASR/objects pick candidate videos (text-side), CLIP reranks
within candidates (visual-side). ASR is optional and activates only when
``ensemble_index/asr_index.jsonl`` exists, so old installations remain valid.

Pipeline per query:
  1. BM25 over media-info (title+desc+keywords) with pyvi tokenization.
  2. Object TF-IDF (Vietnamese noun phrase -> OpenImages class) -> video score.
  3. Combine BM25 + OBJ -> candidate video ranking.
  4. For top-N candidate videos (N=50), compute per-frame CLIP scores.
  5. Re-rank frames by fused score = video_score * alpha + clip_frame_score.
  6. Apply temporal coherence within top result video.

Each retrieval returns list of (video_id, frame_idx, score) tuples.
"""
from __future__ import annotations

import json
import math
import os
import pickle
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable

import faiss
import numpy as np
import torch
from pyvi import ViTokenizer
from rank_bm25 import BM25Okapi
from transformers import CLIPModel, CLIPProcessor

ROOT = Path(os.environ.get(
    "AIC_SOLUTION_ROOT", str(Path(__file__).resolve().parent)
))
INDEX_DIR = ROOT / "index"
ENS_DIR = ROOT / "ensemble_index"
MAP_DIR = ROOT.parent / "extracted" / "map-keyframes-aic25-b1" / "map-keyframes"

# ============================================================
# Stopwords / common Vietnamese noise
# ============================================================
VN_STOPWORDS = {
    "là", "của", "và", "có", "cho", "không", "những", "một", "như", "này",
    "thì", "ở", "được", "tại", "với", "nhưng", "để", "khi", "từ", "trong",
    "trên", "vào", "ra", "đến", "lên", "xuống", "nữa", "thêm", "đó", "đây",
    "nào", "gì", "ai", "sao", "thế", "vậy", "thì", "rồi", "sẽ", "đã", "đang",
    "vẫn", "còn", "lại", "rất", "quá", "khá", "hơi", "hay", "làm", "đi",
    "thấy", "biết", "thành", "thể", "loại", "sự", "việc",
    "tìm", "video", "video", "video", "đây", "nào", "cần",
    "xem", "thêm", "nhiều", "ít",
}

# Vietnamese noun -> OpenImages class names mapping.
VN_TO_OPENIMAGES = {
    "ô_tô": ["Car"], "xe_hơi": ["Car"], "xe": ["Car", "Land vehicle"],
    "xe_máy": ["Motorcycle"], "xe_đạp": ["Bicycle"], "xe_buýt": ["Bus"],
    "xe_tải": ["Truck"], "tàu": ["Train", "Boat"], "thuyền": ["Boat"],
    "máy_bay": ["Airplane"], "trực_thăng": ["Helicopter"],
    "người": ["Person", "Human face"], "đàn_ông": ["Man"], "phụ_nữ": ["Woman"],
    "trẻ_em": ["Child", "Baby"], "em_bé": ["Baby"],
    "diễn_giả": ["Person", "Human face", "Microphone"],
    "chó": ["Dog"], "mèo": ["Cat"], "ngựa": ["Horse"], "bò": ["Cow"],
    "gà": ["Chicken"], "vịt": ["Duck"], "cá": ["Fish"], "chim": ["Bird"],
    "khỉ": ["Monkey"], "voi": ["Elephant"], "hổ": ["Tiger"], "sư_tử": ["Lion"],
    "tòa_nhà": ["Building", "Tower", "Skyscraper"], "nhà": ["Building", "House"],
    "tháp": ["Tower", "Skyscraper"], "cầu": ["Bridge"],
    "đường": ["Street light", "Land vehicle"], "phố": ["Street light", "Building"],
    "sân": ["Sports equipment", "Person"], "bãi_biển": ["Beach", "Sea"],
    "biển": ["Sea", "Beach"], "núi": ["Mountain"],
    "rừng": ["Tree", "Plant"], "cây": ["Tree", "Plant"], "hoa": ["Flower"],
    "vườn": ["Plant", "Tree"], "công_viên": ["Tree", "Plant"],
    "trường": ["Building", "School bus"], "trường_học": ["Building"],
    "bệnh_viện": ["Building"],
    "bóng_đá": ["Sports equipment", "Ball"], "bóng_rổ": ["Ball", "Sports equipment"],
    "quả_bóng": ["Ball"], "sân_vận_động": ["Sports equipment", "Person"],
    "áo": ["Clothing"], "quần": ["Clothing"], "mũ": ["Hat"],
    "giày": ["Footwear"], "túi": ["Bag"], "kính": ["Glasses"],
    "điện_thoại": ["Mobile phone", "Telephone"], "laptop": ["Laptop"],
    "máy_tính": ["Laptop", "Computer monitor"], "bàn": ["Table"],
    "ghế": ["Chair"], "cửa": ["Door"], "cửa_sổ": ["Window"],
    "đèn": ["Lamp", "Light", "Street light"], "bóng_đèn": ["Lamp", "Light"],
    "cơm": ["Food"], "bánh": ["Food"], "nước": ["Drink", "Bottle"],
    "trà": ["Drink"], "cà_phê": ["Drink"], "bia": ["Beer"],
    "lửa": ["Fire"], "khói": ["Smoke"], "mưa": ["Rain"], "tuyết": ["Snow"],
    "nắng": ["Sunlight"],
    "hoa_hậu": ["Person", "Woman", "Crown"], "ca_sĩ": ["Person", "Microphone"],
    "diễn_giả": ["Person", "Microphone"],
    "họp_báo": ["Person", "Microphone"], "hội_thảo": ["Person", "Building"],
    "hội_nghị": ["Person", "Building"],
    "ô_tô": ["Car"], "xe_ô_tô": ["Car"], "xe_điện": ["Car"],
    "áo_dài": ["Clothing"], "áo_đỏ": ["Clothing"], "áo_trắng": ["Clothing"],
    "áo_xanh": ["Clothing"],
}


# ============================================================
# Helpers
# ============================================================
def tokenize_vn(text: str) -> list[str]:
    text = text.lower()
    text = re.sub(r"https?://\S+", " ", text)
    text = re.sub(r"[^\w\sàáảãạăắằẳẵặâấầẩẫậèéẻẽẹêếềểễệìíỉĩịòóỏõọôốồổỗộơớờởỡợùúủũụưứừửữựỳýỷỹỵđ]",
                 " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    text = ViTokenizer.tokenize(text)
    out = []
    for t in text.split():
        if len(t) <= 1:
            continue
        if t in VN_STOPWORDS:
            continue
        out.append(t)
    return out


def extract_query_objects(tokens: list[str]) -> list[str]:
    cls = set()
    for t in tokens:
        if t in VN_TO_OPENIMAGES:
            for c in VN_TO_OPENIMAGES[t]:
                cls.add(c)
    for t in tokens:
        if t[0].isupper():
            cls.add(t.capitalize())
    return list(cls)


# ============================================================
# Main ensemble
# ============================================================
class EnsembleRetriever:
    def __init__(self, device: str = "cuda", load_clip: bool = True,
                 load_siglip: bool = False):
        self.device = device

        # CLIP index
        self.index = faiss.read_index(str(INDEX_DIR / "clip_b32.faiss"))
        with open(INDEX_DIR / "meta.json") as f:
            self.meta = json.load(f)
        self.meta_videos = [m[0] for m in self.meta]
        # vid -> sorted list of frame_idx (global), for frame iteration
        self.vid_to_fidx = defaultdict(list)
        for fidx, (vid, fi) in enumerate(self.meta):
            self.vid_to_fidx[vid].append(fidx)
        self.video_ids = sorted(set(self.meta_videos))

        # CLIP is optional because solver_v7/v8/v9 each own a text encoder.
        # Loading another copy here wastes several GB and can cause OOM before
        # inference starts.  Keep it enabled for the standalone ensemble.
        self.clip = None
        self.proc = None
        if load_clip:
            self.clip = CLIPModel.from_pretrained(
                "openai/clip-vit-base-patch32").to(self.device).eval()
            self.proc = CLIPProcessor.from_pretrained(
                "openai/clip-vit-base-patch32")
            if device == "cuda":
                self.clip = self.clip.half()

        # Optional separately-built SigLIP2 channel.  It is opt-in so the
        # default path keeps one vision tower and remains memory-safe.
        self.siglip = None
        if load_siglip:
            try:
                from siglip_channel import SigLIPChannel
                self.siglip = SigLIPChannel(device=device)
            except Exception as exc:
                print(f"SigLIP2 channel unavailable: {exc}", flush=True)

        # BM25
        with open(ENS_DIR / "bm25.pkl", "rb") as f:
            bm25p = pickle.load(f)
        self.bm25_video_ids = bm25p["video_ids"]
        self.bm25 = BM25Okapi(bm25p["tokenized"])
        self.bm25.idf = bm25p["idf"]
        self.bm25_vid2pos = {v: i for i, v in enumerate(self.bm25_video_ids)}

        # Objects
        with open(ENS_DIR / "objects.pkl", "rb") as f:
            objp = pickle.load(f)
        self.inv_class = objp["inv_class"]
        self.video_class_score = objp["video_class_score"]
        self.video_class_count = objp["video_class_count"]
        self.class_idf = self._compute_class_idf()
        self.video_obj_vec = self._compute_video_obj_vecs()

        # Optional timestamped Vietnamese ASR branch.  It is deliberately
        # loaded only when an index exists: building ASR requires decoding all
        # MP4 audio and should never be an implicit production startup step.
        self.asr_bm25 = None
        self.asr_owners = []
        self._asr_frame_hits = {}
        self._map_cache = {}
        # Prefer the atomically merged full-corpus index; keep the legacy
        # filename as a fallback for older dataset-free deployments.
        asr_path = ENS_DIR / "asr_full.jsonl"
        if not asr_path.exists():
            asr_path = ENS_DIR / "asr_index.jsonl"
        if asr_path.exists():
            try:
                asr_rows = [json.loads(line) for line in asr_path.open()
                            if line.strip()]
                covered = {row.get("video_id") for row in asr_rows}
                if covered != set(self.video_ids):
                    print(f"ASR index incomplete: {len(covered)}/{len(self.video_ids)} videos; skipped", flush=True)
                    asr_rows = []
                asr_docs = []
                for row in asr_rows:
                    for chunk in row.get("chunks", []):
                        toks = tokenize_vn(chunk.get("text", ""))
                        if toks:
                            asr_docs.append(toks)
                            self.asr_owners.append((row["video_id"], chunk))
                if asr_docs:
                    self.asr_bm25 = BM25Okapi(asr_docs)
                    print(f"ASR index loaded: {len(asr_docs)} chunks", flush=True)
            except Exception as exc:
                print(f"ASR index unavailable: {exc}", flush=True)

        self.ocr_bm25 = None
        self.ocr_owners = []
        self._ocr_frame_hits = {}
        ocr_path = ENS_DIR / "ocr_index.jsonl"
        if ocr_path.exists():
            try:
                ocr_rows = [json.loads(line) for line in ocr_path.open()
                            if line.strip()]
                covered = {row.get("video_id") for row in ocr_rows}
                if covered != set(self.video_ids):
                    print(f"OCR index incomplete: {len(covered)}/{len(self.video_ids)} videos; skipped", flush=True)
                    ocr_rows = []
                ocr_docs = []
                for row in ocr_rows:
                    for frame in row.get("frames", []):
                        toks = tokenize_vn(frame.get("text", ""))
                        if toks:
                            ocr_docs.append(toks)
                            self.ocr_owners.append((row["video_id"], frame))
                if ocr_docs:
                    self.ocr_bm25 = BM25Okapi(ocr_docs)
                    print(f"OCR index loaded: {len(ocr_docs)} frames", flush=True)
            except Exception as exc:
                print(f"OCR index unavailable: {exc}", flush=True)

    def _compute_class_idf(self) -> dict[str, float]:
        n_videos = len(self.video_ids)
        df = Counter()
        for vid in self.video_ids:
            classes = set(self.video_class_count.get(vid, {}).keys())
            for c in classes:
                df[c] += 1
        return {c: math.log(1 + n_videos / (1 + d)) for c, d in df.items()}

    def _compute_video_obj_vecs(self) -> dict[str, dict[str, float]]:
        out = {}
        for vid in self.video_ids:
            counts = self.video_class_count.get(vid, {})
            if not counts:
                out[vid] = {}
                continue
            total = sum(counts.values())
            vec = {c: (n / total) * self.class_idf.get(c, 1.0)
                   for c, n in counts.items()}
            norm = math.sqrt(sum(v * v for v in vec.values()))
            if norm > 0:
                vec = {k: v / norm for k, v in vec.items()}
            out[vid] = vec
        return out

    def _object_video_score(self, query_classes: list[str]) -> dict[str, float]:
        if not query_classes:
            return {}
        q_counts = Counter(query_classes)
        q_total = sum(q_counts.values())
        q_vec = {c: (n / q_total) * self.class_idf.get(c, 1.0)
                 for c, n in q_counts.items()}
        q_norm = math.sqrt(sum(v * v for v in q_vec.values()))
        if q_norm > 0:
            q_vec = {k: v / q_norm for k, v in q_vec.items()}
        scores = {}
        for vid, vvec in self.video_obj_vec.items():
            if not vvec:
                continue
            s = 0.0
            if len(q_vec) < len(vvec):
                for k, qv in q_vec.items():
                    if k in vvec:
                        s += qv * vvec[k]
            else:
                for k, vv in vvec.items():
                    if k in q_vec:
                        s += q_vec[k] * vv
            if s > 0:
                scores[vid] = s
        return scores

    @torch.no_grad()
    def encode_text(self, text: str | list[str]) -> np.ndarray:
        if self.clip is None or self.proc is None:
            raise RuntimeError("EnsembleRetriever was created with load_clip=False")
        texts = [text] if isinstance(text, str) else list(text)
        inputs = self.proc(text=texts, return_tensors="pt", padding=True,
                           truncation=True).to(self.device)
        for k in inputs:
            if inputs[k].dtype == torch.float32:
                inputs[k] = inputs[k].half()
        out = self.clip.get_text_features(**inputs)
        if hasattr(out, "pooler_output"):
            feats = out.pooler_output
        elif hasattr(out, "text_embeds"):
            feats = out.text_embeds
        else:
            feats = out[0] if isinstance(out, tuple) else out
        feats = feats / feats.norm(dim=-1, keepdim=True)
        return feats.cpu().numpy().astype(np.float32)

    def clip_search(self, text: str, top_k: int,
                    video_filter: set[str] | None = None,
                    per_video_top: int = 5
                    ) -> list[tuple[int, float]]:
        """Search top_k CLIP frames. If video_filter given, only return frames
        from those videos.

        A global FAISS overfetch is insufficient here: a valid candidate's
        frames can all fall below the global top-N.  For filtered searches,
        score the cached vectors of every candidate and retain a bounded top
        set per video.
        """
        q = self.encode_text(text)
        if video_filter:
            self._load_per_video()
            out = []
            for vid in sorted(video_filter):
                info = self.per_video.get(vid, {})
                vecs = info.get("vecs")
                rows = info.get("rows")
                if vecs is None or rows is None or len(vecs) == 0:
                    continue
                vals = np.asarray(vecs, dtype=np.float32) @ q[0]
                keep = min(max(1, int(per_video_top)), len(vals))
                order = np.argpartition(-vals, keep - 1)[:keep]
                order = order[np.argsort(-vals[order])]
                for pos in order:
                    out.append((int(rows[int(pos)]), float(vals[int(pos)])))
            out.sort(key=lambda x: -x[1])
            return out[:int(top_k)]

        overfetch = top_k
        scores, idx = self.index.search(
            q, min(overfetch, self.index.ntotal))
        out = []
        for s, i in zip(scores[0], idx[0]):
            if i < 0:
                continue
            vid = self.meta[i][0]
            if video_filter and vid not in video_filter:
                continue
            out.append((int(i), float(s)))
            if len(out) >= top_k:
                break
        return out

    def refine_source_frames(self, text: str, results,
                             max_videos: int = 3, window: int = 150,
                             per_video: int = 5):
        """Decode and CLIP-rerank dense source frames near top candidates."""
        if self.clip is None or self.proc is None or not results:
            return list(results)
        try:
            from dense_refine import DenseFrameRefiner
        except ImportError:
            from solution.dense_refine import DenseFrameRefiner

        query = self.encode_text(text)[0]
        refiner = DenseFrameRefiner(self.clip, self.proc, self.device)
        by_video = {}
        for vid, fi, score in results:
            by_video.setdefault(vid, (int(fi), float(score)))
        refined = []
        refined_videos = set()
        for vid, (center, base_score) in list(by_video.items())[:max_videos]:
            hits = refiner.refine(vid, center, query, window=window,
                                  top_k=per_video)
            if not hits:
                continue
            refined_videos.add(vid)
            for frame, local_score in hits:
                refined.append((vid, int(frame),
                                float(base_score + 0.1 * local_score)))
        out = []
        seen = set()
        for row in refined + list(results):
            key = (row[0], int(row[1]))
            if key in seen:
                continue
            seen.add(key)
            out.append((row[0], int(row[1]), float(row[2])))
            if len(out) >= len(results):
                break
        return out

    def bm25_search(self, text: str, top_k: int) -> list[tuple[str, float]]:
        tokens = tokenize_vn(text)
        if not tokens:
            return []
        scores = self.bm25.get_scores(tokens)
        order = np.argsort(-scores)[:top_k]
        return [(self.bm25_video_ids[i], float(scores[i])) for i in order
                if scores[i] > 0]

    def asr_search(self, text: str, top_k: int) -> list[tuple[str, float]]:
        if self.asr_bm25 is None:
            return []
        tokens = tokenize_vn(text)
        if not tokens:
            return []
        scores = self.asr_bm25.get_scores(tokens)
        best = {}
        for score, (vid, chunk) in zip(scores, self.asr_owners):
            if score > best.get(vid, (-1.0, None))[0]:
                best[vid] = (float(score), chunk)
        ranked = sorted(best.items(), key=lambda x: -x[1][0])[:top_k]
        self._asr_frame_hits = {
            vid: self._frame_for_time(vid, chunk.get("start"))
            for vid, (_score, chunk) in ranked
        }
        return [(vid, score) for vid, (score, _chunk) in ranked]

    def _frame_for_time(self, video_id: str, seconds) -> int | None:
        """Map an ASR timestamp to the nearest original frame index."""
        if seconds is None:
            return None
        if video_id not in self._map_cache:
            path = MAP_DIR / f"{video_id}.csv"
            rows = []
            if path.exists():
                try:
                    with path.open() as f:
                        next(f, None)
                        for line in f:
                            p = line.strip().split(",")
                            if len(p) >= 4:
                                rows.append((float(p[1]), int(p[3])))
                except (OSError, ValueError):
                    rows = []
            self._map_cache[video_id] = rows
        rows = self._map_cache[video_id]
        if not rows:
            return None
        times = np.asarray([x[0] for x in rows], dtype=np.float32)
        pos = int(np.searchsorted(times, float(seconds), side="left"))
        pos = min(pos, len(rows) - 1)
        if pos and abs(times[pos - 1] - seconds) < abs(times[pos] - seconds):
            pos -= 1
        return rows[pos][1]

    def ocr_search(self, text: str, top_k: int) -> list[tuple[str, float]]:
        if self.ocr_bm25 is None:
            return []
        tokens = tokenize_vn(text)
        if not tokens:
            return []
        scores = self.ocr_bm25.get_scores(tokens)
        best = {}
        for score, (vid, frame) in zip(scores, self.ocr_owners):
            if score > best.get(vid, (-1.0, None))[0]:
                best[vid] = (float(score), frame)
        ranked = sorted(best.items(), key=lambda x: -x[1][0])[:top_k]
        self._ocr_frame_hits = {vid: int(frame["frame_idx"])
                                for vid, (_score, frame) in ranked}
        return [(vid, score) for vid, (score, _frame) in ranked]

    # --------------------------------------------------------
    # Main retrieval: BM25 -> candidates; CLIP reranks frames within.
    # --------------------------------------------------------
    def retrieve(
        self,
        text: str,
        top_k: int = 100,
        n_candidates: int = 50,   # number of candidate videos to consider
        w_bm25: float = 0.7,
        w_obj: float = 0.3,
        w_asr: float = 0.2,
        w_ocr: float = 0.2,
        w_clip_video: float = 0.0,  # CLIP-as-reranker is OFF by default
        clip_per_vid: int = 5,     # top-K frames per video to keep
        clip_text: str | None = None,
        w_siglip: float = 0.0,
        siglip_text: str | None = None,
    ) -> list[tuple[str, int, float]]:
        # ---- BM25 candidates ----
        self._asr_frame_hits = {}
        self._ocr_frame_hits = {}
        bm25_hits = self.bm25_search(text, n_candidates * 2)
        asr_hits = self.asr_search(text, n_candidates * 2) if self.asr_bm25 else []
        ocr_hits = self.ocr_search(text, n_candidates * 2) if self.ocr_bm25 else []
        siglip_hits = []
        if self.siglip is not None and w_siglip > 0:
            try:
                siglip_hits = self.siglip.search(
                    siglip_text or text, n_candidates * clip_per_vid * 3
                )
            except Exception as exc:
                print(f"SigLIP2 search unavailable: {exc}", flush=True)
        bm25_n = {}
        if bm25_hits:
            bs = np.array([s for _, s in bm25_hits])
            bmin, bmax = bs.min(), bs.max()
            if bmax > bmin:
                bm25_n = {v: (s - bmin) / (bmax - bmin) for v, s in bm25_hits}
            else:
                bm25_n = {v: 1.0 for v, _ in bm25_hits}

        # ---- Object candidates ----
        toks = tokenize_vn(text)
        for w in text.split():
            if w[0:1].isupper() and len(w) > 2:
                toks.append(w)
        qcls = extract_query_objects(toks)
        obj_video_scores = self._object_video_score(qcls)
        if obj_video_scores:
            arr = np.array(list(obj_video_scores.values()))
            omin, omax = arr.min(), arr.max()
            if omax > omin:
                obj_video_n = {v: (s - omin) / (omax - omin)
                               for v, s in obj_video_scores.items()}
            else:
                obj_video_n = {v: 1.0 for v in obj_video_scores}
        else:
            obj_video_n = {}

        # ---- Optional ASR candidates ----
        asr_video_n = {}
        if asr_hits:
            arr = np.array([s for _, s in asr_hits])
            amin, amax = arr.min(), arr.max()
            if amax > amin:
                asr_video_n = {v: (s - amin) / (amax - amin)
                               for v, s in asr_hits}
        else:
            asr_video_n = {v: 1.0 for v, _ in asr_hits}

        ocr_video_n = {}
        if ocr_hits:
            arr = np.array([s for _, s in ocr_hits])
            omin, omax = arr.min(), arr.max()
            if omax > omin:
                ocr_video_n = {v: (s - omin) / (omax - omin)
                               for v, s in ocr_hits}
            else:
                ocr_video_n = {v: 1.0 for v, _ in ocr_hits}

        # ---- Optional SigLIP2 visual channel ----
        siglip_video_n = {}
        if siglip_hits:
            per_video_scores = defaultdict(list)
            for fidx, score in siglip_hits:
                per_video_scores[self.siglip.meta[fidx][0]].append(float(score))
            values = {
                vid: float(np.mean(sorted(vals, reverse=True)[:5]))
                for vid, vals in per_video_scores.items()
            }
            arr = np.asarray(list(values.values()), dtype=np.float32)
            smin, smax = float(arr.min()), float(arr.max())
            if smax > smin:
                siglip_video_n = {v: (s - smin) / (smax - smin)
                                  for v, s in values.items()}
            else:
                siglip_video_n = {v: 1.0 for v in values}

        # ---- Per-video fused (text + object) ----
        all_videos = (set(bm25_n) | set(obj_video_n) | set(asr_video_n)
                      | set(ocr_video_n) | set(siglip_video_n))
        video_text_score = {}
        for v in all_videos:
            video_text_score[v] = (
                w_bm25 * bm25_n.get(v, 0.0) + w_obj * obj_video_n.get(v, 0.0)
                + w_asr * asr_video_n.get(v, 0.0)
                + w_ocr * ocr_video_n.get(v, 0.0)
                + w_siglip * siglip_video_n.get(v, 0.0)
            )

        # Sort videos by text score, take top N candidates
        ranked_videos = sorted(video_text_score.items(),
                               key=lambda x: -x[1])
        # If too few text candidates, fill deterministically.  Random padding
        # makes submissions non-reproducible and, when the text signal is
        # empty, can change the score between identical runs.
        if len(ranked_videos) < n_candidates:
            extras = sorted(v for v in self.video_ids if v not in video_text_score)
            for v in extras[:n_candidates - len(ranked_videos)]:
                ranked_videos.append((v, 0.0))
        candidates = ranked_videos[:n_candidates]
        cand_set = {v for v, _ in candidates}

        # ---- CLIP rerank: search within candidates (optional) ----
        if w_clip_video > 0:
            clip_hits = self.clip_search(
                clip_text or text, n_candidates * clip_per_vid,
                video_filter=cand_set, per_video_top=clip_per_vid
            )
            if clip_hits:
                cs = np.array([s for _, s in clip_hits])
                cmin, cmax = cs.min(), cs.max()
                if cmax > cmin:
                    clip_n = {fidx: (s - cmin) / (cmax - cmin)
                              for fidx, s in clip_hits}
                else:
                    clip_n = {fidx: 1.0 for fidx, _ in clip_hits}
            else:
                clip_n = {}
            # Aggregate per-video CLIP score (max)
            clip_video_score = {}
            for fidx, cn in clip_n.items():
                vid = self.meta[fidx][0]
                if cn > clip_video_score.get(vid, 0.0):
                    clip_video_score[vid] = cn
        else:
            clip_n = {}
            clip_video_score = {}

        # Final per-video score = text_score + clip_video_bonus
        video_final = {}
        for v, ts in candidates:
            cv = clip_video_score.get(v, 0.0)
            video_final[v] = ts + w_clip_video * cv

        # ---- Per-frame final score ----
        # For each video in candidates, pick top clip_per_vid frames
        # by CLIP score, propagate video score to each.
        per_video_frames = defaultdict(list)
        for fidx, cn in clip_n.items():
            vid = self.meta[fidx][0]
            if vid in cand_set:
                per_video_frames[vid].append((fidx, cn))
        if siglip_hits:
            ss = np.asarray([score for _fidx, score in siglip_hits],
                            dtype=np.float32)
            smin, smax = float(ss.min()), float(ss.max())
            for sfidx, score in siglip_hits:
                cn = ((float(score) - smin) / (smax - smin)
                      if smax > smin else 1.0)
                vid = self.siglip.meta[sfidx][0]
                if vid in cand_set:
                    # Negative ids tag SigLIP rows without colliding with the
                    # production CLIP index; conversion happens at output.
                    per_video_frames[vid].append((-(sfidx + 1), cn))
        # If a video has zero CLIP frames, take a sample of frames from it
        for v, _ in candidates:
            if not per_video_frames[v]:
                fidxs = self.vid_to_fidx.get(v, [])
                if fidxs:
                    asr_frame = self._asr_frame_hits.get(v)
                    ocr_frame = self._ocr_frame_hits.get(v)
                    target_frame = asr_frame if asr_frame is not None else ocr_frame
                    if target_frame is not None:
                        frame_ids = np.asarray([self.meta[p][1] for p in fidxs], dtype=np.int64)
                        nearest = int(np.argmin(np.abs(frame_ids - int(target_frame))))
                        per_video_frames[v] = [(fidxs[nearest], 1.0)]
                        continue
                    # Spread picks: middle + quarter + 3-quarter
                    n = len(fidxs)
                    picks = [fidxs[n // 2]]
                    if n > 2:
                        picks.append(fidxs[n // 4])
                        picks.append(fidxs[3 * n // 4])
                    per_video_frames[v] = [(p, 0.0) for p in picks]

        # Build final ranked list
        ranked_videos2 = sorted(video_final.items(), key=lambda x: -x[1])
        out = []
        # Put several high-confidence frames from the best few videos early,
        # then broaden coverage.  The official score rewards the first answer
        # at R@1/R@5/R@20, while the tail still needs backup videos for R@100.
        ranked_frames = []
        for v, vs in ranked_videos2:
            frames = per_video_frames.get(v, [])
            frames.sort(key=lambda x: -x[1])
            ranked_frames.append((v, vs, frames[:clip_per_vid]))
        priority = min(3, len(ranked_frames))
        for round_idx in range(clip_per_vid):
            for v, vs, frames in ranked_frames[:priority]:
                if round_idx >= len(frames):
                    continue
                fidx, cn = frames[round_idx]
                # final frame score = video_score + intra-video CLIP norm
                score = vs + 0.1 * cn
                if fidx < 0:
                    sfidx = -fidx - 1
                    vid, fi = self.siglip.meta[sfidx]
                else:
                    vid, fi = self.meta[fidx]
                out.append((vid, int(fi), float(score)))
                if len(out) >= top_k:
                    return out
        for round_idx in range(clip_per_vid):
            for v, vs, frames in ranked_frames[priority:]:
                if round_idx >= len(frames):
                    continue
                fidx, cn = frames[round_idx]
                score = vs + 0.1 * cn
                if fidx < 0:
                    sfidx = -fidx - 1
                    vid, fi = self.siglip.meta[sfidx]
                else:
                    vid, fi = self.meta[fidx]
                out.append((vid, int(fi), float(score)))
                if len(out) >= top_k:
                    return out
        return out

    def _load_per_video(self):
        if hasattr(self, "per_video"):
            return
        path = INDEX_DIR / "per_video.pkl"
        self.per_video = {}
        if path.exists():
            with path.open("rb") as f:
                self.per_video = pickle.load(f)

    @staticmethod
    def _align_event_scores(frames, event_scores):
        """Monotonic DP alignment over all keyframes in one video."""
        frames = np.asarray(frames, dtype=np.int64)
        if not event_scores or len(frames) == 0:
            return []
        dp = np.asarray(event_scores[0], dtype=np.float32).copy()
        backpointers = []
        for scores in event_scores[1:]:
            scores = np.asarray(scores, dtype=np.float32)
            prefix = np.maximum.accumulate(dp)
            prefix_idx = np.zeros(len(dp), dtype=np.int64)
            best = -np.inf
            best_i = 0
            for i, value in enumerate(dp):
                if value > best:
                    best, best_i = float(value), i
                prefix_idx[i] = best_i
            # The next event only needs a strictly later keyframe.  Using
            # ``frames - 1`` here accidentally skipped adjacent source-frame
            # entries (and could backtrack to the current index).
            allowed = np.arange(len(frames), dtype=np.int64) - 1
            valid = allowed >= 0
            safe_allowed = np.clip(allowed, 0, len(frames) - 1)
            cur = scores + prefix[safe_allowed]
            # A position is invalid only when no earlier frame exists.
            cur[~valid] = -np.inf
            backpointers.append(prefix_idx[safe_allowed])
            dp = cur
        pos = int(np.nanargmax(dp))
        chosen = [pos]
        for bp in reversed(backpointers):
            pos = int(bp[pos])
            chosen.append(pos)
        chosen.reverse()
        return [int(frames[i]) for i in chosen]

    def trake(self, text: str, events: list[dict], top_k: int = 100,
              event_texts: list[str] | None = None,
              clip_text: str | None = None) -> list[str]:
        """Retrieve candidates, align every event, and emit k-best hypotheses.

        TRAKE is scored on the complete answer, not just the video.  We
        therefore retain alignment confidence when ranking videos and
        interleave base hypotheses from several videos before generating local
        alternatives.  This lets a strong second candidate contribute at R@5
        or R@20 instead of waiting behind dozens of offsets for the first.
        """
        if self.clip is None:
            raise RuntimeError("EnsembleRetriever.trake requires load_clip=True")
        top_k = max(1, min(100, int(top_k)))
        if not events:
            return []
        self._load_per_video()
        event_texts = event_texts or [
            (e.get("text") or e.get("desc") or e.get("name") or e.get("event", ""))
            for e in events
        ]
        if len(event_texts) != len(events):
            return []

        # Candidate union is intentionally wide: event alignment can rescue a
        # video whose metadata score is weaker than the top lexical hit.
        raw = self.retrieve(text, clip_text=clip_text,
                            top_k=100,
                            n_candidates=50, w_clip_video=0.25)
        candidates = []
        seen = set()
        for vid, _fi, score in raw:
            if vid not in seen:
                candidates.append((vid, float(score)))
                seen.add(vid)
        aligned = []
        query_parts = []
        clip_query = clip_text or text
        for ev in event_texts:
            query_parts.extend([ev, f"a frame from a video showing {ev}",
                                 f"{clip_query} - {ev}".strip(" -")])
        embeddings = self.encode_text(query_parts)
        for candidate_rank, (vid, video_score) in enumerate(candidates[:50]):
            info = self.per_video.get(vid, {})
            frames = np.asarray(info.get("frames", []), dtype=np.int64)
            vecs = info.get("vecs")
            if frames is None or vecs is None or len(frames) == 0:
                continue
            scores = []
            for i in range(len(event_texts)):
                ev_emb = embeddings[i * 3:(i + 1) * 3]
                s = (vecs @ ev_emb.T).max(axis=1)
                # A small neighbourhood bonus is restricted to TRAKE; it
                # rewards sustained evidence without changing KIS/QA ranking.
                if len(s) >= 3:
                    neigh = np.maximum(np.r_[s[0], s[:-1]], np.r_[s[1:], s[-1]])
                    s = 0.85 * s + 0.15 * neigh
                scores.append(s)
            seq = self._align_event_scores(frames, scores)
            if len(seq) == len(event_texts):
                positions = [int(np.argmin(np.abs(frames - frame)))
                             for frame in seq]
                align_score = float(np.mean([
                    scores[i][positions[i]] for i in range(len(scores))
                ]))
                # Prefer temporally coherent event chains when visual
                # evidence is otherwise similar.  This is a soft penalty:
                # long videos and genuinely long events remain eligible, but
                # accidental matches scattered across the whole timeline do
                # not dominate the TRAKE ranking.
                span = max(0, int(seq[-1]) - int(seq[0])) if seq else 0
                coherent_align = align_score - 0.02 * math.log1p(span / 30.0)
                aligned.append((vid, seq, video_score, coherent_align,
                                frames, candidate_rank))
        if not aligned:
            return []

        # Normalize both priors before combining.  The alignment term is a
        # real event-level signal; it must be able to reorder candidates.
        vs = np.asarray([x[2] for x in aligned], dtype=np.float32)
        als = np.asarray([x[3] for x in aligned], dtype=np.float32)
        def norm(values):
            lo, hi = float(values.min()), float(values.max())
            return ((values - lo) / (hi - lo)) if hi > lo else np.ones_like(values)
        fused = 0.55 * norm(vs) + 0.45 * norm(als)
        order = np.argsort(-fused, kind="stable")
        aligned = [aligned[int(i)] + (float(fused[int(i)]),)
                   for i in order]
        out = []

        def clamp_sequence(seq, frames):
            lo, hi = int(frames[0]), int(frames[-1])
            values = [min(hi, max(lo, int(x))) for x in seq]
            for i in range(1, len(values)):
                values[i] = max(values[i], values[i - 1] + 1)
            # If a large jump made the tail exceed the video, shift the whole
            # sequence back while preserving its order.
            if values and values[-1] > hi:
                shift = values[-1] - hi
                values = [max(lo, x - shift) for x in values]
            return values

        def add(vid, seq, frames):
            seq = clamp_sequence(seq, frames)
            line = f"{vid}, " + ", ".join(str(int(x)) for x in seq)
            if line not in out and len(out) < top_k:
                out.append(line)

        # Base sequence from every candidate first: a correct backup video can
        # now enter the early operating points.
        for vid, seq, _vs, _as, frames, _rank, _fused in aligned:
            add(vid, seq, frames)
            if len(out) >= top_k:
                return out

        # Event-specific local alternatives.  Varying one event at a time is
        # much more useful than shifting every event by the same offset.
        variants = []
        for vid, seq, _vs, _as, frames, _rank, _fused in aligned[:20]:
            base = [int(x) for x in seq]
            for event_idx in range(len(base)):
                for delta in (1, -1, 2, -2, 3, -3, 5, -5, 8, -8, 13, -13):
                    candidate = base[:]
                    candidate[event_idx] += delta
                    variants.append((vid, candidate, frames))
        for vid, seq, frames in variants:
            add(vid, seq, frames)
            if len(out) >= top_k:
                return out

        # Deterministic tail fill if there are fewer unique local hypotheses.
        primary_vid, primary, _vs, _as, primary_frames, _rank, _fused = aligned[0]
        offset = 1
        while len(out) < top_k and offset <= 10000:
            seq = [max(0, int(f) + offset) for f in primary]
            add(primary_vid, seq, primary_frames)
            offset += 1
        return out


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--probe", action="store_true")
    ap.add_argument("--queries")
    ap.add_argument("--out")
    ap.add_argument("--top-k", type=int, default=100)
    args = ap.parse_args()

    print("Loading ensemble...")
    er = EnsembleRetriever()

    if args.probe:
        samples = [
            "Tìm video về một diễn giả mặc áo đỏ phát biểu tại một cuộc họp báo ngoài trời",
            "a person opening a laptop",
            "đám cưới cô dâu chú rể trong nhà thờ",
        ]
        for q in samples:
            print("\nQUERY:", q)
            res = er.retrieve(q, top_k=10)
            for vid, fi, s in res:
                print(f"  {vid} frame={fi} score={s:.4f}")
    elif args.queries and args.out:
        with open(args.queries) as f:
            qs = [json.loads(l) for l in f if l.strip()]
        with open(args.out, "w") as fout:
            for q in qs:
                qid = q.get("query_id", "?")
                qt = q.get("query_type", "KIS")
                text = q.get("query_text", q.get("description", ""))
                events = q.get("events", [])
                ans = ""
                if qt == "TRAKE":
                    res = er.retrieve(text, top_k=100)
                    lines = []
                    for rank, (vid, fi, s) in enumerate(res, 1):
                        ef = [fi]
                        for ev in events[1:]:
                            evtext = f"{text} - {ev.get('desc', ev.get('name', ''))}"
                            sub = er.retrieve(evtext, top_k=20,
                                              n_candidates=20,
                                              clip_per_vid=3)
                            sub_filt = [(v, f, ss) for v, f, ss in sub
                                        if v == vid]
                            if sub_filt:
                                ef.append(sub_filt[0][1])
                            else:
                                ef.append(fi)
                        line = f"{vid}, " + ", ".join(str(x) for x in ef)
                        lines.append(line)
                elif qt == "QA":
                    res = er.retrieve(text, top_k=args.top_k)
                    lines = [f"{vid}, {fi}, {ans}"
                             for vid, fi, _ in res]
                else:
                    res = er.retrieve(text, top_k=args.top_k)
                    lines = [f"{vid}, {fi}" for vid, fi, _ in res]
                for rank, line in enumerate(lines, 1):
                    fout.write(json.dumps(
                        {"query_id": qid, "rank": rank,
                         "answer": line}) + "\n")
        print(f"Wrote {args.out}")
