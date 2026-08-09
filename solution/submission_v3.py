"""
Enhanced retriever with Vietnamese->English translation + multi-template ensemble.
This is the production-grade version for the >90% accuracy goal.
"""
import os
import sys
import json
import argparse
from pathlib import Path
from typing import List, Tuple
import numpy as np
import faiss
import torch

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

ROOT = Path(os.environ.get("AIC_ROOT", str(Path(__file__).resolve().parents[1])))
INDEX_DIR = ROOT / "solution" / "index"


class CLIPRetriever:
    def __init__(self, device="cuda"):
        self.device = device
        from transformers import CLIPModel, CLIPProcessor
        self.index = faiss.read_index(str(INDEX_DIR / "clip_b32.faiss"))
        with open(INDEX_DIR / "meta.json") as f:
            self.meta = json.load(f)
        self.by_vid = {}
        for i, (vid, fi) in enumerate(self.meta):
            self.by_vid.setdefault(vid, []).append((i, int(fi)))
        self.model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device).eval()
        self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        if device == "cuda":
            self.model = self.model.half()

    @torch.no_grad()
    def encode_text(self, texts):
        if isinstance(texts, str):
            texts = [texts]
        inputs = self.processor(text=texts, return_tensors="pt", padding=True, truncation=True, max_length=77).to(self.device)
        for k in inputs:
            if inputs[k].dtype == torch.float32:
                inputs[k] = inputs[k].half()
        out = self.model.get_text_features(**inputs)
        if hasattr(out, "pooler_output"):
            feats = out.pooler_output
        elif hasattr(out, "text_embeds"):
            feats = out.text_embeds
        elif isinstance(out, tuple):
            feats = out[0]
        else:
            feats = out
        feats = feats / feats.norm(dim=-1, keepdim=True)
        return feats.cpu().numpy().astype(np.float32)

    def search(self, query, top_k=100, templates=None):
        feats = self.encode_with_expansion(query, templates)
        scores, idx = self.index.search(feats, min(top_k * 2, self.index.ntotal))
        # Merge all prompt templates before truncating.  Returning as soon as
        # the first template fills top-k silently discarded translated and
        # video-frame prompts.
        best = {}
        for t_idx in range(scores.shape[0]):
            for s, i in zip(scores[t_idx], idx[t_idx]):
                if i < 0:
                    continue
                if i not in best or float(s) > best[i]:
                    best[int(i)] = float(s)
        order = sorted(best, key=best.get, reverse=True)[:top_k]
        return [(self.meta[i][0], int(self.meta[i][1])) for i in order]

    def encode_with_expansion(self, query, templates=None):
        if templates is None:
            templates = [
                query,
                f"a photo of {query}",
                f"an image of {query}",
                f"a frame from a video showing {query}",
            ]
        return self.encode_text(templates)

    def search_per_video(self, query, top_videos=10, templates=None):
        feats = self.encode_with_expansion(query, templates)
        K = top_videos * 30
        scores, idx = self.index.search(feats, min(K, self.index.ntotal))
        max_scores = {}
        for t_idx in range(scores.shape[0]):
            for s, i in zip(scores[t_idx], idx[t_idx]):
                if i < 0: continue
                if i not in max_scores or s > max_scores[i]:
                    max_scores[i] = s
        per_vid = {}
        for i, s in max_scores.items():
            vid, fi = self.meta[i]
            if vid not in per_vid or s > per_vid[vid][2]:
                per_vid[vid] = (vid, fi, s)
        ranked = sorted(per_vid.values(), key=lambda x: -x[2])
        return ranked[:top_videos]


def _align_events_dp_strided(scores_per_event, frames, stride_min=3):
    """Strided DP."""
    N = len(scores_per_event)
    M = len(scores_per_event[0])
    if M == 0 or N == 0:
        return [0] * N
    NEG_INF = -1e18
    dp = np.full((N, M), NEG_INF, dtype=np.float32)
    bp = np.full((N, M), -1, dtype=np.int32)
    dp[0] = scores_per_event[0]
    for i in range(1, N):
        s = scores_per_event[i]
        prev = np.full(M, NEG_INF, dtype=np.float32)
        for j in range(M):
            end_k = j - stride_min
            if end_k < 0: continue
            prev[j] = np.max(dp[i - 1][:end_k + 1])
        dp[i] = prev + s
        for j in range(M):
            end_k = j - stride_min
            if end_k < 0: continue
            bp[i][j] = int(np.argmax(dp[i - 1][:end_k + 1]))
    j = int(np.argmax(dp[N - 1]))
    out = [0] * N
    for i in range(N - 1, -1, -1):
        out[i] = j
        j = int(bp[i][j])
    return out


def generate_trake(retriever, query, events, top_videos=5, templates=None):
    video_candidates = retriever.search_per_video(query, top_videos=top_videos, templates=templates)
    if not video_candidates:
        return []

    event_texts = [e.get("desc", e.get("name", "")) for e in events]
    event_templates_list = []
    for ev in event_texts:
        event_templates_list.append([
            f"{query} - {ev}",
            f"a photo of {ev}",
            f"an image of {ev}",
            ev,
        ])

    results = []
    for vid, top_fi, top_s in video_candidates:
        rows = retriever.by_vid.get(vid, [])
        if not rows: continue
        rows.sort(key=lambda x: x[1])
        all_idx = [r[0] for r in rows]
        all_fi = [r[1] for r in rows]
        vecs = np.vstack([retriever.index.reconstruct(i) for i in all_idx])
        per_event_scores = []
        for ev_templates in event_templates_list:
            ef = retriever.encode_text(ev_templates)
            s = (ef @ vecs.T).max(axis=0)
            per_event_scores.append(s)
        frames = np.array(all_fi)
        n = len(events)
        if n > 1 and len(frames) > n * 3:
            stride = max(1, len(frames) // (n * 3))
            aligned = _align_events_dp_strided(per_event_scores, frames, stride_min=stride)
        else:
            aligned = [0] * n
            for ei, s in enumerate(per_event_scores):
                aligned[ei] = int(np.argmax(s))
        score = float(np.mean([s[idx] for s, idx in zip(per_event_scores, aligned)]))
        results.append((vid, [int(frames[i]) for i in aligned], score))

    if not results:
        return []
    results.sort(key=lambda x: -x[2])
    primary_vid, primary_frames, _ = results[0]
    out_lines = []
    for rank in range(1, 101):
        vid = primary_vid
        frames = primary_frames[:]
        if rank > 1:
            if rank <= 10:
                offset = (rank - 1) * 3
                frames = [f + offset for f in primary_frames]
            elif rank % 10 == 0 and len(results) > 1:
                idx_alt = (rank // 10) % len(results)
                vid, frames, _ = results[idx_alt]
            else:
                offset = (rank - 1) * 5
                frames = [f + offset for f in primary_frames]
        out_lines.append(f"{vid}, " + ", ".join(str(f) for f in frames))
    return out_lines


def is_vietnamese(text):
    """Detect Vietnamese text by diacritics."""
    vn_chars = "àáảãạăằắẳẵặâầấẩẫậèéẻẽẹêềếểễệìíỉĩịòóỏõọôồốổỗộơờớởỡợùúủũụưừứửữựỳýỷỹỵđ"
    return any(c in vn_chars for c in text.lower())


def build_templates_for_query(query, en_translation=None):
    """Build expanded templates - both Vietnamese and English if available."""
    templates = []
    # Vietnamese templates
    templates.append(query)
    templates.append(f"a photo of {query}")
    templates.append(f"hình ảnh về {query}")
    templates.append(f"khung hình trong video về {query}")
    if en_translation:
        # English templates (CLIP works better on English)
        templates.append(en_translation)
        templates.append(f"a photo of {en_translation}")
        templates.append(f"an image of {en_translation}")
        templates.append(f"a frame from a video showing {en_translation}")
    return templates


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--queries", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--top-k", type=int, default=100)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--translate", action="store_true", help="Use Qwen3 translation")
    ap.add_argument("--use-vlm", action="store_true", help="Use Qwen2.5-VL for QA")
    args = ap.parse_args()

    retriever = CLIPRetriever(device=args.device)
    translator = None
    if args.translate:
        from translator import Translator
        translator = Translator(device=args.device)
    vlm = None
    if args.use_vlm:
        from qa_vlm import VLMAnswerer, keyframe_path
        vlm = VLMAnswerer(device=args.device)

    with open(args.queries) as f:
        queries = [json.loads(line) for line in f if line.strip()]
    print(f"[pipeline] {len(queries)} queries, translate={args.translate}, vlm={args.use_vlm}")

    with open(args.out, "w") as fout:
        for q in queries:
            qid = q.get("query_id", "?")
            qt = q.get("query_type", "KIS").upper()
            text = q.get("query_text", q.get("description", ""))

            # translate if Vietnamese
            en_text = None
            if translator and is_vietnamese(text):
                try:
                    en_text = translator.translate(text)
                except Exception as e:
                    print(f"[q{qid}] translate err: {e}")
                    en_text = None

            templates = build_templates_for_query(text, en_text)

            if qt == "KIS":
                ans = retriever.search(text, top_k=args.top_k, templates=templates)
                lines = [f"{v}, {f}" for v, f in ans]

            elif qt == "QA":
                question = q.get("question", text)
                results = retriever.search(text, top_k=20, templates=templates)
                answer = ""
                
                if vlm is not None and results:
                    # 1. Identify the best matching video ID
                    top_vid = results[0][0]
                    
                    # 2. Extract all retrieved frame indices for THIS specific video
                    vid_frames = [f for v, f in results if v == top_vid]
                    
                    # 3. Sort chronologically and limit to 4-8 frames to prevent memory overflow
                    vid_frames = sorted(vid_frames)[:6] 
                    
                    # 4. Resolve the file paths using your existing keyframe_path function
                    img_paths = [keyframe_path(top_vid, f) for f in vid_frames]
                    
                    try:
                        # 5. Pass the chronological sequence to the VLM
                        answer = vlm.answer_multi(img_paths, question)
                    except Exception as e:
                        print(f"[qa] VLM err {qid}: {e}")
                        answer = ""
                        
                # Never copy a reference answer from the query fixture.  If
                # the optional VLM is unavailable, emit an empty answer so
                # blind QA runs remain honest rather than leaking GT.
                lines = [f"{v}, {f}, {answer}" for v, f in results]

            elif qt == "TRAKE":
                events = q.get("events", [])
                # also translate event descriptions
                if translator:
                    for ev in events:
                        ev_text = ev.get("desc", ev.get("name", ""))
                        if is_vietnamese(ev_text):
                            try:
                                ev["en_desc"] = translator.translate(ev_text)
                            except:
                                ev["en_desc"] = ev_text
                lines = generate_trake(retriever, text, events, templates=templates)
            else:
                lines = []

            for rank, line in enumerate(lines, 1):
                fout.write(json.dumps({"query_id": qid, "rank": rank, "answer": line}) + "\n")
            print(f"Q{qid} {qt}: {len(lines)} candidates, en={en_text[:60] if en_text else None}")

    print(f"Saved {args.out}")


if __name__ == "__main__":
    main()
