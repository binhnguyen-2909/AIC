"""
Production submission: CLIP + BM25 + Objects fused + Qwen3 translation + Qwen2.5-VL QA.
Uses EnsembleRetriever from ensemble.py for retrieval.
"""
import argparse
import json
import os
import sys
import re
from collections import Counter
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

# The public repository contains code only.  Keep the runtime root portable so
# another machine can checkout the repo anywhere (or override it explicitly).
ROOT = Path(os.environ.get("AIC_ROOT", str(Path(__file__).resolve().parents[1])))
sys.path.insert(0, str(ROOT / "solution"))


def _normalize_result(item):
    """Normalize legacy 2-tuples and ensemble 3-tuples."""
    if len(item) == 2:
        video, frame = item
        return str(video), int(frame), 0.0
    video, frame, score = item[:3]
    return str(video), int(frame), float(score)


def _dedupe_lines(lines, top_k):
    """Keep first exact candidate and never spend rank slots on duplicates."""
    out = []
    seen = set()
    for line in lines:
        key = tuple(part.strip() for part in str(line).split(","))
        if not key or key in seen:
            continue
        seen.add(key)
        out.append(str(line).strip())
        if len(out) >= top_k:
            break
    return out


def _qa_answer_or_fail(answer, *, allow_blank: bool, query_id, rank: int) -> str:
    """Refuse to publish a blank QA answer outside explicit diagnostic mode."""
    value = str(answer or "").strip()
    if not value and not allow_blank:
        raise RuntimeError(
            f"blank QA answer for query_id={query_id} rank={rank}; "
            "use --allow-blank-qa only for an explicit diagnostic run"
        )
    return value


def _query_type(query):
    raw = str(query.get("query_type", query.get("type", "KIS"))).upper()
    return {"Q&A": "QA", "VQA": "QA", "TEXTUAL_KIS": "KIS"}.get(raw, raw)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--queries", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--top-k", type=int, default=100)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--translate", action="store_true")
    ap.add_argument("--use-vlm", action="store_true")
    ap.add_argument("--qa-frames", type=int, default=20,
                    help="number of candidate rows to answer with VLM (default: 20)")
    ap.add_argument("--allow-given-answer", action="store_true",
                    help="use q.answer fallback; disabled by default to avoid leakage")
    ap.add_argument("--use-ensemble", action="store_true", help="Use CLIP+BM25+Objects ensemble")
    ap.add_argument("--clip-only", action="store_true", help="Use CLIP only (faster)")
    ap.add_argument("--use-siglip", action="store_true",
                    help="enable the separately-built full SigLIP2 channel")
    ap.add_argument("--siglip-weight", type=float, default=0.15,
                    help="RRF-like video weight for SigLIP2 (default: 0.15)")
    ap.add_argument("--dense-refine-videos", type=int, default=3,
                    help="source videos to locally decode for KIS/QA (default: 3)")
    ap.add_argument("--dense-window", type=int, default=150,
                    help="source frames on each side for local refinement")
    ap.add_argument("--no-dense-refine", action="store_true",
                    help="disable source-frame refinement for a fast diagnostic run")
    ap.add_argument("--allow-blank-qa", action="store_true",
                    help="continue with blank QA answers when VLM is unavailable")
    args = ap.parse_args()

    if not 1 <= args.top_k <= 100:
        ap.error("--top-k must be between 1 and 100")
    if args.qa_frames < 1:
        ap.error("--qa-frames must be positive")
    if args.dense_refine_videos < 0 or args.dense_window < 0:
        ap.error("dense refinement settings must be nonnegative")

    with open(args.queries) as f:
        queries = [json.loads(line) for line in f if line.strip()]
    query_types = [_query_type(q) for q in queries]
    unknown_types = sorted(set(query_types) - {"KIS", "QA", "TRAKE"})
    if unknown_types:
        raise SystemExit(f"unsupported query type(s): {unknown_types}")
    has_qa = "QA" in query_types
    has_trake = "TRAKE" in query_types
    if has_qa and not args.use_vlm and not args.allow_given_answer \
            and not args.allow_blank_qa:
        raise SystemExit("QA queries require --use-vlm (or explicit --allow-blank-qa)")

    if args.use_ensemble:
        from ensemble import EnsembleRetriever
        # Keep the CLIP text tower for visual frame localization in every
        # ensemble run.  It is small relative to Qwen2.5-VL and is essential
        # for KIS/QA frame selection; TRAKE also needs it for event DP.
        retriever = EnsembleRetriever(device=args.device,
                                      load_clip=not args.clip_only,
                                      load_siglip=args.use_siglip)
    else:
        from submission_v3 import CLIPRetriever
        retriever = CLIPRetriever(device=args.device)
    from submission_v3 import build_templates_for_query

    translator = None
    if args.translate:
        try:
            from translator import Translator
            translator = Translator(device=args.device)
        except Exception as exc:
            print(f"[translator] unavailable; using original query: {exc}",
                  flush=True)
    vlm = None
    keyframe_path = None
    if args.use_vlm:
        from qa_vlm import VLMAnswerer, keyframe_path
        try:
            vlm = VLMAnswerer(device=args.device)
        except Exception as exc:
            if has_qa and not args.allow_blank_qa and not args.allow_given_answer:
                raise RuntimeError(
                    "Qwen2.5-VL could not be loaded while QA queries exist; "
                    "free GPU or pass --allow-blank-qa for a diagnostic run"
                ) from exc
            print(f"[vlm] unavailable; continuing without QA model: {exc}", flush=True)
    print(f"[ens-pipeline] {len(queries)} queries, ensemble={args.use_ensemble}, translate={args.translate}, vlm={args.use_vlm}")

    out_path = Path(args.out)
    tmp_path = out_path.with_name(out_path.name + ".tmp")
    with open(tmp_path, "w") as fout:
        for q in queries:
            qid = q.get("query_id", "?")
            qt = _query_type(q)
            text = q.get("query_text", q.get("description", ""))

            question = q.get("question", text) if qt == "QA" else ""
            retrieval_text = (f"{text}. {question}".strip(". ")
                              if qt == "QA" and question else text)

            en_text = None
            if translator:
                try:
                    from submission_v3 import is_vietnamese
                    if is_vietnamese(retrieval_text):
                        en_text = translator.translate(retrieval_text)
                except Exception as e:
                    print(f"[q{qid}] translate err: {e}")

            # Keep the Vietnamese/original query for metadata retrieval.  The
            # BM25 index is built from Vietnamese media-info; passing only the
            # English translation here silently removes that signal and can
            # cause the ensemble retriever to fill candidates randomly.
            search_q = en_text if en_text else text

            if args.use_ensemble:
                # ensemble returns (vid, frame_idx, score)
                # Keep Vietnamese for BM25/ASR/OCR.  Use the translated query
                # only for the English CLIP reranker, and only when metadata
                # does not already provide a strong lexical match.
                bm25_probe = retriever.bm25_search(retrieval_text, 1)
                strong_lexical = bool(bm25_probe and bm25_probe[0][1] > 5.0)
                dense_weight = 0.0 if strong_lexical else (0.25 if en_text else 0.10)
                if getattr(retriever, "clip", None) is None:
                    dense_weight = 0.0
                siglip_weight = (0.0 if strong_lexical else args.siglip_weight)
                results = retriever.retrieve(
                    retrieval_text, clip_text=en_text or retrieval_text,
                    top_k=args.top_k,
                    w_clip_video=dense_weight,
                    w_siglip=siglip_weight,
                    siglip_text=en_text or retrieval_text,
                )
            else:
                # plain CLIP search
                from submission_v3 import build_templates_for_query
                templates = build_templates_for_query(retrieval_text, en_text)
                results = retriever.search(retrieval_text, top_k=args.top_k,
                                           templates=templates)
            results = [_normalize_result(item) for item in results]
            if (args.use_ensemble and qt in {"KIS", "QA"}
                    and not args.no_dense_refine
                    and args.dense_refine_videos > 0
                    and args.dense_window > 0):
                results = retriever.refine_source_frames(
                    retrieval_text, results,
                    max_videos=args.dense_refine_videos,
                    window=args.dense_window,
                    per_video=max(5, args.qa_frames if qt == "QA" else 5),
                )

            if qt == "KIS":
                lines = [f"{v}, {f}" for v, f, s in results]

            elif qt == "QA":
                lines = []
                
                # We only need to answer once using the very best video's temporal context
                answered = False
                final_answer = ""
                
                # Import the new sequence extractor
                try:
                    from dense_refine import get_contiguous_keyframes
                except ImportError:
                    from solution.dense_refine import get_contiguous_keyframes
                
                for row_idx, (v, f, _score) in enumerate(results):
                    answer = ""
                    
                    if vlm is not None and row_idx == 0: # Only generate answer for the top-1 candidate
                        # Get a 5-frame sequence around the best frame (center - 2, center + 2)
                        sequence_paths = get_contiguous_keyframes(v, f, window=2)
                        
                        if sequence_paths:
                            prompt = (
                                f"Sự kiện cần tìm: {text}\n"
                                f"Câu hỏi: {question}\n"
                            )
                            try:
                                final_answer = vlm.answer_multi(sequence_paths, prompt).strip()
                                answered = True
                            except Exception as e:
                                print(f"[qa] VLM err {qid} sequence: {e}")
                                
                        # Fallback to single frame if sequence extraction failed
                        if not answered and keyframe_path:
                            img_path = keyframe_path(v, f)
                            if img_path:
                                prompt = (
                                    f"Sự kiện cần tìm: {text}\n"
                                    f"Câu hỏi: {question}\n"
                                    "Chỉ trả lời ngắn gọn dựa trên hình ảnh này."
                                )
                                try:
                                    final_answer = vlm.answer(str(img_path), prompt).strip()
                                except Exception as e:
                                    print(f"[qa] VLM err {qid} single-frame: {e}")

                    if not final_answer and args.allow_given_answer:
                        final_answer = q.get("answer", "")

                    final_answer = _qa_answer_or_fail(
                        final_answer, allow_blank=args.allow_blank_qa,
                        query_id=qid, rank=1,
                    )
                        
                    lines.append(f"{v}, {f}, {final_answer}")

            elif qt == "TRAKE":
                events = q.get("events", [])
                if not events:
                    raise ValueError(f"TRAKE query {qid} has no events")
                if args.use_ensemble:
                    # Keep the same multimodal candidate union as KIS/QA, then
                    # align event-frame scores with monotonic DP.  The old
                    # path created a second CLIP retriever and discarded the
                    # ensemble/ASR/OCR candidate scores for TRAKE.
                    event_texts = []
                    for ev in events:
                        et = (ev.get("text") or ev.get("desc") or
                              ev.get("name") or ev.get("event", ""))
                        if translator and et:
                            try:
                                if is_vietnamese(et):
                                    et = translator.translate(et) or et
                            except Exception as e:
                                print(f"[q{qid}] event translate err: {e}")
                        event_texts.append(et)
                    lines = retriever.trake(
                        text, events, top_k=args.top_k,
                        event_texts=event_texts,
                        clip_text=en_text,
                    )
                else:
                    from submission_v3 import generate_trake as gen_trake_v3
                    templates = build_templates_for_query(text, en_text)
                    lines = gen_trake_v3(retriever, search_q, events, templates=templates)
            else:
                lines = []

            lines = _dedupe_lines(lines, args.top_k)

            for rank, line in enumerate(lines, 1):
                fout.write(json.dumps({"query_id": qid, "rank": rank, "answer": line}) + "\n")
            print(f"Q{qid} {qt}: {len(lines)} candidates")

    os.replace(tmp_path, out_path)
    print(f"Saved {args.out}")


if __name__ == "__main__":
    main()
