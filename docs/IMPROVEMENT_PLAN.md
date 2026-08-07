# AIC 2026 Vòng Sơ Tuyển — Improvement Plan toward >90%

## Current State

| Verified proxy (200 queries) | Title (R@1) | Title (FinalScore) | Desc (R@1) | Desc (FinalScore) |
|--------|------------|-------------------|-----------|-------------------|
| BM25 / ensemble round-robin | 95.5% | 99.1% | 46.0% | 63.5% |
| Official-style frame/answer/TRAKE GT | — | **not available** | — | **not available** |

**Key insight**: BM25 is already excellent for title queries. The gap is entirely
in visual content / description queries where Vietnamese text → image matching is weak.

## Four-Pronged Improvement Strategy

### 1. English Translation Bridge (v8 — implemented, not validated on official GT)
- Translate Vietnamese → English before CLIP encoding
- English CLIP-B/32 text encoder is significantly better aligned to image space
- Multi-template ensemble (6+ templates)
- **Hypothesis**: may help visual descriptions; no official-GT gain is measured.

### 2. SigLIP2-Based Dense Retrieval (isolated benchmark complete)
- SigLIP2-base-patch16-224 is multilingual and is the strongest cached-compatible
  candidate tested here.
- The isolated branch encodes 6,984 sampled keyframes and builds its own FAISS
  index; it does not overwrite the 176,707-frame production index.
- Description proxy: R@1=4.5%, R@100=54.5%, FinalScore=27.1%.
- Keep optional; no official-GT gain is measured, so production remains
  BM25-led.

### 3. Two-Stage Retrieval + Reranking
- Stage 1: BM25 picks top-50 candidate videos (fast, text-based)
- Stage 2: Dense (CLIP/SigLIP) reranks within candidates (visual)
- Avoids CLIP's weakness at full-corpus Vietnamese text matching
- **Hypothesis**: improves robustness; no official-GT gain is measured.

### 4. Per-Event TRAKE Alignment (v2 — COMPLETED)
- Frame-level CLIP search within candidate videos per event
- DP alignment with temporal constraints
- English translation for better event queries
- **Hypothesis**: improves ordered event localization; no official-GT gain is measured.

### 5. Better Vietnamese Tokenization & Stopword Handling
- Already improved in v8
- Compound word tokenization (pyvi) is already used

## Recommended Production Pipeline

```bash
# PRIMARY: v8 with English translation bridge
CUDA_VISIBLE_DEVICES=0 python3 solver_v8.py \
  --queries queries.jsonl --out submission.jsonl

# IF SIGLIP INDEX AVAILABLE: v9 (SigLIP-based)
CUDA_VISIBLE_DEVICES=0 python3 solver_v9.py \
  --queries queries.jsonl --out submission.jsonl

# Q&A with VLM
CUDA_VISIBLE_DEVICES=0 python3 qa_solver_v2.py \
  --queries queries.jsonl --out submission.jsonl --retriever v8

# TRAKE-specific
CUDA_VISIBLE_DEVICES=0 python3 trake_solver_v2.py \
  --queries queries.jsonl --out submission.jsonl

# Quick benchmark
CUDA_VISIBLE_DEVICES=0 python3 benchmark_harness.py \
  --solvers v7 v8 --test-sets title desc
```

## Unverified projections (do not treat as results)

| Component | Title FS | Desc FS | Combined |
|-----------|---------|---------|----------|
| BM25/ensemble (metadata proxy) | 99.1% | 63.5% | not comparable to official |
| v8 (+EN bridge) | not measured on official GT | not measured | not measurable |
| SigLIP2 visual-only (isolated proxy) | — | 27.1% | not measurable |
| Vortex-style fusion (SigLIP2 + caption/OCR + ASR + temporal rerank) | not measured | not measured | not measurable |

The only >90% figure currently available is a result reported by the Vortex
paper on AIC'25-like data; it is not a result of this workspace. See
`RESEARCH_REPORT.md` for the source and limitations.

## Key Papers & Research

1. **CLIP (Radford et al., 2021)**: Zero-shot image-text matching
   - Our base: CLIP-ViT-B/32, 512-dim, ~63% ImageNet zero-shot

2. **SigLIP (Zhai et al., 2023)**: Sigmoid loss for language-image pre-training
   - Better zero-shot than CLIP, especially on fine-grained tasks
   - SigLIP-base-patch16-224: ~768-dim, ~78% ImageNet zero-shot

3. **Multilingual CLIP (Reimers & Gurevych, 2020)**:
   - sentence-transformers/clip-ViT-B-32-multilingual-v1
   - DistilBERT text tower, supports Vietnamese
   - Already used in v7 (but weaker than English CLIP for image alignment)

4. **ColPali / ColQwen2 (Faysse et al., 2024)**: Late interaction multi-vector retrieval
   - Would give +5-10 NDCG@10 but 50-80GB extra index (too heavy for this setup)

5. **RRF (Cormack et al., 2009)**: Reciprocal Rank Fusion
   - Already used for BM25+CLIP+Objects fusion

6. **DP Alignment for Video Moment Retrieval**:
   - Standard Viterbi-style DP for temporal event alignment
   - Used in trake_solver_v2 with per-event CLIP search
