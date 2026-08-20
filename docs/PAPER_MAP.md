# AIC26 paper map

Reviewed `2026-08-20`. This is a transfer map, not a claim that any paper's
score transfers to AIC26. The [official AIC homepage](https://aichallenge.hochiminhcity.gov.vn/)
describes multimedia retrieval, a planned automatic format, and encourages
LVLM/GenAI integration, but the inspected public page does not expose an
official query/ground-truth/scorer package.

| Paper/system | Relevant evidence | Transferable component | Experiment direction | Risk |
|---|---|---|---|---|
| [MERVIN](https://arxiv.org/abs/2605.16120) | Vietnamese news retrieval; reports 79/88 on AIC HCMC 2025 | keyframes + transcripts + summaries; Vietnamese text embeddings; multimodal index | BM25 + ASR/OCR + visual candidate union | AIC25 result, not AIC26; abstract-level transfer |
| [Vortex](https://arxiv.org/abs/2606.19682) | AIC HCMC 2025 system; reports 79.6/88 | CLIP/SigLIP2 RRF, speech/vision metadata, Rocchio and temporal rerank | fusion/temporal hypotheses; SigLIP remains opt-in | external score and paper query-count details need audit |
| [DANTE](https://arxiv.org/abs/2512.13169) | TRAKE-oriented AIC25 system | dynamic programming for temporally incoherent event alignment | monotonic DP and ordered-event smoke | needs real event intervals and official partial-credit scorer |
| [Enhanced cross-modal temporal retrieval](https://arxiv.org/abs/2512.06334) | AIC HCMC 2025-adjacent event retrieval | event-specific query modalities and adaptive scene/slide boundaries | boundary/query-expansion ablation candidate | thresholds and external results not reproduced |
| [SigLIP 2](https://arxiv.org/abs/2502.14786) | multilingual image-text representation | optional dense visual channel | full local proxy was below BM25-led route | domain/language shift; compute cost |
| [CLIP4Clip](https://arxiv.org/abs/2104.08860) / [X-CLIP](https://arxiv.org/abs/2207.07285) | video-text retrieval baselines | temporal pooling and video-level reranking | candidate-level visual rerank | benchmark/domain mismatch |
| [QD-DETR](https://arxiv.org/abs/2303.13874) | temporal grounding | query-dependent moment reranking | future held-out ablation | requires labeled temporal training |
| [PhoWhisper](https://arxiv.org/abs/2406.02555) | Vietnamese ASR | timestamped speech retrieval channel | ASR shard/merge/resume path | recognition and timestamp errors |
| [Qwen2.5-VL](https://arxiv.org/abs/2502.13923) | multimodal QA/reasoning | selective multi-frame evidence answer | QA route is optional and fail-closed | GPU cost and hidden answer semantics |

## Decisions

- Promote only correctness fixes and reproducible gains on the same frozen
  protocol; do not promote a paper because it is recent or reports a high
  external score.
- Keep BM25/ASR/OCR/object candidate union as the high-recall base, visual
  reranking for candidate refinement, monotonic DP for TRAKE, and selective
  VLM for QA.
- Keep dense SigLIP2 and translation as opt-in until an independent AIC-like
  labeled split shows a gain in the full `R@1/R@5/R@20/R@50/R@100` FinalScore.

## Evidence boundary

The public repository has no dataset, model weights, private GT, or generated
index. A local data-bearing checkout must record manifest hashes, coverage and
the exact evaluator before any result is labeled `HELD_OUT` or `OFFICIAL`.
