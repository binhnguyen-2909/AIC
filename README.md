# AIC 2026 — best current solution

This repository contains the current production route for the AIC 2026
multimedia retrieval task. It intentionally contains **code, documentation,
and tests only**. The competition videos, keyframes, feature matrices,
detector outputs, FAISS indexes, model weights, and query/ground-truth files
are not committed.

## Current route

The default high-recall pipeline is:

```text
Vietnamese query
  -> BM25 over media-info
  -> object TF-IDF candidate videos
  -> CLIP-B/32 frame reranking
  -> temporal coherence / monotonic DP for TRAKE
  -> optional source-frame refinement
  -> optional Qwen2.5-VL answer for QA
```

The implementation is in [`solution/submission_ens.py`](solution/submission_ens.py).
It supports KIS, QA, and TRAKE JSONL queries and emits the official ranked
JSONL shape. SigLIP2, ASR, and OCR are optional channels; they are kept
separate because the available proxy validation did not show that enabling
them improves the BM25-led baseline.

There is no public AIC 2026 query/ground-truth/scorer bundle in this checkout,
so an official claim of `>90%` cannot be verified locally. The proxy numbers
and known limitations are recorded in
[`docs/RESEARCH_REPORT.md`](docs/RESEARCH_REPORT.md).

## Quick start

```bash
git clone https://github.com/binhnguyen-2909/AIC.git
cd AIC
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Place the organizer-provided extracted data under `extracted/` (or make that
directory available in the checkout). Nothing in this repository downloads or
commits the dataset. The expected extracted folders are:

```text
extracted/
  clip-features-32-aic25-b1/clip-features-32/*.npy
  map-keyframes-aic25-b1/map-keyframes/*.csv
  media-info-aic25-b1/media-info/*.json
  objects-aic25-b1/objects/<video_id>/*.json       # optional
  Keyframes_*/keyframes/<video_id>/*.jpg           # needed for QA
  Videos_*/video/<video_id>.mp4                    # needed for dense refine
```

Build the local artifacts from the same snapshot:

```bash
python solution/rebuild_consistent.py --skip-objects
python solution/build_text_index.py
python solution/build_object_index.py              # optional but recommended
```

These commands write only local runtime artifacts under
`solution/index/` and `solution/ensemble_index/`; those paths are ignored by
Git.

Run retrieval:

```bash
python solution/submission_ens.py \
  --queries /path/to/queries.jsonl \
  --out /path/to/submission.jsonl \
  --use-ensemble
```

For Vietnamese-to-English visual expansion, add `--translate` when the
Qwen3-8B checkpoint is available. For QA, add `--use-vlm`; this requires the
Qwen2.5-VL-3B checkpoint and the keyframe images:

```bash
python solution/submission_ens.py \
  --queries /path/to/queries.jsonl \
  --out /path/to/submission.jsonl \
  --use-ensemble --translate --use-vlm
```

Use `--device cpu` only for smoke tests. On a shared GPU, leave SigLIP2
disabled unless it has been benchmarked on the target query distribution.

## Input and output

Each input line is one query. Supported type aliases are `KIS`, `QA`/`Q&A`/
`VQA`, and `TRAKE`. Output is one ranked line per query:

```json
{"query_id": 1, "rank": 1, "answer": "L25_V085, 18951"}
```

KIS answers are `video_id, frame_id`; QA answers are
`video_id, frame_id, answer`; TRAKE answers are
`video_id, f1, f2, ..., fN`.

Validate a submission against a supplied official-style ground truth with:

```bash
python solution/eval_official.py \
  --gt /path/to/ground_truth.jsonl \
  --pred /path/to/submission.jsonl
```

Run the dataset-free regression test:

```bash
python solution/synthetic_test.py
python -m py_compile solution/*.py
```

## Portability and resource controls

Paths default to the repository root. Override them when embedding the code
elsewhere:

```bash
export AIC_ROOT=/path/to/checkout
export AIC_SOLUTION_ROOT=/path/to/checkout/solution
```

The ASR/OCR builders are resumable and optional. Long-form Whisper pipelines
may internally reduce worker count to one for timestamp correctness; `--workers`
is therefore a request, not a guarantee. Do not commit their generated JSONL
indexes.

For a multi-GPU or multi-process run, split the sorted video list into
disjoint shards and write one output per worker:

```bash
CUDA_VISIBLE_DEVICES=0 python solution/asr_index.py build \
  --out solution/ensemble_index/asr_shard_0.jsonl \
  --shard-index 0 --num-shards 8 --batch-size 64 --num-beams 1 \
  --skip-file solution/ensemble_index/asr_index.jsonl
```

Run the other shard indices concurrently, then merge atomically with
`solution/merge_indices.py` and validate against the media-info directory.

## Project notes

- `docs/RESEARCH_REPORT.md`: papers, official-data audit, benchmarks, and
  limitations.
- `docs/IMPROVEMENT_PLAN.md`: next experiments and failure modes.
- `solution/ensemble.py`: production fusion and TRAKE alignment.
- `solution/submission_ens.py`: single entry point for all query types.
