# Reproducible quality audit

This checkout contains the production route and tests, but not the organizer's
private query, ground-truth, or scorer files. The following checks separate
engineering completeness from an official accuracy claim.

## Local completeness checks

On the dataset-bearing checkout, verify that both merged runtime indexes cover
the complete media manifest:

```bash
wc -l solution/ensemble_index/asr_full.jsonl \
      solution/ensemble_index/ocr_full.jsonl
jq -r '.video_id' solution/ensemble_index/asr_full.jsonl | sort -u | wc -l
jq -r '.video_id' solution/ensemble_index/ocr_full.jsonl | sort -u | wc -l
python -m py_compile solution/*.py
python solution/synthetic_test.py
```

The current local run has 873 ASR rows and 873 OCR rows. The generated JSONL
indexes are intentionally not part of this public repository.

## Official score gate

After receiving the organizer files, run:

```bash
python solution/submission_ens.py \
  --queries /path/to/queries.jsonl \
  --out /tmp/aic_submission.jsonl \
  --use-ensemble

python solution/eval_official.py \
  --pred /tmp/aic_submission.jsonl \
  --gt /path/to/ground_truth.jsonl
```

Only the `FinalScore` from that evaluator on the organizer-compatible labels
can support a claim above 90%. Metadata pseudo-labels, self-retrieval tests,
and the dataset-free synthetic test are diagnostics, not substitutes for this
gate.
