# AIC26 evaluation and submission contract

## Submission format (updated 2026-08-20)

The organizer's updated preliminary-round guide requires one CSV per query,
then a ZIP containing a directory named `submission/`. This is separate from
the internal JSONL format used by the retrieval pipeline and smoke tests.

- Query suffixes: `kis`, `qa`, `trake`.
- Every CSV: UTF-8, comma-delimited, no header, at most 100 rows.
- KIS: `video_name, frame_id`.
- QA: `video_name, frame_id, answer`; answer ≤100 characters, Vietnamese or
  English.
- TRAKE: `video_name, frame_1, ..., frame_N`; exactly N event frames in order.
- Video names do not include `.mp4`; frame IDs are integers.
- ZIP member paths must be `submission/<query>.csv`.
- At most three submissions per query package; the last is used for ranking.

The source hashes and the full operational notes are in
[`AIC26_OFFICIAL_UPDATE_20260820.md`](AIC26_OFFICIAL_UPDATE_20260820.md).

Use [`solution/package_submission.py`](../solution/package_submission.py) to
validate and package CSVs. It is a format checker, not an accuracy scorer.

## Accuracy contract status

The updated submission guide does not provide an executable organizer scorer,
official query/ground-truth package, R-Score/FinalScore formula, or frame
coordinate convention. Therefore this repository must not claim official
accuracy or `>90%` from local proxy/smoke results. See the research and
validation notes for the current evidence labels.
