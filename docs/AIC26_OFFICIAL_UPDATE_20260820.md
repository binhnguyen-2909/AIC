# AIC26 — updated preliminary-round submission rules

Recorded on `2026-08-20` from two user-provided captures of the updated AIC26
submission guide:

| Source capture | SHA256 |
|---|---|
| capture 1 | `786b4d3bb8d8033aa6f4f96cc6530d8eb3a994c553bc436adc0fce44aa0112bb` |
| capture 2 | `c3d9494b8650c661df82fe13e7324b0334d73ff58640441c1d78d400a9469b34` |

## Submission contract

The preliminary round has `kis`, `qa`, and `trake` query files. The organizer
can release query packages in multiple rounds; the filename suffix identifies
the query type.

Each query requires one UTF-8, comma-delimited `.csv` file with no header and
at most 100 rows:

- KIS: `<video_name>, <frame_id>`.
- Q&A: `<video_name>, <frame_id>, <answer>`.
- TRAKE: `<video_name>, <frame_1>, ..., <frame_N>`, with exactly `N` event
  frames in event order.

Q&A answers are at most 100 characters and may be Vietnamese or English. CSV
quoting is mandatory when an answer contains a comma, quote, or newline;
embedded quotes use the CSV `""` escape. Simple answers may be unquoted.

The final archive must be:

```text
submission.zip
└── submission/
    ├── query-1-kis.csv
    ├── query-2-qa.csv
    └── query-3-trake.csv
```

Video names omit `.mp4`, frame IDs are integers, and the archive—not loose CSV
files—is submitted. A package may be submitted at most three times; the last
submission is used. Public leaderboard scoring uses 50% of organizer answers,
while the final private ranking uses 100%. One team uses one account, and a
malformed submission still consumes one attempt.

## Ambiguities kept explicit

The guide describes Q&A matching both as semantic comparison and, later, as an
exact string comparison. No organizer executable scorer is available here, so
the project does not invent a resolution. The guide also does not restate the
R-Score/FinalScore formula or 0/1-based frame convention; those remain separate
contract fields requiring the official statement/scorer.

Internal JSONL is still supported for debugging. Run the packaging validator
before submitting to the organizer:

```bash
python3 solution/package_submission.py \
  --input-dir /path/to/csv_results \
  --out /path/to/submission.zip
```
