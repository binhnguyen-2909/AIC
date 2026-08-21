#!/usr/bin/env python3
"""Convert internal JSONL predictions into the AIC26 CSV-per-query ZIP."""
from __future__ import annotations
import argparse, csv, json, re, tempfile
from collections import defaultdict
from pathlib import Path
from typing import Iterable
try:
    from .package_submission import SubmissionFormatError, package
except ImportError:  # direct execution: python solution/make_submission.py
    from package_submission import SubmissionFormatError, package

def _query_type(row: dict) -> str:
    raw = str(row.get("query_type", row.get("type", "KIS"))).upper()
    return {"Q&A": "QA", "VQA": "QA", "TEXTUAL_KIS": "KIS"}.get(raw, raw)

def _safe_id(value: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value)).strip("._")
    return safe or "query"

def _read_jsonl(path: Path) -> list[dict]:
    rows = []
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip(): continue
        try: row = json.loads(line)
        except json.JSONDecodeError as exc: raise SubmissionFormatError(f"{path.name}: invalid JSON on line {number}: {exc}") from exc
        if not isinstance(row, dict): raise SubmissionFormatError(f"{path.name}: line {number} must be a JSON object")
        rows.append(row)
    return rows

def _answer_fields(answer: object, query_type: str, query_id: str) -> list[str]:
    raw = str(answer or "")
    if query_type == "KIS": fields, expected = [part.strip() for part in raw.split(",")], 2
    elif query_type == "QA": fields, expected = [part.strip() for part in raw.split(",", 2)], 3
    elif query_type == "TRAKE": fields, expected = [part.strip() for part in raw.split(",")], None
    else: raise SubmissionFormatError(f"query {query_id}: unsupported query type {query_type}")
    if expected is not None and len(fields) != expected: raise SubmissionFormatError(f"query {query_id}: expected {expected} answer fields, got {len(fields)}")
    if len(fields) < 2 or not fields[0]: raise SubmissionFormatError(f"query {query_id}: answer has no video name")
    if query_type == "QA" and not fields[2]: raise SubmissionFormatError(f"query {query_id}: QA answer is blank")
    return fields

def make_submission(queries: Path, predictions: Path, output: Path, *, force: bool = False) -> dict:
    specs = {}
    for row in _read_jsonl(queries):
        qid = str(row.get("query_id", ""))
        if not qid: raise SubmissionFormatError("query spec contains a row without query_id")
        if qid in specs: raise SubmissionFormatError(f"duplicate query_id in spec: {qid}")
        qt = _query_type(row)
        if qt not in {"KIS", "QA", "TRAKE"}: raise SubmissionFormatError(f"query {qid}: unsupported query type {qt}")
        event_count = len(row.get("events", [])) if qt == "TRAKE" else None
        if qt == "TRAKE" and not event_count: raise SubmissionFormatError(f"query {qid}: TRAKE events list is empty")
        specs[qid] = (qt, event_count)
    by_query = defaultdict(list)
    for ordinal, row in enumerate(_read_jsonl(predictions)):
        qid = str(row.get("query_id", ""))
        if qid not in specs: raise SubmissionFormatError(f"prediction references unknown query_id: {qid}")
        try: rank = int(row.get("rank", ordinal + 1))
        except (TypeError, ValueError) as exc: raise SubmissionFormatError(f"query {qid}: rank is not an integer") from exc
        by_query[qid].append((rank, ordinal, str(row.get("answer", ""))))
    output = output.expanduser().resolve()
    with tempfile.TemporaryDirectory(prefix="aic26_csv_") as temporary:
        csv_dir = Path(temporary)
        for qid, (qt, event_count) in specs.items():
            candidates = sorted(by_query.get(qid, []), key=lambda item: (item[0], item[1]))[:100]
            if not candidates: raise SubmissionFormatError(f"query {qid}: no prediction rows")
            path = csv_dir / f"query-{_safe_id(qid)}-{qt.lower()}.csv"
            with path.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.writer(handle, lineterminator="\n")
                for _rank, _ordinal, answer in candidates:
                    fields = _answer_fields(answer, qt, qid)
                    if qt == "TRAKE" and len(fields) != int(event_count) + 1: raise SubmissionFormatError(f"query {qid}: expected {int(event_count)} TRAKE frames, got {len(fields) - 1}")
                    writer.writerow(fields)
        return package(csv_dir, output, force=force)

def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--queries", type=Path, required=True, help="internal query JSONL")
    parser.add_argument("--predictions", type=Path, required=True, help="submission_ens JSONL")
    parser.add_argument("--out", type=Path, required=True, help="submission.zip")
    parser.add_argument("--force", action="store_true")
    return parser

def main(argv: Iterable[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try: result = make_submission(args.queries, args.predictions, args.out, force=args.force)
    except (OSError, SubmissionFormatError) as exc: _parser().error(str(exc))
    print(json.dumps(result, ensure_ascii=False, indent=2)); return 0

if __name__ == "__main__": raise SystemExit(main())
