#!/usr/bin/env python3
"""Validate AIC26 CSV files and create the official submission archive.

The organizer's internal JSONL format is useful for retrieval debugging, but
the preliminary-round upload is a ZIP containing a directory named exactly
``submission``. This module deliberately uses only the Python standard
library so it can run before the model environment is installed.
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import os
import re
import tempfile
import zipfile
from pathlib import Path
from typing import Iterable


CSV_NAME_RE = re.compile(r"(?:^|[-_])(kis|qa|trake)\.csv$", re.IGNORECASE)
INTEGER_RE = re.compile(r"[0-9]+\Z")
MAX_ROWS = 100
MAX_QA_ANSWER_CHARS = 100


class SubmissionFormatError(ValueError):
    """Raised when a CSV or archive would violate the published contract."""


def _query_type(path: Path) -> str:
    match = CSV_NAME_RE.search(path.name)
    if not match:
        raise SubmissionFormatError(
            f"{path.name}: filename must end in -kis.csv, -qa.csv, or -trake.csv"
        )
    return match.group(1).lower()


def _is_header(row: list[str], query_type: str) -> bool:
    normalized = [field.strip().lower() for field in row]
    if query_type == "kis":
        return normalized == ["video_name", "frame_id"]
    if query_type == "qa":
        return normalized == ["video_name", "frame_id", "answer"]
    return len(normalized) >= 2 and normalized[:2] == ["video_name", "frame_id"]


def _validate_frame(value: str, path: Path, row_number: int, column: int) -> None:
    if not INTEGER_RE.fullmatch(value.strip()):
        raise SubmissionFormatError(
            f"{path.name}: row {row_number}, column {column} is not a non-negative integer"
        )


def _validate_csv(path: Path) -> dict[str, object]:
    query_type = _query_type(path)
    try:
        raw = path.read_bytes()
        text = raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise SubmissionFormatError(f"{path.name}: file is not valid UTF-8") from exc

    try:
        rows = list(csv.reader(io.StringIO(text, newline=""), strict=True))
    except csv.Error as exc:
        raise SubmissionFormatError(f"{path.name}: invalid CSV: {exc}") from exc

    if not rows:
        raise SubmissionFormatError(f"{path.name}: CSV must contain at least one row")
    if len(rows) > MAX_ROWS:
        raise SubmissionFormatError(
            f"{path.name}: contains {len(rows)} rows; maximum is {MAX_ROWS}"
        )
    if _is_header(rows[0], query_type):
        raise SubmissionFormatError(f"{path.name}: header row is not allowed")

    expected_columns = {"kis": 2, "qa": 3, "trake": None}[query_type]
    for row_number, row in enumerate(rows, start=1):
        if not row or all(not field.strip() for field in row):
            raise SubmissionFormatError(f"{path.name}: row {row_number} is blank")
        if expected_columns is not None and len(row) != expected_columns:
            raise SubmissionFormatError(
                f"{path.name}: row {row_number} has {len(row)} columns; "
                f"expected {expected_columns}"
            )
        if query_type == "trake" and len(row) < 2:
            raise SubmissionFormatError(
                f"{path.name}: row {row_number} must contain a video and at least one frame"
            )

        video_name = row[0].strip()
        if not video_name:
            raise SubmissionFormatError(f"{path.name}: row {row_number} has an empty video name")
        if video_name.lower().endswith(".mp4"):
            raise SubmissionFormatError(
                f"{path.name}: row {row_number} must omit the .mp4 suffix"
            )
        if "\n" in video_name or "\r" in video_name:
            raise SubmissionFormatError(
                f"{path.name}: row {row_number} has a multiline video name"
            )

        for column, value in enumerate(row[1:], start=2):
            if query_type == "qa" and column == 3:
                if len(value) > MAX_QA_ANSWER_CHARS:
                    raise SubmissionFormatError(
                        f"{path.name}: row {row_number} answer exceeds "
                        f"{MAX_QA_ANSWER_CHARS} characters"
                    )
                continue
            _validate_frame(value, path, row_number, column)

    return {"file": path.name, "type": query_type, "rows": len(rows)}


def _csv_files(input_dir: Path) -> list[Path]:
    if not input_dir.is_dir():
        raise SubmissionFormatError(f"input directory does not exist: {input_dir}")
    files = sorted(input_dir.iterdir(), key=lambda item: item.name.lower())
    if not files:
        raise SubmissionFormatError("input directory contains no CSV files")
    if any(not path.is_file() for path in files):
        bad = next(path for path in files if not path.is_file())
        raise SubmissionFormatError(
            f"input directory must contain direct CSV files only; found {bad.name}"
        )
    if any(path.suffix.lower() != ".csv" for path in files):
        bad = next(path for path in files if path.suffix.lower() != ".csv")
        raise SubmissionFormatError(f"unexpected non-CSV file in input directory: {bad.name}")
    return files


def package(input_dir: Path, output: Path, *, force: bool = False) -> dict[str, object]:
    """Validate *input_dir* and atomically write *output* as a submission ZIP."""

    files = _csv_files(input_dir)
    manifest = [_validate_csv(path) for path in files]
    output = output.expanduser().resolve()
    if output.exists() and not force:
        raise SubmissionFormatError(f"output already exists (use --force): {output}")
    output.parent.mkdir(parents=True, exist_ok=True)

    fd, temporary_name = tempfile.mkstemp(
        prefix=f".{output.name}.", suffix=".tmp", dir=output.parent
    )
    os.close(fd)
    temporary = Path(temporary_name)
    try:
        with zipfile.ZipFile(temporary, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            for path in files:
                info = zipfile.ZipInfo(f"submission/{path.name}")
                info.date_time = (1980, 1, 1, 0, 0, 0)
                info.compress_type = zipfile.ZIP_DEFLATED
                info.external_attr = 0o644 << 16
                archive.writestr(info, path.read_bytes())
        os.replace(temporary, output)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise

    return {
        "archive": str(output),
        "members": [f"submission/{path.name}" for path in files],
        "files": manifest,
    }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--force", action="store_true", help="replace an existing archive")
    parser.add_argument("--json", action="store_true", help="print a JSON manifest")
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    try:
        result = package(args.input_dir, args.out, force=args.force)
    except SubmissionFormatError as exc:
        _build_parser().error(str(exc))
    if args.json:
        print(json.dumps(result, ensure_ascii=False, indent=2))
    else:
        print(f"created {result['archive']} with {len(result['members'])} CSV files")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
