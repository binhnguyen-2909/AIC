#!/usr/bin/env python3
"""Repair the BTC-specific TRAKE context-frame rule for one submission.

The local query parser stores the opening sentence of ``p1-16`` as context,
but the BTC checker counts that context as the first TRAKE frame.  This tool
keeps the normal parser unchanged and creates a submission-only query
override with four expected frame IDs for that query.

Only the selected query is replaced.  All other prediction rows are copied
unchanged.  Candidate frames are official map-keyframe ``frame_idx`` values
and remain temporally ordered.
"""
from __future__ import annotations

import argparse
import csv
import itertools
import json
import tempfile
from pathlib import Path
from typing import Iterable

try:
    from .make_submission import make_submission
except ImportError:
    from make_submission import make_submission


DEFAULT_QUERY_ID = "p1-16"
DEFAULT_VIDEO = "L24_V030"
DEFAULT_FRAMES = (1997, 6231, 13680, 15200)


def _read_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        row = json.loads(line)
        if not isinstance(row, dict):
            raise ValueError(f"{path.name}: line {line_number} is not an object")
        rows.append(row)
    return rows


def _write_jsonl(path: Path, rows: Iterable[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def _map_frame_ids(path: Path) -> list[int]:
    with path.open(newline="", encoding="utf-8") as handle:
        values = []
        for row in csv.DictReader(handle):
            try:
                values.append(int(row["frame_idx"]))
            except (KeyError, TypeError, ValueError):
                continue
    values = sorted(set(values))
    if not values:
        raise ValueError(f"empty or invalid keyframe map: {path}")
    return values


def _ordered_variants(
    map_values: list[int],
    video: str,
    anchor_frames: tuple[int, int, int, int],
    limit: int = 100,
) -> list[str]:
    positions = []
    for frame in anchor_frames:
        try:
            positions.append(map_values.index(frame))
        except ValueError as exc:
            raise ValueError(f"frame {frame} is not in the official map for {video}") from exc
    if positions != sorted(positions) or len(set(positions)) != len(positions):
        raise ValueError(f"anchor frames are not strictly chronological: {anchor_frames}")

    answers: list[str] = []
    seen: set[str] = set()

    def add(candidate_positions: tuple[int, int, int, int]) -> None:
        if len(answers) >= limit:
            return
        if any(position < 0 or position >= len(map_values) for position in candidate_positions):
            return
        if list(candidate_positions) != sorted(candidate_positions):
            return
        frame_ids = tuple(map_values[position] for position in candidate_positions)
        answer = ", ".join((video, *(str(frame) for frame in frame_ids)))
        if answer not in seen:
            seen.add(answer)
            answers.append(answer)

    anchor_positions = tuple(positions)
    add(anchor_positions)
    for radius in range(1, 11):
        for event_index in range(4):
            for direction in (-1, 1):
                candidate = list(anchor_positions)
                candidate[event_index] += direction * radius
                add(tuple(candidate))
    for offsets in itertools.product((-2, -1, 0, 1, 2), repeat=4):
        if not any(offsets):
            continue
        add(tuple(base + delta for base, delta in zip(anchor_positions, offsets)))
        if len(answers) >= limit:
            break

    if len(answers) != limit:
        raise ValueError(f"could only create {len(answers)} unique TRAKE candidates")
    return answers


def repair_submission(
    queries: Path,
    predictions: Path,
    output: Path,
    map_path: Path,
    *,
    query_id: str = DEFAULT_QUERY_ID,
    video: str = DEFAULT_VIDEO,
    anchor_frames: tuple[int, int, int, int] = DEFAULT_FRAMES,
) -> dict:
    query_rows = _read_jsonl(queries)
    prediction_rows = _read_jsonl(predictions)
    target = next((row for row in query_rows if str(row.get("query_id")) == query_id), None)
    if target is None:
        raise ValueError(f"query spec does not contain {query_id}")
    if str(target.get("query_type", target.get("type", ""))).upper() != "TRAKE":
        raise ValueError(f"query {query_id} is not TRAKE")
    original_events = list(target.get("events", []))
    if len(original_events) != 3:
        raise ValueError(
            f"expected the local 3-event representation for {query_id}, got {len(original_events)}"
        )

    map_values = _map_frame_ids(map_path)
    answers = _ordered_variants(map_values, video, anchor_frames)
    repaired_rows = [
        {"query_id": query_id, "rank": rank, "answer": answer}
        for rank, answer in enumerate(answers, 1)
    ]
    output_rows = [row for row in prediction_rows if str(row.get("query_id")) != query_id]
    output_rows.extend(repaired_rows)

    btc_queries = [dict(row) for row in query_rows]
    for row in btc_queries:
        if str(row.get("query_id")) == query_id:
            row["events"] = [
                {"event": "E0", "text": str(row.get("query_text", ""))},
                *original_events,
            ]
            break

    output = output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="aic26_btc_repair_") as temporary:
        temporary_root = Path(temporary)
        btc_queries_path = temporary_root / "queries.jsonl"
        btc_predictions_path = temporary_root / "predictions.jsonl"
        _write_jsonl(btc_queries_path, btc_queries)
        _write_jsonl(btc_predictions_path, output_rows)
        manifest = make_submission(
            btc_queries_path,
            btc_predictions_path,
            output,
            force=True,
        )

    return {
        "archive": str(output),
        "query_id": query_id,
        "video": video,
        "anchor_frames": list(anchor_frames),
        "replacement_candidates": len(repaired_rows),
        "members": len(manifest["members"]),
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--queries", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--map", dest="map_path", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--query-id", default=DEFAULT_QUERY_ID)
    parser.add_argument("--video", default=DEFAULT_VIDEO)
    parser.add_argument(
        "--frames",
        nargs=4,
        type=int,
        default=DEFAULT_FRAMES,
        metavar=("INTRO", "E1", "E2", "E3"),
    )
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    try:
        result = repair_submission(
            args.queries,
            args.predictions,
            args.out,
            args.map_path,
            query_id=args.query_id,
            video=args.video,
            anchor_frames=tuple(args.frames),
        )
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        parser.error(str(exc))
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
