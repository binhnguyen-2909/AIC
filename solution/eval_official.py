"""Evaluate an AIC submission against official-style JSONL ground truth.

The local synthetic evaluators are useful for development but do not score
frame ranges, TRAKE partial credit, or the official five operating points.
This script accepts a permissive GT schema so it can be used as soon as the
organizer's query/answer file is available.

GT fields accepted per line:
  query_id, query_type, video_id (or gt_video),
  frame_start/frame_end (or frame_range), answer (for QA),
  events=[{"frame_start":..., "frame_end":...}, ...] (for TRAKE).
"""
from __future__ import annotations

import argparse
import json
import re
import unicodedata
from collections import defaultdict
from pathlib import Path

KS = (1, 5, 20, 50, 100)


def _range(value, start=None, end=None):
    if isinstance(value, (list, tuple)) and len(value) >= 2:
        return int(value[0]), int(value[1])
    if isinstance(value, dict):
        start = value.get("start", value.get("frame_start", start))
        end = value.get("end", value.get("frame_end", end))
    if start is None or end is None:
        return None
    return int(start), int(end)


_NUM_WORDS = {
    # Common Vietnamese answer spellings in the task examples.
    "không": "0", "một": "1", "mốt": "1",
    "hai": "2", "ba": "3", "bốn": "4",
    "năm": "5", "sáu": "6", "bảy": "7", "tám": "8",
    "chín": "9", "mười": "10",
    "one": "1", "two": "2", "three": "3", "four": "4",
    "five": "5", "six": "6", "seven": "7", "eight": "8",
    "nine": "9", "ten": "10", "zero": "0",
}


def _answer_norm(value):
    if isinstance(value, (list, tuple)):
        value = " ".join(str(x) for x in value)
    text = str("" if value is None else value).strip().lower()
    text = unicodedata.normalize("NFC", text)
    text = re.sub(r"[\.,;:!?()\[\]{}]", " ", text)
    tokens = []
    for token in re.findall(r"\w+", text, flags=re.UNICODE):
        tokens.append(_NUM_WORDS.get(token, token))
    return " ".join(tokens)


def _answer_match(pred, expected):
    """Conservative semantic normalization for the official QA rule.

    The organizer's phrase "khớp ngữ nghĩa" allows e.g. ``5`` and ``năm``.
    We normalize number words and punctuation, while retaining ordinary text
    equality so that unrelated answers are never treated as equal.
    """
    p = _answer_norm(pred)
    if isinstance(expected, (list, tuple, set)):
        return any(p == _answer_norm(x) for x in expected)
    e = _answer_norm(expected)
    if p == e:
        return True
    # Allow a bare number to match the same number with a classifier such as
    # "người"/"people", which is common in the supplied QA example.
    p_num = re.fullmatch(r"\d+", p or "")
    e_num = re.search(r"(?<!\w)(\d+)(?!\w)", e or "")
    if p_num and e_num and p_num.group(0) == e_num.group(1):
        return True
    e_num = re.fullmatch(r"\d+", e or "")
    p_num = re.search(r"(?<!\w)(\d+)(?!\w)", p or "")
    return bool(e_num and p_num and e_num.group(0) == p_num.group(1))


def _video_norm(value):
    text = str(value or "").strip()
    return re.sub(r"\.mp4$", "", text, flags=re.IGNORECASE)


def _query_type(row):
    raw = str(row.get("query_type", row.get("type", "KIS"))).strip().upper()
    aliases = {"Q&A": "QA", "VQA": "QA", "TEXTUAL_KIS": "KIS"}
    raw = aliases.get(raw, raw)
    if raw not in {"KIS", "QA", "TRAKE"}:
        raise ValueError(f"unknown query type {raw!r} for query_id={row.get('query_id')}")
    return raw


def _validate_range(value, query_id, label):
    try:
        result = _range(value)
    except (TypeError, ValueError, KeyError):
        result = None
    if result is None or result[0] < 0 or result[1] < result[0]:
        raise ValueError(f"invalid {label} range for query_id={query_id}: {value!r}")
    return result


def _validate_gt_row(gt):
    qid = gt.get("query_id")
    qt = _query_type(gt)
    video = gt.get("video_id", gt.get("gt_video", gt.get("video", "")))
    if not _video_norm(video):
        raise ValueError(f"missing GT video_id for query_id={qid}")
    if qt in {"KIS", "QA"}:
        value = gt.get("frame_range")
        if value is None:
            value = [gt.get("frame_start"), gt.get("frame_end")]
        _validate_range(value, qid, "frame")
    if qt == "QA":
        answer = gt.get("answer", gt.get("gt_answer"))
        if answer is None or not _answer_norm(answer):
            raise ValueError(f"missing QA answer for query_id={qid}")
    if qt == "TRAKE":
        events = gt.get("events")
        if not isinstance(events, list) or not events:
            raise ValueError(f"TRAKE GT has no events for query_id={qid}")
        for i, event in enumerate(events):
            value = event.get("frame_range")
            if value is None:
                value = [event.get("frame_start"), event.get("frame_end")]
            _validate_range(value, qid, f"TRAKE event {i}")


def _parse_answer(value, query_type="KIS"):
    fields = [x.strip() for x in str(value or "").split(",")]
    if len(fields) < 2:
        return "", [], ""
    video = fields[0]
    if str(query_type).upper() == "QA":
        # QA answers are allowed to be numeric (e.g. "5").  Only the first
        # field after video is the frame; all remaining fields are answer text.
        try:
            frame = int(fields[1])
        except ValueError:
            return video, [], ",".join(fields[1:]).strip()
        return video, [frame], ",".join(fields[2:]).strip()
    nums = []
    for x in fields[1:]:
        try:
            nums.append(int(x))
        except ValueError:
            break
    answer = ",".join(fields[1 + len(nums):]).strip()
    return video, nums, answer


def _score(pred, gt):
    qt = _query_type(gt)
    raw = pred.get("answer", "")
    fields = [x.strip() for x in str(raw).split(",")]
    events = gt.get("events", []) if qt == "TRAKE" else []
    expected_fields = 2 if qt == "KIS" else (3 if qt == "QA" else len(events) + 1)
    # A malformed candidate cannot receive partial credit.  In particular,
    # extra TRAKE frames must not be silently ignored.
    if len(fields) != expected_fields or not fields[0]:
        return 0.0
    video, frames, answer = _parse_answer(pred.get("answer", ""), qt)
    if not frames or any(frame < 0 for frame in frames):
        return 0.0
    if qt == "QA" and not answer:
        return 0.0
    if qt == "TRAKE" and len(frames) != len(events):
        return 0.0
    expected_video = gt.get("video_id", gt.get("gt_video", gt.get("video", "")))
    if _video_norm(video) != _video_norm(expected_video):
        return 0.0
    if qt == "QA":
        frame_range = _range(gt.get("frame_range"), gt.get("frame_start"), gt.get("frame_end"))
        frame_ok = bool(frames) and (frame_range is None or frame_range[0] <= frames[0] <= frame_range[1])
        answer_ok = _answer_match(
            answer, gt.get("answer", gt.get("gt_answer", ""))
        )
        return float(frame_ok and answer_ok)
    if qt == "TRAKE":
        events = gt.get("events", [])
        if not events:
            return 1.0
        if len(frames) < len(events):
            return 0.0
        good = 0
        for frame, event in zip(frames, events):
            r = _range(event.get("frame_range"), event.get("frame_start"), event.get("frame_end"))
            good += int(r is not None and r[0] <= frame <= r[1])
        return good / len(events)
    frame_range = _range(gt.get("frame_range"), gt.get("frame_start"), gt.get("frame_end"))
    return float(bool(frames) and (frame_range is None or frame_range[0] <= frames[0] <= frame_range[1]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", required=True, type=Path)
    ap.add_argument("--gt", required=True, type=Path)
    args = ap.parse_args()

    gt = {}
    with args.gt.open() as f:
        for line in f:
            if line.strip():
                row = json.loads(line)
                qid = str(row.get("query_id"))
                if not row.get("query_id") or qid in gt:
                    raise SystemExit(f"duplicate/missing GT query_id: {qid!r}")
                _validate_gt_row(row)
                gt[qid] = row

    preds = defaultdict(list)
    with args.pred.open() as f:
        for line in f:
            if line.strip():
                row = json.loads(line)
                qid = str(row.get("query_id"))
                if not row.get("query_id"):
                    raise SystemExit("prediction is missing query_id")
                preds[qid].append(row)

    unknown = sorted(set(preds) - set(gt))
    if unknown:
        raise SystemExit(f"predictions contain unknown query_id(s): {unknown[:5]}")

    # Ranks are part of the local JSONL transport.  Respect them rather than
    # silently trusting physical file order; allow plain answer rows only
    # when every row for that query omits rank.
    ranked_preds = {}
    for qid, rows in preds.items():
        if len(rows) > 100:
            raise SystemExit(f"more than 100 predictions for query_id={qid}")
        has_rank = ["rank" in row for row in rows]
        if any(has_rank) and not all(has_rank):
            raise SystemExit(f"mixed ranked/unranked predictions for query_id={qid}")
        if all(has_rank) and rows:
            try:
                ranks = [int(row["rank"]) for row in rows]
            except (TypeError, ValueError) as exc:
                raise SystemExit(f"invalid rank for query_id={qid}") from exc
            expected = list(range(1, len(rows) + 1))
            if sorted(ranks) != expected:
                raise SystemExit(f"ranks must be contiguous 1..N for query_id={qid}")
            ranked_preds[qid] = sorted(rows, key=lambda row: int(row["rank"]))
        else:
            ranked_preds[qid] = rows

    per_query = {}
    for qid, row in gt.items():
        ranked = ranked_preds.get(qid, [])
        scores = [_score(p, row) for p in ranked]
        per_query[qid] = scores

    n = len(per_query)
    if not n:
        raise SystemExit("No matching query_id in ground truth")
    r_at = {}
    for k in KS:
        r_at[k] = sum(max(scores[:k], default=0.0) for scores in per_query.values()) / n
    print(json.dumps({"n": n, **{f"R@{k}": v for k, v in r_at.items()},
                      "FinalScore": sum(r_at.values()) / len(KS)}, indent=2))


if __name__ == "__main__":
    main()
