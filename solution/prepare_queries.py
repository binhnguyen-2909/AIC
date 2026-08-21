#!/usr/bin/env python3
"""Convert the AIC26 official query TXT package into the internal JSONL input."""
from __future__ import annotations
import argparse, json, re, zipfile
from collections import Counter
from pathlib import Path
from typing import Iterable

NAME_RE = re.compile(r"^query-(?P<query_id>.+)-(?P<query_type>kis|qa|trake)\.txt$", re.IGNORECASE)
EVENT_RE = re.compile(r"^\s*E\s*(?P<number>\d+)\s+(?P<text>.+?)\s*$", re.IGNORECASE | re.MULTILINE)

class QueryPreparationError(ValueError): pass

def _decode(raw: bytes, name: str) -> str:
    try: text = raw.decode("utf-8-sig").replace("\r\n", "\n").replace("\r", "\n").strip()
    except UnicodeDecodeError as exc: raise QueryPreparationError(f"{name}: expected UTF-8 text") from exc
    if not text: raise QueryPreparationError(f"{name}: query is empty")
    return text

def _source_files(source: Path) -> list[tuple[str, bytes]]:
    if source.is_dir(): files = sorted((path.name, path.read_bytes()) for path in source.glob("*.txt"))
    elif source.is_file() and source.suffix.lower() == ".zip":
        with zipfile.ZipFile(source) as archive:
            files = []; seen: set[str] = set()
            for info in sorted(archive.infolist(), key=lambda item: item.filename.lower()):
                if info.is_dir() or not info.filename.lower().endswith(".txt"): continue
                name = Path(info.filename).name
                if name in seen: raise QueryPreparationError(f"duplicate TXT basename in ZIP: {name}")
                seen.add(name); files.append((name, archive.read(info)))
    else: raise QueryPreparationError(f"input must be a TXT directory or .zip: {source}")
    if not files: raise QueryPreparationError("input contains no .txt query files")
    return files

def _qa_fields(text: str) -> tuple[str, str]:
    question_end = text.rfind("?")
    if question_end < 0: return text, text
    boundary = max(text.rfind(".", 0, question_end), text.rfind("!", 0, question_end), text.rfind("?", 0, question_end), text.rfind("\n", 0, question_end))
    question = text[boundary + 1:].strip(); scene = text[:boundary + 1].strip(" .\n\r")
    return (scene or text), (question or text)

def _parse(name: str, raw: bytes) -> dict:
    match = NAME_RE.fullmatch(name)
    if not match: raise QueryPreparationError(f"{name}: expected query-<id>-kis.txt, query-<id>-qa.txt, or query-<id>-trake.txt")
    query_id, query_type, text = match.group("query_id"), match.group("query_type").upper(), _decode(raw, name)
    row: dict = {"query_id": query_id, "query_type": query_type}
    if query_type == "QA":
        query_text, question = _qa_fields(text); row.update({"query_text": query_text, "question": question})
    elif query_type == "TRAKE":
        events = [{"event": f"E{item.group('number')}", "text": item.group("text").strip()} for item in EVENT_RE.finditer(text)]
        if not events: raise QueryPreparationError(f"{name}: TRAKE has no E1/E2/... event lines")
        numbers = [int(event["event"][1:]) for event in events]
        if numbers != list(range(1, len(numbers) + 1)): raise QueryPreparationError(f"{name}: TRAKE event labels must be E1..E{len(numbers)}")
        first = EVENT_RE.search(text); row.update({"query_text": text[:first.start()].strip() if first else text, "events": events})
    else: row["query_text"] = text
    return row

def prepare_queries(source: Path, output: Path) -> dict:
    rows = [_parse(name, raw) for name, raw in _source_files(source)]
    ids = [str(row["query_id"]) for row in rows]
    if len(ids) != len(set(ids)): raise QueryPreparationError("duplicate query_id after filename parsing")
    output = output.expanduser().resolve(); output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")
    counts = Counter(row["query_type"] for row in rows)
    return {"output": str(output), "queries": len(rows), "by_type": dict(sorted(counts.items()))}

def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument("--input", type=Path, required=True); parser.add_argument("--out", type=Path, required=True); return parser

def main(argv: Iterable[str] | None = None) -> int:
    parser = _parser(); args = parser.parse_args(argv)
    try: result = prepare_queries(args.input.expanduser().resolve(), args.out)
    except (OSError, QueryPreparationError) as exc: parser.error(str(exc))
    print(json.dumps(result, ensure_ascii=False, indent=2)); return 0

if __name__ == "__main__": raise SystemExit(main())
