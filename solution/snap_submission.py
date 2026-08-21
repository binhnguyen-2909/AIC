#!/usr/bin/env python3
"""Snap prediction frames to official keyframe frame_idx values."""
from __future__ import annotations
import argparse, csv, json
from collections import defaultdict
from pathlib import Path
from typing import Iterable

def _type(row: dict) -> str:
    raw = str(row.get("query_type", row.get("type", "KIS"))).upper()
    return {"Q&A":"QA", "VQA":"QA", "TEXTUAL_KIS":"KIS"}.get(raw, raw)

def _maps(root: Path, video_ids: Iterable[str]) -> dict[str, list[int]]:
    base=root/"extracted"/"map-keyframes-aic25-b1"/"map-keyframes"; result={}
    for video_id in sorted(set(video_ids)):
        path=base/f"{video_id}.csv"
        if not path.is_file(): raise ValueError(f"missing keyframe map for video {video_id}")
        values=[]
        with path.open(newline="") as handle:
            for row in csv.DictReader(handle):
                try: values.append(int(row["frame_idx"]))
                except (KeyError,TypeError,ValueError): continue
        if not values: raise ValueError(f"empty keyframe map for video {video_id}")
        result[video_id]=sorted(set(values))
    return result

def snap_predictions(root: Path, queries: Path, predictions: Path, output: Path) -> dict:
    query_types={}
    for line in queries.read_text(encoding="utf-8").splitlines():
        if line.strip():
            row=json.loads(line); query_types[str(row["query_id"])] = _type(row)
    raw=[json.loads(line) for line in predictions.read_text(encoding="utf-8").splitlines() if line.strip()]
    maps=_maps(root,[str(row["answer"]).split(",",1)[0].strip() for row in raw]); grouped=defaultdict(list)
    for row in raw:
        qid=str(row["query_id"]); qt=query_types[qid]; text=str(row.get("answer","")); parts=[p.strip() for p in (text.split(",",2) if qt=="QA" else text.split(","))]
        if len(parts)<2: raise ValueError(f"query {qid}: malformed answer")
        for index in range(1, 2 if qt in {"KIS","QA"} else len(parts)):
            frame=int(parts[index]); parts[index]=str(min(maps[parts[0]],key=lambda candidate:abs(candidate-frame)))
        grouped[qid].append({"query_id":row["query_id"],"rank":int(row.get("rank",0)),"answer":", ".join(parts)})
    output=output.expanduser().resolve(); output.parent.mkdir(parents=True,exist_ok=True); written=0
    with output.open("w",encoding="utf-8") as handle:
        for qid in query_types:
            seen=set()
            for rank,row in enumerate(sorted(grouped.get(qid,[]),key=lambda item:(item["rank"],item["answer"])),1):
                if row["answer"] in seen: continue
                seen.add(row["answer"]); row["rank"]=rank; handle.write(json.dumps(row,ensure_ascii=False)+"\n"); written+=1
    return {"output":str(output),"input_rows":len(raw),"output_rows":written,"queries":len(query_types)}

def _parser() -> argparse.ArgumentParser:
    parser=argparse.ArgumentParser(description=__doc__); parser.add_argument("--root",type=Path,default=None); parser.add_argument("--queries",type=Path,required=True); parser.add_argument("--predictions",type=Path,required=True); parser.add_argument("--out",type=Path,required=True); return parser

def main(argv: Iterable[str] | None = None) -> int:
    parser=_parser(); args=parser.parse_args(argv); root=(args.root or Path(__file__).resolve().parents[1]).resolve()
    try: result=snap_predictions(root,args.queries,args.predictions,args.out)
    except (OSError,KeyError,ValueError,json.JSONDecodeError) as exc: parser.error(str(exc))
    print(json.dumps(result,ensure_ascii=False,indent=2)); return 0

if __name__ == "__main__": raise SystemExit(main())
