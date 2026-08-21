#!/usr/bin/env python3
"""Fail-closed readiness check for an AIC26 submission run.

The check is intentionally read-only: it validates code dependencies, cached
models, production indexes, the manifest, optional query JSONL, and CUDA
availability without loading a model or allocating GPU memory.
"""
from __future__ import annotations

import argparse
import importlib
import json
import os
import sys
from pathlib import Path
from typing import Iterable

QUERY_TYPES = {"KIS", "QA", "Q&A", "VQA", "TEXTUAL_KIS", "TRAKE"}
REQUIRED_MODULES = ("numpy", "torch", "faiss", "transformers", "rank_bm25", "pyvi", "PIL")
REQUIRED_INDEXES = (
    "solution/index/clip_b32.faiss", "solution/index/meta.json", "solution/index/per_video.pkl",
    "solution/ensemble_index/bm25.pkl", "solution/ensemble_index/objects.pkl",
    "solution/ensemble_index/asr_full.jsonl", "solution/ensemble_index/ocr_full.jsonl",
    "solution/manifests/production_manifest_20260820.json",
)
DATA_DIRS = (
    "extracted/media-info-aic25-b1/media-info", "extracted/map-keyframes-aic25-b1/map-keyframes",
    "extracted/clip-features-32-aic25-b1/clip-features-32", "extracted/objects-aic25-b1/objects",
)
MODEL_CACHE_NAMES = {
    "clip": "models--openai--clip-vit-base-patch32", "vlm": "models--Qwen--Qwen2.5-VL-3B-Instruct",
    "translator": "models--Qwen--Qwen3-8B", "siglip": "models--google--siglip2-base-patch16-224",
}


def normalize_query_type(value: object) -> str:
    raw = str(value or "KIS").upper()
    return {"Q&A": "QA", "VQA": "QA", "TEXTUAL_KIS": "KIS"}.get(raw, raw)


def validate_query_rows(rows: list[dict]) -> list[str]:
    errors: list[str] = []
    seen: set[str] = set()
    for number, row in enumerate(rows, 1):
        if not isinstance(row, dict):
            errors.append(f"line {number}: JSON value must be an object")
            continue
        qid = str(row.get("query_id", ""))
        if not qid:
            errors.append(f"line {number}: missing query_id")
        elif qid in seen:
            errors.append(f"line {number}: duplicate query_id={qid}")
        seen.add(qid)
        raw_type = str(row.get("query_type", row.get("type", "KIS"))).upper()
        if raw_type not in QUERY_TYPES:
            errors.append(f"line {number}: unsupported query type={raw_type}")
            continue
        query_type = normalize_query_type(raw_type)
        if not str(row.get("query_text", row.get("description", ""))).strip():
            errors.append(f"line {number}: missing query_text/description")
        if query_type == "QA" and not str(row.get("question", "")).strip():
            errors.append(f"line {number}: QA query is missing question")
        if query_type == "TRAKE" and not isinstance(row.get("events"), list):
            errors.append(f"line {number}: TRAKE query is missing events list")
        if query_type == "TRAKE" and not row.get("events"):
            errors.append(f"line {number}: TRAKE events list is empty")
    return errors


def _cache_root() -> Path:
    hf_home = os.environ.get("HF_HOME")
    return Path(hf_home).expanduser() / "hub" if hf_home else Path.home() / ".cache/huggingface/hub"


def inspect(root: Path, *, require_data: bool = False, require_vlm: bool = False,
            require_gpu: bool = False, query_path: Path | None = None) -> dict:
    checks: dict[str, object] = {}
    errors: list[str] = []
    warnings: list[str] = []
    missing_indexes = [p for p in REQUIRED_INDEXES if not (root / p).is_file()]
    checks["missing_indexes"] = missing_indexes
    if missing_indexes:
        (errors if require_data else warnings).append("missing production artifacts: " + ", ".join(missing_indexes))
    manifest_path = root / "solution/manifests/production_manifest_20260820.json"
    if manifest_path.is_file():
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            complete = manifest.get("production_gate", {}).get("complete") is True
            checks["manifest_complete"] = complete
            if not complete:
                errors.append("production manifest is not complete")
        except Exception as exc:
            errors.append(f"cannot parse production manifest: {exc}")
    elif require_data:
        errors.append("production manifest is missing")
    missing_data = [p for p in DATA_DIRS if not (root / p).is_dir()]
    checks["missing_data_dirs"] = missing_data
    if missing_data:
        (errors if require_data else warnings).append("missing extracted data directories: " + ", ".join(missing_data))
    module_status: dict[str, str] = {}
    for name in REQUIRED_MODULES:
        try:
            module = importlib.import_module(name)
            module_status[name] = "OK" + (f" {getattr(module, '__version__')}" if getattr(module, "__version__", None) else "")
        except Exception as exc:
            module_status[name] = f"FAIL: {type(exc).__name__}: {exc}"
            errors.append(f"dependency {name} unavailable")
    checks["dependencies"] = module_status
    cache_root = _cache_root()
    cache_status = {}
    for kind, name in MODEL_CACHE_NAMES.items():
        present = (cache_root / name).is_dir()
        cache_status[kind] = {"present": present, "path": str(cache_root / name)}
    checks["model_cache"] = cache_status
    if not cache_status["clip"]["present"]:
        (errors if require_data else warnings).append("CLIP checkpoint cache is missing")
    if require_vlm and not cache_status["vlm"]["present"]:
        errors.append("Qwen2.5-VL checkpoint cache is missing")
    try:
        torch = importlib.import_module("torch")
        cuda = bool(torch.cuda.is_available())
        checks["cuda_available"] = cuda
        if require_gpu and not cuda:
            errors.append("CUDA is required but unavailable")
    except Exception as exc:
        checks["cuda_available"] = False
        if require_gpu:
            errors.append(f"cannot inspect CUDA: {exc}")
    if query_path is not None:
        try:
            rows = [json.loads(line) for line in query_path.read_text(encoding="utf-8").splitlines() if line.strip()]
            query_errors = validate_query_rows(rows)
            checks["query_count"] = len(rows)
            checks["query_errors"] = query_errors
            errors.extend(f"queries: {item}" for item in query_errors)
        except Exception as exc:
            errors.append(f"cannot read queries: {exc}")
    checks["root"] = str(root)
    checks["ready"] = not errors
    return {"ready": not errors, "errors": errors, "warnings": warnings, "checks": checks}


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=None)
    parser.add_argument("--queries", type=Path, default=None)
    parser.add_argument("--require-data", action="store_true")
    parser.add_argument("--require-vlm", action="store_true")
    parser.add_argument("--require-gpu", action="store_true")
    parser.add_argument("--json", action="store_true")
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    root = (args.root or Path(os.environ.get("AIC_ROOT", Path(__file__).resolve().parents[1]))).resolve()
    result = inspect(root, require_data=args.require_data, require_vlm=args.require_vlm,
                     require_gpu=args.require_gpu, query_path=args.queries.resolve() if args.queries else None)
    if args.json:
        print(json.dumps(result, ensure_ascii=False, indent=2))
    else:
        print("READY" if result["ready"] else "NOT_READY")
        for item in result["errors"]:
            print(f"ERROR: {item}")
        for item in result["warnings"]:
            print(f"WARN: {item}")
        if result["ready"]:
            print("All requested checks passed; no model was loaded.")
    return 0 if result["ready"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
