"""Optional SigLIP2 frame channel for the AIC ensemble.

The channel is deliberately opt-in.  It reads an isolated FAISS index and
never modifies the production CLIP index.  Set ``AIC_SIGLIP_ART_DIR`` to the
artifact directory containing ``siglip.faiss`` and ``clip_meta.json``; when
unset, the full isolated artifact is preferred if it exists.
"""
from __future__ import annotations

import os
from pathlib import Path

import faiss
import numpy as np
import torch


ROOT = Path(os.environ.get("AIC_ROOT", str(Path(__file__).resolve().parents[1])))
EXP = ROOT / "experiments" / "siglip_temporal"
DEFAULT_ART = EXP / "artifacts_full"
if not DEFAULT_ART.exists():
    DEFAULT_ART = EXP / "artifacts"
ART = Path(os.environ.get("AIC_SIGLIP_ART_DIR", str(DEFAULT_ART)))
MAP_DIR = ROOT / "extracted" / "map-keyframes-aic25-b1" / "map-keyframes"
MODEL_DIR = os.environ.get("AIC_SIGLIP_DIR", "google/siglip2-base-patch16-224")


def _unwrap(out):
    if hasattr(out, "text_embeds"):
        return out.text_embeds
    if hasattr(out, "pooler_output"):
        return out.pooler_output
    if isinstance(out, tuple):
        return out[0]
    return out


class SigLIPChannel:
    """Load and search a precomputed SigLIP2 frame index."""

    def __init__(self, device: str = "cuda"):
        if not (ART / "siglip.faiss").exists():
            raise FileNotFoundError(f"missing {ART / 'siglip.faiss'}")
        if not (ART / "clip_meta.json").exists():
            raise FileNotFoundError(f"missing {ART / 'clip_meta.json'}")
        from transformers import SiglipModel, SiglipProcessor

        self.device = device
        self.index = faiss.read_index(str(ART / "siglip.faiss"))
        import json
        with (ART / "clip_meta.json").open() as f:
            raw_meta = json.load(f)
        if self.index.ntotal != len(raw_meta):
            raise RuntimeError(
                f"SigLIP index/meta mismatch: {self.index.ntotal} != {len(raw_meta)}"
            )
        # Isolated manifests identify a keyframe by zero-based row ordinal;
        # the competition requires original video frame indices.  Resolve the
        # ordinal through the organizer map CSV before any answer is emitted.
        map_cache = {}
        self.meta = []
        for vid, row in raw_meta:
            if vid not in map_cache:
                mapping = {}
                path = MAP_DIR / f"{vid}.csv"
                if path.exists():
                    with path.open() as mf:
                        next(mf, None)
                        for line in mf:
                            parts = line.strip().split(",")
                            if len(parts) >= 4:
                                try:
                                    mapping[int(parts[0]) - 1] = int(parts[3])
                                except ValueError:
                                    continue
                map_cache[vid] = mapping
            frame_idx = map_cache[vid].get(int(row))
            if frame_idx is None:
                raise RuntimeError(f"no frame mapping for {vid} row={row}")
            self.meta.append([vid, frame_idx])
        local_only = Path(MODEL_DIR).is_dir()
        self.processor = SiglipProcessor.from_pretrained(
            MODEL_DIR, local_files_only=local_only
        )
        self.model = SiglipModel.from_pretrained(
            MODEL_DIR, local_files_only=local_only
        ).to(device).eval()
        if device.startswith("cuda"):
            self.model = self.model.half()
        print(f"SigLIP2 channel loaded: {self.index.ntotal} frames", flush=True)

    @torch.no_grad()
    def encode_text(self, texts: str | list[str]) -> np.ndarray:
        if isinstance(texts, str):
            texts = [texts]
        inputs = self.processor(
            text=list(texts), padding="max_length", truncation=True,
            max_length=64, return_tensors="pt"
        ).to(self.device)
        out = _unwrap(self.model.get_text_features(input_ids=inputs["input_ids"]))
        out = out / out.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        return out.float().cpu().numpy().astype(np.float32)

    def search(self, text: str, top_k: int = 1000):
        query = self.encode_text(text)
        scores, ids = self.index.search(
            query, min(int(top_k), self.index.ntotal)
        )
        return [
            (int(i), float(s)) for i, s in zip(ids[0], scores[0]) if int(i) >= 0
        ]
