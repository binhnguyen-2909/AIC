"""Query-time dense source-frame refinement around retrieved keyframes."""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import torch


import os


ROOT = Path(os.environ.get("AIC_ROOT", str(Path(__file__).resolve().parents[1])))
VIDEO_ROOTS = sorted((ROOT / "extracted").glob("Videos_*/video"))


def _unwrap(out):
    if hasattr(out, "image_embeds"):
        return out.image_embeds
    if hasattr(out, "pooler_output"):
        return out.pooler_output
    if isinstance(out, tuple):
        return out[0]
    return out


class DenseFrameRefiner:
    """Use the already-loaded CLIP image tower on raw source frames.

    Organizer keyframes can be tens of source frames apart while the GT
    interval is often under ten frames.  This local decoder bridges that
    coordinate-resolution gap without materializing a second corpus index.
    """

    def __init__(self, model, processor, device: str):
        self.model = model
        self.processor = processor
        self.device = device
        self._paths = {}

    def _video_path(self, video_id: str):
        if video_id not in self._paths:
            found = []
            for root in VIDEO_ROOTS:
                path = root / f"{video_id}.mp4"
                if path.exists():
                    found.append(path)
            self._paths[video_id] = found[0] if found else None
        return self._paths[video_id]

    @torch.no_grad()
    def refine(self, video_id: str, center_frame: int, text_embedding,
               window: int = 150, top_k: int = 5):
        path = self._video_path(video_id)
        if path is None:
            return []
        start = max(0, int(center_frame) - int(window))
        end = int(center_frame) + int(window)
        cap = cv2.VideoCapture(str(path))
        if not cap.isOpened():
            return []
        cap.set(cv2.CAP_PROP_POS_FRAMES, start)
        images = []
        frame_ids = []
        pos = start
        while pos <= end:
            ok, bgr = cap.read()
            if not ok:
                break
            # Sample every source frame in the local window.  This is only
            # used for a few candidate videos, so it is cheaper than a corpus
            # re-index and preserves narrow frame intervals.
            images.append(bgr[:, :, ::-1].copy())
            frame_ids.append(pos)
            pos += 1
        cap.release()
        if not images:
            return []
        inputs = self.processor(images=images, return_tensors="pt")
        pixels = inputs["pixel_values"].to(self.device)
        if self.device.startswith("cuda"):
            pixels = pixels.half()
        image_emb = _unwrap(self.model.get_image_features(pixel_values=pixels))
        image_emb = image_emb / image_emb.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        scores = (image_emb.float() @ text_embedding.reshape(-1)).cpu().numpy()
        order = np.argsort(-scores)[:max(1, int(top_k))]
        return [(int(frame_ids[int(i)]), float(scores[int(i)])) for i in order]
