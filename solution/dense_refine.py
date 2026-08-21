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

def _device_vector(value, device: str) -> torch.Tensor:
    """Convert a NumPy/CPU/CUDA embedding to one float32 device vector."""
    if isinstance(value, torch.Tensor): return value.to(device=device, dtype=torch.float32).reshape(-1)
    return torch.as_tensor(value, device=device, dtype=torch.float32).reshape(-1)


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
        text_vector = _device_vector(text_embedding, str(image_emb.device))
        image_vector = image_emb.float()
        if text_vector.numel() != image_vector.shape[-1]:
            raise ValueError(f"embedding dimension mismatch: image={image_vector.shape[-1]} text={text_vector.numel()}")
        scores = (image_vector @ text_vector).cpu().numpy()
        order = np.argsort(-scores)[:max(1, int(top_k))]
        return [(int(frame_ids[int(i)]), float(scores[int(i)])) for i in order]
    
def get_contiguous_keyframes(video_id: str, center_frame: int, window: int = 2):
    """Retrieves paths for the center keyframe and its adjacent temporal neighbors."""
    map_csv = ROOT / "extracted" / "map-keyframes-aic25-b1" / "map-keyframes" / f"{video_id}.csv"
    if not map_csv.exists():
        return []
    
    # Read the mapping: n (ordinal) -> frame_idx
    rows = []
    with open(map_csv) as f:
        next(f)
        for line in f:
            parts = line.strip().split(",")
            if len(parts) >= 4:
                rows.append((int(parts[0]), int(parts[3]))) 
                
    if not rows:
        return []
        
    # Find the row index of the closest keyframe to our center_frame
    best_idx = min(range(len(rows)), key=lambda i: abs(rows[i][1] - center_frame))
    
    start_idx = max(0, best_idx - window)
    end_idx = min(len(rows), best_idx + window + 1)
    
    # We must import keyframe_path locally to avoid circular imports
    try:
        from qa_vlm import keyframe_path
    except ImportError:
        from solution.qa_vlm import keyframe_path
        
    sequence_paths = []
    # Reconstruct the original frame_idx for each neighbor and resolve its path
    for i in range(start_idx, end_idx):
        neighbor_frame_idx = rows[i][1]
        path = keyframe_path(video_id, neighbor_frame_idx)
        if path:
            sequence_paths.append(path)
            
    return sequence_paths
