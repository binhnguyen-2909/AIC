"""
Q&A pipeline with Qwen2.5-VL-3B visual answer.
For each query: top-1 frame -> ask VLM the question -> return answer.
"""
import os
import sys
import json
import argparse
import csv
import re
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import torch
from PIL import Image
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor

ROOT = Path(os.environ.get("AIC_ROOT", str(Path(__file__).resolve().parents[1])))
KEYFRAMES_BASE = ROOT / "extracted"
MODEL_CACHE = os.environ.get(
    "AIC_MODEL_CACHE", str(Path.home() / ".cache" / "huggingface" / "hub")
)


def clean_vlm_answer(value: str) -> str:
    """Keep one concise answer line and remove common generation wrappers."""
    answer = str(value or "").strip()
    answer = answer.splitlines()[0].strip() if answer else ""
    answer = re.sub(
        r"^(?:Trả lời|Đáp án|Kết quả|Answer|答案)\s*[:：\-]\s*",
        "", answer, flags=re.IGNORECASE,
    )
    return answer.strip().strip('"“”\'')


class VLMAnswerer:
    def __init__(self, device="cuda"):
        print("[vlm] loading Qwen2.5-VL-3B-Instruct...")
        self.processor = AutoProcessor.from_pretrained(
            "Qwen/Qwen2.5-VL-3B-Instruct",
            cache_dir=MODEL_CACHE,
        )
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            "Qwen/Qwen2.5-VL-3B-Instruct",
            torch_dtype=torch.bfloat16,
            device_map=device,
            cache_dir=MODEL_CACHE,
        ).eval()
        self.device = device

    @torch.no_grad()
    def answer(self, image_path, question, max_new_tokens=64):
        with Image.open(image_path) as source:
            img = source.convert("RGB").copy()
        messages = [{
            "role": "user",
            "content": [
                {"type": "image", "image": img},
                {"type": "text", "text": (
                    "Trả lời thật ngắn gọn bằng tiếng Việt hoặc tiếng Anh. "
                    "Nếu câu hỏi hỏi số lượng, chỉ trả về số hoặc số bằng chữ. "
                    "Không giải thích, không thêm tiền tố.\n"
                    f"Câu hỏi: {question}"
                )},
            ],
        }]
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.processor(text=[text], images=[img], return_tensors="pt").to(self.device)
        out = self.model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
        gen = out[0][inputs.input_ids.shape[1]:]
        ans = self.processor.decode(gen, skip_special_tokens=True)
        return clean_vlm_answer(ans)

    @torch.no_grad()
    def answer_multi(self, image_paths, question, max_new_tokens=64):
        """Answer from a bounded chronological keyframe neighborhood.

        This is opt-in because the competition scorer/answer semantics are not
        available locally.  A single path delegates to ``answer`` so the
        default route remains identical to the single-frame path.
        """
        paths = [Path(path) for path in image_paths if path]
        if not paths:
            return ""
        if len(paths) == 1:
            return self.answer(str(paths[0]), question, max_new_tokens)

        images = []
        for path in paths:
            with Image.open(path) as source:
                images.append(source.convert("RGB").copy())
        content = [{"type": "image", "image": image} for image in images]
        content.append({
            "type": "text",
            "text": (
                "Đây là các keyframe liên tiếp theo thứ tự thời gian của cùng một video. "
                "Trả lời thật ngắn gọn bằng tiếng Việt hoặc tiếng Anh. "
                "Nếu câu hỏi hỏi số lượng, chỉ trả về số hoặc số bằng chữ. "
                "Không giải thích, không thêm tiền tố.\n"
                f"Câu hỏi: {question}"
            ),
        })
        messages = [{"role": "user", "content": content}]
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.processor(
            text=[text], images=images, return_tensors="pt"
        ).to(self.device)
        out = self.model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
        gen = out[0][inputs.input_ids.shape[1]:]
        ans = self.processor.decode(gen, skip_special_tokens=True)
        return clean_vlm_answer(ans)


def _keyframe_rows(video_id):
    """Read ``(ordinal, source_frame_idx)`` rows in temporal order."""
    map_csv = (
        ROOT / "extracted" / "map-keyframes-aic25-b1" / "map-keyframes"
        / f"{video_id}.csv"
    )
    if not map_csv.exists():
        return []
    rows = []
    with map_csv.open(newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            try:
                rows.append((int(row["n"]), int(row["frame_idx"])))
            except (KeyError, TypeError, ValueError):
                continue
    return sorted(rows)


def _path_for_ordinal(video_id, ordinal):
    parts = video_id.split("_", 1)
    if len(parts) != 2:
        return None
    batch = parts[0]
    fname = f"{int(ordinal):03d}.jpg"
    batch_dirs = [KEYFRAMES_BASE / f"Keyframes_{batch}"]
    batch_dirs.extend(sorted(KEYFRAMES_BASE.glob(f"Keyframes_{batch}_*")))
    for batch_dir in batch_dirs:
        path = batch_dir / "keyframes" / video_id / fname
        if path.exists():
            return path
    return None


def keyframe_path(video_id, frame_idx):
    """Resolve the closest organizer keyframe to a source frame index."""
    rows = _keyframe_rows(video_id)
    if not rows:
        return None
    ordinal, _ = min(rows, key=lambda row: abs(row[1] - int(frame_idx)))
    return _path_for_ordinal(video_id, ordinal)


def keyframe_paths(video_id, frame_idx, max_frames=1):
    """Resolve a centered, temporally ordered keyframe neighborhood."""
    rows = _keyframe_rows(video_id)
    if not rows:
        return []
    limit = max(1, int(max_frames))
    center = min(
        range(len(rows)), key=lambda index: abs(rows[index][1] - int(frame_idx))
    )
    if limit >= len(rows):
        selected = rows
    else:
        start = max(0, min(center - limit // 2, len(rows) - limit))
        selected = rows[start:start + limit]
    return [path for ordinal, _ in selected
            if (path := _path_for_ordinal(video_id, ordinal)) is not None]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--queries", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--top-k", type=int, default=100)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    sys.path.insert(0, str(ROOT / "solution"))
    from submission_v2 import Retriever

    retriever = Retriever(device=args.device)
    vlm = VLMAnswerer(device=args.device)

    with open(args.queries) as f:
        queries = [json.loads(line) for line in f if line.strip()]
    print(f"[qa] {len(queries)} queries")

    with open(args.out, "w") as fout:
        for q in queries:
            qid = q.get("query_id", "?")
            qt = q.get("query_type", "?")
            text = q.get("query_text", "")
            question = q.get("question", text)

            if qt != "QA":
                # fallback to submission_v2 behavior
                if qt == "KIS":
                    ans = retriever.search(text, args.top_k)
                    lines = [f"{v}, {f}" for v, f in ans]
                elif qt == "TRAKE":
                    events = q.get("events", [])
                    from submission_v2 import generate_trake
                    lines = generate_trake(retriever, text, events)
                else:
                    lines = []
                for rank, line in enumerate(lines, 1):
                    fout.write(json.dumps({"query_id": qid, "rank": rank, "answer": line}) + "\n")
                continue

            # QA: retrieve + VLM answer
            results = retriever.search(text, top_k=20)
            top_vid, top_fi = results[0] if results else (None, None)
            img_path = keyframe_path(top_vid, top_fi) if top_vid else None
            answer = ""
            if img_path:
                try:
                    answer = vlm.answer(str(img_path), question)
                except Exception as e:
                    print(f"[qa] VLM err {qid}: {e}")
                    answer = ""
            if not answer:
                answer = ""
            lines = [f"{v}, {f}, {answer}" for v, f in results]
            for rank, line in enumerate(lines, 1):
                fout.write(json.dumps({"query_id": qid, "rank": rank, "answer": line}) + "\n")
            print(f"Q{qid} QA: {top_vid} {top_fi} -> '{answer[:80]}'")

    print(f"Saved {args.out}")


if __name__ == "__main__":
    main()
