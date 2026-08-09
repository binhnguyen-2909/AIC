"""
Q&A pipeline with Qwen2.5-VL-3B visual answer.
For each query: top-1 frame -> ask VLM the question -> return answer.
"""
import os
import sys
import json
import argparse
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import torch
from PIL import Image
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor

ROOT = Path(os.environ.get("AIC_ROOT", str(Path(__file__).resolve().parents[1])))
KEYFRAMES_BASE = ROOT / "extracted"


class VLMAnswerer:
    def __init__(self, device="cuda"):
        print("[vlm] loading Qwen2.5-VL-3B-Instruct...")
        self.processor = AutoProcessor.from_pretrained(
            "Qwen/Qwen2.5-VL-3B-Instruct",
        )
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            "Qwen/Qwen2.5-VL-3B-Instruct",
            torch_dtype=torch.bfloat16,
            device_map=device,
        ).eval()
        self.device = device

    @torch.no_grad()
    def answer(self, image_path, question, max_new_tokens=64):
        img = Image.open(image_path).convert("RGB")
        messages = [{
            "role": "user",
            "content": [
                {"type": "image", "image": img},
                {"type": "text", "text": f"Trả lời ngắn gọn bằng tiếng Việt hoặc tiếng Anh: {question}"},
            ],
        }]
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.processor(text=[text], images=[img], return_tensors="pt").to(self.device)
        out = self.model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
        gen = out[0][inputs.input_ids.shape[1]:]
        ans = self.processor.decode(gen, skip_special_tokens=True).strip()
        return ans
    
    @torch.no_grad()
    def answer_multi(self, image_paths, question, max_new_tokens=64):
        """Processes multiple chronological frames as a single video sequence."""
        # Open and filter valid images
        images = [Image.open(p).convert("RGB") for p in image_paths if p is not None]
        
        if not images:
            return ""

        # Build the multi-image content payload
        content = []
        for img in images:
            content.append({"type": "image", "image": img})
            
        # Add context to the prompt telling the model these are sequential frames
        content.append({
            "type": "text", 
            "text": f"Những hình ảnh này là các khung hình liên tiếp từ một video. Trả lời ngắn gọn dựa trên toàn bộ chuỗi hình ảnh: {question}"
        })

        messages = [{"role": "user", "content": content}]
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        
        inputs = self.processor(text=[text], images=images, return_tensors="pt").to(self.device)
        out = self.model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
        
        gen = out[0][inputs.input_ids.shape[1]:]
        ans = self.processor.decode(gen, skip_special_tokens=True).strip()
        return ans


def keyframe_path(video_id, frame_idx):
    """Resolve to actual JPG path."""
    # parse video_id like L21_V001 -> batch L21
    parts = video_id.split("_")
    batch = parts[0]  # L21
    vid_short = parts[1]  # V001
    # frame_idx is original index from map-keyframes; keyframe filename is the n-th keyframe
    # we need the mapping from frame_idx to keyframe n-th
    map_csv = ROOT / "extracted" / "map-keyframes-aic25-b1" / "map-keyframes" / f"{video_id}.csv"
    if not map_csv.exists():
        return None
    rows = []
    with open(map_csv) as f:
        next(f)
        for line in f:
            parts = line.strip().split(",")
            if len(parts) >= 4:
                rows.append((int(parts[0]), int(parts[3])))  # (n, frame_idx)
    # find closest keyframe
    best = min(rows, key=lambda r: abs(r[1] - frame_idx))
    n = best[0]
    fname = f"{n:03d}.jpg"
    batch_dirs = [KEYFRAMES_BASE / f"Keyframes_{batch}"]
    batch_dirs.extend(sorted(KEYFRAMES_BASE.glob(f"Keyframes_{batch}_*")))
    for batch_dir in batch_dirs:
        p = batch_dir / "keyframes" / video_id / fname
        if p.exists():
            return p
    return None


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
