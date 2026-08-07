"""
Qwen3-8B-based Vietnamese->English translator (cached translation).
Used to bridge CLIP-B/32 weakness on Vietnamese text.
"""
import os
os.environ.setdefault("HF_HUB_OFFLINE", "1")
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


class Translator:
    def __init__(self, device="cuda"):
        self.device = device
        self.model = AutoModelForCausalLM.from_pretrained(
            "Qwen/Qwen3-8B",
            dtype=torch.bfloat16,
            device_map=device,
        )
        self.tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-8B")
        self.model.eval()

    @torch.no_grad()
    def translate(self, vi_text, max_new_tokens=80):
        prompt = (
            f"Translate the following Vietnamese text into English concisely. "
            f"Output ONLY the English translation, no commentary.\n\n"
            f"Vietnamese: {vi_text}\nEnglish:"
        )
        messages = [{"role": "user", "content": prompt}]
        text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
        inputs = self.tokenizer(text, return_tensors="pt").to(self.device)
        out = self.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=1.0,
            repetition_penalty=1.0,
        )
        gen = out[0][inputs.input_ids.shape[1]:]
        raw = self.tokenizer.decode(gen, skip_special_tokens=True).strip()
        # strip <think> block if present
        if "</think>" in raw:
            raw = raw.split("</think>")[-1].strip()
        # take first line only (avoid extra commentary)
        return raw.split("\n")[0].strip()


if __name__ == "__main__":
    t = Translator()
    samples = [
        "một diễn giả mặc áo đỏ phát biểu tại cuộc họp báo ngoài trời có cây xanh phía sau",
        "vận động viên nhảy cao thực hiện cú nhảy",
        "chạy đà",
        "giậm nhảy",
    ]
    for s in samples:
        print(f"VI: {s}\nEN: {t.translate(s)}\n")
