"""
Build BM25 text index over media-info (title + description + keywords).

Each video gets one BM25 doc. Tokenize via pyvi ViTokenizer to handle Vietnamese
compound words. Store the BM25 index and the parallel list of video_ids.

Output:
  solution/ensemble_index/bm25.pkl
    keys: 'video_ids' (list[str]), 'doc_lens' (list[int]),
          'idf' (np.array), 'avgdl', 'k1', 'b'
          plus 'tokenized' (list[list[str]]) so we don't re-tokenize at query time.
"""
import os, json, pickle, re, math
from pathlib import Path
import numpy as np
from pyvi import ViTokenizer
from rank_bm25 import BM25Okapi

ROOT = Path(os.environ.get("AIC_ROOT", str(Path(__file__).resolve().parents[1])))
EXTRACTED = ROOT / "extracted"
META_DIR = EXTRACTED / "media-info-aic25-b1" / "media-info"
OUT = ROOT / "solution" / "ensemble_index"
OUT.mkdir(parents=True, exist_ok=True)


def clean(text: str) -> str:
    text = text.lower()
    # remove urls
    text = re.sub(r"https?://\S+", " ", text)
    # strip non-vietnamese noise (keep diacritics + ascii letters/digits)
    text = re.sub(r"[^\w\sàáảãạăắằẳẵặâấầẩẫậèéẻẽẹêếềểễệìíỉĩịòóỏõọôốồổỗộơớờởỡợùúủũụưứừửữựỳýỷỹỵđ]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def tokenize(text: str) -> list[str]:
    text = clean(text)
    text = ViTokenizer.tokenize(text)
    return [t for t in text.split() if len(t) > 1]


def build_doc(meta: dict) -> str:
    parts = [meta.get("title", "")]
    parts.append(meta.get("description", ""))
    # keywords heavily weighted by repetition
    kw = meta.get("keywords", [])
    if kw:
        parts.append(" ".join(kw * 3))
    return " ".join(parts)


def main():
    video_ids = []
    docs = []
    files = sorted(META_DIR.glob("*.json"))
    print(f"Found {len(files)} media-info files")
    for f in files:
        try:
            with open(f) as fp:
                meta = json.load(fp)
        except Exception:
            continue
        vid = f.stem
        video_ids.append(vid)
        docs.append(build_doc(meta))

    print("Tokenizing docs...")
    tokenized = [tokenize(d) for d in docs]
    print(f"avg tokens/doc: {np.mean([len(t) for t in tokenized]):.1f}")
    print("Building BM25...")
    bm25 = BM25Okapi(tokenized)

    # Save
    payload = {
        "video_ids": video_ids,
        "tokenized": tokenized,
        "avgdl": bm25.doc_len,  # raw list of doc lens
        "doc_len": [len(t) for t in tokenized],
        "idf": bm25.idf,
        "k1": bm25.k1,
        "b": bm25.b,
    }
    with open(OUT / "bm25.pkl", "wb") as fp:
        pickle.dump(payload, fp)
    print(f"Saved to {OUT/'bm25.pkl'}, {len(video_ids)} videos")


if __name__ == "__main__":
    main()
