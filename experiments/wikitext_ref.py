"""Phase 3 reference corpus (PROXY validation). BERT was not trained on WikiText,
so a WikiText-103 slice is only a proxy for 'natural English n-gram structure' —
the census validation against it is a lower bound, not ground truth (unlike the
toy, where we censused against the actual training corpus). Stated as a limitation.

Loads the WikiText-103 validation split, tokenizes it with the bert-base-uncased
WordPiece vocab, and saves a token-id stream to data_mlm/ref_ids.npy.
"""
import sys as _sys, pathlib as _pathlib
_ROOT = _pathlib.Path(__file__).resolve().parents[1]
_sys.path[:0] = [str(_ROOT / "src"), str(_ROOT / "experiments")]
import os, json
os.environ.setdefault("HF_HOME", "./hf_cache")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
import numpy as np
from datasets import load_dataset
from mlm_ca import get_tokenizer

MAX_TOKENS = 400_000
OUT = "data_mlm"


def build():
    os.makedirs(OUT, exist_ok=True)
    tok = get_tokenizer()
    # Salesforce/wikitext is the namespaced parquet mirror (no loading script, so
    # it works with recent huggingface_hub); content is identical to WikiText-103.
    ds = load_dataset("Salesforce/wikitext", "wikitext-103-raw-v1", split="validation")
    text = "\n".join(t for t in ds["text"] if t.strip())
    # encode in chunks, no special tokens (we want the raw token stream)
    ids = []
    step = 2000
    lines = text.split("\n")
    buf = ""
    for ln in lines:
        buf += " " + ln
        if len(buf) > 20000:
            ids.extend(tok.encode(buf, add_special_tokens=False))
            buf = ""
            if len(ids) >= MAX_TOKENS:
                break
    if buf and len(ids) < MAX_TOKENS:
        ids.extend(tok.encode(buf, add_special_tokens=False))
    ids = np.array(ids[:MAX_TOKENS], dtype=np.int64)
    np.save(f"{OUT}/ref_ids.npy", ids)
    meta = dict(source="wikitext-103-raw-v1/validation", tokens=int(len(ids)),
                distinct=int(len(np.unique(ids))), vocab=int(tok.vocab_size))
    json.dump(meta, open(f"{OUT}/meta.json", "w"), indent=1)
    print(json.dumps(meta))
    print("sample:", repr(tok.decode(ids[:40].tolist())))
    print("WIKITEXT REF DONE")


if __name__ == "__main__":
    build()
