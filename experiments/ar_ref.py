"""Phase C reference corpus for the AR port: the same WikiText-103 slice, tokenized
with the Pythia (GPT-NeoX) BPE vocab. PROXY (Pythia trained on the Pile, not
WikiText) -- a lower bound on natural-English recovery, stated. -> data_ar/ref_ids.npy
"""
import sys as _sys, pathlib as _pathlib
_ROOT = _pathlib.Path(__file__).resolve().parents[1]
_sys.path[:0] = [str(_ROOT / "src"), str(_ROOT / "experiments")]
import os, json
os.environ.setdefault("HF_HOME", "./hf_cache")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
import numpy as np
from transformers import AutoTokenizer
from datasets import load_dataset

MODEL = "EleutherAI/pythia-160m"
MAX_TOKENS = 400_000
OUT = "data_ar"


def build():
    os.makedirs(OUT, exist_ok=True)
    tok = AutoTokenizer.from_pretrained(MODEL)
    ds = load_dataset("Salesforce/wikitext", "wikitext-103-raw-v1", split="validation")
    ids, buf = [], ""
    for ln in ("\n".join(t for t in ds["text"] if t.strip())).split("\n"):
        buf += " " + ln
        if len(buf) > 20000:
            ids.extend(tok.encode(buf)); buf = ""
            if len(ids) >= MAX_TOKENS:
                break
    if buf and len(ids) < MAX_TOKENS:
        ids.extend(tok.encode(buf))
    ids = np.array(ids[:MAX_TOKENS], dtype=np.int64)
    np.save(f"{OUT}/ref_ids.npy", ids)
    meta = dict(model=MODEL, tokens=int(len(ids)), distinct=int(len(np.unique(ids))),
                vocab=int(tok.vocab_size))
    json.dump(meta, open(f"{OUT}/meta.json", "w"), indent=1)
    print(json.dumps(meta))
    print("sample:", repr(tok.decode(ids[:40].tolist())))
    print("AR REF DONE")


if __name__ == "__main__":
    build()
