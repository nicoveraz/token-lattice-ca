"""Phase 2: byte-level BPE vocab for tinyshakespeare, replacing the word-level
2000-type vocab whose <unk> token (8% of the stream) acted as artifact attractor
material (F3). Byte-level BPE has NO <unk> — every byte is covered — so the
census should no longer be polluted by <unk>-rich patterns.

Reserves id 0 = <mask> (matches ca.MASK). Lowercase normalizer matches the
word-level pilot's preprocessing so the two vocabs are comparable on the same text.

Outputs data_bpe/{tokenizer.json, train_ids.npy, val_ids.npy, vocab.json, meta.json}.
"""
import sys as _sys, pathlib as _pathlib
_ROOT = _pathlib.Path(__file__).resolve().parents[1]
_sys.path[:0] = [str(_ROOT / "src"), str(_ROOT / "experiments")]
import json, os
import numpy as np
from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders, normalizers

VOCAB_SIZE = 4096
OUT = "data_bpe"
SRC = "data/shakespeare.txt"


def build():
    os.makedirs(OUT, exist_ok=True)
    tok = Tokenizer(models.BPE(unk_token=None))          # byte-level: no <unk> possible
    tok.normalizer = normalizers.Lowercase()
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tok.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(
        vocab_size=VOCAB_SIZE,
        special_tokens=["<mask>"],                        # -> id 0
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),  # all 256 bytes -> full coverage
        show_progress=False,
    )
    tok.train([SRC], trainer)
    tok.save(f"{OUT}/tokenizer.json")

    assert tok.token_to_id("<mask>") == 0, "mask must be id 0 to match ca.MASK"
    V = tok.get_vocab_size()

    text = open(SRC).read()
    ids = np.array(tok.encode(text).ids, dtype=np.int32)
    assert int(ids.max()) < V and int(ids.min()) >= 0
    n_val = int(0.05 * len(ids))
    np.save(f"{OUT}/train_ids.npy", ids[:-n_val])
    np.save(f"{OUT}/val_ids.npy", ids[-n_val:])

    # itos for display/decoding (single-token strings, byte-decoded best-effort)
    vocab = tok.get_vocab()                               # {token_str: id}
    itos = [None] * V
    for s, i in vocab.items():
        itos[i] = s
    json.dump(itos, open(f"{OUT}/vocab.json", "w"))

    # coverage is 1.0 by construction (byte-level). Report a few stats.
    cov = 1.0
    meta = dict(vocab_size=V, tokens=len(ids), train=len(ids) - n_val, val=n_val,
                coverage=cov, mask_id=0,
                compression_chars_per_token=round(len(text) / len(ids), 3))
    json.dump(meta, open(f"{OUT}/meta.json", "w"), indent=1)
    print(json.dumps(meta))
    # sanity: decode the first 20 tokens
    print("sample decode:", repr(tok.decode(ids[:20].tolist())))
    print("BPE DONE")


if __name__ == "__main__":
    build()
