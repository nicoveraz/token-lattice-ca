"""Build word-level vocab + token stream for tinyshakespeare."""
import json, re
import numpy as np

MASK, UNK = 0, 1  # special ids

def tokenize(text):
    text = text.lower()
    # newline -> <nl> token; words; punctuation as single tokens
    out = []
    for line in text.split("\n"):
        toks = re.findall(r"[a-z']+|[^\sa-z']", line)
        out.extend(toks)
        out.append("<nl>")
    return out

def main(vocab_size=2000):
    text = open("data/shakespeare.txt").read()
    toks = tokenize(text)
    from collections import Counter
    cnt = Counter(toks)
    keep = [w for w, _ in cnt.most_common(vocab_size - 2)]
    itos = ["<mask>", "<unk>"] + keep
    stoi = {w: i for i, w in enumerate(itos)}
    ids = np.array([stoi.get(t, UNK) for t in toks], dtype=np.int32)
    cov = float((ids != UNK).mean())
    n_val = int(0.05 * len(ids))
    np.save("data/train_ids.npy", ids[:-n_val])
    np.save("data/val_ids.npy", ids[-n_val:])
    json.dump(itos, open("data/vocab.json", "w"))
    print(f"tokens={len(ids)}  types={len(cnt)}  vocab={len(itos)}  coverage={cov:.3f}")
    print("sample:", " ".join(itos[i] for i in ids[:30]))

if __name__ == "__main__":
    main()
