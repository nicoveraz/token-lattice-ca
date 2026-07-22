"""Phase 2: re-run the attractor census + corpus-recovery validation on the BPE
model, and compare against the word-level pilot. Tests whether removing the
<unk> artifact (F3) improves corpus recovery (top-50 trigram overlap, Spearman rho).

Same protocol as census.py, but the ca context points at the BPE model/data and
decoding uses the byte-level tokenizer so example attractors are readable.
"""
import sys as _sys, pathlib as _pathlib
_ROOT = _pathlib.Path(__file__).resolve().parents[1]
_sys.path[:0] = [str(_ROOT / "src"), str(_ROOT / "experiments")]
import json, time
import numpy as np
import ca
from model import load
from ca import run
from census import ngrams, validation
from tokenizers import Tokenizer

N = 48


def census_at(params, T, r=2, B=32, sweeps=150, seed=7):
    out = run(params, B=B, N=N, r=r, T=T, sweeps=sweeps, mode="async",
              init="random", seed=seed)
    rows = out["snaps"][-30:].reshape(-1, N).tolist()
    return ngrams(rows, 3), out["final"]


if __name__ == "__main__":
    ca.DATA_DIR, ca.VOCAB, ca.INIT_LO = "data_bpe", 4096, 1
    params = load("ckpt_bpe/final.npz")
    tok = Tokenizer.from_file("data_bpe/tokenizer.json")

    ids = np.load("data_bpe/train_ids.npy").tolist()
    corpus_tri = ngrams([ids], 3)
    rand = np.random.default_rng(0)
    base_tri = ngrams(rand.integers(1, 4096, size=(960, N)).tolist(), 3)

    results = {}
    for T in [0.3, 0.7, 1.0]:
        t0 = time.time()
        tri, final = census_at(params, T)
        ov, rho, nsh = validation(tri, corpus_tri)
        bov, _, _ = validation(base_tri, corpus_tri)
        results[str(T)] = dict(
            overlap50=ov, spearman=rho, shared=nsh, baseline_overlap50=bov,
            secs=round(time.time() - t0, 1),
            top_trigrams=[repr(tok.decode(list(g))) for g, _ in tri.most_common(15)],
            examples=[repr(tok.decode(final[b].tolist())) for b in range(3)],
        )
        print(f"BPE T={T}: overlap50={ov:.3f} rho={rho:.3f} shared={nsh} "
              f"baseline={bov:.3f}  ({results[str(T)]['secs']}s)", flush=True)

    # side-by-side with the word-level pilot census
    try:
        wl = json.load(open("results/census.json"))["census"]
        cmp = {T: dict(bpe_overlap=results[T]["overlap50"], word_overlap=wl[T]["overlap50"],
                       bpe_rho=results[T]["spearman"], word_rho=wl[T]["spearman"])
               for T in ["0.3", "0.7", "1.0"]}
    except Exception as e:
        cmp = {"error": str(e)}
    json.dump(dict(census=results, vs_word_level=cmp),
              open("results/census_bpe.json", "w"), indent=1)
    print("\n=== BPE vs word-level corpus recovery ===")
    for T in ["0.3", "0.7", "1.0"]:
        c = cmp.get(T)
        if c:
            print(f"  T={T}: overlap {c['word_overlap']:.3f} (word) -> {c['bpe_overlap']:.3f} (BPE)"
                  f"   rho {c['word_rho']:.3f} -> {c['bpe_rho']:.3f}")
    print("CENSUS(bpe) DONE", flush=True)
