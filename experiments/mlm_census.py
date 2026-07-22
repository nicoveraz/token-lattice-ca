"""Phase 3: attractor census for a real MLM, validated (PROXY) against WikiText-103
trigram statistics. BERT was not trained on WikiText, so overlap/rho are a lower
bound on 'recovers natural English structure', not ground-truth recovery like the
toy. Usage: mlm_census.py --model tiny
"""
import sys as _sys, pathlib as _pathlib
_ROOT = _pathlib.Path(__file__).resolve().parents[1]
_sys.path[:0] = [str(_ROOT / "src"), str(_ROOT / "experiments")]
import argparse, json, time
import numpy as np
from mlm_ca import MLMRule, run
from mlm_lib import MODELS, RESDIR, ref_trigram_counter, ensure_resdir
from census import ngrams, validation


def main(tag, B, N, sweeps, scheme):
    ensure_resdir()
    rule = MLMRule(MODELS[tag])
    corpus_tri = ref_trigram_counter()
    rng = np.random.default_rng(0)
    base_rows = rule.random_lattice(rng, 480, N).tolist()
    base_tri = ngrams(base_rows, 3)
    bov, _, _ = validation(base_tri, corpus_tri)

    results = {}
    t0 = time.time()
    for T in [0.7, 1.0]:
        tc = time.time()
        out = run(rule, B=B, N=N, r=2, T=T, sweeps=sweeps, scheme=scheme,
                  init="random", seed=7)
        rows = out["snaps"][-25:].reshape(-1, N).tolist()
        tri = ngrams(rows, 3)
        ov, rho, nsh = validation(tri, corpus_tri)
        results[str(T)] = dict(overlap50=ov, spearman=rho, shared=nsh,
                               baseline_overlap50=bov, secs=round(time.time() - tc, 1),
                               top_trigrams=[rule.tok.decode(list(g)) for g, _ in tri.most_common(15)],
                               examples=[rule.tok.decode(out["final"][b].tolist()) for b in range(3)])
        print(f"[{tag}] census T={T}: overlap50={ov:.3f} rho={rho:.3f} shared={nsh} "
              f"baseline={bov:.3f} ({results[str(T)]['secs']}s)", flush=True)
        for g in results[str(T)]["top_trigrams"][:6]:
            print("      ", repr(g))
    json.dump(results, open(f"{RESDIR}/{tag}_census.json", "w"), indent=1)
    print(f"[{tag}] CENSUS DONE ({time.time()-t0:.0f}s)", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=list(MODELS))
    ap.add_argument("--B", type=int, default=24)
    ap.add_argument("--N", type=int, default=48)
    ap.add_argument("--sweeps", type=int, default=60)
    ap.add_argument("--scheme", default="cls_sep")
    a = ap.parse_args()
    main(a.model, a.B, a.N, a.sweeps, a.scheme)
