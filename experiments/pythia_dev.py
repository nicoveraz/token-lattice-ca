"""Phase D (developmental): run the instrument across Pythia-160m's public training
checkpoints. The real-training analog of F7 (toy crystallization). Two questions:
  (1) Does the repair length / capacity->sensitivity climb EMERGE during pretraining
      — i.e. does D_norm (asymptotic perturbation persistence) rise as the model trains?
  (2) Order of acquisition: does local/format structure (bigram overlap, census) form
      BEFORE the model becomes dynamically sensitive?
Compact: D_norm at the discriminating radii r=2,4; order (bigram overlap vs the Pythia-
tokenized WikiText proxy); census trigram overlap. Usage: pythia_dev.py
"""
import sys as _sys, pathlib as _pathlib
_ROOT = _pathlib.Path(__file__).resolve().parents[1]
_sys.path[:0] = [str(_ROOT / "src"), str(_ROOT / "experiments")]
import os, json, time
os.environ.setdefault("HF_HOME", "./hf_cache")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
from collections import Counter
import numpy as np
from ar_ca import ARRule, run
from ar_probe import block_damage, drift_floor
from mlm_lib import RESDIR, ensure_resdir
from census import ngrams, validation

MODEL = "EleutherAI/pythia-160m"
CKPTS = ["step0", "step512", "step4000", "step32000", "step143000"]
RS = [2, 4]
T = 0.7
SEEDS = [21, 22]
N, B, SW = 48, 20, 24


def probe(rule, ref_bi, ref_tri):
    out = {}
    for r in RS:
        dn = []
        for sd in SEEDS:
            d = block_damage(rule, T, r, B=B, N=N, sweeps=SW, seed=sd)["mean_damage"]
            d0, _ = drift_floor(rule, T, r, B=B, N=N, sweeps=SW, seed=sd)
            dn.append(d / max(d0, 1e-3))
        out[f"Dnorm_r{r}"] = round(float(np.mean(dn)), 4)
    o = run(rule, B=B, N=N, r=4, T=T, sweeps=40, scheme="none", init="random", seed=7)
    rows = o["snaps"][-15:].reshape(-1, N)
    out["order_bigram"] = round(float(np.mean(
        [[(int(row[i]), int(row[i + 1])) in ref_bi for i in range(N - 1)] for row in rows])), 4)
    tri = ngrams(rows.tolist(), 3)
    ov, rho, _ = validation(tri, ref_tri)
    out["census_overlap50"], out["census_rho"] = round(ov, 4), round(float(rho), 4)
    return out


def main():
    ensure_resdir()
    ref = np.load("data_ar/ref_ids.npy")
    ref_bi = set(zip(ref[:-1].tolist(), ref[1:].tolist()))
    ref_tri = Counter(tuple(ref[i:i + 3].tolist()) for i in range(len(ref) - 2))
    res = {"model": MODEL, "ckpts": CKPTS, "RS": RS, "T": T}
    t0 = time.time()
    for ck in CKPTS:
        tc = time.time()
        rule = ARRule(MODEL, revision=ck)
        res[ck] = probe(rule, ref_bi, ref_tri)
        res[ck]["secs"] = round(time.time() - tc, 1)
        print(f"[{ck}] {json.dumps(res[ck])}", flush=True)
        del rule
    json.dump(res, open(f"{RESDIR}/pythia_dev.json", "w"), indent=1)
    print(f"PYTHIA DEV DONE ({time.time()-t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
