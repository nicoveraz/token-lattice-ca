"""Cross-level, type-matched follow-up (issue #4). The primary pairing (white-box
lambda_top vs black-box asymptotic D_norm) was null (rho=-0.49, p=0.33) -- but those
are different TYPES (growth-rate vs persistence). Here we test the pre-registered
dimensional twin: the BLACK-BOX token-space finite-size Lyapunov lambda_ca (early
exponential growth rate of the damage cone) vs the WHITE-BOX activation-space
lambda_top. Both are top-Lyapunov exponents; if the CA proxies internal criticality,
THIS is where it should show.

Two pre-registered operating points (reported both, no cherry-pick):
  * lambda_ca_max : max over a small (r,T) grid  -- twin of the power-iteration MAX in lambda_top
  * lambda_ca_r2  : at (r=2, T=0.7)              -- matched to the D_norm probe cell
Merges into results/mlm/crosslevel.json; correlates each vs white lambda_top and vs size.
Resumable, MPS flush, caffeinate-friendly. Usage: crosslevel_lyap.py
"""
import sys as _sys, pathlib as _pathlib
_ROOT = _pathlib.Path(__file__).resolve().parents[1]
_sys.path[:0] = [str(_ROOT / "src"), str(_ROOT / "experiments")]
import os, json, gc, time
os.environ.setdefault("HF_HOME", "./hf_cache")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
import numpy as np
import torch
from scipy import stats
from mlm_lib import RESDIR
from lyapunov import lyap_from_cone

RS, TS, SEEDS, N = [1, 2, 4, 8], [0.7, 0.9], [21, 22], 48
LADDER = [
    ("pythia-14m",  "EleutherAI/pythia-14m",  20),
    ("pythia-31m",  "EleutherAI/pythia-31m",  20),
    ("pythia-70m",  "EleutherAI/pythia-70m",  20),
    ("pythia-160m", "EleutherAI/pythia-160m", 20),
    ("pythia-410m", "EleutherAI/pythia-410m", 16),
    ("pythia-1b",   "EleutherAI/pythia-1b",   12),
]
OUT = f"{RESDIR}/crosslevel.json"


def black_lyap(name, B):
    from ar_ca import ARRule
    from ar_probe import block_damage
    rule = ARRule(name)
    lam = {}
    for r in RS:
        for T in TS:
            ls = []
            for sd in SEEDS:
                d = block_damage(rule, T, r, block=3, B=B, N=N, settle=12,
                                 sweeps=22, seed=sd, scheme="none")
                ls.append(lyap_from_cone(d["cone"], N)[0])
            lam[(r, T)] = float(np.mean(ls))
            print(f"    lambda_ca r={r:>2} T={T}: {lam[(r,T)]:+.4f}/sweep", flush=True)
    rule.model = None; del rule; gc.collect()
    try: torch.mps.empty_cache()
    except Exception: pass
    lam_max = max(lam.values())
    at = [k for k, v in lam.items() if v == lam_max][0]
    return dict(grid={f"{r}|{T}": round(lam[(r, T)], 4) for r in RS for T in TS},
                lambda_ca_max=round(lam_max, 4), at_r=at[0], at_T=at[1],
                lambda_ca_r2=round(lam[(2, 0.7)], 4))


def correlate(res):
    rows = [(t, res[t]["size_M"], res[t]["white"]["lambda_top"],
             res[t]["black_lyap"]["lambda_ca_max"], res[t]["black_lyap"]["lambda_ca_r2"])
            for t in res if isinstance(res[t], dict) and "black_lyap" in res[t]]
    if len(rows) < 3:
        return None
    _, size, wtop, lmax, lr2 = zip(*rows)
    out = {"n": len(rows), "models": [r[0] for r in rows]}
    for lab, q in [("lambda_ca_max", lmax), ("lambda_ca_r2", lr2)]:
        sp = stats.spearmanr(wtop, q); pr = stats.pearsonr(wtop, q)
        out[f"white_top_vs_{lab}"] = dict(spearman_rho=round(float(sp.correlation), 3),
                                          spearman_p=round(float(sp.pvalue), 4),
                                          pearson_r=round(float(pr[0]), 3), pearson_p=round(float(pr[1]), 4))
        s = stats.spearmanr(size, q)
        out[f"{lab}_vs_size"] = dict(rho=round(float(s.correlation), 3), p=round(float(s.pvalue), 4))
    return out


def main():
    res = json.load(open(OUT)) if os.path.exists(OUT) else {}
    for tag, name, B in LADDER:
        if tag in res and "black_lyap" in res.get(tag, {}):
            print(f"[{tag}] SKIP (done)", flush=True); continue
        if tag not in res:
            print(f"[{tag}] WARN: no white-box entry; run crosslevel.py first", flush=True); continue
        print(f"[{tag}] black-box token-space Lyapunov ...", flush=True)
        t0 = time.time()
        res[tag]["black_lyap"] = black_lyap(name, B)
        bl = res[tag]["black_lyap"]
        print(f"[{tag}] lambda_ca_max={bl['lambda_ca_max']:+.4f} (r={bl['at_r']},T={bl['at_T']})  "
              f"lambda_ca_r2={bl['lambda_ca_r2']:+.4f}  ({time.time()-t0:.0f}s)", flush=True)
        json.dump(res, open(OUT, "w"), indent=1)
    corr = correlate(res)
    if corr:
        res["_correlation_lyap"] = corr
        json.dump(res, open(OUT, "w"), indent=1)
        print("\n=== TYPE-MATCHED CROSS-LEVEL (n=%d) ===" % corr["n"], flush=True)
        for lab in ["lambda_ca_max", "lambda_ca_r2"]:
            c = corr[f"white_top_vs_{lab}"]
            print(f"  white lambda_top vs black {lab}:  Spearman rho={c['spearman_rho']} p={c['spearman_p']}  "
                  f"(Pearson r={c['pearson_r']} p={c['pearson_p']})", flush=True)
            s = corr[f"{lab}_vs_size"]
            print(f"      {lab} vs size: rho={s['rho']} p={s['p']}", flush=True)
    print("CROSSLEVEL_LYAP DONE", flush=True)


if __name__ == "__main__":
    main()
