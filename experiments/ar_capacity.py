"""Firm up (or honestly kill) the AR capacity->sensitivity trend (F24). Focused probe:
the diversity-controlled damping-length metric D_norm at the DISCRIMINATING radius r=2,
over T in {0.5,0.7}, with 5 seeds, across four Pythia sizes (70m/160m/410m/1b). Reports
per-model mean +/- SE and a rank-correlation trend test + adjacent-pair tests, so the
capacity claim gets real error bars instead of 2-seed point estimates. r=2 keeps windows
tiny (seq~3) so even 1b fits at fp16 on 16GB. Usage: ar_capacity.py
"""
import sys as _sys, pathlib as _pathlib
_ROOT = _pathlib.Path(__file__).resolve().parents[1]
_sys.path[:0] = [str(_ROOT / "src"), str(_ROOT / "experiments")]
import os, json, time
os.environ.setdefault("HF_HOME", "./hf_cache")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
import numpy as np
from scipy import stats
from ar_ca import ARRule
from ar_probe import block_damage, drift_floor, MODELS
from mlm_lib import RESDIR, ensure_resdir

ORDER = ["pythia-70m", "pythia-160m", "pythia-410m", "pythia-1b"]
SIZE = {"pythia-70m": 70, "pythia-160m": 160, "pythia-410m": 410, "pythia-1b": 1000}
R, TS, SEEDS, N, B, SW = 2, [0.5, 0.7], [21, 22, 23, 24, 25], 48, 20, 26


OUT = None  # set in main


def _analyze_and_save(res):
    """Rebuild the trend test from whatever per-model results exist, and save."""
    points = []
    for tag in ORDER:
        if tag in res and "vals" in res[tag]:
            points += [(np.log10(SIZE[tag]), v) for v in res[tag]["vals"]]
    out = {tag: dict(res[tag]) for tag in res if tag in ORDER}   # keep vals for resume
    if len(points) > 4:
        x, y = zip(*points)
        sp = stats.spearmanr(x, y)
        out["_trend"] = dict(spearman_rho=round(float(sp.correlation), 3),
                             spearman_p=round(float(sp.pvalue), 4), n_points=len(points))
        raw = {tag: res[tag]["vals"] for tag in res if tag in ORDER and "vals" in res[tag]}
        adj = {}
        for a, b in zip(ORDER[:-1], ORDER[1:]):
            if raw.get(a) and raw.get(b):
                u = stats.mannwhitneyu(raw[b], raw[a], alternative="greater")
                adj[f"{b}>{a}"] = dict(gap=round(float(np.mean(raw[b]) - np.mean(raw[a])), 3),
                                       p=round(float(u.pvalue), 4))
        out["_adjacent_tests"] = adj
    json.dump(out, open(OUT, "w"), indent=1)
    return out


def main():
    global OUT
    ensure_resdir()
    OUT = f"{RESDIR}/ar_capacity.json"
    res = {}
    if os.path.exists(OUT):                          # resume: skip models already done
        prev = json.load(open(OUT))
        for tag in ORDER:
            if tag in prev and "mean" in prev[tag]:
                res[tag] = prev[tag]; res[tag]["vals"] = res[tag].get("vals", [])
                print(f"[{tag}] SKIP (already done)", flush=True)
    for tag in ORDER:
        if tag in res:
            continue
        Bm = 12 if tag == "pythia-1b" else B         # smaller batch for 1b (memory)
        try:
            rule = ARRule(MODELS[tag])
        except Exception as e:
            print(f"[{tag}] SKIP ({e})", flush=True); continue
        vals = []
        for T in TS:
            for sd in SEEDS:
                d = block_damage(rule, T, R, block=3, B=Bm, N=N, settle=12, sweeps=SW, seed=sd)
                d0, _ = drift_floor(rule, T, R, B=Bm, N=N, settle=12, sweeps=SW, seed=sd)
                vals.append(d["mean_damage"] / max(d0, 1e-3))
        v = np.array(vals)
        res[tag] = dict(size_M=SIZE[tag], n=len(v), mean=round(float(v.mean()), 4),
                        se=round(float(v.std(ddof=1) / np.sqrt(len(v))), 4),
                        by_T={str(T): round(float(np.mean([vals[i] for i in range(len(vals))
                              if TS[i // len(SEEDS)] == T])), 4) for T in TS},
                        vals=[round(float(x), 4) for x in vals])
        print(f"[{tag}] D_norm(r=2)={res[tag]['mean']:.3f}+/-{res[tag]['se']:.3f} (n={len(v)})", flush=True)
        _analyze_and_save(res)                        # incremental save after each model
        del rule
    out = _analyze_and_save(res)
    if "_trend" in out:
        print(f"\nTREND: Spearman rho(log-size, D_norm)={out['_trend']['spearman_rho']} "
              f"p={out['_trend']['spearman_p']}")
        for k, v in out.get("_adjacent_tests", {}).items():
            print(f"  {k}: gap={v['gap']:+.3f} p={v['p']}")
    print("AR CAPACITY DONE", flush=True)


if __name__ == "__main__":
    main()
