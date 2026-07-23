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


def main():
    ensure_resdir()
    res, points = {}, []          # points: (log-size, D_norm) for the trend test
    for tag in ORDER:
        try:
            rule = ARRule(MODELS[tag])
        except Exception as e:
            print(f"[{tag}] SKIP ({e})", flush=True); continue
        vals = []
        for T in TS:
            for sd in SEEDS:
                d = block_damage(rule, T, R, block=3, B=B, N=N, settle=12, sweeps=SW, seed=sd)
                d0, _ = drift_floor(rule, T, R, B=B, N=N, settle=12, sweeps=SW, seed=sd)
                dn = d["mean_damage"] / max(d0, 1e-3)
                vals.append(dn); points.append((np.log10(SIZE[tag]), dn))
        vals = np.array(vals)
        res[tag] = dict(size_M=SIZE[tag], n=len(vals),
                        mean=round(float(vals.mean()), 4),
                        se=round(float(vals.std(ddof=1) / np.sqrt(len(vals))), 4),
                        by_T={str(T): round(float(np.mean([v for i, v in enumerate(vals)
                              if TS[i // len(SEEDS)] == T])), 4) for T in TS})
        print(f"[{tag}] D_norm(r=2)={res[tag]['mean']:.3f}+/-{res[tag]['se']:.3f} (n={len(vals)})", flush=True)
        del rule
    # trend test across all points
    if len(points) > 4:
        x, y = zip(*points)
        sp = stats.spearmanr(x, y)
        res["_trend"] = dict(spearman_rho=round(float(sp.correlation), 3),
                             spearman_p=round(float(sp.pvalue), 4), n_points=len(points))
        # adjacent-pair Mann-Whitney (one-sided greater for the larger model)
        raw = {tag: [] for tag in ORDER}
        for (lx, dn) in points:
            for tag in ORDER:
                if abs(lx - np.log10(SIZE[tag])) < 1e-9: raw[tag].append(dn)
        adj = {}
        for a, b in zip(ORDER[:-1], ORDER[1:]):
            if raw[a] and raw[b]:
                u = stats.mannwhitneyu(raw[b], raw[a], alternative="greater")
                adj[f"{b}>{a}"] = dict(gap=round(float(np.mean(raw[b]) - np.mean(raw[a])), 3),
                                       p=round(float(u.pvalue), 4))
        res["_adjacent_tests"] = adj
        print(f"\nTREND: Spearman rho(log-size, D_norm)={res['_trend']['spearman_rho']} "
              f"p={res['_trend']['spearman_p']}")
        for k, v in adj.items():
            print(f"  {k}: gap={v['gap']:+.3f} p={v['p']}")
    json.dump(res, open(f"{RESDIR}/ar_capacity.json", "w"), indent=1)
    print("AR CAPACITY DONE", flush=True)


if __name__ == "__main__":
    main()
