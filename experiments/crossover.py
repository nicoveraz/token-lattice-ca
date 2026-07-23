"""Crossover-relative probing: is the capacity plateau (F23/F24) real, or an artifact of
probing every model at a fixed (r,T)? Each model has its own heal->spread crossover T_c(r);
a fixed-T probe samples different parts of each phase diagram. We locate T_c(r=2) per model
via a T-sweep of the diversity-controlled D_norm, then ask whether T_c scales with capacity
(a properly-located capacity effect) or is flat/non-monotone (plateau real). Also reports the
slope dD_norm/dT at T_c (a susceptibility). Masked (mlm_ca) and AR (ar_ca) in one run.
Incremental save + MPS cache flush (survives kills). Usage: crossover.py
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
from mlm_lib import MODELS as MLM_MODELS, RESDIR, ensure_resdir

R = 2
OUT = None
# (backend, tag, hf_name, size_M, T-grid, seeds, B)
JOBS = [
    ("mlm", "bert-tiny", "prajjwal1/bert-tiny", 4, [0.3, 0.4, 0.5, 0.6, 0.7], [21, 22, 23], 24),
    ("mlm", "bert-mini", "prajjwal1/bert-mini", 11, [0.3, 0.4, 0.5, 0.6, 0.7], [21, 22, 23], 24),
    ("mlm", "bert-base", "bert-base-uncased", 110, [0.3, 0.4, 0.5, 0.6, 0.7], [21, 22, 23], 24),
    ("ar", "pythia-70m", "EleutherAI/pythia-70m", 70, [0.4, 0.5, 0.6, 0.7, 0.8], [21, 22], 20),
    ("ar", "pythia-160m", "EleutherAI/pythia-160m", 160, [0.4, 0.5, 0.6, 0.7, 0.8], [21, 22], 20),
    ("ar", "pythia-410m", "EleutherAI/pythia-410m", 410, [0.4, 0.5, 0.6, 0.7, 0.8], [21, 22], 16),
    ("ar", "pythia-1b", "EleutherAI/pythia-1b", 1000, [0.5, 0.6, 0.7], [21, 22], 12),
]


def cross(ts, ys, lv=0.5):
    ts, ys = np.asarray(ts, float), np.asarray(ys, float)
    for i in range(len(ts) - 1):
        a, b = ys[i], ys[i + 1]
        if (a - lv) * (b - lv) <= 0 and a != b:
            return float(ts[i] + (a - lv) / (a - b) * (ts[i + 1] - ts[i]))
    return None


def dnorm_at(backend, rule, T, sweeps, N, Bm, seed):
    if backend == "mlm":
        from mlm_damage import block_damage, drift_floor
        d = block_damage(rule, T, R, block=3, B=Bm, N=N, settle=12, sweeps=sweeps, seed=seed, scheme="cls_sep")
        d0, _ = drift_floor(rule, T, R, B=Bm, N=N, settle=12, sweeps=sweeps, seed=seed, scheme="cls_sep")
    else:
        from ar_probe import block_damage, drift_floor
        d = block_damage(rule, T, R, block=3, B=Bm, N=N, settle=12, sweeps=sweeps, seed=seed)
        d0, _ = drift_floor(rule, T, R, B=Bm, N=N, settle=12, sweeps=sweeps, seed=seed)
    return d["mean_damage"] / max(d0, 1e-3)


def save(res):
    out = {tag: dict(res[tag]) for tag in res}
    json.dump(out, open(OUT, "w"), indent=1)


def main():
    global OUT
    ensure_resdir()
    OUT = f"{RESDIR}/crossover.json"
    res = json.load(open(OUT)) if os.path.exists(OUT) else {}
    for backend, tag, name, sizeM, TS, SEEDS, B in JOBS:
        if tag in res and "T_c" in res[tag]:
            print(f"[{tag}] SKIP (done)", flush=True); continue
        if backend == "mlm":
            from mlm_ca import MLMRule
            rule = MLMRule(name)
        else:
            from ar_ca import ARRule
            rule = ARRule(name)
        prof = {}
        for T in TS:
            vals = [dnorm_at(backend, rule, T, 26, 48, B, sd) for sd in SEEDS]
            prof[T] = round(float(np.mean(vals)), 4)
            print(f"[{tag}] T={T}: D_norm(r=2)={prof[T]:.3f}", flush=True)
        Tc = cross(TS, [prof[T] for T in TS], 0.5)
        slope = float(np.polyfit(TS, [prof[T] for T in TS], 1)[0])
        res[tag] = dict(backend=backend, size_M=sizeM, r=R, profile=prof,
                        T_c=round(Tc, 4) if Tc else None, slope=round(slope, 4))
        print(f"[{tag}] T_c(r=2)={res[tag]['T_c']}  slope={res[tag]['slope']:+.3f}", flush=True)
        save(res)
        rule.model = None; del rule; gc.collect()
        try:
            torch.mps.empty_cache()
        except Exception:
            pass
    # capacity trend of T_c, per backend
    for be in ["mlm", "ar"]:
        pts = [(np.log10(res[t]["size_M"]), res[t]["T_c"]) for t in res
               if res[t]["backend"] == be and res[t].get("T_c")]
        if len(pts) >= 3:
            x, y = zip(*pts)
            sp = stats.spearmanr(x, y)
            res[f"_trend_{be}"] = dict(spearman_rho=round(float(sp.correlation), 3),
                                       spearman_p=round(float(sp.pvalue), 4), n=len(pts),
                                       note="rho<0 => T_c falls with capacity (bigger=more sensitive)")
            print(f"[{be}] T_c vs log-size: Spearman rho={sp.correlation:.3f} p={sp.pvalue:.4f}")
    save(res)
    print("CROSSOVER DONE", flush=True)


if __name__ == "__main__":
    main()
