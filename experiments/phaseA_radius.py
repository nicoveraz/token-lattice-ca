"""Phase A1+A3: harden the F15 radius headline.

For one model, sweep radius r at a fixed ordered T, with >=5 seeds, under BOTH the
default cls_sep scheme and the no-special-tokens scheme. Per (scheme, r) record:
  order            bigram overlap vs WikiText proxy
  k3, k4           raw k-gram overlap (the F15 metric)
  k3c, k4c         REPEAT-COLLAPSED k-gram overlap (A3: repetition-robust)
  mi_len, mi_int   coarse-grained long-range MI decay length / integrated MI (A3)

A1 (scheme apparatus): the cls_sep vs none profiles quantify the apparatus shift to
compare against the cross-model shift. A3 (repetition): k?c and MI test whether the
intermediate-radius structure peak is real or repetition.
Usage: phaseA_radius.py --model tiny
"""
import sys as _sys, pathlib as _pathlib
_ROOT = _pathlib.Path(__file__).resolve().parents[1]
_sys.path[:0] = [str(_ROOT / "src"), str(_ROOT / "experiments")]
import argparse, json, time
import numpy as np
import mlm_ca
from mlm_ca import MLMRule, run
from mlm_lib import (MODELS, RESDIR, load_ref, ref_kgram_sets, kgram_overlap,
                     distinct_corpus_kgrams, order_param, coarse_mi_decay, ensure_resdir)

RS = [1, 2, 4, 8, 16]
SEEDS = [11, 22, 33, 44, 55]
T_ORDERED = 0.7
N, B, SWEEPS, TAIL = 48, 16, 40, 10


def profile(rule, ref_bi, ksets, scheme):
    """Return per-r dict of metric-lists (over seeds)."""
    keys = ("order", "k3", "k4", "k3c", "k4c", "k3d", "k4d", "distinct", "mi_len", "mi_int")
    prof = {r: {k: [] for k in keys} for r in RS}
    for r in RS:
        for sd in SEEDS:
            out = run(rule, B=B, N=N, r=r, T=T_ORDERED, sweeps=SWEEPS, scheme=scheme,
                      init="random", seed=sd)
            tail = out["snaps"][-TAIL:]                      # (TAIL, B, N)
            lat = out["final"]
            prof[r]["order"].append(order_param(lat, ref_bi)[0])
            ko = kgram_overlap(lat, ksets)                   # raw (F15 metric)
            koc = kgram_overlap(lat, ksets, collapse=True)   # immediate-repeat collapsed
            kod = distinct_corpus_kgrams(lat, ksets)         # distinct corpus k-grams (repetition-robust)
            prof[r]["k3"].append(ko[3]); prof[r]["k4"].append(ko[4])
            prof[r]["k3c"].append(koc[3]); prof[r]["k4c"].append(koc[4])
            prof[r]["k3d"].append(kod[3]); prof[r]["k4d"].append(kod[4])
            prof[r]["distinct"].append(kod["distinct_tok"])
            mi = coarse_mi_decay(tail.reshape(-1, N))
            prof[r]["mi_len"].append(mi["decay_length"]); prof[r]["mi_int"].append(mi["integrated"])
    return prof


def summarize(prof):
    out = {}
    for r in RS:
        out[str(r)] = {}
        for k, v in prof[r].items():
            out[str(r)][k] = round(float(np.mean(v)), 4)
            out[str(r)][k + "_std"] = round(float(np.std(v)), 4)
        out[str(r)]["order_seeds"] = [round(x, 4) for x in prof[r]["order"]]
    return out


def main(tag):
    ensure_resdir()
    rule = MLMRule(MODELS[tag])
    ref_bi = mlm_ca.ref_bigrams(load_ref())
    ksets = ref_kgram_sets(4)
    res = {"model": tag, "T": T_ORDERED, "N": N, "seeds": SEEDS}
    t0 = time.time()
    for scheme in ["cls_sep", "none"]:
        tc = time.time()
        res[scheme] = summarize(profile(rule, ref_bi, ksets, scheme))
        print(f"[{tag}/{scheme}] done ({time.time()-tc:.0f}s)")
        for r in RS:
            d = res[scheme][str(r)]
            print(f"  r={r:>2}: order={d['order']:.3f}±{d['order_std']:.3f}  "
                  f"k4={d['k4']:.3f} k4c={d['k4c']:.3f} k4d={d['k4d']:.3f}  "
                  f"distinct={d['distinct']:.2f}  mi_len={d['mi_len']:.1f}", flush=True)
    json.dump(res, open(f"{RESDIR}/phaseA_radius_{tag}.json", "w"), indent=1)
    print(f"[{tag}] PHASE-A RADIUS DONE ({time.time()-t0:.0f}s)", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=list(MODELS))
    a = ap.parse_args()
    main(a.model)
