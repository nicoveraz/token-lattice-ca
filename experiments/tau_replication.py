"""Does the resolution ladder peak at tau=0.5 on models that did not produce the observation?

Registered in experiments/prereg_tau_replication.json (frozen `5dccdb37...` before any held-out cell
was measured). This is the only thing that can turn F185's tau=0.5 observation into a claim or retire
it, and KR1 makes failure a PERMANENT DEMOTION rather than a disappointment.

WHY IT EXISTS. F185's ladder was registered in full with tau=1.0 named PRIMARY in advance. At that
primary the decisive pair reached 1.12x -- indistinguishable from numeric noise. The ladder peaked at
tau=0.5 at 11.76x, and that rung was not promoted, because selecting the best rung after seeing the
ladder is exactly the threshold-shopping registering it was meant to prevent.

THE DESIGN'S LIMIT, REGISTERED RATHER THAN DISCOVERED. The held-out set contains no second corpus
manipulation -- the cohort has exactly one and it generated the observation. So the near pair here is
same-family-different-SCALE, and a pass licenses "the ladder is non-monotone on held-out models",
never "tau=0.5 resolves the deduped pair at 11.76x".
"""
import sys as _sys, pathlib as _pathlib
_ROOT = _pathlib.Path(__file__).resolve().parents[1]
_sys.path[:0] = [str(_ROOT / "src"), str(_ROOT / "experiments"), str(_ROOT / "gatecheck" / "src")]
import collections, gc, json, os, time

os.environ.setdefault("HF_HOME", str(_ROOT / "hf_cache"))
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import numpy as np
import torch

from provenance import stamp, rel
from selfcont_set import DTYPE, BATCH, BATCH_BIG, BIG, cell_key, out_path
from escape_confidence import top2                     # imported, not reimplemented

PREREG = "experiments/prereg_tau_replication.json"
PR = json.load(open(_ROOT / PREREG))
OUT = _ROOT / "results" / "tau_replication.json"
CACHE = _ROOT / "results" / "tau_replication_cells.json"
TAUS = PR["thresholds"]["tau_ladder"]
MIN_SRC = PR["thresholds"]["min_shared_sources"]
NEAR = ("EleutherAI/pythia-160m", "EleutherAI/pythia-1b")
FLOOR = ("EleutherAI/pythia-160m", "EleutherAI/pythia-160m@bf16")
FAR = [("EleutherAI/pythia-160m", "state-spaces/mamba-130m-hf"),
       ("EleutherAI/pythia-160m", "RWKV/rwkv-4-169m-pile"),
       ("EleutherAI/pythia-160m", "EleutherAI/gpt-neo-1.3B")]
CELLS = [("EleutherAI/pythia-160m", "fp32"), ("EleutherAI/pythia-160m", "bf16"),
         ("EleutherAI/pythia-1b", "fp32"), ("state-spaces/mamba-130m-hf", "fp32"),
         ("RWKV/rwkv-4-169m-pile", "fp32"), ("EleutherAI/gpt-neo-1.3B", "fp32")]


def main():
    from transformers import AutoTokenizer, AutoModelForCausalLM
    cache = json.load(open(CACHE)) if CACHE.exists() else {}
    for m, dt in CELLS:
        key = cell_key(m, dt)
        if key in cache:
            print(f"  {key:<34} cached", flush=True); continue
        p = out_path(m, dt)
        if not p.exists():
            # the bf16 floor cell for pythia-160m has no selfcont run; resolve probes directly
            tokp = AutoTokenizer.from_pretrained(m)
            base = json.load(open(out_path(m, "fp32")))
            pid = np.array(base["probe_token_ids"])
        else:
            pid = np.array(json.load(open(p))["probe_token_ids"])
        idx = np.flatnonzero(pid >= 0)
        ids = [int(i) for i in pid[idx]]
        t0 = time.time()
        tok = AutoTokenizer.from_pretrained(m)
        model = AutoModelForCausalLM.from_pretrained(m).eval().to("cpu", DTYPE[dt])
        tid, tlg = top2(model, ids, "cpu", BATCH_BIG if m in BIG else BATCH)
        dec = [tok.decode([int(i)]) for i in tid[:, 0]]
        del model; gc.collect()
        cache[key] = dict(cell=key, model=m, dtype=dt, probe_positions=[int(i) for i in idx],
                          source_ids=ids, top1_id=[int(i) for i in tid[:, 0]], top1_str=dec,
                          conf_e4=[int(round(x)) for x in (tlg[:, 0] - tlg[:, 1]) * 1e4],
                          escapes=[bool(int(a) != int(b)) for a, b in zip(tid[:, 0], ids)],
                          secs=round(time.time() - t0, 1))
        json.dump(cache, open(CACHE, "w"))
        print(f"  {key:<34} n={len(ids):<6} escaping={sum(cache[key]['escapes']):<6} "
              f"({cache[key]['secs']:.0f}s)", flush=True)
    _verdict(cache)


def _verdict(cache):
    keys = [k for k in cache if not k.startswith("_")]
    shared = None
    for k in keys:
        s = set(cache[k]["probe_positions"])
        shared = s if shared is None else (shared & s)
    shared = sorted(shared)
    pos = {k: {p: i for i, p in enumerate(cache[k]["probe_positions"])} for k in keys}

    def arrays(k):
        i = [pos[k][p] for p in shared]
        return (np.array(cache[k]["conf_e4"], np.float64)[i] / 1e4,
                np.array([cache[k]["top1_str"][j] for j in i], dtype=object),
                np.array(cache[k]["escapes"], bool)[i])
    A = {k: arrays(k) for k in keys}

    def rung(a, b, tau):
        ca, sa, ea = A[a]; cb, sb, eb = A[b]
        keep = ea & eb & (ca >= tau) & (cb >= tau)
        n = int(keep.sum())
        if n == 0:
            return dict(tau=tau, n=0, agreement=None, not_decidable=True)
        return dict(tau=tau, n=n, agreement=round(float(np.mean(sa[keep] == sb[keep])), 4),
                    not_decidable=bool(n < MIN_SRC))

    res = dict(_preregistration_file=PREREG,
               _prereg_sha256=open(_ROOT / "experiments" / "prereg_tau_replication.sha256").read().split()[0],
               cells=keys, n_shared_sources=len(shared),
               registered_prediction=PR["the_prediction_registered_before_the_data"],
               design_limit=PR["A_LIMIT_OF_THIS_DESIGN_REGISTERED_NOW"])
    res["floor"] = [rung(*FLOOR, t) for t in TAUS]
    res["near"] = [rung(*NEAR, t) for t in TAUS]
    res["far"] = [dict(pair=list(p), rungs=[rung(*p, t) for t in TAUS]) for p in FAR]

    ladder = []
    for i, tau in enumerate(TAUS):
        f, nr = res["floor"][i], res["near"][i]
        r = None
        if f["agreement"] is not None and nr["agreement"] is not None:
            den = 1.0 - f["agreement"]
            r = None if den <= 1e-12 else round((1.0 - nr["agreement"]) / den, 2)
        ladder.append(dict(tau=tau, floor=f["agreement"], near=nr["agreement"], resolution=r,
                           n=nr["n"], not_decidable=bool(f["not_decidable"] or nr["not_decidable"])))
    res["resolution_ladder"] = ladder

    def at(tau):
        return next(x for x in ladder if x["tau"] == tau)
    h, z, o = at(0.5), at(0.0), at(1.0)
    f0 = res["floor"][0]["agreement"]
    if h["not_decidable"] or h["resolution"] is None:
        res["verdict_kind"] = "NOT DECIDABLE"
        res["KR1_permanent_demotion"] = None
    elif f0 is not None and f0 >= 1.0:
        res["verdict_kind"] = "NOT DECIDABLE"; res["KR1_permanent_demotion"] = None
        res["KR3_fires"] = True
    else:
        passed = (z["resolution"] is not None and o["resolution"] is not None
                  and h["resolution"] > z["resolution"] and h["resolution"] > o["resolution"])
        res["verdict_kind"] = "REPLICATED" if passed else "FAILED"
        res["KR1_permanent_demotion"] = bool(not passed)

    p = [f"TAU=0.5 REPLICATION, registered in {PREREG} (sha256 {res['_prereg_sha256'][:12]}..., "
         f"frozen before any held-out cell was measured). Six cells, none of which produced the "
         f"observation; {len(shared)} shared probe sources. The prediction, written down in advance: "
         f"resolution peaks at tau=0.5, above both tau=0 and tau=1. "]
    p.append("THE HELD-OUT LADDER: " + "; ".join(
        f"tau={x['tau']} floor {x['floor']} near {x['near']} -> {x['resolution']}x (n={x['n']}"
        + (", NOT DECIDABLE" if x["not_decidable"] else "") + ")" for x in ladder) + ". ")
    if res["verdict_kind"] == "NOT DECIDABLE":
        p.append("NOT DECIDABLE: the tau=0.5 rung is too thin, or the floor is degenerate (KR3), so "
                 "there is nothing to resolve against. The observation stays an observation. ")
    elif res["verdict_kind"] == "REPLICATED":
        p.append(f"THE PREDICTION HOLDS on held-out models: {h['resolution']}x at tau=0.5 against "
                 f"{z['resolution']}x at tau=0 and {o['resolution']}x at tau=1. Per the registered "
                 f"design limit this licenses 'the resolution ladder is non-monotone with an "
                 f"interior peak on held-out models' -- and NEVER the deduped-pair number, because "
                 f"the near pair here is same-family-different-SCALE, not a corpus manipulation. ")
    else:
        p.append(f"KR1 FIRES -- THE TAU=0.5 RUNG IS PERMANENTLY DEMOTED. The prediction fails on "
                 f"held-out models: {h['resolution']}x at tau=0.5 against {z['resolution']}x at "
                 f"tau=0 and {o['resolution']}x at tau=1. F185's 11.76x may never be quoted as a "
                 f"claim, and F185 gains a note saying the replication failed. Registered as the "
                 f"outcome before the run, not discovered as a disappointment. ")
    p.append("REFUSALS: no p-value; no adjustment of the ladder, the prediction or the source floor; "
             "and explicitly NO promotion of any other rung if this one failed -- that would be the "
             "same threshold-shopping one level down, and the prediction named tau=0.5 alone.")
    res["verdict"] = " ".join(p)
    res["_analysis_provenance"] = stamp(__file__)
    json.dump(res, open(OUT, "w"), indent=1)
    print("\n" + res["verdict"]); print("\nwrote", rel(str(OUT)))


if __name__ == "__main__":
    main()
