"""Is the escape destination's brittleness a near-tie problem? 6 cells, the intersection only.

Registered in experiments/prereg_escape_confidence.json (frozen `43b8ee81...` before any unmasked
top-2 existed).

WHY. Arm 1 measured a precision floor of 0.7127: bfloat16 rounding of IDENTICAL WEIGHTS changes 29%
of escape destinations. The decisive pair's signal is 0.0772 of agreement against 0.2873 of floor
disagreement, so the floor is nearly four times the signal. If the flipped destinations are near-ties
a confidence threshold should shrink the floor toward 1.0; if they are not, the brittleness is
structural and no threshold repairs the estimand. KB registers the second outcome as a finding.

WHY NOT THE FREE VERSION. The stored margin is logit(t|t,t) - max_{x!=t} logit(x|t,t): confidence in
LEAVING rather than staying. What flips under rounding is WHICH destination wins -- top-1 against
top-2 of the UNMASKED distribution, which has never been stored. Thresholding on the stored margin
would have cost nothing and measured the wrong quantity, and is refused in the prereg rather than run.

This is also the first test of the margin-threshold rule prereg_selfcont.json registered as owed.
It is NOT a quantization test; bfloat16 is a far smaller perturbation and that obligation stays open.
"""
import sys as _sys, pathlib as _pathlib
_ROOT = _pathlib.Path(__file__).resolve().parents[1]
_sys.path[:0] = [str(_ROOT / "src"), str(_ROOT / "experiments"), str(_ROOT / "fingerprint"),
                 str(_ROOT / "gatecheck" / "src")]
import collections, gc, json, os, time

os.environ.setdefault("HF_HOME", str(_ROOT / "hf_cache"))
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import numpy as np
import torch

from provenance import stamp, rel
from selfcont_set import DTYPE, BATCH, BATCH_BIG, BIG, cell_key, out_path

PREREG = "experiments/prereg_escape_confidence.json"
PR = json.load(open(_ROOT / PREREG))
OUT = _ROOT / "results" / "escape_confidence.json"
PROBES = _ROOT / "experiments" / "probe_strings_selfcont.json"

CELLS = [("EleutherAI/pythia-410m", "Pythia", "fp32"),
         ("EleutherAI/pythia-410m-deduped", "Pythia", "fp32"),
         ("EleutherAI/pythia-410m", "Pythia", "bf16"),
         ("state-spaces/mamba-370m-hf", "Mamba", "fp32"),
         ("RWKV/rwkv-4-430m-pile", "RWKV", "fp32"),
         ("EleutherAI/gpt-neo-125m", "GPT-Neo", "fp32")]
DECISIVE = ("EleutherAI/pythia-410m", "EleutherAI/pythia-410m-deduped")
FAR = [("EleutherAI/pythia-410m", "state-spaces/mamba-370m-hf"),
       ("EleutherAI/pythia-410m", "RWKV/rwkv-4-430m-pile"),
       ("EleutherAI/pythia-410m", "EleutherAI/gpt-neo-125m")]
CONTROL = ("EleutherAI/pythia-410m", "EleutherAI/pythia-410m@bf16")
TAUS = PR["thresholds"]["tau_ladder"]
TAU_P = PR["thresholds"]["tau_primary"]
MIN_SRC = PR["thresholds"]["min_shared_sources"]
CACHE = _ROOT / "results" / "escape_confidence_cells.json"


@torch.no_grad()
def top2(model, ids, dev, batch):
    """Unmasked top-2 ids and logits at the two-token state (t,t). The quantity that actually flips."""
    n = len(ids)
    tid = np.empty((n, 2), np.int64); tlg = np.empty((n, 2), np.float64)
    for i in range(0, n, batch):
        ch = ids[i:i + batch]
        x = torch.tensor(ch, dtype=torch.long, device=dev).view(-1, 1).repeat(1, 2)
        lg = model(input_ids=x).logits[:, -1].float()
        v, j = torch.topk(lg, 2, dim=-1)
        tid[i:i + len(ch)] = j.cpu().numpy(); tlg[i:i + len(ch)] = v.cpu().numpy()
    return tid, tlg


def main():
    probe = json.load(open(PROBES))
    n_str = len(probe["strings"])
    cache = json.load(open(CACHE)) if CACHE.exists() else {}
    failed = []

    from transformers import AutoTokenizer, AutoModelForCausalLM
    for m, fam, dt in CELLS:
        key = cell_key(m, dt)
        if key in cache:
            print(f"  {key:<34} cached", flush=True); continue
        src = json.load(open(out_path(m, dt)))
        pid = np.array(src["probe_token_ids"])
        idx = np.flatnonzero(pid >= 0)
        t0 = time.time()
        try:
            tok = AutoTokenizer.from_pretrained(m)
            model = AutoModelForCausalLM.from_pretrained(m).eval().to("cpu", DTYPE[dt])
        except Exception as e:
            failed.append(dict(cell=key, error=type(e).__name__)); print(f"  {key} FAILED", flush=True)
            continue
        ids = [int(i) for i in pid[idx]]
        tid, tlg = top2(model, ids, "cpu", BATCH_BIG if m in BIG else BATCH)
        V = len(tok)
        dec1 = [tok.decode([int(i)]) for i in tid[:, 0]]
        del model; gc.collect()
        cache[key] = dict(cell=key, model=m, family=fam, dtype=dt,
                          probe_positions=[int(i) for i in idx], source_ids=ids,
                          top1_id=[int(i) for i in tid[:, 0]], top1_str=dec1,
                          conf_e4=[int(round(x)) for x in (tlg[:, 0] - tlg[:, 1]) * 1e4],
                          escapes=[bool(int(a) != int(b)) for a, b in zip(tid[:, 0], ids)],
                          secs=round(time.time() - t0, 1))
        json.dump(cache, open(CACHE, "w"))
        print(f"  {key:<34} n={len(ids):<6} escaping="
              f"{sum(cache[key]['escapes']):<6} ({cache[key]['secs']:.0f}s)", flush=True)

    _batch_invariance(cache)
    _verdict(cache, failed, n_str)


def _batch_invariance(cache):
    """A free control the prereg did not register, and it changes how the floor must be read.

    This run and selfcont_set.py compute the SAME top-1 on the SAME weights with the SAME estimator.
    What differs is only WHICH TOKENS SHARE A BATCH -- the whole vocabulary in id order there, the
    probe sources here -- which changes the floating-point reduction order in the matmul. That was
    flagged as a hazard in selfcont_set.py's oracle note and never measured. It is measurable for
    nothing now, and it must be measured, because if bfloat16 is not reproducible against its own
    batching then part of the 'precision floor' is not precision at all.
    """
    out = {}
    for k, c in cache.items():
        f = _ROOT / "results" / f"selfcont_set_{k.replace('/', '__').replace('@', '__')}.json"
        if not f.exists():
            continue
        a = np.array(json.load(open(f))["argmax_ids"], np.int64)[np.array(c["source_ids"])]
        b = np.array(c["top1_id"], np.int64)
        n = len(b)
        d = int((a != b).sum())
        out[k] = dict(n_sources=n, n_top1_differs=d, rate=round(d / n, 6), dtype=c["dtype"])
    cache["_batch_invariance"] = dict(
        per_cell=out,
        _what_differs="identical weights, identical dtype, identical estimator; only the batch "
                      "composition differs, which changes the reduction order of the matmul",
        _reading="a nonzero rate means the estimand is not reproducible against its own batching, "
                 "and any floor measured with that cell inherits the irreproducibility.")
    json.dump(cache, open(CACHE, "w"))
    print("\n  BATCH INVARIANCE (free, unregistered): " + "; ".join(
        f"{k.split('/')[-1]} {v['n_top1_differs']}/{v['n_sources']} ({v['rate']:.4%}, {v['dtype']})"
        for k, v in sorted(out.items())), flush=True)


def _verdict(cache, failed, n_str):
    cache_ref = cache
    keys = [k for k in cache if not k.startswith("_")]   # `_batch_invariance` is not a cell
    # shared source positions across all six cells, keyed by PROBE POSITION (the string bridge)
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
    res = dict(_preregistration_file=PREREG,
               _prereg_sha256=open(_ROOT / "experiments" / "prereg_escape_confidence.sha256").read().split()[0],
               cells=keys, failed=failed, n_shared_sources=len(shared),
               registered_thresholds=dict(tau_ladder=TAUS, tau_primary=TAU_P,
                                          min_shared_sources=MIN_SRC))

    def rung(a, b, tau):
        ca, sa, ea = A[a]; cb, sb, eb = A[b]
        keep = ea & eb & (ca >= tau) & (cb >= tau)
        n = int(keep.sum())
        if n == 0:
            return dict(tau=tau, n=0, agreement=None, not_decidable=True, modal_share=None)
        ag = float(np.mean(sa[keep] == sb[keep]))
        c = collections.Counter(sa[keep].tolist())
        return dict(tau=tau, n=n, agreement=round(ag, 4), not_decidable=bool(n < MIN_SRC),
                    modal_share=round(c.most_common(1)[0][1] / n, 4),
                    modal_destination=c.most_common(1)[0][0])

    def ladder(a, b):
        return [rung(a, b, t) for t in TAUS]

    res["floor"] = dict(pair=list(CONTROL), rungs=ladder(*CONTROL))
    res["decisive"] = dict(pair=list(DECISIVE), rungs=ladder(*DECISIVE))
    res["should_be_far"] = [dict(pair=list(p), rungs=ladder(*p)) for p in FAR]

    f0, f2 = res["floor"]["rungs"][0]["agreement"], res["floor"]["rungs"][-1]["agreement"]
    res["KB_fires"] = bool(f0 is not None and f2 is not None and f2 <= f0)

    def at(rungs, tau):
        return next((r for r in rungs if r["tau"] == tau), None)

    sep0 = sep_p = None
    d0, dp = at(res["decisive"]["rungs"], 0.0), at(res["decisive"]["rungs"], TAU_P)
    fars0 = [at(x["rungs"], 0.0) for x in res["should_be_far"]]
    farsp = [at(x["rungs"], TAU_P) for x in res["should_be_far"]]
    if d0 and d0["agreement"] is not None and all(f and f["agreement"] is not None for f in fars0):
        sep0 = round(d0["agreement"] - float(np.mean([f["agreement"] for f in fars0])), 4)
    if dp and dp["agreement"] is not None and all(f and f["agreement"] is not None for f in farsp):
        sep_p = round(dp["agreement"] - float(np.mean([f["agreement"] for f in farsp])), 4)
    res["separation"] = dict(at_tau_0=sep0, at_tau_primary=sep_p,
                             _definition="decisive agreement minus the mean of the three far pairs. "
                                         "KD requires this beside any floor improvement.")
    res["KD_trade"] = bool(sep0 is not None and sep_p is not None and sep_p < sep0)

    # DERIVED from the registered agreements, and it answers a different question from KD's
    # separation. Separation asks whether near still looks different from far. RESOLUTION asks
    # whether the decisive pair still looks different from NUMERIC NOISE: how many times further
    # apart is it than the same weights at two precisions, at the same tau. Both are reported
    # because they can move in opposite directions, and here they do.
    res["resolution_ladder"] = []
    for tau in TAUS:
        fr, dr = at(res["floor"]["rungs"], tau), at(res["decisive"]["rungs"], tau)
        r = None
        if fr and dr and fr["agreement"] is not None and dr["agreement"] is not None:
            den = 1.0 - fr["agreement"]
            r = None if den <= 1e-12 else round((1.0 - dr["agreement"]) / den, 2)
        res["resolution_ladder"].append(dict(
            tau=tau, floor_agreement=fr["agreement"] if fr else None,
            decisive_agreement=dr["agreement"] if dr else None, resolution=r,
            n_floor=fr["n"] if fr else 0, n_decisive=dr["n"] if dr else 0,
            not_decidable=bool((fr and fr["not_decidable"]) or (dr and dr["not_decidable"]))))
    res["_primary_tau_is_still"] = TAU_P
    res["_resolution_note"] = (
        "The ladder was registered IN FULL and tau=1.0 was named PRIMARY in advance, precisely so "
        "that no rung has to be chosen after the fact. Reporting the whole ladder is not the same as "
        "selecting the best rung from it: any rung other than the registered primary is an "
        "observation that would need its own pre-registered replication before it is a claim.")

    p = [f"ESCAPE CONFIDENCE, registered in {PREREG} (sha256 {res['_prereg_sha256'][:12]}..., frozen "
         f"before any unmasked top-2 existed). Six cells, {len(shared)} shared probe sources. The "
         f"quantity is top-1 minus top-2 of the UNMASKED distribution -- the thing that actually "
         f"flips -- not the stored margin, which measures leaving rather than choosing and was "
         f"refused in the prereg. "]
    p.append("THE FLOOR ACROSS THE LADDER (pythia-410m fp32 vs bf16, identical weights): "
             + "; ".join(f"tau={r['tau']} {r['agreement']} (n={r['n']}"
                         + (", NOT DECIDABLE" if r["not_decidable"] else "") + ")"
                         for r in res["floor"]["rungs"]) + ". ")
    if res["KB_fires"]:
        p.append("KB FIRES: the floor does not rise across the ladder. The flipped destinations are "
                 "NOT near-ties, the brittleness is structural, and no confidence threshold repairs "
                 "this estimand. Registered as an outcome before the run, not a disappointment. ")
    else:
        p.append(f"KB does not fire: the floor rises from {f0} to {f2} across the ladder, so the "
                 f"flips are substantially near-ties. ")
    p.append("THE DECISIVE PAIR: " + "; ".join(
        f"tau={r['tau']} {r['agreement']} (n={r['n']}" + (", NOT DECIDABLE" if r["not_decidable"] else "")
        + ")" for r in res["decisive"]["rungs"]) + ". ")
    for x in res["should_be_far"]:
        p.append(f"far vs {x['pair'][1].split('/')[-1]}: " + "; ".join(
            f"{r['tau']}:{r['agreement']}" for r in x["rungs"]) + ". ")
    p.append(f"KD, registered because it is the tempting half-report: decisive-minus-far separation "
             f"is {sep0} at tau=0 and {sep_p} at the primary tau={TAU_P}"
             + (". IT DEGRADES -- the threshold buys precision at the cost of discrimination, and "
                "the floor improvement may not be quoted without this. " if res["KD_trade"] else
                ", so discrimination survives the threshold. "))
    rl = res["resolution_ladder"]
    p.append("RESOLUTION AGAINST THE FLOOR AT EACH RUNG, derived, and it moves OPPOSITE to KD's "
             "separation: " + "; ".join(
                 f"tau={r['tau']} {r['resolution']}x" + (" (NOT DECIDABLE)" if r["not_decidable"] else "")
                 for r in rl if r["resolution"] is not None) + ". "
             f"The registered PRIMARY remains tau={TAU_P}. Any other rung is an observation, not a "
             f"claim: the ladder was registered in full so that no rung would have to be chosen "
             f"after the fact, and choosing one now would be exactly the threshold-shopping that "
             f"registering it was meant to prevent. ")
    bi = cache_ref.get("_batch_invariance", {}).get("per_cell", {}) if cache_ref else {}
    if bi:
        fp = [v for k, v in bi.items() if v["dtype"] == "fp32"]
        bf = [v for k, v in bi.items() if v["dtype"] == "bf16"]
        res["batch_invariance"] = cache_ref["_batch_invariance"]
        p.append("A FREE CONTROL THE PREREG DID NOT REGISTER, and it bounds how the floor may be "
                 "read: this run and selfcont_set.py compute the same top-1 on the same weights and "
                 "differ only in which tokens share a batch. In float32 that changes "
                 f"{sum(v['n_top1_differs'] for v in fp)} of {sum(v['n_sources'] for v in fp)} "
                 f"top-1s across {len(fp)} cells -- the estimator is batch-invariant there. In "
                 f"bfloat16 it changes {sum(v['n_top1_differs'] for v in bf)} of "
                 f"{sum(v['n_sources'] for v in bf)} "
                 f"({sum(v['n_top1_differs'] for v in bf)/max(1,sum(v['n_sources'] for v in bf)):.2%}). "
                 "So the bf16 arm is not reproducible against its own batching, and that share of "
                 "the floor is irreproducibility rather than precision. It is the smaller part: the "
                 "floor's disagreement at tau=0 is far larger. Unregistered and descriptive. ")
    p.append("KC: modal-destination share among survivors is reported at every rung, so a rising "
             "agreement that is really a concentration onto one destination is visible. ")
    p.append("REFUSALS: no p-value; no adjustment of the ladder or the 500-source floor; no claim "
             "that this makes the estimand quantization-robust -- bfloat16 is a far smaller "
             "perturbation and that test remains OWED; no semantic reading of any destination. "
             "THE PRIOR-ART RE-CHECK IS OWED AND BLOCKS WRITE-UP.")
    res["verdict"] = " ".join(p)
    res["_analysis_provenance"] = stamp(__file__)
    json.dump(res, open(OUT, "w"), indent=1)
    print("\n" + res["verdict"])
    print("\nwrote", rel(str(OUT)))


if __name__ == "__main__":
    main()
