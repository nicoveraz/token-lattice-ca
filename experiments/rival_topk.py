"""ARM 2: the rival and the top-k AT self-continuing tokens -- the part of the discard that is not free.

Registered in experiments/prereg_escape_rival.json (frozen `cf1e02ff...` before any destination was
decoded or any rival measured).

WHY THIS RUN EXISTS AND ARM 1 DOES NOT. The estimator computes, for every token t, the margin
logit(t | t,t) - max_{x != t} logit(x | t,t). The MAXIMUM is kept and the ARGMAX of the masked tensor
is thrown away. Where the bit is 0 that argmax is the unmasked argmax, which selfcont_set.py already
stores -- so the rival is free there, and it is also the escape destination: the same quantity split
by the bit, verified on 546823 tokens with zero exceptions before the prereg was frozen. Where the
bit is 1 the rival is the runner-up and was never stored. That is these ~60k forward passes, and it
is precisely the set the paper is about: 146 tokens for pythia-410m, 20774 for gpt-neo-1.3B.

WHAT IS STORED, and why more than the rival. The rival alone answers half a question. The prereg also
registers Q3 -- is the attractor a narrow spike or a spike on a neighbourhood? -- so the top-8 ids and
logits go in, plus the full-vocabulary logsumexp so exact top-k PROBABILITIES are recoverable without
keeping 50k floats. Logits are stored as integers scaled by 1e-4 for the reason recorded in
selfcont_set.py: floats here would swamp the pool tests/test_findings_numbers.py traces against.

THE ESTIMATOR IS THE SAME ONE. newline_margin_freeze.margin is imported and used as the reference
oracle exactly as in selfcont_set.py, so a rival measured here is a rival of the same map whose bits
were measured there.

Usage:  caffeinate -dimsu .venv/bin/python -u experiments/rival_topk.py
        (resumable, one file per cell)
"""
import sys as _sys, pathlib as _pathlib
_ROOT = _pathlib.Path(__file__).resolve().parents[1]
_sys.path[:0] = [str(_ROOT / "src"), str(_ROOT / "experiments"), str(_ROOT / "fingerprint"),
                 str(_ROOT / "gatecheck" / "src")]
import gc, json, os, time

os.environ.setdefault("HF_HOME", str(_ROOT / "hf_cache"))
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("HF_HUB_OFFLINE", "1")

import numpy as np
import torch

from provenance import stamp, rel
from newline_margin_freeze import margin as oracle_margin      # imported, never reimplemented
from selfcont_set import COHORT, DTYPE, BATCH, BATCH_BIG, BIG, cell_key, out_path

PREREG = "experiments/prereg_escape_rival.json"
RESULTS = _ROOT / "results"
TOPK = 8
N_ORACLE = 32


def rival_path(cell):
    return RESULTS / f"rival_topk_{cell.replace('/', '__').replace('@', '__')}.json"


@torch.no_grad()
def topk_batched(model, ids, dev, batch, k):
    """top-k ids and logits from the two-token state (t,t), plus the full-vocab logsumexp.

    Same forward pass as selfcont_set.batched; what differs is only what is kept from the logits.
    """
    n = len(ids)
    tid = np.empty((n, k), np.int64)
    tlg = np.empty((n, k), np.float64)
    lse = np.empty(n, np.float64)
    for i in range(0, n, batch):
        chunk = ids[i:i + batch]
        x = torch.tensor(chunk, dtype=torch.long, device=dev).view(-1, 1).repeat(1, 2)
        lg = model(input_ids=x).logits[:, -1].float()
        v, j = torch.topk(lg, k, dim=-1)
        tid[i:i + len(chunk)] = j.cpu().numpy()
        tlg[i:i + len(chunk)] = v.cpu().numpy()
        lse[i:i + len(chunk)] = torch.logsumexp(lg, dim=-1).cpu().numpy()
    return tid, tlg, lse


def measure(m, fam, dt):
    from transformers import AutoTokenizer, AutoModelForCausalLM
    cell = cell_key(m, dt)
    src = json.load(open(out_path(m, dt)))
    sc = [int(i) for i in src["self_continuing_ids"]]
    if not sc:
        return dict(cell=cell, model=m, family=fam, dtype=dt, n_self_continuing=0,
                    _note="no self-continuing tokens; nothing to measure")

    dev, t0 = "cpu", time.time()
    tok = AutoTokenizer.from_pretrained(m)
    model = AutoModelForCausalLM.from_pretrained(m).eval().to(dev, DTYPE[dt])
    batch = BATCH_BIG if m in BIG else BATCH

    tid, tlg, lse = topk_batched(model, sc, dev, batch, TOPK)

    # The top-1 MUST be the token itself: these are the tokens whose bit is 1, measured by the same
    # estimator. If it is not, the two runs disagree about the same map and nothing downstream is
    # comparable -- so this raises rather than being recorded as a discrepancy.
    bad = int(np.sum(tid[:, 0] != np.array(sc)))
    if bad:
        raise AssertionError(
            f"{cell}: {bad} of {len(sc)} self-continuing tokens do not have themselves as top-1 in "
            f"this run. selfcont_set.py and rival_topk.py disagree about the same argmax map; the "
            f"rival is not a rival of the bit that selected it.")

    # THE ORACLE, as in selfcont_set.py: the imported one-at-a-time estimator must agree that the
    # margin here is logit(t) - logit(rival). This checks the RIVAL, not just the margin, which is
    # the whole point of the run.
    o = sorted({sc[int(k)] for k in np.linspace(0, len(sc) - 1, min(N_ORACLE, len(sc))).astype(int)})
    gaps = []
    for t in o:
        om, _ = oracle_margin(model, dev, [], int(t))
        r = int(np.flatnonzero(np.array(sc) == t)[0])
        gaps.append(abs(om - (tlg[r, 0] - tlg[r, 1])))

    dec = [tok.decode([int(i)]) for i in tid[:, 1]]          # the rival's STRING (F166 inversion)
    own = [tok.decode([int(i)]) for i in sc]
    del model
    gc.collect()

    e4 = np.round(tlg * 1e4).astype(np.int64)
    return dict(
        cell=cell, model=m, family=fam, dtype=dt,
        _preregistration_file=PREREG,
        _prereg_sha256=open(_ROOT / "experiments" / "prereg_escape_rival.sha256").read().split()[0],
        _estimator="newline_margin_freeze.margin(prefix=[], nl=t) imported as the oracle; the "
                   "production path is the same forward pass as selfcont_set.batched, keeping the "
                   f"top-{TOPK} instead of the maximum",
        _source_cell=str(rel(str(out_path(m, dt)))),
        _logit_scale="1e-4, stored as integers; see selfcont_set.py for why floats are not stored",
        _oracle_check=dict(n_tokens=len(o), max_abs_margin_gap=round(float(max(gaps)), 8),
                           note="the imported estimator's margin against top1-minus-top2 here. This "
                                "checks the RIVAL, not only the margin."),
        n_self_continuing=len(sc), topk=TOPK,
        token_ids=sc, token_strings=own,
        topk_ids=tid.tolist(), topk_logits_e4=e4.tolist(),
        logsumexp_e4=[int(x) for x in np.round(lse * 1e4)],
        rival_ids=[int(i) for i in tid[:, 1]], rival_strings=dec,
        secs=round(time.time() - t0, 1))


def main():
    RESULTS.mkdir(exist_ok=True)
    failed = []
    # cheapest first, so a failure late in the run does not cost the whole cohort
    order = sorted(COHORT, key=lambda c: len(json.load(open(out_path(c[0], c[2])))["self_continuing_ids"])
                   if out_path(c[0], c[2]).exists() else 0)
    for m, fam, dt in order:
        p = rival_path(cell_key(m, dt))
        if p.exists():
            print(f"  {cell_key(m, dt):<34} cached", flush=True)
            continue
        if not out_path(m, dt).exists():
            failed.append(dict(cell=cell_key(m, dt), error="source cell missing"))
            print(f"  {cell_key(m, dt):<34} SOURCE MISSING", flush=True)
            continue
        try:
            res = measure(m, fam, dt)
        except AssertionError:
            raise
        except Exception as e:
            failed.append(dict(cell=cell_key(m, dt), error=type(e).__name__, detail=str(e)[:200]))
            print(f"  {cell_key(m, dt):<34} FAILED {type(e).__name__}", flush=True)
            continue
        res["_analysis_provenance"] = stamp(__file__)
        json.dump(res, open(p, "w"), indent=1)
        print(f"  {res['cell']:<34} n={res['n_self_continuing']:>6}  "
              f"oracle gap {res['_oracle_check']['max_abs_margin_gap']:.2e}  "
              f"({res['secs']:.0f}s)", flush=True)
    if failed:
        json.dump(dict(failed=failed, _analysis_provenance=stamp(__file__)),
                  open(RESULTS / "rival_topk_failures.json", "w"), indent=1)
        print(f"\n  KH: {len(failed)} cell(s) unusable and NAMED: {[f['cell'] for f in failed]}",
              flush=True)
    print("\nwrote", rel(str(RESULTS)) + "/rival_topk_*.json")


if __name__ == "__main__":
    main()
