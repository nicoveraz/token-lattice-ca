"""Freeze the newline-margin predictions BEFORE the fill census exists. 39 forward passes.

WHAT THIS IS. F162 found that phi-raising prefixes install '\\n' as a fixed point. The bilinear
hypothesis then has two directly measurable vectors and needs no internals: u_prefix is the prefix's
newline affordance, v_model is the model's newline self-continuation margin. This measures the joint
quantity per (model, prefix) cell:

    margin(model, prefix) = logit('\\n' | prefix, '\\n', '\\n') - max_{t != '\\n'} logit(t | ...)

margin > 0 means '\\n' IS a fixed point of the argmax map under that prefix, because the diagonal
state ('\\n','\\n') then reproduces itself. That is NECESSARY for a newline funnel and NOT SUFFICIENT:
trajectories must also REACH ('\\n','\\n'). The primary test is therefore a necessary-condition test.
It can fail cleanly; it cannot alone establish sufficiency. Said here so it is not claimed later.

WHY THE ORDER MATTERS MORE THAN THE MEASUREMENT. The fill census -- three models on twelve texts --
was chosen by F164's coverage analysis BEFORE F162 existed, so the newline factor faces a widening it
had no hand in picking. That is only true if the predictions are frozen first. This script writes
per-cell predicted directions derived from the margins ALONE and sha256-hashes them; the census is
launched only afterwards. A prediction written after the cells exist would be unfalsifiable, and the
hash is what makes the ordering checkable by someone who was not here.

The logit margin is the right scale-free quantity: the argmax map cares only about the ORDER of
logits, so a softmax probability would import a temperature the estimator does not have.
"""
import sys as _sys, pathlib as _pathlib
_ROOT = _pathlib.Path(__file__).resolve().parents[1]
_sys.path[:0] = [str(_ROOT / "src"), str(_ROOT / "experiments"), str(_ROOT / "fingerprint"),
                 str(_ROOT / "gatecheck" / "src")]
import gc, hashlib, json, os, time
os.environ.setdefault("HF_HOME", str(_ROOT / "hf_cache"))
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("HF_HUB_OFFLINE", "1")

import numpy as np
import torch

from provenance import stamp, rel
from structural_text import pick_rows                      # not used for arms; kept for parity
from text_interaction import texts as ti_texts, LENGTH     # the IDENTICAL twelve prefixes

OUT = str(_ROOT / "results" / "newline_margin_frozen.json")
PREREG = _ROOT / "experiments" / "prereg_newline_margin.json"

MODELS = ["EleutherAI/pythia-410m", "bigcode/starcoder2-3b", "llm-jp/llm-jp-3-1.8b"]
CENSUS_SEEDS = [20260803, 990017]


def newline_id(tok):
    """The model's own '\\n'. Lowest id whose decoded form is exactly a newline."""
    cands = []
    for s in ("\n", "Ċ", "<0x0A>"):
        try:
            ids = tok(s, add_special_tokens=False)["input_ids"]
        except Exception:
            continue
        for i in ids:
            if tok.decode([int(i)]) == "\n":
                cands.append(int(i))
    if not cands:
        V = int(getattr(tok, "vocab_size", 0)) or len(tok)
        for i in range(min(V, 300)):
            if tok.decode([i]) == "\n":
                cands.append(i)
                break
    return (min(cands) if cands else None)


def margin(model, dev, prefix_ids, nl):
    """logit(nl | prefix, nl, nl) - max logit(other). One forward pass."""
    ids = torch.tensor([list(prefix_ids) + [nl, nl]], device=dev)
    with torch.no_grad():
        lg = model(input_ids=ids).logits[0, -1].float()
    top = lg[nl].item()
    lg2 = lg.clone()
    lg2[nl] = -float("inf")
    rival = lg2.max().item()
    return float(top - rival), int(torch.argmax(lg).item())


def main():
    res = {"_preregistration_file": "experiments/prereg_newline_margin.json",
           "_order_note": "margins and predictions are frozen and hashed here; the fill census is "
                          "launched only after this file exists"}
    from transformers import AutoTokenizer, AutoModelForCausalLM
    dev = "cpu"                      # margins are 39 short passes; CPU keeps this reproducible
    cells, meta = {}, {}
    for m in MODELS:
        t0 = time.time()
        tok = AutoTokenizer.from_pretrained(m)
        nl = newline_id(tok)
        if nl is None:
            meta[m] = dict(error="no exact-newline token id found")
            continue
        model = AutoModelForCausalLM.from_pretrained(m).eval().to(dev, torch.float32)
        arms = ti_texts(tok)                         # the identical twelve, offset-selected
        mr, mtok = margin(model, dev, [], nl)
        cells[f"{m}||raw"] = dict(model=m, arm="raw", margin=round(mr, 4),
                                  argmax_is_newline=bool(mtok == nl), newline_id=nl)
        for a in sorted(arms):
            v, at = margin(model, dev, arms[a], nl)
            cells[f"{m}||{a}"] = dict(model=m, arm=a, margin=round(v, 4),
                                      argmax_is_newline=bool(at == nl), newline_id=nl,
                                      n_prefix_tokens=len(arms[a]))
        meta[m] = dict(newline_id=nl, newline_decoded=repr(tok.decode([nl])),
                       n_arms=len(arms), secs=round(time.time() - t0, 1))
        print(f"  {m:<28} nl={nl:<7} raw margin {mr:+.3f}  "
              f"arms flipped positive: "
              f"{sum(1 for a in arms if cells[f'{m}||{a}']['margin'] > 0)}/{len(arms)}", flush=True)
        del model
        gc.collect()
    res["margins"] = cells
    res["meta"] = meta

    # PREDICTIONS, derived from the margins ALONE -- no census cell exists yet.
    preds = {}
    for k, v in cells.items():
        if v["arm"] == "raw":
            continue
        m = v["model"]
        raw = cells.get(f"{m}||raw")
        if raw is None:
            continue
        flipped = bool(raw["margin"] <= 0 < v["margin"])
        preds[k] = dict(
            model=m, arm=v["arm"], margin_raw=raw["margin"], margin_arm=v["margin"],
            margin_flipped_positive=flipped,
            predicted_direction=("up" if flipped else "not_up"),
            basis="H1: a RAISING cell must have flipped positive. 'not_up' is a necessary-condition "
                  "prediction only -- it forbids raising, it does not predict falling (H3).")
    res["predictions"] = preds
    n_flip = sum(1 for p in preds.values() if p["margin_flipped_positive"])
    res["prediction_summary"] = dict(n_cells=len(preds), n_predicted_up=n_flip,
                                     n_predicted_not_up=len(preds) - n_flip)

    payload = json.dumps({"margins": cells, "predictions": preds}, sort_keys=True).encode()
    res["frozen_sha256"] = hashlib.sha256(payload).hexdigest()
    res["_analysis_provenance"] = stamp(__file__)
    json.dump(res, open(OUT, "w"), indent=1)

    # stamp the hash into the prereg so the ordering is checkable from either file
    pr = json.load(open(PREREG))
    pr["frozen_predictions_sha256"] = res["frozen_sha256"]
    pr["frozen_predictions_file"] = "results/newline_margin_frozen.json"
    pr["frozen_at"] = "before the fill census; see results/newline_margin_frozen.json _order_note"
    json.dump(pr, open(PREREG, "w"), indent=1)

    print(f"\n  {len(preds)} cell predictions frozen; {n_flip} predict UP, "
          f"{len(preds) - n_flip} predict NOT-UP")
    print(f"  sha256 {res['frozen_sha256'][:32]}...")
    print("\nwrote", rel(OUT), "and stamped the hash into", rel(str(PREREG)))


if __name__ == "__main__":
    main()
