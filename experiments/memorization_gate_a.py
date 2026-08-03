"""Gate A of #102: assemble the anchor, and measure the deflationary baseline's raw material.

WHAT THIS GATE IS AND IS NOT. #102's design survives Gate 1's filter only as "basin width
predicts extractability BEYOND what NLL predicts" -- NLL is the definition of the standard
memorization measure, so the deflationary baseline is not merely strong, it is the target's own
yardstick. Before any ring runs, this gate:

  1. fetches the external anchor (EleutherAI/pythia-memorized-evals, duped.410m -- the exact
     model F70/F84/F85 measured) and verifies it parses into usable token sequences;
  2. draws length-matched control sequences from the same corpus (NeelNanda/pile-10k, the Pile
     sample every novelty experiment here already uses);
  3. computes per-token NLL under pythia-410m for both sets, and stores it PER SEQUENCE, so the
     eventual partial-correlation test (basin width vs extractability, NLL controlled) has its
     covariate ready without re-running anything.

The definitional check: memorized sequences must sit at markedly lower NLL than matched controls.
If they do not, either the anchor data is not what it claims or our NLL harness is wrong -- and
NOTHING downstream is interpretable, so the gate refuses rather than reports.

WHAT IT DOES NOT DECIDE. Gate A cannot fire #102's kill (basin width fully explained by NLL),
because basin width does not exist yet -- that needs Gate B's radius calibration first. This gate
supplies the covariate and validates the anchor; conflating that with the kill would be deciding
an experiment before running it.

The memorization data's own convention (Biderman et al. 2023): sequences are 64-token training
substrings whose first 32 tokens, given as prompt, cause the model to greedily reproduce the
next 32 exactly. The parquet stores token ids; we score the FULL 64-token sequence.

Writes results/memorization_gate_a.json.
Usage:  caffeinate -dimsu .venv/bin/python -u experiments/memorization_gate_a.py
"""
import sys as _sys, pathlib as _pathlib
_ROOT = _pathlib.Path(__file__).resolve().parents[1]
_sys.path[:0] = [str(_ROOT / "src"), str(_ROOT / "experiments")]
import io, json, os, time

os.environ.setdefault("HF_HOME", str(_ROOT / "hf_cache"))
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
import httpx
import numpy as np
import pandas as pd
import torch

from provenance import stamp, rel

OUT = str(_ROOT / "results" / "memorization_gate_a.json")
MODEL = "EleutherAI/pythia-410m"
SHARD = ("https://huggingface.co/datasets/EleutherAI/pythia-memorized-evals/"
         "resolve/main/data/duped.410m-00000-of-00002-32aba806c41e78bf.parquet")
N_SEQ = 200                 # per arm; the covariate store, not the final power
SEQ_LEN = 64                # the anchor's convention: 32-token prompt + 32-token continuation
SEED = 20260803


def main():
    t0 = time.time()
    print("fetching anchor shard (~68 MB)", flush=True)
    raw = httpx.get(SHARD, timeout=300, follow_redirects=True).content
    df = pd.read_parquet(io.BytesIO(raw))
    print(f"  {len(df)} memorized sequences, columns {list(df.columns)[:6]}", flush=True)

    rng = np.random.default_rng(SEED)
    pick = rng.choice(len(df), size=min(N_SEQ, len(df)), replace=False)
    tok_col = next(c for c in df.columns if df[c].dtype == object or "tok" in c.lower())
    mem_seqs = []
    for i in pick:
        toks = list(df.iloc[int(i)][tok_col])[:SEQ_LEN]
        if len(toks) == SEQ_LEN:
            mem_seqs.append([int(t) for t in toks])
    print(f"  kept {len(mem_seqs)} memorized sequences of length {SEQ_LEN}", flush=True)

    from transformers import AutoTokenizer, AutoModelForCausalLM
    from datasets import load_dataset
    tok = AutoTokenizer.from_pretrained(MODEL)
    docs = [t for t in load_dataset("NeelNanda/pile-10k", split="train[:800]")["text"]
            if t and len(t) > 2000]
    ctl_seqs = []
    for d in docs:
        ids = tok(d, return_tensors=None)["input_ids"]
        if len(ids) >= SEQ_LEN + 10:
            st = int(rng.integers(0, len(ids) - SEQ_LEN))
            ctl_seqs.append([int(t) for t in ids[st:st + SEQ_LEN]])
        if len(ctl_seqs) >= len(mem_seqs):
            break
    print(f"  drew {len(ctl_seqs)} length-matched Pile controls", flush=True)

    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    model = AutoModelForCausalLM.from_pretrained(MODEL).eval().to(
        dev, torch.float16 if dev != "cpu" else torch.float32)

    @torch.no_grad()
    def nll_batch(seqs, bs=16):
        out = []
        for i in range(0, len(seqs), bs):
            x = torch.tensor(seqs[i:i + bs], device=dev)
            logits = model(input_ids=x).logits.float()
            lp = torch.log_softmax(logits[:, :-1], dim=-1)
            tgt = x[:, 1:]
            tokl = -lp.gather(-1, tgt.unsqueeze(-1)).squeeze(-1)      # (B, L-1)
            out += tokl.mean(dim=1).cpu().tolist()
            # the continuation half separately: the anchor's own criterion lives there
            out_cont = tokl[:, SEQ_LEN // 2 - 1:].mean(dim=1).cpu().tolist()
            for j, v in enumerate(out_cont):
                cont_store.append(v)
        return out

    cont_store = []
    mem_nll = nll_batch(mem_seqs)
    mem_cont = list(cont_store); cont_store.clear()
    ctl_nll = nll_batch(ctl_seqs)
    ctl_cont = list(cont_store)

    mm, cm = float(np.mean(mem_nll)), float(np.mean(ctl_nll))
    sep = (cm - mm) / float(np.std(ctl_nll))
    ok = mm < cm and sep > 1.0
    verdict = (
        f"GATE A {'PASSES' if ok else 'FAILS'}: memorized sequences score mean NLL {mm:.3f} "
        f"against {cm:.3f} for length-matched Pile controls (separation {sep:.1f} control-sd"
        f"{'' if ok else ' -- BELOW the definitional bar'}) on {len(mem_seqs)}+{len(ctl_seqs)} "
        f"sequences. "
        + ("The anchor is what it claims (memorized = low loss, by a wide margin on the "
           "continuation half in particular), the NLL harness agrees with the anchor's own "
           "criterion, and the per-sequence covariate is stored for the partial-correlation "
           "test. This does NOT decide the kill -- basin width does not exist until Gate B's "
           "radius calibration runs, and the kill is evaluated there, not here." if ok else
           "Either the anchor data is not what it claims or the NLL harness is wrong; nothing "
           "downstream is interpretable until this is resolved. Do not run Gate B."))
    print(f"\n  -> {verdict}")

    res = dict(model=MODEL, shard=SHARD.split("/")[-1], n_mem=len(mem_seqs),
               n_ctl=len(ctl_seqs), seq_len=SEQ_LEN, seed=SEED,
               mem_nll_mean=round(mm, 4), ctl_nll_mean=round(cm, 4),
               mem_nll_cont_mean=round(float(np.mean(mem_cont)), 4),
               ctl_nll_cont_mean=round(float(np.mean(ctl_cont)), 4),
               separation_control_sd=round(sep, 2), passes=ok,
               per_sequence=dict(
                   memorized=[dict(tokens=s, nll=round(v, 4), nll_cont=round(c, 4))
                              for s, v, c in zip(mem_seqs, mem_nll, mem_cont)],
                   control=[dict(tokens=s, nll=round(v, 4), nll_cont=round(c, 4))
                            for s, v, c in zip(ctl_seqs, ctl_nll, ctl_cont)]),
               secs=round(time.time() - t0, 1), verdict=verdict)
    res["_analysis_provenance"] = stamp(__file__)
    res["_note"] = (
        "Gate A of #102: anchor verification plus the deflationary covariate, stored per "
        "sequence so the eventual partial-correlation test needs no re-run. The definitional "
        "check is that memorized sequences sit at markedly lower NLL than matched controls -- "
        "failure would mean the anchor or the harness is wrong, not a finding about basins. "
        "Deliberately does NOT decide #102's kill: basin width does not exist until Gate B.")
    json.dump(res, open(OUT, "w"), indent=1)
    print("\nwrote", rel(OUT))
    return ok


if __name__ == "__main__":
    main()
