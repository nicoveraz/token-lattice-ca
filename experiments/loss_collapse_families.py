"""#84 across families: does lambda_ca collapse against LOSS, in a unit that is actually comparable?

WHAT THIS ADDS TO #84. `loss_collapse.py` asked whether the four Pythia SIZES fall onto one curve
when lambda_ca is plotted against loss instead of against step, and returned NOT DECIDABLE (F88):
the across-size residual was 0.0254 at matched loss against 0.0243 at matched log-step, with a
combined seed floor of 0.0247 -- both alignments sat AT the floor and differed by less than it. That
was underpowered, not negative, and its own diagnosis was "the fix is more checkpoints per size, not
more sizes".

F98 opens a different and better-posed version. Two non-Pythia families now have measured lambda_ca
(OLMo-2-0425-1B and OLMo-1B-0724-hf, both endpoints reproducing Pythia's), and F98's binding
limitation was that TIMING cannot be compared across families because no public non-Pythia family
has a checkpoint inside Pythia's dip window of 0.034-1.07B TOKENS. Loss does not have that problem:
it is a property of the model, not of anyone's checkpoint schedule. If the transition is a function
of HOW GOOD the model is rather than HOW LONG it trained, the three families should land on one
curve in (loss, lambda_ca) even though their checkpoint grids cannot be aligned in tokens.

THE UNIT IS THE WHOLE PROBLEM, AND IT IS WHY loss_collapse.py's METRIC CANNOT BE REUSED.
Cross-entropy in NATS PER TOKEN is not comparable across tokenizers. A coarser tokenizer packs more
text into each token and scores a lower per-token loss for free, with no change in how well the
model models the language. The three families do not share a tokenizer:

    EleutherAI/pythia-410m    vocab  50277   GPTNeoXTokenizer
    allenai/OLMo-1B-0724-hf   vocab  50280   GPTNeoXTokenizer
    allenai/OLMo-2-0425-1B    vocab 100278   (its own)

`loss_collapse.py` was within-Pythia so it never had to care; here the comparison is exactly the one
that breaks. So this measures BITS PER UTF-8 BYTE:

    bpb = (sum of token NLLs in nats) / (ln 2 * number of UTF-8 bytes of the same raw text)

Bytes are a property of the text, not of the tokenizer, so bpb is directly comparable across
vocabularies. The SAME raw Pile documents are scored by every model, and the byte count is taken
from the raw text once, not from any model's tokenization.

PRE-REGISTERED:
  PRIMARY      Do the three families' (bpb, lambda_ca) curves collapse onto one? Measured as the
               across-family spread of lambda_ca at matched bpb, against the same lambda seed
               floor F88 used. Collapse if the spread is BELOW the floor.
  CONTRAST     The same spread at matched TOKEN COUNT, which F98 showed cannot align the families.
               Loss is only the better organising variable if it beats this.
  POWER, and it is registered because F88 died of its absence: the test is only readable if the
               two alignments differ by MORE than the seed floor. If they do not, this returns
               NOT DECIDABLE exactly as F88 did, and the honest conclusion is that the grid is
               still too coarse -- not that loss fails.
  UNIT GATE    bpb must be finite and in a sane range (0.4-2.5) for every cell, and the byte count
               must come from the raw text. A tokenizer-dependent number here would silently
               reproduce the confound this script exists to remove.
  COHORT       gatecheck.cohort guards the family x checkpoint grid: a checkpoint that fails to
               load makes the verdict NOT DECIDABLE rather than shrinking the comparison.
  KILL         Families do not collapse against bpb either -> lambda_ca is not a function of model
               quality in any unit available, and the transition's timing is not comparable across
               families by any route this project can reach.

Writes results/loss_collapse_families.json.
Usage:  caffeinate -dimsu .venv/bin/python -u experiments/loss_collapse_families.py
        (resumable per (model, revision))
"""
import sys as _sys, pathlib as _pathlib
_ROOT = _pathlib.Path(__file__).resolve().parents[1]
_sys.path[:0] = [str(_ROOT / "src"), str(_ROOT / "experiments"),
                 str(_ROOT / "gatecheck" / "src")]
import gc, json, os, time

os.environ.setdefault("HF_HOME", str(_ROOT / "hf_cache"))
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
import numpy as np
import torch

from provenance import stamp, rel
from lyapunov import run_ignited
from gatecheck import dynamic_range, carries_verdict
from gatecheck.cohort import cohort_complete

OUT = str(_ROOT / "results" / "loss_collapse_families.json")

# (family label, repo, [(revision, tokens_B)]) -- exactly the checkpoints whose lambda_ca is
# already measured, so no new lattice runs are needed and the pairing is one-to-one.
FAMILIES = [
    ("pythia-410m", "EleutherAI/pythia-410m",
     [("step128", 0.27), ("step256", 0.54), ("step512", 1.07),
      ("step1000", 2.10), ("step2000", 4.19), ("step4000", 8.39)],
     "results/dev_transition_scale.json"),
    ("olmo2-1b", "allenai/OLMo-2-0425-1B",
     [("stage1-step0-tokens0B", 0.0), ("stage1-step300-tokens1B", 1.0),
      ("stage1-step10000-tokens21B", 21.0), ("stage1-step20000-tokens42B", 42.0),
      ("stage1-step40000-tokens84B", 84.0)],
     "results/generality_olmo2.json"),
    ("olmo1-0724", "allenai/OLMo-1B-0724-hf",
     [("step0-tokens0B", 0.0), ("step1000-tokens2B", 2.0), ("step2000-tokens4B", 4.0),
      ("step4000-tokens8B", 8.0), ("step10000-tokens20B", 20.0)],
     "results/generality_olmo1_0724.json"),
]

N_DOCS = 64             # fixed Pile documents, the SAME raw text for every model
MAX_CHARS = 4000        # per document, so the byte count is bounded and identical across models
SEED = 20260806
BPB_RANGE = (0.4, 2.5)  # the unit gate: anything outside this is not a bits-per-byte number


def pile_slice():
    """The same raw text for every model. Bytes are counted from THIS, not from any tokenization."""
    from datasets import load_dataset
    docs = [d for d in load_dataset("NeelNanda/pile-10k", split="train[:2000]")["text"]
            if len(d) > MAX_CHARS]
    rng = np.random.default_rng(SEED)
    idx = rng.choice(len(docs), size=min(N_DOCS, len(docs)), replace=False)
    return [docs[i][:MAX_CHARS] for i in sorted(idx)]


def bits_per_byte(repo, revision, texts, dev):
    """Total token NLL in nats / (ln2 * total UTF-8 bytes of the raw text).

    The denominator is a property of the TEXT. That is the entire point: nats-per-token rewards a
    coarse tokenizer for packing more characters into each prediction, which is exactly the
    confound that would otherwise make a cross-family comparison meaningless.
    """
    from transformers import AutoTokenizer, AutoModelForCausalLM
    tok = AutoTokenizer.from_pretrained(repo)
    m = AutoModelForCausalLM.from_pretrained(repo, revision=revision).eval().to(
        dev, torch.float16 if dev != "cpu" else torch.float32)
    tot_nll, tot_bytes, tot_tok = 0.0, 0, 0
    with torch.no_grad():
        for t in texts:
            ids = tok(t, return_tensors="pt").input_ids.to(dev)
            if ids.shape[1] < 2:
                continue
            out = m(input_ids=ids)
            lg = out.logits[:, :-1].float()
            tgt = ids[:, 1:]
            nll = torch.nn.functional.cross_entropy(
                lg.reshape(-1, lg.shape[-1]), tgt.reshape(-1), reduction="sum")
            tot_nll += float(nll)
            tot_bytes += len(t.encode("utf-8"))
            tot_tok += int(tgt.numel())
    del m
    try: torch.mps.empty_cache()
    except Exception: pass
    gc.collect()
    return dict(bpb=round(tot_nll / (np.log(2) * tot_bytes), 5),
                nats_per_token=round(tot_nll / max(tot_tok, 1), 5),
                n_tokens=tot_tok, n_bytes=tot_bytes)


def lambda_for(results_path, revision):
    """lambda_ca for this checkpoint, F42 filter applied, from the run that already measured it."""
    try:
        d = json.load(open(_ROOT / results_path))
    except FileNotFoundError:
        return None
    vals = []
    for v in d.get("runs", {}).values():
        if not isinstance(v, dict) or "lambda_ca" not in v or not run_ignited(v):
            continue
        rev = v.get("revision") or (f"step{v['step']}" if "step" in v else None)
        if rev != revision:
            continue
        if v.get("N") not in (None, 48) or (v.get("size_m") not in (None, 410)):
            continue
        vals.append(v["lambda_ca"])
    if not vals:
        return None
    return dict(lambda_ca=round(float(np.mean(vals)), 5),
                lambda_sd=round(float(np.std(vals)), 5), n_seeds=len(vals))


def main():
    res = json.load(open(OUT)) if os.path.exists(OUT) else {"cells": {}}
    res["_preregistration"] = dict(
        families=[f[0] for f in FAMILIES], n_docs=N_DOCS, max_chars=MAX_CHARS, seed=SEED,
        metric="bits per UTF-8 byte on a FIXED Pile slice; the byte count comes from the raw text",
        why_bpb="nats-per-token is not comparable across tokenizers (pythia 50277, olmo2 100278) "
                "-- a coarser tokenizer scores lower per-token loss for free. loss_collapse.py was "
                "within-Pythia so it never had to care; here that is the whole comparison",
        primary="do the three families' (bpb, lambda_ca) curves collapse? across-family spread of "
                "lambda_ca at matched bpb, against the lambda seed floor",
        contrast="the same spread at matched TOKEN COUNT, which F98 showed cannot align families",
        power="registered because F88 died of its absence: readable only if the two alignments "
              "differ by MORE than the seed floor; otherwise NOT DECIDABLE, as F88 returned",
        unit_gate=f"bpb must be finite and within {BPB_RANGE} for every cell",
        cohort="a checkpoint that fails to load makes the verdict NOT DECIDABLE (gatecheck.cohort)",
        kill="no collapse against bpb either -> lambda_ca is not a function of model quality in "
             "any unit available, and cross-family timing is unreachable by this route",
        follows="#84 (F88 NOT DECIDABLE, within-Pythia) extended cross-family by F98")

    texts = pile_slice()
    nbytes = sum(len(t.encode("utf-8")) for t in texts)
    res["_slice"] = dict(n_docs=len(texts), n_bytes=nbytes,
                         note="identical raw text scored by every model; bytes from the text")
    print(f"Pile slice: {len(texts)} docs, {nbytes} UTF-8 bytes -- the same for every model",
          flush=True)

    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    for fam, repo, ckpts, lam_src in FAMILIES:
        for rev, tokB in ckpts:
            k = f"{fam}|{rev}"
            if k in res["cells"]:
                continue
            t0 = time.time()
            try:
                row = bits_per_byte(repo, rev, texts, dev)
            except Exception as e:
                print(f"  {k}: FAILED ({type(e).__name__}: {e})"[:150], flush=True)
                res["cells"][k] = dict(family=fam, revision=rev, tokens_B=tokB,
                                       failed=f"{type(e).__name__}: {e}"[:200])
                json.dump(res, open(OUT, "w"), indent=1)
                continue
            lam = lambda_for(lam_src, rev)
            row.update(family=fam, revision=rev, tokens_B=tokB,
                       secs=round(time.time() - t0, 1), **(lam or {}))
            res["cells"][k] = row
            print(f"  {k:44s} bpb={row['bpb']:.4f}  nats/tok={row['nats_per_token']:.4f}  "
                  f"lam={row.get('lambda_ca')}  ({row['secs']:.0f}s)", flush=True)
            json.dump(res, open(OUT, "w"), indent=1)

    analyse(res)
    json.dump(res, open(OUT, "w"), indent=1)
    print("\nwrote", rel(OUT))


def _spread_at_matched(curves, key):
    """Across-family spread of lambda_ca at matched `key`, by interpolation onto a common grid."""
    usable = {f: sorted([(c[key], c["lambda_ca"]) for c in cs if c.get("lambda_ca") is not None])
              for f, cs in curves.items()}
    usable = {f: v for f, v in usable.items() if len(v) >= 3}
    if len(usable) < 2:
        return None, 0
    lo = max(min(x for x, _ in v) for v in usable.values())
    hi = min(max(x for x, _ in v) for v in usable.values())
    if not (hi > lo):
        return None, 0                       # no overlapping range: the families cannot be compared
    grid = np.linspace(lo, hi, 12)
    stack = []
    for f, v in usable.items():
        xs = np.array([x for x, _ in v]); ys = np.array([y for _, y in v])
        o = np.argsort(xs)
        stack.append(np.interp(grid, xs[o], ys[o]))
    return float(np.mean(np.std(np.stack(stack), axis=0))), len(usable)


def analyse(res):
    cells = [c for c in res["cells"].values() if "bpb" in c]
    parts = []

    declared = [f"{f}|{r}" for f, _, cks, _ in FAMILIES for r, _ in cks]
    coh = cohort_complete(declared, [f"{c['family']}|{c['revision']}" for c in cells],
                          unit="checkpoint")
    parts.append(f"COHORT: {coh.reason}")

    bad = [f"{c['family']}|{c['revision']}={c['bpb']}" for c in cells
           if not (BPB_RANGE[0] <= c["bpb"] <= BPB_RANGE[1])]
    parts.append(
        f"UNIT GATE: bits-per-byte on the shared {res['_slice']['n_bytes']}-byte Pile slice, byte "
        f"count taken from the raw text so the number is tokenizer-independent. "
        + (f"ALL {len(cells)} cells inside {BPB_RANGE}." if not bad else
           f"OUTSIDE RANGE: {bad} -- these are not bits-per-byte numbers and nothing below is read."))

    print(f"\n  {'family':<12} {'checkpoint':<28} {'tokens':>8} {'bpb':>8} {'nats/tok':>9} {'lambda':>9}")
    curves = {}
    for c in sorted(cells, key=lambda c: (c["family"], c["tokens_B"])):
        lam = c.get("lambda_ca")
        print(f"  {c['family']:<12} {c['revision']:<28} {c['tokens_B']:>7.1f}B {c['bpb']:>8.4f} "
              f"{c['nats_per_token']:>9.4f} {lam if lam is None else f'{lam:+.4f}':>9}")
        curves.setdefault(c["family"], []).append(c)

    if bad or not coh.complete:
        res["analysis"] = dict(cohort=coh.block(), unit_gate_failures=bad)
        res["verdict"] = " ".join(parts) + " NOT DECIDABLE."
        res["_analysis_provenance"] = stamp(__file__)
        print(f"\n  -> {res['verdict']}")
        return

    s_bpb, n_bpb = _spread_at_matched(curves, "bpb")
    s_tok, n_tok = _spread_at_matched(curves, "tokens_B")
    floors = [c["lambda_sd"] for c in cells if c.get("lambda_sd")]
    floor = float(np.mean(floors)) / np.sqrt(8) if floors else None

    if s_bpb is None:
        parts.append(
            "NO OVERLAPPING bpb RANGE across families, so a matched-loss comparison does not "
            "exist on this grid. That is itself the finding: the families are not merely offset "
            "in tokens, they do not share a loss interval at the checkpoints anyone published.")
        decided = False
    else:
        better = s_bpb < (s_tok if s_tok is not None else np.inf)
        readable = floor is not None and abs(s_bpb - (s_tok or 0)) > floor
        parts.append(
            f"PRIMARY: across-family spread of lambda_ca at matched bits-per-byte is {s_bpb:.4f} "
            f"over {n_bpb} families"
            + (f", against {s_tok:.4f} at matched token count over {n_tok}." if s_tok is not None
               else ", with no token-matched comparison available.")
            + (f" Seed floor {floor:.4f}." if floor else ""))
        if not readable:
            parts.append(
                f"NOT DECIDABLE, and for exactly the reason F88 was: the two alignments differ by "
                f"{abs(s_bpb - (s_tok or 0)):.4f}, less than the {floor:.4f} seed floor, so this "
                f"grid cannot say which organising variable is better. Not a null about loss -- "
                f"the test is underpowered, and the fix is finer checkpoint spacing, which for the "
                f"non-Pythia families does not exist to be had.")
            decided = False
        elif better and s_bpb <= floor:
            parts.append(
                f"COLLAPSE: at matched model quality the families agree to within the seed floor, "
                f"so lambda_ca is a function of HOW GOOD the model is rather than how long it "
                f"trained. That is the cross-family statement F98 could not make in token units, "
                f"reached in a unit that is comparable across tokenizers.")
            decided = True
        else:
            parts.append(
                f"NO COLLAPSE: the families do not agree at matched bits-per-byte "
                f"({s_bpb:.4f} against a {floor:.4f} floor), so lambda_ca is not a function of "
                f"model quality in this unit either, and cross-family timing stays unreachable.")
            decided = True

    parts.append(
        "BOUNDARY: bits-per-byte removes the tokenizer confound, not the corpus one -- all three "
        "families are scored on Pile text, which is training distribution for Pythia and OLMo but "
        "not identically weighted for either. And architecture, data order and optimiser still "
        "differ across families simultaneously (F98's attribution note applies unchanged).")

    v = " ".join(parts)
    print(f"\n  -> {v}")
    res["analysis"] = dict(
        spread_at_matched_bpb=None if s_bpb is None else round(s_bpb, 5),
        spread_at_matched_tokens=None if s_tok is None else round(s_tok, 5),
        lambda_seed_floor=None if floor is None else round(floor, 5),
        n_families_bpb=n_bpb, n_families_tokens=n_tok, decided=decided, cohort=coh.block())
    res["verdict"] = v
    res["_analysis_provenance"] = stamp(__file__)
    res["_note"] = (
        "#84 extended across families, which F98 made possible and F98's own limitation made "
        "necessary: timing cannot be compared across families in TOKENS because no public "
        "non-Pythia family has a checkpoint inside Pythia's dip window, but loss is a property of "
        "the model rather than of a checkpoint schedule. Measured in bits per UTF-8 byte because "
        "nats-per-token is not comparable across tokenizers (pythia 50277 vs olmo2 100278) -- the "
        "byte count comes from the raw text, identical for every model. No new lattice runs: every "
        "lambda_ca here was already measured by the run that is cited for it.")


if __name__ == "__main__":
    main()
