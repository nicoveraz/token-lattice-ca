"""Does lambda_ca collapse against LOSS rather than against STEP? (#84)

WHY THIS AND NOT THE EXISTING BASELINE. loss_baseline.py (#72) already answered "is lambda_ca a
perplexity proxy" -- no, it overshoots where loss falls monotonically, and the loss elbow is
size-invariant where the lambda_ca crossing is not. It deliberately used WikiText and used only
the SHAPE of the curve, because WikiText is not Pythia's training distribution and the LEVEL of
those numbers is not comparable across corpora.

A collapse test needs the level. If lambda_ca is a function of how good the model is rather than
how long it trained, the four size-curves should fall onto ONE curve when plotted against loss --
and that requires a loss measured on the distribution the models were actually trained on. Hence
a fixed Pile slice.

WHY NOT THE PUBLISHED NUMBERS. Pythia's per-checkpoint loss is not obtainable. The eval JSONs at
github.com/EleutherAI/pythia/evals/pythia-v1 carry acc, acc_norm, likelihood_difference and
lambada_openai.ppl -- no loss -- and contain no 14m or 31m directories at all. Loss lives only in
the maintainers' self-described "(messy!)" W&B project with no export path. So it is recomputed
here, which also covers the sizes the published evals omit and evaluates on exactly the token
distribution driving the lattice.

THE STANDARD THIS IS HELD TO. arXiv:2507.02119 (Qiu et al., "supercollapse") sets the bar: a
collapse residual is measured against a per-model SEED NOISE FLOOR, not eyeballed. Anything less
reads as curve-fitting. We have eight seeds per (model, checkpoint) cell, so the floor is
measurable from our own data and PolyPythias is not needed:

    floor    = mean over cells of the standard error of lambda_ca across seeds
    residual = spread of lambda_ca about a single monotone fit in loss, across ALL models

    residual <= ~floor   -> the curves collapse; lambda_ca tracks capability, not schedule
    residual >> floor    -> no collapse; lambda_ca is not a function of loss alone

PRE-REGISTERED BEFORE RUNNING:
  * Primary: is the collapse residual within a factor of 2 of the seed noise floor?
  * Reported either way. A NO-COLLAPSE result is informative: it would mean lambda_ca carries
    information about training that loss does not, which strengthens rather than weakens the
    paper's position, and it is the outcome consistent with #72's finding that the loss elbow is
    size-invariant while the lambda_ca crossing is not.
  * The comparison against step is reported alongside, so "collapses against loss" is a claim
    about which x-axis is better, not an unanchored statement.
  * lambda statistics exclude unignited runs (F42), n stated.
  * FRAMED AS AN EXTENSION, NOT AN INVENTION: arXiv:2403.15796 (Du et al.) owns
    loss-reparameterisation collapse for TASK metrics. The contribution here would be extending
    it to a DYNAMICAL observable -- measured from the model's own generation dynamics rather
    than from benchmark outputs. Stated that way in any write-up.

NO NEW LATTICE RUNS. Every lambda_ca value comes from dev_transition_scale.json (and
dev_transition_width.json when present); this script only adds teacher-forced forward passes.

Writes results/loss_collapse_pile.json.
Usage:  caffeinate -i .venv/bin/python experiments/loss_collapse_pile.py
"""
import sys as _sys, pathlib as _pathlib
_ROOT = _pathlib.Path(__file__).resolve().parents[1]
_sys.path[:0] = [str(_ROOT / "src"), str(_ROOT / "experiments")]
import os, json, gc, time
os.environ.setdefault("HF_HOME", str(_ROOT / "hf_cache"))
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
import numpy as np
import torch
from scipy import stats

from provenance import stamp, rel
from lyapunov import run_ignited

GRID = {
    "EleutherAI/pythia-14m":  [128, 256, 512, 1000, 2000, 4000],
    "EleutherAI/pythia-31m":  [128, 256, 512, 1000, 2000, 4000],
    "EleutherAI/pythia-70m":  [128, 256, 512, 1000, 2000, 4000],
    "EleutherAI/pythia-160m": [128, 256, 512, 1000, 2000, 4000],
    "EleutherAI/pythia-410m": [128, 256, 512, 1000, 2000, 4000],
    "EleutherAI/pythia-1b":   [128, 256, 512, 1000, 2000, 4000],
}
PILE = ("NeelNanda/pile-10k", "train")     # a fixed, ungated 10k-document Pile sample
SEQ_LEN, N_SEQ = 1024, 96                  # 98304 tokens, same budget as the WikiText baseline
OUT = _ROOT / "results" / "loss_collapse_pile.json"
SOURCES = ["dev_transition_scale.json", "dev_transition_width.json"]


def eval_tokens(tok):
    """A FIXED held-out slice of the Pile sample. No shuffling, no sampling."""
    from datasets import load_dataset
    ds = load_dataset(PILE[0], split=PILE[1])
    text = "\n\n".join(t for t in ds["text"][:4000] if t and t.strip())
    ids = tok(text, return_tensors="pt").input_ids[0]
    need = SEQ_LEN * N_SEQ
    assert ids.numel() >= need, f"slice too short: {ids.numel()} < {need}"
    return ids[:need].view(N_SEQ, SEQ_LEN)


def loss_at(name, revision, batches, device):
    from transformers import AutoModelForCausalLM
    m = AutoModelForCausalLM.from_pretrained(name, revision=revision).to(device).eval()
    tot, ntok = 0.0, 0
    with torch.no_grad():
        for i in range(batches.shape[0]):
            x = batches[i:i + 1].to(device)
            lg = m(x).logits[:, :-1].float()
            tg = x[:, 1:]
            tot += float(torch.nn.functional.cross_entropy(
                lg.reshape(-1, lg.shape[-1]), tg.reshape(-1), reduction="sum"))
            ntok += int(tg.numel())
    del m
    try: torch.mps.empty_cache()
    except Exception: pass
    gc.collect()
    return tot / ntok


def lambda_cells():
    """(model, step) -> per-seed lambda_ca, deduplicated on (model, step, seed).

    Dedup is not optional: loss_baseline.py shipped with 32 rows that were the same run present
    in two files, which doubled every reported n until it was caught.
    """
    seen, cells = {}, {}
    for src in SOURCES:
        p = _ROOT / "results" / src
        if not p.exists():
            continue
        for v in json.load(open(p))["runs"].values():
            if not (isinstance(v, dict) and "lambda_ca" in v):
                continue
            k = (v["model"], v["step"], v["seed"])
            if k in seen:
                assert seen[k] == v["lambda_ca"], f"files disagree about {k}"
                continue
            seen[k] = v["lambda_ca"]
            if run_ignited(v):
                cells.setdefault((v["model"], v["step"]), []).append(v["lambda_ca"])
    return cells


def main():
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    res = json.load(open(OUT)) if OUT.exists() else {"loss": {}}
    res["_preregistration"] = dict(
        grid=GRID, pile_source=PILE, seq_len=SEQ_LEN, n_seq=N_SEQ,
        primary="is the collapse residual within a factor of 2 of the seed noise floor?",
        standard="Qiu et al. arXiv:2507.02119 -- residual measured against a seed noise floor",
        framing="extension of Du et al. arXiv:2403.15796 (task metrics) to a dynamical observable",
        no_collapse_is_informative=True)

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained("EleutherAI/pythia-160m")
    batches = eval_tokens(tok)
    print(f"Pile slice: {batches.shape[0]}x{batches.shape[1]} = {batches.numel()} tokens "
          f"from {PILE[0]}, device={device}\n", flush=True)

    todo = [(m, s) for m, ss in GRID.items() for s in ss]
    for k, (name, step) in enumerate(todo, 1):
        key = f"{name.split('-')[-1]}_step{step}"
        if key in res["loss"]:
            continue
        t0 = time.time()
        try:
            L = loss_at(name, f"step{step}", batches, device)
        except Exception as e:
            print(f"[{k}/{len(todo)}] {key}: FAILED ({type(e).__name__})", flush=True)
            continue
        res["loss"][key] = dict(model=name, step=step, nll_per_token=round(L, 6),
                                secs=round(time.time() - t0, 1))
        print(f"[{k}/{len(todo)}] {key}: loss={L:.4f} ({res['loss'][key]['secs']}s)", flush=True)
        json.dump(res, open(OUT, "w"), indent=1)

    analyse(res)
    json.dump(res, open(OUT, "w"), indent=1)
    print("\nwrote", rel(OUT))


def analyse(res):
    cells = lambda_cells()
    rows = []
    for (name, step), lams in cells.items():
        key = f"{name.split('-')[-1]}_step{step}"
        if key not in res["loss"] or len(lams) < 2:
            continue
        a = np.array(lams)
        rows.append(dict(model=name.split("-")[-1], step=step,
                         loss=res["loss"][key]["nll_per_token"],
                         lam=float(a.mean()), sem=float(a.std(ddof=1) / np.sqrt(len(a))),
                         n=len(a)))
    if len(rows) < 6:
        print(f"\nonly {len(rows)} paired cells -- not enough to test a collapse"); return

    rows.sort(key=lambda r: r["loss"])
    loss = np.array([r["loss"] for r in rows])
    lam = np.array([r["lam"] for r in rows])
    step = np.array([float(r["step"]) for r in rows])
    floor = float(np.mean([r["sem"] for r in rows]))

    print(f"\n{'model':>7} {'step':>7} {'loss':>8} {'lambda':>9} {'sem':>7} {'n':>3}")
    for r in rows:
        print(f"{r['model']:>7} {r['step']:>7} {r['loss']:>8.4f} {r['lam']:>+9.4f} "
              f"{r['sem']:>7.4f} {r['n']:>3}")

    def residual(x):
        """Spread about a single smooth monotone fit in x, across ALL models pooled."""
        o = np.argsort(x)
        xs, ys = x[o], lam[o]
        # isotonic-free, assumption-light: a low-order polynomial in log x
        c = np.polyfit(np.log(xs), ys, 3)
        return float(np.std(ys - np.polyval(c, np.log(xs)), ddof=1))

    r_loss, r_step = residual(loss), residual(step)
    print(f"\n=== collapse residuals, against a single pooled curve ===")
    print(f"  seed noise floor (mean SEM over cells) : {floor:.4f}")
    print(f"  residual vs LOSS                        : {r_loss:.4f}  ({r_loss/floor:.2f}x floor)")
    print(f"  residual vs STEP                        : {r_step:.4f}  ({r_step/floor:.2f}x floor)")

    collapses = r_loss <= 2.0 * floor
    better = r_loss < r_step
    if collapses:
        verdict = (f"COLLAPSES against loss: residual {r_loss:.4f} is {r_loss/floor:.2f}x the "
                   f"seed noise floor ({floor:.4f}), within the pre-registered factor of 2. "
                   f"lambda_ca tracks how good the model is, not how long it trained.")
    elif better:
        verdict = (f"NO COLLAPSE, but loss is the better axis: residual {r_loss:.4f} "
                   f"({r_loss/floor:.2f}x floor) vs {r_step:.4f} against step. lambda_ca is not "
                   f"a function of loss alone -- it carries training information loss does not.")
    else:
        verdict = (f"NO COLLAPSE, and loss is not even the better axis ({r_loss:.4f} vs "
                   f"{r_step:.4f} against step). lambda_ca is not explained by capability.")
    print(f"\n  -> {verdict}")

    res["analysis"] = dict(
        cells=rows, seed_noise_floor=round(floor, 5),
        residual_vs_loss=round(r_loss, 5), residual_vs_step=round(r_step, 5),
        residual_over_floor=round(r_loss / floor, 3),
        collapses_within_2x_floor=bool(collapses), loss_is_better_axis=bool(better))
    res["verdict"] = verdict
    res["_analysis_provenance"] = stamp(__file__)
    res["_note"] = (
        "Does lambda_ca collapse against held-out PILE loss rather than against step (#84)? "
        "loss_baseline.py (#72) used WikiText and deliberately used only the SHAPE, since "
        "WikiText is not Pythia's training distribution. A collapse test needs the LEVEL, hence "
        "a fixed Pile slice. Pythia's published per-checkpoint loss is not obtainable -- the eval "
        "JSONs carry accuracy metrics but no loss, and omit 14m/31m entirely -- so it is "
        "recomputed. Held to Qiu et al.'s standard: the residual is measured against a seed "
        "noise floor computed from our own 8 seeds per cell, not eyeballed. A no-collapse result "
        "is informative and was pre-registered as such. No new lattice runs.")
    json.dump(res, open(OUT, "w"), indent=1)


if __name__ == "__main__":
    main()
