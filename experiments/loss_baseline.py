"""Is lambda_ca an expensive perplexity proxy? Compare it against held-out loss (issue #72).

THE OBJECTION THIS ANSWERS. The developmental transition sits around 0.5-1B tokens, which is
where essentially everything in a language model changes at once. A reviewer will ask the
obvious question: what does lambda_ca tell you that the training loss does not? If lambda_ca is
a monotone function of held-out loss, the instrument is an expensive perplexity proxy with a
good test suite, and the paper's contribution shrinks to the calibration ladder. That question
should be answered here rather than in review.

NO NEW LATTICE RUNS. Every (model, revision) pair below is already measured in
dev_transition_scale.json or dev_transition_phase3.json, and every revision is already in
hf_cache. This adds only teacher-forced forward passes on a fixed held-out text.

PRE-REGISTERED BEFORE RUNNING:
  * P1  Spearman rho between per-checkpoint mean lambda_ca and held-out loss, within each model
        (checkpoints are the unit; n is small and stated). A tight monotone relation is the
        PROXY-CONSISTENT outcome and must be reported as such.
  * P2  The discriminating test is SHAPE, not correlation. Held-out loss over this range is
        expected to fall monotonically. Phase 3's verdict (b) for lambda_ca is rise ->
        overshoot -> plateau. A non-monotone function of a monotone variable cannot be a
        monotone transform of it, so an overshoot that survives here is positive evidence that
        the two instruments are not measuring the same thing. If lambda_ca turns out to be
        monotone in loss with no overshoot, THE PROXY OBJECTION STANDS and the paper says so.
  * P3  Location. Compare the bracket where loss falls fastest (per log-step) against the
        bracket where lambda_ca crosses zero. Coincidence supports redundancy; separation is
        the finding.
  * P4  Reproducibility has no loss analog, and this is a conceptual claim, not a measured one.
        Under this protocol loss is DETERMINISTIC -- one number per checkpoint, seed variance
        identically zero -- whereas lambda_ca's seed-level sign agreement goes from mixed to
        unanimous. What is measured here is only whether the sign-agreement transition lands
        where the loss curve has a distinguishing feature. It is reported as descriptive.

WHAT THIS CANNOT SETTLE. Held-out loss on WikiText is not the Pile validation loss Pythia was
trained against, so the LEVEL of the numbers is not comparable to published Pythia curves. The
SHAPE over checkpoints is what P1-P3 use, and shape is robust across evaluation corpora in a
way level is not. This is stated rather than assumed: an absolute-loss claim is not made.

Writes results/loss_baseline.json.
Usage:  caffeinate -i .venv/bin/python experiments/loss_baseline.py
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

from provenance import stamp
from lyapunov import is_unignited

# The union of every (model, revision) that carries a lambda_ca measurement.
GRID = {
    "EleutherAI/pythia-70m":  [128, 256, 512, 1000, 2000, 4000],
    "EleutherAI/pythia-160m": [128, 256, 512, 1000, 2000, 4000],
    "EleutherAI/pythia-410m": [128, 256, 512, 1000, 2000, 4000, 8000, 143000],
    "EleutherAI/pythia-1b":   [128, 256, 512, 1000, 2000, 4000],
}
SEQ_LEN, N_SEQ = 1024, 96          # 98304 held-out tokens, fixed slice, no sampling
OUT = _ROOT / "results" / "loss_baseline.json"
SCALE = _ROOT / "results" / "dev_transition_scale.json"
PHASE3 = _ROOT / "results" / "dev_transition_phase3.json"


def eval_tokens(tokenizer):
    """A FIXED held-out slice of WikiText-103 validation, tokenised once.

    Deterministic by construction: no sampling, no shuffle, first N_SEQ*SEQ_LEN tokens of the
    concatenated validation split. Pythia sizes share a tokenizer, so the same token ids are
    scored by every model and the loss is comparable across sizes.
    """
    from datasets import load_dataset
    ds = load_dataset("Salesforce/wikitext", "wikitext-103-raw-v1", split="validation")
    text = "\n\n".join(t for t in ds["text"] if t.strip())
    ids = tokenizer(text, return_tensors="pt").input_ids[0]
    need = SEQ_LEN * N_SEQ
    assert ids.numel() >= need, f"held-out slice too short: {ids.numel()} < {need}"
    return ids[:need].view(N_SEQ, SEQ_LEN)


def loss_at(model_name, revision, batches, device):
    """Mean teacher-forced NLL per token. Deterministic given (model, revision, batches)."""
    from transformers import AutoModelForCausalLM
    m = AutoModelForCausalLM.from_pretrained(model_name, revision=revision).to(device).eval()
    tot, ntok = 0.0, 0
    with torch.no_grad():
        for i in range(batches.shape[0]):
            x = batches[i:i + 1].to(device)
            logits = m(x).logits[:, :-1].float()
            tgt = x[:, 1:]
            nll = torch.nn.functional.cross_entropy(
                logits.reshape(-1, logits.shape[-1]), tgt.reshape(-1), reduction="sum")
            tot += float(nll)
            ntok += int(tgt.numel())
    del m
    try: torch.mps.empty_cache()
    except Exception: pass
    gc.collect()
    return tot / ntok


def lambda_by_checkpoint():
    """Per-(model, step) mean lambda_ca and seed sign agreement, from the existing results.

    F42: unignited runs are dropped from lambda statistics (lambda is undefined without a cone).
    Sign agreement is computed over the SAME ignited set, and n is carried through so the
    downstream analysis can state it.
    """
    out = {}
    rows = []
    if SCALE.exists():
        for v in json.load(open(SCALE))["runs"].values():
            if isinstance(v, dict) and "lambda_ca" in v:
                rows.append((v["model"], v["step"], v))
    if PHASE3.exists():
        for v in json.load(open(PHASE3))["runs"].values():
            # phase3 is 410m; keep N=48 only so the lattice size matches the scale run
            if isinstance(v, dict) and "lambda_ca" in v and v.get("N") == 48:
                rows.append(("EleutherAI/pythia-410m", v["step"], v))

    # DEDUPLICATE ON (model, step, seed). The scale run and Phase 3 both measured Pythia-410m at
    # N=48, B=16 with seeds 21-28 under the identical protocol, so 32 of these rows are the SAME
    # run appearing in two files -- byte-identical lambda values, not independent replicates.
    # Pooling them leaves the means untouched but doubles the reported n, which is precisely the
    # pseudoreplication W1 forced out of the headline. Assert agreement rather than assume it: a
    # disagreement here would mean the two files disagree about a run they both claim to have.
    seen, dedup = {}, []
    for name, step, v in rows:
        k = (name, step, v["seed"])
        if k in seen:
            assert seen[k]["lambda_ca"] == v["lambda_ca"], (
                f"two results files disagree about {k}: {seen[k]['lambda_ca']} vs "
                f"{v['lambda_ca']} -- same model, checkpoint, seed, lattice size and protocol")
            continue
        seen[k] = v
        dedup.append((name, step, v))
    n_dropped = len(rows) - len(dedup)
    if n_dropped:
        print(f"  ({n_dropped} duplicate (model, step, seed) rows dropped -- the scale run and "
              f"Phase 3 overlap at 410m/N=48; values verified identical)")
    for name, step, v in dedup:
        out.setdefault((name, step), []).append(v)
    agg = {}
    for (name, step), vs in out.items():
        ign = [v for v in vs if not (is_unignited(mean_damage=v["mean_damage"])
                                     if "mean_damage" in v else is_unignited(D_norm=v["D_norm"]))]
        if not ign:
            continue
        lam = np.array([v["lambda_ca"] for v in ign])
        agg[(name, step)] = dict(
            lambda_mean=float(lam.mean()), n_ignited=len(ign), n_runs=len(vs),
            n_unignited=len(vs) - len(ign),
            frac_positive=float((lam > 0).mean()),
            sign_unanimous=bool((lam > 0).all() or (lam < 0).all()))
    return agg


def crossing_bracket(steps, values):
    """(a, b) where the sequence changes sign between consecutive checkpoints, else None."""
    for i in range(len(steps) - 1):
        if values[i] < 0 <= values[i + 1] or values[i] >= 0 > values[i + 1]:
            return (steps[i], steps[i + 1])
    return None


def steepest_bracket(steps, values):
    """(a, b) with the largest drop in `values` per unit log-step."""
    best, out = None, None
    for i in range(len(steps) - 1):
        dl = (values[i] - values[i + 1]) / (np.log(steps[i + 1]) - np.log(steps[i]))
        if best is None or dl > best:
            best, out = dl, (steps[i], steps[i + 1])
    return out, float(best)


def main():
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    res = json.load(open(OUT)) if OUT.exists() else {"loss": {}}
    res["_preregistration"] = dict(
        grid={k: v for k, v in GRID.items()}, seq_len=SEQ_LEN, n_seq=N_SEQ,
        eval_set="Salesforce/wikitext wikitext-103-raw-v1 validation, first "
                 f"{SEQ_LEN * N_SEQ} tokens of the concatenated split",
        P1="Spearman(lambda_ca mean, held-out loss) within model, checkpoints as unit",
        P2="shape: an overshoot in lambda_ca against monotone loss refutes a monotone proxy; "
           "no overshoot means the proxy objection STANDS and is reported",
        P3="loss steepest-descent bracket vs lambda_ca zero-crossing bracket",
        P4="descriptive only: loss is deterministic here, so seed sign-agreement has no analog")

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained("EleutherAI/pythia-160m")
    batches = eval_tokens(tok)
    print(f"held-out slice: {batches.shape[0]} x {batches.shape[1]} = {batches.numel()} tokens "
          f"of WikiText-103 validation, device={device}\n", flush=True)

    todo = [(m, s) for m, ss in GRID.items() for s in ss]
    for k, (name, step) in enumerate(todo, 1):
        key = f"{name.split('-')[-1]}_step{step}"
        if key in res["loss"]:
            continue
        t0 = time.time()
        L = loss_at(name, f"step{step}", batches, device)
        res["loss"][key] = dict(model=name, step=step, nll_per_token=round(L, 6),
                                ppl=round(float(np.exp(L)), 4), secs=round(time.time() - t0, 1))
        print(f"[{k}/{len(todo)}] {key}: loss={L:.4f} ppl={np.exp(L):.1f} "
              f"({res['loss'][key]['secs']}s)", flush=True)
        json.dump(res, open(OUT, "w"), indent=1)

    if len(res["loss"]) < len(todo):
        print(f"partial: {len(res['loss'])}/{len(todo)}")
        json.dump(res, open(OUT, "w"), indent=1); return

    # ---- analysis --------------------------------------------------------------------------
    lam = lambda_by_checkpoint()
    analysis, verdicts = {}, []
    print(f"\n{'model':>8} {'step':>7} {'loss':>8} {'lambda':>9} {'n_ign':>6} {'unanimous':>10}")
    for name, steps in GRID.items():
        tag = name.split("-")[-1]
        shared = [s for s in steps if (name, s) in lam]
        if len(shared) < 3:
            print(f"  ({tag}: only {len(shared)} checkpoints with lambda -- skipped)")
            continue
        L = [res["loss"][f"{tag}_step{s}"]["nll_per_token"] for s in shared]
        M = [lam[(name, s)]["lambda_mean"] for s in shared]
        for s, l_, m_ in zip(shared, L, M):
            a = lam[(name, s)]
            print(f"{tag:>8} {s:>7} {l_:>8.4f} {m_:>+9.4f} {a['n_ignited']:>6} "
                  f"{str(a['sign_unanimous']):>10}")

        rho, p = stats.spearmanr(M, L)
        loss_monotone = all(L[i] > L[i + 1] for i in range(len(L) - 1))
        # overshoot: lambda rises to an interior maximum and then falls back
        imax = int(np.argmax(M))
        overshoot = 0 < imax < len(M) - 1 and (M[imax] - M[-1]) > 0
        xb = crossing_bracket(shared, M)
        sb, srate = steepest_bracket(shared, L)
        analysis[tag] = dict(
            checkpoints=shared, loss=[round(x, 6) for x in L],
            lambda_mean=[round(x, 5) for x in M],
            n_ignited=[lam[(name, s)]["n_ignited"] for s in shared],
            sign_unanimous=[lam[(name, s)]["sign_unanimous"] for s in shared],
            spearman_rho=round(float(rho), 4), spearman_p=round(float(p), 5), n=len(shared),
            loss_monotone_decreasing=bool(loss_monotone),
            lambda_overshoot=bool(overshoot),
            lambda_argmax_step=shared[imax],
            lambda_crossing_bracket=xb, loss_steepest_bracket=sb,
            loss_steepest_rate_per_lognat=round(srate, 5),
            brackets_coincide=bool(xb is not None and tuple(xb) == tuple(sb)))
        verdicts.append((tag, overshoot, loss_monotone, xb, sb))

    print(f"\n{'model':>8} {'rho':>8} {'p':>8} {'loss mono':>10} {'overshoot':>10} "
          f"{'lam cross':>14} {'loss steep':>14}")
    for tag, a in analysis.items():
        print(f"{tag:>8} {a['spearman_rho']:>+8.3f} {a['spearman_p']:>8.4f} "
              f"{str(a['loss_monotone_decreasing']):>10} {str(a['lambda_overshoot']):>10} "
              f"{str(a['lambda_crossing_bracket']):>14} {str(a['loss_steepest_bracket']):>14}")

    n_over = sum(1 for _, o, _, _, _ in verdicts if o)
    n_mono = sum(1 for _, _, mo, _, _ in verdicts if mo)
    n_apart = sum(1 for _, _, _, xb, sb in verdicts if xb is not None and tuple(xb) != tuple(sb))
    if n_over and n_mono == len(verdicts):
        verdict = (f"NOT A MONOTONE PROXY: held-out loss falls monotonically in all "
                   f"{n_mono}/{len(verdicts)} models, while lambda_ca overshoots in {n_over}. A "
                   f"non-monotone function of a monotone variable is not a monotone transform "
                   f"of it. Separately, the zero-crossing and steepest-loss brackets differ in "
                   f"{n_apart} of {len(verdicts)} models.")
    elif not n_over:
        verdict = ("PROXY OBJECTION STANDS on this evidence: lambda_ca shows no overshoot "
                   "against loss on this grid, so it is consistent with a monotone transform "
                   "of held-out loss. The paper must report this.")
    else:
        verdict = (f"MIXED: overshoot in {n_over}/{len(verdicts)} models but loss is not "
                   f"monotone in all ({n_mono}/{len(verdicts)}) -- read the per-model rows, "
                   f"not this line.")
    print(f"\n  -> {verdict}")

    res["analysis"] = analysis
    res["verdict"] = verdict
    res["_analysis_provenance"] = stamp(__file__)
    res["_note"] = (
        "Held-out loss per checkpoint, to test whether lambda_ca is an expensive perplexity "
        "proxy. No new lattice runs: every (model, revision) already carries a lambda_ca "
        "measurement and every revision was already cached. Loss is teacher-forced NLL per "
        "token on a FIXED slice of WikiText-103 validation and is therefore deterministic -- "
        "zero seed variance by construction, which is itself the point of P4. The LEVEL of "
        "these numbers is not comparable to published Pythia curves (different corpus); only "
        "the SHAPE across checkpoints is used. lambda statistics drop unignited runs per F42 "
        "with n carried through.")
    res["_config"] = dict(grid=GRID, seq_len=SEQ_LEN, n_seq=N_SEQ, device=device,
                          eval_set="wikitext-103-raw-v1 validation")
    json.dump(res, open(OUT, "w"), indent=1)
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
