"""Ablate a component, re-measure lambda_ca: attribution instead of co-timing. (#100)

ROUTE 3 OF THE EXPLANANDUM PROGRAMME (`critical_analysis.md` §3), and the only one that explains
rather than correlates. The flagship is a *when*: lambda_ca crosses between step256 and step512
(F39/F46) at every radius (F77). What changed inside the model is unanswered.

WHY CO-TIMING IS NOT ENOUGH. Route 1 (`context_onset.py`, F78) and route 2 (#69/#70) both
CORRELATE an internal event with the crossing. F78 showed how thin that can be: the declared
co-timing test returned a null while the saturation matched, and neither outcome would have shown
lambda_ca MEASURES context use. F26-F29 already failed this way -- correlating lambda_ca against
white-box lambda_top across six models -- and F31 diagnosed a type mismatch as part of the cause.

The repair is not a better correlation. It is to hold the black-box measurement fixed and
MANIPULATE THE INTERNALS: ablate a component in a post-crossing model, re-measure lambda_ca, and
ask whether it falls back toward the pre-crossing level. F64 is the same move one level up --
RWKV, Pile-trained, no attention, no attractor -- so this is the within-model version of a test
this project has already run across architectures.

THE CONFOUND THAT DECIDES WHETHER THIS WORKS, MEASURED BEFORE THE GRID WAS CHOSEN. Any ablation
degrades the model, so a raw drop in lambda_ca proves nothing. On this model, at step143000:

    zero all attention outputs   held-out loss  +4.67
    zero all MLP outputs         held-out loss  +11.49

MLP ablation is more than twice as damaging. So the measurement is SELECTIVITY -- does lambda_ca
fall more than the loss degradation predicts? -- and never the raw drop.

F77 supplies the reason to expect selectivity is even possible: lambda_ca is FLAT from step1000 to
step143000 while held-out loss falls substantially over the same span, so the two are already
dissociated in training time. If they are also dissociated under ablation, that is the attribution.
If lambda_ca simply tracks loss damage, lambda_ca is a loss proxy and this line closes.

THE PROTOCOL IS IMPORTED, NOT COPIED. `dev_transition_phase3.measure` is called unchanged --
byte-identical to the code behind F39, F46 and F77 -- by patching `ar_ca.ARRule` so every rule it
builds carries the ablation. That file is NOT edited: it is stamped against
`dev_transition_phase3.json`, whose regeneration is a multi-hour run, and the import-closure guard
would (correctly) invalidate every consumer.

REFERENCE LEVELS, from F77's r=3 arm rather than re-measured:
    step256    lambda_ca = -0.0433   pre-crossing; the level an ablation must reach to have
                                     "undone" the transition
    step143000 lambda_ca = +0.3566   the plateau this experiment ablates from

PRE-REGISTERED:
  Primary.    Is there an ablation whose lambda_ca drop is LARGER than its loss degradation
              predicts -- i.e. that sits off the lambda-vs-loss line the other ablations define?
  Control.    `none` must reproduce F77's +0.3566 within between-seed spread. If the harness
              alone moves lambda_ca, nothing below is interpretable. This is the anti-vacuity
              check, not a formality.
  Null.       lambda_ca falls in proportion to loss for every ablation. Then lambda_ca carries no
              component-specific information under intervention, it is a loss proxy, and the
              explanandum programme closes. A NULL IS A GOOD RESULT.
  Kill.       If ablation drives most runs unignited, lambda is undefined (F42) and the
              comparison is not decidable -- report NOT DECIDABLE rather than reading the
              estimator floor (F40) as a measurement.
  Boundary.   Even a clean positive attributes lambda_ca to a COMPONENT, not to a mechanism.
              "Attention layers 8-15 carry it" is not "induction heads carry it"; naming the
              circuit needs #69/#70 first.

Writes results/ablate_lambda.json.
Usage:  caffeinate -dimsu .venv/bin/python -u experiments/ablate_lambda.py
        (safe to interrupt and re-run -- resumes, keyed by (ablation, seed))
"""
import sys as _sys, pathlib as _pathlib
_ROOT = _pathlib.Path(__file__).resolve().parents[1]
_sys.path[:0] = [str(_ROOT / "src"), str(_ROOT / "experiments")]
import os, json, gc, time, contextlib, statistics
os.environ.setdefault("HF_HOME", str(_ROOT / "hf_cache"))
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
import numpy as np
import torch

from provenance import stamp, rel
from dev_transition_phase3 import measure, BASE, SEEDS, T          # protocol, imported unchanged
from lyapunov import lambda_of, run_ignited
import ar_ca

STEP = "step143000"                  # post-crossing, on the flat plateau
R, N, B = 3, 48, 16                  # r=3: clear of the F62-F70 window, effect largest there (F77)
LAMBDA_PRE = -0.0433                 # F77, r=3, step256 -- the "undone" level
LAMBDA_PLATEAU = +0.3566             # F77, r=3, step143000 -- what `none` must reproduce
N_LAYERS = 24
LOSS_TOKENS = 40_000                 # fixed slice of data_ar/ref_ids.npy, same for every arm
OUT = str(_ROOT / "results" / "ablate_lambda.json")

# Layer groups rather than 24 singles: 9 arms x 8 seeds is ~2.7 h, 48 singles would be ~17 h.
# Groups still answer the selectivity question; singles are the follow-up if a group separates.
GROUPS = {"early": range(0, 8), "mid": range(8, 16), "late": range(16, 24)}
ABLATIONS = ["none",
             "attn_all", "mlp_all",
             "attn_early", "attn_mid", "attn_late",
             "mlp_early", "mlp_mid", "mlp_late"]


def _targets(model, spec):
    """The modules `spec` names, on this model."""
    if spec == "none":
        return []
    kind, where = spec.split("_", 1)
    layers = model.gpt_neox.layers
    if where == "all":
        idx = range(N_LAYERS)
    elif where.startswith("L") and where[1:].isdigit():
        idx = [int(where[1:])]              # single layer, e.g. attn_L07 -- the #100 follow-up
    else:
        idx = GROUPS[where]
    return [(layers[i].attention if kind == "attn" else layers[i].mlp) for i in idx]


def _zero(mod, inp, out):
    # GPTNeoX attention returns a 2-tuple; MLP returns a Tensor. Verified by observation before
    # this file was written, not assumed from the class name.
    if isinstance(out, tuple):
        return (torch.zeros_like(out[0]),) + out[1:]
    return torch.zeros_like(out)


def apply_ablation(model, spec):
    return [m.register_forward_hook(_zero) for m in _targets(model, spec)]


@contextlib.contextmanager
def ablating(spec):
    """Patch ARRule so every rule built inside the block carries `spec`.

    `dev_transition_phase3.measure` constructs its own ARRule and is deliberately left
    byte-identical -- it is stamped against a results file whose regeneration is a multi-hour run,
    and editing it would invalidate every consumer through the import-closure guard. Patching the
    constructor is the least invasive way to drive the IDENTICAL protocol on a modified model.
    """
    orig = ar_ca.ARRule.__init__

    def patched(self, *a, **kw):
        orig(self, *a, **kw)
        apply_ablation(self.model, spec)

    ar_ca.ARRule.__init__ = patched
    try:
        yield
    finally:
        ar_ca.ARRule.__init__ = orig


@torch.no_grad()
def held_out_loss(spec):
    """Mean cross-entropy on a FIXED token slice, under `spec`.

    data_ar/ref_ids.npy is WikiText tokenised with Pythia's tokenizer -- a proxy corpus, not the
    Pile (README's standing caveat). That is acceptable here because only the RELATIVE degradation
    across ablations is used, and every arm sees the identical slice.
    """
    from transformers import AutoModelForCausalLM
    model = AutoModelForCausalLM.from_pretrained(BASE, revision=STEP).eval()
    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    model = model.to(dev, torch.float16 if dev != "cpu" else torch.float32)
    apply_ablation(model, spec)
    ids = np.load(_ROOT / "data_ar" / "ref_ids.npy")[:LOSS_TOKENS]
    tot, n, win = 0.0, 0, 512
    for i in range(0, len(ids) - win, win):
        chunk = torch.tensor(ids[i:i + win][None, :], dtype=torch.long, device=dev)
        tot += float(model(input_ids=chunk, labels=chunk).loss)
        n += 1
    del model
    try: torch.mps.empty_cache()
    except Exception: pass
    gc.collect()
    return tot / max(n, 1)


def main():
    res = json.load(open(OUT)) if os.path.exists(OUT) else {"runs": {}, "loss": {}}
    res["_preregistration"] = dict(
        base=BASE, step=STEP, r=R, N=N, B=B, T=T, seeds=list(SEEDS), ablations=ABLATIONS,
        groups={k: [min(v), max(v)] for k, v in GROUPS.items()},
        lambda_pre=LAMBDA_PRE, lambda_plateau=LAMBDA_PLATEAU,
        reference="F77 r=3 arm, read not re-measured",
        primary="is there an ablation whose lambda drop EXCEEDS what its loss degradation "
                "predicts -- off the lambda-vs-loss line the others define?",
        control="`none` must reproduce F77's +0.3566 within between-seed spread; if the harness "
                "alone moves lambda, nothing else is interpretable",
        null="lambda falls in proportion to loss for every ablation -> lambda_ca is a loss proxy "
             "under intervention and the explanandum programme closes. A NULL IS A GOOD RESULT",
        kill="most runs unignited -> lambda undefined (F42) -> NOT DECIDABLE",
        boundary="a positive attributes lambda_ca to a COMPONENT, not to a mechanism; naming the "
                 "circuit needs #69/#70",
        loss_corpus="data_ar/ref_ids.npy (WikiText proxy), fixed slice, relative use only",
        resumable="keyed by (ablation, seed)")

    for spec in ABLATIONS:
        if spec not in res["loss"]:
            t0 = time.time()
            res["loss"][spec] = round(held_out_loss(spec), 4)
            print(f"  loss[{spec:11s}] = {res['loss'][spec]:8.4f}   ({time.time()-t0:.0f}s)",
                  flush=True)
            json.dump(res, open(OUT, "w"), indent=1)

    todo = [(a, s) for a in ABLATIONS for s in SEEDS if f"{a}|s{s}" not in res["runs"]]
    print(f"\n{len(res['runs'])} cached, {len(todo)} lambda cells to run "
          f"(~{len(todo)*140/3600:.1f} h)\n", flush=True)

    for spec, seed in todo:
        t0 = time.time()
        with ablating(spec):
            lam, dn, md, ig = measure(STEP, N, B, seed, r=R)
        res["runs"][f"{spec}|s{seed}"] = dict(
            ablation=spec, seed=seed, step=STEP, r=R, N=N, B=B, T=T,
            lambda_ca=lam, D_norm=dn, mean_damage=md, ignition_prob=ig,
            secs=round(time.time() - t0, 1))
        print(f"  {spec:11s} s={seed}  lambda={lam:+.4f}  D_norm={dn:.4f}  ign={ig:.3f}  "
              f"{time.time()-t0:.0f}s", flush=True)
        json.dump(res, open(OUT, "w"), indent=1)

    analyse(res)
    json.dump(res, open(OUT, "w"), indent=1)
    print("\nwrote", rel(OUT))


def analyse(res):
    runs, loss = res["runs"], res["loss"]
    per = {}
    for spec in ABLATIONS:
        rs = [v for v in runs.values() if v["ablation"] == spec]
        if not rs:
            continue
        lams = lambda_of(rs)                      # F42: ignited runs only
        per[spec] = dict(n=len(rs), n_ignited=len(lams),
                         lambda_median=round(statistics.median(lams), 4) if lams else None,
                         lambda_sd=round(statistics.pstdev(lams), 4) if len(lams) > 1 else 0.0,
                         loss=loss.get(spec))

    have = [s for s in ABLATIONS if s in per and per[s]["lambda_median"] is not None]
    if len(have) < len(ABLATIONS):
        res["analysis"] = dict(complete=False, have=len(have), need=len(ABLATIONS), per=per)
        res["verdict"] = (f"INCOMPLETE -- {len(have)}/{len(ABLATIONS)} ablations have a usable "
                          f"lambda. Absence of data is not absence of effect.")
        res["_analysis_provenance"] = stamp(__file__)
        print(f"\n  -> {res['verdict']}")
        return

    base_lam, base_loss = per["none"]["lambda_median"], per["none"]["loss"]
    rows = []
    for spec in ABLATIONS:
        if spec == "none":
            continue
        d_lam = base_lam - per[spec]["lambda_median"]
        d_loss = per[spec]["loss"] - base_loss
        rows.append(dict(ablation=spec, d_lambda=round(d_lam, 4), d_loss=round(d_loss, 4),
                         per_nat=round(d_lam / d_loss, 4) if d_loss > 1e-9 else None,
                         recovered_frac=round(d_lam / (base_lam - LAMBDA_PRE), 4)))

    # selectivity: lambda damage per nat of loss damage, against the mean of the others
    vals = [r["per_nat"] for r in rows if r["per_nat"] is not None]
    mu, sd = (statistics.fmean(vals), statistics.pstdev(vals)) if len(vals) > 1 else (0, 0)
    for r in rows:
        r["z_selectivity"] = (round((r["per_nat"] - mu) / sd, 2)
                              if r["per_nat"] is not None and sd > 1e-9 else None)

    harness_ok = abs(base_lam - LAMBDA_PLATEAU) <= max(2 * per["none"]["lambda_sd"], 0.05)
    ign_frac = statistics.fmean([per[s]["n_ignited"] / max(per[s]["n"], 1) for s in ABLATIONS])

    print(f"\n  {'ablation':12s} {'lambda':>8s} {'sd':>6s} {'ign':>6s} {'loss':>8s} "
          f"{'d_lam':>7s} {'d_loss':>7s} {'per_nat':>8s} {'z':>6s}")
    print(f"  {'none':12s} {base_lam:>8.4f} {per['none']['lambda_sd']:>6.3f} "
          f"{per['none']['n_ignited']}/{per['none']['n']:<4} {base_loss:>8.4f}")
    # `or -9`: z_selectivity of exactly 0.0 is falsy and would sort as -9. Harmless for display
    # ordering but the same latent bug as conditional_sensitivity's gap==0 -- made explicit.
    for r in sorted(rows, key=lambda x: -(x["z_selectivity"]
                                          if x["z_selectivity"] is not None else -9)):
        p = per[r["ablation"]]
        print(f"  {r['ablation']:12s} {p['lambda_median']:>8.4f} {p['lambda_sd']:>6.3f} "
              f"{p['n_ignited']}/{p['n']:<4} {p['loss']:>8.4f} {r['d_lambda']:>7.3f} "
              f"{r['d_loss']:>7.3f} {(r['per_nat'] if r['per_nat'] is not None else float('nan')):>8.4f} "
              f"{(r['z_selectivity'] if r['z_selectivity'] is not None else float('nan')):>6.2f}")

    if not harness_ok:
        verdict = (f"CONTROL FAILED: with no ablation lambda_ca reads {base_lam:+.4f} against "
                   f"F77's {LAMBDA_PLATEAU:+.4f} at the same checkpoint, radius and protocol. The "
                   f"harness itself moves the measurement, so nothing below is interpretable. Fix "
                   f"the harness before reading any ablation.")
    elif ign_frac < 0.5:
        verdict = (f"NOT DECIDABLE: only {ign_frac*100:.0f}% of runs ignited across ablations, so "
                   f"lambda is undefined in most cells (F42) and the estimator floor (F40) would "
                   f"be read as a measurement.")
    else:
        top = max(rows, key=lambda r: (r["z_selectivity"]
                                       if r["z_selectivity"] is not None else -9))
        if (top["z_selectivity"] or 0) >= 2.0:
            verdict = (f"SELECTIVE: {top['ablation']} costs {top['per_nat']:.3f} of lambda_ca per "
                       f"nat of held-out loss, z={top['z_selectivity']:+.2f} against the other "
                       f"ablations -- it damages lambda_ca out of proportion to how much it "
                       f"damages the model, and recovers {top['recovered_frac']*100:.0f}% of the "
                       f"distance back to the pre-crossing level. lambda_ca carries "
                       f"component-specific information under intervention. BOUNDARY: this "
                       f"attributes lambda_ca to a COMPONENT, not to a mechanism.")
        else:
            verdict = (f"NULL, AND IT IS A CLEAN ONE: no ablation damages lambda_ca out of "
                       f"proportion to its loss cost (max z={top['z_selectivity']:+.2f} for "
                       f"{top['ablation']}, threshold 2.0). lambda_ca falls with general model "
                       f"degradation and carries no component-specific information under "
                       f"intervention -- it is a loss proxy in this regime, and the explanandum "
                       f"programme closes here rather than at a fourth route.")

    print(f"\n  -> {verdict}")
    res["analysis"] = dict(complete=True, per_ablation=per, contrasts=rows,
                           harness_reproduces_F77=harness_ok, ignited_fraction=round(ign_frac, 3),
                           selectivity_mean=round(mu, 4), selectivity_sd=round(sd, 4))
    res["verdict"] = verdict
    res["_analysis_provenance"] = stamp(__file__)
    res["_note"] = (
        "Route 3 (#100). Ablate a component in a post-crossing pythia-410m, re-measure lambda_ca "
        "with dev_transition_phase3.measure driven UNCHANGED via an ARRule patch, and ask whether "
        "the drop exceeds what the ablation's held-out loss cost predicts. Raw drops are "
        "meaningless here -- zeroing all MLPs costs 2.5x the loss that zeroing all attention does "
        "-- so SELECTIVITY is the measurement. A null closes the explanandum programme: it would "
        "mean lambda_ca is a loss proxy under intervention.")


if __name__ == "__main__":
    main()
