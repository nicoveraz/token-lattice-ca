"""Falsification test for F111: does DIVERSITY collapse against loss where lambda_ca does not?

WHY THIS IS THE SHARPEST TEST AVAILABLE. F111 claims lambda_ca is a function of the settled ring's
diversity -- that training step enters only through the diversity it produces. F100 established
that lambda_ca does NOT collapse against model quality: across-family spread at matched
bits-per-byte is 0.0588 against a 0.0197 seed floor, and at bpb ~2.3 Pythia sits in its dip while
OLMo-2 is already at plateau.

If lambda_ca really is a function of diversity, then diversity must fail against loss in the SAME
WAY. Two quantities that are functions of one another cannot disagree about whether a third
variable organises them. So:

  diversity ALSO fails to collapse  ->  consistent with F111; the reduction survives a test that
                                       could have killed it
  diversity DOES collapse           ->  lambda_ca and diversity behave differently under the same
                                       comparison, they are NOT the same quantity, and F111's
                                       reduction is incomplete and must be amended

This is a falsification attempt on a finding recorded yesterday, run on the grid F100 already
built. The bpb values and lambda values are re-used UNCHANGED from loss_collapse_families.json;
only the diversity axis is new, which is one settle per cell and no damage run.

PRE-REGISTERED:
  PRIMARY   across-family spread of DIVERSITY at matched bits-per-byte, against its own seed floor,
            compared with the same statistic at matched token count. The comparison is the one F100
            ran on lambda_ca, so the two are commensurable by construction.
  READING   F111 predicts diversity fails to collapse, mirroring lambda_ca. A collapse is the
            falsification.
  CONTROL   the diagonal check: within Pythia, diversity must reproduce the 8/24/41/193/191/188
            already measured by transplant_s at the same geometry. A mismatch means the settle
            differs and nothing else is read.
  BOUNDARY  three families, one radius, one temperature. Bits-per-byte removes the tokenizer
            confound, not the corpus one (F100's boundary applies unchanged).

Writes results/diversity_vs_loss.json.  Resumable per cell.
"""
import sys as _sys, pathlib as _pathlib
_ROOT = _pathlib.Path(__file__).resolve().parents[1]
_sys.path[:0] = [str(_ROOT / "src"), str(_ROOT / "experiments"), str(_ROOT / "gatecheck" / "src")]
import gc, json, os, time
os.environ.setdefault("HF_HOME", str(_ROOT / "hf_cache"))
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
import numpy as np, torch
from provenance import stamp, rel
from gatecheck.cohort import cohort_complete

OUT = str(_ROOT / "results" / "diversity_vs_loss.json")
SRC = _ROOT / "results" / "loss_collapse_families.json"
R, N, B, SWEEPS, T = 2, 48, 8, 30, 0.7
# MULTI-SEED, and the first version was not. Its control failed at 62% because it compared a
# single-seed settle against transplant_s's single-seed values, and settled diversity in the
# low-diversity regime has an across-seed sd of 4-12 on means of 7-32 (F111 amendment). A single
# draw of that is not a measurement, which is precisely what the control existed to detect.
# DELIBERATELY DISJOINT FROM diversity_multiseed's [21..28]. The control compares this run's
# Pythia diversity against that run's 8-seed means -- and with the SAME seeds the settle is
# bit-identical, so the control returned exactly 0.00 sigma on all six checkpoints and could not
# have failed. A control that cannot fail is not a control. These seeds make the comparison an
# independent draw from the same distribution, which is what the check was supposed to be.
SEEDS = [31, 32, 33, 34, 35, 36, 37, 38]
REPOS = {"pythia-410m": "EleutherAI/pythia-410m",
         "olmo2-1b": "allenai/OLMo-2-0425-1B",
         "olmo1-0724": "allenai/OLMo-1B-0724-hf"}
# The SEED-AVERAGED pythia values from diversity_multiseed.json (8 seeds each), with their
# measured across-seed sd. The control checks agreement in units of that sd rather than as a
# percentage, because a percentage tolerance on a quantity whose sd is 55% of its mean is
# meaningless -- that is the error the first version of this control made.
PYTHIA_DIAGONAL = {128: (7.5, 4.093), 256: (26.25, 11.443), 512: (31.5, 11.906),
                   1000: (185.125, 7.574), 2000: (205.125, 10.252), 4000: (196.125, 9.867)}
CONTROL_SIGMA = 2.5          # agreement required, in units of the measured across-seed sd


def evict(repo, revision):
    try:
        from huggingface_hub import scan_cache_dir
        info = scan_cache_dir()
        hits = [rv.commit_hash for rp in info.repos if rp.repo_id == repo
                for rv in rp.revisions if revision in rv.refs]
        if not hits:
            return None
        st = info.delete_revisions(*hits); sz = st.expected_freed_size_str; st.execute(); return sz
    except Exception:
        return None


def main():
    res = json.load(open(OUT)) if os.path.exists(OUT) else {"cells": {}}
    src = json.load(open(SRC))["cells"]
    res["_preregistration"] = dict(
        r=R, N=N, B=B, sweeps=SWEEPS, T=T, repos=REPOS,
        source="loss_collapse_families.json -- bpb and lambda re-used UNCHANGED",
        primary="across-family spread of DIVERSITY at matched bits-per-byte vs at matched tokens, "
                "the same comparison F100 ran on lambda_ca",
        reading="F111 predicts diversity FAILS to collapse, mirroring lambda_ca; a collapse "
                "falsifies the reduction",
        control=f"within Pythia, seed-averaged diversity must agree with diversity_multiseed's "
                f"8-seed means to within {CONTROL_SIGMA} sigma of their measured across-seed sd",
        seeds=SEEDS,
        boundary="bits-per-byte removes the tokenizer confound, not the corpus one")
    from ar_ca import ARRule, run
    for k, c in sorted(src.items(), key=lambda kv: (kv[1]["family"], kv[1]["tokens_B"])):
        if "bpb" not in c or k in res["cells"]:
            continue
        repo = REPOS[c["family"]]
        t0 = time.time()
        try:
            rule = ARRule(repo, revision=c["revision"])
            pooled, per = [], []
            for sd in SEEDS:
                fin = run(rule, B=B, N=N, r=R, T=T, sweeps=SWEEPS, scheme="none",
                          seed=sd, order="per_replica")["final"]
                pooled.append(int(len(np.unique(fin.reshape(-1)))))
                per.append(float(np.mean([len(np.unique(fin[b])) for b in range(B)])))
        except Exception as e:
            print(f"  {k}: FAILED {type(e).__name__}"[:120], flush=True)
            res["cells"][k] = dict(**{a: c[a] for a in ("family", "revision", "tokens_B", "bpb")},
                                   failed=f"{type(e).__name__}: {e}"[:180])
            json.dump(res, open(OUT, "w"), indent=1); continue
        row = dict(family=c["family"], revision=c["revision"], tokens_B=c["tokens_B"],
                   bpb=c["bpb"], lambda_ca=c.get("lambda_ca"), lambda_sd=c.get("lambda_sd"),
                   n_seeds=len(SEEDS), pooled_per_seed=pooled,
                   distinct=round(float(np.mean(pooled)), 3),          # POOLED, seed-averaged
                   distinct_sd=round(float(np.std(pooled)), 3),
                   per_replica=round(float(np.mean(per)), 3),
                   secs=round(time.time() - t0, 1))
        res["cells"][k] = row
        print(f"  {k:<44} bpb={row['bpb']:.4f} distinct={row['distinct']:>6.2f} "
              f"sd={row['distinct_sd']:>5.1f} lam={row['lambda_ca']} ({row['secs']:.0f}s)",
              flush=True)
        json.dump(res, open(OUT, "w"), indent=1)
        del rule
        try: torch.mps.empty_cache()
        except Exception: pass
        gc.collect()
        if c["family"] != "pythia-410m":
            evict(repo, c["revision"])
    analyse(res)
    json.dump(res, open(OUT, "w"), indent=1)
    print("\nwrote", rel(OUT))


def _spread(curves, key, val):
    use = {f: sorted([(c[key], c[val]) for c in cs if c.get(val) is not None])
           for f, cs in curves.items()}
    use = {f: v for f, v in use.items() if len(v) >= 3}
    if len(use) < 2:
        return None, 0
    lo = max(min(x for x, _ in v) for v in use.values())
    hi = min(max(x for x, _ in v) for v in use.values())
    if not hi > lo:
        return None, 0
    g = np.linspace(lo, hi, 12)
    st = []
    for f, v in use.items():
        xs = np.array([x for x, _ in v]); ys = np.array([y for _, y in v]); o = np.argsort(xs)
        st.append(np.interp(g, xs[o], ys[o]))
    return float(np.mean(np.std(np.stack(st), axis=0))), len(use)


def analyse(res):
    cells = [c for c in res["cells"].values() if "distinct" in c]
    parts = []
    py = {int(c["revision"].replace("step", "")): c["distinct"]
          for c in cells if c["family"] == "pythia-410m" and c["revision"].startswith("step")}
    ok = True
    if py:
        z = {st: abs(py[st] - PYTHIA_DIAGONAL[st][0]) / max(PYTHIA_DIAGONAL[st][1], 1e-9)
             for st in py if st in PYTHIA_DIAGONAL}
        ok = bool(z and max(z.values()) <= CONTROL_SIGMA)
        parts.append(
            f"CONTROL: Pythia's seed-averaged diversity against diversity_multiseed's 8-seed means, "
            f"in units of their measured across-seed sd: "
            f"{ {s: round(v, 2) for s, v in sorted(z.items())} } sigma, worst "
            f"{max(z.values()):.2f} against a {CONTROL_SIGMA} gate. "
            + ("Agreement within seed noise, so the settle matches the geometry F111 was measured "
               "at." if ok else "IT DOES NOT -- the settle differs and nothing below is read."))
    curves = {}
    for c in cells:
        curves.setdefault(c["family"], []).append(c)
    coh = cohort_complete(sorted(REPOS), sorted(curves), unit="family")
    parts.append(f"COHORT: {coh.reason}")
    print(f"\n  {'family':<12} {'tokens':>8} {'bpb':>8} {'distinct':>9} {'lambda':>9}")
    for c in sorted(cells, key=lambda c: (c["family"], c["tokens_B"])):
        lam = c.get("lambda_ca")
        print(f"  {c['family']:<12} {c['tokens_B']:>7.1f}B {c['bpb']:>8.4f} "
              f"{c['distinct']:>9.2f} {lam if lam is None else f'{lam:+.4f}':>9}")
    if not (ok and coh.complete):
        res["analysis"] = dict(control_ok=ok, cohort=coh.block())
        res["verdict"] = " ".join(parts) + " NOT DECIDABLE."
        res["_analysis_provenance"] = stamp(__file__); print(f"\n  -> {res['verdict']}"); return

    d_bpb, n1 = _spread(curves, "bpb", "distinct")
    d_tok, n2 = _spread(curves, "tokens_B", "distinct")
    floor = float(np.mean([c["distinct_sd"] for c in cells])) / np.sqrt(B)
    collapses = bool(d_bpb is not None and d_bpb <= floor)
    parts.append(
        f"PRIMARY: across-family spread of DIVERSITY at matched bits-per-byte is "
        f"{d_bpb if d_bpb is None else round(d_bpb, 4)} over {n1} families, against "
        f"{d_tok if d_tok is None else round(d_tok, 4)} at matched token count, with a seed floor "
        f"of {floor:.4f}. "
        + (f"DIVERSITY COLLAPSES AGAINST LOSS where lambda_ca did not (F100: 0.0588 against a "
           f"0.0197 floor). The two quantities therefore behave DIFFERENTLY under the same "
           f"comparison, so they are not the same quantity and F111's reduction is INCOMPLETE. "
           f"That is the falsification this test was built to look for, and it fired."
           if collapses else
           f"Diversity does NOT collapse against loss, mirroring lambda_ca's failure to (F100). "
           f"Two quantities claimed to be functions of one another agree about whether model "
           f"quality organises them, which is what F111 requires and what this test could have "
           f"refuted. The reduction survives."))
    parts.append(
        "BOUNDARY: three families, one radius, one temperature; bpb removes the tokenizer confound "
        "but not the corpus one, and architecture, data order and optimiser still differ across "
        "families simultaneously. bpb and lambda were re-used unchanged, so only diversity is new.")
    v = " ".join(parts)
    print(f"\n  -> {v}")
    res["analysis"] = dict(control_ok=ok, cohort=coh.block(),
                           spread_diversity_at_bpb=None if d_bpb is None else round(d_bpb, 5),
                           spread_diversity_at_tokens=None if d_tok is None else round(d_tok, 5),
                           diversity_seed_floor=round(floor, 5), collapses=collapses,
                           lambda_reference=dict(spread_at_bpb=0.0588, floor=0.0197, source="F100"))
    res["verdict"] = v
    res["_analysis_provenance"] = stamp(__file__)
    res["_note"] = ("Falsification attempt on F111. If lambda_ca is a function of diversity, the two "
                    "must agree about whether loss organises them. F100 showed lambda_ca does not "
                    "collapse against bits-per-byte; this asks the same of diversity on the same grid.")


if __name__ == "__main__":
    main()
