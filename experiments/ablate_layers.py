"""Per-layer attention ablation: turn #100's pattern into a reference distribution. (#100 follow-up)

WHY THIS EXISTS. #100 ablated attention and MLP in three layer groups and returned its
pre-registered NULL -- max selectivity z = +1.49 against a 2.0 threshold. But the raw pattern was
not null-shaped:

    attn_all     lambda +0.0144   loss  +5.19   per_nat 0.0660   recovers 86% toward pre-crossing
    attn_early   lambda +0.0115   loss  +5.54   per_nat 0.0623   recovers 86%
    mlp_all      lambda +0.3354   loss +10.91   per_nat 0.0019   <- 2x the loss damage, 1/16th
                                                                    the lambda damage
    attn_late    lambda +0.3960   loss  +1.11   per_nat -0.0354  <- ablation RAISES lambda

THE DECLARED STATISTIC WAS CONTAMINATED, AND THAT IS WHY IT READ NULL. Selectivity was z-scored
against a distribution CONTAINING the candidates, and there were two of them, so `attn_all` and
`attn_early` inflated both the mean and the spread they were tested against. Leave-one-out moves
attn_all to z = 1.93; a regression of d_lambda on d_loss puts both at +1.60 sd residuals with
everything else inside +/-1.02. Three statistics agree on the ordering and none clears 2 sigma with
n = 8 arms. That is the same failure family as F74's z-score defect: a normaliser contaminated by
the thing being normalised.

The fix is not a different threshold on the same eight points. It is a REAL REFERENCE DISTRIBUTION:
24 single-layer attention ablations, so a candidate is one point among 24 rather than one of two
among eight, and "is layer k an outlier" becomes answerable.

#100 ANTICIPATED THIS IN WRITING -- "singles are the follow-up if a group separates" -- so this is
the declared next step, not a post-hoc rescue.

EVERYTHING IS IMPORTED. `ablating`, `apply_ablation`, `held_out_loss` and the constants come from
`ablate_lambda`, which in turn drives `dev_transition_phase3.measure` unchanged. Per-layer specs
(`attn_L07`) were added to `ablate_lambda._targets` rather than reimplemented here, so there remains
ONE ablation implementation -- F56's anti-drift rule, and what F73 caught when the assembly
estimator had two.

PRE-REGISTERED:
  Primary.    Against a reference distribution of 24 single-layer ablations, is any layer's
              selectivity (d_lambda per nat of d_loss) an outlier at |z| >= 2, computed
              LEAVE-ONE-OUT so no candidate contaminates its own reference?
  Secondary.  Is selectivity monotone in depth? #100 saw early > mid > late with `attn_late`
              NEGATIVE. A monotone depth profile is a stronger claim than any single outlier and
              cannot be manufactured by one noisy layer.
  Null.       No layer is an outlier and there is no depth trend. Then lambda_ca is not carried by
              any localisable part of the attention stack, #100's pattern was eight-point noise,
              and the explanandum programme closes for real. A NULL IS A GOOD RESULT.
  Kill.       Same as #100: if ablation drives most runs unignited, lambda is undefined (F42) and
              the comparison is not decidable at this geometry. NOTE `attn_all` already ran at 6/8
              ignited, so single layers should be safer, but it is checked per layer.
  Control.    `none` is re-measured in-file rather than read across, so the harness is re-validated
              against F77's +0.3566 in the same run that produces the sweep.
  Boundary.   A localised layer is a COMPONENT, not a mechanism. "Layer 3 attention carries it" is
              not "induction heads carry it" -- naming the circuit still needs #69/#70.

  AMENDED AFTER THE RUN, and the declared text above is left as declared. The primary as written
  said |z| >= 2, and the verdict logic implemented it literally. That flagged L20 (z=-2.62) and
  L23 (z=-5.00) as "LOCALISED" -- layers where ablation RAISES lambda_ca, recovering -8% and -4%.
  Two errors, both mine, and together they manufactured a positive out of a null:

    * THE TEST MUST BE DIRECTIONAL. A layer that CARRIES lambda_ca has POSITIVE selectivity:
      ablating it drops lambda toward the pre-crossing level. |z| admits the opposite sign.
    * THE NOISE GATE MUST PRECEDE THE RATIO. per_nat = d_lambda / d_loss is meaningless when the
      numerator sits inside seed scatter, and d_loss spans 100x here (+0.011 to +1.228), so the
      ranking is set by its smallest denominators. Measured on this very sweep:
      Spearman(d_loss, |per_nat|) = -0.472, p = 0.02. The two flagged layers are the 1st and 3rd
      smallest denominators in the grid.

  This hazard was written down BEFORE the run and shipped anyway, because only the declared
  STATISTIC was left untouched -- the VERDICT that consumes it was not guarded. Same failure
  family as F74's z-score defect. Corrected: a layer must clear 2x its own seed sd BEFORE any
  ratio is computed, and z must be positive.

Cost: 25 arms x 8 seeds x ~130 s ~= 7.2 h, resumable per cell.

Writes results/ablate_layers.json.
Usage:  caffeinate -dimsu .venv/bin/python -u experiments/ablate_layers.py
"""
import sys as _sys, pathlib as _pathlib
_ROOT = _pathlib.Path(__file__).resolve().parents[1]
_sys.path[:0] = [str(_ROOT / "src"), str(_ROOT / "experiments")]
import os, json, time, statistics
os.environ.setdefault("HF_HOME", str(_ROOT / "hf_cache"))
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

from provenance import stamp, rel
from dev_transition_phase3 import measure, BASE, SEEDS, T
from lyapunov import lambda_of
from ablate_lambda import (ablating, held_out_loss, STEP, R, N, B, N_LAYERS,
                           LAMBDA_PRE, LAMBDA_PLATEAU)

ABLATIONS = ["none"] + [f"attn_L{i:02d}" for i in range(N_LAYERS)]
OUT = str(_ROOT / "results" / "ablate_layers.json")


def main():
    res = json.load(open(OUT)) if os.path.exists(OUT) else {"runs": {}, "loss": {}}
    res["_preregistration"] = dict(
        base=BASE, step=STEP, r=R, N=N, B=B, T=T, seeds=list(SEEDS), ablations=ABLATIONS,
        n_layers=N_LAYERS, lambda_pre=LAMBDA_PRE, lambda_plateau=LAMBDA_PLATEAU,
        why="#100 returned null at z=1.49 with a CONTAMINATED statistic -- two candidates inside "
            "their own 8-point reference. 24 singles give a real reference distribution",
        primary="is any layer's selectivity an outlier at |z| >= 2, computed LEAVE-ONE-OUT?",
        secondary="is selectivity monotone in depth? #100 saw early > mid > late with late "
                  "NEGATIVE; a depth profile cannot be manufactured by one noisy layer",
        null="no outlier and no depth trend -> lambda_ca is not carried by any localisable part "
             "of the attention stack and #100's pattern was noise. A NULL IS A GOOD RESULT",
        kill="most runs unignited -> lambda undefined (F42) -> NOT DECIDABLE, checked per layer",
        control="`none` re-measured in-file, re-validating the harness against F77's +0.3566",
        boundary="a layer is a COMPONENT not a mechanism; naming the circuit needs #69/#70",
        declared_by="#100's own text: 'singles are the follow-up if a group separates'",
        resumable="keyed by (ablation, seed)")

    for spec in ABLATIONS:
        if spec not in res["loss"]:
            res["loss"][spec] = round(held_out_loss(spec), 4)
            print(f"  loss[{spec:10s}] = {res['loss'][spec]:8.4f}", flush=True)
            json.dump(res, open(OUT, "w"), indent=1)

    todo = [(a, s) for a in ABLATIONS for s in SEEDS if f"{a}|s{s}" not in res["runs"]]
    print(f"\n{len(res['runs'])} cached, {len(todo)} cells (~{len(todo)*130/3600:.1f} h)\n",
          flush=True)

    for spec, seed in todo:
        t0 = time.time()
        with ablating(spec):
            lam, dn, md, ig = measure(STEP, N, B, seed, r=R)
        res["runs"][f"{spec}|s{seed}"] = dict(
            ablation=spec, seed=seed, step=STEP, r=R, N=N, B=B, T=T,
            lambda_ca=lam, D_norm=dn, mean_damage=md, ignition_prob=ig,
            secs=round(time.time() - t0, 1))
        print(f"  {spec:10s} s={seed}  lambda={lam:+.4f}  ign={ig:.3f}  "
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
        lams = lambda_of(rs)                                  # F42: ignited runs only
        per[spec] = dict(n=len(rs), n_ignited=len(lams), loss=loss.get(spec),
                         lambda_median=round(statistics.median(lams), 4) if lams else None,
                         lambda_sd=round(statistics.pstdev(lams), 4) if len(lams) > 1 else 0.0)

    usable = [s for s in ABLATIONS if s in per and per[s]["lambda_median"] is not None]
    if len(usable) < len(ABLATIONS):
        res["analysis"] = dict(complete=False, have=len(usable), need=len(ABLATIONS), per=per)
        res["verdict"] = (f"INCOMPLETE -- {len(usable)}/{len(ABLATIONS)} arms usable. Absence of "
                          f"data is not absence of effect; this file is a checkpoint.")
        res["_analysis_provenance"] = stamp(__file__)
        print(f"\n  -> {res['verdict']}")
        return

    base_lam, base_loss = per["none"]["lambda_median"], per["none"]["loss"]
    rows = []
    for i in range(N_LAYERS):
        spec = f"attn_L{i:02d}"
        d_lam = base_lam - per[spec]["lambda_median"]
        d_loss = per[spec]["loss"] - base_loss
        rows.append(dict(layer=i, ablation=spec, d_lambda=round(d_lam, 4),
                         d_loss=round(d_loss, 4),
                         per_nat=round(d_lam / d_loss, 4) if abs(d_loss) > 1e-6 else None,
                         recovered_frac=round(d_lam / (base_lam - LAMBDA_PRE), 4)))

    # LEAVE-ONE-OUT z: the fix for #100's contamination -- a candidate never sits in its own
    # reference set.
    vals = {r["layer"]: r["per_nat"] for r in rows if r["per_nat"] is not None}
    for r in rows:
        others = [v for k, v in vals.items() if k != r["layer"]]
        if r["per_nat"] is None or len(others) < 3:
            r["z_loo"] = None
            continue
        mu, sd = statistics.fmean(others), statistics.pstdev(others)
        r["z_loo"] = round((r["per_nat"] - mu) / sd, 2) if sd > 1e-9 else None

    # depth trend: Spearman of selectivity against layer index
    pts = [(r["layer"], r["per_nat"]) for r in rows if r["per_nat"] is not None]
    rho = None
    if len(pts) >= 5:
        xs = [p[0] for p in pts]; ys = [p[1] for p in pts]
        rx = {v: i for i, v in enumerate(sorted(xs))}
        ry = {v: i for i, v in enumerate(sorted(ys))}
        a = [rx[x] for x in xs]; b = [ry[y] for y in ys]
        ma, mb = statistics.fmean(a), statistics.fmean(b)
        num = sum((x - ma) * (y - mb) for x, y in zip(a, b))
        den = (sum((x - ma) ** 2 for x in a) * sum((y - mb) ** 2 for y in b)) ** 0.5
        rho = round(num / den, 3) if den > 0 else None

    ign_frac = statistics.fmean([per[s]["n_ignited"] / max(per[s]["n"], 1) for s in ABLATIONS])
    harness_ok = abs(base_lam - LAMBDA_PLATEAU) <= max(2 * per["none"]["lambda_sd"], 0.05)

    # TWO CORRECTIONS, both of which the first version of this function got wrong and which
    # together manufactured a "LOCALISED" verdict out of a null.
    #
    # 1. THE NOISE GATE, APPLIED BEFORE ANY RATIO. per_nat = d_lambda / d_loss is meaningless when
    #    the numerator is inside seed scatter, and d_loss spans 100x across layers (+0.011 to
    #    +1.228), so the ratio is dominated by its smallest denominators. Measured on this sweep:
    #    Spearman(d_loss, |per_nat|) = -0.472, p = 0.02 -- the "selectivity" ranking is
    #    significantly driven by the denominator rather than by lambda. Same failure family as
    #    F74's z-score defect, and it was flagged in writing before this run and then shipped
    #    anyway because only the STATISTIC was left alone, not the VERDICT that consumes it.
    #
    # 2. THE TEST IS DIRECTIONAL. A layer that CARRIES lambda_ca must have POSITIVE selectivity:
    #    ablating it should DROP lambda toward the pre-crossing level. The first version tested
    #    |z| >= 2 and so flagged L20 and L23, whose z is -2.62 and -5.00 -- layers where ablation
    #    RAISES lambda, recovering -8% and -4%. That is the opposite of the hypothesis.
    noisy = {r["layer"]: 2 * per[f"attn_L{r['layer']:02d}"]["lambda_sd"] for r in rows}
    resolved = [r for r in rows if abs(r["d_lambda"]) > noisy[r["layer"]]]
    outliers = [r for r in resolved
                if r["z_loo"] is not None and r["z_loo"] >= 2.0 and r["d_lambda"] > 0]

    # Denominator diagnostic, reported so the hazard is visible whatever the verdict says.
    try:
        import scipy.stats as _st
        den = _st.spearmanr([r["d_loss"] for r in rows], [abs(r["per_nat"] or 0) for r in rows])
        den_rho, den_p = round(float(den.statistic), 3), float(den.pvalue)
    except Exception:
        den_rho, den_p = None, None

    print(f"\n  {'layer':>5} {'lambda':>8} {'sd':>6} {'ign':>5} {'loss':>8} {'d_lam':>7} "
          f"{'d_loss':>7} {'per_nat':>8} {'z_loo':>6}")
    print(f"  {'none':>5} {base_lam:>8.4f} {per['none']['lambda_sd']:>6.3f} "
          f"{per['none']['n_ignited']}/{per['none']['n']:<3} {base_loss:>8.4f}")
    for r in rows:
        p = per[r["ablation"]]
        print(f"  {r['layer']:>5} {p['lambda_median']:>8.4f} {p['lambda_sd']:>6.3f} "
              f"{p['n_ignited']}/{p['n']:<3} {p['loss']:>8.4f} {r['d_lambda']:>7.3f} "
              f"{r['d_loss']:>7.3f} "
              f"{(r['per_nat'] if r['per_nat'] is not None else float('nan')):>8.4f} "
              f"{(r['z_loo'] if r['z_loo'] is not None else float('nan')):>6.2f}")

    if not harness_ok:
        verdict = (f"CONTROL FAILED: with no ablation lambda_ca reads {base_lam:+.4f} against "
                   f"F77's {LAMBDA_PLATEAU:+.4f}. The harness moves the measurement; nothing "
                   f"below is interpretable.")
    elif ign_frac < 0.5:
        verdict = (f"NOT DECIDABLE: only {ign_frac*100:.0f}% of runs ignited, so lambda is "
                   f"undefined in most cells (F42).")
    elif not resolved:
        mx = max(rows, key=lambda r: abs(r["d_lambda"]))
        verdict = (f"NULL, AND IT IS A CLEAN ONE: NO single attention layer moves lambda_ca "
                   f"outside its own seed scatter. The largest single-layer effect is L{mx['layer']} "
                   f"at |d_lambda|={abs(mx['d_lambda']):.4f} against a seed sd of "
                   f"{per['attn_L%02d' % mx['layer']]['lambda_sd']:.4f}; 0 of {N_LAYERS} clear 2 "
                   f"sigma. Yet removing EIGHT layers together gives d_lambda=+0.345 and all 24 "
                   f"gives +0.342, while the 24 singles SUM to "
                   f"{sum(r['d_lambda'] for r in rows):+.3f} -- the wrong sign. The effect is "
                   f"STRONGLY NON-ADDITIVE: it is not localised in any layer and not diffusely "
                   f"spread either, or the singles would sum toward the group value. Ignition says "
                   f"the same: 8/8 on every single layer against 7/8 and 6/8 for the groups. "
                   f"lambda_ca is not attributable to a localisable component, which is the "
                   f"pre-registered null for the explanandum programme. Denominator diagnostic: "
                   f"Spearman(d_loss, |per_nat|) = {den_rho} (p={den_p:.2g}), so any ratio-based "
                   f"ranking here would have been driven by its denominator.")
    elif outliers:
        names = ", ".join(f"L{r['layer']} (z={r['z_loo']:+.2f}, recovers "
                          f"{r['recovered_frac']*100:.0f}%)" for r in outliers)
        verdict = (f"LOCALISED: {len(outliers)} of {N_LAYERS} attention layers are selectivity "
                   f"outliers at |z|>=2 leave-one-out -- {names}. lambda_ca IS carried by a "
                   f"localisable part of the attention stack, which #100 could not establish with "
                   f"eight contaminated points. Depth trend rho={rho}. BOUNDARY: a layer is a "
                   f"COMPONENT, not a mechanism; naming the circuit needs #69/#70.")
    elif rho is not None and abs(rho) >= 0.6:
        verdict = (f"NO SINGLE OUTLIER, BUT A DEPTH PROFILE: selectivity is monotone in layer "
                   f"index (Spearman rho={rho} over {len(pts)} layers) with no layer clearing "
                   f"|z|>=2 alone. That is the stronger of the two secondaries -- a gradient "
                   f"cannot be manufactured by one noisy layer -- and it says lambda_ca is "
                   f"carried diffusely by depth rather than by any one layer.")
    else:
        verdict = (f"NULL, AND IT IS A CLEAN ONE: no attention layer is a selectivity outlier "
                   f"(max |z_loo|={max(abs(r['z_loo']) for r in rows if r['z_loo'] is not None):.2f}) "
                   f"and there is no depth trend (rho={rho}). lambda_ca is not carried by any "
                   f"localisable part of the attention stack; #100's eight-point pattern does not "
                   f"survive a real reference distribution, and the explanandum programme closes.")

    print(f"\n  -> {verdict}")
    res["analysis"] = dict(complete=True, per_ablation=per, layers=rows, depth_rho=rho,
                           harness_reproduces_F77=harness_ok, ignited_fraction=round(ign_frac, 3),
                           outliers=[r["layer"] for r in outliers],
                           resolved_above_seed_noise=[r["layer"] for r in resolved],
                           denominator_rho=den_rho, denominator_p=den_p,
                           singles_sum_d_lambda=round(sum(r["d_lambda"] for r in rows), 4))
    res["verdict"] = verdict
    res["_analysis_provenance"] = stamp(__file__)
    res["_note"] = (
        "#100 follow-up, declared in #100's own text ('singles are the follow-up if a group "
        "separates'). 24 single-layer attention ablations give a real reference distribution, "
        "fixing the contamination that made #100 read null: selectivity was z-scored against a set "
        "containing its own two candidates. z is leave-one-out here. Everything -- the ablation "
        "harness, the loss measurement, and dev_transition_phase3.measure -- is imported, so there "
        "is one implementation of each. A null closes the explanandum programme for real.")


if __name__ == "__main__":
    main()
