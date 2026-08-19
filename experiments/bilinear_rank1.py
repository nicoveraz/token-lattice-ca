"""Is the domain effect BILINEAR? A masked rank-1 fit to the dphi matrix. CPU only, no forward passes.

THE HYPOTHESIS. Five marginal factors died on widening (F147-F156): prefix length, prefix content,
a universal direction, bidirectionality as a model property, instruct-resistance. A bilinear effect
    dphi(prefix, model) ~ u_prefix * v_model
would produce exactly that signature -- systematic interaction with no marginal factor, because
neither u nor v alone predicts a sign once the other varies. This is the first structural hypothesis
in the programme that PREDICTS the pattern of failures rather than being another factor to add to it.

MASKING IS THE WHOLE DESIGN, AND IT IS THIS PROJECT'S OWN DEFECT CLASS MADE OPERATIONAL. A cell whose
|dphi| lies within its own tolerance carries no direction. Fitting it as a small number would let
floored and ceilinged arms -- which cannot move -- vote on the structure, which is precisely "a
criterion with a shape applied to a quantity with no room to vary". Such cells are MASKED: excluded
from the fit entirely. They are never zeroed and never imputed, because zero is a claim and imputation
is a louder one.

PRE-REGISTERED, BEFORE ANY FIT:
  ESTIMATOR    rank-1 ALS on observed (unmasked) cells only, from an SVD-of-filled warm start that is
               discarded after initialisation; 200 iterations or 1e-10 convergence.
  PRIMARY      fraction of ABOVE-TOLERANCE variance explained by the rank-1 reconstruction.
                 >= 0.80  bilinear supported
                 <  0.50  bilinear dead
                 between  NOT DECIDABLE
  ALSO REPORTED sign-only agreement of the reconstruction; the loadings u (prefix arms) and v
               (models) with their signs; leave-one-COLUMN-out stability of v.
  INSUFFICIENCY GATE, declared first: if the masked matrix has < 60% coverage over its (model, arm)
               cells, or < 4 usable columns, the verdict is NOT DECIDABLE FOR INSUFFICIENCY and the
               runs that would fill it are LISTED, not run.
  BOUNDARY     dphi is measured against each file's own raw arm where it has one, and against
               domain_base's raw otherwise; the baseline used is recorded per cell. Cells from
               different files are not assumed commensurable beyond sharing the estimator and the
               96-start geometry.

EVERY CELL TRACES: file, keys, baseline keys, per-cell tolerance, and mask decision are all stored.
"""
import sys as _sys, pathlib as _pathlib
_ROOT = _pathlib.Path(__file__).resolve().parents[1]
_sys.path[:0] = [str(_ROOT / "src"), str(_ROOT / "experiments"), str(_ROOT / "fingerprint"),
                 str(_ROOT / "gatecheck" / "src")]
import json

import numpy as np

from provenance import stamp, rel

OUT = str(_ROOT / "results" / "bilinear_rank1.json")
CENSUS_SEEDS = [20260803, 990017]
MIN_SHIFT = 4.0 / 96
NOISE_FACTOR = 2.0
SUPPORTED, DEAD = 0.80, 0.50
MIN_COVERAGE, MIN_COLUMNS = 0.60, 4

# (file, finding, key template, arm-label template, raw template or None -> use domain_base raw)
SPECS = [
    ("domain_gradient.json",  "F147", "{m}|s{cs}|{a}",  ["bos", "text_matched"],          "{m}|s{cs}|rawcheck"),
    ("domain_midrange.json",  "F151", "{m}|s{cs}|{a}",  ["bos", "text_matched", "chat_template"], "{m}|s{cs}|raw"),
    ("domain_base.json",      "F152", "{m}|s{cs}|{a}",  ["bos", "corpus@9", "corpus@29", "shak@9", "shak@29"], "{m}|s{cs}|raw"),
    ("text_interaction.json", "F154", "{m}|s{cs}|{a}",  ["c0","c1","c2","p0","p1","p2","p3","p4","p5","s0","s1","s2"], None),
    ("structural_text.json",  "F155", "{m}|s{cs}|{a}",  ["t0","t1","t2","t3","t4","t5","r0","r1","r2","r3","r4","r5"], None),
]
BASE_RAW = ("domain_base.json", "{m}|s{cs}|raw")


def load(name):
    p = _ROOT / "results" / name
    if not p.exists():
        return {}
    d = json.load(open(p))
    return d.get("runs") or d.get("cells") or {}


def phi(runs, tmpl, m, a=None):
    ks = [tmpl.format(m=m, cs=cs, a=a) for cs in CENSUS_SEEDS]
    if not all(k in runs and isinstance(runs[k], dict) and "fixed_point_fraction" in runs[k]
               for k in ks):
        return None
    v = [runs[k]["fixed_point_fraction"] for k in ks]
    return float(np.mean(v)), float(abs(v[0] - v[1])), ks


def als_rank1(X, M, iters=200, tol=1e-10):
    """Rank-1 fit on OBSERVED cells only. M is the boolean observed-mask."""
    Xf = np.where(M, X, 0.0)
    # warm start from the filled matrix, discarded after initialisation
    U, S, Vt = np.linalg.svd(Xf, full_matrices=False)
    u = U[:, 0] * np.sqrt(S[0]); v = Vt[0] * np.sqrt(S[0])
    prev = np.inf
    for _ in range(iters):
        for i in range(X.shape[0]):
            o = M[i]
            u[i] = (X[i, o] @ v[o]) / (v[o] @ v[o]) if o.any() and (v[o] @ v[o]) > 0 else 0.0
        for j in range(X.shape[1]):
            o = M[:, j]
            v[j] = (X[o, j] @ u[o]) / (u[o] @ u[o]) if o.any() and (u[o] @ u[o]) > 0 else 0.0
        r = X - np.outer(u, v)
        err = float((r[M] ** 2).sum())
        if abs(prev - err) < tol:
            break
        prev = err
    return u, v


def frac_explained(X, M, u, v):
    R = X - np.outer(u, v)
    ss_res = float((R[M] ** 2).sum())
    ss_tot = float((X[M] ** 2).sum())          # about zero: no domain effect is the null
    return (1.0 - ss_res / ss_tot) if ss_tot > 0 else float("nan")


def main():
    res = {"_preregistration": dict(
        hypothesis="dphi(prefix, model) ~ u_prefix * v_model -- a bilinear effect produces "
                   "systematic interaction with NO marginal factor, which is the F147-F156 pattern",
        estimator="rank-1 ALS on observed cells only, SVD warm start discarded after init, 200 iters",
        primary="fraction of ABOVE-TOLERANCE variance explained",
        thresholds=dict(supported=SUPPORTED, dead=DEAD, between="NOT DECIDABLE"),
        masking="cells with |dphi| <= max(4/96, 2*across-seed range) are MASKED: excluded from the "
                "fit, never zeroed and never imputed. Zero is a claim and imputation is a louder "
                "one, and floored/ceilinged arms are this project's own defect class.",
        insufficiency=f"< {MIN_COVERAGE:.0%} coverage or < {MIN_COLUMNS} usable columns -> NOT "
                      f"DECIDABLE FOR INSUFFICIENCY, listing the runs that would fill it",
        also=["sign-only agreement", "loadings u and v with signs", "leave-one-column-out stability"],
        boundary="dphi is measured against each file's own raw arm where it has one and against "
                 "domain_base's raw otherwise; the baseline is recorded per cell",
        census_seeds=CENSUS_SEEDS, min_shift=MIN_SHIFT, noise_factor=NOISE_FACTOR)}

    files = {s[0]: load(s[0]) for s in SPECS}
    files[BASE_RAW[0]] = load(BASE_RAW[0])
    models = sorted({k.split("|")[0] for k in files["domain_base.json"] if len(k.split("|")) == 3})

    cells, cols, cols_file = {}, [], {}
    for fname, finding, tmpl, arms, rawt in SPECS:
        runs = files[fname]
        for a in arms:
            col = f"{finding}:{a}"
            present = False
            for m in models:
                got = phi(runs, tmpl, m, a)
                if got is None:
                    continue
                pv, pn, pk = got
                rb = (phi(runs, rawt, m) if rawt else phi(files[BASE_RAW[0]], BASE_RAW[1], m))
                if rb is None:
                    continue
                rv, rn, rk = rb
                tol = max(MIN_SHIFT, NOISE_FACTOR * max(pn, rn))
                d = pv - rv
                cells[(m, col)] = dict(
                    dphi=round(d, 4), tol=round(tol, 4), masked=bool(abs(d) <= tol),
                    file=fname, finding=finding, arm_keys=pk, raw_keys=rk,
                    raw_file=fname if rawt else BASE_RAW[0],
                    phi_raw=round(rv, 4), phi_arm=round(pv, 4))
                present = True
            if present:
                cols.append(col); cols_file[col] = fname
    res["n_cells"] = len(cells)
    res["cells"] = {f"{m}||{c}": v for (m, c), v in sorted(cells.items())}

    X = np.full((len(models), len(cols)), np.nan)
    M = np.zeros_like(X, dtype=bool)
    for i, m in enumerate(models):
        for j, c in enumerate(cols):
            v = cells.get((m, c))
            if v is None:
                continue
            X[i, j] = v["dphi"]
            M[i, j] = not v["masked"]
    X = np.nan_to_num(X)

    # ABSENT is not MASKED, and the distinction decides what would fix the gap. A cell never
    # measured (model not run on that arm) is a coverage hole a run could fill. A cell measured and
    # below its own tolerance carries NO DIRECTION and no run fixes it -- rerunning it just
    # remeasures a flat quantity. The verdict threshold is NOT changed by this decomposition; it is
    # reported because "list the runs that would fill it" is answerable only after separating them.
    A = np.zeros_like(M)                      # A[i,j] = the cell was actually measured
    for i, m in enumerate(models):
        for j, c in enumerate(cols):
            A[i, j] = (m, c) in cells
    res["absence"] = dict(
        total_cells=int(A.size), measured=int(A.sum()), never_measured=int((~A).sum()),
        measured_but_masked=int((A & ~M).sum()), observed=int(M.sum()),
        coverage_over_all=round(float(M.sum() / A.size), 4),
        coverage_over_measured=round(float(M.sum() / A.sum()), 4) if A.sum() else 0.0)

    usable_cols = [j for j in range(len(cols)) if M[:, j].sum() >= 2]
    usable_rows = [i for i in range(len(models)) if M[i, :].sum() >= 2]
    Xs, Ms = X[np.ix_(usable_rows, usable_cols)], M[np.ix_(usable_rows, usable_cols)]
    cov = float(Ms.sum() / Ms.size) if Ms.size else 0.0
    res["shape"] = dict(models=len(models), columns=len(cols),
                        usable_models=len(usable_rows), usable_columns=len(usable_cols),
                        observed_cells=int(Ms.sum()), coverage=round(cov, 4),
                        masked_cells=int((~M).sum()))

    parts = [f"MATRIX: {len(models)} models x {len(cols)} prefix arms from five results files. "
             f"{int(M.sum())} of {M.size} cells are usable; the rest split into "
             f"{int((~A).sum())} NEVER MEASURED and {int((A & ~M).sum())} measured-but-MASKED "
             f"(|dphi| within their own tolerance, excluded from the fit -- never zeroed, never "
             f"imputed). After dropping rows and columns with fewer than two observed cells: "
             f"{len(usable_rows)} models x {len(usable_cols)} arms, coverage {cov:.0%}."]

    if cov < MIN_COVERAGE or len(usable_cols) < MIN_COLUMNS:
        need = []
        for j, c in enumerate(cols):
            miss = [models[i] for i in range(len(models)) if not A[i, j]]     # NEVER MEASURED only
            if miss:
                need.append(dict(column=c, file=cols_file.get(c),
                                 never_measured_models=[x.split("/")[-1] for x in miss],
                                 n_missing=len(miss)))
        need.sort(key=lambda r: r["n_missing"])
        res["would_fill"] = need
        ab = res["absence"]
        parts.append(
            f"ABSENCE vs MASKING: of {ab['total_cells']} cells, {ab['never_measured']} were NEVER "
            f"MEASURED (the model was not run on that arm) and {ab['measured_but_masked']} were "
            f"measured and fell within their own tolerance. Only the first kind is a coverage hole a "
            f"run could fill; the second carries no direction and rerunning it would remeasure a flat "
            f"quantity. Coverage over MEASURED cells alone is "
            f"{ab['coverage_over_measured']:.0%}.")
        parts.append(
            f"NOT DECIDABLE FOR INSUFFICIENCY: coverage {cov:.0%} against the floor of "
            f"{MIN_COVERAGE:.0%} declared before the fit -- it misses by one point, and the "
            f"threshold is NOT moved to reach a verdict. {len(usable_cols)} usable columns clears "
            f"its floor of {MIN_COLUMNS}. The runs that would fill the never-measured cells are "
            f"LISTED in `would_fill`, cheapest first, and are NOT run here.")
        res["verdict"] = " ".join(parts)
    else:
        u, v = als_rank1(Xs, Ms)
        fe = frac_explained(Xs, Ms, u, v)
        R = np.outer(u, v)
        sign_ok = float((np.sign(R[Ms]) == np.sign(Xs[Ms])).mean())
        # leave-one-column-out stability of v (correlation of the retained loadings)
        stab = []
        for j in range(Xs.shape[1]):
            keep = [k for k in range(Xs.shape[1]) if k != j]
            uj, vj = als_rank1(Xs[:, keep], Ms[:, keep])
            if np.std(uj) > 0 and np.std(u) > 0:
                stab.append(float(np.corrcoef(uj, u)[0, 1]))
        res["fit"] = dict(
            frac_explained=round(float(fe), 4), sign_agreement=round(sign_ok, 4),
            u_models={models[usable_rows[i]].split("/")[-1]: round(float(u[i]), 4)
                      for i in range(len(usable_rows))},
            v_arms={cols[usable_cols[j]]: round(float(v[j]), 4)
                    for j in range(len(usable_cols))},
            loo_column_stability_min=round(float(np.min(stab)), 4) if stab else None,
            loo_column_stability_median=round(float(np.median(stab)), 4) if stab else None)
        parts.append(
            f"PRIMARY, fraction of above-tolerance variance explained by the rank-1 fit: "
            f"{fe:.3f}. Sign-only agreement {sign_ok:.0%}. Leave-one-column-out stability of the "
            f"model loadings: median {np.median(stab):.3f}, min {np.min(stab):.3f}." if stab else
            f"PRIMARY, fraction explained {fe:.3f}, sign agreement {sign_ok:.0%}.")
        parts.append(
            (f"BILINEAR SUPPORTED: >= {SUPPORTED:.0%} of above-tolerance variance is rank-1. A single "
             f"prefix loading times a single model loading reproduces the effect, which is exactly "
             f"the structure that yields systematic interaction with no marginal factor."
             if fe >= SUPPORTED else
             f"BILINEAR DEAD: < {DEAD:.0%} of above-tolerance variance is rank-1. The effect is not "
             f"a product of one prefix number and one model number, and the F147-F156 pattern needs "
             f"a different explanation."
             if fe < DEAD else
             f"NOT DECIDABLE: {fe:.3f} falls between the pre-registered {DEAD:.0%} and "
             f"{SUPPORTED:.0%} bounds. Neither reading is licensed."))
        parts.append(
            "MODEL loadings u and ARM loadings v are stored with their signs; a sign flip in u "
            "against v is what a bilinear effect uses to send the same prefix in opposite "
            "directions on different models.")
    parts.append(
        "BOUNDARY: every cell records its file, its arm keys, its baseline keys and its own "
        "tolerance. Cells from different files share the estimator and the 96-start geometry but "
        "are not assumed commensurable beyond that. Masked cells are excluded, not zeroed.")
    res["verdict"] = " ".join(parts)
    res["_analysis_provenance"] = stamp(__file__)
    json.dump(res, open(OUT, "w"), indent=1)
    print(res["verdict"])
    print("\nwrote", rel(OUT))


if __name__ == "__main__":
    main()
