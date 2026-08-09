"""Four measures this project cannot explain, drawn from committed results.

Not a new measurement: every number here is already in results/ and in findings.md. The purpose is
to make the open problems visible in one place, with what has been ruled out drawn alongside what
survives -- because the discarded explanations are what make each panel a puzzle rather than a gap.

  A  T* predicts degeneration and nobody knows why. The only externally-predictive result the
     project has. Ruled out: it is not the conditional's heat capacity (F97, disjoint ranges), and
     it is not settled diversity in other clothing (F112, |rho| <= 0.11 at four temperatures).
  B  The SLOPE of lambda(T) at its zero crossing tracks degeneration BETTER than T* does, on the
     same six models. Registered as exploratory with no analogue in T*. Nothing explains it, and
     n = 6.
  C  lambda_ca is largely fixed by settled diversity (rho = +0.771) -- but ~40% of its variance is
     not, and that residual has never been searched for structure.
  D  Attractor share correlates with IFEval at rho = +0.709 and with nothing else on the benchmark
     panel. Unexplained, and with a live deflation: the annealed/mid-trained recipe may drive both.

Writes fig/unexplained.png.
Usage:  .venv/bin/python experiments/fig_unexplained.py
"""
import sys as _sys, pathlib as _pathlib
_ROOT = _pathlib.Path(__file__).resolve().parents[1]
_sys.path[:0] = [str(_ROOT / "src"), str(_ROOT / "experiments")]
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from provenance import rel

R = _ROOT / "results"
L = lambda n: json.load(open(R / n))
INK, MUTED, ACC, WARN = "#1f2328", "#6e7781", "#0969da", "#bc4c00"


def main():
    fig, ax = plt.subplots(2, 2, figsize=(13, 11.5))
    fig.suptitle("Four measures this project cannot explain", fontsize=15, fontweight="bold",
                 x=0.5, y=0.985)

    # ---- A: T* vs rep_4, with the two ruled-out explanations annotated
    deg = L("degeneration_vs_tstar.json")
    tgt = {m: v for s in ("runs", "censored_above") for m, v in deg.get(s, {}).items()
           if v.get("rep_4") is not None}
    fin = [(m, v["t_star"], v["rep_4"]) for m, v in tgt.items()
           if isinstance(v.get("t_star"), (int, float))]
    a = ax[0, 0]
    a.scatter([x[1] for x in fin], [x[2] for x in fin], s=70, color=ACC, zorder=3, edgecolor="w")
    a.set_xlabel("T*  (temperature at which the attractor melts)")
    a.set_ylabel("rep_4  (greedy-decoding repetition)")
    a.set_title("A · T* predicts degeneration — mechanism unknown", fontsize=11, fontweight="bold",
                loc="left", color=INK)
    caps = []
    caps.append((a, f"$\\rho$ = +0.547 (n={len(fin)}); +0.833 at family level.  No fit line — $\\rho$ is a RANK "
                    f"statistic.\nRULED OUT: not heat capacity (F97, disjoint ranges); not diversity "
                    f"(F112, |$\\rho$|≤0.11)."))

    # ---- B: slope at lambda's zero crossing vs rep_4
    lt = L("lambda_temperature_crossing.json")["analysis"]["rows"]
    b = ax[0, 1]
    sl = [v["slope"] for v in lt.values()]; r4 = [v["rep_4"] for v in lt.values()]
    b.scatter(sl, r4, s=70, color=WARN, zorder=3, edgecolor="w")
    for m, v in lt.items():
        b.annotate(m.split("/")[-1], (v["slope"], v["rep_4"]), fontsize=7, color=MUTED,
                   xytext=(4, 4), textcoords="offset points")
    b.set_xlabel("slope of $\\lambda_{ca}(T)$ at its zero crossing")
    b.set_ylabel("rep_4")
    b.set_title("B · the crossing's STEEPNESS beats T* — and has no theory",
                fontsize=11, fontweight="bold", loc="left", color=INK)
    caps.append((b, "$\\rho$ = +0.771 (p = 0.103, n = 6) vs T*'s +0.547. Exploratory; no fit — n=6 supports "
                    "no functional form,\nand pythia-31m alone sets the right-hand end.  "
                    "8 of 14 never cross — every crosser is Pythia/GPT-Neo."))

    # ---- C: lambda vs settled diversity, residual highlighted
    dv = L("diversity_multiseed.json")["analysis"]["rows"]
    from meanfield_lambda import lambda_measured
    meas = lambda_measured()
    steps = sorted(int(k) for k in dv)
    x = np.array([dv[str(s)]["mean"] for s in steps])
    xe = np.array([dv[str(s)]["sd"] for s in steps])
    y = np.array([meas[s][0] for s in steps])
    c = ax[1, 0]
    # NO STRAIGHT-LINE FIT. Every statistic quoted in this figure is a SPEARMAN rank correlation,
    # which assumes monotonicity and nothing more. An OLS line asserts a linear model that was
    # never fitted and never tested -- and here it would be visibly wrong, because lambda saturates
    # at both ends (a dip floor near -0.09, a plateau near +0.17) so the relation is sigmoid, not
    # log-linear. The residual is therefore shown against the best MONOTONE fit (pool-adjacent-
    # violators), which is the shape Spearman actually presumes.
    o = np.argsort(x)
    xs_, ys_, ws_ = x[o], y[o].astype(float).copy(), np.ones(len(y))
    lvl = list(ys_); wts = list(ws_)                      # PAVA, isotonic increasing
    i = 0
    while i < len(lvl) - 1:
        if lvl[i] > lvl[i + 1]:
            tot = wts[i] + wts[i + 1]
            m = (lvl[i] * wts[i] + lvl[i + 1] * wts[i + 1]) / tot
            lvl[i:i + 2] = [m]; wts[i:i + 2] = [tot]
            i = max(i - 1, 0)
        else:
            i += 1
    iso, k = [], 0
    for v, w in zip(lvl, wts):
        iso += [v] * int(round(w)); k += int(round(w))
    iso = np.array(iso[:len(xs_)])
    c.errorbar(x, y, xerr=xe, fmt="o", ms=8, color=ACC, ecolor=MUTED, elinewidth=1,
               capsize=3, zorder=3, mec="w")
    c.step(xs_, iso, where="mid", color=ACC, lw=1.5, alpha=.45, zorder=2)
    for xi, yi, fi in zip(xs_, ys_, iso):
        if abs(yi - fi) > 1e-9:
            c.plot([xi, xi], [yi, fi], color=WARN, lw=2, alpha=.8, zorder=4)
    c.axhline(0, color=MUTED, lw=.8, ls=":")
    c.set_xscale("log")
    c.set_xlabel("settled-ring diversity (distinct tokens, 8 seeds)")
    c.set_ylabel("$\\lambda_{ca}$")
    c.set_title("C · $\\lambda_{ca}$ tracks diversity, but not perfectly — and n=6 hides how much",
                fontsize=11, fontweight="bold", loc="left", color=INK)
    caps.append((c, "$\\rho$ = +0.771, CI [+0.714, +0.829], so $1-\\rho^2\\approx$ 40% of rank variance is unshared.\n"
                    "Step = best MONOTONE fit (what $\\rho$ presumes); orange = its residual — but at n=6 it "
                    "nearly interpolates,\nso the bars are a LOWER BOUND on the unexplained part, not a "
                    "picture of it. No line: $\\lambda$ saturates at both ends."))

    # ---- D: attractor share vs the benchmark panel
    be = L("band_screen.json")["analysis"]["benchmark_exploratory"]
    d = ax[1, 1]
    names = list(be); vals = [be[k]["rho"] for k in names]
    cols = [WARN if abs(v) > .6 else MUTED for v in vals]
    d.barh(range(len(names)), vals, color=cols, height=.6)
    d.set_yticks(range(len(names))); d.set_yticklabels(names, fontsize=9)
    d.axvline(0, color=INK, lw=.8)
    d.set_xlabel("Spearman $\\rho$  (attractor share vs benchmark)")
    d.set_title("D · one benchmark correlates, four do not — why IFEval?",
                fontsize=11, fontweight="bold", loc="left", color=INK)
    caps.append((d, "n = 10 models, exploratory (declared before data).\n"
                    "LIVE DEFLATION: the annealed / mid-trained recipe may drive both — "
                    "F91, F87. Not yet tested."))

    # A caption written over the data is unreadable and hides points -- the first version of this
    # figure buried a Pythia label under panel B's text and a data point under panel A's. Reserve a
    # band at the bottom of each axis by extending the limit, and put the caption there.
    for a_ in ax.flat:
        a_.spines[["top", "right"]].set_visible(False)
        a_.grid(alpha=.15, lw=.6)
        a_.tick_params(labelsize=9)
    fig.tight_layout(rect=[0, 0.045, 1, 0.955])
    fig.subplots_adjust(hspace=0.46)
    for a_, txt in caps:
        bb = a_.get_position()
        fig.text(bb.x0, bb.y0 - 0.038, txt, fontsize=8, color=MUTED, va="top", ha="left",
                 linespacing=1.5)
    out = _ROOT / "fig" / "unexplained.png"
    out.parent.mkdir(exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor="white")
    print("wrote", rel(out))


if __name__ == "__main__":
    main()
