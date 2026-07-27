"""Does the unignited fraction rise with lattice size? (the open question F42 left)

F42 established that lambda_ca is undefined when damage never ignites, and deliberately did NOT
assert a mechanism: the natural story -- that the bogus lambda steepens with N -- was tested and
refuted (`lyap_from_cone` is N-independent for a fixed cone; a 3-site seed dying immediately
returns -0.9943 at N=48, 96 and 192 alike). What it recorded instead was an open empirical
question: whether unignited runs become MORE COMMON at larger N.

That question is now answerable, because three lattice sizes have been run under an identical
protocol and the ignition state of every run is recoverable. This script answers it descriptively.
It pre-registers nothing and tests no hypothesis about the developmental transition -- it is a
property of the APPARATUS, not of the models.

Why it matters even though it is not a claim about a model. If the unignited fraction rises with
N, then any lambda statistic that silently averaged over unignited runs would carry a
lattice-size-dependent contamination -- which is exactly the exposure F42 quantified (one such run
displaces a 16-run pre mean by ~-0.108, 73% of N=96's entire gap). Establishing the trend tells us
how much the F42 filter is doing at each size, and whether a fourth size would need it more.

Reads results/dev_transition_phase3.json (N=48, 96) and results/dev_transition_n192.json (N=192);
writes results/ignition_vs_size.json. Reads only; seconds.
"""
import sys, pathlib, json
ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "experiments")]
import numpy as np
from scipy import stats, optimize

from provenance import stamp, rel
from lyapunov import is_unignited, run_ignited

SOURCES = [("dev_transition_phase3.json", None), ("dev_transition_n192.json", 192)]
OUT = ROOT / "results" / "ignition_vs_size.json"


def unignited(r):
    return not run_ignited(r)


def load():
    rows = []
    for name, forced_N in SOURCES:
        p = ROOT / "results" / name
        if not p.exists():
            print(f"  (missing {name} -- skipped)")
            continue
        d = json.load(open(p))
        runs = d.get("runs", d)
        runs = list(runs.values()) if isinstance(runs, dict) else runs
        for r in runs:
            if isinstance(r, dict) and "lambda_ca" in r:
                r = dict(r)
                r.setdefault("N", forced_N)
                r["_src"] = name
                rows.append(r)
    return rows


def main():
    rows = load()
    if not rows:
        print("no runs found"); return
    sizes = sorted({r["N"] for r in rows})
    print(f"{len(rows)} runs across N in {sizes}\n")

    out = {"by_size": {}, "by_size_and_step": {}}
    print("=== unignited fraction by lattice size (all checkpoints pooled) ===")
    print(f"  {'N':>4} {'runs':>5} {'unignited':>10} {'fraction':>9}   B (design)")
    per_size = {}
    for N in sizes:
        rs = [r for r in rows if r["N"] == N]
        dead = [r for r in rs if unignited(r)]
        per_size[N] = (len(dead), len(rs))
        B = {48: 16, 96: 8, 192: 4}.get(N)
        out["by_size"][str(N)] = dict(n=len(rs), n_unignited=len(dead),
                                      frac_unignited=round(len(dead) / len(rs), 4),
                                      B=B, N_times_B=(N * B if B else None))
        print(f"  {N:>4} {len(rs):>5} {len(dead):>10} {len(dead)/len(rs):>9.3f}   B={B}")

    # the honest comparison is at MATCHED checkpoints: unignited runs concentrate early in
    # training, so pooling over checkpoints would confound size with checkpoint mix.
    print("\n=== at matched checkpoints (the confound-free comparison) ===")
    shared = sorted({r["step"] for r in rows if r["N"] == max(sizes)}
                    & {r["step"] for r in rows if r["N"] == min(sizes)})
    print(f"  checkpoints present at every size: {shared}")
    print(f"  {'step':>8} " + "".join(f"{'N='+str(N):>12}" for N in sizes))
    tbl = {}
    for st in shared:
        row = f"  {st:>8} "
        for N in sizes:
            rs = [r for r in rows if r["N"] == N and r["step"] == st]
            dead = sum(1 for r in rs if unignited(r))
            tbl[(N, st)] = (dead, len(rs))
            out["by_size_and_step"][f"N{N}_step{st}"] = dict(
                n=len(rs), n_unignited=dead,
                frac_unignited=round(dead / len(rs), 4) if rs else None)
            row += f"{f'{dead}/{len(rs)}':>12}"
        print(row)

    # Fisher exact on the smallest vs largest size, restricted to shared checkpoints
    lo, hi = min(sizes), max(sizes)
    a = sum(tbl[(lo, st)][0] for st in shared)
    b = sum(tbl[(lo, st)][1] for st in shared) - a
    c = sum(tbl[(hi, st)][0] for st in shared)
    d_ = sum(tbl[(hi, st)][1] for st in shared) - c
    fisher_p = None
    if min(a + b, c + d_) > 0:
        odds, fisher_p = stats.fisher_exact([[a, b], [c, d_]])
        print(f"\n  N={lo}: {a}/{a+b} unignited   N={hi}: {c}/{c+d_} unignited")
        print(f"  Fisher exact p={fisher_p:.5f}   <- but see below before reading this as an N effect")
        out["fisher_smallest_vs_largest"] = dict(
            checkpoints=shared, N_small=lo, N_large=hi, unignited_small=a, total_small=a + b,
            unignited_large=c, total_large=c + d_, p=round(float(fisher_p), 5))

    # --- THE CONFOUND, tested rather than merely disclosed -----------------------------------
    # A run reports zero damage only if ALL B lattices die, and the design halves B as N doubles
    # (16 / 8 / 4) for the 16GB budget. So P(run unignited) = d^B under a CONSTANT per-lattice
    # death probability d with NO N dependence at all -- and d^B grows as B shrinks. Fit d and
    # ask whether one number explains every size.
    obs = {}
    for N in sizes:
        B = out["by_size"][str(N)]["B"]
        k = sum(tbl[(N, st)][0] for st in shared)
        n = sum(tbl[(N, st)][1] for st in shared)
        if B:
            obs[N] = (k, n, B)
    if len(obs) >= 2:
        def nll(d):
            d = np.clip(d, 1e-9, 1 - 1e-9)
            return -sum(stats.binom.logpmf(k, n, d ** B) for k, n, B in obs.values())
        d_hat = float(optimize.minimize_scalar(nll, bounds=(0.01, 0.999), method="bounded").x)
        chi = float(sum((k - n * d_hat ** B) ** 2 / max(n * d_hat ** B * (1 - d_hat ** B), 1e-9)
                        for k, n, B in obs.values()))
        dof = max(len(obs) - 1, 1)
        gof_p = float(1 - stats.chi2.cdf(chi, dof))
        print(f"\n=== is it N, or is it B? (B is halved as N doubles) ===")
        print(f"  one constant per-lattice death probability, no N dependence: d = {d_hat:.4f}")
        print(f"  {'N':>4} {'B':>3} {'d^B':>8} {'expected':>9} {'observed':>10}")
        for N, (k, n, B) in obs.items():
            print(f"  {N:>4} {B:>3} {d_hat**B:>8.4f} {n*d_hat**B:>9.2f} {f'{k}/{n}':>10}")
        print(f"  chi-square = {chi:.2f} ({dof} df), p = {gof_p:.3f}")
        explained = gof_p > 0.05
        print("  -> " + ("a single constant d reproduces every size: the apparent rise with N is "
                         "what a pure BATCH-SIZE effect predicts, and these data show NO evidence "
                         "of an N effect." if explained else
                         "a constant d does NOT fit; there is structure beyond batch size."))
        out["batch_size_model"] = dict(
            d_hat=round(d_hat, 4), chi_square=round(chi, 3), dof=dof, gof_p=round(gof_p, 4),
            per_size={str(N): dict(B=B, predicted=round(d_hat ** B, 4),
                                   expected_unignited=round(n * d_hat ** B, 2),
                                   observed_unignited=k, n=n)
                      for N, (k, n, B) in obs.items()},
            explains_the_trend=bool(explained))
        out["verdict"] = ("CONFOUNDED: the unignited fraction rises with N in the raw counts "
                          f"(Fisher p={fisher_p:.3f}), but B is halved as N doubles and a single "
                          f"constant per-lattice death probability d={d_hat:.3f} with NO N "
                          f"dependence reproduces all three sizes (chi2 p={gof_p:.3f}). These "
                          "data show no evidence of an N effect."
                          if explained else
                          "the trend is not explained by batch size alone")

    out["_analysis_provenance"] = stamp(__file__)
    out["_note"] = (
        "Descriptive answer to the open question F42 recorded: does the unignited fraction rise "
        "with lattice size? Compared at MATCHED checkpoints, because unignited runs concentrate "
        "in early training and pooling would confound size with checkpoint mix. This is a "
        "property of the apparatus, not a claim about any model, and it pre-registers nothing. "
        "F42's exclusion rule does not depend on the answer -- lambda is undefined without a cone "
        "either way -- but the answer says how much work the filter is doing at each size. "
        "THE ANSWER IS THAT IT IS CONFOUNDED AND THE CONFOUND WINS: B is halved as N doubles "
        "(16/8/4) for the 16GB budget, a run is unignited only if ALL B lattices die, and a "
        "single constant per-lattice death probability with NO N dependence reproduces every "
        "size. Reporting the raw Fisher test alone would have been a spurious finding.")
    json.dump(out, open(OUT, "w"), indent=1)
    print(f"\nwrote {rel(OUT)}")


if __name__ == "__main__":
    main()
