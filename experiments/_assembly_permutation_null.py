"""Permutation null for the r=2 rung's shape statistic -- F76's caveat 1. (#20)

WHY. F76 reported that two of sixteen measures show an interior temperature maximum clearing
between-seed noise. Sixteen measures with no multiple-comparison correction is exactly the shape
F39 applied BH-FDR to, and this had nothing. "Two survive" is a description until it is compared
against how many would survive with NO temperature structure at all.

THE NULL. Permute the temperature labels WITHIN each seed. That destroys any dependence on
temperature while preserving, exactly, each seed's own distribution of values -- so a measure whose
apparent peak is really between-seed scatter keeps its scatter and loses its shape. The shape
statistic is then recomputed the same way it is on the real data, including the between-seed noise
term, so the test is self-consistent rather than comparing against a fixed threshold.

WHAT IT REPORTS. Per measure, p = P(clears the interior-peak test | no temperature structure), then
BH-FDR across the measures. Also the expected number of survivors under the null, which is the
direct answer to "would two of sixteen have cleared anyway?"

Reuses `assembly_temperature._shape` and `dev_transition_phase3.bh_fdr` rather than reimplementing
either -- the anti-drift rule (F56, and the shadowed duplicate F73 caught on its first run).

Usage:  .venv/bin/python experiments/_assembly_permutation_null.py [n_permutations] [kind] [radius]
"""
import sys as _sys, pathlib as _pathlib
_ROOT = _pathlib.Path(__file__).resolve().parents[1]
_sys.path[:0] = [str(_ROOT / "src"), str(_ROOT / "experiments")]
import json
import random
import statistics

import assembly_temperature as AT
from dev_transition_phase3 import bh_fdr

OUT = _ROOT / "results" / "assembly_temperature.json"


def collect(runs, kind, r):
    """{measure: {seed: {T: value}}} for every measure present, dropping Nones."""
    cells = [(k, v) for k, v in runs.items() if v["kind"] == kind and v["r"] == r]
    if not cells:
        return {}, []
    names = sorted((cells[0][1].get("oriented") or {}).keys()) + ["logA_ring_n2", "logA_ring_n3"]
    out = {}
    for m in names:
        per_seed = {}
        for k, v in cells:
            try:
                val = AT._val(v, m)
            except (KeyError, TypeError):
                val = None
            if val is None:
                continue
            per_seed.setdefault(v["seed"], {})[v["T"]] = val
        # keep only seeds that carry the full temperature grid, so a permutation is well defined
        per_seed = {s: d for s, d in per_seed.items() if len(d) == len(AT.TEMPS)}
        if len(per_seed) >= 2:
            out[m] = per_seed
    return out, sorted({v["seed"] for _, v in cells})


def shape_of(per_seed, temps):
    """Median-per-T with between-seed sd, then the project's own _shape."""
    series = {}
    for T in temps:
        vs = [d[T] for d in per_seed.values()]
        series[T] = dict(value=statistics.median(vs),
                         sd=statistics.pstdev(vs) if len(vs) > 1 else 0.0,
                         n_seeds=len(vs))
    return AT._shape(series)


def main(n_perm=2000, kind="ar", r=2, seed=0):
    runs = json.loads(OUT.read_text())["runs"]
    data, seeds = collect(runs, kind, r)
    if not data:
        print(f"no {kind} r={r} cells yet")
        return 1
    temps = list(AT.TEMPS)
    n_seeds = len(next(iter(data.values())))
    print(f"{kind} r={r}: {len(data)} measures, {n_seeds} complete seeds, "
          f"{len(temps)} temperatures, {n_perm} permutations\n")

    rng = random.Random(seed)
    observed, pvals, names = {}, [], []
    null_hits = {m: 0 for m in data}

    for m, per_seed in data.items():
        observed[m] = shape_of(per_seed, temps)

    # One shared set of permutations across measures, so the "expected survivors" count below is
    # computed on the same draws rather than on independent ones per measure.
    survivors_per_perm = []
    for _ in range(n_perm):
        perms = {s: rng.sample(temps, len(temps)) for s in next(iter(data.values()))}
        k = 0
        for m, per_seed in data.items():
            shuffled = {s: {temps[i]: d[perms[s][i]] for i in range(len(temps))}
                        for s, d in per_seed.items()}
            if shape_of(shuffled, temps).get("interior_peak"):
                null_hits[m] += 1
                k += 1
        survivors_per_perm.append(k)

    for m in data:
        names.append(m)
        pvals.append((null_hits[m] + 1) / (n_perm + 1))       # add-one, never reports p = 0
    adj = bh_fdr(pvals)

    print(f"{'measure':22s} {'observed':>10s} {'margin':>9s} {'noise':>8s} "
          f"{'p_perm':>8s} {'p_BH':>8s}  verdict")
    order = sorted(range(len(names)), key=lambda i: pvals[i])
    n_sig = 0
    for i in order:
        m, o = names[i], observed[names[i]]
        obs = ("interior peak" if o.get("interior_peak")
               else "degenerate end" if o.get("peaks_at_degenerate_end") else "no peak")
        sig = o.get("interior_peak") and adj[i] <= 0.05
        n_sig += bool(sig)
        print(f"{m:22s} {obs:>10s} {o.get('margin', 0):>9.3f} "
              f"{o.get('between_seed_noise', 0):>8.3f} {pvals[i]:>8.4f} {adj[i]:>8.4f}  "
              f"{'SURVIVES' if sig else ''}")

    exp = statistics.fmean(survivors_per_perm)
    hi = sorted(survivors_per_perm)[int(0.95 * len(survivors_per_perm))]
    obs_n = sum(1 for m in data if observed[m].get("interior_peak"))
    print(f"\n  observed interior peaks         : {obs_n} of {len(data)}")
    print(f"  expected under the null         : {exp:.2f}  (95th pct {hi})")
    print(f"  P(>= {obs_n} survivors by chance)     : "
          f"{(sum(1 for k in survivors_per_perm if k >= obs_n) + 1) / (n_perm + 1):.4f}")
    print(f"  survive BH-FDR at 0.05          : {n_sig}")
    return 0


if __name__ == "__main__":
    a = _sys.argv[1:]
    _sys.exit(main(int(a[0]) if a else 2000,
                   a[1] if len(a) > 1 else "ar",
                   int(a[2]) if len(a) > 2 else 2))
