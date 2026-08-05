"""Gate B of #102: at what radius does the ring retain a memorized sequence at all?

WHY THIS GATE EXISTS. F72 measured that the prompt is ERASED at r=2 -- only 2% of a corpus seed
survives settling -- so at the radius this project has mostly used, "is a memorized sequence
retained more than a control?" is vacuous by construction: nothing is retained. Before #102's
basin-width measurement can mean anything, the smallest radius at which retention EXISTS and
DIFFERS between memorized and control sequences has to be found. That number is itself the
gate's result, and published memorization uses a 32-token prefix -- far beyond the r <= 16 this
project has ever run -- so "no practical radius works" is a live outcome that would close #102
honestly.

DESIGN. The ring IS the sequence: N = 64 cells initialised to the exact 64-token sequences Gate A
stored (results/memorization_gate_a.json, per_sequence) -- memorized (Biderman et al.'s
duped.410m set) against the length-matched Pile controls, THE SAME model (pythia-410m) supplying
the dynamics. T = 0.02, the greedy regime the memorization criterion is defined in. After 16
sweeps, retention = fraction of sites still holding their seeded token.

  r = 2      the F72 CONTROL: both arms should erase to ~0. If they do not, F72's erasure
             reading needs revisiting before anything else here is trusted.
  r in {4, 8, 16, 32}   the sweep. 32 covers the anchor's own prefix convention.

PRE-REGISTERED:
  Primary   the smallest r at which memorized retention exceeds control retention beyond the
            bootstrap 95% CI over sequences.
  Control   at r = 2 both arms sit near zero (F72). A large retention at r=2 fails the gate's
            own known-answer check.
  Kill      no separation at ANY r <= 32: retention is radius-limited, not memory-limited, and
            #102 closes as "vacuous at practical radii" -- an honest end, cheaper than the
            basin measurement it would otherwise have licensed.
  Note      separation here is NOT yet basin width. Retention of the exact sequence is the
            precondition; basin width (corrupted starts flowing back) is the experiment this
            gate licenses, and the NLL covariate for it is already stored by Gate A.

Writes results/memorization_gate_b.json.
Usage:  caffeinate -dimsu .venv/bin/python -u experiments/memorization_gate_b.py
        (resumable per (radius, arm, batch))
"""
import sys as _sys, pathlib as _pathlib
_ROOT = _pathlib.Path(__file__).resolve().parents[1]
_sys.path[:0] = [str(_ROOT / "src"), str(_ROOT / "experiments")]
import gc, json, os, time

os.environ.setdefault("HF_HOME", str(_ROOT / "hf_cache"))
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
import numpy as np
import torch

from provenance import stamp, rel

OUT = str(_ROOT / "results" / "memorization_gate_b.json")
GATE_A = _ROOT / "results" / "memorization_gate_a.json"
MODEL = "EleutherAI/pythia-410m"
RADII = [2, 4, 8, 16, 32]
N, SWEEPS, T = 64, 16, 0.02
B = 16                       # sequences per settle batch
N_BATCH = 3                  # 48 sequences per arm
SEED = 20260804
BOOT = 2000


def main():
    res = json.load(open(OUT)) if os.path.exists(OUT) else {"runs": {}}
    res["_preregistration"] = dict(
        model=MODEL, radii=RADII, N=N, sweeps=SWEEPS, T=T, B=B, n_batches=N_BATCH,
        seed=SEED, bootstrap=BOOT,
        primary="smallest r at which memorized retention exceeds control beyond the bootstrap "
                "95% CI over sequences",
        control="at r=2 both arms erase to ~0 (F72); large retention there fails the gate's own "
                "known-answer check",
        kill="no separation at any r<=32 -> #102 closes as vacuous at practical radii",
        note="retention is the PRECONDITION for basin width, not basin width itself")
    runs = res["runs"]

    ga = json.load(open(GATE_A))
    seqs = {"mem": [d["tokens"] for d in ga["per_sequence"]["memorized"]],
            "ctl": [d["tokens"] for d in ga["per_sequence"]["control"]]}
    rng = np.random.default_rng(SEED)
    for arm in seqs:
        idx = rng.choice(len(seqs[arm]), size=B * N_BATCH, replace=False)
        seqs[arm] = [seqs[arm][i] for i in idx]

    from ar_ca import ARRule, run
    rule = ARRule(MODEL)
    print(f"{MODEL} loaded", flush=True)

    for r in RADII:
        for arm in ("mem", "ctl"):
            for b in range(N_BATCH):
                k = f"r{r}|{arm}|b{b}"
                if k in runs: continue
                t0 = time.time()
                init = np.array(seqs[arm][b * B:(b + 1) * B], dtype=np.int64)
                fin = run(rule, B=B, N=N, r=r, T=T, sweeps=SWEEPS, scheme="none",
                          init_state=init, seed=SEED + b, order="per_replica")["final"]
                ret = (fin == init).mean(axis=1)          # per-sequence retention
                runs[k] = dict(r=r, arm=arm, batch=b,
                               retention=[round(float(x), 4) for x in ret],
                               secs=round(time.time() - t0, 1))
                json.dump(res, open(OUT, "w"), indent=1)
            vals = [x for bb in range(N_BATCH) for x in runs[f"r{r}|{arm}|b{bb}"]["retention"]]
            print(f"  r={r:<3} {arm}: retention {np.mean(vals):.3f} "
                  f"({runs[f'r{r}|{arm}|b0']['secs']:.0f}s/batch)", flush=True)

    analyse(res)
    json.dump(res, open(OUT, "w"), indent=1)
    print("\nwrote", rel(OUT))


def analyse(res):
    runs = res["runs"]
    rng = np.random.default_rng(1)
    rows, first = {}, None
    for r in RADII:
        vals = {}
        for arm in ("mem", "ctl"):
            v = [x for b in range(N_BATCH)
                 for x in runs.get(f"r{r}|{arm}|b{b}", {}).get("retention", [])]
            if v:
                vals[arm] = np.array(v)
        if len(vals) < 2:
            continue
        diff = float(vals["mem"].mean() - vals["ctl"].mean())
        boots = []
        for _ in range(BOOT):
            m = rng.choice(vals["mem"], size=len(vals["mem"]))
            c = rng.choice(vals["ctl"], size=len(vals["ctl"]))
            boots.append(m.mean() - c.mean())
        lo, hi = np.percentile(boots, [2.5, 97.5])
        sep = bool(lo > 0)
        rows[r] = dict(mem=round(float(vals["mem"].mean()), 4),
                       ctl=round(float(vals["ctl"].mean()), 4),
                       diff=round(diff, 4), ci95=[round(float(lo), 4), round(float(hi), 4)],
                       separates=sep)
        if sep and first is None:
            first = r
    print(f"\n  {'r':>4} {'memorized':>10} {'control':>9} {'diff':>8} {'95% CI':>18} sep")
    for r, v in rows.items():
        print(f"  {r:>4} {v['mem']:>10.3f} {v['ctl']:>9.3f} {v['diff']:>+8.3f} "
              f"[{v['ci95'][0]:+.3f},{v['ci95'][1]:+.3f}] {v['separates']}")

    f72_ok = 2 in rows and rows[2]["mem"] < 0.15 and rows[2]["ctl"] < 0.15
    parts = []
    if not f72_ok and 2 in rows:
        parts.append(
            f"KNOWN-ANSWER CHECK FAILED: r=2 retains {rows[2]['mem']:.2f}/{rows[2]['ctl']:.2f} "
            f"where F72 measured erasure (~0.02 corpus survival). Either the N=64 exact-sequence "
            f"ring differs from F72's N=96 corpus-seed regime in a way that matters, or the "
            f"harness is wrong -- resolve before believing anything below.")
    else:
        parts.append(f"F72 control holds: r=2 erases both arms "
                     f"({rows.get(2, {}).get('mem', float('nan')):.2f}/"
                     f"{rows.get(2, {}).get('ctl', float('nan')):.2f}).")
    if first is not None:
        parts.append(
            f"GATE B PASSES: memorized sequences are retained above matched controls from "
            f"r={first} (diff {rows[first]['diff']:+.3f}, CI excludes 0). The basin-width "
            f"experiment is licensed at r>={first}, and its cost model should use that radius, "
            f"not the project's habitual r=2. Retention is the precondition, not basin width.")
    else:
        parts.append(
            "KILL: no separation at any r<=32. Retention is radius-limited, not memory-limited "
            "-- the ring erases memorized and control sequences alike at every practical radius, "
            "and #102 closes as vacuous by erasure. An honest end, bought cheaply.")
    verdict = " ".join(parts)
    print(f"\n  -> {verdict}")
    res["analysis"] = dict(rows={str(r): v for r, v in rows.items()},
                           first_separating_radius=first, f72_control_ok=bool(f72_ok))
    res["verdict"] = verdict
    res["_analysis_provenance"] = stamp(__file__)
    res["_note"] = (
        "Gate B of #102: the radius calibration F72 makes mandatory. The ring IS the 64-token "
        "sequence (init_state), the model is the one the memorization labels were computed for "
        "(duped pythia-410m), T=0.02 is the greedy regime the criterion is defined in, and the "
        "sequences are exactly Gate A's stored sets, so the NLL covariate pairs one-to-one. "
        "r=2 is the known-answer arm (F72 erasure); r=32 covers the anchor's own 32-token prefix "
        "convention. Bootstrap CI over sequences; retention is the precondition for basin width, "
        "not the quantity itself.")


if __name__ == "__main__":
    main()
