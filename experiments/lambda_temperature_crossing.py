"""Does lambda_ca's TEMPERATURE RESPONSE predict degeneration, as T*'s does?

THE STRUCTURAL LESSON THIS TESTS. F112 measured that the settled state predicts nothing external
(diversity vs greedy rep_4: |rho| <= 0.11 across four temperatures on 26 models, every p > 0.59)
while T* -- where the diversity curve crosses a threshold as temperature varies -- reaches rho =
0.547 on the same target. The lesson drawn was that the useful quantity is a RESPONSE, not a LEVEL.

lambda_ca has only ever been used as a level, and as a level it buys no decision. This applies the
same move to it: not lambda at a temperature, but the TEMPERATURE AT WHICH lambda CROSSES ZERO, and
the slope there. If the lesson is right, that derived scalar should behave like T* rather than like
diversity.

THREE OUTCOMES, ALL WORTH HAVING:
  PREDICTS      lambda_ca acquires an external use by the same move that made T* work, and the
                project gains a second predictor of the same target -- not a second target, but
                converging evidence from a differently-derived scalar, which matters because F93
                left the anchor with one leg.
  DOES NOT      the structural lesson is confirmed on a second quantity and the negative gets
                stronger: it is not that lambda is the wrong level, it is that this family does not
                transfer at all.
  COINCIDES     if T_cross and T* land on top of each other, two of the project's readings are one
                measurement. That is F112's worry one level over, and better found here than by a
                reviewer.

PRE-REGISTERED:
  PRIMARY    rho(T_cross, rep_4) across models where lambda actually crosses, against T*'s rho =
             0.547 measured on the same target in F112.
  SECOND     rho(slope at crossing, rep_4). A crossing has a location and a steepness; the location
             is the analogue of T* and the steepness has no analogue, so it is exploratory.
  CENSORING  a model whose lambda does not change sign on the grid has NO T_cross, exactly as
             gpt-neo-125M has no T* (censored above). Censored models are reported and EXCLUDED,
             not imputed -- F87's distinction between "no attractor" and "attractor not yet melted"
             applies here unchanged.
  RANGE GATE T_cross must span more than RANGE_K times its own seed floor across models before any
             correlation is quoted.
  COINCIDENCE rho(T_cross, T*) is reported on the subset with both, as the check against having
             measured one thing twice.
  BOUNDARY   greedy-scoped target, one radius, one lattice size. A positive result gives lambda_ca
             a use in the regime rep_4 is defined in, not in general.

Writes results/lambda_temperature_crossing.json.  Resumable per (model, T, seed).
"""
import sys as _sys, pathlib as _pathlib
_ROOT = _pathlib.Path(__file__).resolve().parents[1]
_sys.path[:0] = [str(_ROOT / "src"), str(_ROOT / "experiments"), str(_ROOT / "gatecheck" / "src")]
import gc, itertools, json, os, time
os.environ.setdefault("HF_HOME", str(_ROOT / "hf_cache"))
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
import numpy as np, torch
from provenance import stamp, rel
from lyapunov import lyap_from_cone, is_unignited
from dev_transition_phase3 import FIT_KW
from gatecheck import dynamic_range, carries_verdict

OUT = str(_ROOT / "results" / "lambda_temperature_crossing.json")
DEG = _ROOT / "results" / "degeneration_vs_tstar.json"
# EXTENDED DOWN after the first pass censored 7 of 10 models: every crossing found sat in the
# 0.3-0.5 interval, i.e. squeezed against the grid's lower edge, and the models that did not cross
# were positive-but-small at T=0.3 (gpt-neo-125M +0.014, pythia-31m +0.049). A crossing pinned to
# the scan boundary is F59's retracted defect on a different axis: the extremum you find is the
# edge, not an extremum. 0.1 and 0.2 put the crossings inside the scan.
TEMPS = [0.1, 0.2, 0.3, 0.5, 0.7, 0.9, 1.1]
SEEDS = [21, 22, 23]
R, N, B, SETTLE, SWEEPS, BLOCK = 2, 48, 16, 12, 22, 3
RANGE_K = 2.0
TSTAR_REFERENCE = 0.547            # F112, T* vs rep_4 on the same target
MODELS = ["EleutherAI/pythia-14m", "EleutherAI/pythia-31m", "EleutherAI/pythia-70m",
          "EleutherAI/pythia-160m", "EleutherAI/pythia-410m", "EleutherAI/gpt-neo-125M",
          "gpt2", "gpt2-medium", "gpt2-large", "facebook/opt-350m",
          "bigscience/bloom-560m", "state-spaces/mamba-130m-hf", "RWKV/rwkv-4-169m-pile",
          "Salesforce/codegen-350M-mono"]


def evict(repo):
    try:
        from huggingface_hub import scan_cache_dir
        info = scan_cache_dir()
        h = [rv.commit_hash for rp in info.repos if rp.repo_id == repo for rv in rp.revisions]
        if not h: return
        st = info.delete_revisions(*h); st.execute()
    except Exception:
        pass


def lam_at(rule, T, seed):
    from ar_ca import run
    rng = np.random.default_rng(seed)
    base = run(rule, B=B, N=N, r=R, T=T, sweeps=SETTLE, scheme="none", init="random",
               seed=seed)["final"]
    c = N // 2
    idx = [c + k for k in range(-(BLOCK // 2), BLOCK - BLOCK // 2)]
    fl = base.copy()
    for j in idx:
        fl[:, j] = rng.choice(rule.init_pool, size=B)
    u = np.random.default_rng(seed + 1).random(SWEEPS * N * B)
    u2 = np.concatenate([u.reshape(SWEEPS * N, B)] * 2, axis=1).reshape(-1)
    c2 = run(rule, B=2 * B, N=N, r=R, T=T, sweeps=SWEEPS, scheme="none",
             init_state=np.concatenate([base, fl], axis=0), seed=seed + 2, u_stream=u2)
    s = c2["snaps"]
    diff = (s[:, :B] != s[:, B:])
    cone = np.roll(diff, N // 2 - idx[len(idx) // 2], axis=2).mean(axis=1)
    md = float(diff[-1].mean())
    return float(lyap_from_cone(cone, N, **FIT_KW)[0]), md, bool(not is_unignited(mean_damage=md))


def crossing(temps, lams):
    """Interpolated temperature where lambda changes sign, and the slope there. None if no sign change."""
    for i in range(len(temps) - 1):
        a, b = lams[i], lams[i + 1]
        if a < 0 <= b or a >= 0 > b:
            t = temps[i] + (temps[i + 1] - temps[i]) * (0 - a) / (b - a)
            return float(t), float((b - a) / (temps[i + 1] - temps[i]))
    return None, None


def main():
    res = json.load(open(OUT)) if os.path.exists(OUT) else {"cells": {}}
    res["_preregistration"] = dict(
        models=MODELS, temps=TEMPS, seeds=SEEDS, r=R, N=N, B=B, settle=SETTLE, sweeps=SWEEPS,
        range_k=RANGE_K, tstar_reference=TSTAR_REFERENCE,
        primary="rho(T_cross, rep_4) against T*'s 0.547 on the same target (F112)",
        second="rho(slope at crossing, rep_4) -- exploratory, no analogue in T*",
        censoring="a model whose lambda does not change sign has NO T_cross and is EXCLUDED, not "
                  "imputed (F87's distinction). A temperature where damage never ignites is "
                  "MISSING DATA at that temperature (F42: lambda is undefined), not evidence about "
                  "the model -- the cell is dropped and the model kept, with the count recorded",
        grid_extension="the first pass used [0.3..1.1] and censored 7 of 10 models, with every "
                       "crossing found sitting in the 0.3-0.5 interval against the lower edge. "
                       "0.1 and 0.2 were added so crossings fall inside the scan rather than on "
                       "its boundary -- F59's defect on a different axis",
        coincidence="rho(T_cross, T*) on the subset with both, as the check against measuring one "
                    "thing twice",
        boundary="greedy-scoped target, one radius, one lattice size")
    from ar_ca import ARRule
    for name in MODELS:
        if all(f"{name}|T{T}|s{sd}" in res["cells"] for T in TEMPS for sd in SEEDS):
            continue
        try:
            rule = ARRule(name)
        except Exception as e:
            print(f"  {name}: LOAD FAILED {type(e).__name__}", flush=True)
            res["cells"][f"{name}|failed"] = dict(model=name, error=repr(e)[:160])
            json.dump(res, open(OUT, "w"), indent=1); continue
        for T in TEMPS:
            for sd in SEEDS:
                k = f"{name}|T{T}|s{sd}"
                if k in res["cells"]: continue
                t0 = time.time()
                try:
                    lam, md, ig = lam_at(rule, T, sd)
                except Exception as e:
                    res["cells"][k] = dict(model=name, T=T, seed=sd, failed=repr(e)[:160])
                    json.dump(res, open(OUT, "w"), indent=1); continue
                res["cells"][k] = dict(model=name, T=T, seed=sd, lambda_ca=round(lam, 5),
                                       mean_damage=md, ignited=ig, secs=round(time.time()-t0, 1))
                json.dump(res, open(OUT, "w"), indent=1)
        ls = [np.mean([res["cells"][f"{name}|T{T}|s{s}"]["lambda_ca"] for s in SEEDS
                       if res["cells"].get(f"{name}|T{T}|s{s}", {}).get("ignited")] or [np.nan])
              for T in TEMPS]
        print(f"  {name:<32} lam(T) = {[None if np.isnan(x) else round(float(x),4) for x in ls]}",
              flush=True)
        del rule
        try: torch.mps.empty_cache()
        except Exception: pass
        gc.collect()
        if not name.startswith("EleutherAI/pythia-410m"):
            evict(name)
    analyse(res)
    json.dump(res, open(OUT, "w"), indent=1)
    print("\nwrote", rel(OUT))


def _rho_p(a, b, seed=0):
    a, b = np.array(a, float), np.array(b, float)
    rk = lambda x: np.argsort(np.argsort(x))
    r = float(np.corrcoef(rk(a), rk(b))[0, 1]); n = len(a)
    if n <= 8:
        null = [np.corrcoef(np.array(p), rk(b))[0, 1] for p in itertools.permutations(rk(a))]
    else:
        g = np.random.default_rng(seed)
        null = [np.corrcoef(g.permutation(rk(a)), rk(b))[0, 1] for _ in range(20000)]
    return r, float(np.mean(np.abs(np.array(null)) >= abs(r) - 1e-12)), n


def analyse(res):
    deg = json.load(open(DEG))
    tgt = {m: v for s in ("runs", "censored_above") for m, v in deg.get(s, {}).items()
           if v.get("rep_4") is not None}
    rows, censored = {}, []
    for name in MODELS:
        ls, sds = [], []
        for T in TEMPS:
            v = [res["cells"][f"{name}|T{T}|s{s}"]["lambda_ca"] for s in SEEDS
                 if res["cells"].get(f"{name}|T{T}|s{s}", {}).get("ignited")]
            ls.append(float(np.mean(v)) if v else np.nan)
            sds.append(float(np.std(v)) if len(v) > 1 else np.nan)
        ok = [(T, l, sd) for T, l, sd in zip(TEMPS, ls, sds) if not np.isnan(l)]
        if len(ok) < 3:
            censored.append((name, "fewer than 3 igniting temperatures")); continue
        t_ok = [x[0] for x in ok]; l_ok = [x[1] for x in ok]
        n_dead = len(TEMPS) - len(ok)
        tc, sl = crossing(t_ok, l_ok)
        if tc is None:
            censored.append((name, round(min(l_ok), 4), round(max(l_ok), 4))); continue
        rows[name] = dict(temps=t_ok, lam=[round(x, 5) for x in l_ok], n_unignited=n_dead,
                          seed_sd=[round(x, 5) for x in sds if not np.isnan(x)],
                          T_cross=round(tc, 4), slope=round(sl, 4),
                          rep_4=tgt.get(name, {}).get("rep_4"),
                          t_star=tgt.get(name, {}).get("t_star"))
    print(f"\n  {'model':<32} {'T_cross':>8} {'slope':>8} {'rep_4':>7} {'T*':>8}")
    for m, v in rows.items():
        r4 = "   -" if v["rep_4"] is None else f"{v['rep_4']:.3f}"
        print(f"  {m:<32} {v['T_cross']:>8.4f} {v['slope']:>8.3f} {r4:>7} {str(v['t_star']):>8}")

    parts = [f"CENSORING: {len(censored)} of {len(MODELS)} models never change sign on "
             f"{TEMPS} and have no T_cross -- {censored}. Reported and EXCLUDED, not imputed."]
    use = {m: v for m, v in rows.items() if v["rep_4"] is not None}
    if len(use) < 5:
        res["analysis"] = dict(rows=rows, censored=censored)
        res["verdict"] = " ".join(parts) + f" Only {len(use)} usable models -- NOT DECIDABLE."
        res["_analysis_provenance"] = stamp(__file__); print(f"\n  -> {res['verdict']}"); return
    tc = [v["T_cross"] for v in use.values()]; r4 = [v["rep_4"] for v in use.values()]
    floor = float(np.nanmean([np.nanmean(v["seed_sd"]) for v in use.values()]))
    lev = dynamic_range(tc, floor=floor / max(abs(np.mean([v["slope"] for v in use.values()])), 1e-9),
                        k=RANGE_K, name="T_cross across models")
    r, p, n = _rho_p(tc, r4)
    v = carries_verdict([lev], value=r)
    parts.append(
        f"PRIMARY: rho(T_cross, rep_4) = {r:+.3f} (permutation p = {p:.4f}, n = {n}), against T*'s "
        f"{TSTAR_REFERENCE:+.3f} on the same target. {lev.reason}"
        + ("" if v.status == "DECIDED" else f" NOT DECIDABLE: {v.reason}"))
    sl = [v["slope"] for v in use.values()]
    rs, ps, _ = _rho_p(sl, r4)
    parts.append(f"SECOND (exploratory): rho(slope at crossing, rep_4) = {rs:+.3f} (p = {ps:.4f}).")
    both = [(v["T_cross"], v["t_star"]) for v in use.values() if isinstance(v["t_star"], (int, float))]
    if len(both) >= 4:
        rc, pc, nc = _rho_p([x[0] for x in both], [x[1] for x in both])
        parts.append(
            f"COINCIDENCE CHECK: rho(T_cross, T*) = {rc:+.3f} (p = {pc:.4f}, n = {nc}). "
            + ("They track each other closely, so these may be one measurement derived two ways --"
               " which would consolidate rather than extend the anchor."
               if abs(rc) > 0.8 else
               "They do not track closely, so T_cross is a differently-derived scalar rather than "
               "T* in other clothes."))
    parts.append(
        "READING: if T_cross predicts rep_4 comparably to T*, lambda_ca acquires an external use by "
        "the same move that made T* work -- a response rather than a level. If it does not, the "
        "structural lesson is confirmed on a second quantity and the one-token response family does "
        "not transfer at all. BOUNDARY: greedy-scoped target, one radius, one lattice size.")
    verdict = " ".join(parts)
    print(f"\n  -> {verdict}")
    res["analysis"] = dict(rows=rows, censored=censored, rho_tcross_rep4=round(r, 4),
                           perm_p=round(p, 4), n=n, tstar_reference=TSTAR_REFERENCE,
                           rho_slope_rep4=round(rs, 4), leverage=lev.block())
    res["verdict"] = verdict
    res["_analysis_provenance"] = stamp(__file__)
    res["_note"] = ("Applies F112's structural lesson to lambda_ca: not lambda at a temperature but "
                    "the temperature at which it crosses zero. Tests whether a derived response "
                    "predicts greedy degeneration as T* does, where the level does not.")


if __name__ == "__main__":
    main()
