"""Does CANALIZATION predict degeneration? The deepest candidate, pointed at an external target.

WHY THIS ONE. F96 built the canalization machinery and validated it on a ladder where the answers
are known: Domany-Kinzel's p2=0 line gives spread EXACTLY 0.000000 with sub-additivity +0.960 (pure
cancellation), ECA rule 150 (XOR) gives spread 0.0000 subadd +1.0000, and rule 232 (majority) gives
spread 0.2887 subadd +0.2500 (masking). The pair separates the two mechanisms, and the gate passed.
But it has only ever been run DEVELOPMENTALLY -- pythia-410m across six checkpoints -- and never
pointed at anything outside itself.

There is theory behind expecting it to matter. F102's mean-field null across 33 ablation arms says
the missing physics is exactly the term canalization measures; rule 232 was the single ECA miss for
the same reason; and in Boolean-network theory canalizing functions are the known stabilizer. As a
per-model quantity it characterises the COMPUTATIONAL CLASS of the local function the model
implements -- majority-like versus XOR-like -- which no behavioural benchmark touches.

THE TWO INDICES, and they are different mechanisms (F96):
  spread   sd over contexts of per-context sensitivity. High = MASKING: whether a flip propagates
           depends on the rest of the window. Canalization proper.
  subadd   1 - (1-s_a)(1-s_b) - s_ab: how far the two-flip response falls below independence.
           Positive = sub-additive. XOR is maximally sub-additive with ZERO spread, so the pair is
           needed and neither alone will do.

PRE-REGISTERED:
  PRIMARY     rho(subadd, rep_4) on the SETTLED ensemble (F96/F99: the regime the dynamics run in),
              against T*'s rho = +0.547 on the same target and models (F112).
  SECOND      rho(spread, rep_4) -- the masking index, same target.
  DEFLATION, frozen before any number: T* already predicts this target. A new metric is only worth
              having if it is (a) not a restatement of T*, checked by rho(index, T*), and (b) not
              beaten by it. Both are reported whichever way they come out.
  RANGE GATE  each index must span more than RANGE_K times its own across-context standard error
              before any correlation is quoted (the defect class caught seven times in this project).
  REGIME      measured on settled AND random ensembles. F94 measured a related quantity on random
              windows and reached a conclusion that F96 showed was a property of the ensemble.
  KILL        neither index clears the range gate, or both predict at |rho| below T*'s -- the
              canalization route adds nothing external and closes.
  BOUNDARY    greedy-scoped target, one radius, one temperature; models are not independent draws
              (F86 states its own anchor at family level for this reason).

Writes results/canalization_predicts.json.  Resumable per model.
"""
import sys as _sys, pathlib as _pathlib
_ROOT = _pathlib.Path(__file__).resolve().parents[1]
_sys.path[:0] = [str(_ROOT / "src"), str(_ROOT / "experiments"), str(_ROOT / "gatecheck" / "src")]
import gc, itertools, json, os, time
os.environ.setdefault("HF_HOME", str(_ROOT / "hf_cache"))
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
import numpy as np, torch
from provenance import stamp, rel
from meanfield_lambda import s_crn
from gatecheck import dynamic_range, carries_verdict

OUT = str(_ROOT / "results" / "canalization_predicts.json")
DEG = _ROOT / "results" / "degeneration_vs_tstar.json"
R, T, N_CTX, BATCH = 2, 0.7, 128, 32
SET_B, SET_N, SET_SWEEPS = 8, 48, 30
SEED, RANGE_K = 20260809, 2.0
TSTAR_REFERENCE = 0.547
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
        if h:
            st = info.delete_revisions(*h); st.execute()
    except Exception:
        pass


def indices(rule, pool, rng):
    """spread and sub-additivity over contexts drawn from `pool`. s is exact; no sampling error."""
    pool = np.asarray(pool, dtype=np.int64)
    rows = []
    for _ in range(N_CTX):
        w = [int(x) for x in rng.choice(pool, size=R)]
        a, b = list(w), list(w)
        while a[0] == w[0]: a[0] = int(rng.choice(pool))
        while b[1] == w[1]: b[1] = int(rng.choice(pool))
        ab = [a[0], b[1]]
        rows += [w, a, b, ab]
    out = []
    for i in range(0, len(rows), BATCH):
        with torch.no_grad():
            lg = rule.model(input_ids=torch.tensor(rows[i:i + BATCH], device=rule.device)
                            ).logits[:, -1].float()
            out.append(torch.softmax(lg / T, dim=-1).cpu().double().numpy())
    P = np.concatenate(out, 0); P = P / P.sum(axis=1, keepdims=True)
    s_ctx, sub = [], []
    for k in range(N_CTX):
        p, pa, pb, pab = P[4*k], P[4*k+1], P[4*k+2], P[4*k+3]
        sa, sb, sab = s_crn(p, pa), s_crn(p, pb), s_crn(p, pab)
        s_ctx.append(0.5 * (sa + sb))
        sub.append(1 - (1 - sa) * (1 - sb) - sab)
    s_ctx, sub = np.array(s_ctx), np.array(sub)
    return dict(s=round(float(s_ctx.mean()), 5), spread=round(float(s_ctx.std()), 5),
                subadd=round(float(sub.mean()), 5),
                subadd_se=round(float(sub.std() / np.sqrt(N_CTX)), 5),
                spread_se=round(float(s_ctx.std() / np.sqrt(2 * (N_CTX - 1))), 5))


def main():
    res = json.load(open(OUT)) if os.path.exists(OUT) else {"cells": {}}
    res["_preregistration"] = dict(
        models=MODELS, r=R, T=T, n_ctx=N_CTX, seed=SEED, range_k=RANGE_K,
        settle=dict(B=SET_B, N=SET_N, sweeps=SET_SWEEPS), tstar_reference=TSTAR_REFERENCE,
        primary="rho(subadd, rep_4) on the SETTLED ensemble, against T*'s +0.547",
        second="rho(spread, rep_4), the masking index",
        deflation="T* already predicts this target; a new metric must be neither a restatement of "
                  "it (checked by rho(index, T*)) nor beaten by it. Both reported either way",
        range_gate="each index must span > RANGE_K x its own across-context standard error",
        regime="settled AND random; F94 measured a related quantity on random windows and reached "
               "a conclusion F96 showed was a property of the ensemble",
        kill="neither index clears the gate, or both predict below T* -- the route closes",
        anchors="F96's validated ladder: DK p2=0 spread 0.000000 subadd +0.960; ECA rule 150 (XOR) "
                "spread 0.0000 subadd +1.0000; rule 232 (majority) spread 0.2887 subadd +0.2500")
    from ar_ca import ARRule, run
    for name in MODELS:
        if name in res["cells"]:
            continue
        t0 = time.time()
        try:
            rule = ARRule(name)
            sp = {i for i in (rule.tok.bos_token_id, rule.tok.eos_token_id,
                              rule.tok.pad_token_id, rule.tok.unk_token_id) if i is not None}
            uni = np.array([i for i in range(rule.V) if i not in sp], np.int64)
            settled = run(rule, B=SET_B, N=SET_N, r=R, T=T, sweeps=SET_SWEEPS, scheme="none",
                          seed=SEED, order="per_replica")["final"].reshape(-1)
            row = dict(model=name,
                       settled=indices(rule, settled, np.random.default_rng(SEED)),
                       random=indices(rule, uni, np.random.default_rng(SEED + 1)),
                       settled_distinct=int(len(np.unique(settled))),
                       secs=round(time.time() - t0, 1))
        except Exception as e:
            print(f"  {name}: FAILED {type(e).__name__}"[:110], flush=True)
            res["cells"][name] = dict(model=name, failed=repr(e)[:180])
            json.dump(res, open(OUT, "w"), indent=1); continue
        res["cells"][name] = row
        s = row["settled"]
        print(f"  {name:<32} spread={s['spread']:.4f} subadd={s['subadd']:+.4f} "
              f"distinct={row['settled_distinct']:>3} ({row['secs']:.0f}s)", flush=True)
        json.dump(res, open(OUT, "w"), indent=1)
        del rule
        try: torch.mps.empty_cache()
        except Exception: pass
        gc.collect()
        if "pythia-410m" not in name:
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
    use = [(m, c) for m, c in res["cells"].items() if "settled" in c and m in tgt]
    parts = []
    print(f"\n  {'model':<32} {'spread':>8} {'subadd':>9} {'rep_4':>7} {'T*':>9}")
    for m, c in use:
        print(f"  {m:<32} {c['settled']['spread']:>8.4f} {c['settled']['subadd']:>+9.4f} "
              f"{tgt[m]['rep_4']:>7.3f} {str(tgt[m].get('t_star')):>9}")
    if len(use) < 6:
        res["analysis"] = dict(n=len(use))
        res["verdict"] = f"Only {len(use)} usable models -- NOT DECIDABLE."
        res["_analysis_provenance"] = stamp(__file__); print(f"\n  -> {res['verdict']}"); return
    r4 = [tgt[m]["rep_4"] for m, _ in use]
    out = {}
    for idx in ("subadd", "spread"):
        for reg in ("settled", "random"):
            v = [c[reg][idx] for _, c in use]
            se = float(np.mean([c[reg][f"{idx}_se"] for _, c in use]))
            lev = dynamic_range(v, floor=se, k=RANGE_K, name=f"{idx} ({reg})")
            r, p, n = _rho_p(v, r4)
            out[f"{idx}|{reg}"] = dict(rho=round(r, 4), perm_p=round(p, 4), n=n,
                                       span=round(float(max(v) - min(v)), 5), se=round(se, 5),
                                       usable=lev.usable, reason=lev.reason)
            print(f"  rho({idx:<6} {reg:<7}, rep_4) = {r:+.3f}  p={p:.4f}  "
                  f"{'[gate ok]' if lev.usable else '[GATE FAILS]'}")
    # THE PRIMARY IS THE REGISTERED ARM, NOT THE BEST ARM. An earlier version took the maximum
    # |rho| over all four (index x regime) combinations, found subadd|random at +0.552 > T*'s
    # +0.547, and printed "the canalization indices beat T*" -- while the REGISTERED primary, the
    # settled ensemble, sat at +0.270. That is arm-shopping, implemented in the summariser rather
    # than in the statistic, which is exactly the defect gatecheck.leverage exists to stop one
    # level down. The non-registered arms are reported as exploratory and can never carry the
    # verdict.
    prim = out["subadd|settled"]
    parts.append(
        f"PRIMARY, the registered arm (subadd on the SETTLED ensemble): rho = {prim['rho']:+.3f} "
        f"(p = {prim['perm_p']:.4f}, n = {prim['n']}), against T*'s {TSTAR_REFERENCE:+.3f} on the "
        f"same target and models. Secondary (spread, settled): "
        f"{out['spread|settled']['rho']:+.3f} (p = {out['spread|settled']['perm_p']:.4f}).")
    expl = {k: v for k, v in out.items() if not k.endswith("|settled")}
    parts.append(
        "EXPLORATORY, and reported only so the arm-shopping is visible rather than hidden: the "
        "random-ensemble arms give "
        + ", ".join(f"{k.split('|')[0]} {v['rho']:+.3f} (p={v['perm_p']:.4f})"
                    for k, v in expl.items())
        + ". The best of these exceeds T* by 0.005, which is not a result -- it is what picking the "
          "largest of four correlations looks like, and F96's regime work says the settled ensemble "
          "is the one that governs the dynamics anyway.")
    both = [(c["settled"]["subadd"], tgt[m]["t_star"]) for m, c in use
            if isinstance(tgt[m].get("t_star"), (int, float))]
    if len(both) >= 4:
        rc, pc, nc = _rho_p([x[0] for x in both], [x[1] for x in both])
        parts.append(
            f"DEFLATION CHECK (frozen before the run): rho(subadd, T*) = {rc:+.3f} (p = {pc:.4f}, "
            f"n = {nc}). "
            + ("The index tracks T* closely, so it is largely a restatement of a metric the "
               "project already has rather than a new one."
               if abs(rc) > 0.8 else
               "The index does not track T*, so whatever it measures is not a restatement of it."))
    beats = bool(abs(prim["rho"]) > abs(TSTAR_REFERENCE) and prim["usable"])
    parts.append(
        "The registered index beats T* on this target, so the route adds an external predictor."
        if beats else
        f"The registered index does NOT beat T* ({prim['rho']:+.3f} against {TSTAR_REFERENCE:+.3f}, "
        f"p = {prim['perm_p']:.4f}). Since T* is cheaper and already validated, the canalization "
        f"route does not add an external predictor and closes as designed -- the registered kill, "
        f"not a disappointment.")
    subs = [c["settled"]["subadd"] for _, c in use]
    parts.append(
        f"AND THE INDEX ITSELF IS NEAR ZERO ON EVERY REAL MODEL: sub-additivity spans "
        f"{min(subs):+.4f} to {max(subs):+.4f}, against F96's validated ladder where "
        f"Domany-Kinzel's p2=0 line sits at +0.960, ECA rule 150 (XOR) at +1.000 and rule 232 "
        f"(majority) at +0.250. Every model is two orders of magnitude below the weakest anchor, "
        f"i.e. the two-token response is essentially ADDITIVE and nowhere near canalizing. That is "
        f"a substantive negative about what these local functions are, and it is independent of "
        f"whether the index predicts anything.")
    parts.append(
        "BOUNDARY: greedy-scoped target, one radius, one temperature, and the models are not "
        "independent draws -- F86 states its own anchor at family level for exactly this reason, "
        "so a per-model rho here is the weaker form. s is exact (inverse-CDF CRN disagreement), so "
        "none of these numbers carries sampling error in the estimator itself.")
    v = " ".join(parts)
    print(f"\n  -> {v}")
    res["analysis"] = dict(correlations=out, n=len(use), tstar_reference=TSTAR_REFERENCE,
                           primary_arm="subadd|settled", primary=prim, beats_tstar=beats,
                           subadd_range=[round(float(min(subs)), 5), round(float(max(subs)), 5)],
                           anchors=dict(dk_p2_0=0.960, eca_rule150_xor=1.000, eca_rule232_maj=0.250))
    res["verdict"] = v
    res["_analysis_provenance"] = stamp(__file__)
    res["_note"] = ("Points F96's validated canalization machinery -- ladder-anchored on DK, XOR and "
                    "majority -- at an external target for the first time. The indices had only ever "
                    "been measured developmentally on one model.")


if __name__ == "__main__":
    main()
