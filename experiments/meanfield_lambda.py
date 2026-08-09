"""Can lambda_ca be DERIVED rather than detected? Annealed mean field, on the ladder.

THE IDEA. Every route to an explanandum so far tried to correlate lambda_ca with something
INTERNAL, and all three returned negative (F78 indeterminate, F79 underpowered, F80 null). The
alternative is to derive it from something simpler and still black-box. Damage spreading on a
lattice has a classical annealed mean-field theory -- Derrida & Pomeau's, the one that gives random
Boolean networks their order-chaos boundary from sensitivity alone -- and the token-lattice
analogue is direct.

A flipped site can only affect sites whose window contains it, and only when those are next
resampled. For the causal AR rule the window is the r cells to the LEFT, so site i sits in the
window of sites i+1..i+r: r children. When such a child is resampled under CRN, it becomes damaged
with probability

    s = P( sample differs | the window differs in exactly one position, shared uniform )

and a damaged site whose own window is clean HEALS, because identical windows plus a shared uniform
give identical draws. To first order, damage multiplies by r*s per sweep, so

    lambda_MF = log(r * s)        and criticality sits at   s = 1/r.

For r=2 that predicts the transition at s = 0.5.

WHY s IS EXACTLY COMPUTABLE, NOT ESTIMATED. Sampling is inverse-CDF against a shared uniform
(`ar_ca.sample_device`: `(cdf < u).sum()`), so for two conditionals p and q the disagreement
probability is a deterministic functional of the pair:

    s(p, q) = 1 - sum_v | [F_p(v-1), F_p(v)) INTERSECT [F_q(v-1), F_q(v)) |

No Monte-Carlo, no seed. Two forward passes per measurement.

THE LADDER, AND IT IS THE POINT. A mean-field prediction is worth nothing until its error is
calibrated on systems whose answer is known -- which is this project's founding rule, applied to a
theory instead of an estimator.

  RUNG 1 (DK, analytic).  On the p2=0 line a single-parent flip changes the emission probability
                          from 0 to p1 or back, so s = p1 exactly, and each site feeds 2 children:
                          mean field predicts damage criticality at p1 = 0.5. The literature puts
                          it at 0.8087 (Hinrichsen) / 0.801 (Zebende). Mean field is therefore
                          KNOWN to be wrong here, in a known direction and by a known amount --
                          it ignores that a damaged site's descendants overlap, so it overcounts
                          spreading and underestimates p_c. Measuring that error is the rung.
  RUNG 2 (ECA, exact).    For each of the 19 rules of known class, s is the rule table's average
                          Boolean sensitivity; mean field predicts damage survives iff 3*s > 1.
                          Compare against the classes F36 measured.
  RUNG 3 (the model).     s from forward passes at the run's own (r, T), lambda_MF = log(r*s),
                          against the lambda_ca already measured per checkpoint.

PRE-REGISTERED, before any model number:
  Primary       Does lambda_MF track lambda_ca across checkpoints -- Spearman over the developmental
                grid, and does the s = 1/r crossing land in the same bracket as the lambda_ca
                crossing (steps 256-512)?
  Calibration   Rungs 1 and 2 run FIRST and their errors are reported BEFORE rung 3 is read. A
                mean-field theory that misses DK's p_c by 38% cannot be quoted to two decimals on
                a language model, and the rungs say how much to discount.
  DEFLATIONARY, and registered as a real outcome: if s alone predicts lambda_ca to within the
                lambda seed floor, then the ring is redundant -- a few thousand forward passes give
                the same number as N*sweeps*B of them, and THE TOOL IS s. That is K1 one level
                deeper. A derived relation is an explanation rather than a demotion (temperature is
                "just" mean kinetic energy and thermometers still exist), but if it holds it must
                be said, and it must be said having been written down first.
  Expected failure mode: the annealed approximation ignores correlations and this lattice is
                emphatically correlated -- it settles into text-like states. A MEASURED discrepancy
                against a derived prediction is still far more informative than the current
                silence, and it is the natural home for F60's unexplained 7% bias.

Writes results/meanfield_lambda.json.
Usage:  caffeinate -dimsu .venv/bin/python -u experiments/meanfield_lambda.py
"""
import sys as _sys, pathlib as _pathlib
_ROOT = _pathlib.Path(__file__).resolve().parents[1]
_sys.path[:0] = [str(_ROOT / "src"), str(_ROOT / "experiments")]
import gc, json, os, time

os.environ.setdefault("HF_HOME", str(_ROOT / "hf_cache"))
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
import numpy as np
from ranking import rank as _rank
import torch

from provenance import stamp, rel
from lyapunov import run_ignited

OUT = str(_ROOT / "results" / "meanfield_lambda.json")
SCALE = _ROOT / "results" / "dev_transition_scale.json"
PHASE3 = _ROOT / "results" / "dev_transition_phase3.json"

MODEL = "EleutherAI/pythia-410m"
STEPS = [128, 256, 512, 1000, 2000, 4000]
R, T = 2, 0.7                  # the developmental grid's own geometry
N_CTX = 128                    # random windows per checkpoint
SEED = 20260805

# A CORRELATION NEEDS THE PREDICTOR TO MOVE. F93's gate, pointed at the predictor instead of the
# target: rho between two series is uninterpretable when one of them has no range to speak from.
# The predictor must span at least this fraction of what the target spans, in the target's units,
# before rho is quoted in either direction. This gate was NOT in the pre-registration -- see
# _posthoc_gate in the results -- and it is applied here because the registered primary asked for
# a Spearman without asking whether lambda_MF varies enough for one to mean anything.
RANGE_RATIO = 0.5


# ------------------------------------------------------------------ the estimator

def s_crn(p, q):
    """Exact P(inverse-CDF draws differ) for two distributions under a shared uniform.

    X = F_p^{-1}(u), Y = F_q^{-1}(u) with the SAME u and the same vocabulary order, which is what
    `ar_ca.sample_device` does. They agree exactly when u lands in the intersection of the two
    CDF cells for the same token, so

        P(X = Y) = sum_v |[F_p(v-1), F_p(v)) & [F_q(v-1), F_q(v))|

    computed as the total overlap of two interval partitions of [0,1]. Deterministic: no sampling.
    """
    cp = np.concatenate([[0.0], np.cumsum(p)])
    cq = np.concatenate([[0.0], np.cumsum(q)])
    cp[-1] = cq[-1] = 1.0
    lo = np.maximum(cp[:-1], cq[:-1])
    hi = np.minimum(cp[1:], cq[1:])
    return float(1.0 - np.clip(hi - lo, 0.0, None).sum())


def lambda_mf(r, s):
    """Annealed mean field: damage multiplies by r*s per sweep, so lambda = log(r*s)."""
    return float(np.log(max(r * s, 1e-12)))


# ---------------------------------------------------------------- rung 1: DK, analytic

def rung_dk():
    """On p2=0 a single-parent flip moves the emission probability between 0 and p1, so s = p1.

    Two children per site (s'_i depends on s_{i-1} XOR s_{i+1}), so mean field puts damage
    criticality at 2*p1 = 1. The published p_c is 0.8087 (Hinrichsen) / 0.801 (Zebende), so the
    error is known, large, and in the direction annealed mean field always errs: it ignores the
    overlap between a damaged site's descendants, overcounts spreading, and puts p_c too low.
    """
    from dk import ANCHORS
    mf_pc = 0.5
    pub = {k: ANCHORS[k]["p1"] for k in ANCHORS if "w18" in k}
    errs = {k: round((mf_pc - v) / v * 100, 1) for k, v in pub.items()}
    return dict(mf_critical_p1=mf_pc, published=pub, pct_error=errs,
                worst_pct_error=min(errs.values()),
                reading="mean field predicts damage criticality at p1=0.5 where the literature "
                        "puts it near 0.80 -- it is wrong by ~38%, low, exactly as annealed "
                        "theory errs when it ignores descendant overlap. Any lambda_MF on a "
                        "language model inherits an error of this order and must not be quoted "
                        "as if it were a measurement.")


# ------------------------------------------------------------- rung 2: ECA, exact

def rung_eca():
    """Average Boolean sensitivity of each rule table; mean field says damage survives iff 3s > 1.

    This is Derrida's parameter for random Boolean networks computed exactly rather than annealed,
    so it is the strongest form of the prediction, tested where the classes are known (F36).
    """
    try:
        eca = json.load(open(_ROOT / "results" / "eca_ordered_vs_rest.json"))
    except FileNotFoundError:
        return None
    # the mapping lives inside groups[*].rules as {rule: ignition_prob}
    classes = {int(rl): grp
               for grp in ("ordered", "edge", "chaotic")
               for rl in eca.get("groups", {}).get(grp, {}).get("rules", {})}
    if not classes:
        return dict(skipped="rule->class mapping not present in eca_ordered_vs_rest.json")
    rows = {}
    for rule, grp in sorted(classes.items()):
        bits = [(rule >> i) & 1 for i in range(8)]          # bits[(l,c,r) as 3-bit index]
        flips = 0
        for cfg in range(8):
            for b in range(3):
                if bits[cfg] != bits[cfg ^ (1 << b)]:
                    flips += 1
        s = flips / (8 * 3)                                  # average sensitivity per input bit
        rows[rule] = dict(group=grp, s=round(s, 4), mf_growth=round(3 * s, 3),
                          mf_survives=bool(3 * s > 1))
    ok = sum(1 for v in rows.values()
             if (v["group"] == "ordered") != v["mf_survives"])
    return dict(rules=rows, n=len(rows), correct=ok,
                accuracy=round(ok / max(len(rows), 1), 3),
                reading="mean field classifies a rule as damage-surviving iff 3s > 1; scored "
                        "against F36's ordered-vs-rest split, which is the only ECA distinction "
                        "the project claims")


# ------------------------------------------------------- rung 3: the model

def measure_s(model, tok, dev, r, T, n_ctx, rng, pool):
    """s at this (r, T): mean exact CRN disagreement over random windows with one token flipped."""
    vals = []
    for _ in range(n_ctx):
        win = [int(x) for x in rng.choice(pool, size=r)]
        pos = int(rng.integers(0, r))
        alt = list(win)
        while alt[pos] == win[pos]:
            alt[pos] = int(rng.choice(pool))
        with torch.no_grad():
            x = torch.tensor([win, alt], device=dev)
            lg = model(input_ids=x).logits[:, -1].float()
            pr = torch.softmax(lg / T, dim=-1).cpu().double().numpy()
        vals.append(s_crn(pr[0] / pr[0].sum(), pr[1] / pr[1].sum()))
    return float(np.mean(vals)), float(np.std(vals))


def lambda_measured():
    """lambda_ca per checkpoint from the developmental grid, F42 filter applied."""
    out = {}
    for path, key in ((SCALE, "size_m"), (PHASE3, None)):
        try:
            d = json.load(open(path))
        except FileNotFoundError:
            continue
        for v in d.get("runs", {}).values():
            if not isinstance(v, dict) or "lambda_ca" not in v or not run_ignited(v):
                continue
            if key and v.get(key) != 410:
                continue
            if v.get("N") not in (None, 48):
                continue
            out.setdefault(v["step"], []).append(v["lambda_ca"])
    return {k: (float(np.mean(v)), float(np.std(v)), len(v)) for k, v in out.items()}


def main():
    res = json.load(open(OUT)) if os.path.exists(OUT) else {"s": {}}
    res["_preregistration"] = dict(
        model=MODEL, steps=STEPS, r=R, T=T, n_ctx=N_CTX, seed=SEED,
        theory="annealed mean field: damage multiplies by r*s per sweep, lambda_MF = log(r*s), "
               "criticality at s = 1/r",
        estimator="s computed EXACTLY as the inverse-CDF disagreement probability between two "
                  "conditionals; deterministic, no sampling",
        primary="does lambda_MF track lambda_ca across checkpoints, and does the s = 1/r crossing "
                "land in the lambda_ca crossing bracket (256-512)?",
        calibration="rungs 1 (DK, known p_c) and 2 (ECA, known classes) run FIRST and their errors "
                    "are reported before rung 3 is read",
        deflationary="if s alone predicts lambda_ca within the lambda seed floor, the ring is "
                     "redundant and THE TOOL IS s -- registered as a real outcome, K1 one level "
                     "deeper",
        expected_failure="annealed theory ignores correlations and this lattice settles into "
                         "text-like states, so a quantitative miss is expected; a measured "
                         "discrepancy still beats silence")

    print("=== RUNG 1: Domany-Kinzel, where the answer is known ===", flush=True)
    res["rung1_dk"] = rung_dk()
    r1 = res["rung1_dk"]
    print(f"  mean-field p_c = {r1['mf_critical_p1']}, published {r1['published']}")
    print(f"  error {r1['pct_error']}  -> {r1['worst_pct_error']}%", flush=True)

    print("\n=== RUNG 2: elementary CA, classes known (F36) ===", flush=True)
    res["rung2_eca"] = rung_eca()
    r2 = res["rung2_eca"]
    if r2 and "rules" in r2:
        print(f"  mean field agrees with the ordered-vs-rest split on {r2['correct']}/{r2['n']} "
              f"rules ({r2['accuracy']:.0%})", flush=True)
    else:
        print(f"  {r2}", flush=True)

    print(f"\n=== RUNG 3: {MODEL}, r={R}, T={T} ===", flush=True)
    from transformers import AutoTokenizer, AutoModelForCausalLM
    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(MODEL)
    for st in STEPS:
        k = f"step{st}"
        if k in res["s"]:
            continue
        t0 = time.time()
        m = AutoModelForCausalLM.from_pretrained(MODEL, revision=k).eval().to(
            dev, torch.float16 if dev != "cpu" else torch.float32)
        V = int(getattr(m.config, "vocab_size", len(tok)))
        sp = {i for i in (tok.bos_token_id, tok.eos_token_id, tok.pad_token_id,
                          tok.unk_token_id) if i is not None}
        pool = np.array([i for i in range(min(V, len(tok))) if i not in sp], np.int64)
        s, sd = measure_s(m, tok, dev, R, T, N_CTX, np.random.default_rng(SEED), pool)
        res["s"][k] = dict(step=st, s=round(s, 5), s_sd=round(sd, 5),
                           lambda_mf=round(lambda_mf(R, s), 5), secs=round(time.time() - t0, 1))
        print(f"  step{st:<7} s={s:.4f}  r*s={R*s:.3f}  lambda_MF={res['s'][k]['lambda_mf']:+.4f} "
              f"({time.time()-t0:.0f}s)", flush=True)
        json.dump(res, open(OUT, "w"), indent=1)
        del m
        try: torch.mps.empty_cache()
        except Exception: pass
        gc.collect()

    analyse(res)
    json.dump(res, open(OUT, "w"), indent=1)
    print("\nwrote", rel(OUT))


def analyse(res):
    meas = lambda_measured()
    rows = []
    for k, v in res["s"].items():
        st = v["step"]
        if st in meas:
            rows.append(dict(step=st, s=v["s"], lambda_mf=v["lambda_mf"],
                             lambda_ca=round(meas[st][0], 4), lambda_sd=round(meas[st][1], 4),
                             n_seeds=meas[st][2]))
    rows.sort(key=lambda x: x["step"])
    print(f"\n  {'step':>7} {'s':>7} {'r*s':>6} {'lambda_MF':>10} {'lambda_ca':>10} {'seed sd':>8}")
    for r in rows:
        print(f"  {r['step']:>7} {r['s']:7.4f} {R*r['s']:6.3f} {r['lambda_mf']:+10.4f} "
              f"{r['lambda_ca']:+10.4f} {r['lambda_sd']:8.4f}")

    parts = []
    r1 = res["rung1_dk"]
    parts.append(
        f"CALIBRATION FIRST (the rungs, read before the model). Rung 1: mean field puts DK's "
        f"damage criticality at p1 = {r1['mf_critical_p1']} where the literature puts it near "
        f"0.80 -- an error of {r1['worst_pct_error']}%, low, which is the direction annealed "
        f"theory always errs because it ignores descendant overlap. ")
    r2 = res.get("rung2_eca") or {}
    if "accuracy" in r2:
        parts.append(
            f"Rung 2: on {r2['n']} elementary rules of known class it recovers the "
            f"ordered-vs-rest split {r2['accuracy']:.0%} of the time. ")
    parts.append(
        "So lambda_MF is a SHAPE prediction, not a value prediction, and nothing below is quoted "
        "as a measurement.")

    if len(rows) >= 4:
        s_arr = np.array([r["s"] for r in rows])
        mf = np.array([r["lambda_mf"] for r in rows])
        ca = np.array([r["lambda_ca"] for r in rows])
        rk = lambda x: _rank(x)
        rho = float(np.corrcoef(rk(mf), rk(ca))[0, 1])
        # n is 6, so the null is enumerable exactly -- 720 permutations, no sampling error.
        from itertools import permutations
        rmf, rca = rk(mf), rk(ca)
        null = [np.corrcoef(np.array(pm), rca)[0, 1] for pm in permutations(rmf)]
        p_perm = float(np.mean(np.abs(np.array(null)) >= abs(rho) - 1e-12))
        mf_span, ca_span = float(mf.max() - mf.min()), float(ca.max() - ca.min())
        span_ratio = mf_span / ca_span if ca_span else 0.0
        rho_usable = bool(span_ratio >= RANGE_RATIO)
        # where does s cross 1/r, and where does lambda_ca cross zero?
        s_cross = next((f"{rows[i]['step']}-{rows[i+1]['step']}" for i in range(len(rows) - 1)
                        if s_arr[i] < 1.0 / R <= s_arr[i + 1]), None)
        ca_cross = next((f"{rows[i]['step']}-{rows[i+1]['step']}" for i in range(len(rows) - 1)
                         if ca[i] < 0 <= ca[i + 1]), None)
        floor = float(np.mean([r["lambda_sd"] for r in rows])) / np.sqrt(8)
        resid = float(np.mean(np.abs(mf - ca)))
        deflates = resid <= 2 * floor
        parts.append(
            f"RUNG 3, and the two legs of the registered primary must be read differently. "
            f"THE CROSSING LEG IS DECIDED: "
            f"s crosses 1/r = {1/R:.2f} in bracket {s_cross or 'NEVER on this grid'}; lambda_ca "
            f"crosses zero in {ca_cross or 'no bracket'}. "
            + ("THE BRACKETS AGREE -- the conditional's single-token sensitivity crosses 1/r in "
               "the same interval where the ring's exponent crosses zero, which is the derivation "
               "the three internal routes failed to produce, stated in black-box vocabulary."
               if s_cross and s_cross == ca_cross else
               "THE BRACKETS DO NOT AGREE, so the mean-field crossing is not the lambda_ca "
               "crossing on this grid; the theory does not locate the transition even if it "
               "tracks its shape. Saturation is a real answer to 'does it cross', so this leg is "
               "genuinely falsified, not merely unsupported."))
        parts.append(
            f"THE CORRELATIONAL LEG IS NOT DECIDABLE, and saying so is the point: lambda_MF spans "
            f"{mf_span:.3f} against lambda_ca's {ca_span:.3f}, a ratio of {span_ratio:.2f} below "
            f"the {RANGE_RATIO} a correlation needs to have leverage (s itself spans {s_arr.max()-s_arr.min():.3f} "
            f"on [0,1] and sits {min(abs(s_arr.min()-1/R), abs(s_arr.max()-1/R)):.2f} from 1/r "
            f"throughout). rho = {rho:+.3f}, exact permutation p = {p_perm:.2f} over all "
            f"{len(rows)}! orderings. It is quoted here ONLY to record that it must not be quoted "
            f"in either direction -- neither as weak evidence against the theory nor, had it come "
            f"out positive, for it. This is F93's defect recurring inside the very finding that "
            f"eliminated a hypothesis: a statistically-shaped criterion registered against a "
            f"quantity with no room to vary. The gate is POST-HOC here and marked as such."
            if not rho_usable else
            f"rho = {rho:+.3f} (exact permutation p = {p_perm:.2f}), and lambda_MF spans "
            f"{span_ratio:.2f} of lambda_ca's range, so the correlational leg has leverage.")
        parts.append(
            f"DEFLATION CHECK (registered): mean |lambda_MF - lambda_ca| = {resid:.4f} against a "
            f"lambda seed floor of {floor:.4f}. "
            + ("s reproduces lambda_ca WITHIN the floor, so the ring is redundant for this "
               "quantity and THE TOOL IS s -- a few thousand forward passes replace N*sweeps*B of "
               "them. Registered as a real outcome before the run: a derived relation is an "
               "explanation, not a demotion, but it must be reported as found."
               if deflates else
               "s does NOT reproduce lambda_ca within the floor, so the ring is not redundant: "
               "mean field captures shape without capturing value, which is what the DK rung "
               "predicted it would do."))
        res["analysis"] = dict(rows=rows, rho=round(rho, 3), rho_perm_p=round(p_perm, 4),
                               rho_usable=rho_usable, s_crossing=s_cross,
                               lambda_crossing=ca_cross, residual=round(resid, 4),
                               lambda_seed_floor=round(floor, 4), deflates=bool(deflates),
                               s_span=round(float(s_arr.max() - s_arr.min()), 4),
                               lambda_mf_span=round(mf_span, 4), lambda_ca_span=round(ca_span, 4),
                               span_ratio=round(span_ratio, 3), range_ratio_gate=RANGE_RATIO)
        res["_posthoc_gate"] = dict(
            gate="rho is quoted only if lambda_MF spans >= RANGE_RATIO of lambda_ca's range",
            registered=False,
            why="The pre-registered primary had two legs, a crossing and a Spearman, and only the "
                "crossing leg carried a range check. The Spearman leg was registered against a "
                "predictor that turned out to be saturated, which is exactly the defect F93 "
                "found and F89 before it. Applying the gate after the fact cannot rescue the "
                "leg -- it can only stop the number being read as evidence, which is what it "
                "does here. The crossing leg is unaffected: it was decided by saturation, and "
                "saturation is an answer.",
            affects="the correlational sub-claim only; the elimination rests on the crossing leg "
                    "and on the measured flatness of s, neither of which needs rho.")
    verdict = " ".join(parts)
    print(f"\n  -> {verdict}")
    res["verdict"] = verdict
    res["_analysis_provenance"] = stamp(__file__)
    res["_note"] = (
        "Derives lambda_ca from the conditional's single-token sensitivity via annealed mean field "
        "(Derrida-Pomeau), tested on the project's own ladder: DK first (known p_c, so the "
        "theory's error is calibrated before use), then ECA (known classes), then the model. s is "
        "computed EXACTLY as the inverse-CDF disagreement probability between two conditionals -- "
        "the same coupling ar_ca samples with -- so it is deterministic and needs two forward "
        "passes. The deflationary outcome (if s alone reproduces lambda_ca within the seed floor, "
        "the ring is redundant and the tool is s) was registered before the run as a real "
        "possibility, which is K1 one level deeper. WHERE THE RESIDUAL LIVES, sharpened: healing "
        "is not a free rate in this system. A damaged site whose window is clean redraws the same "
        "token deterministically -- same context, same uniform -- so healing is forced to exactly "
        "1 by the same CRN property that makes twins diverge by zero. It is therefore not a "
        "per-site probability at all but a property of whether clean windows RE-FORM around "
        "damaged sites. Creation is now measured flat. So the two terms collapse into one object: "
        "THE SPATIAL STRUCTURE OF THE DAMAGE CLOUD, which is precisely what annealed theory "
        "discards by construction, since its premise is that sites are independently "
        "re-randomised each step. Rung 2 already contains a validated instance: the single rule "
        "mean field misses out of 19 is 232, majority -- the canonical CANALIZING function, whose "
        "response to perturbation is nonlinear in how many inputs are perturbed and which "
        "per-site sensitivity averages away by construction.")


if __name__ == "__main__":
    main()
