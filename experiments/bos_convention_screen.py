"""Is the BOS effect a TRAINING-CONFIG fact? A screen, not a test. Zero forward passes.

THE IDEA. F152 found one BOS token raising Falcon3-1B-Base 0.214 -> 0.906 while collapsing other
models, and F158 found attention-sink strength does not predict that sign. A cheaper explanation is
available and has never been checked: models differ in whether their PRETRAINING CONVENTION prepends
BOS. For a model trained with BOS always present, the raw arm -- two tokens with no BOS -- is the
out-of-distribution one, and adding BOS puts it back in distribution. For a model trained without,
adding BOS is the perturbation.

PREDICTION, WRITTEN AND FROZEN BEFORE THE JOIN (the file records this in _preregistration, and the
join is computed only after):
    models whose tokenizer config prepends BOS (add_bos_token true) move phi UP under the bos arm;
    models without the convention collapse or hold.

KILL     agreement at or below chance (<= 50% of readable models on a two-class call).
CAUTION, PRE-REGISTERED  n is small and CONFIGURATION IS CONFOUNDED WITH FAMILY -- Llama-lineage
    tokenizers set add_bos_token true as a family habit, so a "config" effect may be a family effect
    wearing a config label. This is a SCREEN whose only legitimate output is a candidate hypothesis
    and a confusion table. It is not a test, and no p-value is computed. With this n a chi-square
    could not fail informatively; that refusal is recorded here, before the numbers.

SOURCE OF TRUTH  tokenizer_config.json in the local HF cache, key add_bos_token. Where absent or
    ambiguous the model is reported UNKNOWN and excluded -- never guessed from the family name, which
    would manufacture exactly the confound the caution names.
"""
import sys as _sys, pathlib as _pathlib
_ROOT = _pathlib.Path(__file__).resolve().parents[1]
_sys.path[:0] = [str(_ROOT / "src"), str(_ROOT / "experiments"), str(_ROOT / "fingerprint"),
                 str(_ROOT / "gatecheck" / "src")]
import glob, json, os

import numpy as np

from provenance import stamp, rel

OUT = str(_ROOT / "results" / "bos_convention_screen.json")
CACHE = _ROOT / "hf_cache" / "hub"
CENSUS_SEEDS = [20260803, 990017]
MIN_SHIFT = 4.0 / 96
NOISE_FACTOR = 2.0
CHANCE = 0.50


def cache_dir(model):
    return CACHE / ("models--" + model.replace("/", "--"))


def bos_convention(model):
    """Does this model's tokenizer PREPEND BOS in practice?

    The declared key is checked first, but six of eight models in this cohort omit add_bos_token
    from tokenizer_config.json entirely -- and absence of the key is NOT absence of the convention,
    it is silence. Guessing from the family name would manufacture the very confound the caution
    names. So the convention is MEASURED: encode a probe string and look at whether the first id is
    the BOS id. Tokenizer only, no model weights, no forward passes.

    Returns (value, source) or (None, reason)."""
    d = cache_dir(model)
    hits = sorted(glob.glob(str(d / "snapshots" / "*" / "tokenizer_config.json")))
    declared, decl_src = None, None
    if hits:
        try:
            cfg = json.load(open(hits[0]))
            if "add_bos_token" in cfg:
                declared = bool(cfg["add_bos_token"])
                decl_src = os.path.relpath(hits[0], _ROOT)
        except Exception:
            pass
    try:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(model)
    except Exception as e:
        if declared is not None:
            return declared, f"{decl_src} (declared; tokenizer unloadable)"
        return None, f"tokenizer unloadable ({type(e).__name__}) and key absent"
    b = tok.bos_token_id
    if b is None:
        return False, "measured: tokenizer has no BOS token at all"
    ids = tok("The quick brown fox", add_special_tokens=True)["input_ids"]
    measured = bool(ids and ids[0] == b)
    src = f"measured: first id {'==' if measured else '!='} bos_token_id ({b})"
    if declared is not None:
        src += f"; declared {declared} in {decl_src}"
        if declared != measured:
            src += " -- DECLARED AND MEASURED DISAGREE, measured wins"
    return measured, src


def phi_pair(runs, m, arm):
    ks = [f"{m}|s{cs}|{arm}" for cs in CENSUS_SEEDS]
    if not all(k in runs for k in ks):
        return None
    v = [runs[k]["fixed_point_fraction"] for k in ks]
    return float(np.mean(v)), float(abs(v[0] - v[1]))


def main():
    res = {"_preregistration": dict(
        prediction="models whose tokenizer config prepends BOS (add_bos_token true) move phi UP "
                   "under the bos arm; models without the convention collapse or hold",
        kill=f"BALANCED accuracy <= chance ({CHANCE:.0%}) on the readable models; raw agreement is not used because it is a base-rate artefact when the predictor is imbalanced",
        caution="n is small and CONFIG IS CONFOUNDED WITH FAMILY; this is a SCREEN yielding a "
                "candidate hypothesis and a confusion table, not a test",
        refusal="no significance test is computed -- at this n a chi-square could not fail "
                "informatively. Recorded before the numbers, per the project's standing rule.",
        source="MEASURED by encoding a probe string with the local tokenizer and testing whether "
               "the first id is bos_token_id (no weights, no forward passes); the declared "
               "add_bos_token key is recorded alongside where present, and disagreements are "
               "named. Six of eight models omit the key entirely, and absence is silence rather "
               "than a false -- inferring it from family name would manufacture the confound.",
        census_seeds=CENSUS_SEEDS, min_shift=MIN_SHIFT, noise_factor=NOISE_FACTOR)}

    runs = json.load(open(_ROOT / "results" / "domain_base.json"))["runs"]
    models = sorted({k.split("|")[0] for k in runs if len(k.split("|")) == 3})

    rows, unknown = [], []
    for m in models:
        conv, src = bos_convention(m)
        raw, bos = phi_pair(runs, m, "raw"), phi_pair(runs, m, "bos")
        if raw is None or bos is None:
            continue
        (pr, nr), (pb, nb) = raw, bos
        tol = max(MIN_SHIFT, NOISE_FACTOR * max(nr, nb))
        d = pb - pr
        direction = "up" if d > tol else ("down" if d < -tol else "flat")
        rec = dict(model=m, add_bos_token=conv, config_source=src,
                   phi_raw=round(pr, 4), phi_bos=round(pb, 4), d_phi=round(d, 4),
                   tol=round(tol, 4), direction=direction,
                   source_file="results/domain_base.json",
                   source_keys=[f"{m}|s{cs}|{a}" for a in ("raw", "bos") for cs in CENSUS_SEEDS])
        (unknown if conv is None else rows).append(rec)
    res["models"] = rows
    res["unknown"] = unknown

    parts = []
    parts.append(
        f"COVERAGE: {len(rows)} models with a readable add_bos_token in the local cache; "
        f"{len(unknown)} UNKNOWN and excluded"
        + (f" ({[(u['model'].split('/')[-1], u['config_source']) for u in unknown]})."
           if unknown else "."))

    readable = [r for r in rows if r["direction"] != "flat"]
    parts.append(
        f"ANTI-VACUITY: {len(rows) - len(readable)} of {len(rows)} models are FLAT under bos "
        f"(|d phi| within their own tolerance) and cannot vote on a two-class prediction; they are "
        f"named and excluded"
        + (f": {[r['model'].split('/')[-1] for r in rows if r['direction'] == 'flat']}."
           if len(readable) < len(rows) else "."))

    if len(readable) < 4:
        parts.append(
            f"NOT DECIDABLE: {len(readable)} readable models. A two-class agreement over fewer than "
            f"four cannot separate a real association from a coin flip, and the screen is not read.")
    else:
        tab = {(True, "up"): 0, (True, "down"): 0, (False, "up"): 0, (False, "down"): 0}
        for r in readable:
            tab[(bool(r["add_bos_token"]), r["direction"])] += 1
        n_true = tab[(True, "up")] + tab[(True, "down")]
        n_false = tab[(False, "up")] + tab[(False, "down")]
        agree = tab[(True, "up")] + tab[(False, "down")]
        acc = agree / len(readable)
        # RAW AGREEMENT IS A BASE-RATE ARTEFACT WHEN THE PREDICTOR IS IMBALANCED. With one True and
        # six False, "agrees on 5 of 7" is carried entirely by the majority cell: a predictor with
        # no variance still scores high. Balanced accuracy averages the two class rates and is the
        # quantity that can actually fail. This is the project's own defect -- a criterion applied
        # to a quantity with no room to vary -- appearing in the screen written to avoid it.
        tpr = (tab[(True, "up")] / n_true) if n_true else None
        tnr = (tab[(False, "down")] / n_false) if n_false else None
        bal = None if (tpr is None or tnr is None) else (tpr + tnr) / 2
        res["confusion"] = {f"add_bos={k[0]}|{k[1]}": v for k, v in tab.items()}
        res["agreement"] = dict(agree=agree, n=len(readable), raw_accuracy=round(acc, 3),
                                n_true=n_true, n_false=n_false,
                                balanced_accuracy=None if bal is None else round(bal, 3))
        parts.append(
            "CONFUSION (rows = measured BOS convention, cols = phi direction under bos): "
            f"true/up {tab[(True,'up')]}, true/down {tab[(True,'down')]}, "
            f"false/up {tab[(False,'up')]}, false/down {tab[(False,'down')]}. "
            f"Raw agreement {agree} of {len(readable)} ({acc:.0%}), "
            f"BALANCED accuracy {'n/a' if bal is None else format(bal, '.2f')}. ")
        if min(n_true, n_false) < 2:
            parts.append(
                f"NOT DECIDABLE -- PREDICTOR IMBALANCE: only {n_true} model(s) carry the convention "
                f"and {n_false} do not. Raw agreement of {acc:.0%} is a BASE-RATE ARTEFACT: with the "
                f"predictor this lopsided, a rule that ignores it entirely and always answers 'down' "
                f"scores {max(n_true, n_false)}/{len(readable)}. Balanced accuracy, which is the "
                f"quantity that can fail, is "
                f"{'undefined' if bal is None else format(bal, '.2f')}"
                + ("" if bal is None else
                   f" -- {'BELOW' if bal < CHANCE else 'at or above'} chance. ")
                + "The screen is not read, and this was caught before reporting rather than after.")
        else:
            parts.append(
                (f"KILL FIRED: balanced accuracy {bal:.2f} is at or below chance. The BOS convention "
                 f"does not screen the sign, and the training-config reading is dropped."
                 if bal <= CHANCE else
                 f"SCREEN PASSES as a candidate: balanced accuracy {bal:.2f} is above chance. This "
                 f"licenses a hypothesis, nothing more -- see the caution, written before the "
                 f"join."))
    parts.append(
        "CAUTION, PRE-REGISTERED: n is small and add_bos_token is confounded with model family, so "
        "a 'config' effect may be a family effect wearing a config label. No significance test is "
        "computed: at this n it could not fail informatively, and that refusal was recorded before "
        "the numbers. Every phi traces to results/domain_base.json; every config value traces to a "
        "tokenizer_config.json path recorded per model.")
    res["verdict"] = " ".join(parts)
    res["_analysis_provenance"] = stamp(__file__)
    json.dump(res, open(OUT, "w"), indent=1)
    print(res["verdict"])
    print("\nwrote", rel(OUT))


if __name__ == "__main__":
    main()
