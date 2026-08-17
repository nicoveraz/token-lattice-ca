"""Does STRUCTURAL text raise fixed-point structure, or was p1 just one idiosyncratic up-text?

THE LEAD, AND WHY IT NEEDS ITS OWN RUN. F154 found the only text that raises two models: `p1`, Pile
row 101, which begins "\\n\\nGreat Britain\\n\\n# Contents\\n\\n## Plan Your Trip\\n\\n### Welcome
to..." -- a table of contents, not prose. It takes `Minerva-3B` to 1.000 and `Falcon3-1B` to 0.979
while driving four other models to ~0.000. That suggests a mechanism: highly templated boilerplate is
exactly the context in which a model's next-token distribution collapses onto a single continuation,
so a structural prefix might build fixed points wherever the weights permit it at all.

BUT IT IS ONE TEXT, and a hypothesis read off one observation is the defect this project keeps
catching. F153's "all up-texts are Shakespeare" died the moment a real corpus arrived. This gives the
structural hypothesis the same chance to die.

"STRUCTURAL" IS DEFINED MECHANICALLY AND FIXED HERE, BEFORE SELECTION. Over the first 200 characters
of a Pile row, `score` = fraction of characters that are newline, markup (#|*-=>[]{}_`) or digit.
Measured over 3000 rows the median is 0.025 and the 95th percentile is 0.150; `p1` scores **0.230**,
the 97.6th percentile. So:
    STRUCTURAL  score >= 0.15   (top ~5%, 146 of 3000 rows qualify)
    PROSE       score <= 0.02   (bottom ~43%)
Texts are then taken in ROW-INDEX ORDER, never by reading them. `p1` is included by construction as
the reproducibility anchor; the other five structural texts are the next five qualifying rows.

MODELS: the three F154 showed are RAISABLE (some text moves them up) plus one that is NOT, as a
control. If structural text also raises the unraisable model, then "raisable" is not the fixed model
property F154 claimed and that finding needs revising too.

PRE-REGISTERED:
  RUNG       `p1` must reproduce text_interaction's stored fixed_point_fraction EXACTLY for every
             model it was measured on. Same prefix, same seeds.
  ANTI-VACUITY  each model's headroom on both sides must exceed its own tolerance, from its stored
             raw seed noise. Verified, not assumed.
  PRIMARY    per RAISABLE model, the up-rate under STRUCTURAL texts vs under PROSE texts. Registered
             readings:
               structural up-rate clearly higher on the raisable models -> the mechanism is real:
                 templated boilerplate builds fixed points wherever the weights permit, and `p1` was
                 an instance rather than a coincidence. This would be the first PREDICTIVE statement
                 in the whole domain programme -- everything so far says what cannot be predicted.
               rates similar -> `p1` was one idiosyncratic up-text that happened to be structural.
                 The lead DIES, and F154's "up-sets are idiosyncratic" stands unqualified.
  CONTROL    the unraisable model must stay unraisable under structural text. If it does not,
             F154's claim that bidirectionality is a MODEL property is wrong.
  NOT A TEST With three raisable models this estimates RATES; it is not a significance test and no
             p-value will be computed. Declared before the numbers -- three clusters cannot fail one
             informatively, the F149/F153 refusal.
  BOUNDARY   one length (9 tokens), one corpus, one operationalisation of "structural".
"""
import sys as _sys, pathlib as _pathlib
_ROOT = _pathlib.Path(__file__).resolve().parents[1]
_sys.path[:0] = [str(_ROOT / "src"), str(_ROOT / "experiments"), str(_ROOT / "fingerprint"),
                 str(_ROOT / "gatecheck" / "src")]
import gc, json, os, time
os.environ.setdefault("HF_HOME", str(_ROOT / "hf_cache"))
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("HF_HUB_OFFLINE", "1")

import numpy as np
import torch

from provenance import stamp, rel
from gate1 import argmax_census
from argmax_census_hardened import classify, N_STARTS, CENSUS_SEEDS
from argmax_census_templated import _Prefixed

OUT = str(_ROOT / "results" / "structural_text.json")
TI = _ROOT / "results" / "text_interaction.json"
BASE = _ROOT / "results" / "domain_base.json"

RAISABLE = ["tiiuae/Falcon3-1B-Base", "sapienzanlp/Minerva-3B-base-v1.0", "Qwen/Qwen1.5-1.8B"]
CONTROL = ["HuggingFaceTB/SmolLM-1.7B"]          # 0/12 up in F154
MODELS = RAISABLE + CONTROL

LENGTH = 9
ANCHOR_ROW = 101                                  # p1, the F154 shared up-text -> the RUNG
STRUCT_MIN, PROSE_MAX = 0.15, 0.02                # fixed here; p95 and ~p43 of the Pile
N_PER_ARM = 6
SCAN_ROWS = 3000
MARK = set("\n#|*-=>[]{}_`")
MIN_SHIFT = 4.0 / N_STARTS
NOISE_FACTOR = 2.0


def struct_score(t, n=200):
    h = t[:n]
    return sum(1 for c in h if c in MARK or c.isdigit()) / max(len(h), 1)


def pick_rows():
    """Row indices only -- selection never reads a text for its content."""
    from datasets import load_dataset
    ds = load_dataset("NeelNanda/pile-10k", split="train")
    struct, prose = [ANCHOR_ROW], []
    for i in range(SCAN_ROWS):
        if len(struct) >= N_PER_ARM and len(prose) >= N_PER_ARM:
            break
        t = ds[i]["text"]
        if len(t) < 400 or i == ANCHOR_ROW:
            continue
        s = struct_score(t)
        if s >= STRUCT_MIN and len(struct) < N_PER_ARM:
            struct.append(i)
        elif s <= PROSE_MAX and len(prose) < N_PER_ARM:
            prose.append(i)
    return ds, struct, prose


def texts(tok, ds, struct, prose):
    out = {}
    for tag, rows in (("t", struct), ("r", prose)):        # t=structural, r=prose control
        for j, i in enumerate(rows):
            ids = tok(ds[int(i)]["text"][:4000], add_special_tokens=False)["input_ids"]
            if len(ids) >= LENGTH:
                out[f"{tag}{j}"] = [int(x) for x in ids[:LENGTH]]
    return out


def _raw_of(m):
    src = json.load(open(BASE))["runs"]
    ks = [f"{m}|s{cs}|raw" for cs in CENSUS_SEEDS]
    if not all(k in src for k in ks):
        return None
    v = [src[k]["fixed_point_fraction"] for k in ks]
    return float(np.mean(v)), float(abs(v[0] - v[1]))


def analyse(res):
    runs, parts, analysis = res["runs"], [], {}
    ti = json.load(open(TI))["runs"] if TI.exists() else {}

    errs = []
    for m in MODELS:
        for cs in CENSUS_SEEDS:
            a, b = runs.get(f"{m}|s{cs}|t0"), ti.get(f"{m}|s{cs}|p1")
            if a and b:
                errs.append(abs(a["fixed_point_fraction"] - b["fixed_point_fraction"]))
    worst = max(errs, default=float("inf"))
    ok = bool(errs) and worst == 0.0
    parts.append(
        f"RUNG (t0 is F154's p1 and must reproduce it): {len(errs)} cells, worst error {worst:.2e}. "
        + ("Identical." if ok else "NOT reproduced -- nothing below is read."))
    if not ok:
        res["analysis"] = dict(rung_passes=False, worst=worst)
        res["verdict"] = " ".join(parts)
        return

    rows, excluded = {}, []
    for m in MODELS:
        r = _raw_of(m)
        if r is None:
            continue
        raw, raw_n = r
        if min(raw, 1 - raw) <= max(MIN_SHIFT, NOISE_FACTOR * raw_n):
            excluded.append((m, round(raw, 3)))
            continue
        per = {}
        for k in sorted({x.split("|")[2] for x in runs
                         if x.startswith(f"{m}|") and len(x.split("|")) == 3}):
            ks = [f"{m}|s{cs}|{k}" for cs in CENSUS_SEEDS]
            if not all(x in runs for x in ks):
                continue
            v = [runs[x]["fixed_point_fraction"] for x in ks]
            mu, n = float(np.mean(v)), float(abs(v[0] - v[1]))
            tol = max(MIN_SHIFT, NOISE_FACTOR * max(n, raw_n))
            d = mu - raw
            per[k] = dict(value=round(mu, 4), shift=round(d, 4), tol=round(tol, 4),
                          dir="up" if d > tol else ("down" if d < -tol else "flat"))
        if per:
            rows[m] = dict(raw=round(raw, 4), texts=per)
    analysis["rows"], analysis["excluded"] = rows, excluded
    parts.append(
        f"ANTI-VACUITY: {len(excluded)} model(s) lack room to move both ways"
        + (f" -- {excluded}, excluded." if excluded else ", so all can show either direction."))
    if not rows:
        res["analysis"] = analysis
        res["verdict"] = " ".join(parts) + " No model complete yet."
        return

    def rate(m, tag):
        ks = [k for k in rows[m]["texts"] if k.startswith(tag)]
        up = sum(1 for k in ks if rows[m]["texts"][k]["dir"] == "up")
        return up, len(ks)

    counts = {}
    for m in rows:
        su, sn = rate(m, "t")
        pu, pn = rate(m, "r")
        counts[m] = dict(struct_up=su, struct_n=sn, prose_up=pu, prose_n=pn,
                         raisable=m in RAISABLE)
    analysis["counts"] = counts
    rz = [m for m in rows if m in RAISABLE]
    tot_s = sum(counts[m]["struct_up"] for m in rz), sum(counts[m]["struct_n"] for m in rz)
    tot_p = sum(counts[m]["prose_up"] for m in rz), sum(counts[m]["prose_n"] for m in rz)
    parts.append(
        "PRIMARY, up-rate under STRUCTURAL vs PROSE text, on the models F154 showed are raisable: "
        + "; ".join("{} struct {}/{} vs prose {}/{}".format(
            m.split("/")[-1], counts[m]["struct_up"], counts[m]["struct_n"],
            counts[m]["prose_up"], counts[m]["prose_n"]) for m in rz)
        + f". Totals: structural {tot_s[0]}/{tot_s[1]}, prose {tot_p[0]}/{tot_p[1]}. ")
    # CONSISTENCY BEFORE AGGREGATE. A pooled rate over models that disagree is the Simpson's shape
    # this project has already been caught by (F141). The aggregate is only read if the models
    # individually agree; otherwise the per-model split IS the result.
    agree = [m for m in rz if counts[m]["struct_n"] and counts[m]["prose_n"]
             and counts[m]["struct_up"] / counts[m]["struct_n"]
             > counts[m]["prose_up"] / counts[m]["prose_n"]]
    against = [m for m in rz if m not in agree]
    analysis["per_model_direction"] = dict(supports=[m.split("/")[-1] for m in agree],
                                           reverses_or_flat=[m.split("/")[-1] for m in against])
    pooled_passes = (tot_s[1] and tot_p[1]
                     and tot_s[0] / tot_s[1] >= 2 * max(tot_p[0] / tot_p[1], 1e-9)
                     and tot_s[0] >= 3)
    parts.append(
        f"CONSISTENCY: {len(agree)} of {len(rz)} raisable models individually show a HIGHER up-rate "
        f"under structural than under prose ({[m.split('/')[-1] for m in agree] or '-'}); "
        f"{len(against)} do not ({[m.split('/')[-1] for m in against] or '-'}). "
        + ("All raisable models agree, so the pooled rate is a fair summary of them. "
           if not against else
           "The models DISAGREE, so the pooled rate is NOT read as a summary -- pooling across units "
           "that point different ways is the Simpson's shape this project was already caught by "
           "(F141), and the per-model split below is the result instead. "))
    parts.append(
        ("VERDICT: structural text raises these models far more often than prose, consistently "
         "across every raisable model, so `p1` was an INSTANCE of a mechanism rather than a "
         "coincidence: templated boilerplate builds fixed points wherever the weights permit. This "
         "is the first PREDICTIVE statement in the domain programme."
         if pooled_passes and not against else
         f"VERDICT: PARTIAL and model-dependent. Structural text raises "
         f"{len(agree)} of {len(rz)} raisable models more often than prose does, and REVERSES or "
         f"fails on the rest. The pooled ratio ({tot_s[0]}/{tot_s[1]} vs {tot_p[0]}/{tot_p[1]}) "
         f"clears the pre-registered bar, but it is carried by a subset, so the honest claim is that "
         f"the MECHANISM IS ITSELF MODEL-DEPENDENT -- which is what every other finding in this "
         f"programme also says. It does NOT yield a predictive rule about prefixes."
         if pooled_passes else
         f"VERDICT: the lead DIES. Structural and prose texts raise these models at comparable "
         f"rates, so `p1` was one idiosyncratic up-text that happened to be structural, and F154's "
         f"'up-sets are idiosyncratic' stands unqualified."))
    ctl = [m for m in rows if m in CONTROL]
    if ctl:
        parts.append(
            "CONTROL, the model F154 found unraisable (0/12 up): "
            + "; ".join("{} struct {}/{} vs prose {}/{}".format(
                m.split("/")[-1], counts[m]["struct_up"], counts[m]["struct_n"],
                counts[m]["prose_up"], counts[m]["prose_n"]) for m in ctl)
            + ". "
            + ("It stays unraisable, so bidirectionality is a MODEL property as F154 claimed."
               if all(counts[m]["struct_up"] == 0 for m in ctl) else
               "STRUCTURAL TEXT RAISES IT. F154's claim that bidirectionality is a fixed model "
               "property is WRONG -- the model was only unraisable by the texts tried, and that "
               "finding needs revising."))
    parts.append(
        f"NOT A TEST, declared before the numbers: {len(rz)} raisable model clusters cannot fail a "
        f"significance test informatively, so this estimates RATES and no p-value is computed.")
    parts.append(
        f"BOUNDARY: {len(rows)} models, {N_PER_ARM} structural and {N_PER_ARM} prose texts from ONE "
        f"corpus at ONE length ({LENGTH} tokens), under ONE operationalisation of 'structural' "
        f"(newline/markup/digit density >= {STRUCT_MIN} over the first 200 chars, vs <= {PROSE_MAX}). "
        f"Rows were chosen by INDEX ORDER after thresholding, never by reading them.")
    res["analysis"] = analysis
    res["verdict"] = " ".join(parts)


def main():
    res = json.load(open(OUT)) if os.path.exists(OUT) else {"runs": {}}
    # The selected row indices ARE provenance -- which texts were used cannot be recovered from the
    # token ids alone. An --analyse pass must not drop them by rebuilding the block from scratch.
    _keep = {k: v for k, v in res.get("_preregistration", {}).items()
             if k in ("struct_rows", "prose_rows")}
    res["_preregistration"] = dict(
        models=MODELS, raisable=RAISABLE, control=CONTROL, length=LENGTH, anchor_row=ANCHOR_ROW,
        struct_min=STRUCT_MIN, prose_max=PROSE_MAX, n_per_arm=N_PER_ARM, scan_rows=SCAN_ROWS,
        n_starts=N_STARTS, census_seeds=CENSUS_SEEDS,
        definition="structural = fraction of newline/markup/digit chars in the first 200 >= 0.15 "
                   "(p95 of the Pile; p1 scores 0.230, p97.6). prose control <= 0.02. Rows taken in "
                   "INDEX ORDER after thresholding, never read for content.",
        rung="t0 IS F154's p1 and must reproduce it exactly",
        primary="up-rate under structural vs prose text, on the three models F154 showed raisable",
        control_check="the unraisable model must stay unraisable, or F154's model-property claim "
                      "is wrong",
        not_a_test="three raisable clusters cannot fail a significance test informatively; rates "
                   "only, no p-value",
        why="p1 is one text, and a hypothesis read off one observation is the defect this project "
            "keeps catching -- F153's 'all up-texts are Shakespeare' died the same way")
    res["_preregistration"].update(_keep)
    if "--analyse" not in _sys.argv:
        from transformers import AutoTokenizer, AutoModelForCausalLM
        dev = "mps" if torch.backends.mps.is_available() else "cpu"
        limit = int(_sys.argv[_sys.argv.index("--limit") + 1]) if "--limit" in _sys.argv else 0
        ds, struct, prose = pick_rows()
        res["_preregistration"]["struct_rows"] = struct
        res["_preregistration"]["prose_rows"] = prose
        print(f"  structural rows {struct}\n  prose rows      {prose}", flush=True)
        done = 0
        for m in MODELS:
            t0 = time.time()
            try:
                tok = AutoTokenizer.from_pretrained(m)
            except Exception as e:
                print(f"  {m}: TOK FAILED ({type(e).__name__})", flush=True)
                continue
            tx = texts(tok, ds, struct, prose)
            if all(f"{m}|s{cs}|{k}" in res["runs"] for k in tx for cs in CENSUS_SEEDS):
                continue
            if limit and done >= limit:
                print(f"  (stopping after {done}; re-run to continue)", flush=True)
                break
            try:
                model = AutoModelForCausalLM.from_pretrained(m).eval().to(
                    dev, torch.float16 if dev != "cpu" else torch.float32)
            except Exception as e:
                res["runs"][f"{m}|failed"] = dict(model=m, error=type(e).__name__)
                json.dump(res, open(OUT, "w"), indent=1)
                print(f"  {m}: LOAD FAILED ({type(e).__name__})", flush=True)
                continue
            V = int(getattr(model.config, "vocab_size", len(tok)))
            sp = {i for i in (tok.bos_token_id, tok.eos_token_id, tok.pad_token_id,
                              tok.unk_token_id) if i is not None}
            pool = np.array([i for i in range(min(V, len(tok))) if i not in sp], np.int64)
            print(f"  {m:<34} {len(tx)} texts at {LENGTH} tok", flush=True)
            for k, pre in tx.items():
                for cs in CENSUS_SEEDS:
                    key = f"{m}|s{cs}|{k}"
                    if key in res["runs"]:
                        continue
                    c = argmax_census(_Prefixed(model, pre), tok, dev, pool,
                                      np.random.default_rng(cs), n_starts=N_STARTS)
                    c.update(cls=classify(c), model=m, census_seed=cs, text=k,
                             n_prefix_tokens=len(pre), arm="structural" if k[0] == "t" else "prose")
                    res["runs"][key] = c
                    json.dump(res, open(OUT, "w"), indent=1)
                    print(f"  {m:<34} {k:<4} s={cs} cls={c['cls']:<11} "
                          f"fix={c['fixed_point_fraction']:.3f}", flush=True)
            done += 1
            print(f"  {m:<34} model done in {time.time()-t0:.0f}s", flush=True)
            del model
            gc.collect()
            try:
                torch.mps.empty_cache()
            except Exception:
                pass
    analyse(res)
    res["_analysis_provenance"] = stamp(__file__)
    json.dump(res, open(OUT, "w"), indent=1)
    print(f"\n  -> {res['verdict']}")
    print("\nwrote", rel(OUT))


if __name__ == "__main__":
    main()
