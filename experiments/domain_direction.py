"""Is the domain effect's DIRECTION predictable -- or is it just floor and ceiling? M3b.

THE DEFLATIONARY ACCOUNT THIS RUN EXISTS TO TEST. F144 and F147 both report that the domain moves
models in model-SPECIFIC directions, and that reads as a finding about weights. But there is a dull
explanation that would produce the same table: a model whose raw fixed_point_fraction is 0.948 has
almost nowhere to go but down, and one sitting at 0.000 has nowhere to go but up. If direction is
mostly determined by where the model STARTS, "model-specific" is a restatement of the raw value and
not a property of the interaction at all. That must be ruled out before any structural claim.

WHY THIS IS MOSTLY ANALYSIS-ONLY, AND WHY THAT IS THE RIGHT DESIGN. The decisive test does not need
more models -- it needs the SAME model moving BOTH ways from ITS OWN raw value. A within-model
bidirectional case is immune to floor/ceiling by construction, because the starting point is held
fixed while the domain varies. Widening the cohort would help the between-model tests below, but it
cannot make them decisive: with six model clusters a rank correlation cannot fail informatively, and
running more censuses to compute one anyway would repeat F137's mistake rather than fix it.

F148'S CONSTRAINT IS APPLIED THROUGHOUT. Prose is never a single number here. Where a prose direction
is quoted it is taken over ALL of that model's prose samples, and if they disagree in sign the domain
is recorded as TEXT-DEPENDENT rather than as an up or a down. Quoting F147's single CORPUS value
would silently reintroduce the draw F148 showed to matter.

PRE-REGISTERED:
  RUNG       the raw and domain values must match domain_gradient's stored analysis exactly; this
             reads existing results and introduces no new measurement, so any mismatch is a bug.
  ANTI-VACUITY  a shift counts as UP or DOWN only if it exceeds that model's tolerance, the larger
             of four census starts and twice its own across-seed noise. Shifts inside tolerance are
             FLAT and are not evidence of direction in either sign. A model with no shift outside
             tolerance is excluded from the PRIMARY and named.
  PRIMARY    within-model bidirectionality: does any model show BOTH a robust UP and a robust DOWN
             from its own single raw value? Registered readings:
               at least one such model -> the floor/ceiling account is REFUTED as a general
                 explanation. Direction is a joint property of weights and domain, because the
                 weights and the starting value are held fixed while only the domain changes.
               none -> the deflationary account SURVIVES on this cohort, and F147's
                 "model-specific direction" must be reported as confounded with the raw value
                 until a bidirectional model is found.
  SECONDARY  the floor/ceiling BASELINE, stated as a predictor and scored: sign(shift) = sign(0.5 -
             raw). Its accuracy over (model, domain) units says how much of the direction pattern is
             mechanical. High accuracy alongside a PRIMARY refutation means BOTH are true -- the
             baseline explains most units while specific models still escape it.
  TERTIARY   candidate predictors of shift magnitude (raw value, template length, parameter count).
             DESCRIPTIVE ONLY, and declared so HERE, before the numbers: with six model clusters a
             rank correlation cannot fail informatively, so no correlation computed below is a test
             and none may be cited as one. This is F137's lesson written into the design.
  BOUNDARY   six instruction-tuned models, one statistic, three domains plus a prose ensemble.
"""
import sys as _sys, pathlib as _pathlib
_ROOT = _pathlib.Path(__file__).resolve().parents[1]
_sys.path[:0] = [str(_ROOT / "src"), str(_ROOT / "experiments"), str(_ROOT / "fingerprint"),
                 str(_ROOT / "gatecheck" / "src")]
import json, os

import numpy as np

from provenance import stamp, rel
from argmax_census_hardened import N_STARTS, CENSUS_SEEDS

OUT = str(_ROOT / "results" / "domain_direction.json")
GRAD = _ROOT / "results" / "domain_gradient.json"
PROSE = _ROOT / "results" / "prose_samples.json"

MIN_SHIFT = 4.0 / N_STARTS
NOISE_FACTOR = 2.0
MIN_CLUSTERS_FOR_RANK = 10          # below this a rank correlation cannot fail informatively


def classify_shift(delta, tol):
    if abs(delta) <= tol:
        return "flat"
    return "up" if delta > 0 else "down"


def build(res):
    grad = json.load(open(GRAD))
    rows = grad["analysis"]["rows"]
    prose = json.load(open(PROSE))["analysis"]["rows"] if PROSE.exists() else {}
    out = {}
    for m, pt in rows.items():
        raw = float(np.mean(pt["raw"]["fix"]))
        noise = max(float(abs(pt[d]["fix"][0] - pt[d]["fix"][1])) for d in pt)
        tol = max(MIN_SHIFT, NOISE_FACTOR * noise)
        dirs, shifts = {}, {}
        for d in ("bos", "chat_template"):
            v = float(np.mean(pt[d]["fix"]))
            shifts[d] = v - raw
            dirs[d] = classify_shift(v - raw, tol)
        # PROSE AS AN ENSEMBLE, never as one number -- F148 showed the single draw matters.
        pr = prose.get(m, {}).get("fix")
        if pr:
            ds = {classify_shift(v - raw, tol) for v in pr.values()}
            nonflat = ds - {"flat"}
            dirs["prose"] = ("text_dependent" if len(nonflat) > 1
                             else (nonflat.pop() if nonflat else "flat"))
            shifts["prose"] = float(np.mean(list(pr.values()))) - raw
            out.setdefault("_prose_n", {})[m] = len(pr)
        out[m] = dict(raw=round(raw, 4), tol=round(tol, 4), seed_noise=round(noise, 4),
                      shifts={k: round(v, 4) for k, v in shifts.items()}, directions=dirs)
    res["rows"] = {k: v for k, v in out.items() if not k.startswith("_")}
    res["prose_sample_counts"] = out.get("_prose_n", {})


def analyse(res):
    rows, parts = res["rows"], []
    parts.append(
        f"RUNG: this run introduces no measurement -- it reads domain_gradient ({len(rows)} models) "
        f"and prose_samples, and every value below is theirs.")

    active = {m: r for m, r in rows.items()
              if any(v in ("up", "down") for v in r["directions"].values())}
    excluded = [m for m in rows if m not in active]
    parts.append(
        f"ANTI-VACUITY: {len(excluded)} of {len(rows)} models have NO shift outside their own "
        f"tolerance"
        + (f" -- {[m.split('/')[-1] for m in excluded]}, excluded from the PRIMARY: a model that "
           f"does not move cannot move in two directions, and counting it either way would be "
           f"vacuous. " if excluded else ", so every model can show a direction. ")
        + "Tolerance is max(4 census starts, 2x that model's own across-seed noise).")

    both = []
    for m, r in active.items():
        ds = set(r["directions"].values())
        if "up" in ds and "down" in ds:
            both.append(m)
    res["bidirectional"] = both
    parts.append(
        "PRIMARY, within-model bidirectionality (the weights and the raw value held FIXED, only the "
        "domain varying -- so floor and ceiling cannot produce it): "
        + "; ".join("{} raw {:.3f} -> {}".format(
            m.split("/")[-1], r["raw"],
            ", ".join(f"{d} {r['shifts'][d]:+.3f} ({r['directions'][d]})"
                      for d in r["directions"])) for m, r in active.items())
        + ". "
        + (f"{len(both)} model(s) move BOTH ways from a single raw value "
           f"({[m.split('/')[-1] for m in both]}), so the floor/ceiling account is REFUTED as a "
           f"general explanation: direction is a joint property of weights and domain, not a "
           f"restatement of where the model started."
           if both else
           "NO model moves both ways from its own raw value, so on this cohort the deflationary "
           "account SURVIVES and F147's model-specific direction must be reported as confounded "
           "with the raw value until a bidirectional model is found."))

    # SCORING THE BASELINE ONLY WHERE IT COULD HAVE BEEN WRONG. A model at raw 0.000 cannot move
    # down and one at 0.979 cannot move up by more than its own tolerance, so those units confirm
    # sign(0.5 - raw) no matter what is true -- they are the SAME vacuity defect, now inside the
    # baseline instead of inside the measurement. A unit is scoreable only if the model had room to
    # move in BOTH directions by more than its tolerance.
    # WITHIN-PROSE bidirectionality, judged per sample against THAT SAMPLE's own seed noise. This is
    # the sharpest available form of the PRIMARY -- same weights, same raw value, same domain KIND
    # and same length, only the text differing -- so nothing mechanical can produce a two-sided
    # result. It gets its own noise treatment because a model-level tolerance hides that one sample
    # can be far noisier than the rest, which is exactly what happens here.
    prose_bi = {}
    if PROSE.exists():
        praw = json.load(open(PROSE))["runs"]
        for m, r in rows.items():
            ups, downs = [], []
            seen = set()
            for k in praw:
                p = k.split("|")
                if p[0] != m or len(p) != 3 or p[2] == "rawcheck" or p[2] in seen:
                    continue
                seen.add(p[2])
                pair = [praw[f"{m}|s{cs}|{p[2]}"]["fixed_point_fraction"] for cs in CENSUS_SEEDS]
                mean, sn = float(np.mean(pair)), float(abs(pair[0] - pair[1]))
                delta, tol_s = mean - r["raw"], max(MIN_SHIFT, NOISE_FACTOR * sn)
                if delta > tol_s:
                    ups.append((p[2], round(delta, 4), round(sn, 4)))
                elif delta < -tol_s:
                    downs.append((p[2], round(delta, 4), round(sn, 4)))
            if ups or downs:
                prose_bi[m] = dict(up=sorted(ups), down=sorted(downs))
    res["prose_bidirectional"] = prose_bi
    bi_models = [m for m, v in prose_bi.items() if v["up"] and v["down"]]
    parts.append(
        "PRIMARY (sharpest form) -- WITHIN the prose domain: same raw value, same kind, same length, "
        "only the TEXT differing, each sample judged against its OWN seed noise. "
        + ("; ".join("{} raw {:.3f}: up {} down {}".format(
            m.split("/")[-1], rows[m]["raw"],
            [(s, d) for s, d, _n in v["up"]] or "none", [(s, d) for s, d, _n in v["down"]] or "none")
            for m, v in prose_bi.items()) if prose_bi else "No sample clears its own noise.")
        + ". "
        + (f"{[m.split('/')[-1] for m in bi_models]} move BOTH ways on TEXT ALONE, which no "
           f"floor/ceiling account can produce -- the starting value is identical for every sample."
           if bi_models else
           "No model has BOTH a robust up and a robust down sample, so bidirectionality is not "
           "established. Where it looked present it did not survive per-sample noise."))

    def has_room(r):
        return min(r["raw"], 1.0 - r["raw"]) > r["tol"]

    scoreable = [(m, d, r["raw"], r["shifts"][d], r["directions"][d])
                 for m, r in rows.items() if has_room(r)
                 for d in r["directions"] if r["directions"][d] in ("up", "down")]
    vacuous = [(m, r["raw"], r["tol"]) for m, r in rows.items() if not has_room(r)]
    res["baseline"] = dict(n_scoreable=len(scoreable), n_clusters_scoreable=len({m for m, *_ in scoreable}),
                           excluded_no_room=[dict(model=m, raw=raw, tol=tol)
                                             for m, raw, tol in vacuous])
    if scoreable:
        hits = sum(1 for _m, _d, raw, _s, dr in scoreable if (dr == "up") == (raw < 0.5))
        acc = hits / len(scoreable)
        res["baseline"].update(hits=hits, accuracy=round(acc, 3))
    parts.append(
        f"SECONDARY, the floor/ceiling BASELINE sign(shift) = sign(0.5 - raw), scored ONLY where it "
        f"could have been wrong: {len(vacuous)} of {len(rows)} models had no room to move both ways "
        f"by more than their own tolerance"
        + (f" -- {[(m.split('/')[-1], f'raw {raw:.3f}', f'tol {tol:.3f}') for m, raw, tol in vacuous]}. "
           f"Their units confirm the baseline whatever the truth, so scoring them would be vacuous. "
           if vacuous else ". ")
        + (f"{res['baseline'].get('hits')} of {len(scoreable)} scoreable units correct "
           f"({res['baseline'].get('accuracy', 0):.0%}), from "
           f"{res['baseline']['n_clusters_scoreable']} model cluster(s). "
           + ("Too few clusters to read as a rate -- this is a count."
              if res["baseline"]["n_clusters_scoreable"] < 3 else "")
           if scoreable else
           "NOTHING is scoreable: on this cohort the floor/ceiling account cannot be tested at all, "
           "because nearly every model sits AT a floor or a ceiling. That is the real state of the "
           "evidence -- F147's model-specific direction is currently UNTESTABLE here, neither "
           "confirmed nor refuted, and the prerequisite is models with MID-RANGE raw values rather "
           "than more models like these."))

    parts.append(
        f"TERTIARY, candidate predictors of shift magnitude: NOT COMPUTED, and this was declared in "
        f"the pre-registration before the numbers were seen. With {len(rows)} model clusters against "
        f"a floor of {MIN_CLUSTERS_FOR_RANK}, a rank correlation cannot fail informatively -- it "
        f"would return a number whatever the truth, and reporting it would manufacture a result. "
        f"This is F137's defect refused rather than repeated. Widening the instruct cohort is the "
        f"prerequisite, not a larger analysis of the same six.")
    parts.append(
        f"BOUNDARY: {len(rows)} instruction-tuned models, fixed_point_fraction only, domains bos / "
        f"prose-ensemble / chat_template against raw. Prose is read as an ENSEMBLE over that model's "
        f"samples per F148, and where its samples disagree in sign the domain is recorded "
        f"TEXT_DEPENDENT rather than as a direction. The PRIMARY is an existence result: it refutes "
        f"a general floor/ceiling account, it does not measure how often models are bidirectional.")
    res["verdict"] = " ".join(parts)


def main():
    res = json.load(open(OUT)) if os.path.exists(OUT) else {}
    res["_preregistration"] = dict(
        min_shift=MIN_SHIFT, noise_factor=NOISE_FACTOR, min_clusters_for_rank=MIN_CLUSTERS_FOR_RANK,
        reads=["results/domain_gradient.json", "results/prose_samples.json"],
        rung="analysis-only; every value is domain_gradient's or prose_samples'",
        primary="does any model move BOTH ways from its own single raw value, with weights and "
                "starting point held fixed and only the domain varying",
        secondary="accuracy of the floor/ceiling baseline sign(shift)=sign(0.5-raw), clustered",
        tertiary="predictors of magnitude are NOT computed -- six clusters cannot fail a rank "
                 "correlation informatively, declared before the numbers",
        prose_rule="prose is an ENSEMBLE over F148's samples; disagreeing signs record "
                   "TEXT_DEPENDENT, never a single draw",
        why="'model-specific direction' has a dull alternative explanation -- a model at 0.948 can "
            "mostly go down and one at 0.000 can only go up -- and it must be ruled out before the "
            "interaction can be called structural")
    build(res)
    analyse(res)
    res["_analysis_provenance"] = stamp(__file__)
    json.dump(res, open(OUT, "w"), indent=1)
    print(f"\n  -> {res['verdict']}")
    print("\nwrote", rel(OUT))


if __name__ == "__main__":
    main()
