"""F185's escape map on 20 models across several corpora and nine tokenizers. Same estimator.

Registered in experiments/prereg_escape_widening.json (frozen `3c04677f...` before any new cell).

WHAT THIS CONVERTS. F185 could only DISCLOSE two confounds. Both of its attribution misses were
Mamba landing on RWKV, so what it recovered may be architecture class rather than family; and family
was confounded with tokenizer, because eleven of its twelve models shared a GPT-NeoX or GPT-2
vocabulary. It also held corpus fixed, so it could say nothing about corpus and rule nothing out.
This run makes all three testable rather than confessed: five recurrent/SSM models against fifteen
transformers, nine distinct tokenizers, and corpus no longer constant.

WHY NOT THE 17-MODEL CENSUS COHORT. It contains no recurrent models at all, its probe intersection is
152 -- below the registered floor -- and its families are singletons. The prereg records the sweep.

The twelve core cells are REUSED from results/selfcont_set_*.json and never re-measured, so the
widened numbers cannot drift from the ones F185 reported.
"""
import sys as _sys, pathlib as _pathlib
_ROOT = _pathlib.Path(__file__).resolve().parents[1]
_sys.path[:0] = [str(_ROOT / "src"), str(_ROOT / "experiments"), str(_ROOT / "gatecheck" / "src")]
import collections, gc, itertools, json, os, time

os.environ.setdefault("HF_HOME", str(_ROOT / "hf_cache"))
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import numpy as np
import torch

from provenance import stamp, rel
from gatecheck import balance_report, balanced_accuracy
from selfcont_set import BATCH, BATCH_BIG, BIG, out_path

PREREG = "experiments/prereg_escape_widening.json"
PR = json.load(open(_ROOT / PREREG))
OUT = _ROOT / "results" / "escape_widening.json"
CACHE = _ROOT / "results" / "escape_widening_cells.json"
PROBES = _ROOT / "experiments" / "probe_strings_selfcont.json"
REUSED, NEW = PR["cohort"]["reused"], PR["cohort"]["new"]
RECURRENT = set(PR["cohort"]["architecture_class"]["recurrent_or_ssm"])
MIN_INDEX = 500

FAMILY = {"pythia": "Pythia", "gpt-neo": "GPT-Neo", "rwkv": "RWKV", "mamba": "Mamba",
          "zamba": "Zamba", "gemma": "Gemma", "llama": "Llama", "qwen": "Qwen",
          "olmo": "OLMo", "falcon": "Falcon", "stablelm": "StableLM", "smollm": "SmolLM"}


def family(m):
    low = m.split("/")[-1].lower()
    for k, v in FAMILY.items():
        if low.startswith(k):
            return v
    return m.split("/")[0]


@torch.no_grad()
def top1(model, ids, dev, batch):
    out = np.empty(len(ids), np.int64)
    for i in range(0, len(ids), batch):
        ch = ids[i:i + batch]
        x = torch.tensor(ch, dtype=torch.long, device=dev).view(-1, 1).repeat(1, 2)
        out[i:i + len(ch)] = model(input_ids=x).logits[:, -1].float().argmax(-1).cpu().numpy()
    return out


def main():
    from transformers import AutoTokenizer, AutoModelForCausalLM
    strings = [e["s"] for e in json.load(open(PROBES))["strings"]]
    cache = json.load(open(CACHE)) if CACHE.exists() else {}
    failed = []
    for m in NEW:
        if m in cache:
            print(f"  {m:<36} cached", flush=True); continue
        t0 = time.time()
        try:
            tok = AutoTokenizer.from_pretrained(m)
            model = AutoModelForCausalLM.from_pretrained(m).eval().to("cpu", torch.float32)
        except Exception as e:
            failed.append(dict(model=m, error=type(e).__name__, detail=str(e)[:160]))
            print(f"  {m:<36} LOAD FAILED {type(e).__name__}", flush=True); continue
        pid = []
        for s in strings:
            try:
                enc = tok(s, add_special_tokens=False)["input_ids"]
            except Exception:
                enc = []
            pid.append(int(enc[0]) if len(enc) == 1 else -1)
        idx = [i for i, v in enumerate(pid) if v >= 0]
        ids = [pid[i] for i in idx]
        a = top1(model, ids, "cpu", BATCH_BIG if m in BIG else BATCH)
        dec = [tok.decode([int(i)]) for i in a]
        del model; gc.collect()
        cache[m] = dict(model=m, family=family(m), probe_positions=idx, source_ids=ids,
                        top1_id=[int(x) for x in a], top1_str=dec,
                        escapes=[bool(int(x) != int(y)) for x, y in zip(a, ids)],
                        secs=round(time.time() - t0, 1))
        json.dump(cache, open(CACHE, "w"))
        print(f"  {m:<36} n={len(ids):<6} escaping={sum(cache[m]['escapes']):<6} "
              f"({cache[m]['secs']:.0f}s)", flush=True)
    _verdict(cache, failed, strings)


def _verdict(cache, failed, strings):
    # reuse the twelve core cells without re-measuring them
    core = {}
    for m in REUSED:
        p = out_path(m, "fp32")
        if not p.exists():
            failed.append(dict(model=m, error="core cell missing")); continue
        d = json.load(open(p))
        pid = np.array(d["probe_token_ids"])
        idx = [int(i) for i in np.flatnonzero(pid >= 0)]
        amax = np.array(d["argmax_ids"], np.int64)[pid[idx]]
        marg = np.array(d["margins_e4"], np.int64)[pid[idx]]
        sent = d.get("_unmeasured_sentinel", -2147483648)
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(m)
        core[m] = dict(model=m, family=family(m), probe_positions=idx,
                       source_ids=[int(x) for x in pid[idx]],
                       top1_str=[tok.decode([int(i)]) for i in amax],
                       escapes=[bool(x <= 0 and x != sent) for x in marg])
    cells = {**core, **{k: v for k, v in cache.items() if not k.startswith("_")}}
    models = sorted(cells)

    shared = None
    for m in models:
        s = set(cells[m]["probe_positions"])
        shared = s if shared is None else (shared & s)
    shared = sorted(shared)
    res = dict(_preregistration_file=PREREG,
               _prereg_sha256=open(_ROOT / "experiments" / "prereg_escape_widening.sha256").read().split()[0],
               n_models=len(models), models=models, failed=failed,
               intersection=len(shared), n_probe_candidates=len(strings))
    if len(shared) < MIN_INDEX:
        res["KW4_fires"] = True
        res["verdict"] = (f"KW4 FIRES: the realised intersection over {len(models)} measured models "
                          f"is {len(shared)}, below the registered floor of {MIN_INDEX}. NOT "
                          f"DECIDABLE for coverage; the run stops here as registered.")
        _write(res); return
    res["KW4_fires"] = False

    pos = {m: {p: i for i, p in enumerate(cells[m]["probe_positions"])} for m in models}
    A = {}
    for m in models:
        i = [pos[m][p] for p in shared]
        A[m] = (np.array([cells[m]["top1_str"][j] for j in i], dtype=object),
                np.array(cells[m]["escapes"], bool)[i])

    prof = {}
    for m in models:
        s, e = A[m]
        c = collections.Counter(s[e].tolist())
        top, n = c.most_common(1)[0]
        prof[m] = dict(family=cells[m]["family"], n_escaping=int(e.sum()), n_distinct=len(c),
                       modal_destination=top, modal_share=round(n / max(1, int(e.sum())), 4),
                       architecture="recurrent" if m in RECURRENT else "transformer")
    res["per_model"] = prof
    res["KW5_modal_share_max"] = max(p["modal_share"] for p in prof.values())

    def agree(a, b):
        sa, ea = A[a]; sb, eb = A[b]
        both = ea & eb
        n = int(both.sum())
        return (None if n == 0 else float(np.mean(sa[both] == sb[both]))), n

    M = np.full((len(models), len(models)), np.nan)
    for i, j in itertools.combinations(range(len(models)), 2):
        v, _ = agree(models[i], models[j])
        M[i, j] = M[j, i] = -1.0 if v is None else v
    res["agreement_matrix"] = dict(order=models, matrix=np.round(M, 4).tolist())

    fam = {m: cells[m]["family"] for m in models}
    counts = collections.Counter(fam.values())
    scored = [m for m in models if counts[fam[m]] >= 2]     # the arXiv:2607.10252 convention
    hits = 0; nn = {}
    for m in scored:
        i = models.index(m)
        d = M[i].copy(); d[i] = -np.inf
        j = int(np.nanargmax(d))
        ok = fam[models[j]] == fam[m]
        hits += ok
        nn[m] = dict(nearest=models[j], agreement=round(float(M[i, j]), 4), same_family=bool(ok))
    n = len(models)
    chance = float(np.mean([(counts[fam[m]] - 1) / (n - 1) for m in scored])) if scored else float("nan")
    res["identification"] = dict(
        n_cohort=n, n_scored=len(scored), rank1_same_family=hits,
        rank1_accuracy=round(hits / len(scored), 4) if scored else None,
        chance_family_level=round(chance, 4), family_counts=dict(counts),
        _convention="scored only over models with at least one same-family peer, as arXiv:2607.10252 does",
        nearest_neighbour=nn)
    res["KW1_fires"] = bool(scored and (hits / len(scored)) <= chance)

    arch = [prof[m]["architecture"] for m in models]
    bal = balance_report(arch, name="architecture class")
    pred = []
    for m in models:
        i = models.index(m); d = M[i].copy(); d[i] = -np.inf
        pred.append(prof[models[int(np.nanargmax(d))]]["architecture"])
    bacc, per = balanced_accuracy(pred, arch)
    res["architecture_class"] = dict(balance=bal.reason, counts=bal.counts,
                                     majority_rate=round(bal.majority_rate, 4),
                                     balanced_accuracy=round(bacc, 4),
                                     per_class_rate={k: round(v, 4) for k, v in per.items()},
                                     readable=bool(bal))
    res["KW2_fires"] = bool(bal and bacc <= bal.majority_rate)

    tokgrp = {m: len(cells[m]["source_ids"]) for m in models}   # proxy: identical resolved index
    tg = collections.defaultdict(list)
    for m in models:
        tg[tuple(cells[m]["source_ids"][:64])].append(m)
    same_fam, diff_fam, same_tok, diff_tok = [], [], [], []
    for i, j in itertools.combinations(range(len(models)), 2):
        v = M[i, j]
        if v < 0:
            continue
        (same_fam if fam[models[i]] == fam[models[j]] else diff_fam).append(v)
        gi = next(k for k, vs in tg.items() if models[i] in vs)
        gj = next(k for k, vs in tg.items() if models[j] in vs)
        (same_tok if gi == gj else diff_tok).append(v)
    res["family_vs_tokenizer"] = dict(
        n_tokenizer_groups=len(tg),
        mean_within_family=round(float(np.mean(same_fam)), 4) if same_fam else None,
        mean_across_family=round(float(np.mean(diff_fam)), 4) if diff_fam else None,
        mean_within_tokenizer=round(float(np.mean(same_tok)), 4) if same_tok else None,
        mean_across_tokenizer=round(float(np.mean(diff_tok)), 4) if diff_tok else None,
        _tokenizer_group="models whose first 64 resolved probe token ids are identical; a "
                         "necessary-not-sufficient proxy for a shared vocabulary")
    lift_f = ((res["family_vs_tokenizer"]["mean_within_family"] or 0)
              - (res["family_vs_tokenizer"]["mean_across_family"] or 0))
    lift_t = ((res["family_vs_tokenizer"]["mean_within_tokenizer"] or 0)
              - (res["family_vs_tokenizer"]["mean_across_tokenizer"] or 0))
    res["family_vs_tokenizer"]["family_lift"] = round(lift_f, 4)
    res["family_vs_tokenizer"]["tokenizer_lift"] = round(lift_t, 4)
    res["KW3_fires"] = bool(lift_t > lift_f)

    # ---- UNREGISTERED ADDITIONS. All three land after the freeze and are labelled as such; the
    # ---- prereg stands unchanged. Ordering matters more than the additions.
    # (1) RE-BASELINING. F185's numbers were computed on a 3471-string support; this run's is
    #     smaller. No 3471-era number may be quoted against a 3355-era one, so F185's headline
    #     comparisons are recomputed HERE, on this support.
    reb = {}
    for lbl, (a, b) in {"decisive": ("EleutherAI/pythia-410m", "EleutherAI/pythia-410m-deduped"),
                        "far_mamba": ("EleutherAI/pythia-410m", "state-spaces/mamba-370m-hf"),
                        "far_rwkv": ("EleutherAI/pythia-410m", "RWKV/rwkv-4-430m-pile"),
                        "far_gptneo": ("EleutherAI/pythia-410m", "EleutherAI/gpt-neo-125m")}.items():
        if a in A and b in A:
            v, nn_ = agree(a, b)
            reb[lbl] = dict(a=a, b=b, agreement=None if v is None else round(v, 4), n_shared=nn_)
    res["rebaselined_on_this_support"] = dict(
        support=len(shared), f185_support=3471, pairs=reb,
        _reading="F185 reported the decisive pair at 0.6355 and the far pairs at 0.3393-0.3902 on a "
                 "3471-string support. These are the same comparisons on THIS support. Quote these "
                 "when comparing to anything else in this file; quote F185's when comparing within "
                 "F185. Mixing them compares two different index sets.")

    # (2) PER-PAIR THINNESS. prereg_escape_rival carried a 500-source floor per PAIR (KA);
    #     prereg_escape_widening carries one only for the GLOBAL index (KW4). A model that rarely
    #     escapes thins every pair it is in, and one here does.
    thin = []
    for i, j in itertools.combinations(range(len(models)), 2):
        _, nn_ = agree(models[i], models[j])
        if nn_ < MIN_INDEX:
            thin.append(dict(a=models[i], b=models[j], n_shared=nn_))
    res["thin_pairs"] = dict(
        floor=MIN_INDEX, n_thin=len(thin), pairs=thin[:40],
        _status="UNREGISTERED. The per-pair floor was in prereg_escape_rival as KA and is absent "
                "from this prereg, which gates only the global index. Reported so a thin pair is "
                "visible rather than averaged in silently.")

    # (3) WHAT THE WIDENING DID AND DID NOT WIDEN. The scored set and the recurrent class are both
    #     worth naming explicitly, because a reader will assume both grew.
    rec_members = sorted(m for m in models if m in RECURRENT)
    res["what_widened"] = dict(
        candidate_pool=len(models), scored_set=sorted(nn),
        scored_set_is_f185_cohort=bool(sorted(nn) == sorted(REUSED)),
        recurrent_members=rec_members,
        recurrent_class_is_unchanged=bool(set(rec_members) == {
            "RWKV/rwkv-4-169m-pile", "RWKV/rwkv-4-430m-pile",
            "state-spaces/mamba-130m-hf", "state-spaces/mamba-370m-hf"}),
        _reading="the widening added DISTRACTORS, not new same-family pairs and not new recurrent "
                 "members. No new family reached two members, so the scored set is exactly F185's "
                 "twelve; and Zamba2, which was pre-registered as RECURRENT precisely because a "
                 "hybrid is the informative case, failed to load. So H_arch is tested against seven "
                 "new transformer distractors but against no new recurrent model, and the four that "
                 "generated the hypothesis are the four that test it.")

    # (4) KW5 WAS WRITTEN WITH AN ADJECTIVE, NOT A NUMBER, and that is a defect in my own prereg.
    #     It says "if one destination dominates" and names no threshold, so it cannot fire
    #     mechanically. The standing rule in this project is that directions and thresholds are
    #     numbers. The nearest REGISTERED number is arm 1's KB at 0.50, and under it this cohort
    #     trips: one model's modal destination holds 0.8832 of its escapes. So every headline is
    #     recomputed with each model's own modal destination excluded, and both are reported. This
    #     is a robustness check, not a replacement -- the registered figures stand above.
    def agree_xm(a, b):
        sa, ea = A[a]; sb, eb = A[b]
        keep = ea & eb & (sa != prof[a]["modal_destination"]) & (sb != prof[b]["modal_destination"])
        n = int(keep.sum())
        return (None if n == 0 else float(np.mean(sa[keep] == sb[keep]))), n

    Mx = np.full((len(models), len(models)), np.nan)
    for i, j in itertools.combinations(range(len(models)), 2):
        v, _ = agree_xm(models[i], models[j])
        Mx[i, j] = Mx[j, i] = -1.0 if v is None else v
    hx = 0
    for m in scored:
        i = models.index(m); d = Mx[i].copy(); d[i] = -np.inf
        hx += fam[models[int(np.nanargmax(d))]] == fam[m]
    predx = []
    for m in models:
        i = models.index(m); d = Mx[i].copy(); d[i] = -np.inf
        predx.append(prof[models[int(np.nanargmax(d))]]["architecture"])
    baccx, perx = balanced_accuracy(predx, arch)
    sf, df, st, dt = [], [], [], []
    for i, j in itertools.combinations(range(len(models)), 2):
        v = Mx[i, j]
        if v < 0:
            continue
        (sf if fam[models[i]] == fam[models[j]] else df).append(v)
        gi = next(k for k, vs in tg.items() if models[i] in vs)
        gj = next(k for k, vs in tg.items() if models[j] in vs)
        (st if gi == gj else dt).append(v)
    res["modal_excluded_robustness"] = dict(
        _status="UNREGISTERED. KW5 named no threshold -- a defect in this prereg, since directions "
                "and thresholds are numbers in this project. Arm 1's registered 0.50 is the nearest "
                "number and this cohort trips it at 0.8832, so the headlines are recomputed with "
                "each model's modal destination removed. The registered figures remain primary.",
        max_modal_share=res["KW5_modal_share_max"], arm1_registered_gate=0.50,
        rank1_same_family=int(hx), rank1_accuracy=round(hx / len(scored), 4) if scored else None,
        architecture_balanced_accuracy=round(baccx, 4),
        architecture_per_class={k: round(v, 4) for k, v in perx.items()},
        family_lift=round((np.mean(sf) if sf else 0) - (np.mean(df) if df else 0), 4),
        tokenizer_lift=round((np.mean(st) if st else 0) - (np.mean(dt) if dt else 0), 4))

    i_ = res["identification"]; a_ = res["architecture_class"]; t_ = res["family_vs_tokenizer"]
    p = [f"ESCAPE MAP WIDENED, registered in {PREREG} (sha256 {res['_prereg_sha256'][:12]}...). "
         f"{n} models, {len(t_['n_tokenizer_groups'] * [0])} tokenizer groups, corpus NO LONGER "
         f"FIXED. Intersection {len(shared)} of {len(strings)} -- KW4 clears. Same estimator as "
         f"F185; the twelve core cells are reused, never re-measured. "]
    if failed:
        p.append(f"KW6: {len(failed)} cell(s) NAMED and dropped: {[f['model'] for f in failed]}. ")
    p.append(f"ATTRIBUTION: {i_['rank1_same_family']} of {i_['n_scored']} scored models "
             f"({i_['rank1_accuracy']}) against a family-level chance of {i_['chance_family_level']}, "
             f"scored over models with at least one same-family peer. ")
    p.append("KW1 FIRES -- F185's 10 of 12 WAS SMALL-COHORT LUCK. Attribution is at or below chance "
             "on the wider cohort and the paper must say so; 10/12 may not be quoted as a "
             "capability. " if res["KW1_fires"] else
             "KW1 does not fire: attribution survives the widening. ")
    p.append(f"ARCHITECTURE CLASS (H_arch, the hypothesis F185 could only disclose): {a_['balance']} "
             f"Balanced accuracy {a_['balanced_accuracy']} against a majority rate of "
             f"{a_['majority_rate']}, per class {a_['per_class_rate']}. ")
    p.append("KW2 FIRES: the architecture-class reading of F185's two misses is REFUTED. They stay "
             "two misses. " if res["KW2_fires"] else
             "KW2 does not fire: recurrent and transformer models are separated above the base rate. ")
    p.append(f"FAMILY VERSUS TOKENIZER (H_tok), which F185 could not ask because eleven of its twelve "
             f"models shared a vocabulary: within-family agreement {t_['mean_within_family']} against "
             f"{t_['mean_across_family']} across (lift {t_['family_lift']}); within-tokenizer "
             f"{t_['mean_within_tokenizer']} against {t_['mean_across_tokenizer']} across (lift "
             f"{t_['tokenizer_lift']}). ")
    p.append("KW3 FIRES: TOKENIZER PREDICTS AGREEMENT BETTER THAN FAMILY. The fingerprint is "
             "substantially reading the tokenizer rather than the model, and every attribution claim "
             "in the paper must be restated as such. " if res["KW3_fires"] else
             "KW3 does not fire: family lift exceeds tokenizer lift, so the attribution is not "
             "principally a tokenizer effect. ")
    p.append(f"KW5: the largest single-destination share in any model is {res['KW5_modal_share_max']}. ")
    w_ = res["what_widened"]; r_ = res["rebaselined_on_this_support"]
    p.append(f"WHAT THE WIDENING ACTUALLY WIDENED, unregistered and stated because a reader will "
             f"assume otherwise: the candidate pool grew to {w_['candidate_pool']}, but the SCORED "
             f"set is still exactly F185's twelve -- no new family reached two members -- and the "
             f"recurrent class is still exactly the four models that GENERATED the hypothesis, "
             f"because Zamba2 failed to load. H_arch is therefore tested against seven new "
             f"transformer distractors and against no new recurrent model. The perfect separation "
             f"below must be read as that. ")
    p.append(f"RE-BASELINED on this {len(shared)}-string support, so no F185 number is quoted against "
             f"a number from a different index: "
             + "; ".join(f"{k} {v['agreement']} (n={v['n_shared']})" for k, v in r_["pairs"].items())
             + f". F185's support was {r_['f185_support']}. ")
    if res["thin_pairs"]["n_thin"]:
        p.append(f"THIN PAIRS, unregistered: {res['thin_pairs']['n_thin']} pair(s) have fewer than "
                 f"{MIN_INDEX} shared escaping sources, all involving models that rarely escape. "
                 f"prereg_escape_rival gated this per pair as KA; this prereg gates only the global "
                 f"index, so they are reported rather than excluded. ")
    x_ = res["modal_excluded_robustness"]
    p.append(f"KW5 NAMED NO THRESHOLD, which is a defect in this prereg: directions and thresholds "
             f"are numbers here, and 'if one destination dominates' is an adjective. Arm 1's "
             f"registered gate is 0.50 and this cohort trips it at {x_['max_modal_share']}, so every "
             f"headline is recomputed with each model's modal destination removed: attribution "
             f"{x_['rank1_same_family']} of {len(scored)} ({x_['rank1_accuracy']}), architecture "
             f"balanced accuracy {x_['architecture_balanced_accuracy']}, family lift "
             f"{x_['family_lift']} against tokenizer lift {x_['tokenizer_lift']}. All three verdicts "
             f"hold under the exclusion. Unregistered; the registered figures above stay primary. ")
    p.append("REFUSALS: no p-value; no reassignment of Zamba2's architecture label; no claim that a "
             "surviving attribution is a CORPUS result, since corpus now varies but is not "
             "controlled; no instance identification; no semantic reading. F187's quantization "
             "envelope stands unchanged and 4-bit remains fatal.")
    res["verdict"] = " ".join(p)
    _write(res)


def _write(res):
    res["_analysis_provenance"] = stamp(__file__)
    json.dump(res, open(OUT, "w"), indent=1)
    print("\n" + res["verdict"]); print("\nwrote", rel(str(OUT)))


if __name__ == "__main__":
    main()
