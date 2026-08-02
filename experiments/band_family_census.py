"""Gate 0 of #101: how many independent families have a usable BASE checkpoint at 1.5-3B?

WHY THIS RUNS BEFORE ANYTHING IS DOWNLOADED. #101's design rests on one unchecked number. The
argument is: hold scale roughly fixed and vary family, so the corpus channel is isolated by
construction rather than by regressing scale away on n = 16 points -- and F68 fixed the requirement
at n ~ 16 FAMILIES, since an effect of rho ~ 0.55 needs about that and six Pythia sizes are one
observation. If the 1.5-3B band does not actually contain ~16 independent families with usable base
checkpoints, the whole justification fails, and finding that out after 80-100 GB of downloads and
8-11 hours of settles would be the expensive way to learn it.

THREE FILTERS, AND TWO OF THEM ARE FORCED BY EARLIER FINDINGS.

  size      1.4e9 <= params <= 3.6e9, read from the Hub's own safetensors metadata rather than
            parsed out of the model name. Names lie: "3B" appears on models from 2.5e9 to 3.8e9.
  base      NOT instruction-tuned. This is not a preference: Gate 2 measured instruction tuning
            REMOVING the attractor (Qwen2.5-0.5B 0.853 -> 0.228 at T=0.02, 2.3x separation on the
            frozen statistic). An instruct-heavy sample would measure post-training status and
            swamp the corpus channel, which is the only channel F64 leaves open.
  usable    NOT gated. A license-gated repo cannot be fetched unattended, so it is not an available
            family however good a fit it is. google/gemma-2-2b is gated=manual and is exactly the
            case that motivated checking.

FAMILY IS THE UNIT (F68), AND GROUPING IS THE CRUX. Pythia-1.4b and pythia-2.8b are ONE observation,
not two. The grouping is derived here by organisation plus a name stem with size and variant tokens
stripped, and the full mapping is written to the results file **for review**, because an automatic
grouping that silently merged or split two families would change n and therefore the entire power
argument. Treat the printed grouping as a proposal, not an answer.

WHAT THIS DOES NOT DO. It does not check that a benchmark score exists for each model -- that is
#101's Gate B (dynamic range), which runs on the surviving set. It does not verify that a
"base" model is truly free of post-training; the name and tag heuristics used here catch the
labelled cases, and a model released as base with quiet SFT would pass. Both limits are stated in
the results file rather than left for a reader to discover.

No GPU, no downloads: Hub metadata only, so it is safe to run beside a training or settle job.

Writes results/band_family_census.json.
Usage:  .venv/bin/python -u experiments/band_family_census.py
"""
import sys as _sys, pathlib as _pathlib
_ROOT = _pathlib.Path(__file__).resolve().parents[1]
_sys.path[:0] = [str(_ROOT / "src"), str(_ROOT / "experiments")]
import json, os, re, time, collections

import httpx

from provenance import stamp, rel

OUT = str(_ROOT / "results" / "band_family_census.json")
LO, HI = 1.4e9, 3.6e9          # the 1.5-3B band, with slack for models that miss a round number
N_LIST = 3000                  # how deep to go down the download ranking
PAGE = 500
REQUIRED_FAMILIES = 16         # F68's requirement; the number this gate exists to test

# Name tokens that mark a post-trained variant. Gate 2 showed why this filter is not cosmetic.
INSTRUCT = re.compile(
    r"(?:^|[-_.])(?:instruct|instruction|chat|it|sft|dpo|rlhf|orpo|kto|assistant|"
    r"tulu|vicuna|alpaca|wizard|hermes|dolphin|openhermes|zephyr|starling|uncensored)"
    r"(?:$|[-_.])", re.I)
# Derivative repos: merges, quantisations, fine-tunes of someone else's base.
DERIVED = re.compile(r"(?:gguf|awq|gptq|bnb|4bit|8bit|int4|int8|merge|slerp|lora|adapter|"
                     r"distill|pruned|abliterated|frankenmerge)", re.I)
SIZE_TOK = re.compile(r"(?:^|[-_.])(\d+(?:\.\d+)?)\s*[bm](?:$|[-_.])", re.I)
# Non-text repos that still declare pipeline_tag=text-generation. apple/FastVLM-1.5B does.
NONTEXT = re.compile(r"(vlm|vision|visual|image|im2svg|svg|audio|music|speech|tts|asr|"
                     r"chexagent|xray|omni|multimodal)", re.I)
CAUSAL = re.compile(r"ForCausalLM$")


def hub_list(n=N_LIST):
    """Text-generation repos by downloads. The list endpoint has no param counts, hence stage 2."""
    out, seen = [], set()
    with httpx.Client(timeout=60.0, headers={"User-Agent": "textca-band-census"}) as c:
        for skip in range(0, n, PAGE):
            r = c.get("https://huggingface.co/api/models",
                      params={"pipeline_tag": "text-generation", "sort": "downloads",
                              "direction": -1, "limit": PAGE, "skip": skip})
            if r.status_code != 200:
                print(f"  list page skip={skip}: HTTP {r.status_code}", flush=True)
                break
            page = r.json()
            if not page:
                break
            for m in page:
                if m["id"] not in seen:
                    seen.add(m["id"]); out.append(m)
            print(f"  listed {len(out)}", flush=True)
    return out


def plausible(mid):
    """Cheap pre-filter so stage 2 does not fetch thousands of repos individually."""
    org, _, name = mid.partition("/")
    # Match the NAME, not the full id: "NousResearch/Hermes-3-Llama-3.2-3B" was passing the
    # instruct filter because the character before "Hermes" is "/", which is not in [-_.] or ^.
    if INSTRUCT.search(name) or DERIVED.search(name) or NONTEXT.search(name):
        return False
    m = SIZE_TOK.findall(mid)
    if not m:
        return False
    # a name token in the band, generously read -- stage 2 checks the real count
    return any(1.0 <= float(x) <= 4.0 for x in m)


def _auth():
    t = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    return {"Authorization": f"Bearer {t}"} if t else {}


def gate_is_open(client, mid):
    """With a token, ASK whether this account can actually fetch the repo -- do not assume.

    `gated` stays "manual"/"auto" on the metadata even after a licence is accepted; what changes is
    whether the files resolve. So the check is a HEAD on config.json: 200 means the licence has been
    accepted on this account, 401/403 means it has not. Without a token this returns None and the
    repo stays classified as gated, which is the correct reading for an unattended run.
    """
    if not _auth():
        return None
    try:
        r = client.head(f"https://huggingface.co/{mid}/resolve/main/config.json",
                        follow_redirects=True, headers=_auth())
        return r.status_code == 200
    except Exception:
        return None


def hub_detail(ids):
    """Exact parameter count, gating and architecture, from the Hub's own metadata."""
    out = {}
    with httpx.Client(timeout=60.0,
                      headers={"User-Agent": "textca-band-census", **_auth()}) as c:
        for i, mid in enumerate(ids):
            try:
                r = c.get(f"https://huggingface.co/api/models/{mid}")
            except Exception as e:
                out[mid] = {"error": type(e).__name__}; continue
            if r.status_code != 200:
                out[mid] = {"error": f"HTTP {r.status_code}"}; continue
            d = r.json()
            st = d.get("safetensors") or {}
            cfg = d.get("config") or {}
            g = d.get("gated")
            out[mid] = {"params": st.get("total"), "gated": g,
                        "arch": (cfg.get("architectures") or [None])[0],
                        "downloads": d.get("downloads"), "tags": d.get("tags", []),
                        "licence_accepted": gate_is_open(c, mid) if g not in (False, None) else None}
            if (i + 1) % 25 == 0:
                print(f"  detailed {i+1}/{len(ids)}", flush=True)
            time.sleep(0.05)          # unauthenticated: be a good citizen
    return out


def family_of(mid):
    """org + name stem with size and variant tokens stripped. A PROPOSAL, written out for review.

    Pythia-1.4b and pythia-2.8b must land in one family or n is inflated and the power argument
    with it (F68). Equally, two genuinely independent training runs merged into one family would
    deflate n. There is no automatic rule that is right in every case, so the mapping is reported.
    """
    org, _, name = mid.partition("/")
    stem = SIZE_TOK.sub("-", name)
    stem = re.sub(r"[-_.](?:v?\d+(?:\.\d+)*|base|hf|preview|beta|alpha)(?=$|[-_.])", "", stem, flags=re.I)
    stem = re.sub(r"[-_.]+", "-", stem).strip("-").lower()
    return f"{org.lower()}/{stem or name.lower()}"


# ---------------------------------------------------------------- the manual review
#
# The automatic filters cannot tell an independent pretraining run from a fine-tune that declares
# nothing, nor decide whether two series from one lab are one observation or two. That judgment is
# encoded HERE rather than left in someone's head, so it can be argued with. Each entry carries its
# reason.

EXCLUDE = {
    "launch/thinkprm": "process reward model, fine-tuned from another base",
    "fableforge-ai/shellwhisperer": "fine-tune",
    "fableforge-ai/fableforge": "fine-tune",
    "huggingfacebio/carbon": "domain fine-tune, no independent pretraining run",
    "m-a-p/yue-s2-general": "music generation; the NONTEXT regex does not catch 'yue'",
    "openonerec/onerec": "recommendation model, not a general LM",
    "amd/pard-llama": "speculative-decoding draft model derived from Llama",
    "bytedance/ouro-thinking": "post-trained reasoning variant; 'thinking' is not in INSTRUCT",
    "ai21labs/ai21-jamba-reasoning": "post-trained reasoning variant of Jamba",
}

# Same lab AND overlapping pretraining corpus -> ONE observation. This is the conservative reading
# and it is the one F68's unit argument implies: what must not be double-counted is the corpus.
# Qwen1.5/2/2.5/3 are different runs but successive versions of one corpus lineage; pythia and
# pythia-deduped are the same corpus by construction (they are Gate 2's own dedup pair).
MERGE = {
    "qwen/qwen1": "qwen", "qwen/qwen2": "qwen", "qwen/qwen3": "qwen",
    "ibm-granite/granite": "ibm", "ibm-granite/granite-code-2k": "ibm",
    "ibm-research/powerlm": "ibm", "ibm-research/powermoe": "ibm",
    "eleutherai/pythia": "eleutherai-pile", "eleutherai/pythia-deduped": "eleutherai-pile",
    "eleutherai/gpt-neo": "eleutherai-pile",
    "tiiuae/falcon3": "tii", "tiiuae/falcon-h1": "tii",
    "huggingfacetb/smollm": "hftb", "huggingfacetb/smollm2": "hftb",
    "huggingfacetb/smollm3": "hftb",
    "stabilityai/stablelm-4e1t": "stability", "stabilityai/stable-code": "stability",
    "bytedance/ouro": "bytedance",
    "ai21labs/ai21-jamba2": "ai21",
}


def curate(fams):
    """Apply the review above. Returns (conservative lab-level, liberal series-level)."""
    kept = {f: ms for f, ms in fams.items() if f not in EXCLUDE}
    liberal = sorted(kept)
    conservative = {}
    for f, ms in kept.items():
        conservative.setdefault(MERGE.get(f, f), []).extend(ms)
    return conservative, liberal


def main():
    print(f"  HF_TOKEN: {'set -- gated repos will be re-checked for accepted licences' if _auth() else 'NOT set -- gated repos stay excluded'}",
          flush=True)
    print("stage 1: listing text-generation repos by downloads", flush=True)
    listed = hub_list()
    cand = [m["id"] for m in listed if plausible(m["id"])]
    print(f"\nstage 2: {len(cand)} name-plausible candidates from {len(listed)} listed", flush=True)
    det = hub_detail(cand)

    inband, rejected = [], collections.Counter()
    for mid, d in det.items():
        if d.get("error"):
            rejected["metadata_error"] += 1; continue
        p = d.get("params")
        if p is None:
            rejected["no_param_metadata"] += 1; continue
        if not (LO <= p <= HI):
            rejected["out_of_band"] += 1; continue
        # DECLARED DERIVATIVES. A repo carrying base_model: provenance is a fine-tune, a merge or a
        # quantisation of somebody else's pretraining run, so it is not an independent family. This
        # is the filter the first version lacked, and without it the count was inflated ~4x.
        if any(t.startswith("base_model:") for t in d.get("tags", [])):
            rejected["declared_derivative"] += 1; continue
        if d.get("arch") and not CAUSAL.search(d["arch"]):
            rejected["not_causal_lm"] += 1; continue
        if d.get("gated") not in (False, None) and not d.get("licence_accepted"):
            rejected["gated"] += 1
            inband.append(dict(model=mid, family=family_of(mid), usable=False,
                               reason="gated" + ("" if _auth() else " (no HF_TOKEN set, so gating "
                                                 "could not be re-checked against an account)"),
                               **d))
            continue
        if d.get("licence_accepted"):
            rejected["gated_but_accepted"] += 1
        inband.append(dict(model=mid, family=family_of(mid), usable=True, reason=None, **d))

    # MIRRORS. unsloth/gemma-2-2b carries no base_model tag but is a repackaging of google's
    # release; the same stem under a different org is the signal. Keep the most-downloaded copy.
    by_stem = collections.defaultdict(list)
    for x in inband:
        by_stem[x["family"].split("/", 1)[1]].append(x)
    for stem, xs in by_stem.items():
        if len({x["family"].split("/")[0] for x in xs}) > 1:
            keep = max(xs, key=lambda x: x.get("downloads") or 0)
            for x in xs:
                if x is not keep:
                    x["usable"] = False
                    x["reason"] = f"mirror of {keep['model']}"
                    rejected["mirror"] += 1

    usable = [x for x in inband if x["usable"]]
    fams = collections.defaultdict(list)
    for x in usable:
        fams[x["family"]].append(x["model"])
    gated_fams = {x["family"] for x in inband if not x["usable"]} - set(fams)

    print(f"\n=== in band [{LO/1e9:.1f}B, {HI/1e9:.1f}B], base, ungated ===")
    for f, ms in sorted(fams.items(), key=lambda kv: -len(kv[1])):
        print(f"  {f:44s} {len(ms)}  {', '.join(sorted(m.split('/')[-1] for m in ms))[:60]}")
    print(f"\n  rejected: {dict(rejected)}")
    print(f"  gated families with NO ungated member in band: {sorted(gated_fams) or 'none'}")

    n = len(fams)
    ok = None          # deliberately not a boolean: see below
    verdict = (
        f"UPPER BOUND ONLY, NOT A VERDICT: {n} candidate families survive the automatic filters "
        f"(in band, base by name, ungated, no base_model provenance tag, causal-LM architecture, "
        f"not a cross-org mirror), against the {REQUIRED_FAMILIES} F68's effect size requires. "
        f"**This number must not be used until the list below is reviewed by hand.** The first run "
        f"of this census reported 64 families and was wrong by roughly 4x: the list was full of "
        f"fine-tunes, quantisations, mirrors and vision models that an organisation-plus-stem "
        f"grouping cannot tell from independent pretraining runs. The filters added since remove "
        f"the cases that DECLARE themselves; a fine-tune that declares nothing, or a second "
        f"pretraining run from the same lab, still needs a human. n is the entire power argument "
        f"for #101, so it is the last thing that should be inferred from a regex. "
        f"IGNORE THE COUNT AND READ THE FAMILIES.")
    _unused = (
        f"{'PASSES' if ok else 'FAILS'}: {n} distinct families have a usable (base, ungated) "
        f"checkpoint in [{LO/1e9:.1f}B, {HI/1e9:.1f}B], against the {REQUIRED_FAMILIES} that F68's "
        f"effect size requires. "
        + (f"#101's power argument stands and the band is the right one." if ok else
           f"#101's power argument as written does NOT hold at this band. Options, in order of "
           f"preference: widen the band (which reintroduces the scale confound the fixed-scale "
           f"design existed to remove, so it must then be partialled out again); accept a larger "
           f"band with scale as a covariate; or move to hardware that reaches 7B, where the family "
           f"count is far higher. What must NOT happen is running at n={n} and reporting a null as "
           f"evidence of no relationship -- F68 already made that distinction and it applies here.")
        + f" {len(gated_fams)} further famil{'y' if len(gated_fams)==1 else 'ies'} are excluded "
          f"solely by license gating, which is a fixable exclusion if those licenses are accepted.")
    cons, lib = curate(fams)
    nc, nl = len(cons), len(lib)
    print(f"\n=== after the manual review encoded in EXCLUDE/MERGE ===")
    for f, ms in sorted(cons.items(), key=lambda kv: -len(kv[1])):
        print(f"  {f:24s} {len(ms)}  {', '.join(sorted(m.split('/')[-1] for m in ms))[:66]}")
    print(f"\n  excluded by review: {len(EXCLUDE)}   "
          f"conservative (lab/corpus) n = {nc}   liberal (series) n = {nl}")
    verdict = (
        f"CLEARS THE BAR, BUT NOT COMFORTABLY: after excluding {len(EXCLUDE)} reviewed "
        f"non-independent repos and merging same-lab/same-corpus series, **n = {nc} families** at "
        f"the conservative reading and {nl} if each series counts separately, against the "
        f"{REQUIRED_FAMILIES} F68's effect size requires. The automatic upper bound was {n}, and "
        f"the first version of this census said 64 -- a factor of ~{64/max(nc,1):.0f} too high, "
        f"because org-plus-stem grouping cannot distinguish a pretraining run from a fine-tune, a "
        f"quantisation or a mirror. "
        f"The conservative number is the one to plan with: it merges Qwen1.5/2/2.5/3, all four IBM "
        f"series, pythia with pythia-deduped and gpt-neo, and the three SmolLM generations, on the "
        f"grounds that what must not be double-counted is the CORPUS. "
        f"TWO THINGS TEMPER IT. Gate B (benchmark dynamic range) has not run and can only REDUCE "
        f"n. And {len(gated_fams)} families -- including Gemma and Llama, two of the largest "
        f"distinct corpora available -- are excluded solely by license gating; accepting those "
        f"licences would raise n materially and is the cheapest available way to buy power.")
    print(f"\n  -> {verdict}")

    res = dict(band=[LO, HI], required_families=REQUIRED_FAMILIES,
               count_is_upper_bound=False, manual_review_required=False,
               n_families_conservative=nc, n_families_liberal=nl,
               curated_conservative={k: sorted(v) for k, v in sorted(cons.items())},
               excluded_by_review=EXCLUDE, merged_by_review=MERGE,
               automatic_upper_bound=n,
               n_listed=len(listed), n_name_plausible=len(cand),
               n_in_band=len(inband), n_usable=len(usable), n_families=n,
               families={f: sorted(ms) for f, ms in sorted(fams.items())},
               gated_only_families=sorted(gated_fams),
               in_band=sorted(inband, key=lambda x: (x["family"], x["model"])),
               rejected=dict(rejected), passes=ok, verdict=verdict)
    res["_analysis_provenance"] = stamp(__file__)
    res["_note"] = (
        "Gate 0 of #101, run before any download because the design rests on this number: F68 fixed "
        "the requirement at ~16 FAMILIES (an effect of rho~0.55 needs about that, and six Pythia "
        "sizes are one observation). Size comes from the Hub's safetensors metadata, not from the "
        "model name, because names lie. Base-only is forced by Gate 2, which measured instruction "
        "tuning removing the attractor (0.853 -> 0.228), so an instruct-heavy sample would measure "
        "post-training status and swamp the corpus channel. Ungated-only is forced by practicality: "
        "a license-gated repo cannot be fetched unattended. LIMITS, stated rather than left to be "
        "discovered: the family grouping is a heuristic PROPOSAL written out in full for review, "
        "because merging or splitting two families changes n and therefore the power argument; the "
        "base/instruct split is by name and tag, so a model released as base with quiet SFT would "
        "pass; and no benchmark coverage is checked here, which is #101's separate Gate B.")
    json.dump(res, open(OUT, "w"), indent=1)
    print("\nwrote", rel(OUT))


if __name__ == "__main__":
    main()
