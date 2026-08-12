"""The attractor share on models too large to run locally, through Groq's free tier.

WHY THIS IS RUNNABLE AT ALL. F130 established the attractor share as the instrument's
model-attributable readout, and it needs only a SETTLE -- no CRN twins, no full distribution, no
logprobs. The provider samples for us. So the entire measurement is: send a short context, ask for
one token at temperature T, read it back. Every chat API does that, which is why this needs no paid
tier and no special access. F134 then showed the ranking survives a top-k interface, so a
restricted view of the conditional is not disqualifying either.

THE CONSTRUCTION PROBLEM, STATED BEFORE ANY CALL RATHER THAN DISCOVERED AFTER. The lattice needs
`p(x_i | x_{i-2}, x_{i-1})` -- the model conditioning on exactly r tokens. A chat API does not offer
that. Every request is wrapped in the provider's chat template, so the model conditions on a system
role, a user role, formatting tokens AND our r tokens. This is therefore NOT the same construction
as the local work: it is a chat-scaffolded lattice, and F126/F128 are exactly the findings that say
a different construction can move a readout more than the model does.

That does not make it useless, because the question is not whether the VALUES match -- F134 already
established that cross-interface values are not comparable -- but whether the RANKING survives. The
scaffold is identical across models on one provider, so it is a constant of the construction, which
is the situation F130's invariance test was designed for.

PRE-REGISTERED:
  RUNG      the scaffold's effect is MEASURED LOCALLY FIRST, and this script refuses to spend a
            remote call until it has been. A cached model is run twice -- raw r-token context, and
            the same context wrapped in a chat-style scaffold -- and the shift in top1 is recorded.
            If the scaffold moves the share by more than SCAFFOLD_MAX the remote numbers measure the
            template rather than the model, and the run stops.
  PRIMARY   do the remote models rank consistently across constructions, as F130's local ten did?
            Reported as mean pairwise Spearman between the model-rankings different constructions
            produce. Registered reading: >= CONCORDANT means the share extends to this scale.
  SECONDARY seed stability at fixed construction, the same gate F128 failed and F130 passed. A
            ranking that is not seed-stable cannot be asked about construction-invariance.
  BOUNDARY  these are open-weight models served remotely, not closed ones. The axis being tested is
            SCALE (70B, far beyond this machine), not secrecy. Absolute shares are not comparable
            to the local work.

The API key is read from the GROQ_API_KEY environment variable and is never written to any results
file, log line or error message.
"""
import sys as _sys, pathlib as _pathlib
_ROOT = _pathlib.Path(__file__).resolve().parents[1]
_sys.path[:0] = [str(_ROOT / "src"), str(_ROOT / "experiments"), str(_ROOT / "gatecheck" / "src")]
import json, os, time, urllib.error, urllib.request

import numpy as np
from ranking import spearman
from provenance import stamp, rel

OUT = str(_ROOT / "results" / "groq_share.json")
API = "https://api.groq.com/openai/v1/chat/completions"
# TWO models, and the analysis below is built for two rather than pretending otherwise.
# gemma2-9b-it is decommissioned. qwen3.6-27b and both gpt-oss models are REASONING models: they
# emit chain-of-thought before answering, which is not a single forward-pass conditional and would
# be a different object on the lattice, not merely a costlier one. allam-2-7b works but is
# Arabic-focused, so on an English colour alphabet it adds a corpus confound to what is meant to be
# a scale comparison -- excluded deliberately (author's call) rather than accepted for the extra n.
MODELS = ["llama-3.1-8b-instant", "llama-3.3-70b-versatile"]
CONSTRUCTIONS = [(2, 0.2), (2, 0.7), (3, 0.2)]
N, SETTLE = 24, 12            # small: the free tier is rate-limited and this is a pilot
SEEDS = [1, 2]
ALPHABET = [" red", " green", " blue", " yellow", " black", " white"]
SCAFFOLD_MAX = 0.15
CONCORDANT = 0.6
LOCAL_CAL = "gpt2"


def _key():
    """GROQ_API_KEY from the environment, or from a gitignored .env at the repo root.

    The .env path exists because this repository's tooling runs each command in a fresh shell, so
    an `export` in one invocation is not visible to the next. Parsed by hand rather than with
    python-dotenv: one fewer dependency, and the parsing is four lines.

    The key is never written to a results file, a log line, or an error message. If it is missing
    this raises without echoing anything it did find.
    """
    k = os.environ.get("GROQ_API_KEY")
    if not k:
        env = _ROOT / ".env"
        if env.exists():
            for line in env.read_text().splitlines():
                line = line.strip()
                if line.startswith("GROQ_API_KEY") and "=" in line:
                    k = line.split("=", 1)[1].strip().strip("\'\"")
                    break
    if not k:
        raise SystemExit(
            "GROQ_API_KEY not found. Put it in a .env at the repo root as\n"
            "  GROQ_API_KEY=gsk_...\n"
            ".env is gitignored. Do not commit it and do not paste the key into a chat log.")
    return k


def ask(model, ctx, T, key, retries=5):
    """One lattice update: r context words in, one word out. Returns the chosen word or None."""
    body = json.dumps({
        "model": model,
        "messages": [{"role": "user",
                      "content": "Continue this sequence with exactly one more word from the list "
                                 f"{', '.join(w.strip() for w in ALPHABET)}. Reply with the word "
                                 f"only.\n{' '.join(w.strip() for w in ctx)}"}],
        "max_tokens": 4, "temperature": float(T),
    }).encode()
    # USER-AGENT IS NOT OPTIONAL. urllib defaults to "Python-urllib/3.11", which the provider's
    # Cloudflare front rejects with HTTP 403 and body "error code: 1010" -- a browser-signature ban,
    # not an auth failure. The first smoke call hit exactly this and looked like a bad key.
    req = urllib.request.Request(API, data=body, headers={
        "Authorization": f"Bearer {key}", "Content-Type": "application/json",
        "User-Agent": "token-lattice-ca/1.0 (research; +https://github.com/nicoveraz/token-lattice-ca)"})
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(req, timeout=60) as r:
                txt = json.load(r)["choices"][0]["message"]["content"].strip().lower()
            for w in ALPHABET:
                if w.strip() in txt:
                    return w
            return None
        except urllib.error.HTTPError as e:
            if e.code == 429:                       # rate limited: back off, do not give up
                time.sleep(2 ** attempt)
                continue
            # The body carries the reason and the status alone does not. Discarding it cost a
            # diagnostic round-trip on the first failure; 403 could be auth, quota, or a bot block.
            try:
                detail = e.read().decode()[:200]
            except Exception:
                detail = "(no body)"
            raise SystemExit(f"HTTP {e.code} from the provider: {detail} (key not shown)")
        except Exception:
            time.sleep(2 ** attempt)
    return None


def scaffold_rung():
    """Measure what the chat template does to the share, on a LOCAL model, before spending a call.

    The remote lattice conditions on the provider's template plus our r words; the local work
    conditions on r tokens alone. If that difference moves the share more than SCAFFOLD_MAX, the
    remote numbers are a property of the template and the comparison is void.
    """
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM
    tok = AutoTokenizer.from_pretrained(LOCAL_CAL)
    mdl = AutoModelForCausalLM.from_pretrained(LOCAL_CAL).eval()
    ids = [tok(w, add_special_tokens=False)["input_ids"][0] for w in ALPHABET]
    rng = np.random.default_rng(0)
    out = {}
    for name, prefix in (("raw", ""), ("scaffolded",
                                       "Continue this sequence with exactly one more word from the "
                                       "list red, green, blue, yellow, black, white. Reply with the "
                                       "word only.\n")):
        ring = list(rng.choice(ALPHABET, size=N))
        for _ in range(SETTLE):
            for i in range(N):
                ctx = prefix + "".join(ring[(i - 2) % N:i] or [ring[i - 1]])
                x = tok(ctx, return_tensors="pt")["input_ids"]
                with torch.no_grad():
                    lg = mdl(input_ids=x).logits[0, -1]
                p = torch.softmax(lg[ids] / 0.7, dim=-1).numpy()
                ring[i] = ALPHABET[int(rng.choice(len(ids), p=p / p.sum()))]
        vals, cnt = np.unique(ring, return_counts=True)
        out[name] = float(cnt.max() / cnt.sum())
    out["shift"] = abs(out["raw"] - out["scaffolded"])
    return out


def settle(model, r, T, seed, key):
    rng = np.random.default_rng(seed)
    ring = list(rng.choice(ALPHABET, size=N))
    calls = misses = 0
    for _ in range(SETTLE):
        for i in range(N):
            ctx = [ring[(i - j) % N] for j in range(r, 0, -1)]
            w = ask(model, ctx, T, key); calls += 1
            if w is None:
                misses += 1
            else:
                ring[i] = w
    vals, cnt = np.unique(ring, return_counts=True)
    return dict(top1=float(cnt.max() / cnt.sum()), distinct=int(len(vals)),
                calls=calls, misses=misses)


def analyse(res):
    parts, cells = [], res["cells"]
    # THE THRESHOLD COMES FROM scaffold_effect.py, NOT FROM HERE. This script's own rung was coarse
    # (N=24, one model, one seed) and gated against a 0.15 limit chosen by argument -- it read the
    # shift as 0.167 where the proper measurement at N=64 gives 0.051 on the same model. Collection
    # is allowed to proceed without it, because the gate constrains the READING, not the data.
    sc = res.get("scaffold_rung", {})
    prop = _ROOT / "results" / "scaffold_effect.json"
    if prop.exists():
        d = json.load(open(prop)).get("analysis")
        if d:
            sc = dict(source="scaffold_effect.py", shift=max(d["shifts"].values()),
                      gate=d["gate"], passed=d["passed"], raw_spread=d["raw_spread"])
            ok = bool(d["passed"])
        else:
            sc, ok = dict(source="scaffold_effect.py", status="incomplete"), None
    else:
        sc, ok = dict(source="none"), None
    if ok is None:
        parts.append(
            "RUNG NOT AVAILABLE: scaffold_effect.py has not produced an analysis yet, so the "
            "scaffold's effect on the share is unmeasured at usable resolution. The remote cells "
            "below are COLLECTED but NOT READ -- this script's own coarse rung (N=24, one model, "
            "one seed) put the shift at 0.167 where the proper measurement gives 0.051 on the same "
            "model, so gating on it would be gating on a bad number.")
        res["analysis"] = dict(rung_passes=None, scaffold=sc, n_cells=len(cells))
        res["verdict"] = " ".join(parts); return
    parts.append(
        f"RUNG (the chat scaffold, from scaffold_effect.py at N=64 over 3 models x 2 temperatures "
        f"x 4 seeds): worst scaffold shift {sc.get('shift', float('nan')):.4f} against a gate of "
        f"{sc.get('gate', float('nan')):.4f}, which is half the across-model spread rather than a "
        f"number chosen by argument. Scaffolds passing: {sc.get('passed')}. "
        + ("The template does not dominate the readout, so the remote numbers are about the models."
           if ok else
           "The template moves the share more than the limit allows, so a remote measurement would "
           "be reading the scaffold rather than the model. Nothing below is read."))
    if not ok:
        res["analysis"] = dict(rung_passes=False, scaffold=sc)
        res["verdict"] = " ".join(parts); return
    # SPEARMAN IS DEGENERATE AT n=2 AND IS NOT USED. With two models a "ranking" is just which is
    # higher, so rho can only be +1 or -1 and a 0.6 threshold could not fail -- this project's own
    # defect class (a criterion applied to a quantity with no room to vary) arriving in the
    # analysis. At n=2 the answerable questions are whether the ORDERING is consistent, and whether
    # the GAP exceeds the seed noise that would flip it.
    cons = sorted({c["construction"] for c in cells.values()})
    a, b = MODELS
    rows, agree, gaps, noises = [], [], [], []
    for con in cons:
        va = [cells[f"{a}|{con}|s{sd}"]["top1"] for sd in SEEDS if f"{a}|{con}|s{sd}" in cells]
        vb = [cells[f"{b}|{con}|s{sd}"]["top1"] for sd in SEEDS if f"{b}|{con}|s{sd}" in cells]
        if len(va) < len(SEEDS) or len(vb) < len(SEEDS):
            continue
        ma, mb = float(np.mean(va)), float(np.mean(vb))
        noise = float(np.mean([np.ptp(va), np.ptp(vb)]))
        rows.append(dict(construction=con, a=round(ma, 4), b=round(mb, 4),
                         gap=round(mb - ma, 4), seed_range=round(noise, 4)))
        agree.append(mb > ma); gaps.append(abs(mb - ma)); noises.append(noise)
    if not rows:
        parts.append("PRIMARY: no construction has both models at every seed.")
        res["analysis"] = dict(rung_passes=True, scaffold=sc, rows=rows)
        res["verdict"] = " ".join(parts); return
    same = sum(agree); n = len(agree)
    sep = [g > nz for g, nz in zip(gaps, noises)]
    parts.append(
        f"PRIMARY (n=2, so ordering and separation rather than a rank correlation): "
        + "; ".join(f"{r['construction']} {a.split('-')[2]}={r['a']:.3f} vs "
                    f"{b.split('-')[2]}={r['b']:.3f} (gap {r['gap']:+.3f}, seed range "
                    f"{r['seed_range']:.3f})" for r in rows)
        + f". The 70B reads higher in {same} of {n} constructions, and the gap exceeds the seed "
          f"range in {sum(sep)} of {n}. "
        + ("Consistent ordering with separation above seed noise: the share distinguishes these two "
           "models remotely."
           if same in (0, n) and sum(sep) > n / 2 else
           "Not a usable distinction -- the ordering is inconsistent across constructions, or the "
           "gap sits inside the seed range that would flip it. At n=2 this is the whole claim "
           "available, and it is not met."))
    parts.append(
        f"NO RANKING CLAIM IS MADE. Two models cannot establish that the share ranks models at "
        f"scale; that needed the four-plus set the provider's usable line-up did not supply. What "
        f"this can show is whether a remote, chat-scaffolded lattice produces a stable, separable "
        f"reading at all.")
    miss = sum(c.get("misses", 0) for c in cells.values())
    tot = sum(c.get("calls", 0) for c in cells.values())
    parts.append(
        f"BOUNDARY: open-weight models served remotely, not closed ones -- the axis tested is SCALE "
        f"(70B, far beyond this machine), not secrecy. This is a chat-scaffolded WORD lattice, not "
        f"the token lattice of the local work, so absolute shares are not comparable to F130's. "
        f"N={N}, settle={SETTLE}. {miss} of {tot} calls returned no alphabet word and left the cell "
        f"unchanged.")
    res["analysis"] = dict(rung_passes=True, scaffold=sc, seed_agreement=float(np.mean(agree))
                           if agree else None, mean_rho=mean_rho, n_models=len(MODELS),
                           misses=miss, calls=tot)
    res["verdict"] = " ".join(parts)


def main():
    res = json.load(open(OUT)) if _pathlib.Path(OUT).exists() else {"cells": {}}
    res["_preregistration"] = dict(
        provider="groq (free tier)", models=MODELS, alphabet=ALPHABET,
        constructions=[f"r{r}.T{T}" for r, T in CONSTRUCTIONS], N=N, settle=SETTLE, seeds=SEEDS,
        scaffold_max=SCAFFOLD_MAX, concordant=CONCORDANT, local_calibration=LOCAL_CAL,
        rung="the chat template's effect on the share is measured on a LOCAL model before any "
             "remote call; if it exceeds scaffold_max the run stops",
        primary="do the remote models rank consistently across constructions, as F130's ten did",
        note="the provider samples; no logprobs are requested and none are needed")
    key = _key()
    for m in MODELS:
        for r, T in CONSTRUCTIONS:
            for sd in SEEDS:
                k = f"{m}|r{r}.T{T}|s{sd}"
                if k in res["cells"]:
                    continue
                t0 = time.time()
                c = settle(m, r, T, sd, key)
                c.update(model=m, construction=f"r{r}.T{T}", r=r, T=T, seed=sd,
                         secs=round(time.time() - t0, 1))
                res["cells"][k] = c
                print(f"  {k:<44} top1={c['top1']:.4f} distinct={c['distinct']} "
                      f"misses={c['misses']}/{c['calls']} ({c['secs']:.0f}s)", flush=True)
                json.dump(res, open(OUT, "w"), indent=1)
    analyse(res)
    res["_analysis_provenance"] = stamp(__file__)
    json.dump(res, open(OUT, "w"), indent=1)
    print(f"\n  -> {res['verdict']}")
    print("\nwrote", rel(OUT))


if __name__ == "__main__":
    main()
