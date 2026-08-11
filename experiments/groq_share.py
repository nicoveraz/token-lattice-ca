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
MODELS = ["llama-3.3-70b-versatile", "llama-3.1-8b-instant", "gemma2-9b-it"]
CONSTRUCTIONS = [(2, 0.2), (2, 0.7), (3, 0.2)]
N, SETTLE = 24, 12            # small: the free tier is rate-limited and this is a pilot
SEEDS = [1, 2]
ALPHABET = [" red", " green", " blue", " yellow", " black", " white"]
SCAFFOLD_MAX = 0.15
CONCORDANT = 0.6
LOCAL_CAL = "gpt2"


def _key():
    k = os.environ.get("GROQ_API_KEY")
    if not k:
        raise SystemExit("GROQ_API_KEY not set. export it; it is never written to disk by this "
                         "script.")
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
    req = urllib.request.Request(API, data=body, headers={
        "Authorization": f"Bearer {key}", "Content-Type": "application/json"})
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
            raise SystemExit(f"HTTP {e.code} from the provider (key not shown)")
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
    sc = res.get("scaffold_rung", {})
    ok = sc and sc.get("shift", 1.0) <= SCAFFOLD_MAX
    parts.append(
        f"RUNG (the chat scaffold, measured locally BEFORE any remote call): on {LOCAL_CAL} the "
        f"same lattice reads top1 = {sc.get('raw', float('nan')):.4f} raw and "
        f"{sc.get('scaffolded', float('nan')):.4f} wrapped in the provider-style template, a shift "
        f"of {sc.get('shift', float('nan')):.4f} against a limit of {SCAFFOLD_MAX}. "
        + ("The template does not dominate the readout, so the remote numbers are about the models."
           if ok else
           "The template moves the share more than the limit allows, so a remote measurement would "
           "be reading the scaffold rather than the model. Nothing below is read."))
    if not ok:
        res["analysis"] = dict(rung_passes=False, scaffold=sc)
        res["verdict"] = " ".join(parts); return
    cons = sorted({c["construction"] for c in cells.values()})
    def rank(con, seed):
        v = []
        for m in MODELS:
            c = cells.get(f"{m}|{con}|s{seed}")
            if c is None:
                return None
            v.append(c["top1"])
        return v
    agree = [spearman(rank(c, SEEDS[0]), rank(c, SEEDS[1]))
             for c in cons if rank(c, SEEDS[0]) and rank(c, SEEDS[1])]
    agree = [a for a in agree if np.isfinite(a)]
    seed_ok = bool(agree) and float(np.mean(agree)) >= CONCORDANT
    parts.append(
        f"SECONDARY (seed stability at fixed construction, the gate F128 failed and F130 passed): "
        f"{np.mean(agree):+.3f} over {len(agree)} constructions."
        if agree else "SECONDARY: no comparable seed pairs.")
    import itertools
    live = [c for c in cons if rank(c, SEEDS[0])]
    ps = [spearman(rank(x, SEEDS[0]), rank(y, SEEDS[0])) for x, y in itertools.combinations(live, 2)]
    ps = [p for p in ps if np.isfinite(p)]
    mean_rho = float(np.mean(ps)) if ps else float("nan")
    parts.append(
        f"PRIMARY: mean pairwise agreement between the model-rankings different constructions "
        f"produce is {mean_rho:+.3f} over {len(live)} constructions, {len(MODELS)} models. "
        + ("Only three models, so a single swap moves this a long way; read it as a direction, not "
           "a measurement." if len(MODELS) < 5 else "")
        + (f"At or above {CONCORDANT}: the share ranks these models consistently across "
           f"constructions, as it did locally (F130)."
           if np.isfinite(mean_rho) and mean_rho >= CONCORDANT and seed_ok else
           "Below the registered threshold, or the seed gate failed, so no ranking claim is made."))
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
    if "scaffold_rung" not in res:
        print("  measuring the scaffold effect locally first (no remote calls yet)...", flush=True)
        res["scaffold_rung"] = scaffold_rung()
        print(f"    raw={res['scaffold_rung']['raw']:.4f} "
              f"scaffolded={res['scaffold_rung']['scaffolded']:.4f} "
              f"shift={res['scaffold_rung']['shift']:.4f} (limit {SCAFFOLD_MAX})", flush=True)
        json.dump(res, open(OUT, "w"), indent=1)
    if res["scaffold_rung"]["shift"] > SCAFFOLD_MAX:
        analyse(res); json.dump(res, open(OUT, "w"), indent=1)
        print(f"\n  -> {res['verdict']}")
        return
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
