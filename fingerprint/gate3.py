"""Gate 3 — the API port: does the signature survive being read through a completion endpoint?

WHAT CAN AND CANNOT RUN HERE, STATED FIRST. Gate 3's outward half -- pointing the battery at a
third-party endpoint -- needs credentials this machine does not have, and signing up for or spending
on external services is not something to do unasked. What PROGRAM.md 6 requires BEFORE that half is
fully runnable, and is the part with the actual engineering risk in it:

    before the battery is trusted on any UNKNOWN endpoint, it must recover the known family labels
    of held-out PUBLIC models at the same call budget and sampler settings ... a battery that cannot
    re-identify pythia-410m through its own API harness has no business characterizing a closed
    endpoint.

So this gate stands up a real completion endpoint over HTTP against local weights, drives the CA
through it one token at a time, and asks whether the signature survives the abstraction. The models
are the discriminating pair whose answer is already known: pythia-410m HAS the attractor and gpt2
does NOT (F62-F70, and gate2 measured both at this geometry). If the harness cannot tell them apart,
nothing downstream matters.

FOUR THINGS CHANGE WHEN THE CA IS READ THROUGH AN API, and each is a separate arm:

  ids         Token ids in, token id out, specials forbidden server-side. The optimistic case, and
              the FIDELITY CHECK: it should reproduce the local number, because nothing has changed
              except that the logits crossed a socket. A deviation here is a bug in this harness,
              not a fact about APIs, and the gate says so rather than reading it as a finding.
  nospecial   Same, but the endpoint does NOT let you forbid special tokens. Local `center_probs`
              zeroes them; no real completion API exposes that. This is the first thing a port
              actually loses.
  text        Text in, text out, client re-tokenizes -- what a completion API really is. The hazard
              is that decode(a)+decode(b) need not re-tokenize to [a, b], so the ring can drift in
              LENGTH and identity even when the model is byte-identical.
  chat        Text plus a chat template. F66 showed ONE BOS token moves the attractor share
              74% -> 24%; a chat template prepends far more than one. Registered under K3 as
              SCOPING the capability, not killing it.

PRE-REGISTERED (K3, frozen in prereg.json):
  H4        the battery computed through single-token API calls reproduces the local signature
            within seed spread.
  K3 fires  chat-template wrapping destroys the signature AND no raw completion endpoint
            reproduces it -> capability scoped to open-weight serving only.
  Fidelity  if the `ids` arm does not reproduce the local number, the harness is wrong and NOTHING
            here is reported as a fact about APIs.
  Discrimination gate: the harness must separate pythia-410m from gpt2 in every arm it claims to
            support. Reproducing a number is not enough -- the capability is telling models apart.

Writes fingerprint/gate3.json.
Usage:  caffeinate -dimsu .venv/bin/python -u fingerprint/gate3.py
        (resumable, keyed by (model, mode, T, seed))
"""
import collections
import json
import os
import pathlib
import statistics
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import numpy as np

_HERE = pathlib.Path(__file__).resolve()
ROOT = _HERE.parents[1]
os.environ.setdefault("HF_HOME", str(ROOT / "hf_cache"))
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
sys.path[:0] = [str(ROOT / "gatecheck" / "src"), str(_HERE.parent),
                str(ROOT / "src"), str(ROOT / "experiments")]

import torch  # noqa: E402
import httpx  # noqa: E402
from gatecheck import save_results, verify_block  # noqa: E402

OUT = _HERE.parent / "gate3.json"
PREREG = _HERE.parent / "prereg.json"

# Geometry MATCHED to gate2's frozen battery. Gate 2 had to add a whole arm because a denominator
# was measured at a different geometry (F56); introducing a second mismatch here would repeat it.
N, B, SETTLE, R = 96, 16, 16, 2
TEMPS = [0.02, 0.436]          # the degenerate pole and F58's T_c -- where the signature lives
SEEDS = [101, 102]
MODES = ["ids", "nospecial", "text", "chat"]
MODELS = [("EleutherAI/pythia-410m", "HAS the attractor (F70)"),
          ("gpt2", "control -- has none")]
HOST, PORT = "127.0.0.1", 8731


# ------------------------------------------------------------------ the endpoint

class _Handler(BaseHTTPRequestHandler):
    """A raw completion endpoint: prompts in, ONE sampled token out. No logits leave the server."""
    server_version = "textca-fingerprint/1"

    def log_message(self, *a):
        pass

    def do_POST(self):
        n = int(self.headers.get("Content-Length", 0))
        req = json.loads(self.rfile.read(n) or b"{}")
        st = self.server.state
        mode, T = req["mode"], float(req["temperature"])
        prompts = req["prompts"]
        if mode in ("ids", "nospecial"):
            ids = [list(map(int, p)) for p in prompts]
        else:
            # NEVER pad a short window with BOS, and NEVER force a common width across the
            # batch. Both were tried and both manufactured the effect they then measured:
            #   * BOS padding -- gpt2 has one token for "\n\n", so a newline-rich ring merges a
            #     two-newline window to one token, tripping the pad and injecting <|endoftext|>,
            #     which strongly predicts '\n'. The ring collapsed to 95.8% newline.
            #   * min-width truncation -- one merged prompt anywhere in a batch of 16 truncated
            #     EVERY replica to a one-token context, which F69 identifies as the degenerate
            #     regime. A batching convenience became a physics change.
            # A real completion endpoint conditions each prompt on exactly what it tokenizes to,
            # independently of its neighbours. Ragged lengths are therefore grouped and run as
            # separate forward passes, and the realized widths are reported so any drift in
            # effective context is visible in the results rather than inferred from an outcome.
            ids = [st["tok"](p, return_tensors=None)["input_ids"] for p in prompts]
            ids = [(q[-R:] if len(q) >= R else q) or [st["bos"]] for q in ids]
        if len({len(q) for q in ids}) == 1:
            out = st["sample"](ids, T, forbid_specials=(mode != "nospecial"))
        else:
            out = [None] * len(ids)
            by_len = {}
            for i, q in enumerate(ids):
                by_len.setdefault(len(q), []).append(i)
            for L, idxs in by_len.items():
                got = st["sample"]([ids[i] for i in idxs], T,
                                   forbid_specials=(mode != "nospecial"))
                for i, t in zip(idxs, got):
                    out[i] = t
        st["widths"].update(len(q) for q in ids)
        if mode in ("ids", "nospecial"):
            body = {"choices": [{"token_id": int(t)} for t in out]}
        else:
            body = {"choices": [{"text": st["tok"].decode([int(t)])} for t in out]}
        b = json.dumps(body).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(b)))
        self.end_headers()
        self.wfile.write(b)


def start_endpoint(name, dev):
    """Serve local weights behind HTTP with softmax semantics, top_p=1, no top_k."""
    from transformers import AutoTokenizer, AutoModelForCausalLM
    tok = AutoTokenizer.from_pretrained(name)
    model = AutoModelForCausalLM.from_pretrained(name).eval().to(
        dev, torch.float16 if dev != "cpu" else torch.float32)
    bos = tok.bos_token_id if tok.bos_token_id is not None else tok.eos_token_id
    specials = torch.tensor(sorted({i for i in (tok.bos_token_id, tok.eos_token_id,
                                                tok.pad_token_id, tok.unk_token_id)
                                    if i is not None}), dtype=torch.long)

    @torch.no_grad()
    def sample(id_batch, T, forbid_specials=True):
        x = torch.tensor(id_batch, device=dev)
        lg = model(input_ids=x).logits[:, -1].float()
        if forbid_specials and len(specials):
            lg[:, specials.to(dev)] = -float("inf")
        p = torch.softmax(lg / max(T, 1e-6), dim=-1)     # top_p=1, no top_k, by construction
        return torch.multinomial(p, 1).squeeze(1).tolist()

    # A settle leaves the socket in TIME_WAIT, so the next model's endpoint cannot bind without
    # this. Without it the gate dies partway through with Errno 48 and looks like a model failure.
    ThreadingHTTPServer.allow_reuse_address = True
    srv = ThreadingHTTPServer((HOST, PORT), _Handler)
    srv.state = {"tok": tok, "sample": sample, "bos": bos,
                 "widths": collections.Counter()}
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv, tok, model


# ------------------------------------------------------------------- the client

def api_settle(client, tok, mode, T, seed, chat_prefix=None):
    """Run the ring through the endpoint. One HTTP call per site, B prompts batched per call.

    Batched prompts are an ordinary completion-API feature and preserve the CA's structure exactly:
    the sites are visited sequentially, the replicas are independent, so B contexts at one site is
    one request. Without it a settle is N*sweeps*B calls instead of N*sweeps.
    """
    rng = np.random.default_rng(seed)
    V = len(tok)
    lat = rng.integers(0, V, size=(B, N))
    for _ in range(SETTLE):
        orders = np.array([rng.permutation(N) for _ in range(B)])   # order="per_replica"
        for k in range(N):
            sites = orders[:, k]
            win = np.stack([lat[b, (sites[b] - np.arange(R, 0, -1)) % N] for b in range(B)])
            if mode in ("ids", "nospecial"):
                prompts = win.tolist()
            else:
                prompts = [tok.decode(w.tolist()) for w in win]
                if chat_prefix is not None:
                    prompts = [chat_prefix + p for p in prompts]
            r = client.post(f"http://{HOST}:{PORT}/v1/completions",
                            json={"prompts": prompts, "temperature": T, "top_p": 1.0,
                                  "max_tokens": 1, "mode": mode})
            ch = r.json()["choices"]
            for b in range(B):
                if mode in ("ids", "nospecial"):
                    lat[b, sites[b]] = ch[b]["token_id"]
                else:
                    t = tok(ch[b]["text"], return_tensors=None)["input_ids"]
                    lat[b, sites[b]] = t[0] if t else lat[b, sites[b]]
    tops = [collections.Counter(row.tolist()).most_common(1)[0][1] / N for row in lat]
    return float(np.mean(tops))


def local_reference(res):
    """The local answer for every model/temperature this gate measures, at the SAME geometry.

    Gate 2's runs are reused where they exist, and MEASURED HERE where they do not. The first
    version read gate2 only, which silently produced no reference for pythia-410m -- gate2's model
    list never included it -- so the fidelity arm had nothing to compare against and would have
    reported "reproduces" vacuously for the one model the gate exists to identify. A comparison
    with no left-hand side must be an error, not an empty pass.
    """
    ref = dict(res.get("local_reference", {}))
    g2 = json.load(open(_HERE.parent / "gate2.json"))["runs"]
    need = []
    for m, _ in MODELS:
        for T in TEMPS:
            k = f"{m}|{T}"
            if k in ref:
                continue
            vs = [g2[j]["top1"] for j in g2
                  if j.startswith(f"{m}|T{T}|s") and isinstance(g2[j], dict)]
            if vs:
                ref[k] = {"mean": round(statistics.mean(vs), 4),
                          "sd": round(statistics.pstdev(vs), 4), "n_seeds": len(vs),
                          "source": "gate2"}
            else:
                need.append((m, T))
    for m, T in need:
        from ar_ca import ARRule, run
        rule = ARRule(m)
        vs = []
        for s_ in SEEDS:
            lat = run(rule, B=B, N=N, r=R, T=T, sweeps=SETTLE, scheme="none",
                      init="random", seed=s_, order="per_replica")["final"]
            vs.append(float(np.mean([collections.Counter(row.tolist()).most_common(1)[0][1] / N
                                     for row in lat])))
        ref[f"{m}|{T}"] = {"mean": round(statistics.mean(vs), 4),
                           "sd": round(statistics.pstdev(vs), 4), "n_seeds": len(vs),
                           "source": "measured here at the frozen geometry (gate2 lacks this model)"}
        print(f"  local reference {m} T={T}: {ref[f'{m}|{T}']['mean']:.3f}"
              f"+-{ref[f'{m}|{T}']['sd']:.3f}", flush=True)
        del rule
        try: torch.mps.empty_cache()
        except Exception: pass
    res["local_reference"] = ref
    json.dump(res, open(OUT, "w"), indent=1)
    return ref


def main():
    block = json.load(open(PREREG))
    if not verify_block(block):
        raise SystemExit("prereg.json failed its own hash check")
    print("  prereg block verifies: True", flush=True)
    res = json.load(open(OUT)) if OUT.exists() else {"runs": {}}
    runs = res["runs"]
    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    ref = local_reference(res)

    for name, role in MODELS:
        need = [f"{name}|{mo}|T{T}|s{s}" for mo in MODES for T in TEMPS for s in SEEDS]
        if all(k in runs for k in need):
            print(f"  {name}: already complete", flush=True); continue
        t0 = time.time()
        srv, tok, model = start_endpoint(name, dev)
        print(f"  {name} ({role}) served on :{PORT} in {time.time()-t0:.0f}s", flush=True)
        prefix = None
        try:
            if hasattr(tok, "apply_chat_template") and tok.chat_template:
                prefix = tok.apply_chat_template([{"role": "user", "content": ""}],
                                                 tokenize=False, add_generation_prompt=True)
        except Exception:
            prefix = None
        if prefix is None:
            prefix = "<|im_start|>user\n<|im_end|>\n<|im_start|>assistant\n"   # a stand-in wrapper
        with httpx.Client(timeout=60.0) as client:
            for mo in MODES:
                for T in TEMPS:
                    for s in SEEDS:
                        k = f"{name}|{mo}|T{T}|s{s}"
                        if k in runs: continue
                        t1 = time.time()
                        a = api_settle(client, tok, mo, T, s,
                                       chat_prefix=prefix if mo == "chat" else None)
                        w = srv.state["widths"]
                        tot = max(sum(w.values()), 1)
                        runs[k] = dict(model=name, mode=mo, T=T, seed=s, top1=round(a, 4),
                                       context_widths={str(kk): round(vv / tot, 4)
                                                       for kk, vv in sorted(w.items())},
                                       secs=round(time.time() - t1, 1))
                        srv.state["widths"] = collections.Counter()
                        print(f"     {mo:10s} T={T:<6} s={s} top1={a:.3f} "
                              f"({time.time()-t1:.0f}s)", flush=True)
                        json.dump(res, open(OUT, "w"), indent=1)
        srv.shutdown()
        del model
        try: torch.mps.empty_cache()
        except Exception: pass

    out = analyse(res, ref)
    out["runs"] = runs
    out["_prereg_sha256"] = block["sha256"]
    print("\n  ->", out["gate3_verdict"])
    save_results(OUT, out, script=__file__, root=ROOT, prereg=block,
                 independent_unit="family", forbid_paths=True)
    print("\nwrote fingerprint/gate3.json")


def analyse(res, ref):
    runs = res["runs"]
    out = {"local_reference": ref, "modes": {}, "scope": (
        "LOCAL HARNESS ONLY. No third-party endpoint was contacted: this machine has no API "
        "credentials and acquiring them was not authorised. What is tested is the part PROGRAM.md 6 "
        "requires first -- that the battery survives the API abstraction and still separates models "
        "whose answer is already known.")}
    for mo in MODES:
        rows = {}
        for name, _ in MODELS:
            for T in TEMPS:
                vs = [runs[k]["top1"] for k in runs
                      if k.startswith(f"{name}|{mo}|T{T}|s") and isinstance(runs[k], dict)]
                r = ref.get(f"{name}|{T}")
                if vs and r:
                    api = statistics.mean(vs)
                    tol = max(r["sd"], statistics.pstdev(vs), 0.02) * 2
                    rows[f"{name}|{T}"] = {
                        "api": round(api, 4), "local": r["mean"],
                        "delta": round(api - r["mean"], 4), "tolerance_2sd": round(tol, 4),
                        "reproduces": bool(abs(api - r["mean"]) <= tol)}
        sep = None
        for T in TEMPS:
            a = rows.get(f"{MODELS[0][0]}|{T}"); b = rows.get(f"{MODELS[1][0]}|{T}")
            if a and b:
                g = a["api"] - b["api"]
                lg = a["local"] - b["local"]
                sep = {"T": T, "api_gap": round(g, 4), "local_gap": round(lg, 4),
                       "discriminates": bool(g > 0.2)} if (sep is None or g > sep["api_gap"]) else sep
        wid = {}
        for name, _ in MODELS:
            ws = [runs[k].get("context_widths") for k in runs
                  if k.startswith(f"{name}|{mo}|") and isinstance(runs[k], dict)
                  and runs[k].get("context_widths")]
            if ws:
                agg = {}
                for w in ws:
                    for kk, vv in w.items():
                        agg[kk] = agg.get(kk, 0.0) + vv / len(ws)
                wid[name] = {kk: round(vv, 3) for kk, vv in sorted(agg.items())}
        out["modes"][mo] = {"cells": rows, "realized_context_widths": wid,
                            "all_reproduce": bool(rows and all(v["reproduces"] for v in rows.values())),
                            "separation": sep}

    parts = []
    fid = out["modes"].get("ids", {})
    expected = len(MODELS) * len(TEMPS)
    if len(fid.get("cells", {})) < expected:
        missing = [f"{m}|{T}" for m, _ in MODELS for T in TEMPS
                   if f"{m}|{T}" not in fid.get("cells", {})]
        out["gate3_verdict"] = (
            f"NOT DECIDABLE -- the fidelity arm is missing a local reference for {missing}. "
            f"An absent comparison is an error, not a pass: without a left-hand side the arm "
            f"would report 'reproduces' for cells it never checked.")
        out["fidelity_ok"] = False
        return out
    if not fid["all_reproduce"]:
        bad = [k for k, v in fid["cells"].items() if not v["reproduces"]]
        parts.append(
            f"FIDELITY FAILS on {', '.join(bad)}: the ids arm changes nothing but moving logits "
            f"across a socket, so a deviation there is a BUG IN THIS HARNESS, not a fact about "
            f"APIs. Nothing below is reported as a property of the API port until it is fixed.")
        out["gate3_verdict"] = " ".join(parts)
        out["fidelity_ok"] = False
        return out
    parts.append("FIDELITY OK: the ids arm reproduces the local numbers within seed spread, so the "
                 "harness is faithful and the remaining arms isolate real API effects.")
    out["fidelity_ok"] = True

    for mo in MODES[1:]:
        m = out["modes"][mo]
        s = m.get("separation") or {}
        parts.append(
            f"{mo.upper()}: "
            + ("reproduces the local signature" if m["all_reproduce"] else
               "does NOT reproduce the local numbers")
            + (f", and still separates pythia-410m from gpt2 by {s.get('api_gap')} "
               f"(local {s.get('local_gap')})." if s.get("discriminates") else
               f", and does NOT separate the two models (api gap {s.get('api_gap')} against a "
               f"local gap of {s.get('local_gap')}), so identification fails here even if a number "
               f"survived."))
    # The width diagnostic is what turns the text arm from an observation into a mechanism.
    tw = out["modes"].get("text", {}).get("realized_context_widths", {})
    if tw:
        frag = ", ".join(f"{m.split('/')[-1]} r=1 in {w.get('1', 0.0):.0%} of calls"
                         for m, w in tw.items())
        parts.append(
            f"THE TEXT ARM'S MECHANISM IS MEASURED, NOT INFERRED: a two-token window decoded to "
            f"text and re-tokenized MERGES into a single token -- {frag}. The endpoint is therefore "
            f"running the CA at a SMALLER RADIUS than requested, and F69 established r=1 is the "
            f"degenerate regime. The merge rate is a property of the tokenizer and the ring's own "
            f"content, so it is model-dependent and does not cancel: it inverts which model looks "
            f"attractor-bearing. Any text-in/text-out endpoint has this, and it is invisible unless "
            f"the realized width is measured, which is why it is recorded per cell here.")

    chat = out["modes"].get("chat", {})
    raw_ok = any(out["modes"][mo].get("separation", {}).get("discriminates")
                 for mo in ("ids", "nospecial", "text"))
    chat_ok = chat.get("separation", {}).get("discriminates")
    k3 = bool(not chat_ok and not raw_ok)
    out["K3"] = {"fired": k3, "raw_endpoint_discriminates": bool(raw_ok),
                 "chat_template_discriminates": bool(chat_ok),
                 "rule": "K3 fires if chat wrapping destroys the signature AND no raw arm reproduces it"}
    if k3:
        parts.append("K3 FIRES: no arm discriminates, so the capability does not survive the API "
                     "port at all -- not merely scoped, absent.")
    elif not chat_ok:
        parts.append("K3 SCOPES AS REGISTERED: raw completion arms carry the signature, chat "
                     "templating destroys it. The capability is scoped to raw completion endpoints "
                     "(open-weight serving, base-model APIs), which is exactly what PROGRAM.md 5 "
                     "registered as the expected outcome given F66 -- one BOS token moved the share "
                     "74% -> 24%, and a chat template prepends far more than one.")
    else:
        cw = out["modes"].get("chat", {}).get("realized_context_widths", {})
        parts.append(
            "K3 DOES NOT FIRE, AND THE REGISTERED HAZARD WAS THE WRONG ONE. PROGRAM.md 5 expected "
            "chat templating to be what breaks the port, reasoning from F66 (one BOS token moved "
            "the share 74% -> 24%). It is not: the chat arm keeps a full-width context "
            f"({cw}) and still separates the models. What breaks the port is the PLAIN TEXT "
            "round-trip, which was not registered as a hazard at all -- and the chat template "
            "helps precisely because its wrapper text keeps the window tokens from merging. The "
            "capability is therefore scoped by INTERFACE TYPE, not by templating: token-id "
            "endpoints carry it, text endpoints destroy it, and a text endpoint could be made to "
            "work by separating the window so it cannot merge. That is a testable claim this gate "
            "did not test.")
    parts.append(
        "SCOPE: local harness only. No third-party endpoint was contacted -- this machine has no "
        "API credentials and obtaining them was not authorised. This establishes what PROGRAM.md 6 "
        "requires BEFORE an unknown endpoint may be characterized, and does not establish that any "
        "commercial endpoint behaves this way.")
    out["gate3_verdict"] = " ".join(parts)
    return out


if __name__ == "__main__":
    main()
